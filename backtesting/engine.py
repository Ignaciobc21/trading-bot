"""
engine.py — Motor de backtesting.

Simula la ejecución de una estrategia sobre datos históricos OHLCV
y genera métricas de rendimiento (Sharpe, drawdown, win-rate, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from strategies.base import BaseStrategy, Action
from config.settings import INITIAL_CAPITAL, COMMISSION_PCT
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class BacktestResult:
    """Resultado completo de un backtest."""

    strategy_name: str
    initial_capital: float
    final_capital: float
    total_return_pct: float
    sharpe_ratio: float
    max_drawdown_pct: float
    win_rate: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    profit_factor: float
    trades: List[dict] = field(default_factory=list)
    equity_curve: Optional[pd.Series] = None

    def summary(self) -> str:
        return (
            f"\n{'=' * 50}\n"
            f"  BACKTEST -- {self.strategy_name}\n"
            f"{'=' * 50}\n"
            f"  Capital inicial : ${self.initial_capital:,.2f}\n"
            f"  Capital final   : ${self.final_capital:,.2f}\n"
            f"  Retorno total   : {self.total_return_pct:+.2f}%\n"
            f"  Sharpe Ratio    : {self.sharpe_ratio:.2f}\n"
            f"  Max Drawdown    : {self.max_drawdown_pct:.2f}%\n"
            f"  Win Rate        : {self.win_rate:.1f}%\n"
            f"  Total trades    : {self.total_trades}\n"
            f"  Ganadores       : {self.winning_trades}\n"
            f"  Perdedores      : {self.losing_trades}\n"
            f"  Profit Factor   : {self.profit_factor:.2f}\n"
            f"{'=' * 50}\n"
        )


class BacktestEngine:
    """
    Motor de backtest que recorre datos históricos barra por barra,
    aplica la estrategia y registra operaciones simuladas.
    """

    def __init__(
        self,
        strategy: BaseStrategy,
        initial_capital: float = INITIAL_CAPITAL,
        commission_pct: float = COMMISSION_PCT,
    ):
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.commission_pct = commission_pct / 100.0  # convertir a decimal

    def run(self, df: pd.DataFrame, lookback: int = 50) -> BacktestResult:
        """
        Ejecuta el backtest sobre un DataFrame OHLCV.

        Args:
            df: DataFrame con columnas 'open','high','low','close','volume'
            lookback: Nº de velas mínimas antes de empezar a operar

        Returns:
            BacktestResult con métricas y equity curve.
        """
        logger.info(
            "Iniciando backtest de '%s' — %d velas, capital=$%.2f",
            self.strategy.name,
            len(df),
            self.initial_capital,
        )

        capital = self.initial_capital
        position: Optional[dict] = None  # {"entry_price", "quantity", "side"}
        trades: List[dict] = []
        equity: List[float] = []

        for i in range(lookback, len(df)):
            window = df.iloc[: i + 1]
            current_price = df["close"].iloc[i]
            current_time = df.index[i]

            signal = self.strategy.generate_signal(window)

            # ── ENTRADA ──
            if position is None and signal.action == Action.BUY:
                quantity = capital / current_price
                commission = capital * self.commission_pct
                capital -= commission
                position = {
                    "entry_price": current_price,
                    "quantity": quantity,
                    "entry_time": current_time,
                }
                logger.debug("BUY @ %.4f  qty=%.6f", current_price, quantity)

            # ── SALIDA ──
            elif position is not None and signal.action == Action.SELL:
                exit_value = position["quantity"] * current_price
                commission = exit_value * self.commission_pct
                pnl = exit_value - (position["quantity"] * position["entry_price"]) - commission

                capital = exit_value - commission

                trades.append(
                    {
                        "entry_price": position["entry_price"],
                        "exit_price": current_price,
                        "entry_time": position["entry_time"],
                        "exit_time": current_time,
                        "pnl": pnl,
                        "return_pct": (pnl / (position["quantity"] * position["entry_price"])) * 100,
                    }
                )
                logger.debug(
                    "SELL @ %.4f  PnL=%.2f", current_price, pnl
                )
                position = None

            # Equity mark-to-market
            if position is not None:
                equity.append(position["quantity"] * current_price)
            else:
                equity.append(capital)

        # Si quedó posición abierta, cerrar al último precio
        if position is not None:
            last_price = df["close"].iloc[-1]
            exit_value = position["quantity"] * last_price
            commission = exit_value * self.commission_pct
            pnl = exit_value - (position["quantity"] * position["entry_price"]) - commission
            capital = exit_value - commission
            trades.append(
                {
                    "entry_price": position["entry_price"],
                    "exit_price": last_price,
                    "entry_time": position["entry_time"],
                    "exit_time": df.index[-1],
                    "pnl": pnl,
                    "return_pct": (pnl / (position["quantity"] * position["entry_price"])) * 100,
                }
            )

        # ── Métricas ──
        equity_series = pd.Series(equity, index=df.index[lookback:])
        result = self._calculate_metrics(trades, equity_series)
        result.trades = trades

        logger.info(result.summary())
        return result

    def _calculate_metrics(
        self, trades: List[dict], equity: pd.Series
    ) -> BacktestResult:
        """Calcula métricas de rendimiento a partir de los trades."""
        total_trades = len(trades)
        pnls = [t["pnl"] for t in trades]
        winning = [p for p in pnls if p > 0]
        losing = [p for p in pnls if p <= 0]

        # Sharpe Ratio (anualizado, asumiendo velas diarias)
        if len(pnls) > 1:
            returns = pd.Series(pnls)
            sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() != 0 else 0.0
        else:
            sharpe = 0.0

        # Max Drawdown
        cummax = equity.cummax()
        drawdown = (equity - cummax) / cummax * 100
        max_dd = drawdown.min() if len(drawdown) > 0 else 0.0

        # Profit Factor
        gross_profit = sum(winning) if winning else 0.0
        gross_loss = abs(sum(losing)) if losing else 1.0
        profit_factor = gross_profit / gross_loss if gross_loss != 0 else float("inf")

        final_capital = equity.iloc[-1] if len(equity) > 0 else self.initial_capital

        return BacktestResult(
            strategy_name=self.strategy.name,
            initial_capital=self.initial_capital,
            final_capital=final_capital,
            total_return_pct=((final_capital - self.initial_capital) / self.initial_capital) * 100,
            sharpe_ratio=round(sharpe, 2),
            max_drawdown_pct=round(max_dd, 2),
            win_rate=round((len(winning) / total_trades * 100) if total_trades > 0 else 0.0, 1),
            total_trades=total_trades,
            winning_trades=len(winning),
            losing_trades=len(losing),
            profit_factor=round(profit_factor, 2),
            equity_curve=equity,
        )

    # ──────────────────────────────────────────
    # Visualización
    # ──────────────────────────────────────────
    @staticmethod
    def plot_results(result: BacktestResult, save_path: Optional[str] = None) -> None:
        """Genera gráficos del backtest: equity curve y drawdown."""
        if result.equity_curve is None:
            logger.warning("No hay equity curve para graficar")
            return

        fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
        fig.suptitle(f"Backtest — {result.strategy_name}", fontsize=14, fontweight="bold")

        # Equity curve
        axes[0].plot(result.equity_curve, color="#2196F3", linewidth=1.2)
        axes[0].axhline(
            result.initial_capital, color="gray", linestyle="--", alpha=0.5
        )
        axes[0].set_ylabel("Capital ($)")
        axes[0].set_title("Equity Curve")
        axes[0].grid(alpha=0.3)

        # Drawdown
        cummax = result.equity_curve.cummax()
        drawdown = (result.equity_curve - cummax) / cummax * 100
        axes[1].fill_between(drawdown.index, drawdown, color="#F44336", alpha=0.4)
        axes[1].set_ylabel("Drawdown (%)")
        axes[1].set_title("Drawdown")
        axes[1].grid(alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150)
            logger.info("Gráfico guardado en %s", save_path)
        else:
            plt.show()
