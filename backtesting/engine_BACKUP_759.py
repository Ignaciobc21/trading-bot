"""
engine.py — Motor de backtesting.

Simula la ejecución de una estrategia sobre datos históricos OHLCV
y genera métricas de rendimiento (Sharpe, Sortino, Calmar, drawdown,
win-rate, profit factor, expectancy, exposure, etc.).

Características:
    - Entradas/salidas al OPEN de la siguiente barra (evita look-ahead).
    - Slippage y comisión configurables.
    - Stop-loss y take-profit intrabar (usa high/low).
    - Tamaño de posición por % del capital.
    - Máximo de barras en posición (forzar cierre).
    - Métricas anualizadas según el intervalo del dataset.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from strategies.base import BaseStrategy, Action
from config.settings import (
    INITIAL_CAPITAL,
    COMMISSION_PCT,
    SLIPPAGE_PCT,
    MAX_POSITION_SIZE_PCT,
    STOP_LOSS_PCT,
    TAKE_PROFIT_PCT,
    MAX_HOLDING_BARS,
)
from utils.logger import get_logger

logger = get_logger(__name__)


# ──────────────────────────────────────────────
# Utilidades
# ──────────────────────────────────────────────
def _periods_per_year(index: pd.DatetimeIndex) -> float:
    """
    Estima cuántas barras hay por año a partir del índice temporal.
    Fallback conservador: 252 (diario).
    """
    if not isinstance(index, pd.DatetimeIndex) or len(index) < 2:
        return 252.0
    deltas = index.to_series().diff().dropna()
    if deltas.empty:
        return 252.0
    median_seconds = deltas.dt.total_seconds().median()
    if median_seconds <= 0:
        return 252.0
    seconds_per_year = 365.25 * 24 * 3600
    # Aproximación: si el intervalo es intradía (<1 día), asumimos ~6.5h de
    # sesión por día bursátil y 252 días hábiles.
    if median_seconds < 23 * 3600:
        bars_per_trading_day = (6.5 * 3600) / median_seconds
        return bars_per_trading_day * 252.0
    return seconds_per_year / median_seconds


# ──────────────────────────────────────────────
# Resultado
# ──────────────────────────────────────────────
@dataclass
class BacktestResult:
    strategy_name: str
    initial_capital: float
    final_capital: float
    total_return_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown_pct: float
    win_rate: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    profit_factor: float
    expectancy_pct: float
    exposure_pct: float
    max_consecutive_losses: int
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
            f"  Sortino Ratio   : {self.sortino_ratio:.2f}\n"
            f"  Calmar Ratio    : {self.calmar_ratio:.2f}\n"
            f"  Max Drawdown    : {self.max_drawdown_pct:.2f}%\n"
            f"  Win Rate        : {self.win_rate:.1f}%\n"
            f"  Total trades    : {self.total_trades}\n"
            f"  Ganadores       : {self.winning_trades}\n"
            f"  Perdedores      : {self.losing_trades}\n"
            f"  Profit Factor   : {self.profit_factor:.2f}\n"
            f"  Expectancy/trade: {self.expectancy_pct:+.2f}%\n"
            f"  Exposure        : {self.exposure_pct:.1f}%\n"
            f"  Max losing run  : {self.max_consecutive_losses}\n"
            f"{'=' * 50}\n"
        )


# ──────────────────────────────────────────────
# Motor
# ──────────────────────────────────────────────
class BacktestEngine:
    """
    Motor de backtest con ejecución al open de la siguiente barra,
    slippage, SL/TP intrabar y sizing por % de capital.
    """

    def __init__(
        self,
        strategy: BaseStrategy,
        initial_capital: float = INITIAL_CAPITAL,
        commission_pct: float = COMMISSION_PCT,
        slippage_pct: float = SLIPPAGE_PCT,
        position_size_pct: float = 100.0,
        stop_loss_pct: float = STOP_LOSS_PCT,
        take_profit_pct: float = TAKE_PROFIT_PCT,
        max_holding_bars: int = MAX_HOLDING_BARS,
    ):
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.commission_pct = commission_pct / 100.0
        self.slippage_pct = slippage_pct / 100.0
        self.position_size_pct = max(0.0, min(position_size_pct, 100.0)) / 100.0
        self.stop_loss_pct = stop_loss_pct / 100.0 if stop_loss_pct > 0 else 0.0
        self.take_profit_pct = take_profit_pct / 100.0 if take_profit_pct > 0 else 0.0
        self.max_holding_bars = max(0, max_holding_bars)

    # ──────────────────────────────────────────
    # Ejecución
    # ──────────────────────────────────────────
    def run(self, df: pd.DataFrame, lookback: int = 50) -> BacktestResult:
        """
        Ejecuta el backtest sobre un DataFrame OHLCV.

        Args:
            df: DataFrame con columnas 'open','high','low','close','volume'.
            lookback: barras mínimas antes de empezar a operar.
        """
        required_cols = {"open", "high", "low", "close", "volume"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"Faltan columnas en el DataFrame: {missing}")

        logger.info(
            "Iniciando backtest de '%s' — %d velas, capital=$%.2f, "
            "slippage=%.3f%%, commission=%.3f%%, SL=%.2f%%, TP=%.2f%%, size=%.1f%%",
            self.strategy.name,
            len(df),
            self.initial_capital,
            self.slippage_pct * 100,
            self.commission_pct * 100,
            self.stop_loss_pct * 100,
            self.take_profit_pct * 100,
            self.position_size_pct * 100,
        )

        # Pre-computar señales de la estrategia (vectorizado si la estrategia
        # soporta generate_signals; si no, la base itera por defecto).
        signals = self.strategy.generate_signals(df)

        capital = self.initial_capital
        position: Optional[dict] = None  # {entry_price, quantity, entry_time, entry_idx, invested}
        trades: List[dict] = []
        equity: List[float] = [capital] * lookback  # relleno inicial
        bars_in_position = 0

        n = len(df)
        for i in range(lookback, n):
            current_time = df.index[i]
            open_price = float(df["open"].iloc[i])
            high = float(df["high"].iloc[i])
            low = float(df["low"].iloc[i])
            close = float(df["close"].iloc[i])

            # La señal del bar i-1 se ejecuta al OPEN del bar i (no look-ahead).
            prev_action = signals.iloc[i - 1] if i - 1 >= 0 else Action.HOLD

            # ── Salidas intrabar (SL/TP/max holding) ANTES de mirar la señal ──
            if position is not None:
                exit_price: Optional[float] = None
                exit_reason: Optional[str] = None

                if self.stop_loss_pct > 0:
                    sl_price = position["entry_price"] * (1.0 - self.stop_loss_pct)
                    if low <= sl_price:
                        exit_price = sl_price
                        exit_reason = "STOP_LOSS"

                if exit_price is None and self.take_profit_pct > 0:
                    tp_price = position["entry_price"] * (1.0 + self.take_profit_pct)
                    if high >= tp_price:
                        exit_price = tp_price
                        exit_reason = "TAKE_PROFIT"

                if (
                    exit_price is None
                    and self.max_holding_bars > 0
                    and bars_in_position >= self.max_holding_bars
                ):
                    exit_price = open_price  # forzar al open del bar
                    exit_reason = "MAX_HOLDING"

                if exit_price is not None:
                    capital = self._close_position(
                        position, exit_price, current_time, trades, reason=exit_reason, side_buy=True
                    )
                    # sumamos el cash liberado + el resto que quedó fuera.
                    capital = capital + position.get("reserved_cash", 0.0)
                    position = None
                    bars_in_position = 0

            # ── Ejecutar señal del bar anterior al OPEN del bar actual ──
            if position is None and prev_action == Action.BUY:
                fill_price = open_price * (1.0 + self.slippage_pct)
                invested = capital * self.position_size_pct
                if invested > 0 and fill_price > 0:
                    commission = invested * self.commission_pct
                    quantity = (invested - commission) / fill_price
                    if quantity > 0:
                        position = {
                            "entry_price": fill_price,
                            "quantity": quantity,
                            "entry_time": current_time,
                            "entry_idx": i,
                            "invested": invested,
                            "reserved_cash": capital - invested,
                        }
                        capital = 0.0
                        bars_in_position = 0
                        logger.debug("BUY  @ %.4f  qty=%.6f", fill_price, quantity)

            elif position is not None and prev_action == Action.SELL:
                fill_price = open_price * (1.0 - self.slippage_pct)
                capital = self._close_position(
                    position, fill_price, current_time, trades, reason="SIGNAL", side_buy=True
                )
                capital = capital + position.get("reserved_cash", 0.0)
                position = None
                bars_in_position = 0

            # ── Mark-to-market para equity curve ──
            if position is not None:
                bars_in_position += 1
                equity.append(position["quantity"] * close + position.get("reserved_cash", 0.0))
            else:
                equity.append(capital)

        # Cerrar posición pendiente al cierre final.
        if position is not None:
            last_price = float(df["close"].iloc[-1]) * (1.0 - self.slippage_pct)
            capital = self._close_position(
                position,
                last_price,
                df.index[-1],
                trades,
                reason="END_OF_DATA",
                side_buy=True,
            )
            capital = capital + position.get("reserved_cash", 0.0)
            position = None

        equity_series = pd.Series(equity, index=df.index[: len(equity)])
        result = self._calculate_metrics(trades, equity_series)
        result.trades = trades

        logger.info(result.summary())
        return result

    # ──────────────────────────────────────────
    # Cierre de posición
    # ──────────────────────────────────────────
    def _close_position(
        self,
        position: dict,
        exit_price: float,
        exit_time,
        trades: List[dict],
        reason: str,
        side_buy: bool = True,
    ) -> float:
        exit_value = position["quantity"] * exit_price
        commission = exit_value * self.commission_pct
        proceeds = exit_value - commission
        pnl = proceeds - position["invested"]
        return_pct = (pnl / position["invested"]) * 100 if position["invested"] > 0 else 0.0

        trades.append(
            {
                "entry_price": position["entry_price"],
                "exit_price": exit_price,
                "entry_time": position["entry_time"],
                "exit_time": exit_time,
                "quantity": position["quantity"],
                "invested": position["invested"],
                "pnl": pnl,
                "return_pct": return_pct,
                "exit_reason": reason,
            }
        )
        logger.debug("SELL @ %.4f  PnL=%.2f  reason=%s", exit_price, pnl, reason)
        return proceeds

    # ──────────────────────────────────────────
    # Métricas
    # ──────────────────────────────────────────
    def _calculate_metrics(
        self, trades: List[dict], equity: pd.Series
    ) -> BacktestResult:
        total_trades = len(trades)
        pnls = [t["pnl"] for t in trades]
        return_pcts = [t["return_pct"] for t in trades]
        winning_pnls = [p for p in pnls if p > 0]
        losing_pnls = [p for p in pnls if p <= 0]

<<<<<<< HEAD
        # Sharpe Ratio (anualizado, asumiendo velas diarias)
        if len(pnls) > 1:
            returns = pd.Series([t["return_pct"] / 100 for t in trades])
            sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() != 0 else 0.0
=======
        # Sharpe / Sortino basados en returns del equity curve.
        periods_per_year = _periods_per_year(equity.index) if len(equity) > 1 else 252.0
        equity_returns = equity.pct_change().dropna()

        if len(equity_returns) > 1 and equity_returns.std() > 0:
            sharpe = (equity_returns.mean() / equity_returns.std()) * np.sqrt(periods_per_year)
>>>>>>> 1b8ca7b
        else:
            sharpe = 0.0

        downside = equity_returns[equity_returns < 0]
        if len(downside) > 1 and downside.std() > 0:
            sortino = (equity_returns.mean() / downside.std()) * np.sqrt(periods_per_year)
        else:
            sortino = 0.0

        # Max drawdown (sobre equity curve, %).
        if len(equity) > 0:
            cummax = equity.cummax()
            drawdown = (equity - cummax) / cummax * 100
            max_dd = float(drawdown.min())
        else:
            max_dd = 0.0

        # Calmar: retorno anualizado / |max drawdown|.
        if len(equity) > 1 and max_dd < 0:
            total_ret = (equity.iloc[-1] / equity.iloc[0]) - 1.0
            years = len(equity) / periods_per_year if periods_per_year > 0 else 1.0
            if years > 0 and (1.0 + total_ret) > 0:
                cagr = (1.0 + total_ret) ** (1.0 / years) - 1.0
            else:
                cagr = 0.0
            calmar = (cagr * 100) / abs(max_dd) if max_dd != 0 else 0.0
        else:
            calmar = 0.0

        gross_profit = sum(winning_pnls) if winning_pnls else 0.0
        gross_loss = abs(sum(losing_pnls)) if losing_pnls else 0.0
        if gross_loss > 0:
            profit_factor = gross_profit / gross_loss
        elif gross_profit > 0:
            profit_factor = float("inf")
        else:
            profit_factor = 0.0

        expectancy_pct = float(np.mean(return_pcts)) if return_pcts else 0.0

        # Exposure %: fracción de barras con posición abierta (aproximación por
        # cambios de equity respecto al capital cuando está flat).
        if trades and len(equity) > 0:
            bars_in_pos = 0
            for t in trades:
                try:
                    entry_loc = equity.index.get_loc(t["entry_time"])
                    exit_loc = equity.index.get_loc(t["exit_time"])
                    bars_in_pos += max(0, exit_loc - entry_loc)
                except KeyError:
                    continue
            exposure_pct = (bars_in_pos / len(equity)) * 100
        else:
            exposure_pct = 0.0

        # Max consecutive losses.
        max_losing_run = 0
        run = 0
        for pnl in pnls:
            if pnl <= 0:
                run += 1
                max_losing_run = max(max_losing_run, run)
            else:
                run = 0

        final_capital = float(equity.iloc[-1]) if len(equity) > 0 else self.initial_capital

        return BacktestResult(
            strategy_name=self.strategy.name,
            initial_capital=self.initial_capital,
            final_capital=final_capital,
            total_return_pct=((final_capital - self.initial_capital) / self.initial_capital) * 100,
            sharpe_ratio=round(float(sharpe), 2),
            sortino_ratio=round(float(sortino), 2),
            calmar_ratio=round(float(calmar), 2),
            max_drawdown_pct=round(max_dd, 2),
            win_rate=round((len(winning_pnls) / total_trades * 100) if total_trades > 0 else 0.0, 1),
            total_trades=total_trades,
            winning_trades=len(winning_pnls),
            losing_trades=len(losing_pnls),
            profit_factor=round(profit_factor, 2) if profit_factor != float("inf") else profit_factor,
            expectancy_pct=round(expectancy_pct, 2),
            exposure_pct=round(exposure_pct, 1),
            max_consecutive_losses=int(max_losing_run),
            equity_curve=equity,
        )

    # ──────────────────────────────────────────
    # Visualización / persistencia
    # ──────────────────────────────────────────
    @staticmethod
    def plot_results(result: BacktestResult, save_path: Optional[str] = None) -> None:
        """Genera gráficos del backtest: equity curve y drawdown."""
        if result.equity_curve is None:
            logger.warning("No hay equity curve para graficar")
            return

        fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
        fig.suptitle(f"Backtest — {result.strategy_name}", fontsize=14, fontweight="bold")

        axes[0].plot(result.equity_curve, color="#2196F3", linewidth=1.2)
        axes[0].axhline(result.initial_capital, color="gray", linestyle="--", alpha=0.5)
        axes[0].set_ylabel("Capital ($)")
        axes[0].set_title("Equity Curve")
        axes[0].grid(alpha=0.3)

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
            plt.close(fig)
        else:
            try:
                plt.show()
            except Exception:
                logger.debug("No se pudo mostrar el gráfico en backend no interactivo")
                plt.close(fig)

    @staticmethod
    def save_trades_csv(result: BacktestResult, path: str) -> None:
        """Persiste los trades a CSV para análisis posterior."""
        if not result.trades:
            logger.warning("No hay trades que guardar")
            return
        pd.DataFrame(result.trades).to_csv(path, index=False)
        logger.info("Trades guardados en %s", path)
