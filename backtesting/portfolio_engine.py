"""
portfolio_engine.py — Backtest multi-símbolo (G).

Diseño:
    El motor single-symbol (`BacktestEngine`) opera **una serie por
    ejecución**. Una cartera real elige entre varios símbolos en cada
    barra y reparte el capital entre ellos. Este motor implementa la
    lógica de cartera mínima:

      1. Pre-genera señales por símbolo (reutilizando exactamente la
         misma `strategy.generate_signals(df)` que el motor 1-símbolo).
      2. Itera barras alineadas en un índice común (unión de los índices
         de cada símbolo). En cada barra, primero gestiona salidas
         (SL/TP/MAX_HOLDING/SIGNAL) y luego entradas, en ese orden.
      3. Comparte un único `cash` entre todos los símbolos.
         `position_size_pct` se interpreta sobre el **equity total** del
         portfolio en el momento de la entrada — no por símbolo.
      4. Limita el nº de posiciones simultáneas (`max_positions`) y
         filtra entradas por correlación con las posiciones ya abiertas
         (`corr_threshold` sobre matriz rolling de log-returns).

    No se hace re-balanceo automático: cada posición vive hasta su
    salida (SL/TP/timeout/SELL signal). Esto mantiene el motor honesto y
    fácil de auditar, en línea con `BacktestEngine`.

Limitaciones conocidas (para futuras iteraciones):
    - **Sin shorts**: la cartera es long-only, igual que el motor base.
    - **Sin re-balanceo periódico**: no se redimensionan posiciones tras
      ganancias/pérdidas grandes. Cada trade tiene tamaño fijo decidido
      al abrir.
    - **Sin priorización entre BUY simultáneos**: si en la misma barra
      dos símbolos generan BUY y sólo cabe uno, se abre por orden
      lexicográfico de ticker (estable y reproducible). Versión futura:
      priorizar por `proba` del meta-modelo.
    - **Comisión fija**: el cost model real (slippage dinámico, market
      impact) llegará en la fase I.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from backtesting.engine import BacktestResult, _periods_per_year
from config.settings import (
    INITIAL_CAPITAL,
    COMMISSION_PCT,
    SLIPPAGE_PCT,
    STOP_LOSS_PCT,
    TAKE_PROFIT_PCT,
    MAX_HOLDING_BARS,
)
from strategies.base import Action, BaseStrategy
from utils.logger import get_logger

logger = get_logger(__name__)


# ──────────────────────────────────────────────
# Resultado agregado de cartera
# ──────────────────────────────────────────────
@dataclass
class PortfolioResult:
    """
    Métricas agregadas de la cartera + desglose por símbolo.

    `equity_curve` es la curva única del portfolio (no por símbolo).
    `per_symbol` contiene, para cada ticker, su `BacktestResult`-like
    con métricas calculadas SÓLO sobre los trades de ese símbolo.
    """

    strategy_name: str
    symbols: List[str]
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
    max_concurrent_positions: int
    avg_concurrent_positions: float
    rejected_by_corr: int
    rejected_by_max_positions: int
    per_symbol: Dict[str, BacktestResult] = field(default_factory=dict)
    trades: List[dict] = field(default_factory=list)
    equity_curve: Optional[pd.Series] = None
    concurrent_positions_curve: Optional[pd.Series] = None

    def summary(self) -> str:
        """Resumen humano-legible del backtest de cartera."""
        lines = [
            "\n" + "=" * 60,
            f"  PORTFOLIO BACKTEST -- {self.strategy_name}",
            f"  Símbolos        : {', '.join(self.symbols)}  (n={len(self.symbols)})",
            "=" * 60,
            f"  Capital inicial : ${self.initial_capital:,.2f}",
            f"  Capital final   : ${self.final_capital:,.2f}",
            f"  Retorno total   : {self.total_return_pct:+.2f}%",
            f"  Sharpe Ratio    : {self.sharpe_ratio:.2f}",
            f"  Sortino Ratio   : {self.sortino_ratio:.2f}",
            f"  Calmar Ratio    : {self.calmar_ratio:.2f}",
            f"  Max Drawdown    : {self.max_drawdown_pct:.2f}%",
            f"  Win Rate        : {self.win_rate:.1f}%",
            f"  Total trades    : {self.total_trades}  (W={self.winning_trades} / L={self.losing_trades})",
            f"  Profit Factor   : {self.profit_factor:.2f}",
            f"  Expectancy/trade: {self.expectancy_pct:+.2f}%",
            f"  Exposure        : {self.exposure_pct:.1f}%",
            f"  Max concurrent  : {self.max_concurrent_positions}",
            f"  Avg concurrent  : {self.avg_concurrent_positions:.2f}",
            f"  Rejected (corr) : {self.rejected_by_corr}",
            f"  Rejected (cap)  : {self.rejected_by_max_positions}",
            "─" * 60,
            "  Desglose por símbolo:",
        ]
        for sym, res in self.per_symbol.items():
            lines.append(
                f"    {sym:<10s} trades={res.total_trades:3d}  "
                f"WR={res.win_rate:5.1f}%  PF={res.profit_factor:5.2f}  "
                f"ret={res.total_return_pct:+6.2f}%"
            )
        lines.append("=" * 60 + "\n")
        return "\n".join(lines)


# ──────────────────────────────────────────────
# Configuración del motor
# ──────────────────────────────────────────────
@dataclass
class PortfolioConfig:
    """Parámetros del PortfolioBacktestEngine."""

    initial_capital: float = INITIAL_CAPITAL
    commission_pct: float = COMMISSION_PCT
    slippage_pct: float = SLIPPAGE_PCT
    # % del equity TOTAL a invertir en cada nueva entrada. 20% ⇒ con
    # max_positions=5 puedes llegar a ~100% invertido.
    position_size_pct: float = 20.0
    stop_loss_pct: float = STOP_LOSS_PCT
    take_profit_pct: float = TAKE_PROFIT_PCT
    max_holding_bars: int = MAX_HOLDING_BARS
    # Máximo de posiciones simultáneas. None ⇒ sin límite (no recomendado:
    # con position_size_pct=20% podrías llegar a apalancarte por error).
    max_positions: int = 5
    # Umbral de correlación para el guard. Si la corr media del candidato
    # con las posiciones abiertas supera este valor, la entrada se
    # rechaza. None desactiva el guard.
    corr_threshold: Optional[float] = 0.75
    # Ventana rolling para la matriz de correlación (en barras).
    corr_window: int = 60
    # Mínimo de barras observadas en el rolling antes de aplicar el
    # guard. Si hay menos historia, no rechazamos por corr (no fiable).
    corr_min_obs: int = 30


# ──────────────────────────────────────────────
# Motor de cartera
# ──────────────────────────────────────────────
class PortfolioBacktestEngine:
    """
    Backtest multi-símbolo. Una sola pasada por barra que (en orden):
        a) cierra posiciones por SL/TP/max-holding/SELL,
        b) actualiza la matriz de correlación rolling,
        c) abre nuevas posiciones que pasen los filtros de corr+cap.

    Las señales se pre-computan por símbolo antes del loop, lo que
    permite usar cualquier `BaseStrategy` existente sin modificarla.
    """

    def __init__(self, strategy: BaseStrategy, config: Optional[PortfolioConfig] = None):
        self.strategy = strategy
        self.cfg = config or PortfolioConfig()
        # Casts en proporciones internas (igual que `BacktestEngine`).
        self._comm = self.cfg.commission_pct / 100.0
        self._slip = self.cfg.slippage_pct / 100.0
        self._pos_size = max(0.0, min(self.cfg.position_size_pct, 100.0)) / 100.0
        self._sl = self.cfg.stop_loss_pct / 100.0 if self.cfg.stop_loss_pct > 0 else 0.0
        self._tp = self.cfg.take_profit_pct / 100.0 if self.cfg.take_profit_pct > 0 else 0.0

    # ──────────────────────────────────────────
    # API pública
    # ──────────────────────────────────────────
    def run(
        self,
        data: Dict[str, pd.DataFrame],
        lookback: int = 50,
        size_multipliers: Optional[Dict[str, pd.Series]] = None,
    ) -> PortfolioResult:
        """
        Ejecuta el backtest de cartera.

        Args:
            data: dict ticker → DataFrame OHLCV (mismo formato que el
                  motor single-symbol). Cada DataFrame puede tener su
                  propio rango temporal — el motor alinea por unión.
            lookback: barras mínimas antes de empezar a operar (igual
                  que el motor base).
            size_multipliers: dict opcional ticker → Serie con multiplicador
                  por barra (usado por el RiskOverlay). Símbolos sin
                  entrada → multiplicador implícito 1.0.

        Returns:
            `PortfolioResult` con métricas agregadas + desglose.
        """
        if not data:
            raise ValueError("`data` no puede estar vacío.")

        # Validación mínima de columnas por símbolo.
        for sym, df in data.items():
            req = {"open", "high", "low", "close", "volume"}
            missing = req - set(df.columns)
            if missing:
                raise ValueError(f"{sym}: faltan columnas {missing}")

        symbols = sorted(data.keys())  # orden estable y reproducible.
        n_syms = len(symbols)

        # 1. Pre-computar señales y close-series para correlación.
        signals_by_sym: Dict[str, pd.Series] = {}
        close_by_sym: Dict[str, pd.Series] = {}
        size_mult_by_sym: Dict[str, pd.Series] = {}
        for sym in symbols:
            df = data[sym]
            # Anotar el ticker en attrs por si la estrategia construye
            # features que necesitan saberlo (sentiment de la fase B).
            df.attrs.setdefault("symbol", sym)
            sig = self.strategy.generate_signals(df)
            signals_by_sym[sym] = sig.reindex(df.index)
            close_by_sym[sym] = df["close"].astype(float)
            if size_multipliers and sym in size_multipliers:
                size_mult_by_sym[sym] = (
                    size_multipliers[sym].reindex(df.index).fillna(0.0).astype(float)
                )
            else:
                size_mult_by_sym[sym] = pd.Series(1.0, index=df.index)

        # 2. Índice común de la cartera = unión de los índices de cada
        # símbolo (con orden temporal). Esto soporta símbolos con
        # diferente fecha de listado (ej. una IPO posterior).
        master_index = pd.DatetimeIndex(sorted(set().union(*[df.index for df in data.values()])))
        n = len(master_index)
        logger.info(
            "Portfolio backtest: %d símbolos, %d barras [%s → %s], "
            "max_positions=%s, corr≤%s (window=%d), pos_size=%.1f%%",
            n_syms, n, master_index[0], master_index[-1],
            self.cfg.max_positions, self.cfg.corr_threshold,
            self.cfg.corr_window, self._pos_size * 100,
        )

        # 3. Construir DataFrame de log-returns para correlación rolling.
        # Reindexamos cada close al master_index, hacemos pct_change y
        # log; los NaN (cuando un símbolo aún no cotiza) los dejamos.
        close_panel = pd.DataFrame(
            {sym: close_by_sym[sym].reindex(master_index) for sym in symbols}
        )
        # log-returns: log(p_t / p_{t-1}). Más simétrico que pct_change
        # para correlación, especialmente con activos de vol distinta.
        log_ret_panel = np.log(close_panel / close_panel.shift(1))

        # 4. Estado del portfolio.
        cash = float(self.cfg.initial_capital)
        positions: Dict[str, dict] = {}  # sym → position dict
        trades: List[dict] = []
        # Equity curve y curva de nº de posiciones simultáneas.
        equity_curve = np.full(n, np.nan, dtype=float)
        concur_curve = np.zeros(n, dtype=int)
        # Contadores diagnósticos.
        rej_corr = 0
        rej_cap = 0

        # 5. Loop principal por barra del índice maestro.
        for i in range(n):
            t = master_index[i]

            # ─ Salidas: SL/TP/max-holding ANTES de leer señales ─────
            for sym in list(positions.keys()):
                pos = positions[sym]
                if t not in close_by_sym[sym].index:
                    continue  # no hay barra para este símbolo en t.
                df = data[sym]
                # Localizar el índice posicional (más rápido por
                # `get_loc` que `loc` con timestamp).
                try:
                    j = df.index.get_loc(t)
                except KeyError:
                    continue
                high = float(df["high"].iloc[j])
                low = float(df["low"].iloc[j])
                open_p = float(df["open"].iloc[j])
                exit_price: Optional[float] = None
                exit_reason: Optional[str] = None

                # Stop-loss: si el low toca el SL price.
                if self._sl > 0:
                    sl_p = pos["entry_price"] * (1.0 - self._sl)
                    if low <= sl_p:
                        exit_price = sl_p
                        exit_reason = "STOP_LOSS"

                # Take-profit (sólo si no se disparó SL).
                if exit_price is None and self._tp > 0:
                    tp_p = pos["entry_price"] * (1.0 + self._tp)
                    if high >= tp_p:
                        exit_price = tp_p
                        exit_reason = "TAKE_PROFIT"

                # Timeout por barras en posición.
                if (
                    exit_price is None
                    and self.cfg.max_holding_bars > 0
                    and pos["bars_in_position"] >= self.cfg.max_holding_bars
                ):
                    exit_price = open_p
                    exit_reason = "MAX_HOLDING"

                if exit_price is not None:
                    proceeds = self._close(pos, exit_price, t, sym, trades, exit_reason)
                    cash += proceeds
                    positions.pop(sym, None)

            # ─ Salidas por SEÑAL del bar anterior ────────────────────
            # Igual que el motor 1-símbolo, las señales del bar t-1 se
            # ejecutan al OPEN del bar t (no look-ahead).
            for sym in list(positions.keys()):
                df = data[sym]
                if t not in df.index:
                    continue
                j = df.index.get_loc(t)
                if j == 0:
                    continue
                prev_action = signals_by_sym[sym].iloc[j - 1]
                if prev_action != Action.SELL:
                    continue
                fill = float(df["open"].iloc[j]) * (1.0 - self._slip)
                proceeds = self._close(positions[sym], fill, t, sym, trades, "SIGNAL")
                cash += proceeds
                positions.pop(sym, None)

            # ─ Entradas BUY ─────────────────────────────────────────
            # Recolectar candidatos del bar anterior y filtrar por:
            #   1. cap de posiciones (max_positions),
            #   2. cash disponible,
            #   3. correlation guard.
            buy_candidates: List[Tuple[str, float, float]] = []  # (sym, fill, size_mult)
            for sym in symbols:
                if sym in positions:
                    continue  # ya estamos dentro.
                df = data[sym]
                if t not in df.index:
                    continue
                j = df.index.get_loc(t)
                if j < lookback:
                    continue  # warmup.
                prev_action = signals_by_sym[sym].iloc[j - 1]
                if prev_action != Action.BUY:
                    continue
                size_mult = float(size_mult_by_sym[sym].iloc[j - 1])
                if size_mult <= 0:
                    continue
                fill = float(df["open"].iloc[j]) * (1.0 + self._slip)
                buy_candidates.append((sym, fill, size_mult))

            for sym, fill, size_mult in buy_candidates:
                # ── 1. Cap de posiciones ─────────────────────────
                if (
                    self.cfg.max_positions is not None
                    and len(positions) >= self.cfg.max_positions
                ):
                    rej_cap += 1
                    continue

                # ── 2. Correlation guard ─────────────────────────
                if (
                    self.cfg.corr_threshold is not None
                    and positions
                    and i >= self.cfg.corr_min_obs
                ):
                    # Slice rolling de log-returns hasta t-1 (estricto:
                    # no usar la barra actual para no leakear).
                    lo = max(0, i - self.cfg.corr_window)
                    window_ret = log_ret_panel.iloc[lo:i]
                    open_syms = list(positions.keys())
                    sub = window_ret[[sym] + open_syms].dropna()
                    if len(sub) >= self.cfg.corr_min_obs:
                        corr_with_open = sub.corr().loc[sym, open_syms]
                        # Usamos la corr media en valor absoluto: una
                        # corr -0.9 sería igual de "redundante" en
                        # términos de exposición a un mismo factor de
                        # riesgo (sólo cambia el signo).
                        if corr_with_open.abs().mean() > self.cfg.corr_threshold:
                            rej_corr += 1
                            continue

                # ── 3. Sizing y ejecución ────────────────────────
                # `equity_total` = cash + valor mark-to-market de las
                # posiciones abiertas. El sizing es sobre ESE total para
                # que la cartera se comporte coherentemente: si las
                # posiciones abiertas valen mucho, las nuevas también
                # son proporcionales.
                equity_total = cash + self._mark_to_market(positions, master_index, i, close_panel)
                # `size_mult` ya viene del bar j-1 del propio símbolo (no
                # del master_index) — se calculó en `buy_candidates`.
                target_invest = equity_total * self._pos_size * size_mult
                # Cap por cash disponible (no apalancamos).
                invested = min(target_invest, cash)
                if invested <= 0 or fill <= 0:
                    continue
                commission = invested * self._comm
                quantity = (invested - commission) / fill
                if quantity <= 0:
                    continue

                positions[sym] = {
                    "entry_price": fill,
                    "quantity": quantity,
                    "entry_time": t,
                    "invested": invested,
                    "bars_in_position": 0,
                }
                cash -= invested

            # ─ Mark-to-market y avance de bars_in_position ─────────
            mtm_value = self._mark_to_market(positions, master_index, i, close_panel)
            equity_curve[i] = cash + mtm_value
            concur_curve[i] = len(positions)
            for pos in positions.values():
                pos["bars_in_position"] += 1

        # 6. Cerrar lo que quede al final del backtest.
        for sym in list(positions.keys()):
            df = data[sym]
            last_close = float(df["close"].iloc[-1]) * (1.0 - self._slip)
            proceeds = self._close(
                positions[sym], last_close, df.index[-1], sym, trades, "END_OF_DATA"
            )
            cash += proceeds
            positions.pop(sym)

        # 7. Limpiar NaN del equity curve (huecos al inicio antes del
        # primer dato disponible) → forward-fill desde initial_capital.
        ec = pd.Series(equity_curve, index=master_index, name="equity").ffill()
        ec = ec.fillna(self.cfg.initial_capital)
        cc = pd.Series(concur_curve, index=master_index, name="concurrent_positions")

        # 8. Métricas.
        result = self._build_result(trades, ec, cc, symbols, rej_corr, rej_cap)
        logger.info(result.summary())
        return result

    # ──────────────────────────────────────────
    # Helpers internos
    # ──────────────────────────────────────────
    @staticmethod
    def _mark_to_market(
        positions: Dict[str, dict],
        master_index: pd.DatetimeIndex,
        i: int,
        close_panel: pd.DataFrame,
    ) -> float:
        """Valor mark-to-market de las posiciones abiertas en la barra `i`.

        Si un símbolo no tiene precio en la barra (NaN), se usa el último
        precio disponible (el panel ya está reindexado al master_index;
        como fallback usamos el entry_price para no inflar la equity).
        """
        if not positions:
            return 0.0
        total = 0.0
        for sym, pos in positions.items():
            px = close_panel[sym].iloc[i]
            if pd.isna(px):
                # Fallback: último precio conocido hasta i, o entry.
                prev = close_panel[sym].iloc[: i + 1].dropna()
                px = float(prev.iloc[-1]) if len(prev) > 0 else pos["entry_price"]
            total += pos["quantity"] * float(px)
        return total

    def _close(
        self,
        pos: dict,
        exit_price: float,
        exit_time,
        symbol: str,
        trades: List[dict],
        reason: str,
    ) -> float:
        """Cierra una posición long y registra el trade."""
        exit_value = pos["quantity"] * exit_price
        commission = exit_value * self._comm
        proceeds = exit_value - commission
        pnl = proceeds - pos["invested"]
        return_pct = (pnl / pos["invested"]) * 100 if pos["invested"] > 0 else 0.0
        trades.append({
            "symbol": symbol,
            "entry_price": pos["entry_price"],
            "exit_price": exit_price,
            "entry_time": pos["entry_time"],
            "exit_time": exit_time,
            "quantity": pos["quantity"],
            "invested": pos["invested"],
            "pnl": pnl,
            "return_pct": return_pct,
            "exit_reason": reason,
        })
        return proceeds

    # ──────────────────────────────────────────
    # Métricas agregadas + por símbolo
    # ──────────────────────────────────────────
    def _build_result(
        self,
        trades: List[dict],
        equity_curve: pd.Series,
        concur_curve: pd.Series,
        symbols: List[str],
        rej_corr: int,
        rej_cap: int,
    ) -> PortfolioResult:
        # ── Métricas agregadas ──────────────────────────────────────
        initial = float(equity_curve.iloc[0]) if len(equity_curve) else self.cfg.initial_capital
        final = float(equity_curve.iloc[-1]) if len(equity_curve) else initial
        total_ret_pct = (final / initial - 1.0) * 100 if initial > 0 else 0.0

        eq_ret = equity_curve.pct_change().dropna()
        ppy = _periods_per_year(equity_curve.index) if len(equity_curve) > 1 else 252.0
        if len(eq_ret) > 1 and eq_ret.std() > 0:
            sharpe = (eq_ret.mean() / eq_ret.std()) * np.sqrt(ppy)
        else:
            sharpe = 0.0
        downside = eq_ret[eq_ret < 0]
        if len(downside) > 1 and downside.std() > 0:
            sortino = (eq_ret.mean() / downside.std()) * np.sqrt(ppy)
        else:
            sortino = 0.0

        # MaxDD sobre la equity curve agregada.
        if len(equity_curve) > 0:
            cummax = equity_curve.cummax()
            dd = (equity_curve - cummax) / cummax * 100
            max_dd = float(dd.min())
        else:
            max_dd = 0.0

        # Calmar = CAGR / |MaxDD|.
        years = len(equity_curve) / ppy if ppy > 0 else 1.0
        if years > 0 and (final / initial) > 0:
            cagr = (final / initial) ** (1.0 / years) - 1.0
        else:
            cagr = 0.0
        calmar = (cagr * 100) / abs(max_dd) if max_dd != 0 else 0.0

        # Métricas de trades agregados.
        total_trades = len(trades)
        wins = [t for t in trades if t["pnl"] > 0]
        losses = [t for t in trades if t["pnl"] <= 0]
        win_rate = (len(wins) / total_trades) * 100 if total_trades else 0.0
        gross_profit = sum(t["pnl"] for t in wins)
        gross_loss = abs(sum(t["pnl"] for t in losses))
        if gross_loss > 0:
            pf = gross_profit / gross_loss
        elif gross_profit > 0:
            pf = float("inf")
        else:
            pf = 0.0
        expectancy = (
            float(np.mean([t["return_pct"] for t in trades]))
            if total_trades else 0.0
        )

        # Exposición agregada: % del tiempo con ≥1 posición abierta.
        exp_pct = float((concur_curve > 0).mean() * 100) if len(concur_curve) else 0.0

        # ── Métricas por símbolo ────────────────────────────────────
        per_sym: Dict[str, BacktestResult] = {}
        for sym in symbols:
            sym_trades = [t for t in trades if t["symbol"] == sym]
            per_sym[sym] = self._per_symbol_result(sym, sym_trades, initial)

        return PortfolioResult(
            strategy_name=self.strategy.name,
            symbols=symbols,
            initial_capital=initial,
            final_capital=final,
            total_return_pct=total_ret_pct,
            sharpe_ratio=float(sharpe),
            sortino_ratio=float(sortino),
            calmar_ratio=float(calmar),
            max_drawdown_pct=max_dd,
            win_rate=win_rate,
            total_trades=total_trades,
            winning_trades=len(wins),
            losing_trades=len(losses),
            profit_factor=pf,
            expectancy_pct=expectancy,
            exposure_pct=exp_pct,
            max_concurrent_positions=int(concur_curve.max()) if len(concur_curve) else 0,
            avg_concurrent_positions=float(concur_curve.mean()) if len(concur_curve) else 0.0,
            rejected_by_corr=rej_corr,
            rejected_by_max_positions=rej_cap,
            per_symbol=per_sym,
            trades=trades,
            equity_curve=equity_curve,
            concurrent_positions_curve=concur_curve,
        )

    @staticmethod
    def _per_symbol_result(
        symbol: str, sym_trades: List[dict], initial_capital: float
    ) -> BacktestResult:
        """
        Mini-summary por símbolo. NOTA: Sharpe/Sortino/MaxDD por símbolo
        no se calculan aquí porque requerirían reconstruir su equity curve
        propia, algo poco honesto en un setup de cartera (el capital es
        compartido). Se rellenan a 0 y nos centramos en métricas de
        trades (count, win rate, retorno acumulado).
        """
        n = len(sym_trades)
        wins = [t for t in sym_trades if t["pnl"] > 0]
        losses = [t for t in sym_trades if t["pnl"] <= 0]
        wr = (len(wins) / n) * 100 if n else 0.0
        gp = sum(t["pnl"] for t in wins)
        gl = abs(sum(t["pnl"] for t in losses))
        pf = gp / gl if gl > 0 else (float("inf") if gp > 0 else 0.0)
        # "Retorno por símbolo": suma de PnL del símbolo / capital
        # inicial del portfolio. Es una contribución, no un return real.
        contrib = (sum(t["pnl"] for t in sym_trades) / initial_capital) * 100 if initial_capital > 0 else 0.0
        expectancy = float(np.mean([t["return_pct"] for t in sym_trades])) if n else 0.0
        return BacktestResult(
            strategy_name=symbol,
            initial_capital=initial_capital,
            final_capital=initial_capital + sum(t["pnl"] for t in sym_trades),
            total_return_pct=contrib,
            sharpe_ratio=0.0, sortino_ratio=0.0, calmar_ratio=0.0,
            max_drawdown_pct=0.0,
            win_rate=wr,
            total_trades=n,
            winning_trades=len(wins),
            losing_trades=len(losses),
            profit_factor=pf,
            expectancy_pct=expectancy,
            exposure_pct=0.0,  # sin sentido por símbolo aislado.
            max_consecutive_losses=0,
            trades=sym_trades,
            equity_curve=None,
        )
