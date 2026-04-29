"""
backtesting/walk_forward.py — Walk-Forward Backtest Framework.

¿Qué es walk-forward?
=====================
Un backtest normal sobre 5y produce UN único Sharpe. Eso no te dice si
tu estrategia es robusta o si depende de un período concreto (p.ej. el
bull run de 2021). Walk-forward = correr N backtests sobre ventanas
deslizantes y obtener una DISTRIBUCIÓN de Sharpes.

Métricas que produce:
    - Distribución de Sharpe por ventana (median, Q25, Q75, peor, mejor)
    - % ventanas con Sharpe > 0 / > 1
    - Max drawdown por ventana
    - Consistencia del win rate
    - Análisis de peor cuartil (lo que importa para producción)

Modos:
    1. Rolling: ventanas de tamaño fijo que se deslizan (p.ej. 6 meses
       cada 3 meses). Cada ventana es independiente. Para detectar
       regímenes malos.
    2. Expanding: la ventana de train crece y el test avanza. Simula
       mejor el comportamiento real del bot (ves más historia conforme
       avanza el tiempo).

Uso típico:
    >>> from backtesting.walk_forward import WalkForwardConfig, run_walk_forward
    >>> from backtesting.portfolio_engine import PortfolioBacktestEngine, PortfolioConfig
    >>> from strategies.ensemble import build_default_ensemble
    >>>
    >>> cfg = WalkForwardConfig(window_months=6, step_months=3, n_windows=10)
    >>> results = run_walk_forward(
    ...     data={"AAPL": df_aapl, "MSFT": df_msft},
    ...     strategy_factory=build_default_ensemble,
    ...     config=cfg,
    ... )
    >>> print(results.summary())
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from backtesting.engine import BacktestResult, _periods_per_year
from backtesting.portfolio_engine import (
    PortfolioBacktestEngine,
    PortfolioConfig,
    PortfolioResult,
)
from execution.costs import CostModel
from strategies.base import BaseStrategy
from utils.logger import get_logger

logger = get_logger(__name__)


# ════════════════════════════════════════════════════════════════════
#  Configuración
# ════════════════════════════════════════════════════════════════════
@dataclass
class WalkForwardConfig:
    """
    Parámetros del walk-forward.

    Attributes
    ----------
    window_months : int
        Tamaño de cada ventana de test en meses (default 6).
        Ej: 6 → cada backtest cubre medio año.

    step_months : int
        Desplazamiento entre ventanas en meses (default 3).
        Ej: 3 → solapamiento del 50% entre ventanas.
        step_months == window_months → sin solapamiento.

    n_windows : int
        Número máximo de ventanas a evaluar. None = todas las que quepan
        en los datos disponibles.

    mode : str
        "rolling": ventanas de tamaño fijo. Cada ventana incluye sólo
            `window_months` de datos. Sin contexto histórico previo.
        "expanding": la ventana de test avanza, pero el backtest siempre
            empieza desde el primer dato disponible (más historia = más
            contexto para la estrategia, pero la ventana de test sigue
            siendo `window_months`).

    min_bars : int
        Barras mínimas en una ventana para incluirla. Evita incluir
        ventanas con muy pocos datos (weekends, gaps, etc.). Default 20.

    portfolio_config : PortfolioConfig
        Configuración del motor de cartera que se aplica en cada ventana.
        Si None, se usa PortfolioConfig por defecto.
    """
    window_months: int = 6
    step_months: int = 3
    n_windows: Optional[int] = None
    mode: str = "rolling"        # "rolling" | "expanding"
    min_bars: int = 20
    portfolio_config: Optional[PortfolioConfig] = None


# ════════════════════════════════════════════════════════════════════
#  Resultado de cada ventana
# ════════════════════════════════════════════════════════════════════
@dataclass
class WindowResult:
    """Resultado de una ventana del walk-forward."""
    window_id: int
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    n_bars: int
    n_symbols: int
    sharpe: float
    sortino: float
    calmar: float
    max_drawdown_pct: float
    total_return_pct: float
    win_rate: float
    total_trades: int
    profit_factor: float
    avg_cost_bps: float
    # Flag de si la ventana fue válida (suficientes datos / trades)
    is_valid: bool = True
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "window_id": self.window_id,
            "start_date": self.start_date.strftime("%Y-%m-%d"),
            "end_date": self.end_date.strftime("%Y-%m-%d"),
            "n_bars": self.n_bars,
            "n_symbols": self.n_symbols,
            "sharpe": round(self.sharpe, 3),
            "sortino": round(self.sortino, 3),
            "calmar": round(self.calmar, 3),
            "max_drawdown_pct": round(self.max_drawdown_pct, 2),
            "total_return_pct": round(self.total_return_pct, 2),
            "win_rate": round(self.win_rate, 1),
            "total_trades": self.total_trades,
            "profit_factor": round(self.profit_factor, 2),
            "avg_cost_bps": round(self.avg_cost_bps, 1),
            "is_valid": self.is_valid,
        }


# ════════════════════════════════════════════════════════════════════
#  Resultado agregado
# ════════════════════════════════════════════════════════════════════
@dataclass
class WalkForwardResult:
    """
    Resultado completo del walk-forward.

    Contiene todas las métricas agregadas sobre la distribución
    de resultados por ventana, además de la lista completa de
    resultados individuales.
    """
    config: WalkForwardConfig
    n_windows_total: int
    n_windows_valid: int
    windows: List[WindowResult] = field(default_factory=list)

    # ── Distribución de Sharpe ──────────────────────────────────────
    sharpe_mean: float = 0.0
    sharpe_median: float = 0.0
    sharpe_std: float = 0.0
    sharpe_min: float = 0.0
    sharpe_max: float = 0.0
    sharpe_q25: float = 0.0
    sharpe_q75: float = 0.0

    # ── Porcentajes de ventanas ─────────────────────────────────────
    pct_windows_positive_sharpe: float = 0.0   # % con Sharpe > 0
    pct_windows_sharpe_above_1: float = 0.0    # % con Sharpe > 1
    pct_windows_profitable: float = 0.0        # % con retorno > 0

    # ── Peor cuartil (lo que importa para producción) ──────────────
    worst_quartile_sharpe: float = 0.0
    worst_quartile_return: float = 0.0
    worst_quartile_drawdown: float = 0.0

    # ── Otras distribuciones ────────────────────────────────────────
    return_median: float = 0.0
    drawdown_median: float = 0.0
    win_rate_median: float = 0.0
    trades_median: float = 0.0

    # ── Consistencia ────────────────────────────────────────────────
    # Sharpe rolling (para detectar si hay tendencia de degradación)
    sharpe_trend: float = 0.0    # pendiente de una regresión lineal Sharpe vs tiempo

    def summary(self) -> str:
        valid_tag = f"{self.n_windows_valid}/{self.n_windows_total}"
        lines = [
            "\n" + "=" * 60,
            f"  WALK-FORWARD BACKTEST — {valid_tag} ventanas válidas",
            f"  Modo: {self.config.mode}  "
            f"Ventana: {self.config.window_months}m  "
            f"Step: {self.config.step_months}m",
            "=" * 60,
            "",
            "  ── Distribución de Sharpe ───────────────────────────",
            f"  Media      : {self.sharpe_mean:+.3f}",
            f"  Mediana    : {self.sharpe_median:+.3f}",
            f"  Std        : {self.sharpe_std:.3f}",
            f"  Mín / Máx  : {self.sharpe_min:+.3f} / {self.sharpe_max:+.3f}",
            f"  Q25 / Q75  : {self.sharpe_q25:+.3f} / {self.sharpe_q75:+.3f}",
            "",
            "  ── Robustez ─────────────────────────────────────────",
            f"  Ventanas Sharpe > 0 : {self.pct_windows_positive_sharpe:.1f}%",
            f"  Ventanas Sharpe > 1 : {self.pct_windows_sharpe_above_1:.1f}%",
            f"  Ventanas rentables  : {self.pct_windows_profitable:.1f}%",
            "",
            "  ── Peor cuartil (Q25) ───────────────────────────────",
            f"  Sharpe     : {self.worst_quartile_sharpe:+.3f}",
            f"  Retorno    : {self.worst_quartile_return:+.2f}%",
            f"  Max DD     : {self.worst_quartile_drawdown:.2f}%",
            "",
            "  ── Medianas por ventana ─────────────────────────────",
            f"  Retorno    : {self.return_median:+.2f}%",
            f"  Max DD     : {self.drawdown_median:.2f}%",
            f"  Win rate   : {self.win_rate_median:.1f}%",
            f"  Trades     : {self.trades_median:.0f}",
            "",
            "  ── Tendencia temporal ───────────────────────────────",
            f"  Pendiente Sharpe: {self.sharpe_trend:+.4f} por ventana",
            "  " + ("(Estrategia se DEGRADA con el tiempo)" if self.sharpe_trend < -0.05
                     else "(Estrategia ESTABLE o mejora con el tiempo)"),
            "=" * 60,
        ]
        return "\n".join(lines)

    def to_dataframe(self) -> pd.DataFrame:
        """Devuelve un DataFrame con los resultados por ventana."""
        return pd.DataFrame([w.to_dict() for w in self.windows])


# ════════════════════════════════════════════════════════════════════
#  Generación de ventanas temporales
# ════════════════════════════════════════════════════════════════════
def _generate_windows(
    master_index: pd.DatetimeIndex,
    window_months: int,
    step_months: int,
    n_windows: Optional[int],
    mode: str,
) -> List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    """
    Genera lista de (train_start, test_start, test_end) para cada ventana.

    En modo "rolling", train_start = test_start - window_months.
    En modo "expanding", train_start = siempre el inicio del dataset.

    Devuelve tuplas (data_start, test_start, test_end).
    """
    if len(master_index) == 0:
        return []

    global_start = master_index[0]
    global_end = master_index[-1]

    windows = []
    # Empezamos a generar ventanas de test desde el inicio + window_months
    # (necesitamos al menos una ventana de datos antes del primer test).
    test_start = global_start + pd.DateOffset(months=window_months)

    while test_start < global_end:
        test_end = test_start + pd.DateOffset(months=window_months)
        if test_end > global_end:
            test_end = global_end

        if mode == "expanding":
            data_start = global_start
        else:  # rolling
            data_start = test_start - pd.DateOffset(months=window_months)
            if data_start < global_start:
                data_start = global_start

        windows.append((data_start, test_start, test_end))
        test_start += pd.DateOffset(months=step_months)

        if n_windows is not None and len(windows) >= n_windows:
            break

    return windows


# ════════════════════════════════════════════════════════════════════
#  Runner de una ventana
# ════════════════════════════════════════════════════════════════════
def _run_single_window(
    window_id: int,
    data_start: pd.Timestamp,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
    data: Dict[str, pd.DataFrame],
    strategy_factory: Callable[[], BaseStrategy],
    portfolio_config: PortfolioConfig,
    min_bars: int,
    mode: str,
) -> WindowResult:
    """
    Ejecuta un backtest de portfolio sobre una ventana temporal.

    En modo "rolling": usa [data_start, test_end] como datos totales.
    En modo "expanding": usa [data_start, test_end], pero sólo evaluamos
        las métricas sobre [test_start, test_end] — la parte anterior
        es "train" y se usa para warmup de la estrategia.

    Nota: el PortfolioBacktestEngine actual no soporta modo expanding
    nativo. Lo emulamos corriendo el backtest completo y extrayendo
    sólo los trades que caen en el período de test.
    """
    # Recortar datos a la ventana temporal
    sliced: Dict[str, pd.DataFrame] = {}
    for sym, df in data.items():
        if isinstance(df.index, pd.DatetimeIndex):
            mask = (df.index >= data_start) & (df.index <= test_end)
            sub = df.loc[mask]
        else:
            sub = df
        if len(sub) >= min_bars:
            sliced[sym] = sub

    if not sliced:
        return WindowResult(
            window_id=window_id,
            start_date=test_start,
            end_date=test_end,
            n_bars=0,
            n_symbols=0,
            sharpe=0.0,
            sortino=0.0,
            calmar=0.0,
            max_drawdown_pct=0.0,
            total_return_pct=0.0,
            win_rate=0.0,
            total_trades=0,
            profit_factor=0.0,
            avg_cost_bps=0.0,
            is_valid=False,
            error="No data after slicing",
        )

    try:
        strategy = strategy_factory()
        engine = PortfolioBacktestEngine(strategy=strategy, config=portfolio_config)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = engine.run(sliced, lookback=50)

        # En modo expanding, filtrar sólo los trades del período de test
        # para evitar que el "warmup" infle las métricas.
        if mode == "expanding" and result.trades:
            test_trades = [
                t for t in result.trades
                if isinstance(t.get("entry_time"), pd.Timestamp)
                and t["entry_time"] >= test_start
            ]
            if not test_trades and result.total_trades > 0:
                # Si no hay trades en el test pero sí en warmup, ventana inválida
                return WindowResult(
                    window_id=window_id,
                    start_date=test_start,
                    end_date=test_end,
                    n_bars=sum(len(v) for v in sliced.values()),
                    n_symbols=len(sliced),
                    sharpe=0.0,
                    sortino=0.0,
                    calmar=0.0,
                    max_drawdown_pct=0.0,
                    total_return_pct=0.0,
                    win_rate=0.0,
                    total_trades=0,
                    profit_factor=0.0,
                    avg_cost_bps=0.0,
                    is_valid=False,
                    error="No trades in test period (expanding mode)",
                )
            # Re-calcular métricas sólo sobre los trades del período de test
            # Usamos métricas simples directamente de los trades
            n_test = len(test_trades)
            wins = [t for t in test_trades if t["pnl"] > 0]
            losses = [t for t in test_trades if t["pnl"] <= 0]
            win_rate = (len(wins) / n_test * 100) if n_test else 0.0
            gp = sum(t["pnl"] for t in wins) if wins else 0.0
            gl = abs(sum(t["pnl"] for t in losses)) if losses else 0.0
            pf = (gp / gl) if gl > 0 else (float("inf") if gp > 0 else 0.0)
            avg_cost = float(np.mean([t.get("total_cost_bps", 0.0) for t in test_trades]))

            # Equity sólo sobre el período de test
            ec = result.equity_curve
            if ec is not None and isinstance(ec.index, pd.DatetimeIndex):
                ec_test = ec.loc[ec.index >= test_start]
            else:
                ec_test = ec

            sharpe = _calc_sharpe(ec_test) if ec_test is not None and len(ec_test) > 5 else 0.0
            sortino = _calc_sortino(ec_test) if ec_test is not None and len(ec_test) > 5 else 0.0
            max_dd = _calc_maxdd(ec_test) if ec_test is not None and len(ec_test) > 5 else 0.0
            total_ret = (
                (float(ec_test.iloc[-1]) / float(ec_test.iloc[0]) - 1.0) * 100
                if ec_test is not None and len(ec_test) > 1 else 0.0
            )
            calmar = (total_ret / abs(max_dd)) if max_dd != 0 else 0.0

            return WindowResult(
                window_id=window_id,
                start_date=test_start,
                end_date=test_end,
                n_bars=sum(len(v) for v in sliced.values()),
                n_symbols=len(sliced),
                sharpe=sharpe,
                sortino=sortino,
                calmar=calmar,
                max_drawdown_pct=max_dd,
                total_return_pct=total_ret,
                win_rate=win_rate,
                total_trades=n_test,
                profit_factor=pf if pf != float("inf") else 9.99,
                avg_cost_bps=avg_cost,
                is_valid=n_test >= 2,
            )

        # Modo rolling: usar las métricas del engine directamente
        n_bars = sum(len(v) for v in sliced.values())
        return WindowResult(
            window_id=window_id,
            start_date=test_start,
            end_date=test_end,
            n_bars=n_bars,
            n_symbols=len(sliced),
            sharpe=float(result.sharpe_ratio),
            sortino=float(result.sortino_ratio),
            calmar=float(result.calmar_ratio),
            max_drawdown_pct=float(result.max_drawdown_pct),
            total_return_pct=float(result.total_return_pct),
            win_rate=float(result.win_rate),
            total_trades=int(result.total_trades),
            profit_factor=float(result.profit_factor) if result.profit_factor != float("inf") else 9.99,
            avg_cost_bps=float(result.avg_cost_bps),
            is_valid=result.total_trades >= 2,
        )

    except Exception as exc:
        logger.warning("Ventana %d falló: %s", window_id, exc)
        return WindowResult(
            window_id=window_id,
            start_date=test_start,
            end_date=test_end,
            n_bars=0,
            n_symbols=len(sliced),
            sharpe=0.0,
            sortino=0.0,
            calmar=0.0,
            max_drawdown_pct=0.0,
            total_return_pct=0.0,
            win_rate=0.0,
            total_trades=0,
            profit_factor=0.0,
            avg_cost_bps=0.0,
            is_valid=False,
            error=str(exc),
        )


# ════════════════════════════════════════════════════════════════════
#  Helpers de métricas
# ════════════════════════════════════════════════════════════════════
def _calc_sharpe(equity: pd.Series) -> float:
    if equity is None or len(equity) < 5:
        return 0.0
    ret = equity.pct_change().dropna()
    if ret.std() < 1e-9:
        return 0.0
    ppy = _periods_per_year(equity.index) if isinstance(equity.index, pd.DatetimeIndex) else 252.0
    return float((ret.mean() / ret.std()) * np.sqrt(ppy))


def _calc_sortino(equity: pd.Series) -> float:
    if equity is None or len(equity) < 5:
        return 0.0
    ret = equity.pct_change().dropna()
    downside = ret[ret < 0]
    if len(downside) < 2 or downside.std() < 1e-9:
        return 0.0
    ppy = _periods_per_year(equity.index) if isinstance(equity.index, pd.DatetimeIndex) else 252.0
    return float((ret.mean() / downside.std()) * np.sqrt(ppy))


def _calc_maxdd(equity: pd.Series) -> float:
    if equity is None or len(equity) < 2:
        return 0.0
    cummax = equity.cummax()
    dd = (equity - cummax) / cummax * 100
    return float(dd.min())


# ════════════════════════════════════════════════════════════════════
#  Runner principal
# ════════════════════════════════════════════════════════════════════
def run_walk_forward(
    data: Dict[str, pd.DataFrame],
    strategy_factory: Callable[[], BaseStrategy],
    config: Optional[WalkForwardConfig] = None,
    portfolio_config: Optional[PortfolioConfig] = None,
) -> WalkForwardResult:
    """
    Ejecuta el walk-forward backtest completo.

    Args:
        data             : dict {symbol: OHLCV DataFrame}. Los DataFrames
                           deben tener DatetimeIndex.
        strategy_factory : callable sin args que devuelve una estrategia
                           nueva (se llama una vez por ventana para evitar
                           state leakage entre ventanas).
        config           : WalkForwardConfig. Si None, usa defaults.
        portfolio_config : PortfolioConfig para el motor. Si None, usa defaults.

    Returns:
        WalkForwardResult con distribución de métricas y lista de ventanas.
    """
    cfg = config or WalkForwardConfig()
    pcfg = portfolio_config or cfg.portfolio_config or PortfolioConfig()

    # Construir índice maestro para la generación de ventanas
    all_dates = set()
    for df in data.values():
        if isinstance(df.index, pd.DatetimeIndex):
            all_dates.update(df.index.tolist())
    if not all_dates:
        raise ValueError("Los DataFrames no tienen DatetimeIndex.")
    master_index = pd.DatetimeIndex(sorted(all_dates))

    logger.info(
        "Walk-forward: %d símbolos, %d barras [%s → %s], "
        "modo=%s, ventana=%dm, step=%dm",
        len(data), len(master_index), master_index[0].date(), master_index[-1].date(),
        cfg.mode, cfg.window_months, cfg.step_months,
    )

    # Generar ventanas
    windows_spec = _generate_windows(
        master_index=master_index,
        window_months=cfg.window_months,
        step_months=cfg.step_months,
        n_windows=cfg.n_windows,
        mode=cfg.mode,
    )

    if not windows_spec:
        raise ValueError(
            f"No se pudieron generar ventanas con los datos disponibles "
            f"({master_index[0].date()} → {master_index[-1].date()}) y "
            f"window_months={cfg.window_months}."
        )

    logger.info("Generadas %d ventanas. Ejecutando backtests...", len(windows_spec))

    # Ejecutar cada ventana
    window_results: List[WindowResult] = []
    for i, (data_start, test_start, test_end) in enumerate(windows_spec):
        logger.info(
            "Ventana %d/%d: %s → %s",
            i + 1, len(windows_spec),
            test_start.date(), test_end.date(),
        )
        wr = _run_single_window(
            window_id=i + 1,
            data_start=data_start,
            test_start=test_start,
            test_end=test_end,
            data=data,
            strategy_factory=strategy_factory,
            portfolio_config=pcfg,
            min_bars=cfg.min_bars,
            mode=cfg.mode,
        )
        window_results.append(wr)
        logger.info(
            "  → Sharpe=%.3f  Ret=%.2f%%  DD=%.2f%%  trades=%d  valid=%s",
            wr.sharpe, wr.total_return_pct, wr.max_drawdown_pct,
            wr.total_trades, wr.is_valid,
        )

    # Filtrar válidas para las estadísticas
    valid = [w for w in window_results if w.is_valid]
    n_valid = len(valid)

    if n_valid == 0:
        logger.warning("Ninguna ventana produjo resultados válidos.")
        return WalkForwardResult(
            config=cfg,
            n_windows_total=len(window_results),
            n_windows_valid=0,
            windows=window_results,
        )

    sharpes = np.array([w.sharpe for w in valid])
    returns = np.array([w.total_return_pct for w in valid])
    drawdowns = np.array([w.max_drawdown_pct for w in valid])
    win_rates = np.array([w.win_rate for w in valid])
    trades = np.array([w.total_trades for w in valid])

    # Tendencia del Sharpe: regresión lineal sobre número de ventana
    sharpe_trend = float(np.polyfit(np.arange(n_valid), sharpes, 1)[0]) if n_valid >= 3 else 0.0

    # Peor cuartil
    q25_sharpe = float(np.percentile(sharpes, 25))
    worst_q_mask = sharpes <= q25_sharpe
    worst_returns = returns[worst_q_mask]
    worst_drawdowns = drawdowns[worst_q_mask]

    result = WalkForwardResult(
        config=cfg,
        n_windows_total=len(window_results),
        n_windows_valid=n_valid,
        windows=window_results,
        # Distribución de Sharpe
        sharpe_mean=float(np.mean(sharpes)),
        sharpe_median=float(np.median(sharpes)),
        sharpe_std=float(np.std(sharpes)),
        sharpe_min=float(np.min(sharpes)),
        sharpe_max=float(np.max(sharpes)),
        sharpe_q25=q25_sharpe,
        sharpe_q75=float(np.percentile(sharpes, 75)),
        # Robustez
        pct_windows_positive_sharpe=float(np.mean(sharpes > 0) * 100),
        pct_windows_sharpe_above_1=float(np.mean(sharpes > 1.0) * 100),
        pct_windows_profitable=float(np.mean(returns > 0) * 100),
        # Peor cuartil
        worst_quartile_sharpe=q25_sharpe,
        worst_quartile_return=float(np.mean(worst_returns)) if len(worst_returns) else 0.0,
        worst_quartile_drawdown=float(np.mean(worst_drawdowns)) if len(worst_drawdowns) else 0.0,
        # Medianas
        return_median=float(np.median(returns)),
        drawdown_median=float(np.median(drawdowns)),
        win_rate_median=float(np.median(win_rates)),
        trades_median=float(np.median(trades)),
        # Tendencia
        sharpe_trend=sharpe_trend,
    )

    logger.info(result.summary())
    return result


# ════════════════════════════════════════════════════════════════════
#  CLI helper: integración con main.py
# ════════════════════════════════════════════════════════════════════
def run_walk_forward_from_cli(
    symbols: List[str],
    period: str,
    interval: str,
    strategy_name: str,
    model_path: Optional[str] = None,
    window_months: int = 6,
    step_months: int = 3,
    n_windows: Optional[int] = None,
    mode: str = "rolling",
    max_positions: int = 5,
    corr_threshold: float = 0.75,
    position_size_pct: float = 20.0,
    cost_model_kind: str = "flat",
    commission_bps: float = 1.0,
    spread_bps: float = 4.0,
    save_result: Optional[str] = None,
) -> WalkForwardResult:
    """
    Wrapper de alto nivel para invocar el walk-forward desde `main.py`.

    Descarga datos, construye la strategy factory y lanza el walk-forward.
    """
    from data.fetcher import DataFetcher
    from execution.costs import build_cost_model

    logger.info("Descargando datos para walk-forward...")
    data: Dict[str, pd.DataFrame] = {}
    for sym in symbols:
        df = DataFetcher.fetch_yahoo(ticker=sym, period=period, interval=interval)
        if df is not None and not df.empty:
            df.attrs["symbol"] = sym
            data[sym] = df
            logger.info("  %-8s : %d filas [%s → %s]", sym, len(df), df.index[0], df.index[-1])
        else:
            logger.warning("Sin datos para %s — se omite.", sym)

    if not data:
        raise ValueError("Ningún símbolo produjo datos válidos.")

    cost_model = build_cost_model(
        kind=cost_model_kind,
        commission_bps=commission_bps,
        spread_bps=spread_bps,
    )

    pcfg = PortfolioConfig(
        max_positions=max_positions,
        corr_threshold=corr_threshold if corr_threshold <= 1.0 else None,
        position_size_pct=position_size_pct,
        cost_model=cost_model,
    )

    # Strategy factory: debe devolver una nueva instancia en cada llamada
    def _make_strategy() -> BaseStrategy:
        if strategy_name == "meta_ensemble":
            if not model_path:
                raise ValueError("meta_ensemble requiere --model")
            from strategies.meta_labeled_ensemble import build_meta_labeled_ensemble_from_file
            return build_meta_labeled_ensemble_from_file(model_path)
        elif strategy_name == "ensemble":
            from strategies.ensemble import build_default_ensemble
            return build_default_ensemble()
        elif strategy_name == "donchian":
            from strategies.donchian_trend import DonchianTrendStrategy
            return DonchianTrendStrategy()
        elif strategy_name == "rsi2_mr":
            from strategies.rsi2_mean_reversion import RSI2MeanReversionStrategy
            return RSI2MeanReversionStrategy()
        else:
            from strategies.ensemble import build_default_ensemble
            return build_default_ensemble()

    wf_cfg = WalkForwardConfig(
        window_months=window_months,
        step_months=step_months,
        n_windows=n_windows,
        mode=mode,
        portfolio_config=pcfg,
    )

    result = run_walk_forward(
        data=data,
        strategy_factory=_make_strategy,
        config=wf_cfg,
        portfolio_config=pcfg,
    )

    if save_result:
        import pickle
        from pathlib import Path
        p = Path(save_result).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "wb") as f:
            pickle.dump({
                "kind": "walk_forward",
                "result": result,
                "symbols": symbols,
                "saved_at": pd.Timestamp.utcnow().isoformat(),
            }, f)
        logger.info("Resultado walk-forward guardado en %s", p)

    return result