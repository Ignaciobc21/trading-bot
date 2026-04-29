"""
main.py — Punto de entrada del Trading Bot.

Soporta dos modos:
  1. Live trading  → python main.py --mode live
  2. Backtest      → python main.py --mode backtest

Ejemplo de uso:
  python main.py --mode backtest --symbol AAPL --period 1y
  python main.py --mode live
"""

from __future__ import annotations

import argparse
import time
import sys
from pathlib import Path

# ── Compatibilidad con consolas Windows (cp1252) ──
# El help y los banners contienen caracteres Unicode (—, ─, ═, →, ≈, ≥) que
# revientan en `cmd.exe` y `Git Bash` por defecto (codec cp1252).  Forzamos
# UTF-8 en stdout/stderr — disponible en Python 3.7+; si la terminal no lo
# soporta, hacemos fallback silencioso.
if sys.platform == "win32":
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        except (AttributeError, ValueError):
            pass
from typing import Dict, List, Optional

import pandas as pd

from config.settings import (
    TRADING_SYMBOL,
    TIMEFRAME,
    INITIAL_CAPITAL,
    STOP_LOSS_PCT,
    TAKE_PROFIT_PCT,
    MAX_POSITION_SIZE_PCT,
    SLIPPAGE_PCT,
    COMMISSION_PCT,
    MAX_HOLDING_BARS,
    USE_TREND_FILTER,
    TREND_EMA_PERIOD,
)
from utils.logger import get_logger

logger = get_logger("main")


# ═══════════════════════════════════════════════
#  MODO BACKTEST
# ═══════════════════════════════════════════════
def _build_strategy(
    strategy_name: str,
    use_trend_filter: bool,
    trend_ema_period: int,
    model_path: Optional[str] = None,
    threshold_override: Optional[float] = None,
):
    """Factoría simple de estrategias disponibles en CLI."""
    if strategy_name == "rsi":
        from strategies.rsi_strategy import RSIStrategy
        return RSIStrategy(), 50
    if strategy_name == "mfi_rsi":
        from strategies.mfi_rsi_strategy import MfiRsiStrategy
        strat = MfiRsiStrategy(
            rsi_period=14,
            rsi_oversold=30,
            rsi_overbought=70,
            mfi_period=14,
            mfi_oversold=20,
            mfi_overbought=80,
            use_trend_filter=use_trend_filter,
            trend_ema_period=trend_ema_period,
        )
        return strat, max(50, trend_ema_period if use_trend_filter else 0) + 5
    if strategy_name == "donchian":
        from strategies.donchian_trend import DonchianTrendStrategy
        return DonchianTrendStrategy(), 60
    if strategy_name == "rsi2_mr":
        from strategies.rsi2_mean_reversion import RSI2MeanReversionStrategy
        return RSI2MeanReversionStrategy(), 205
    if strategy_name == "ensemble":
        from strategies.ensemble import build_default_ensemble
        return build_default_ensemble(), 205
    if strategy_name == "meta_ensemble":
        if not model_path:
            raise ValueError("--strategy meta_ensemble requiere --model /ruta/al/modelo.pkl")
        from strategies.meta_labeled_ensemble import (
            MetaLabeledEnsembleStrategy,
        )
        from ml.meta_labeler import load_meta_labeler
        infer = load_meta_labeler(model_path)
        strat = MetaLabeledEnsembleStrategy(inferencer=infer, threshold_override=threshold_override)
        return strat, 205
    raise ValueError(f"Estrategia desconocida: {strategy_name}")


def run_backtest(
    symbol: str,
    period: str,
    interval: str,
    strategy_name: str = "mfi_rsi",
    position_size_pct: float = 100.0,
    stop_loss_pct: float = 0.0,   # desactivado por defecto en backtest
    take_profit_pct: float = 0.0,
    slippage_pct: float = SLIPPAGE_PCT,
    commission_pct: float = COMMISSION_PCT,
    max_holding_bars: int = 0,
    use_trend_filter: bool = False,
    trend_ema_period: int = TREND_EMA_PERIOD,
    save_trades: Optional[str] = None,
    save_plot: Optional[str] = None,
    model_path: Optional[str] = None,
    threshold_override: Optional[float] = None,
    risk_overlay: bool = False,
    target_vol: float = 0.20,
    vol_mult_cap: float = 1.50,
    max_daily_loss_pct: float = 0.0,
    use_regime_sizing: bool = True,
    use_confidence_sizing: bool = True,
    save_result: Optional[str] = None,
    cost_model_kind: str = "flat",
    commission_bps: float = 1.0,
    spread_bps: float = 4.0,
    impact_coef: float = 0.1,
) -> None:
    """Descarga datos y ejecuta un backtest con la estrategia seleccionada."""
    from data.fetcher import DataFetcher
    from backtesting.engine import BacktestEngine
    # I — Cost model configurable. Si kind='flat' usa los flags antiguos
    # (commission_pct/slippage_pct) y mantiene compatibilidad bit-for-bit.
    # Si 'realistic', construye un modelo con spread por hora + impact.
    from execution.costs import build_cost_model

    logger.info("═" * 50)
    logger.info("  MODO BACKTEST — strategy=%s", strategy_name)
    logger.info("═" * 50)

    # 1. Descargar datos
    df = DataFetcher.fetch_yahoo(ticker=symbol, period=period, interval=interval)

    if df.empty:
        logger.error("No se pudieron obtener datos para %s", symbol)
        sys.exit(1)

    # Anotar el símbolo en el DataFrame para que las features de
    # sentimiento (si el modelo las usa) sepan a qué ticker pedir noticias.
    df.attrs["symbol"] = symbol

    logger.info("Datos descargados: %d filas [%s → %s]", len(df), df.index[0], df.index[-1])

    # 2. Configurar estrategia (vía factoría)
    strategy, lookback = _build_strategy(
        strategy_name, use_trend_filter, trend_ema_period,
        model_path=model_path, threshold_override=threshold_override,
    )

    # 3. Risk overlay opcional
    size_multiplier = None
    if risk_overlay:
        from risk import RiskConfig, RiskOverlay
        from strategies.regime import RegimeDetector

        cfg = RiskConfig(
            target_vol=target_vol,
            vol_mult_cap=vol_mult_cap,
            use_confidence_sizing=use_confidence_sizing,
            max_daily_loss_pct=max_daily_loss_pct,
        )
        overlay = RiskOverlay(cfg)

        regime_series = None
        if use_regime_sizing:
            regime_series = RegimeDetector().classify(df)

        proba_series = None
        if use_confidence_sizing and hasattr(strategy, "predict_proba_series"):
            try:
                proba_series = strategy.predict_proba_series(df)
            except Exception as exc:  # fail-open: seguimos sin confidence sizing.
                logger.warning("predict_proba_series falló (%s); deshabilitando confidence sizing.", exc)
                proba_series = None

        size_multiplier = overlay.size_multiplier(df, regime=regime_series, proba=proba_series)
        mean_mult = float(size_multiplier.dropna().mean()) if len(size_multiplier.dropna()) else 0.0
        logger.info(
            "Risk overlay ON — target_vol=%.2f vol_cap=%.2f regime=%s conf=%s daily_loss=%.1f%% avg_mult=%.3f",
            target_vol, vol_mult_cap,
            "on" if use_regime_sizing else "off",
            "on" if use_confidence_sizing and proba_series is not None else "off",
            max_daily_loss_pct, mean_mult,
        )

    # 4. Ejecutar backtest
    # I — Construir cost_model según flags del CLI.
    cost_model = build_cost_model(
        kind=cost_model_kind,
        commission_pct=commission_pct,
        slippage_pct=slippage_pct,
        commission_bps=commission_bps,
        spread_bps=spread_bps,
        impact_coef=impact_coef,
    )
    logger.info(
        "Cost model: %s (%s)",
        cost_model.name,
        f"comm_bps={commission_bps}, spread_bps={spread_bps}, impact_coef={impact_coef}"
        if cost_model_kind == "realistic"
        else f"commission_pct={commission_pct}, slippage_pct={slippage_pct}",
    )
    engine = BacktestEngine(
        strategy=strategy,
        initial_capital=INITIAL_CAPITAL,
        commission_pct=commission_pct,
        slippage_pct=slippage_pct,
        position_size_pct=position_size_pct,
        stop_loss_pct=stop_loss_pct,
        take_profit_pct=take_profit_pct,
        max_holding_bars=max_holding_bars,
        max_daily_loss_pct=max_daily_loss_pct if risk_overlay else 0.0,
        cost_model=cost_model,
    )
    result = engine.run(df, lookback=lookback, size_multiplier=size_multiplier)

    # 4. Mostrar resultados
    print(result.summary())

    # 5. Persistencia opcional
    if save_trades:
        engine.save_trades_csv(result, save_trades)

    # 6. Graficar
    engine.plot_results(result, save_path=save_plot)

    # 7. Persistir resultado para el dashboard (H).
    if save_result:
        _save_result_pickle(result, save_result, kind="backtest", symbol=symbol)


def _save_result_pickle(result, path: str, kind: str, symbol: Optional[str] = None) -> None:
    """
    Persiste un BacktestResult o PortfolioResult en pickle, junto con un
    sidecar JSON con metadata legible (kind, symbol, fecha) para el
    dashboard.

    Uso de pickle: las clases tienen pandas Series y dataclasses con
    campos no triviales; pickle es lo más fiel y compacto. JSON sólo
    se usa para el header del archivo, no para los datos.
    """
    import pickle
    from pathlib import Path

    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "wb") as f:
        pickle.dump({
            "kind": kind,
            "symbol": symbol,
            "result": result,
            "saved_at": pd.Timestamp.utcnow().isoformat(),
        }, f)
    logger.info("Resultado %s guardado en %s", kind, p)


# ═══════════════════════════════════════════════
#  MODO PORTFOLIO  (G — backtest multi-symbol)
# ═══════════════════════════════════════════════
def run_portfolio(
    symbols: list,
    period: str,
    interval: str,
    strategy_name: str = "meta_ensemble",
    model_path: Optional[str] = None,
    threshold_override: Optional[float] = None,
    position_size_pct: float = 20.0,
    stop_loss_pct: float = 0.0,
    take_profit_pct: float = 0.0,
    slippage_pct: float = SLIPPAGE_PCT,
    commission_pct: float = COMMISSION_PCT,
    max_holding_bars: int = 0,
    max_positions: int = 5,
    corr_threshold: Optional[float] = 0.75,
    corr_window: int = 60,
    save_result: Optional[str] = None,
    cost_model_kind: str = "flat",
    commission_bps: float = 1.0,
    spread_bps: float = 4.0,
    impact_coef: float = 0.1,
) -> None:
    """
    G — Backtest multi-símbolo con capital compartido y correlation guard.

    Reutiliza la misma `BaseStrategy` que el backtest single-symbol; la
    única diferencia es el motor: `PortfolioBacktestEngine` itera todos
    los símbolos en paralelo sobre un índice maestro común.

    En la fase I se añadió soporte para cost_model: flat (retro-compat,
    default) o realistic (spread por hora + market impact). El cost_model
    se pasa al motor vía `PortfolioConfig.cost_model`.
    """
    from data.fetcher import DataFetcher
    from backtesting.portfolio_engine import PortfolioBacktestEngine, PortfolioConfig
    from execution.costs import build_cost_model

    logger.info("═" * 50)
    logger.info("  MODO PORTFOLIO — strategy=%s  symbols=%s", strategy_name, ",".join(symbols))
    logger.info("═" * 50)

    # 1. Descargar datos para cada símbolo y filtrar los que se hayan
    # devuelto vacíos (delisted, ticker mal escrito, etc.).
    data: Dict[str, "pd.DataFrame"] = {}
    for sym in symbols:
        df = DataFetcher.fetch_yahoo(ticker=sym, period=period, interval=interval)
        if df.empty:
            logger.warning("Sin datos para %s — se descarta.", sym)
            continue
        df.attrs["symbol"] = sym
        data[sym] = df
        logger.info("  %-8s : %d filas [%s → %s]", sym, len(df), df.index[0], df.index[-1])

    if not data:
        logger.error("Ningún símbolo produjo datos válidos.")
        sys.exit(1)

    # 2. Estrategia. Lookback se hereda de la factoría; todos los
    # símbolos comparten la misma estrategia (lo cual es correcto: el
    # ensemble es genérico, no específico por ticker).
    strategy, lookback = _build_strategy(
        strategy_name, use_trend_filter=False, trend_ema_period=TREND_EMA_PERIOD,
        model_path=model_path, threshold_override=threshold_override,
    )

    # 3. Motor de cartera.
    # I — Construir cost_model antes de crear la config.
    cost_model = build_cost_model(
        kind=cost_model_kind,
        commission_pct=commission_pct,
        slippage_pct=slippage_pct,
        commission_bps=commission_bps,
        spread_bps=spread_bps,
        impact_coef=impact_coef,
    )
    logger.info(
        "Cost model: %s (%s)",
        cost_model.name,
        f"comm_bps={commission_bps}, spread_bps={spread_bps}, impact_coef={impact_coef}"
        if cost_model_kind == "realistic"
        else f"commission_pct={commission_pct}, slippage_pct={slippage_pct}",
    )
    cfg = PortfolioConfig(
        initial_capital=INITIAL_CAPITAL,
        commission_pct=commission_pct,
        slippage_pct=slippage_pct,
        position_size_pct=position_size_pct,
        stop_loss_pct=stop_loss_pct,
        take_profit_pct=take_profit_pct,
        max_holding_bars=max_holding_bars,
        max_positions=max_positions,
        # El usuario puede desactivar el guard pasando un threshold > 1.
        corr_threshold=corr_threshold if (corr_threshold is not None and corr_threshold <= 1.0) else None,
        corr_window=corr_window,
        cost_model=cost_model,
    )
    engine = PortfolioBacktestEngine(strategy=strategy, config=cfg)
    # El motor ya llama a `logger.info(result.summary())` internamente.
    # No imprimimos por stdout para evitar UnicodeEncodeError en consolas
    # Windows con codepage cp1252 (los caracteres `═` y `─` no encajan).
    result = engine.run(data, lookback=lookback)

    if save_result:
        _save_result_pickle(result, save_result, kind="portfolio", symbol=",".join(symbols))


# ═══════════════════════════════════════════════
#  MODO WALK-FORWARD (backtest ventanas deslizantes)
# ═══════════════════════════════════════════════
def run_walk_forward_mode(
    symbols: list,
    period: str,
    interval: str,
    strategy_name: str = "ensemble",
    model_path=None,
    window_months: int = 6,
    step_months: int = 3,
    n_windows=None,
    mode: str = "rolling",
    max_positions: int = 5,
    corr_threshold: float = 0.75,
    position_size_pct: float = 20.0,
    cost_model_kind: str = "flat",
    commission_bps: float = 1.0,
    spread_bps: float = 4.0,
    save_result=None,
) -> None:
    """
    G + Walk-Forward — Backtest walk-forward multi-símbolo.

    Ejecuta N backtests sobre ventanas deslizantes y muestra la
    distribución de Sharpes para evaluar si la estrategia es robusta
    o depende de un período concreto.
    """
    from backtesting.walk_forward import run_walk_forward_from_cli

    logger.info("═" * 60)
    logger.info("  MODO WALK-FORWARD — strategy=%s  symbols=%s", strategy_name, ",".join(symbols))
    logger.info("  Ventana=%dm  Step=%dm  Modo=%s  MaxVentanas=%s",
                window_months, step_months, mode, n_windows or "auto")
    logger.info("═" * 60)

    result = run_walk_forward_from_cli(
        symbols=symbols,
        period=period,
        interval=interval,
        strategy_name=strategy_name,
        model_path=model_path,
        window_months=window_months,
        step_months=step_months,
        n_windows=n_windows,
        mode=mode,
        max_positions=max_positions,
        corr_threshold=corr_threshold,
        position_size_pct=position_size_pct,
        cost_model_kind=cost_model_kind,
        commission_bps=commission_bps,
        spread_bps=spread_bps,
        save_result=save_result,
    )

    print(result.summary())
    print("\nDetalle por ventana:")
    df = result.to_dataframe()
    print(df[["window_id","start_date","end_date","sharpe","total_return_pct",
              "max_drawdown_pct","total_trades","win_rate"]].to_string(index=False))


# ═══════════════════════════════════════════════
#  MODO FEATURES  (vuelca el dataset tabular para ML)
# ═══════════════════════════════════════════════
def run_features(
    symbol: str,
    period: str,
    interval: str,
    out_path: Optional[str] = None,
    include_hurst: bool = True,
    use_cache: bool = True,
    include_sentiment: bool = False,
) -> None:
    """Descarga datos OHLCV y vuelca el DataFrame de features."""
    from data.fetcher import DataFetcher
    from features import FeatureBuilder, FeatureCache, FEATURE_VERSION

    logger.info("═" * 50)
    logger.info("  MODO FEATURES — %s %s %s (v%s)  sentiment=%s",
                symbol, interval, period, FEATURE_VERSION, include_sentiment)
    logger.info("═" * 50)

    df = DataFetcher.fetch_yahoo(ticker=symbol, period=period, interval=interval)
    if df.empty:
        logger.error("No se pudieron obtener datos para %s", symbol)
        sys.exit(1)
    # Anotar el símbolo para que sentiment lo encuentre.
    df.attrs["symbol"] = symbol

    builder = FeatureBuilder(
        include_hurst=include_hurst,
        include_sentiment=include_sentiment,
    )

    def _build():
        return builder.build(df, symbol=symbol)

    if use_cache:
        cache = FeatureCache()
        features = cache.get_or_build(
            symbol, interval, period, _build,
            extra=f"hurst={int(include_hurst)}",
        )
    else:
        features = _build()

    dropped = features.dropna()
    logger.info(
        "Features calculadas: shape=%s  filas válidas (dropna)=%d  cols=%d",
        features.shape, len(dropped), features.shape[1],
    )
    logger.info("Columnas: %s", list(features.columns))

    if out_path:
        p = Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        if p.suffix == ".parquet":
            try:
                features.to_parquet(p)
            except ImportError as exc:
                fallback = p.with_suffix(".csv")
                logger.warning(
                    "pyarrow/fastparquet no disponible (%s). Guardando como CSV en %s.",
                    exc, fallback,
                )
                logger.warning(
                    "Para habilitar parquet: `pip install -r requirements.txt`  (o `pip install pyarrow`)."
                )
                features.to_csv(fallback)
                p = fallback
        elif p.suffix == ".csv":
            features.to_csv(p)
        else:
            raise ValueError("--features-out debe terminar en .parquet o .csv")
        logger.info("Features guardadas en %s (%d bytes)", p, p.stat().st_size)

    # Resumen estadístico rápido (head + describe) al stdout. Usamos ASCII
    # puro en los títulos para evitar UnicodeEncodeError en consolas Windows
    # configuradas con cp1252.
    print("\n-- Primeras filas (no-NaN) --")
    print(dropped.head(3).round(4))
    print("\n-- Resumen estadistico --")
    print(dropped.describe().T[["count", "mean", "std", "min", "max"]].round(4))


# ═══════════════════════════════════════════════
#  MODO TRAIN (entrena meta-labeler LightGBM)
# ═══════════════════════════════════════════════
def run_train(
    symbol: Optional[str],
    period: str,
    interval: str,
    out_path: str,
    tp_mult: float = 2.0,
    sl_mult: float = 1.0,
    max_bars: int = 10,
    symbols: Optional[list] = None,
    regime_split: bool = False,
    threshold_objective: str = "sharpe",
    cv_method: str = "walk_forward",
    purge_bars: Optional[int] = None,
    include_sentiment: bool = False,
    tune_hp: bool = False,
    tune_trials: int = 50,
    tune_timeout: Optional[int] = None,
) -> None:
    """Entrena un MetaLabeler LightGBM sobre señales del ensemble y persiste.

    Modos:
      - single-symbol (P4)        → `symbol` + no split
      - basket (P4.1)             → `symbols` + no split
      - basket + regime-split (P4.2) → `symbols` + regime_split=True:
          entrena DOS modelos (trend-follow / mean-revert) en un único payload.

    Fase J (hyperparameter tuning):
      - `tune_hp=True` activa un sweep Optuna antes del fit final. Usa
        `tune_trials` combinaciones y `tune_timeout` como techo opcional.
      - Con regime_split, cada modelo (trend / mean_revert) se tunea por
        separado.
    """
    from data.fetcher import DataFetcher
    from features import FeatureBuilder
    from ml.meta_labeler import MetaLabelerTrainer, MetaLabelerConfig, save_meta_labeler

    logger.info("═" * 50)
    tag = []
    if symbols:
        tag.append("BASKET x%d" % len(symbols))
    if regime_split:
        tag.append("REGIME-SPLIT")
    label = " ".join(tag) if tag else "SINGLE"
    logger.info("  MODO TRAIN [%s] — %s %s %s  TP=%.2f SL=%.2f Max=%d  thr-obj=%s  cv=%s",
                label, symbol or "basket", interval, period, tp_mult, sl_mult, max_bars,
                threshold_objective, cv_method)
    logger.info("═" * 50)

    if regime_split and not symbols:
        logger.error("--regime-split requiere --symbols (entrenamiento sobre basket).")
        sys.exit(1)

    cfg = MetaLabelerConfig(
        tp_mult=tp_mult,
        sl_mult=sl_mult,
        max_bars=max_bars,
        threshold_objective=threshold_objective,
        cv_method=cv_method,
        purge_bars=purge_bars,
    )
    # B (sentiment): el FeatureBuilder se construye aquí para poder
    # inyectar `include_sentiment`. Si está activo, las columnas
    # `news_*` y `fng_*` se añaden al dataset y quedan codificadas en
    # `feature_cols` del payload — la inferencia las detecta y activa
    # automáticamente el mismo modo (ver MetaLabeledEnsembleStrategy).
    fb = FeatureBuilder(include_sentiment=include_sentiment)
    trainer = MetaLabelerTrainer(config=cfg, feature_builder=fb)

    # J — Tuning opcional de hiperparámetros con Optuna.
    # Si tune_hp es False, tune_config = None y el flujo es 100% igual
    # que antes (defaults del MetaLabelerConfig.lgb_params).
    tune_config = None
    if tune_hp:
        from ml.tuning import OptunaTunerConfig
        tune_config = OptunaTunerConfig(
            n_trials=tune_trials,
            timeout=tune_timeout,
            objective="auc",
        )
        logger.info(
            "J — Optuna tuning ACTIVADO: n_trials=%d%s",
            tune_trials,
            f", timeout={tune_timeout}s" if tune_timeout else "",
        )

    if symbols:
        dfs = {}
        for sym in symbols:
            df_s = DataFetcher.fetch_yahoo(ticker=sym, period=period, interval=interval)
            if df_s is None or df_s.empty:
                logger.warning("Sin datos para %s — se omite.", sym)
                continue
            dfs[sym] = df_s
            logger.info("  %-8s : %d filas [%s → %s]",
                        sym, len(df_s), df_s.index[0], df_s.index[-1])
        if not dfs:
            logger.error("Ningún símbolo del basket produjo datos.")
            sys.exit(1)
        result = (
            trainer.train_multi_regime_split(dfs, tune_config=tune_config)
            if regime_split
            else trainer.train_multi(dfs, tune_config=tune_config)
        )
    else:
        df = DataFetcher.fetch_yahoo(ticker=symbol, period=period, interval=interval)
        if df.empty:
            logger.error("No se pudieron obtener datos para %s", symbol)
            sys.exit(1)
        logger.info("Datos: %d filas [%s → %s]", len(df), df.index[0], df.index[-1])
        result = trainer.train(df, symbol=symbol, tune_config=tune_config)

    if result.get("kind") == "regime_split":
        # Reporting específico para el payload per-régimen.
        logger.info("Muestras por regimen: %s", result.get("samples_per_regime", {}))
        if result.get("samples_per_symbol"):
            logger.info("Muestras por simbolo: %s", result["samples_per_symbol"])
        for name, sub in result["regimes"].items():
            logger.info(
                "Regimen %-11s : %d muestras  base_rate=%.3f  thr=%.3f",
                name, sub["n_samples"], sub["base_rate_global"], sub["threshold"],
            )
            print(f"\n-- [{name}] Metricas por fold (cv={cv_method}) --")
            if sub["fold_metrics"]:
                print(pd.DataFrame(sub["fold_metrics"]).round(4).to_string(index=False))
            else:
                print("(sin folds — muestras insuficientes)")
            print(f"\n-- [{name}] Top 10 features por importancia --")
            for k, v in list(sub["feature_importance"].items())[:10]:
                print(f"  {k:<25}  {v}")
    else:
        logger.info("Muestras: %d  |  base_rate global: %.3f",
                    result["n_samples"], result["base_rate_global"])
        logger.info("Threshold elegido: %.3f", result["threshold"])
        if result.get("samples_per_symbol"):
            logger.info("Muestras por simbolo: %s", result["samples_per_symbol"])

        print(f"\n-- Metricas por fold (cv={cv_method}) --")
        if result["fold_metrics"]:
            folds_df = pd.DataFrame(result["fold_metrics"])
            print(folds_df.round(4).to_string(index=False))
        else:
            print("(sin folds — muestras insuficientes para splits internos)")

        print("\n-- Top 15 features por importancia --")
        top = list(result["feature_importance"].items())[:15]
        for k, v in top:
            print(f"  {k:<25}  {v}")

    path = save_meta_labeler(result, out_path)
    logger.info("Modelo guardado en %s (%d bytes)", path, path.stat().st_size)


# ═══════════════════════════════════════════════
#  MODO LIVE
# ═══════════════════════════════════════════════
# K — Helpers para auto-retrain y drift reference
# ═══════════════════════════════════════════════
def _load_drift_reference(
    model_path: str, symbol: str, strategy
) -> Optional["pd.DataFrame"]:
    """
    Intenta obtener la distribución de features de referencia para el
    detector de drift.

    Orden de preferencia:
      1. Si el pickle trae `X_reference_sample` (versión futura), usarlo.
      2. Fallback: bajar 2 años de datos del símbolo y reconstruir
         features con el mismo FeatureBuilder que la estrategia.

    Devuelve un DataFrame de features o None si todo falla.
    """
    try:
        import joblib
        payload = joblib.load(model_path)
        ref = payload.get("X_reference_sample") if isinstance(payload, dict) else None
        if ref is not None and hasattr(ref, "columns"):
            logger.info("Drift reference tomada del pickle (%d filas).", len(ref))
            return ref
    except Exception as exc:
        logger.debug("No pude leer X_reference_sample del pickle: %s", exc)

    # Fallback: reconstruir con el FeatureBuilder de la estrategia.
    try:
        from data.fetcher import DataFetcher
        df = DataFetcher.fetch_yahoo(ticker=symbol, period="2y", interval="1d")
        if df is None or df.empty:
            return None
        df.attrs["symbol"] = symbol
        infer = getattr(strategy, "inferencer", None)
        fb = getattr(infer, "feature_builder", None) if infer is not None else None
        if fb is None:
            from features import FeatureBuilder
            fb = FeatureBuilder()
        feats = fb.build(df, symbol=symbol)
        feats = feats.select_dtypes(include=[float, int])
        logger.info(
            "Drift reference reconstruida vía yfinance (%d filas, %d features).",
            len(feats), feats.shape[1],
        )
        return feats
    except Exception as exc:
        logger.warning("Fallback drift reference falló: %s", exc)
        return None


def _retrain_fn_for_orchestrator(**kwargs) -> dict:
    """
    Wrapper del pipeline de retrain para el `RetrainOrchestrator`.

    El orchestrator le pasa `out_path`, `symbol`, `interval`, `symbols`,
    `regime_split`, `threshold_objective`, `cv_method`,
    `include_sentiment`, `period`. Internamente reutilizamos
    `run_train` (single path con todas las capas) y al terminar cargamos
    el pickle para devolver el dict de fold_metrics al orchestrator
    (que calcula la AUC media para la decisión de promote).
    """
    import joblib
    # `run_train` escribe el pickle y no devuelve el result en memoria,
    # así que lo releemos desde disco. Es barato (KB).
    out_path = kwargs.pop("out_path")
    period = kwargs.pop("period", "5y")
    symbols = kwargs.pop("symbols", None)

    # run_train acepta exactamente estos args (ver firma arriba).
    run_train(
        symbol=kwargs.get("symbol", TRADING_SYMBOL),
        period=period,
        interval=kwargs.get("interval", "1d"),
        out_path=out_path,
        tp_mult=2.0,
        sl_mult=1.5,
        max_bars=10,
        symbols=symbols,
        regime_split=kwargs.get("regime_split", False),
        threshold_objective=kwargs.get("threshold_objective", "sharpe"),
        cv_method=kwargs.get("cv_method", "walk_forward"),
        include_sentiment=kwargs.get("include_sentiment", False),
    )
    # Releer el pickle para devolver fold_metrics al orchestrator.
    payload = joblib.load(out_path)
    return payload


# ═══════════════════════════════════════════════
def run_live(
    symbol: str = TRADING_SYMBOL,
    timeframe: str = TIMEFRAME,
    strategy_name: str = "meta_ensemble",
    model_path: Optional[str] = None,
    threshold_override: Optional[float] = None,
    dry_run: bool = False,
    data_source: str = "alpaca",
    alpaca_feed: str = "iex",
    risk_overlay: bool = False,
    target_vol: float = 0.20,
    max_daily_loss_pct: float = 0.0,
    disable_confidence_sizing: bool = False,
    base_position_size_pct: float = 2.0,
    stop_loss_pct: float = 0.0,
    take_profit_pct: float = 0.0,
    max_iters: Optional[int] = None,
    sleep_seconds: Optional[int] = None,
    state_path: Optional[str] = None,
    # ── K: Auto-retrain / drift detection ─────────────────────────
    auto_retrain: bool = False,
    retrain_cooldown_days: float = 7.0,
    retrain_period: str = "5y",
    retrain_symbols: Optional[list] = None,
    drift_psi_strong: float = 0.25,
    drift_auc_floor: float = 0.52,
    drift_check_every_iters: int = 20,
    drift_require_multi: bool = True,
) -> None:
    """
    Ejecuta el bot en modo live o paper trading.

    El "live" real requiere credenciales Alpaca en `config/secrets.env`. Con
    `dry_run=True` se puede validar la señal y el sizing sin enviar órdenes
    (y sin necesitar conexión Alpaca si `data_source="yahoo"`).

    Args:
        symbol              : ticker (ej. "AAPL", "BTC-USD").
        timeframe           : granularidad Alpaca-style (1Min/5Min/15Min/1Hour/1Day).
        strategy_name       : ensemble, meta_ensemble, rsi, mfi_rsi, donchian, rsi2_mr.
        model_path          : .pkl del meta-modelo (requerido para meta_ensemble).
        threshold_override  : override del threshold del meta-modelo global.
        dry_run             : si True, no envía órdenes (sólo logging).
        data_source         : "alpaca" (tiempo real) o "yahoo" (delay, free).
        risk_overlay        : activa el overlay (vol targeting + regime + confidence).
        target_vol          : vol anualizada objetivo para el overlay.
        max_daily_loss_pct  : kill-switch diario (%, 0 = off).
        disable_confidence_sizing : desactiva el componente de confidence en el overlay.
        base_position_size_pct   : % del capital base por trade (antes del multiplier).
        stop_loss_pct / take_profit_pct : % SL/TP intra-barra (0 = off).
        max_iters           : para testing; None = loop infinito.
        sleep_seconds       : override del intervalo entre iteraciones.
    """
    # Imports locales para no arrastrar deps si nunca se usa `live`.
    from execution.live_runner import LiveConfig, LiveRunner
    from strategies.ensemble import build_default_ensemble
    from strategies.donchian_trend import DonchianTrendStrategy
    from strategies.rsi2_mean_reversion import RSI2MeanReversionStrategy
    from strategies.meta_labeled_ensemble import build_meta_labeled_ensemble_from_file

    logger.info("═" * 50)
    logger.info("  MODO LIVE TRADING — %s %s  strategy=%s  dry_run=%s",
                symbol, timeframe, strategy_name, dry_run)
    logger.info("═" * 50)

    # ── Construir la estrategia ─────────────────────────────────────────
    if strategy_name == "meta_ensemble":
        if not model_path:
            logger.error("--strategy meta_ensemble requiere --model")
            sys.exit(1)
        strategy = build_meta_labeled_ensemble_from_file(model_path)
        if threshold_override is not None and hasattr(strategy, "threshold"):
            # El override es una red de seguridad: permite endurecer el
            # filtro en caliente sin retrain (ej. si el mercado cambió).
            strategy.threshold = float(threshold_override)
    elif strategy_name == "ensemble":
        strategy = build_default_ensemble()
    elif strategy_name == "donchian":
        strategy = DonchianTrendStrategy()
    elif strategy_name == "rsi2_mr":
        strategy = RSI2MeanReversionStrategy()
    elif strategy_name == "mfi_rsi":
        from strategies.mfi_rsi_strategy import MfiRsiStrategy
        strategy = MfiRsiStrategy()
    else:
        # Fallback al path histórico del bot (RSIStrategy).
        from strategies.rsi_strategy import RSIStrategy
        strategy = RSIStrategy()

    # ── Broker / storage ────────────────────────────────────────────────
    # Si estamos en dry_run Y usamos Yahoo como data source, no hace falta
    # ni siquiera abrir conexión Alpaca — útil para validar en máquinas
    # sin credenciales.
    broker = None
    storage = None
    risk_mgr = None
    if not (dry_run and data_source == "yahoo"):
        from execution.broker import Broker
        from data.storage import StorageManager
        from risk.manager import RiskManager
        try:
            broker = Broker()
            storage = StorageManager()
            risk_mgr = RiskManager()
        except Exception as exc:
            logger.error("No se pudo inicializar Broker/Storage: %s", exc)
            if not dry_run:
                sys.exit(1)

    # ── Risk overlay (opcional) ─────────────────────────────────────────
    overlay = None
    if risk_overlay:
        from risk.overlay import RiskConfig, RiskOverlay
        overlay_cfg = RiskConfig(
            target_vol=target_vol,
            max_daily_loss_pct=max_daily_loss_pct,
            use_confidence_sizing=not disable_confidence_sizing,
        )
        overlay = RiskOverlay(config=overlay_cfg)
        logger.info(
            "  RISK OVERLAY ACTIVO — target_vol=%.2f daily_loss=%.1f%% conf_sizing=%s",
            target_vol, max_daily_loss_pct, not disable_confidence_sizing,
        )

    # ── Telegram (notificaciones) ───────────────────────────────────────
    try:
        from utils.helpers import send_telegram_message
        telegram_fn = send_telegram_message
    except Exception:
        telegram_fn = None

    # ── Config del runner ───────────────────────────────────────────────
    # Mapeo de intervalos "Yahoo-style" (1d, 1h) a "Alpaca-style" para que
    # el usuario pueda pasar lo mismo que en backtest.
    yahoo_interval = "1h" if timeframe == "1Hour" else (
        "1d" if timeframe == "1Day" else "1h"
    )

    cfg = LiveConfig(
        symbol=symbol,
        timeframe=timeframe,
        data_source=data_source,
        alpaca_feed=alpaca_feed,
        dry_run=dry_run,
        base_position_size_pct=base_position_size_pct,
        stop_loss_pct=stop_loss_pct,
        take_profit_pct=take_profit_pct,
        max_iters=max_iters,
        sleep_seconds=sleep_seconds,
        yahoo_interval=yahoo_interval,
        state_path=state_path,
        drift_check_every_iters=drift_check_every_iters,
        # Si auto_retrain está activo, el LiveRunner también debe
        # vigilar el pickle para hot-reload tras un swap.
        model_reload_path=model_path if auto_retrain and model_path else None,
    )

    # ── K: Drift detector + retrain orchestrator ───────────────────────
    drift_detector = None
    retrain_orchestrator = None
    if auto_retrain:
        if not model_path:
            logger.error("--auto-retrain requiere --model.")
            sys.exit(1)
        if strategy_name != "meta_ensemble":
            logger.warning(
                "--auto-retrain sólo tiene sentido con strategy=meta_ensemble. "
                "Con otra estrategia, el monitor queda desactivado."
            )
        else:
            from ml.drift import DriftConfig, DriftDetector
            from ml.retrain import RetrainConfig, RetrainOrchestrator

            drift_detector = DriftDetector(DriftConfig(
                psi_strong=drift_psi_strong,
                auc_floor=drift_auc_floor,
                require_multiple_signals=drift_require_multi,
            ))
            # Reference = features del training. Las obtenemos desde el
            # pickle: guardamos X_reference_sample al entrenar, y si no
            # existe, caemos a un redownload rápido (puede fallar si no
            # hay conexión — en ese caso desactivamos el detector).
            ref_df = _load_drift_reference(
                model_path, symbol, strategy
            )
            if ref_df is not None and not ref_df.empty:
                drift_detector.fit_reference(ref_df)
            else:
                logger.warning(
                    "No pude cargar la referencia de drift; deshabilito el monitor."
                )
                drift_detector = None

            # Orchestrator: usa el mismo run_train que el modo CLI 'train'.
            # train_args replica los flags clave del entreno original.
            train_args = {
                "symbol": symbol,
                "interval": cfg.yahoo_interval if data_source == "yahoo" else "1d",
                "symbols": retrain_symbols or None,
                "regime_split": bool(retrain_symbols),
                "threshold_objective": "sharpe",
                "cv_method": "walk_forward",
                "include_sentiment": False,
            }
            retrain_orchestrator = RetrainOrchestrator(
                model_path=model_path,
                retrain_fn=_retrain_fn_for_orchestrator,
                config=RetrainConfig(
                    cooldown_days=retrain_cooldown_days,
                    retrain_period=retrain_period,
                    retrain_symbols=retrain_symbols or [],
                    train_args=train_args,
                    backup_old_pickle=True,
                    wait_result=False,
                    min_auc_to_promote=drift_auc_floor,
                ),
            )
            logger.info(
                "K — AUTO-RETRAIN ACTIVO (cooldown %.1fd, PSI>%.2f, AUC<%.2f, check cada %d iters)",
                retrain_cooldown_days, drift_psi_strong, drift_auc_floor,
                drift_check_every_iters,
            )

    runner = LiveRunner(
        strategy=strategy,
        config=cfg,
        broker=broker,
        storage=storage,
        risk_mgr=risk_mgr,
        overlay=overlay,
        telegram_fn=telegram_fn,
        drift_detector=drift_detector,
        retrain_orchestrator=retrain_orchestrator,
    )

    if telegram_fn and not dry_run:
        telegram_fn(f"🤖 <b>Trading Bot iniciado</b>\n"
                    f"Symbol: {symbol}  TF: {timeframe}\n"
                    f"Strategy: {strategy.name}")

    try:
        runner.run()
    finally:
        if storage is not None:
            try:
                storage.close()
            except Exception:
                pass


# ═══════════════════════════════════════════════
#  MODO LIVE PORTFOLIO (multi-symbol)
# ═══════════════════════════════════════════════
def run_live_portfolio(
    symbols: List[str],
    timeframe: str = TIMEFRAME,
    strategy_name: str = "meta_ensemble",
    model_path: Optional[str] = None,
    threshold_override: Optional[float] = None,
    dry_run: bool = False,
    data_source: str = "alpaca",
    alpaca_feed: str = "iex",
    risk_overlay: bool = False,
    target_vol: float = 0.20,
    disable_confidence_sizing: bool = False,
    position_size_pct: float = 20.0,
    max_positions: int = 4,
    corr_threshold: float = 0.75,
    corr_window: int = 60,
    stop_loss_pct: float = 0.0,
    take_profit_pct: float = 0.0,
    max_iters: Optional[int] = None,
    sleep_seconds: Optional[int] = None,
    state_path: Optional[str] = None,
) -> None:
    """
    Ejecuta el bot en modo live/paper sobre MÚLTIPLES símbolos.

    Un único proceso itera sobre todos los tickers, compartiendo capital
    y respetando max_positions + correlation guard.

    Args:
        symbols             : lista de tickers (ej. ["NVDA", "AAPL", "NFLX"]).
        max_positions       : máximo de posiciones simultáneas.
        corr_threshold      : umbral de correlación para rechazar entradas.
        position_size_pct   : % del equity total por entrada.
        (resto de args: idénticos a run_live).
    """
    from execution.portfolio_live_runner import PortfolioLiveConfig, PortfolioLiveRunner
    from strategies.ensemble import build_default_ensemble
    from strategies.donchian_trend import DonchianTrendStrategy
    from strategies.rsi2_mean_reversion import RSI2MeanReversionStrategy
    from strategies.meta_labeled_ensemble import build_meta_labeled_ensemble_from_file

    logger.info("═" * 60)
    logger.info("  MODO LIVE PORTFOLIO — %d símbolos  strategy=%s  dry_run=%s",
                len(symbols), strategy_name, dry_run)
    logger.info("  Símbolos: %s", ", ".join(symbols))
    logger.info("  Max positions: %d  Corr threshold: %.2f  Size: %.1f%%",
                max_positions, corr_threshold, position_size_pct)
    logger.info("═" * 60)

    # ── Construir la estrategia ─────────────────────────────────────────
    if strategy_name == "meta_ensemble":
        if not model_path:
            logger.error("--strategy meta_ensemble requiere --model")
            sys.exit(1)
        strategy = build_meta_labeled_ensemble_from_file(model_path)
        if threshold_override is not None and hasattr(strategy, "threshold"):
            strategy.threshold = float(threshold_override)
    elif strategy_name == "ensemble":
        strategy = build_default_ensemble()
    elif strategy_name == "donchian":
        strategy = DonchianTrendStrategy()
    elif strategy_name == "rsi2_mr":
        strategy = RSI2MeanReversionStrategy()
    elif strategy_name == "mfi_rsi":
        from strategies.mfi_rsi_strategy import MfiRsiStrategy
        strategy = MfiRsiStrategy()
    else:
        from strategies.rsi_strategy import RSIStrategy
        strategy = RSIStrategy()

    # ── Broker ──────────────────────────────────────────────────────────
    broker = None
    if not (dry_run and data_source == "yahoo"):
        from execution.broker import Broker
        try:
            broker = Broker()
        except Exception as exc:
            logger.error("No se pudo inicializar Broker: %s", exc)
            if not dry_run:
                sys.exit(1)

    # ── Risk overlay (opcional) ─────────────────────────────────────────
    overlay = None
    if risk_overlay:
        from risk.overlay import RiskConfig, RiskOverlay
        overlay_cfg = RiskConfig(
            target_vol=target_vol,
            use_confidence_sizing=not disable_confidence_sizing,
        )
        overlay = RiskOverlay(config=overlay_cfg)
        logger.info("  RISK OVERLAY ACTIVO — target_vol=%.2f", target_vol)

    # ── Telegram ────────────────────────────────────────────────────────
    try:
        from utils.helpers import send_telegram_message
        telegram_fn = send_telegram_message
    except Exception:
        telegram_fn = None

    # ── Config del runner ───────────────────────────────────────────────
    yahoo_interval = "1h" if timeframe == "1Hour" else (
        "1d" if timeframe == "1Day" else "1h"
    )

    cfg = PortfolioLiveConfig(
        symbols=symbols,
        timeframe=timeframe,
        data_source=data_source,
        alpaca_feed=alpaca_feed,
        dry_run=dry_run,
        position_size_pct=position_size_pct,
        max_positions=max_positions,
        corr_threshold=corr_threshold,
        corr_window=corr_window,
        stop_loss_pct=stop_loss_pct,
        take_profit_pct=take_profit_pct,
        max_iters=max_iters,
        sleep_seconds=sleep_seconds,
        yahoo_interval=yahoo_interval,
        state_path=state_path,
    )

    runner = PortfolioLiveRunner(
        strategy=strategy,
        config=cfg,
        broker=broker,
        overlay=overlay,
        telegram_fn=telegram_fn,
    )

    try:
        runner.run()
    except Exception:
        pass


# ═══════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════

# Yahoo Finance limita los datos historicos segun el intervalo:
INTERVAL_MAX_PERIOD = {
    "1m":  "7d",
    "2m":  "60d",
    "5m":  "60d",
    "15m": "60d",
    "30m": "60d",
    "60m": "730d",
    "1h":  "730d",
    "90m": "60d",
    "1d":  "max",
    "5d":  "max",
    "1wk": "max",
    "1mo": "max",
}


def main() -> None:
    # ──────────────────────────────────────────────────────────────────
    # Construcción del CLI con grupos lógicos para que `--help` sea
    # legible. Cada grupo agrupa flags relacionados con un modo o capa
    # del bot (datos, costes, ML, riesgo, live, auto-retrain, portfolio,
    # dashboard, etc.) en lugar de aparecer todos en una lista plana.
    # ──────────────────────────────────────────────────────────────────
    parser = argparse.ArgumentParser(
        prog="trading-bot",
        description=(
            "Motor de trading algorítmico modular con pipeline ML completo.\n"
            "\n"
            "MODOS DE EJECUCIÓN (--mode):\n"
            "  backtest   Backtest single-symbol con cualquier estrategia.\n"
            "  portfolio  Backtest multi-símbolo con correlation guard y cap de posiciones.\n"
            "  train      Entrena el meta-labeler LightGBM (triple-barrier + CV + threshold).\n"
            "  features   Vuelca el DataFrame de features a parquet/csv (útil para EDA).\n"
            "  live       Live / paper trading en Alpaca (o dry-run con yahoo).\n"
            "  dashboard  Lanza la UI Streamlit (Live Monitor + Backtest Review + Model Inspection).\n"
            "\n"
            "ESTRATEGIAS (--strategy):\n"
            "  rsi          RSI-2 Connors puro.\n"
            "  mfi_rsi      MFI + RSI reversal (estrategia base original).\n"
            "  donchian     Donchian channel breakout (trend-following).\n"
            "  rsi2_mr      RSI2 mean-reversion con SMA200 filter.\n"
            "  ensemble     Donchian trend + RSI2 mean-rev + régimen detector.\n"
            "  meta_ensemble  Ensemble + LightGBM meta-filter (RECOMENDADA, requiere --model).\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "INTERVALOS Y PERIODOS MÁXIMOS DE YAHOO FINANCE:\n"
            "  1m              max 7  días\n"
            "  2m, 5m, 15m     max 60 días\n"
            "  30m, 90m        max 60 días\n"
            "  60m / 1h        max 730 días (~2 años)\n"
            "  1d, 5d          sin límite\n"
            "  1wk, 1mo        sin límite\n"
            "\n"
            "EJEMPLOS POR MODO:\n"
            "\n"
            "  # Backtest single-symbol con risk overlay y cost model realista\n"
            "  python main.py --mode backtest --strategy meta_ensemble \\\n"
            "      --model models/v1.pkl --symbol NFLX --period 5y \\\n"
            "      --risk-overlay --target-vol 0.20 --cost-model realistic\n"
            "\n"
            "  # Portfolio backtest 7 símbolos con correlation guard\n"
            "  python main.py --mode portfolio --strategy meta_ensemble \\\n"
            "      --model models/v1.pkl \\\n"
            "      --symbols NFLX,DIS,KO,BTC-USD,INTC,PFE,NKE \\\n"
            "      --max-positions 4 --corr-threshold 0.75 --period 5y\n"
            "\n"
            "  # Train con regime split, sentiment, purged k-fold y Optuna tuning\n"
            "  python main.py --mode train \\\n"
            "      --symbols SPY,QQQ,AAPL,MSFT,NVDA,GOOGL,META,AMZN,TSLA \\\n"
            "      --period 5y --regime-split --threshold-objective sharpe \\\n"
            "      --cv-method purged_kfold --include-sentiment \\\n"
            "      --tune-hp --tune-trials 50 --model models/v2.pkl\n"
            "\n"
            "  # Paper / dry-run con auto-retrain y monitor de drift\n"
            "  python main.py --mode live --strategy meta_ensemble \\\n"
            "      --model models/v1.pkl --symbol AAPL --data-source yahoo --dry-run \\\n"
            "      --risk-overlay --auto-retrain --drift-check-every-iters 20 \\\n"
            "      --state-path /tmp/live.json\n"
            "\n"
            "  # Dashboard Streamlit (3 páginas)\n"
            "  python main.py --mode dashboard \\\n"
            "      --state-path /tmp/live.json --save-result /tmp/portfolio.pkl \\\n"
            "      --model models/v1.pkl\n"
        ),
    )
    # ── Grupos lógicos: agrupan flags por área para que --help sea legible ──
    g_general   = parser.add_argument_group("general",        "Modo, símbolo, estrategia, período e intervalo (válidos en casi todos los modos).")
    g_backtest  = parser.add_argument_group("backtest",       "Parámetros del modo --mode backtest (sizing, SL/TP, holding, filtros y outputs).")
    g_costs     = parser.add_argument_group("costes",         "Modelo de costes (flat retro-compat o realistic con spread+impact). Aplica a backtest y portfolio.")
    g_features  = parser.add_argument_group("features",       "Parámetros del modo --mode features (volcado de features para EDA).")
    g_train     = parser.add_argument_group("train (ML)",     "Entrenamiento del meta-labeler: triple-barrier, regime split, threshold tuning, CV y sentiment.")
    g_tune      = parser.add_argument_group("tuning Optuna",  "J — sweep de hiperparámetros LightGBM con TPE + MedianPruner antes del fit final.")
    g_risk      = parser.add_argument_group("risk overlay",   "P6 — multiplicador dinámico del sizing (vol target + regime + confidence) y kill-switch diario.")
    g_live      = parser.add_argument_group("live / paper",   "E — flags del modo --mode live (Alpaca o Yahoo, dry-run, sleep, snapshot para dashboard).")
    g_retrain   = parser.add_argument_group("auto-retrain (K)", "Detección de concept drift (KS+PSI+AUC) y reentreno automático en background con cooldown.")
    g_dashboard = parser.add_argument_group("dashboard (H)",  "H — UI Streamlit. --save-result/--state-path/--model conectan los datos.")
    g_portfolio = parser.add_argument_group("portfolio (G)",  "G — multi-símbolo con cap de posiciones, correlation guard y sizing compartido.")
    g_walkfwd   = parser.add_argument_group("walk-forward",   "Análisis de robustez: distribución de Sharpes sobre ventanas deslizantes.")

    # ── general ─────────────────────────────────────────────────────────
    g_general.add_argument(
        "--mode",
        choices=["live", "backtest", "features", "train", "portfolio", "dashboard", "walk_forward"],
        default="backtest",
        help="Modo de ejecución (default: backtest). Ver descripción del programa para qué hace cada uno.",
    )
    g_general.add_argument("--symbol", default="AAPL",
                           help="Ticker single-symbol (default: AAPL). Yahoo-style: 'AAPL', 'BTC-USD', 'EURUSD=X'. Ignorado si se pasa --symbols.")
    g_general.add_argument(
        "--strategy",
        choices=["rsi", "mfi_rsi", "donchian", "rsi2_mr", "ensemble", "meta_ensemble"],
        default="mfi_rsi",
        help="Estrategia a usar (default: mfi_rsi). 'meta_ensemble' es la recomendada y requiere --model.",
    )
    g_general.add_argument(
        "--period",
        default=None,
        help="Período de datos a descargar (ej: 7d, 60d, 6mo, 1y, 5y, max). Si se omite, se infiere del intervalo (ver epílogo).",
    )
    g_general.add_argument(
        "--interval",
        default="1d",
        choices=list(INTERVAL_MAX_PERIOD.keys()),
        help="Granularidad de la vela Yahoo-style (default: 1d). Ver epílogo para periodos máximos.",
    )
    # ── backtest ────────────────────────────────────────────────────────
    g_backtest.add_argument("--position-size-pct", type=float, default=100.0,
                            help="Tamaño de posición como %% del capital (default: 100). Usa 2.0 si quieres replicar el live.")
    g_backtest.add_argument("--stop-loss-pct", type=float, default=0.0,
                            help="Stop-loss en %% del precio de entrada. 0 = desactivado (default).")
    g_backtest.add_argument("--take-profit-pct", type=float, default=0.0,
                            help="Take-profit en %% del precio de entrada. 0 = desactivado (default).")
    g_backtest.add_argument("--max-holding-bars", type=int, default=MAX_HOLDING_BARS,
                            help="Máximo de barras en posición antes de salida forzada. 0 = ilimitado.")
    g_backtest.add_argument("--trend-filter", action="store_true",
                            help="Activa el filtro EMA de tendencia (sólo longs cuando precio > EMA(--trend-ema)).")
    g_backtest.add_argument("--trend-ema", type=int, default=TREND_EMA_PERIOD,
                            help=f"Periodo de la EMA del filtro de tendencia (default: {TREND_EMA_PERIOD}).")
    g_backtest.add_argument("--save-trades", default=None,
                            help="Path CSV para volcar los trades del backtest (opcional).")
    g_backtest.add_argument("--save-plot", default=None,
                            help="Path .png para guardar el gráfico equity-curve (opcional).")

    # ── costes ──────────────────────────────────────────────────────────
    g_costs.add_argument("--slippage-pct", type=float, default=SLIPPAGE_PCT,
                         help=f"Slippage aplicado en cada fill (default: {SLIPPAGE_PCT}%%). Sólo aplica con --cost-model flat.")
    g_costs.add_argument("--commission-pct", type=float, default=COMMISSION_PCT,
                         help=f"Comisión por operación (default: {COMMISSION_PCT}%%). Sólo aplica con --cost-model flat.")
    g_costs.add_argument("--cost-model", choices=["flat", "realistic"], default="flat",
                         help="I — modelo de costes. 'flat' (default, retro-compat) usa comisión y slippage constantes. "
                              "'realistic' aplica spread variable por hora + market impact raíz-cuadrática (Almgren-Chriss) sobre ADV 20d.")
    g_costs.add_argument("--commission-bps", type=float, default=1.0,
                         help="I — comisión en basis points cuando --cost-model=realistic (default: 1.0 bp, ~Alpaca/IBKR retail).")
    g_costs.add_argument("--spread-bps", type=float, default=4.0,
                         help="I — spread bid-ask base en bps (default: 4.0). Se multiplica por factor horario: apertura 1.5x, medio día 1.0x, cierre 1.3x.")
    g_costs.add_argument("--impact-coef", type=float, default=0.1,
                         help="I — coeficiente `k` del market impact (default: 0.1). impact_bps = k*10000*sqrt(notional/ADV). Con k=0.1 y participation=1%% da ~10 bps.")

    # ── features ────────────────────────────────────────────────────────
    g_features.add_argument("--features-out", default=None,
                            help="Path de salida del DataFrame de features (.parquet o .csv).")
    g_features.add_argument("--features-no-hurst", action="store_true",
                            help="Desactiva el cálculo del exponente de Hurst (caro en series largas).")
    g_features.add_argument("--features-no-cache", action="store_true",
                            help="Fuerza recálculo (ignora la cache en ~/.cache/trading-bot/features/).")

    # ── train ───────────────────────────────────────────────────────────
    g_train.add_argument("--model", default=None,
                         help="Path al .pkl del meta-labeler. ENTRADA para --strategy meta_ensemble; SALIDA para --mode train; entrada para dashboard --mode dashboard.")
    g_train.add_argument("--symbols", default=None,
                         help="Lista de tickers separados por coma para --mode train (basket training, anti-overfit). Ej: AAPL,MSFT,NVDA,SPY,GOOGL. Si se indica, ignora --symbol.")
    g_train.add_argument("--train-tp-mult", type=float, default=2.0,
                         help="Triple-barrier: multiplicador ATR del take-profit (default: 2.0).")
    g_train.add_argument("--train-sl-mult", type=float, default=1.0,
                         help="Triple-barrier: multiplicador ATR del stop-loss (default: 1.0).")
    g_train.add_argument("--train-max-bars", type=int, default=10,
                         help="Triple-barrier: barras máximas antes del timeout vertical (default: 10).")
    g_train.add_argument("--train-threshold", type=float, default=None,
                         help="Override del threshold de probabilidad meta_ensemble (default: threshold entrenado del pickle).")
    g_train.add_argument("--regime-split", action="store_true",
                         help="P4.2 — entrena DOS modelos separados (trend-follow y mean-revert) en un único pickle. Requiere --symbols.")
    g_train.add_argument("--threshold-objective", choices=["sharpe", "f1"], default="sharpe",
                         help="D — criterio del threshold del meta-modelo. 'sharpe' (default, maximiza Sharpe de retornos realizados) o 'f1' (legacy: max F1 de clasificación).")
    g_train.add_argument("--cv-method", choices=["walk_forward", "purged_kfold"], default="walk_forward",
                         help="F — método de CV del meta-labeler. 'walk_forward' (default, expanding+embargo) o 'purged_kfold' (López de Prado: K folds con purging+embargo, métricas más honestas pero rompe asimetría temporal).")
    g_train.add_argument("--purge-bars", type=int, default=None,
                         help="F — barras a purgar antes de cada test fold en purged_kfold (default: igual a --train-max-bars).")
    g_train.add_argument("--include-sentiment", action="store_true",
                         help="B — añade features de sentimiento al pipeline (VADER sobre yfinance news + crypto Fear&Greed). Sin keys externas; NaN para barras sin cobertura. La inferencia detecta automáticamente si el modelo fue entrenado con sentiment.")

    # ── tuning Optuna ───────────────────────────────────────────────────
    g_tune.add_argument("--tune-hp", action="store_true",
                        help="J — activa el sweep Optuna (TPE + MedianPruner) sobre los hiperparámetros de LightGBM antes del fit final. Con --regime-split, tunea cada régimen por separado.")
    g_tune.add_argument("--tune-trials", type=int, default=50,
                        help="J — nº de trials del sweep (default: 50). Mínimo útil 30, sweet spot 50-100.")
    g_tune.add_argument("--tune-timeout", type=int, default=None,
                        help="J — techo en segundos para el sweep completo (default: sin límite). Gana el primero que se alcance entre --tune-trials y --tune-timeout.")

    # ── risk overlay ────────────────────────────────────────────────────
    g_risk.add_argument("--risk-overlay", action="store_true",
                        help="P6 — activa el risk overlay (vol target × regime × confidence sizing) en backtest y live.")
    g_risk.add_argument("--target-vol", type=float, default=0.20,
                        help="Volatilidad anualizada objetivo del vol-targeting (default: 0.20 = 20%%).")
    g_risk.add_argument("--vol-mult-cap", type=float, default=1.50,
                        help="Tope del multiplicador por vol targeting (default: 1.5 = hasta 150%% del sizing base).")
    g_risk.add_argument("--max-daily-loss-pct", type=float, default=0.0,
                        help="Kill-switch diario: si pérdida intradía > X%%, bloquea BUYs hasta el día siguiente. 0 = desactivado.")
    g_risk.add_argument("--no-regime-sizing", action="store_true",
                        help="Desactiva el componente regime del risk overlay (CHOP=0.5x, MEAN_REVERT=0.7x).")
    g_risk.add_argument("--no-confidence-sizing", action="store_true",
                        help="Desactiva el componente confidence del risk overlay (mapeo lineal proba -> multiplicador).")

    # ── live / paper ────────────────────────────────────────────────────
    g_live.add_argument("--live-timeframe", default=TIMEFRAME,
                        help=f"E — timeframe Alpaca-style para --mode live (default: {TIMEFRAME}). Valores: 1Min, 5Min, 15Min, 1Hour, 1Day.")
    g_live.add_argument("--data-source", choices=["alpaca", "yahoo"], default="alpaca",
                        help="E — origen de datos en --mode live. 'alpaca' (real-time, requiere creds en config/secrets.env) o 'yahoo' (delay 15-20m, gratis, sin creds).")
    g_live.add_argument("--alpaca-feed", choices=["iex", "sip"], default="iex",
                        help="E — feed Alpaca para equities (default: iex = único gratis). 'sip' (consolidated tape) requiere plan de pago. Ignorado para crypto.")
    g_live.add_argument("--dry-run", action="store_true",
                        help="E — --mode live no envía órdenes reales: solo loguea la decisión y simula fills (útil para validar señales sin riesgo).")
    g_live.add_argument("--live-position-size-pct", type=float, default=2.0,
                        help="E — %% del capital por trade en --mode live (default: 2.0).")
    g_live.add_argument("--live-max-iters", type=int, default=None,
                        help="E — límite de iteraciones del loop live (default: None = infinito). Útil para tests cortos.")
    g_live.add_argument("--live-sleep", type=int, default=None,
                        help="E — override de los segundos de sleep entre iteraciones (default: derivado del timeframe).")
    g_live.add_argument("--state-path", default=None,
                        help="H — path de un JSON donde el LiveRunner escribe un snapshot por iteración. Lo lee el dashboard.")

    # ── auto-retrain (K) ────────────────────────────────────────────────
    g_retrain.add_argument("--auto-retrain", action="store_true",
                           help="K — activa el monitor de drift y el reentreno automático en background. Requiere --model y --strategy meta_ensemble. Opt-in (default off).")
    g_retrain.add_argument("--retrain-cooldown-days", type=float, default=7.0,
                           help="K — días mínimos entre reentrenos automáticos (default: 7). Evita thrashing ante ruido estadístico.")
    g_retrain.add_argument("--retrain-period", default="5y",
                           help="K — período Yahoo a usar para el retrain automático (default: 5y).")
    g_retrain.add_argument("--retrain-symbols", default=None,
                           help="K — basket de símbolos (CSV) para el retrain automático. Si se omite, reentrena sobre el --symbol del live.")
    g_retrain.add_argument("--drift-psi-strong", type=float, default=0.25,
                           help="K — threshold PSI para 'drift fuerte' por feature (default: 0.25, estándar industria).")
    g_retrain.add_argument("--drift-auc-floor", type=float, default=0.52,
                           help="K — AUC rolling mínima aceptable del modelo en producción (default: 0.52). Por debajo, edge perdido.")
    g_retrain.add_argument("--drift-check-every-iters", type=int, default=20,
                           help="K — cada cuántos iters ejecutar el chequeo de drift (default: 20). Evita coste KS+PSI en cada barra.")
    g_retrain.add_argument("--drift-single-signal", action="store_true",
                           help="K — si se pasa, basta con UNA señal (KS, PSI o AUC) para disparar retrain. Por defecto se requieren >=2 — más conservador.")

    # ── dashboard (H) ───────────────────────────────────────────────────
    g_dashboard.add_argument("--save-result", default=None,
                             help="H — path para persistir el resultado de backtest/portfolio (pickle). Útil para revisarlo después en el dashboard.")
    g_dashboard.add_argument("--dashboard-port", type=int, default=8501,
                             help="H — puerto local para --mode dashboard (default: 8501).")

    # ── portfolio (G) ───────────────────────────────────────────────────
    g_portfolio.add_argument("--max-positions", type=int, default=5,
                             help="G — nº máximo de posiciones simultáneas en --mode portfolio (default: 5).")
    g_portfolio.add_argument("--corr-threshold", type=float, default=0.75,
                             help="G — umbral de correlación rolling para rechazar entradas correlacionadas (default: 0.75). >1 desactiva el guard.")
    g_portfolio.add_argument("--corr-window", type=int, default=60,
                             help="G — ventana en barras de la matriz de correlación rolling (default: 60).")
    g_portfolio.add_argument("--portfolio-position-size-pct", type=float, default=20.0,
                             help="G — %% del equity total invertido en cada nueva entrada (default: 20%%). Con max_positions=5 se llega a ~100%% sin apalancar.")

    # ── walk-forward ────────────────────────────────────────────────────
    g_walkfwd.add_argument("--wf-window-months", type=int, default=6,
                           help="Tamaño de cada ventana de test en meses (default: 6).")
    g_walkfwd.add_argument("--wf-step-months", type=int, default=3,
                           help="Desplazamiento entre ventanas en meses (default: 3).")
    g_walkfwd.add_argument("--wf-n-windows", type=int, default=None,
                           help="Máximo de ventanas a evaluar (default: todas las que quepan).")
    g_walkfwd.add_argument("--wf-mode", choices=["rolling","expanding"], default="rolling",
                           help="Modo de ventana: 'rolling' (fija) o 'expanding' (crece desde inicio).")

    args = parser.parse_args()

    if args.mode == "backtest":
        # Auto-seleccionar periodo si no se indico
        period = args.period
        if period is None:
            period = INTERVAL_MAX_PERIOD.get(args.interval, "1y")
            logger.info("Periodo auto-seleccionado: %s (max para %s)", period, args.interval)

        run_backtest(
            symbol=args.symbol,
            period=period,
            interval=args.interval,
            strategy_name=args.strategy,
            position_size_pct=args.position_size_pct,
            stop_loss_pct=args.stop_loss_pct,
            take_profit_pct=args.take_profit_pct,
            slippage_pct=args.slippage_pct,
            commission_pct=args.commission_pct,
            max_holding_bars=args.max_holding_bars,
            use_trend_filter=args.trend_filter or USE_TREND_FILTER,
            trend_ema_period=args.trend_ema,
            save_trades=args.save_trades,
            save_plot=args.save_plot,
            model_path=args.model,
            threshold_override=args.train_threshold,
            risk_overlay=args.risk_overlay,
            target_vol=args.target_vol,
            vol_mult_cap=args.vol_mult_cap,
            max_daily_loss_pct=args.max_daily_loss_pct,
            use_regime_sizing=not args.no_regime_sizing,
            use_confidence_sizing=not args.no_confidence_sizing,
            save_result=args.save_result,
            cost_model_kind=args.cost_model,
            commission_bps=args.commission_bps,
            spread_bps=args.spread_bps,
            impact_coef=args.impact_coef,
        )
    elif args.mode == "train":
        if not args.model:
            logger.error("--mode train requiere --model /ruta/salida.pkl")
            sys.exit(2)
        period = args.period
        if period is None:
            period = INTERVAL_MAX_PERIOD.get(args.interval, "5y")
            logger.info("Periodo auto-seleccionado: %s (max para %s)", period, args.interval)
        symbols_list = None
        if args.symbols:
            symbols_list = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
        run_train(
            symbol=args.symbol,
            period=period,
            interval=args.interval,
            out_path=args.model,
            tp_mult=args.train_tp_mult,
            sl_mult=args.train_sl_mult,
            max_bars=args.train_max_bars,
            symbols=symbols_list,
            regime_split=args.regime_split,
            threshold_objective=args.threshold_objective,
            cv_method=args.cv_method,
            purge_bars=args.purge_bars,
            include_sentiment=args.include_sentiment,
            tune_hp=args.tune_hp,
            tune_trials=args.tune_trials,
            tune_timeout=args.tune_timeout,
        )
    elif args.mode == "features":
        period = args.period
        if period is None:
            period = INTERVAL_MAX_PERIOD.get(args.interval, "1y")
            logger.info("Periodo auto-seleccionado: %s (max para %s)", period, args.interval)
        run_features(
            symbol=args.symbol,
            period=period,
            interval=args.interval,
            out_path=args.features_out,
            include_hurst=not args.features_no_hurst,
            use_cache=not args.features_no_cache,
            include_sentiment=args.include_sentiment,
        )
    elif args.mode == "portfolio":
        # G — Backtest multi-symbol.
        period = args.period
        if period is None:
            period = INTERVAL_MAX_PERIOD.get(args.interval, "1y")
            logger.info("Periodo auto-seleccionado: %s (max para %s)", period, args.interval)
        if not args.symbols:
            logger.error("--mode portfolio requiere --symbols (lista separada por comas).")
            sys.exit(1)
        symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
        run_portfolio(
            symbols=symbols,
            period=period,
            interval=args.interval,
            strategy_name=args.strategy,
            model_path=args.model,
            threshold_override=args.train_threshold,
            position_size_pct=args.portfolio_position_size_pct,
            stop_loss_pct=args.stop_loss_pct,
            take_profit_pct=args.take_profit_pct,
            slippage_pct=args.slippage_pct,
            commission_pct=args.commission_pct,
            max_holding_bars=args.max_holding_bars,
            max_positions=args.max_positions,
            corr_threshold=args.corr_threshold,
            corr_window=args.corr_window,
            save_result=args.save_result,
            cost_model_kind=args.cost_model,
            commission_bps=args.commission_bps,
            spread_bps=args.spread_bps,
            impact_coef=args.impact_coef,
        )
    elif args.mode == "walk_forward":
        if not args.symbols:
            logger.error("--mode walk_forward requiere --symbols")
            sys.exit(1)
        period = args.period
        if period is None:
            period = INTERVAL_MAX_PERIOD.get(args.interval, "5y")
        symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
        run_walk_forward_mode(
            symbols=symbols,
            period=period,
            interval=args.interval,
            strategy_name=args.strategy,
            model_path=args.model,
            window_months=args.wf_window_months,
            step_months=args.wf_step_months,
            n_windows=args.wf_n_windows,
            mode=args.wf_mode,
            max_positions=args.max_positions,
            corr_threshold=args.corr_threshold,
            position_size_pct=args.portfolio_position_size_pct,
            cost_model_kind=args.cost_model,
            commission_bps=args.commission_bps,
            spread_bps=args.spread_bps,
            save_result=args.save_result,
        )
    elif args.mode == "dashboard":
        # H — Lanza el dashboard Streamlit. El binario `streamlit` debe
        # estar instalado en el venv (incluido en requirements.txt).
        import subprocess
        from pathlib import Path

        app_path = Path(__file__).parent / "dashboard" / "app.py"
        if not app_path.exists():
            logger.error("No encuentro %s", app_path)
            sys.exit(2)
        cmd = [
            sys.executable, "-m", "streamlit", "run", str(app_path),
            "--server.port", str(args.dashboard_port),
            "--server.headless", "true",
            "--browser.gatherUsageStats", "false",
            "--",
        ]
        # Pasar paths configurados al app vía args posicionales.
        if args.state_path:
            cmd += ["--state-path", args.state_path]
        if args.save_result:
            cmd += ["--result-path", args.save_result]
        if args.model:
            cmd += ["--model-path", args.model]
        logger.info("Arrancando dashboard: %s", " ".join(cmd))
        subprocess.run(cmd, check=False)
    elif args.mode == "live":
        # Si se pasan --symbols, usar el runner multi-símbolo (portfolio live).
        if args.symbols:
            symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
            if len(symbols) < 2:
                logger.warning("--symbols con 1 solo símbolo — usando runner single-symbol.")
                # Caer al path single-symbol con ese símbolo.
                run_live(
                    symbol=symbols[0],
                    timeframe=args.live_timeframe,
                    strategy_name=args.strategy,
                    model_path=args.model,
                    threshold_override=args.train_threshold,
                    dry_run=args.dry_run,
                    data_source=args.data_source,
                    alpaca_feed=args.alpaca_feed,
                    risk_overlay=args.risk_overlay,
                    target_vol=args.target_vol,
                    max_daily_loss_pct=args.max_daily_loss_pct,
                    disable_confidence_sizing=args.no_confidence_sizing,
                    base_position_size_pct=args.live_position_size_pct,
                    stop_loss_pct=args.stop_loss_pct,
                    take_profit_pct=args.take_profit_pct,
                    max_iters=args.live_max_iters,
                    sleep_seconds=args.live_sleep,
                    state_path=args.state_path,
                    auto_retrain=args.auto_retrain,
                    retrain_cooldown_days=args.retrain_cooldown_days,
                    retrain_period=args.retrain_period,
                    retrain_symbols=None,
                    drift_psi_strong=args.drift_psi_strong,
                    drift_auc_floor=args.drift_auc_floor,
                    drift_check_every_iters=args.drift_check_every_iters,
                    drift_require_multi=(not args.drift_single_signal),
                )
            else:
                run_live_portfolio(
                    symbols=symbols,
                    timeframe=args.live_timeframe,
                    strategy_name=args.strategy,
                    model_path=args.model,
                    threshold_override=args.train_threshold,
                    dry_run=args.dry_run,
                    data_source=args.data_source,
                    alpaca_feed=args.alpaca_feed,
                    risk_overlay=args.risk_overlay,
                    target_vol=args.target_vol,
                    disable_confidence_sizing=args.no_confidence_sizing,
                    position_size_pct=args.portfolio_position_size_pct,
                    max_positions=args.max_positions,
                    corr_threshold=args.corr_threshold,
                    corr_window=args.corr_window,
                    stop_loss_pct=args.stop_loss_pct,
                    take_profit_pct=args.take_profit_pct,
                    max_iters=args.live_max_iters,
                    sleep_seconds=args.live_sleep,
                    state_path=args.state_path,
                )
        else:
            run_live(
                symbol=args.symbol,
                timeframe=args.live_timeframe,
                strategy_name=args.strategy,
                model_path=args.model,
                threshold_override=args.train_threshold,
                dry_run=args.dry_run,
                data_source=args.data_source,
                alpaca_feed=args.alpaca_feed,
                risk_overlay=args.risk_overlay,
                target_vol=args.target_vol,
                max_daily_loss_pct=args.max_daily_loss_pct,
                disable_confidence_sizing=args.no_confidence_sizing,
                base_position_size_pct=args.live_position_size_pct,
                stop_loss_pct=args.stop_loss_pct,
                take_profit_pct=args.take_profit_pct,
                max_iters=args.live_max_iters,
                sleep_seconds=args.live_sleep,
                state_path=args.state_path,
                auto_retrain=args.auto_retrain,
                retrain_cooldown_days=args.retrain_cooldown_days,
                retrain_period=args.retrain_period,
                retrain_symbols=(
                    [s.strip().upper() for s in args.retrain_symbols.split(",") if s.strip()]
                    if args.retrain_symbols else None
                ),
                drift_psi_strong=args.drift_psi_strong,
                drift_auc_floor=args.drift_auc_floor,
                drift_check_every_iters=args.drift_check_every_iters,
                drift_require_multi=(not args.drift_single_signal),
            )


if __name__ == "__main__":
    main()
