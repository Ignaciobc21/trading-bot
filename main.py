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
from typing import Dict, Optional

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
) -> None:
    """Descarga datos y ejecuta un backtest con la estrategia seleccionada."""
    from data.fetcher import DataFetcher
    from backtesting.engine import BacktestEngine

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
    )
    result = engine.run(df, lookback=lookback, size_multiplier=size_multiplier)

    # 4. Mostrar resultados
    print(result.summary())

    # 5. Persistencia opcional
    if save_trades:
        engine.save_trades_csv(result, save_trades)

    # 6. Graficar
    engine.plot_results(result, save_path=save_plot)


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
) -> None:
    """
    G — Backtest multi-símbolo con capital compartido y correlation guard.

    Reutiliza la misma `BaseStrategy` que el backtest single-symbol; la
    única diferencia es el motor: `PortfolioBacktestEngine` itera todos
    los símbolos en paralelo sobre un índice maestro común.
    """
    from data.fetcher import DataFetcher
    from backtesting.portfolio_engine import PortfolioBacktestEngine, PortfolioConfig

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
    )
    engine = PortfolioBacktestEngine(strategy=strategy, config=cfg)
    # El motor ya llama a `logger.info(result.summary())` internamente.
    # No imprimimos por stdout para evitar UnicodeEncodeError en consolas
    # Windows con codepage cp1252 (los caracteres `═` y `─` no encajan).
    engine.run(data, lookback=lookback)


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
) -> None:
    """Entrena un MetaLabeler LightGBM sobre señales del ensemble y persiste.

    Modos:
      - single-symbol (P4)        → `symbol` + no split
      - basket (P4.1)             → `symbols` + no split
      - basket + regime-split (P4.2) → `symbols` + regime_split=True:
          entrena DOS modelos (trend-follow / mean-revert) en un único payload.
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
        result = trainer.train_multi_regime_split(dfs) if regime_split else trainer.train_multi(dfs)
    else:
        df = DataFetcher.fetch_yahoo(ticker=symbol, period=period, interval=interval)
        if df.empty:
            logger.error("No se pudieron obtener datos para %s", symbol)
            sys.exit(1)
        logger.info("Datos: %d filas [%s → %s]", len(df), df.index[0], df.index[-1])
        result = trainer.train(df, symbol=symbol)

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
def run_live(
    symbol: str = TRADING_SYMBOL,
    timeframe: str = TIMEFRAME,
    strategy_name: str = "meta_ensemble",
    model_path: Optional[str] = None,
    threshold_override: Optional[float] = None,
    dry_run: bool = False,
    data_source: str = "alpaca",
    risk_overlay: bool = False,
    target_vol: float = 0.20,
    max_daily_loss_pct: float = 0.0,
    disable_confidence_sizing: bool = False,
    base_position_size_pct: float = 2.0,
    stop_loss_pct: float = 0.0,
    take_profit_pct: float = 0.0,
    max_iters: Optional[int] = None,
    sleep_seconds: Optional[int] = None,
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
        dry_run=dry_run,
        base_position_size_pct=base_position_size_pct,
        stop_loss_pct=stop_loss_pct,
        take_profit_pct=take_profit_pct,
        max_iters=max_iters,
        sleep_seconds=sleep_seconds,
        yahoo_interval=yahoo_interval,
    )

    runner = LiveRunner(
        strategy=strategy,
        config=cfg,
        broker=broker,
        storage=storage,
        risk_mgr=risk_mgr,
        overlay=overlay,
        telegram_fn=telegram_fn,
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
    parser = argparse.ArgumentParser(
        description="Trading Bot -- RSI Strategy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Intervalos disponibles y periodo maximo de Yahoo Finance:\n"
            "  1m         max 7 dias\n"
            "  2m,5m,15m  max 60 dias\n"
            "  30m,90m    max 60 dias\n"
            "  60m / 1h   max 730 dias\n"
            "  1d,5d      sin limite\n"
            "  1wk,1mo    sin limite\n"
            "\n"
            "Ejemplos:\n"
            "  python main.py --mode backtest --symbol AAPL --interval 1d --period 1y\n"
            "  python main.py --mode backtest --symbol TSLA --interval 1h --period 30d\n"
            "  python main.py --mode backtest --symbol NVDA --interval 15m --period 5d\n"
            "  python main.py --mode backtest --symbol AMZN --interval 1m --period 5d\n"
        ),
    )
    parser.add_argument(
        "--mode",
        choices=["live", "backtest", "features", "train", "portfolio"],
        default="backtest",
        help="Modo de ejecucion (default: backtest). 'features' vuelca el DataFrame; 'train' entrena meta-labeler ML.",
    )
    parser.add_argument("--symbol", default="AAPL", help="Simbolo (default: AAPL)")
    parser.add_argument(
        "--strategy",
        choices=["rsi", "mfi_rsi", "donchian", "rsi2_mr", "ensemble", "meta_ensemble"],
        default="mfi_rsi",
        help="Estrategia a usar (default: mfi_rsi). 'meta_ensemble' requiere --model.",
    )
    parser.add_argument(
        "--period",
        default=None,
        help="Periodo de datos (ej: 7d, 60d, 6mo, 1y, max). Si no se indica, se elige automaticamente.",
    )
    parser.add_argument(
        "--interval",
        default="1d",
        choices=list(INTERVAL_MAX_PERIOD.keys()),
        help="Intervalo de vela (default: 1d)",
    )
    # ── Parámetros de riesgo / ejecución (backtest) ──
    parser.add_argument("--position-size-pct", type=float, default=100.0,
                        help="Tamaño de posición como %% del capital (default: 100; usa 2.0 para replicar live)")
    parser.add_argument("--stop-loss-pct", type=float, default=0.0,
                        help="Stop-loss en %% del precio de entrada (default: 0 = desactivado)")
    parser.add_argument("--take-profit-pct", type=float, default=0.0,
                        help="Take-profit en %% del precio de entrada (default: 0 = desactivado)")
    parser.add_argument("--slippage-pct", type=float, default=SLIPPAGE_PCT,
                        help=f"Slippage aplicado en cada fill (default: {SLIPPAGE_PCT}%%)")
    parser.add_argument("--commission-pct", type=float, default=COMMISSION_PCT,
                        help=f"Comisión por operación (default: {COMMISSION_PCT}%%)")
    parser.add_argument("--max-holding-bars", type=int, default=MAX_HOLDING_BARS,
                        help="Máximo de barras en posición (0 = ilimitado)")
    parser.add_argument("--trend-filter", action="store_true",
                        help="Activa el filtro EMA de tendencia (solo longs cuando precio > EMA)")
    parser.add_argument("--trend-ema", type=int, default=TREND_EMA_PERIOD,
                        help=f"Periodo de la EMA del filtro de tendencia (default: {TREND_EMA_PERIOD})")
    parser.add_argument("--save-trades", default=None,
                        help="Ruta CSV donde guardar los trades del backtest")
    parser.add_argument("--save-plot", default=None,
                        help="Ruta (png) donde guardar el gráfico del backtest")
    # ── Parámetros del modo features ──
    parser.add_argument("--features-out", default=None,
                        help="Ruta de salida del DataFrame de features (extensión .parquet / .csv)")
    parser.add_argument("--features-no-hurst", action="store_true",
                        help="Desactiva el cálculo del exponente de Hurst (caro en series largas)")
    parser.add_argument("--features-no-cache", action="store_true",
                        help="Fuerza recálculo (ignora la cache en ~/.cache/trading-bot/features/)")
    # ── Modo train / meta_ensemble ──
    parser.add_argument("--model", default=None,
                        help="Ruta al modelo .pkl (entrada para --strategy meta_ensemble, salida para --mode train)")
    parser.add_argument("--symbols", default=None,
                        help="Lista de tickers separados por coma para --mode train (basket training). "
                             "Ej: AAPL,MSFT,NVDA,SPY,GOOGL. Si se indica, ignora --symbol.")
    parser.add_argument("--train-tp-mult", type=float, default=2.0,
                        help="Multiplicador ATR del take-profit en triple-barrier (default: 2.0)")
    parser.add_argument("--train-sl-mult", type=float, default=1.0,
                        help="Multiplicador ATR del stop-loss en triple-barrier (default: 1.0)")
    parser.add_argument("--train-max-bars", type=int, default=10,
                        help="Barras máximas antes del timeout en triple-barrier (default: 10)")
    parser.add_argument("--train-threshold", type=float, default=None,
                        help="Override del threshold de probabilidad para meta_ensemble (default: threshold entrenado)")
    parser.add_argument("--regime-split", action="store_true",
                        help="P4.2: entrena dos modelos separados (trend-follow y mean-revert) "
                             "en un único .pkl. Requiere --symbols.")
    parser.add_argument("--threshold-objective", choices=["sharpe", "f1"], default="sharpe",
                        help="D: criterio de elección del threshold del meta-modelo. "
                             "'sharpe' (default) maximiza Sharpe de los retornos realizados; "
                             "'f1' replica el comportamiento previo (max F1 de clasificación).")
    # ── F: CV method ──
    parser.add_argument("--cv-method", choices=["walk_forward", "purged_kfold"], default="walk_forward",
                        help="F: método de validación cruzada del meta-labeler. "
                             "'walk_forward' (default, expanding window con embargo). "
                             "'purged_kfold' (López de Prado: K folds disjuntos con purging+embargo). "
                             "purged_kfold da métricas CV más honestas pero rompe la asimetría temporal — "
                             "útil para auditar overfit, no para deploys finales.")
    parser.add_argument("--purge-bars", type=int, default=None,
                        help="F: barras a purgar antes de cada test fold en --cv-method purged_kfold "
                             "(default: igual a --train-max-bars, el horizonte del triple-barrier).")
    # ── B: Sentiment ──
    parser.add_argument("--include-sentiment", action="store_true",
                        help="B: añade features de sentimiento al pipeline (VADER sobre yfinance "
                             "news + crypto Fear & Greed). Sin keys externas. NaN para barras sin "
                             "cobertura. La inferencia detecta automáticamente si el modelo "
                             "guardado fue entrenado con sentiment y activa el modo correspondiente.")
    # ── P6: Risk overlay ──
    parser.add_argument("--risk-overlay", action="store_true",
                        help="P6: activa el risk overlay (vol targeting + regime + confidence sizing).")
    parser.add_argument("--target-vol", type=float, default=0.20,
                        help="Volatilidad anualizada objetivo para el vol-targeting (default: 0.20 = 20%%).")
    parser.add_argument("--vol-mult-cap", type=float, default=1.50,
                        help="Tope del multiplicador por vol targeting (default: 1.5 = hasta 150%% del sizing base).")
    parser.add_argument("--max-daily-loss-pct", type=float, default=0.0,
                        help="Kill-switch diario: si la pérdida intradía supera X%%, bloquea BUYs hasta el día siguiente. 0 = desactivado.")
    parser.add_argument("--no-regime-sizing", action="store_true",
                        help="Desactiva el escalado por régimen dentro del risk overlay.")
    parser.add_argument("--no-confidence-sizing", action="store_true",
                        help="Desactiva el escalado por confianza del meta-modelo.")
    # ── E: Live / paper trading ──
    parser.add_argument("--live-timeframe", default=TIMEFRAME,
                        help=f"E: timeframe Alpaca-style para --mode live (default: {TIMEFRAME}). "
                             f"Valores: 1Min, 5Min, 15Min, 1Hour, 1Day.")
    parser.add_argument("--data-source", choices=["alpaca", "yahoo"], default="alpaca",
                        help="E: origen de datos en --mode live. 'alpaca' (real-time, requiere creds) "
                             "o 'yahoo' (delay 15-20m, gratis).")
    parser.add_argument("--dry-run", action="store_true",
                        help="E: --mode live no envía órdenes reales (simulación con logging).")
    parser.add_argument("--live-position-size-pct", type=float, default=2.0,
                        help="E: % del capital por trade en --mode live (default: 2.0).")
    parser.add_argument("--live-max-iters", type=int, default=None,
                        help="E: limite de iteraciones (None = infinito). Útil para test.")
    parser.add_argument("--live-sleep", type=int, default=None,
                        help="E: override de los segundos de sleep entre iteraciones.")
    # ── G: Portfolio (multi-symbol) ──
    parser.add_argument("--max-positions", type=int, default=5,
                        help="G: nº máximo de posiciones simultáneas en --mode portfolio (default: 5).")
    parser.add_argument("--corr-threshold", type=float, default=0.75,
                        help="G: umbral de correlación rolling media para rechazar nuevas entradas "
                             "que solapen con las posiciones abiertas (default: 0.75). Negativo o "
                             "muy alto desactiva el guard.")
    parser.add_argument("--corr-window", type=int, default=60,
                        help="G: ventana en barras para la matriz de correlación rolling (default: 60).")
    parser.add_argument("--portfolio-position-size-pct", type=float, default=20.0,
                        help="G: % del equity total invertido en cada nueva entrada (default: 20%%). "
                             "Con max_positions=5 se llega a ~100%% sin apalancar.")

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
        )
    elif args.mode == "live":
        run_live(
            symbol=args.symbol,
            timeframe=args.live_timeframe,
            strategy_name=args.strategy,
            model_path=args.model,
            threshold_override=args.train_threshold,
            dry_run=args.dry_run,
            data_source=args.data_source,
            risk_overlay=args.risk_overlay,
            target_vol=args.target_vol,
            max_daily_loss_pct=args.max_daily_loss_pct,
            disable_confidence_sizing=args.no_confidence_sizing,
            base_position_size_pct=args.live_position_size_pct,
            stop_loss_pct=args.stop_loss_pct,
            take_profit_pct=args.take_profit_pct,
            max_iters=args.live_max_iters,
            sleep_seconds=args.live_sleep,
        )


if __name__ == "__main__":
    main()
