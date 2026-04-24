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
from typing import Optional

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

    logger.info("Datos descargados: %d filas [%s → %s]", len(df), df.index[0], df.index[-1])

    # 2. Configurar estrategia (vía factoría)
    strategy, lookback = _build_strategy(
        strategy_name, use_trend_filter, trend_ema_period,
        model_path=model_path, threshold_override=threshold_override,
    )

    # 3. Ejecutar backtest
    engine = BacktestEngine(
        strategy=strategy,
        initial_capital=INITIAL_CAPITAL,
        commission_pct=commission_pct,
        slippage_pct=slippage_pct,
        position_size_pct=position_size_pct,
        stop_loss_pct=stop_loss_pct,
        take_profit_pct=take_profit_pct,
        max_holding_bars=max_holding_bars,
    )
    result = engine.run(df, lookback=lookback)

    # 4. Mostrar resultados
    print(result.summary())

    # 5. Persistencia opcional
    if save_trades:
        engine.save_trades_csv(result, save_trades)

    # 6. Graficar
    engine.plot_results(result, save_path=save_plot)


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
) -> None:
    """Descarga datos OHLCV y vuelca el DataFrame de features."""
    from data.fetcher import DataFetcher
    from features import FeatureBuilder, FeatureCache, FEATURE_VERSION

    logger.info("═" * 50)
    logger.info("  MODO FEATURES — %s %s %s (v%s)", symbol, interval, period, FEATURE_VERSION)
    logger.info("═" * 50)

    df = DataFetcher.fetch_yahoo(ticker=symbol, period=period, interval=interval)
    if df.empty:
        logger.error("No se pudieron obtener datos para %s", symbol)
        sys.exit(1)

    builder = FeatureBuilder(include_hurst=include_hurst)

    def _build():
        return builder.build(df)

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
) -> None:
    """Entrena un MetaLabeler LightGBM sobre señales del ensemble y persiste.

    Modos:
      - single-symbol (P4)        → `symbol` + no split
      - basket (P4.1)             → `symbols` + no split
      - basket + regime-split (P4.2) → `symbols` + regime_split=True:
          entrena DOS modelos (trend-follow / mean-revert) en un único payload.
    """
    from data.fetcher import DataFetcher
    from ml.meta_labeler import MetaLabelerTrainer, MetaLabelerConfig, save_meta_labeler

    logger.info("═" * 50)
    tag = []
    if symbols:
        tag.append("BASKET x%d" % len(symbols))
    if regime_split:
        tag.append("REGIME-SPLIT")
    label = " ".join(tag) if tag else "SINGLE"
    logger.info("  MODO TRAIN [%s] — %s %s %s  TP=%.2f SL=%.2f Max=%d",
                label, symbol or "basket", interval, period, tp_mult, sl_mult, max_bars)
    logger.info("═" * 50)

    if regime_split and not symbols:
        logger.error("--regime-split requiere --symbols (entrenamiento sobre basket).")
        sys.exit(1)

    cfg = MetaLabelerConfig(tp_mult=tp_mult, sl_mult=sl_mult, max_bars=max_bars)
    trainer = MetaLabelerTrainer(config=cfg)

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
        result = trainer.train(df)

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
            print(f"\n-- [{name}] Metricas por fold (walk-forward) --")
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

        print("\n-- Metricas por fold (walk-forward) --")
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
def run_live() -> None:
    """Ejecuta el bot en modo live con la estrategia RSI."""
    from data.fetcher import DataFetcher
    from data.storage import StorageManager
    from strategies.rsi_strategy import RSIStrategy
    from execution.broker import Broker
    from execution.orders import OrderManager
    from risk.manager import RiskManager
    from utils.helpers import send_telegram_message, build_trade_alert

    logger.info("═" * 50)
    logger.info("  MODO LIVE TRADING")
    logger.info("═" * 50)

    # Inicializar componentes
    broker = Broker()
    storage = StorageManager()
    risk_mgr = RiskManager()
    order_mgr = OrderManager(broker, storage, risk_mgr)
    strategy = RSIStrategy()
    fetcher = DataFetcher(alpaca_api=broker.api)

    send_telegram_message(f"🤖 <b>Trading Bot iniciado</b>\n"
                          f"Símbolo: {TRADING_SYMBOL}\nTimeframe: {TIMEFRAME}")

    logger.info("Bot iniciado — Símbolo: %s  Timeframe: %s", TRADING_SYMBOL, TIMEFRAME)

    try:
        while True:
            # 1. Obtener datos recientes
            df = fetcher.fetch_alpaca(symbol=TRADING_SYMBOL, timeframe=TIMEFRAME)

            # 2. Generar señal
            signal = strategy.generate_signal(df)
            logger.info("Señal: %s", signal)

            # 3. Actuar según la señal
            if signal.action.value == "BUY":
                trade = order_mgr.open_position(
                    symbol=TRADING_SYMBOL,
                    side="buy",
                    strategy_name=strategy.name,
                )
                if trade:
                    alert = build_trade_alert("BUY", TRADING_SYMBOL, signal.price, strategy.name)
                    send_telegram_message(alert)

            elif signal.action.value == "SELL":
                open_trades = storage.get_open_trades(symbol=TRADING_SYMBOL)
                for trade in open_trades:
                    order_mgr.close_position(trade, signal.price)
                    alert = build_trade_alert("SELL", TRADING_SYMBOL, signal.price, strategy.name)
                    send_telegram_message(alert)

            # 4. Revisar SL/TP de posiciones abiertas
            current_price = fetcher.get_current_price(TRADING_SYMBOL)
            for trade in storage.get_open_trades(TRADING_SYMBOL):
                order_mgr.check_exit_conditions(trade, current_price)

            # 5. Esperar hasta la siguiente vela
            sleep_map = {
                "1Min": 60, "5Min": 300, "15Min": 900,
                "1Hour": 3600, "4Hour": 14400, "1Day": 86400,
            }
            wait = sleep_map.get(TIMEFRAME, 3600)
            logger.info("Esperando %d segundos hasta la siguiente vela...", wait)
            time.sleep(wait)

    except KeyboardInterrupt:
        logger.info("Bot detenido por el usuario")
        send_telegram_message("🛑 <b>Trading Bot detenido</b>")
    finally:
        storage.close()


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
        choices=["live", "backtest", "features", "train"],
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
        )
    elif args.mode == "live":
        run_live()


if __name__ == "__main__":
    main()
