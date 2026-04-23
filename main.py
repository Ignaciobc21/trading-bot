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
from typing import Optional

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
def run_backtest(
    symbol: str,
    period: str,
    interval: str,
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
) -> None:
    """Descarga datos y ejecuta un backtest con la estrategia MFI+RSI."""
    from data.fetcher import DataFetcher
    from strategies.mfi_rsi_strategy import MfiRsiStrategy
    from backtesting.engine import BacktestEngine

    logger.info("═" * 50)
    logger.info("  MODO BACKTEST")
    logger.info("═" * 50)

    # 1. Descargar datos
    df = DataFetcher.fetch_yahoo(ticker=symbol, period=period, interval=interval)

    if df.empty:
        logger.error("No se pudieron obtener datos para %s", symbol)
        sys.exit(1)

    logger.info("Datos descargados: %d filas [%s → %s]", len(df), df.index[0], df.index[-1])

    # 2. Configurar estrategia
    strategy = MfiRsiStrategy(
        rsi_period=14,
        rsi_oversold=30,
        rsi_overbought=70,
        mfi_period=14,
        mfi_oversold=20,
        mfi_overbought=80,
        use_trend_filter=use_trend_filter,
        trend_ema_period=trend_ema_period,
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
    # Lookback suficiente para que la EMA de tendencia esté estable.
    lookback = max(50, trend_ema_period if use_trend_filter else 0) + 5
    result = engine.run(df, lookback=lookback)

    # 4. Mostrar resultados
    print(result.summary())

    # 5. Persistencia opcional
    if save_trades:
        engine.save_trades_csv(result, save_trades)

    # 6. Graficar
    engine.plot_results(result, save_path=save_plot)


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
        choices=["live", "backtest"],
        default="backtest",
        help="Modo de ejecucion (default: backtest)",
    )
    parser.add_argument("--symbol", default="AAPL", help="Simbolo (default: AAPL)")
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
        )
    elif args.mode == "live":
        run_live()


if __name__ == "__main__":
    main()
