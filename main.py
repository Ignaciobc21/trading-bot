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

from config.settings import TRADING_SYMBOL, TIMEFRAME, INITIAL_CAPITAL
from utils.logger import get_logger

logger = get_logger("main")


# ═══════════════════════════════════════════════
#  MODO BACKTEST
# ═══════════════════════════════════════════════
def run_backtest(symbol: str, period: str, interval: str) -> None:
    """Descarga datos y ejecuta un backtest con la estrategia RSI."""
    from data.fetcher import DataFetcher
    from strategies.rsi_strategy import RSIStrategy
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
    # strategy = RSIStrategy(rsi_period=14, oversold=30, overbought=70)
    strategy = MfiRsiStrategy(rsi_period=14, rsi_oversold=30, rsi_overbought=70, mfi_period=14, mfi_oversold=20, mfi_overbought=80)

    # 3. Ejecutar backtest
    engine = BacktestEngine(
        strategy=strategy,
        initial_capital=INITIAL_CAPITAL,
    )
    result = engine.run(df, lookback=50)

    # 4. Mostrar resultados
    print(result.summary())

    # 5. Graficar
    engine.plot_results(result)


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

    args = parser.parse_args()

    if args.mode == "backtest":
        # Auto-seleccionar periodo si no se indico
        period = args.period
        if period is None:
            period = INTERVAL_MAX_PERIOD.get(args.interval, "1y")
            logger.info("Periodo auto-seleccionado: %s (max para %s)", period, args.interval)

        run_backtest(symbol=args.symbol, period=period, interval=args.interval)
    elif args.mode == "live":
        run_live()


if __name__ == "__main__":
    main()
