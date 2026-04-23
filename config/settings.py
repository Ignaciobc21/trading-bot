"""
settings.py — Configuración global del trading bot.

Carga variables de entorno desde secrets.env y expone constantes
de configuración que utilizan todos los módulos del proyecto.

"""

import os
from pathlib import Path
from dotenv import load_dotenv

# ──────────────────────────────────────────────
# Rutas del proyecto
# ──────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = BASE_DIR / "config" / "secrets.env"

load_dotenv(dotenv_path=ENV_PATH)

# ──────────────────────────────────────────────
# Alpaca API
# ──────────────────────────────────────────────
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY", "")
ALPACA_API_SECRET = os.getenv("ALPACA_API_SECRET", "")
ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

# ──────────────────────────────────────────────
# Trading
# ──────────────────────────────────────────────
TRADING_SYMBOL = os.getenv("TRADING_SYMBOL", "AAPL")
TIMEFRAME = os.getenv("TIMEFRAME", "1Hour")

# ──────────────────────────────────────────────
# Telegram (notificaciones)
# ──────────────────────────────────────────────
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# ──────────────────────────────────────────────
# Gestión de riesgo
# ──────────────────────────────────────────────
MAX_POSITION_SIZE_PCT = float(os.getenv("MAX_POSITION_SIZE_PCT", "2.0"))   # % del capital
STOP_LOSS_PCT = float(os.getenv("STOP_LOSS_PCT", "1.5"))                  # % de stop-loss
TAKE_PROFIT_PCT = float(os.getenv("TAKE_PROFIT_PCT", "3.0"))              # % de take-profit
MAX_DAILY_LOSS_PCT = float(os.getenv("MAX_DAILY_LOSS_PCT", "5.0"))        # % pérdida máxima diaria
MAX_OPEN_TRADES = int(os.getenv("MAX_OPEN_TRADES", "3"))

# ──────────────────────────────────────────────
# Base de datos
# ──────────────────────────────────────────────
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{BASE_DIR / 'data' / 'trading.db'}")

# ──────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = BASE_DIR / "logs" / "trading_bot.log"

# ──────────────────────────────────────────────
# Backtesting
# ──────────────────────────────────────────────
INITIAL_CAPITAL = float(os.getenv("INITIAL_CAPITAL", "10000.0"))
COMMISSION_PCT = float(os.getenv("COMMISSION_PCT", "0.0"))  # Alpaca es commission-free
