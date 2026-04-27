"""
helpers.py — Funciones de utilidad generales.

Contiene helpers reutilizables: notificaciones Telegram,
formateadores de moneda, cálculos técnicos auxiliares, etc.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

import requests

from config.settings import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
from utils.logger import get_logger

logger = get_logger(__name__)


# ──────────────────────────────────────────────
# Telegram
# ──────────────────────────────────────────────
def send_telegram_message(
    message: str,
    token: str = TELEGRAM_BOT_TOKEN,
    chat_id: str = TELEGRAM_CHAT_ID,
    parse_mode: str = "HTML",
) -> bool:
    """
    Envía un mensaje de texto al chat de Telegram configurado.

    Returns:
        True si el mensaje se envió correctamente.
    """
    # Detectamos token/chat_id ausentes o placeholders obvios del .env de ejemplo.
    # Sin esto, el código trataría "tu_telegram_token_aqui" como válido e intentaría
    # hacer POST contra la URL del bot, generando 404s que ensucian los logs.
    placeholders = ("tu_", "your_", "aqui", "here", "xxx", "todo", "placeholder")
    def _is_placeholder(val: str) -> bool:
        low = (val or "").strip().lower()
        return not low or any(p in low for p in placeholders)

    if _is_placeholder(token) or _is_placeholder(chat_id):
        logger.debug("Telegram no configurado — mensaje silencioso")
        return False

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message, "parse_mode": parse_mode}

    try:
        r = requests.post(url, json=payload, timeout=10)
        r.raise_for_status()
        logger.info("📨 Mensaje Telegram enviado")
        return True
    except requests.RequestException as e:
        logger.error("Error enviando mensaje Telegram: %s", e)
        return False


# ──────────────────────────────────────────────
# Formateo
# ──────────────────────────────────────────────
def format_currency(value: float, symbol: str = "$", decimals: int = 2) -> str:
    """Formatea un valor numérico como moneda."""
    return f"{symbol}{value:,.{decimals}f}"


def format_pct(value: float, decimals: int = 2) -> str:
    """Formatea un valor como porcentaje con signo."""
    return f"{value:+.{decimals}f}%"


def format_timestamp(dt: Optional[datetime] = None) -> str:
    """Devuelve un timestamp legible."""
    dt = dt or datetime.utcnow()
    return dt.strftime("%Y-%m-%d %H:%M:%S UTC")


# ──────────────────────────────────────────────
# Cálculos auxiliares
# ──────────────────────────────────────────────
def calculate_pnl(entry_price: float, exit_price: float, quantity: float, side: str = "buy") -> float:
    """
    Calcula el PnL de una operación.

    Args:
        entry_price: Precio de entrada.
        exit_price: Precio de salida.
        quantity: Cantidad.
        side: "buy" para longs, "sell" para shorts.
    """
    if side == "buy":
        return (exit_price - entry_price) * quantity
    else:
        return (entry_price - exit_price) * quantity


def calculate_pnl_pct(entry_price: float, exit_price: float, side: str = "buy") -> float:
    """Calcula el PnL en porcentaje."""
    if entry_price == 0:
        return 0.0
    if side == "buy":
        return ((exit_price - entry_price) / entry_price) * 100
    else:
        return ((entry_price - exit_price) / entry_price) * 100


# ──────────────────────────────────────────────
# Alertas formateadas
# ──────────────────────────────────────────────
def build_trade_alert(
    action: str,
    symbol: str,
    price: float,
    strategy: str = "",
    pnl: Optional[float] = None,
) -> str:
    """
    Construye un mensaje de alerta por HTML para Telegram.

    Args:
        action: "BUY", "SELL", "CLOSE", etc.
        symbol: Par de trading.
        price: Precio de la operación.
        strategy: Nombre de la estrategia.
        pnl: PnL (solo para cierres).
    """
    emoji = {"BUY": "🟢", "SELL": "🔴", "CLOSE": "🔒"}.get(action.upper(), "📊")

    lines = [
        f"{emoji} <b>{action.upper()}</b> — {symbol}",
        f"💰 Precio: {format_currency(price)}",
    ]
    if strategy:
        lines.append(f"📈 Estrategia: {strategy}")
    if pnl is not None:
        lines.append(f"📊 PnL: {format_currency(pnl)} ({format_pct(pnl)})")
    lines.append(f"🕐 {format_timestamp()}")

    return "\n".join(lines)
