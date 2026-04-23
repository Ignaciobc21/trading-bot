"""
logger.py — Configuración centralizada de logging.

Crea un logger con salida a consola (coloreada) y a archivo
rotativo para todos los módulos del proyecto.
"""

from __future__ import annotations

import io
import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler

from config.settings import LOG_LEVEL, LOG_FILE


# ──────────────────────────────────────────────
# Formato con colores para la consola
# ──────────────────────────────────────────────
class ColorFormatter(logging.Formatter):
    """Formatter que añade colores ANSI según el nivel de log."""

    COLORS = {
        logging.DEBUG: "\033[36m",      # cyan
        logging.INFO: "\033[32m",       # verde
        logging.WARNING: "\033[33m",    # amarillo
        logging.ERROR: "\033[31m",      # rojo
        logging.CRITICAL: "\033[1;31m", # rojo bold
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelno, self.RESET)
        record.levelname = f"{color}{record.levelname:<8}{self.RESET}"
        return super().format(record)


# ──────────────────────────────────────────────
# Fábrica de loggers
# ──────────────────────────────────────────────
_initialized = False


def _setup_root_logger() -> None:
    """Configura el root logger una sola vez."""
    global _initialized
    if _initialized:
        return

    root = logging.getLogger()
    root.setLevel(getattr(logging, LOG_LEVEL.upper(), logging.INFO))

    # ── Handler de consola (utf-8 safe para Windows) ──
    utf8_stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    console = logging.StreamHandler(utf8_stdout)
    console.setFormatter(
        ColorFormatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    )
    root.addHandler(console)

    # ── Handler de archivo ──
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    file_handler = RotatingFileHandler(
        LOG_FILE,
        maxBytes=5 * 1024 * 1024,  # 5 MB
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
        )
    )
    root.addHandler(file_handler)

    _initialized = True


def get_logger(name: str) -> logging.Logger:
    """
    Devuelve un logger configurado para el módulo indicado.

    Uso:
        from utils.logger import get_logger
        logger = get_logger(__name__)
        logger.info("Hola mundo")
    """
    _setup_root_logger()
    return logging.getLogger(name)
