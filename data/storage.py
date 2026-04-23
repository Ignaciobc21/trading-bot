"""
storage.py — Persistencia de datos con SQLAlchemy.

Gestiona el almacenamiento de operaciones (trades), señales y
datos OHLCV en una base de datos SQLite (u otra via DATABASE_URL).
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional, List

import pandas as pd
from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    Float,
    String,
    DateTime,
    Boolean,
)
from sqlalchemy.orm import declarative_base, sessionmaker

from config.settings import DATABASE_URL
from utils.logger import get_logger

logger = get_logger(__name__)

# ──────────────────────────────────────────────
# SQLAlchemy setup
# ──────────────────────────────────────────────
engine = create_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()


# ──────────────────────────────────────────────
# Modelos
# ──────────────────────────────────────────────
class Trade(Base):
    """Registro de una operación ejecutada."""

    __tablename__ = "trades"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String, nullable=False)
    side = Column(String, nullable=False)          # "buy" | "sell"
    entry_price = Column(Float, nullable=False)
    exit_price = Column(Float, nullable=True)
    quantity = Column(Float, nullable=False)
    pnl = Column(Float, nullable=True)             # ganancias / pérdidas
    status = Column(String, default="open")         # "open" | "closed"
    strategy = Column(String, nullable=True)
    opened_at = Column(DateTime, default=datetime.utcnow)
    closed_at = Column(DateTime, nullable=True)

    def __repr__(self) -> str:
        return (
            f"<Trade {self.id} {self.side} {self.symbol} "
            f"@ {self.entry_price} — {self.status}>"
        )


class Signal(Base):
    """Señal generada por una estrategia."""

    __tablename__ = "signals"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String, nullable=False)
    action = Column(String, nullable=False)         # "BUY" | "SELL" | "HOLD"
    strategy = Column(String, nullable=True)
    price_at_signal = Column(Float, nullable=True)
    confidence = Column(Float, nullable=True)       # 0.0 – 1.0
    created_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self) -> str:
        return f"<Signal {self.action} {self.symbol} @ {self.price_at_signal}>"


# ──────────────────────────────────────────────
# Controller
# ──────────────────────────────────────────────
class StorageManager:
    """Interfaz de alto nivel para operaciones de lectura/escritura."""

    def __init__(self):
        Base.metadata.create_all(engine)
        self._session = SessionLocal()
        logger.info("StorageManager inicializado — DB: %s", DATABASE_URL)

    # ── Trades ──
    def save_trade(self, trade: Trade) -> Trade:
        self._session.add(trade)
        self._session.commit()
        logger.info("Trade guardado: %s", trade)
        return trade

    def close_trade(
        self, trade_id: int, exit_price: float, pnl: float
    ) -> Optional[Trade]:
        trade = self._session.query(Trade).get(trade_id)
        if trade is None:
            logger.warning("Trade %d no encontrado", trade_id)
            return None

        trade.exit_price = exit_price
        trade.pnl = pnl
        trade.status = "closed"
        trade.closed_at = datetime.utcnow()
        self._session.commit()
        logger.info("Trade %d cerrado con PnL: %.2f", trade_id, pnl)
        return trade

    def get_open_trades(self, symbol: Optional[str] = None) -> List[Trade]:
        query = self._session.query(Trade).filter(Trade.status == "open")
        if symbol:
            query = query.filter(Trade.symbol == symbol)
        return query.all()

    def get_trade_history(self, limit: int = 50) -> List[Trade]:
        return (
            self._session.query(Trade)
            .order_by(Trade.opened_at.desc())
            .limit(limit)
            .all()
        )

    # ── Signals ──
    def save_signal(self, signal: Signal) -> Signal:
        self._session.add(signal)
        self._session.commit()
        logger.debug("Signal guardada: %s", signal)
        return signal

    # ── OHLCV ──
    def save_ohlcv(self, df: pd.DataFrame, table_name: str = "ohlcv") -> None:
        """Guarda un DataFrame OHLCV en la base de datos."""
        df.to_sql(table_name, engine, if_exists="append", index=True)
        logger.info("Guardadas %d filas en tabla '%s'", len(df), table_name)

    def load_ohlcv(self, table_name: str = "ohlcv") -> pd.DataFrame:
        """Carga datos OHLCV desde la base de datos."""
        return pd.read_sql_table(table_name, engine, parse_dates=["timestamp"])

    # ── Cleanup ──
    def close(self) -> None:
        self._session.close()
        logger.info("StorageManager cerrado")
