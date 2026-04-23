"""
fetcher.py — Descarga de datos de mercado.

Proporciona funciones para obtener datos históricos (OHLCV) desde
Alpaca o Yahoo Finance vía yfinance.
"""

from __future__ import annotations

import pandas as pd
from datetime import datetime, timedelta
from typing import Optional

import yfinance as yf

from config.settings import TRADING_SYMBOL, TIMEFRAME
from utils.logger import get_logger

logger = get_logger(__name__)


class DataFetcher:
    """Descarga datos OHLCV desde Alpaca o Yahoo Finance."""

    def __init__(self, alpaca_api=None):
        """
        Args:
            alpaca_api: Instancia de alpaca_trade_api.REST (opcional).
                        Si se pasa, se puede usar fetch_alpaca().
        """
        self._api = alpaca_api

    # ──────────────────────────────────────────
    # Datos desde Alpaca
    # ──────────────────────────────────────────
    def fetch_alpaca(
        self,
        symbol: str = TRADING_SYMBOL,
        timeframe: str = TIMEFRAME,
        limit: int = 500,
    ) -> pd.DataFrame:
        """
        Descarga barras OHLCV desde Alpaca.

        Returns:
            DataFrame con columnas: open, high, low, close, volume
        """
        if self._api is None:
            raise RuntimeError("Alpaca API no inicializada. Pasa el api al constructor.")

        from alpaca_trade_api.rest import TimeFrame

        tf_map = {
            "1Min": TimeFrame.Minute,
            "5Min": TimeFrame(5, "Min"),
            "15Min": TimeFrame(15, "Min"),
            "1Hour": TimeFrame.Hour,
            "1Day": TimeFrame.Day,
        }
        tf = tf_map.get(timeframe, TimeFrame.Hour)

        end = datetime.utcnow()
        start = end - timedelta(days=limit)

        logger.info("Descargando datos Alpaca: %s (%s) ...", symbol, timeframe)
        bars = self._api.get_bars(symbol, tf, start=start.isoformat(), end=end.isoformat()).df

        if bars.empty:
            logger.warning("No se obtuvieron datos de Alpaca para %s", symbol)
            return bars

        # Normalizar nombres de columna
        bars.columns = [c.lower() for c in bars.columns]
        logger.info("Descargadas %d barras de Alpaca", len(bars))
        return bars

    # ──────────────────────────────────────────
    # Datos desde Yahoo Finance
    # ──────────────────────────────────────────
    @staticmethod
    def fetch_yahoo(
        ticker: str,
        period: str = "1y",
        interval: str = "1d",
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Descarga datos históricos desde Yahoo Finance.

        Args:
            ticker: Símbolo (e.g. "AAPL", "BTC-USD")
            period: Período relativo ("1d","5d","1mo","3mo","6mo","1y","2y","5y","10y","ytd","max")
            interval: Intervalo ("1m","2m","5m","15m","30m","60m","90m","1h","1d","5d","1wk","1mo","3mo")
            start: Fecha inicio "YYYY-MM-DD" (override period)
            end: Fecha fin "YYYY-MM-DD"
        """
        logger.info("Descargando datos Yahoo Finance: %s (%s)", ticker, period)

        data = yf.download(
            ticker,
            period=period,
            interval=interval,
            start=start,
            end=end,
            progress=False,
        )

        if data.empty:
            logger.warning("No se obtuvieron datos para %s", ticker)
            return data

        # yfinance puede devolver MultiIndex columns — aplanar
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [col[0].lower() for col in data.columns]
        else:
            data.columns = [c.lower() for c in data.columns]
        logger.info("Descargadas %d filas de %s", len(data), ticker)
        return data

    # ──────────────────────────────────────────
    # Precio actual (requiere Alpaca)
    # ──────────────────────────────────────────
    def get_current_price(self, symbol: str = TRADING_SYMBOL) -> float:
        """Obtiene el último precio del símbolo via Alpaca."""
        if self._api is None:
            raise RuntimeError("Alpaca API no inicializada.")

        snapshot = self._api.get_snapshot(symbol)
        price = float(snapshot.latest_trade.p)
        logger.debug("Precio actual de %s: %.4f", symbol, price)
        return price
