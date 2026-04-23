"""
broker.py — Conexión con Alpaca.

Encapsula la lógica de conexión con Alpaca Markets y expone métodos
de alto nivel para consultar balances, colocar y cancelar órdenes.
"""

from __future__ import annotations

from typing import Optional, Dict, Any, List

import alpaca_trade_api as tradeapi

from config.settings import (
    ALPACA_API_KEY,
    ALPACA_API_SECRET,
    ALPACA_BASE_URL,
)
from utils.logger import get_logger

logger = get_logger(__name__)


class Broker:
    """
    Capa de abstracción sobre la API de Alpaca.

    Uso:
        broker = Broker()
        balance = broker.get_balance()
        order = broker.place_market_order("AAPL", "buy", 10)
    """

    def __init__(
        self,
        api_key: str = ALPACA_API_KEY,
        api_secret: str = ALPACA_API_SECRET,
        base_url: str = ALPACA_BASE_URL,
    ):
        self.api = tradeapi.REST(
            key_id=api_key,
            secret_key=api_secret,
            base_url=base_url,
            api_version="v2",
        )

        # Verificar conexión
        try:
            account = self.api.get_account()
            logger.info(
                "Broker conectado a Alpaca — cuenta %s (status: %s)",
                account.id,
                account.status,
            )
        except Exception as e:
            logger.warning("No se pudo verificar la conexión a Alpaca: %s", e)

    # ──────────────────────────────────────────
    # Cuenta / Balance
    # ──────────────────────────────────────────
    def get_balance(self, currency: str = "USD") -> Dict[str, float]:
        """
        Devuelve el balance de la cuenta.

        Returns:
            {"free": cash, "used": ..., "total": portfolio_value}
        """
        account = self.api.get_account()
        info = {
            "free": float(account.cash),
            "used": float(account.portfolio_value) - float(account.cash),
            "total": float(account.portfolio_value),
        }
        logger.info(
            "Balance → cash=%.2f  portfolio=%.2f",
            info["free"],
            info["total"],
        )
        return info

    # ──────────────────────────────────────────
    # Órdenes de mercado
    # ──────────────────────────────────────────
    def place_market_order(
        self, symbol: str, side: str, qty: float
    ) -> Dict[str, Any]:
        """
        Coloca una orden de mercado.

        Args:
            symbol: Símbolo (e.g. "AAPL", "TSLA")
            side: "buy" | "sell"
            qty: Cantidad de acciones
        """
        logger.info("Orden MARKET %s %s qty=%.2f", side.upper(), symbol, qty)
        order = self.api.submit_order(
            symbol=symbol,
            qty=qty,
            side=side,
            type="market",
            time_in_force="gtc",
        )
        logger.info(
            "Orden ejecutada — id=%s  status=%s", order.id, order.status
        )
        return {
            "id": order.id,
            "status": order.status,
            "symbol": order.symbol,
            "side": order.side,
            "qty": order.qty,
            "filled_avg_price": order.filled_avg_price,
        }

    # ──────────────────────────────────────────
    # Órdenes limit
    # ──────────────────────────────────────────
    def place_limit_order(
        self, symbol: str, side: str, qty: float, price: float
    ) -> Dict[str, Any]:
        """
        Coloca una orden límite.

        Args:
            symbol: Símbolo
            side: "buy" | "sell"
            qty: Cantidad
            price: Precio límite
        """
        logger.info(
            "Orden LIMIT %s %s qty=%.2f @ %.4f", side.upper(), symbol, qty, price
        )
        order = self.api.submit_order(
            symbol=symbol,
            qty=qty,
            side=side,
            type="limit",
            time_in_force="gtc",
            limit_price=str(price),
        )
        logger.info("Orden enviada — id=%s", order.id)
        return {
            "id": order.id,
            "status": order.status,
            "symbol": order.symbol,
        }

    # ──────────────────────────────────────────
    # Cancelar / consultar
    # ──────────────────────────────────────────
    def cancel_order(self, order_id: str, symbol: str = "") -> None:
        """Cancela una orden abierta."""
        logger.info("Cancelando orden %s", order_id)
        self.api.cancel_order(order_id)

    def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict]:
        """Devuelve las órdenes abiertas."""
        orders = self.api.list_orders(status="open")
        if symbol:
            orders = [o for o in orders if o.symbol == symbol]
        return [{"id": o.id, "symbol": o.symbol, "side": o.side, "qty": o.qty} for o in orders]

    def get_order_status(self, order_id: str, symbol: str = "") -> Dict[str, Any]:
        """Consulta el estado de una orden."""
        order = self.api.get_order(order_id)
        return {
            "id": order.id,
            "status": order.status,
            "filled_avg_price": order.filled_avg_price,
        }

    def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """Obtiene el último precio del símbolo."""
        snapshot = self.api.get_snapshot(symbol)
        return {
            "last": float(snapshot.latest_trade.p),
            "bid": float(snapshot.latest_quote.bp),
            "ask": float(snapshot.latest_quote.ap),
        }

    # ──────────────────────────────────────────
    # Posiciones
    # ──────────────────────────────────────────
    def get_positions(self) -> List[Dict]:
        """Devuelve las posiciones abiertas."""
        positions = self.api.list_positions()
        return [
            {
                "symbol": p.symbol,
                "qty": float(p.qty),
                "avg_entry_price": float(p.avg_entry_price),
                "current_price": float(p.current_price),
                "unrealized_pl": float(p.unrealized_pl),
            }
            for p in positions
        ]

    def close_position(self, symbol: str) -> None:
        """Cierra toda la posición de un símbolo."""
        logger.info("Cerrando posición completa de %s", symbol)
        self.api.close_position(symbol)
