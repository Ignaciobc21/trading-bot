"""
orders.py — Gestión de órdenes.

Coordina la creación de órdenes incorporando lógica de riesgo
(stop-loss, take-profit) y registro en base de datos.
"""

from __future__ import annotations

from typing import Optional, Dict, Any

from execution.broker import Broker
from data.storage import StorageManager, Trade
from risk.manager import RiskManager
from config.settings import TRADING_SYMBOL, STOP_LOSS_PCT, TAKE_PROFIT_PCT
from utils.logger import get_logger

logger = get_logger(__name__)


class OrderManager:
    """
    Orquesta la ejecución de operaciones:
    1. Valida contra el RiskManager.
    2. Ejecuta la orden vía Broker.
    3. Registra el trade en StorageManager.
    """

    def __init__(
        self,
        broker: Broker,
        storage: StorageManager,
        risk_manager: RiskManager,
    ):
        self.broker = broker
        self.storage = storage
        self.risk = risk_manager

    # ──────────────────────────────────────────
    # Abrir posición
    # ──────────────────────────────────────────
    def open_position(
        self,
        symbol: str = TRADING_SYMBOL,
        side: str = "buy",
        amount: Optional[float] = None,
        strategy_name: str = "",
    ) -> Optional[Trade]:
        """
        Abre una nueva posición si pasa las validaciones de riesgo.

        Args:
            symbol: Par de trading
            side: "buy" | "sell"
            amount: Cantidad (si None, se calcula automáticamente)
            strategy_name: Nombre de la estrategia que generó la señal

        Returns:
            Trade registrado o None si fue rechazado por risk management.
        """
        # 1. Validar riesgo
        balance = self.broker.get_balance()
        capital = balance.get("free", 0.0)

        if amount is None:
            amount = self.risk.calculate_position_size(capital)

        if not self.risk.can_open_trade():
            logger.warning("⚠️  RiskManager rechazó la operación")
            return None

        # 2. Ejecutar orden
        try:
            order = self.broker.place_market_order(symbol, side, amount)
        except Exception as e:
            logger.error("Error ejecutando orden: %s", e)
            return None

        # 3. Registrar en DB
        entry_price = order.get("average") or order.get("price", 0.0)
        trade = Trade(
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            quantity=amount,
            strategy=strategy_name,
            status="open",
        )
        self.storage.save_trade(trade)
        self.risk.register_trade(trade)

        logger.info(
            "✅ Posición abierta: %s %s %.6f @ %.4f",
            side.upper(),
            symbol,
            amount,
            entry_price,
        )
        return trade

    # ──────────────────────────────────────────
    # Cerrar posición
    # ──────────────────────────────────────────
    def close_position(
        self,
        trade: Trade,
        current_price: Optional[float] = None,
    ) -> Optional[Trade]:
        """
        Cierra una posición abierta.

        Args:
            trade: Trade a cerrar
            current_price: Precio de cierre (si None, se obtiene del broker)
        """
        if trade.status != "open":
            logger.warning("Trade %d ya está cerrado", trade.id)
            return None

        if current_price is None:
            ticker = self.broker.get_ticker(trade.symbol)
            current_price = ticker["last"]

        # Calcular PnL
        if trade.side == "buy":
            pnl = (current_price - trade.entry_price) * trade.quantity
        else:
            pnl = (trade.entry_price - current_price) * trade.quantity

        # Ejecutar orden de cierre
        close_side = "sell" if trade.side == "buy" else "buy"
        try:
            self.broker.place_market_order(trade.symbol, close_side, trade.quantity)
        except Exception as e:
            logger.error("Error cerrando posición: %s", e)
            return None

        # Actualizar en DB
        self.storage.close_trade(trade.id, current_price, pnl)

        logger.info(
            "🔒 Posición cerrada: Trade #%d  PnL=%.2f",
            trade.id,
            pnl,
        )
        return trade

    # ──────────────────────────────────────────
    # Check stop-loss / take-profit
    # ──────────────────────────────────────────
    def check_exit_conditions(self, trade: Trade, current_price: float) -> bool:
        """
        Evalúa si una posición abierta debe cerrarse por SL o TP.

        Returns:
            True si la posición fue cerrada.
        """
        if trade.side == "buy":
            change_pct = ((current_price - trade.entry_price) / trade.entry_price) * 100
        else:
            change_pct = ((trade.entry_price - current_price) / trade.entry_price) * 100

        # Stop-loss
        if change_pct <= -STOP_LOSS_PCT:
            logger.warning(
                "🛑 STOP-LOSS alcanzado en Trade #%d (%.2f%%)", trade.id, change_pct
            )
            self.close_position(trade, current_price)
            return True

        # Take-profit
        if change_pct >= TAKE_PROFIT_PCT:
            logger.info(
                "🎯 TAKE-PROFIT alcanzado en Trade #%d (%.2f%%)", trade.id, change_pct
            )
            self.close_position(trade, current_price)
            return True

        return False
