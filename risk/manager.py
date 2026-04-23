"""
manager.py — Gestión de riesgo.

Controla el tamaño de posición, el número máximo de operaciones
abiertas y la pérdida diaria acumulada para proteger el capital.
"""

from __future__ import annotations

from datetime import datetime, date
from typing import List, Optional

from config.settings import (
    MAX_POSITION_SIZE_PCT,
    MAX_DAILY_LOSS_PCT,
    MAX_OPEN_TRADES,
    STOP_LOSS_PCT,
)
from utils.logger import get_logger

logger = get_logger(__name__)


class RiskManager:
    """
    Valida operaciones contra reglas de riesgo antes de ejecutarlas.

    Reglas:
    - Tamaño máximo de posición (% del capital).
    - Número máximo de operaciones abiertas simultáneas.
    - Pérdida diaria máxima acumulada.
    """

    def __init__(
        self,
        max_position_pct: float = MAX_POSITION_SIZE_PCT,
        max_daily_loss_pct: float = MAX_DAILY_LOSS_PCT,
        max_open_trades: int = MAX_OPEN_TRADES,
        stop_loss_pct: float = STOP_LOSS_PCT,
    ):
        self.max_position_pct = max_position_pct
        self.max_daily_loss_pct = max_daily_loss_pct
        self.max_open_trades = max_open_trades
        self.stop_loss_pct = stop_loss_pct

        # Estado interno
        self._open_trades: list = []
        self._daily_pnl: float = 0.0
        self._last_reset: date = date.today()

        logger.info(
            "RiskManager configurado: pos_max=%.1f%%, loss_max=%.1f%%, trades_max=%d",
            max_position_pct,
            max_daily_loss_pct,
            max_open_trades,
        )

    # ──────────────────────────────────────────
    # Reset diario
    # ──────────────────────────────────────────
    def _check_daily_reset(self) -> None:
        """Resetea el PnL acumulado si cambia la fecha."""
        today = date.today()
        if today != self._last_reset:
            logger.info(
                "Nuevo día — reseteando PnL diario (anterior: %.2f)", self._daily_pnl
            )
            self._daily_pnl = 0.0
            self._last_reset = today

    # ──────────────────────────────────────────
    # Validaciones
    # ──────────────────────────────────────────
    def can_open_trade(self) -> bool:
        """Evalúa si se puede abrir una nueva operación."""
        self._check_daily_reset()

        # ¿Demasiadas operaciones abiertas?
        if len(self._open_trades) >= self.max_open_trades:
            logger.warning(
                "Rechazado: %d trades abiertos (max %d)",
                len(self._open_trades),
                self.max_open_trades,
            )
            return False

        # ¿Pérdida diaria excedida?
        if self._daily_pnl < 0 and abs(self._daily_pnl) >= self.max_daily_loss_pct:
            logger.warning(
                "Rechazado: pérdida diaria %.2f%% >= máximo %.2f%%",
                abs(self._daily_pnl),
                self.max_daily_loss_pct,
            )
            return False

        return True

    # ──────────────────────────────────────────
    # Tamaño de posición
    # ──────────────────────────────────────────
    def calculate_position_size(
        self, capital: float, risk_per_trade_pct: Optional[float] = None
    ) -> float:
        """
        Calcula la cantidad máxima a invertir por operación.

        Args:
            capital: Capital disponible (free balance).
            risk_per_trade_pct: Override del % de riesgo por trade.

        Returns:
            Cantidad en la moneda quote (e.g. USDT).
        """
        pct = risk_per_trade_pct or self.max_position_pct
        size = capital * (pct / 100.0)
        logger.debug(
            "Tamaño de posición: %.2f (%.1f%% de %.2f)", size, pct, capital
        )
        return round(size, 4)

    # ──────────────────────────────────────────
    # Registro de trades
    # ──────────────────────────────────────────
    def register_trade(self, trade) -> None:
        """Registra un trade abierto en el contexto del risk manager."""
        self._open_trades.append(trade)
        logger.info(
            "Trade registrado — operaciones abiertas: %d / %d",
            len(self._open_trades),
            self.max_open_trades,
        )

    def unregister_trade(self, trade, pnl_pct: float = 0.0) -> None:
        """
        Desregistra un trade cerrado y acumula el PnL diario.

        Args:
            trade: El trade que se cierra.
            pnl_pct: PnL de la operación en porcentaje.
        """
        self._check_daily_reset()
        if trade in self._open_trades:
            self._open_trades.remove(trade)

        self._daily_pnl += pnl_pct
        logger.info(
            "Trade desregistrado — PnL diario acumulado: %.2f%%", self._daily_pnl
        )

    # ──────────────────────────────────────────
    # Info
    # ──────────────────────────────────────────
    def status(self) -> dict:
        """Devuelve un resumen del estado del risk manager."""
        self._check_daily_reset()
        return {
            "open_trades": len(self._open_trades),
            "max_open_trades": self.max_open_trades,
            "daily_pnl_pct": round(self._daily_pnl, 2),
            "max_daily_loss_pct": self.max_daily_loss_pct,
            "max_position_pct": self.max_position_pct,
        }
