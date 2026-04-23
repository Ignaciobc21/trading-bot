"""
rsi_strategy.py — Estrategia basada en RSI (Relative Strength Index).

Genera señales BUY cuando el RSI baja de un umbral de sobreventa y
señales SELL cuando supera un umbral de sobrecompra.
"""

from __future__ import annotations

import pandas as pd
import ta

from strategies.base import BaseStrategy, StrategySignal, Action
from utils.logger import get_logger

logger = get_logger(__name__)


class RSIStrategy(BaseStrategy):
    """
    Estrategia RSI clásica.

    Parámetros:
        rsi_period : ventana del RSI (default 14)
        oversold   : umbral de sobreventa (default 30)
        overbought : umbral de sobrecompra (default 70)
    """

    name = "RSI Strategy"

    def __init__(
        self,
        rsi_period: int = 14,
        oversold: float = 30.0,
        overbought: float = 70.0,
    ):
        self.rsi_period = rsi_period
        self.oversold = oversold
        self.overbought = overbought

    def generate_signal(self, df: pd.DataFrame) -> StrategySignal:
        """
        Calcula el RSI y devuelve una señal basada en niveles de
        sobrecompra / sobreventa.
        """
        if len(df) < self.rsi_period + 1:
            return StrategySignal(
                action=Action.HOLD,
                reason="Datos insuficientes para calcular RSI",
            )

        # Calcular RSI usando la librería 'ta'
        rsi_series = ta.momentum.RSIIndicator(
            close=df["close"], window=self.rsi_period
        ).rsi()

        current_rsi = rsi_series.iloc[-1]
        prev_rsi = rsi_series.iloc[-2]
        current_price = df["close"].iloc[-1]

        logger.debug(
            "RSI actual=%.2f  anterior=%.2f  precio=%.4f",
            current_rsi,
            prev_rsi,
            current_price,
        )

        # ── Señal de compra ──
        if current_rsi < self.oversold:
            confidence = min(1.0, (self.oversold - current_rsi) / self.oversold)
            return StrategySignal(
                action=Action.BUY,
                confidence=confidence,
                price=current_price,
                reason=f"RSI={current_rsi:.1f} < {self.oversold} (sobreventa)",
            )

        # ── Señal de venta ──
        if current_rsi > self.overbought:
            confidence = min(1.0, (current_rsi - self.overbought) / (100 - self.overbought))
            return StrategySignal(
                action=Action.SELL,
                confidence=confidence,
                price=current_price,
                reason=f"RSI={current_rsi:.1f} > {self.overbought} (sobrecompra)",
            )

        # ── Sin señal clara ──
        return StrategySignal(
            action=Action.HOLD,
            confidence=0.0,
            price=current_price,
            reason=f"RSI={current_rsi:.1f} en zona neutral [{self.oversold}–{self.overbought}]",
        )
