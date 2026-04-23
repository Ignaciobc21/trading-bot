"""
base.py — Clase base abstracta para estrategias de trading.

Todas las estrategias concretas heredan de `BaseStrategy` e implementan
el método `generate_signal()`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import pandas as pd


class Action(Enum):
    """Señal de acción del mercado."""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass
class StrategySignal:
    """Resultado devuelto por una estrategia."""
    action: Action
    confidence: float = 0.0       # 0.0 – 1.0
    price: Optional[float] = None
    reason: str = ""

    def __str__(self) -> str:
        return (
            f"[{self.action.value}] confianza={self.confidence:.0%} "
            f"precio={self.price} — {self.reason}"
        )


class BaseStrategy(ABC):
    """
    Interfaz que deben implementar todas las estrategias.

    Uso:
        class MiEstrategia(BaseStrategy):
            name = "Mi Estrategia"
            def generate_signal(self, df):
                ...
    """

    name: str = "BaseStrategy"

    @abstractmethod
    def generate_signal(self, df: pd.DataFrame) -> StrategySignal:
        """
        Analiza el DataFrame OHLCV y devuelve una señal.

        Args:
            df: DataFrame con columnas 'open','high','low','close','volume'

        Returns:
            StrategySignal con la acción recomendada.
        """
        ...

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Genera una serie de acciones (una por barra) para backtests vectorizados.

        Implementación por defecto: itera barra a barra llamando a
        `generate_signal`. Las estrategias pueden sobreescribir este método
        para devolver la serie completa en una sola pasada vectorizada
        (mucho más rápido en datasets grandes).

        Returns:
            pd.Series indexada como df, con valores Action (BUY/SELL/HOLD).
        """
        actions = []
        for i in range(len(df)):
            window = df.iloc[: i + 1]
            actions.append(self.generate_signal(window).action)
        return pd.Series(actions, index=df.index, name="action")

    def __repr__(self) -> str:
        return f"<Strategy: {self.name}>"
