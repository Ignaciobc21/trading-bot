"""
donchian_trend.py — Estrategia de trend-following tipo Donchian / Turtle.

Entrada:  ruptura del máximo de los últimos `entry_lookback` bares.
Salida:   ruptura del mínimo de los últimos `exit_lookback` bares, o stop
          de volatilidad tipo "chandelier" (max(high) - ATR·mult).

Diseñada para operar sólo en régimen TREND_UP (la decisión de cuándo se
permite operar la toma la estrategia Ensemble).
"""

from __future__ import annotations

import pandas as pd

from strategies.base import BaseStrategy, StrategySignal, Action
from strategies.regime import atr


class DonchianTrendStrategy(BaseStrategy):
    """Donchian breakout + chandelier exit."""

    def __init__(
        self,
        entry_lookback: int = 20,
        exit_lookback: int = 10,
        atr_period: int = 14,
        chandelier_mult: float = 3.0,
    ):
        self.entry_lookback = entry_lookback
        self.exit_lookback = exit_lookback
        self.atr_period = atr_period
        self.chandelier_mult = chandelier_mult
        self.name = f"Donchian({entry_lookback}/{exit_lookback}) + ATR·{chandelier_mult}"

    # ──────────────────────────────────────────
    # Señales vectorizadas
    # ──────────────────────────────────────────
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        # Canal previo (sin mirar la barra actual para evitar look-ahead en el nivel).
        upper = df["high"].rolling(window=self.entry_lookback, min_periods=self.entry_lookback).max().shift(1)
        lower = df["low"].rolling(window=self.exit_lookback, min_periods=self.exit_lookback).min().shift(1)

        atr_series = atr(df, period=self.atr_period)
        chandelier_high = df["high"].rolling(
            window=self.entry_lookback, min_periods=self.entry_lookback
        ).max()
        chandelier_stop = chandelier_high - self.chandelier_mult * atr_series

        buy = df["close"] > upper
        exit_chan = df["close"] < lower
        exit_atr = df["close"] < chandelier_stop.shift(1)
        sell = exit_chan | exit_atr

        actions = pd.Series(Action.HOLD, index=df.index, name="action", dtype=object)
        actions[sell.fillna(False)] = Action.SELL
        actions[buy.fillna(False) & ~sell.fillna(False)] = Action.BUY
        return actions

    # ──────────────────────────────────────────
    # Señal puntual (modo live)
    # ──────────────────────────────────────────
    def generate_signal(self, df: pd.DataFrame) -> StrategySignal:
        if len(df) < max(self.entry_lookback, self.atr_period) + 2:
            return StrategySignal(Action.HOLD, reason="Datos insuficientes")
        action = self.generate_signals(df).iloc[-1]
        price = float(df["close"].iloc[-1])
        if action == Action.BUY:
            return StrategySignal(Action.BUY, confidence=0.7, price=price,
                                  reason=f"Cierre > máx {self.entry_lookback} barras")
        if action == Action.SELL:
            return StrategySignal(Action.SELL, confidence=0.7, price=price,
                                  reason=f"Cierre < mín {self.exit_lookback} barras o ATR stop")
        return StrategySignal(Action.HOLD, confidence=0.0, price=price, reason="Sin ruptura")
