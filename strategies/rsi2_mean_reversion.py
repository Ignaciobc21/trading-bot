"""
rsi2_mean_reversion.py — Estrategia de mean-reversion tipo Connors.

Entrada:  RSI(2) < `oversold` con precio por encima de la SMA200 (tendencia
          general alcista). La SMA200 es un filtro clásico de Connors para
          evitar mean-revert en bear markets.
Salida:   precio cierra por encima de la SMA(5) (corta duración: 1-4 barras
          promedio).

Pensada para operar en régimen MEAN_REVERT del Ensemble. No tiene sentido
usarla en solitario en 1 único símbolo — funciona mejor sobre baskets.
"""

from __future__ import annotations

import pandas as pd

from strategies.base import BaseStrategy, StrategySignal, Action


class RSI2MeanReversionStrategy(BaseStrategy):
    """Connors RSI(2) mean-reversion con filtro SMA200."""

    def __init__(
        self,
        rsi_period: int = 2,
        oversold: float = 10.0,
        exit_sma: int = 5,
        trend_sma: int = 200,
    ):
        self.rsi_period = rsi_period
        self.oversold = oversold
        self.exit_sma = exit_sma
        self.trend_sma = trend_sma
        self.name = f"Connors RSI({rsi_period}) MR + SMA{trend_sma}"

    # ──────────────────────────────────────────
    # Indicador
    # ──────────────────────────────────────────
    def _compute_rsi(self, close: pd.Series) -> pd.Series:
        delta = close.diff()
        gain = delta.clip(lower=0.0)
        loss = -delta.clip(upper=0.0)
        avg_gain = gain.ewm(alpha=1 / self.rsi_period, min_periods=self.rsi_period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / self.rsi_period, min_periods=self.rsi_period, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, float("nan"))
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return rsi.where(avg_loss != 0, 100.0)

    # ──────────────────────────────────────────
    # Señales vectorizadas
    # ──────────────────────────────────────────
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        close = df["close"]
        rsi2 = self._compute_rsi(close)
        sma_exit = close.rolling(window=self.exit_sma, min_periods=self.exit_sma).mean()
        sma_trend = close.rolling(window=self.trend_sma, min_periods=self.trend_sma).mean()

        long_regime = close > sma_trend
        buy = (rsi2 < self.oversold) & long_regime
        sell = close > sma_exit

        actions = pd.Series(Action.HOLD, index=df.index, name="action", dtype=object)
        actions[sell.fillna(False)] = Action.SELL
        actions[buy.fillna(False) & ~sell.fillna(False)] = Action.BUY
        return actions

    # ──────────────────────────────────────────
    # Señal puntual
    # ──────────────────────────────────────────
    def generate_signal(self, df: pd.DataFrame) -> StrategySignal:
        if len(df) < self.trend_sma + 2:
            return StrategySignal(Action.HOLD, reason="Datos insuficientes")
        action = self.generate_signals(df).iloc[-1]
        price = float(df["close"].iloc[-1])
        if action == Action.BUY:
            return StrategySignal(Action.BUY, confidence=0.7, price=price,
                                  reason=f"RSI({self.rsi_period})<{self.oversold} en tendencia alcista")
        if action == Action.SELL:
            return StrategySignal(Action.SELL, confidence=0.7, price=price,
                                  reason=f"Cierre > SMA{self.exit_sma}")
        return StrategySignal(Action.HOLD, confidence=0.0, price=price, reason="Sin señal")
