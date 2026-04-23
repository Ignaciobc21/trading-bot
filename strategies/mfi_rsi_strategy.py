"""
mfi_rsi_strategy.py — Estrategia combinada RSI + MFI con filtro de tendencia.

Lógica:
    - BUY cuando el RSI cruza al alza el umbral de sobreventa y el MFI está
      o ha estado recientemente en zona de sobreventa. Opcionalmente, solo
      se permiten longs si el precio está por encima de una EMA larga
      (filtro de tendencia).
    - SELL cuando el RSI cruza a la baja el umbral de sobrecompra o el MFI
      cruza a la baja desde sobrecompra.
    - HOLD en cualquier otro caso.

Los indicadores se calculan vectorialmente una sola vez para evitar el
O(n²) del motor de backtesting antiguo.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from strategies.base import BaseStrategy, StrategySignal, Action


class MfiRsiStrategy(BaseStrategy):
    """
    Estrategia combinada RSI + MFI basada en *cruces*, con filtro de
    tendencia opcional por EMA.
    """

    def __init__(
        self,
        rsi_period: int = 14,
        rsi_oversold: float = 30.0,
        rsi_overbought: float = 70.0,
        mfi_period: int = 14,
        mfi_oversold: float = 20.0,
        mfi_overbought: float = 80.0,
        use_trend_filter: bool = True,
        trend_ema_period: int = 200,
        confirmation_bars: int = 3,
    ):
        self.name = (
            f"RSI({rsi_period}) + MFI({mfi_period})"
            + (f" + EMA{trend_ema_period}" if use_trend_filter else "")
        )

        # Parámetros RSI
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought

        # Parámetros MFI
        self.mfi_period = mfi_period
        self.mfi_oversold = mfi_oversold
        self.mfi_overbought = mfi_overbought

        # Filtro de tendencia
        self.use_trend_filter = use_trend_filter
        self.trend_ema_period = trend_ema_period

        # Ventana de confirmación para cruces cercanos entre RSI y MFI.
        self.confirmation_bars = max(1, confirmation_bars)

    # ──────────────────────────────────────────
    # Indicadores (vectorizados)
    # ──────────────────────────────────────────
    def _compute_rsi(self, close: pd.Series) -> pd.Series:
        """RSI (Wilder) calculado vía EWM."""
        delta = close.diff()
        gain = delta.clip(lower=0.0)
        loss = -delta.clip(upper=0.0)
        avg_gain = gain.ewm(
            alpha=1 / self.rsi_period, min_periods=self.rsi_period, adjust=False
        ).mean()
        avg_loss = loss.ewm(
            alpha=1 / self.rsi_period, min_periods=self.rsi_period, adjust=False
        ).mean()
        rs = avg_gain / avg_loss.replace(0.0, np.nan)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        # Cuando avg_loss == 0 y avg_gain > 0 → RSI=100; cuando ambos 0 → neutral 50.
        rsi = rsi.where(avg_loss != 0, 100.0)
        rsi = rsi.where(~((avg_gain == 0) & (avg_loss == 0)), 50.0)
        return rsi

    def _compute_mfi(self, df: pd.DataFrame) -> pd.Series:
        """Money Flow Index con manejo seguro de divisiones por cero."""
        typical_price = (df["high"] + df["low"] + df["close"]) / 3.0
        raw_money_flow = typical_price * df["volume"]

        tp_diff = typical_price.diff()
        positive_flow = raw_money_flow.where(tp_diff > 0, 0.0)
        negative_flow = raw_money_flow.where(tp_diff < 0, 0.0)

        positive_mf = positive_flow.rolling(window=self.mfi_period).sum()
        negative_mf = negative_flow.rolling(window=self.mfi_period).sum()

        # Si no hubo flujo negativo: MFI = 100. Si no hubo flujo positivo: MFI = 0.
        mfi = 100.0 - (100.0 / (1.0 + positive_mf / negative_mf.replace(0.0, np.nan)))
        mfi = mfi.where(negative_mf != 0, 100.0)
        mfi = mfi.where(positive_mf != 0, 0.0)
        return mfi

    # ──────────────────────────────────────────
    # Señales vectorizadas (usado por el backtest)
    # ──────────────────────────────────────────
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Devuelve una Serie con la acción (BUY/SELL/HOLD) por barra."""
        rsi = self._compute_rsi(df["close"])
        mfi = self._compute_mfi(df)

        rsi_prev = rsi.shift(1)
        mfi_prev = mfi.shift(1)

        # ── Triggers ──
        # BUY primario: RSI cruza al alza el umbral de sobreventa.
        rsi_cross_up = (rsi_prev <= self.rsi_oversold) & (rsi > self.rsi_oversold)
        # Confirmación opcional: el MFI ha visitado sobreventa en una ventana
        # reciente O está por debajo del punto medio (mitad inferior del rango),
        # de forma que no compramos contra un MFI alcista ya extendido.
        mfi_recently_oversold = (mfi < self.mfi_oversold).rolling(
            window=self.confirmation_bars, min_periods=1
        ).max().astype(bool)
        mfi_not_overbought = mfi < self.mfi_overbought
        mfi_filter = mfi_recently_oversold | (mfi_not_overbought & (mfi < 50))

        buy_signal = rsi_cross_up & mfi_filter

        # SELL: cualquiera de los dos cruza a la baja desde sobrecompra.
        rsi_cross_down = (rsi_prev >= self.rsi_overbought) & (rsi < self.rsi_overbought)
        mfi_cross_down = (mfi_prev >= self.mfi_overbought) & (mfi < self.mfi_overbought)
        sell_signal = rsi_cross_down | mfi_cross_down

        # Filtro de tendencia por EMA: consideramos "régimen alcista" cuando
        # se cumple al menos UNA de:
        #   - El precio estuvo por encima de la EMA en las últimas `trend_lookback` barras.
        #   - La pendiente de la EMA es positiva en la barra actual.
        # Durante un pullback que genera RSI oversold el precio suele estar
        # momentáneamente por debajo de la EMA; usar sólo "precio > EMA" haría
        # que el filtro rechace casi todas las entradas válidas.
        if self.use_trend_filter and self.trend_ema_period > 0:
            ema = df["close"].ewm(span=self.trend_ema_period, adjust=False).mean()
            trend_lookback = max(10, self.trend_ema_period // 4)
            recent_above = (df["close"] > ema).rolling(
                window=trend_lookback, min_periods=1
            ).max().astype(bool)
            slope_up = (ema.diff() > 0).fillna(False)
            uptrend = recent_above | slope_up
            buy_signal = buy_signal & uptrend

        # Construcción de la serie final.
        actions = pd.Series(Action.HOLD, index=df.index, name="action", dtype=object)
        actions[sell_signal.fillna(False)] = Action.SELL
        # BUY tiene prioridad sobre SELL solo si no hay exit simultáneo;
        # mantenemos SELL si ambos colisionan para ser conservadores.
        buy_only = buy_signal.fillna(False) & ~sell_signal.fillna(False)
        actions[buy_only] = Action.BUY
        return actions

    # ──────────────────────────────────────────
    # Señal puntual (modo live — una sola barra)
    # ──────────────────────────────────────────
    def generate_signal(self, df: pd.DataFrame) -> StrategySignal:
        max_period = max(self.rsi_period, self.mfi_period, self.trend_ema_period)
        if len(df) < max_period + 2:
            return StrategySignal(Action.HOLD, reason="Datos insuficientes")

        actions = self.generate_signals(df)
        action = actions.iloc[-1]
        current_price = float(df["close"].iloc[-1])

        if action == Action.BUY:
            reason = "Cruce alcista RSI+MFI (trend OK)" if self.use_trend_filter else "Cruce alcista RSI+MFI"
            return StrategySignal(Action.BUY, confidence=0.8, price=current_price, reason=reason)
        if action == Action.SELL:
            return StrategySignal(Action.SELL, confidence=0.8, price=current_price, reason="Cruce bajista RSI/MFI")
        return StrategySignal(Action.HOLD, confidence=0.0, price=current_price, reason="Sin cruce")
