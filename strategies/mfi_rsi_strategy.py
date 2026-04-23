from __future__ import annotations
from typing import Optional
import pandas as pd
import numpy as np

from strategies.base import BaseStrategy, StrategySignal, Action

# --- 2. Your Custom Strategy ---

class MfiRsiStrategy(BaseStrategy):
    """
    Estrategia combinada de RSI y MFI.
    Genera señal de COMPRA cuando ambos indicadores están en zona "alcista" 
    (por encima de sobrevendido y por debajo de sobrecomprado).
    Genera señal de VENTA cuando alguno sale de la zona.
    """

    def __init__(
        self,
        rsi_period: int = 14,
        rsi_oversold: float = 30.0,
        rsi_overbought: float = 70.0,
        mfi_period: int = 14,
        mfi_oversold: float = 20.0,
        mfi_overbought: float = 80.0
    ):
        self.name = f"RSI({rsi_period}) + MFI({mfi_period})"
        
        # Parámetros RSI
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        
        # Parámetros MFI
        self.mfi_period = mfi_period
        self.mfi_oversold = mfi_oversold
        self.mfi_overbought = mfi_overbought

    def _compute_rsi(self, series: pd.Series) -> pd.Series:
        """Cálculo interno de RSI."""
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1/self.rsi_period, min_periods=self.rsi_period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/self.rsi_period, min_periods=self.rsi_period, adjust=False).mean()
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def _compute_mfi(self, df: pd.DataFrame) -> pd.Series:
        """Cálculo interno de MFI."""
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        raw_money_flow = typical_price * df['volume']
        
        positive_flow = pd.Series(0.0, index=df.index)
        negative_flow = pd.Series(0.0, index=df.index)
        
        tp_diff = typical_price.diff()
        positive_flow[tp_diff > 0] = raw_money_flow[tp_diff > 0]
        negative_flow[tp_diff < 0] = raw_money_flow[tp_diff < 0]
        
        positive_mf = positive_flow.rolling(window=self.mfi_period).sum()
        negative_mf = negative_flow.rolling(window=self.mfi_period).sum()
        
        return 100 - (100 / (1 + positive_mf / negative_mf.replace(0, np.nan)))

    def generate_signal(self, df: pd.DataFrame) -> StrategySignal:
        """Evalúa los datos más recientes y devuelve una decisión."""
        
        # Validar que hay suficientes datos
        max_period = max(self.rsi_period, self.mfi_period)
        if len(df) < max_period:
            return StrategySignal(Action.HOLD, reason="Not enough data")

        # 1. Calcular indicadores (optimizado: solo evaluamos lo necesario si es posible, 
        # pero aquí calculamos la serie para aplicar tus funciones exactas)
        rsi_series = self._compute_rsi(df['close'])
        mfi_series = self._compute_mfi(df)

        # 2. Extraer el valor de la vela más reciente
        current_rsi = rsi_series.iloc[-1]
        current_mfi = mfi_series.iloc[-1]
        current_price = df['close'].iloc[-1]

        # 3. Lógica de la estrategia (basada en tu código)
        # RSI es alcista si está entre oversold y overbought
        rsi_bullish = (current_rsi > self.rsi_oversold) and (current_rsi <= self.rsi_overbought)
        
        # MFI es alcista si está entre oversold y overbought
        mfi_bullish = (current_mfi > self.mfi_oversold) and (current_mfi <= self.mfi_overbought)

        # 4. Generar la señal
        if rsi_bullish and mfi_bullish:
            return StrategySignal(
                action=Action.BUY,
                confidence=0.9,
                price=current_price,
                reason=f"RSI ({current_rsi:.1f}) y MFI ({current_mfi:.1f}) en zona alcista"
            )
        else:
            return StrategySignal(
                action=Action.SELL,
                confidence=0.9,
                price=current_price,
                reason=f"Condición rota. RSI: {current_rsi:.1f}, MFI: {current_mfi:.1f}"
            )