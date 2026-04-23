"""
builder.py — Construye un DataFrame tabular de features para ML.

Familias de features:
    - price     : retornos log a distintos horizontes y aceleración.
    - momentum  : RSI, MACD, Stochastic, ROC.
    - trend     : distancias/z-scores a EMAs 20/50/200.
    - volatility: ATR normalizada, retorno rolling std, BB width, BB %b.
    - volume    : OBV slope, CMF, MFI, z-score de volumen, VWAP deviation.
    - regime    : ADX, pendiente de EMA(50), exponente de Hurst.
    - calendar  : día de la semana, mes, hora (intraday) como one-hot y seno/coseno.

Uso básico:
    >>> builder = FeatureBuilder()
    >>> features = builder.build(df_ohlcv)
    >>> features.dropna().shape
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd

from strategies.regime import adx, atr, rolling_hurst

# Versión del conjunto de features. Incrementar si cambia cualquier fórmula
# o se añaden columnas (invalida caches en disco).
FEATURE_VERSION = "1.0.0"


# ──────────────────────────────────────────────
# Helpers numéricos
# ──────────────────────────────────────────────
def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    return a / b.replace(0, np.nan)


def _zscore(series: pd.Series, window: int) -> pd.Series:
    roll = series.rolling(window=window, min_periods=window)
    return (series - roll.mean()) / roll.std(ddof=0).replace(0, np.nan)


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.where(avg_loss != 0, 100.0)


def _mfi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    typical = (df["high"] + df["low"] + df["close"]) / 3.0
    raw_mf = typical * df["volume"]
    direction = np.sign(typical.diff()).fillna(0.0)
    positive_mf = raw_mf.where(direction > 0, 0.0)
    negative_mf = raw_mf.where(direction < 0, 0.0)
    pos = positive_mf.rolling(window=period, min_periods=period).sum()
    neg = negative_mf.rolling(window=period, min_periods=period).sum()
    mfr = pos / neg.replace(0, np.nan)
    mfi = 100.0 - (100.0 / (1.0 + mfr))
    mfi = mfi.where(neg != 0, 100.0)
    mfi = mfi.where(pos != 0, 0.0)
    return mfi


def _stoch_k(df: pd.DataFrame, period: int = 14) -> pd.Series:
    low_n = df["low"].rolling(window=period, min_periods=period).min()
    high_n = df["high"].rolling(window=period, min_periods=period).max()
    return 100.0 * (df["close"] - low_n) / (high_n - low_n).replace(0, np.nan)


def _macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    hist = macd - signal_line
    return macd, signal_line, hist


def _obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff()).fillna(0.0)
    return (direction * volume).cumsum()


def _cmf(df: pd.DataFrame, period: int = 20) -> pd.Series:
    hl = (df["high"] - df["low"]).replace(0, np.nan)
    mfm = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / hl
    mfv = mfm * df["volume"]
    return (
        mfv.rolling(window=period, min_periods=period).sum()
        / df["volume"].rolling(window=period, min_periods=period).sum().replace(0, np.nan)
    )


def _vwap_deviation(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """VWAP rolling — en 1d el VWAP diario no aplica, así que usamos una
    ventana móvil de tamaño `window`."""
    typical = (df["high"] + df["low"] + df["close"]) / 3.0
    num = (typical * df["volume"]).rolling(window=window, min_periods=window).sum()
    den = df["volume"].rolling(window=window, min_periods=window).sum().replace(0, np.nan)
    vwap = num / den
    return (df["close"] - vwap) / vwap


def _bollinger(close: pd.Series, period: int = 20, n_std: float = 2.0):
    mid = close.rolling(window=period, min_periods=period).mean()
    std = close.rolling(window=period, min_periods=period).std(ddof=0)
    upper = mid + n_std * std
    lower = mid - n_std * std
    width = (upper - lower) / mid.replace(0, np.nan)
    pct_b = (close - lower) / (upper - lower).replace(0, np.nan)
    return width, pct_b


# ──────────────────────────────────────────────
# Builder
# ──────────────────────────────────────────────
@dataclass
class FeatureBuilder:
    """Construye un DataFrame wide de features por barra."""

    # Ventanas de features ajustables.
    return_horizons: List[int] = field(default_factory=lambda: [1, 3, 5, 10, 20])
    ema_periods: List[int] = field(default_factory=lambda: [20, 50, 200])
    volatility_window: int = 20
    vol_zscore_window: int = 20
    bb_period: int = 20
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    stoch_period: int = 14
    mfi_period: int = 14
    cmf_period: int = 20
    vwap_window: int = 20
    adx_period: int = 14
    hurst_window: int = 100
    include_calendar: bool = True
    include_hurst: bool = True  # caro en series largas; opcional desactivar.

    # ──────────────────────────────────────────
    # API pública
    # ──────────────────────────────────────────
    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Construye el DataFrame de features a partir de OHLCV. Conserva el
        índice de `df`. NO incluye precio/volumen crudos como features
        (esos se mantienen en el DataFrame original; el caller puede hacer
        join si los quiere).
        """
        self._validate(df)
        out = {}

        out.update(self._price_features(df))
        out.update(self._momentum_features(df))
        out.update(self._trend_features(df))
        out.update(self._volatility_features(df))
        out.update(self._volume_features(df))
        out.update(self._regime_features(df))
        if self.include_calendar:
            out.update(self._calendar_features(df))

        features = pd.DataFrame(out, index=df.index)
        # Nada de datos de futuro: todas las features se calculan hasta
        # (inclusive) la barra actual. Si queremos "features al cierre de
        # t-1 usadas para predecir t", el shift(1) lo hace el consumidor
        # (builder de labels) para desacoplar horizonte.
        return features

    # ──────────────────────────────────────────
    # Familias de features
    # ──────────────────────────────────────────
    def _price_features(self, df: pd.DataFrame) -> dict:
        close = df["close"]
        log_ret = np.log(close / close.shift(1))
        feats = {"log_ret_1": log_ret}
        for h in self.return_horizons:
            if h == 1:
                continue
            feats[f"log_ret_{h}"] = np.log(close / close.shift(h))
        # Aceleración = diferencia de retornos (no-stationary-ish pero informativo).
        feats["log_ret_1_diff"] = log_ret.diff()
        # Gap = open relativo al cierre previo.
        feats["gap_pct"] = (df["open"] - close.shift(1)) / close.shift(1).replace(0, np.nan)
        # Rango intra-bar normalizado.
        feats["range_pct"] = (df["high"] - df["low"]) / df["close"].replace(0, np.nan)
        # Posición del cierre dentro del rango intra-bar.
        rng = (df["high"] - df["low"]).replace(0, np.nan)
        feats["close_in_range"] = (df["close"] - df["low"]) / rng
        return feats

    def _momentum_features(self, df: pd.DataFrame) -> dict:
        close = df["close"]
        rsi = _rsi(close, period=self.rsi_period)
        macd, signal, hist = _macd(close, self.macd_fast, self.macd_slow, self.macd_signal)
        stoch_k = _stoch_k(df, period=self.stoch_period)
        stoch_d = stoch_k.rolling(window=3, min_periods=3).mean()
        roc_10 = close.pct_change(10)
        return {
            f"rsi_{self.rsi_period}": rsi,
            "rsi_centered": rsi - 50.0,
            "macd": macd,
            "macd_signal": signal,
            "macd_hist": hist,
            "macd_hist_norm": hist / close.replace(0, np.nan),
            "stoch_k": stoch_k,
            "stoch_d": stoch_d,
            "stoch_k_minus_d": stoch_k - stoch_d,
            "roc_10": roc_10,
        }

    def _trend_features(self, df: pd.DataFrame) -> dict:
        close = df["close"]
        feats: dict = {}
        prev_ema: Optional[pd.Series] = None
        for n in self.ema_periods:
            ema = close.ewm(span=n, adjust=False).mean()
            feats[f"ema_{n}_dev"] = (close - ema) / ema.replace(0, np.nan)
            feats[f"ema_{n}_slope"] = ema.diff() / ema.replace(0, np.nan)
            feats[f"z_price_vs_ema_{n}"] = _zscore(close - ema, window=n)
            if prev_ema is not None:
                feats[f"ema_stack_{n}"] = (ema - prev_ema) / prev_ema.replace(0, np.nan)
            prev_ema = ema
        return feats

    def _volatility_features(self, df: pd.DataFrame) -> dict:
        close = df["close"]
        log_ret = np.log(close / close.shift(1))
        rolling_vol = log_ret.rolling(window=self.volatility_window, min_periods=self.volatility_window).std(ddof=0)
        atr_series = atr(df, period=14)
        atr_norm = atr_series / close.replace(0, np.nan)
        bb_width, bb_pctb = _bollinger(close, period=self.bb_period, n_std=2.0)
        return {
            "realized_vol_20": rolling_vol,
            "atr_14_norm": atr_norm,
            "bb_width_20": bb_width,
            "bb_pctb_20": bb_pctb,
            # Vol-of-vol: cambio reciente del entorno de volatilidad.
            "vol_of_vol_20": rolling_vol.diff().rolling(window=5, min_periods=5).std(ddof=0),
        }

    def _volume_features(self, df: pd.DataFrame) -> dict:
        close = df["close"]
        volume = df["volume"].astype(float)
        obv = _obv(close, volume)
        obv_slope = obv.diff().rolling(window=10, min_periods=10).mean() / volume.rolling(window=10, min_periods=10).mean().replace(0, np.nan)
        vol_z = _zscore(volume, window=self.vol_zscore_window)
        cmf = _cmf(df, period=self.cmf_period)
        mfi = _mfi(df, period=self.mfi_period)
        vwap_dev = _vwap_deviation(df, window=self.vwap_window)
        return {
            "obv_slope_10": obv_slope,
            "vol_zscore_20": vol_z,
            f"cmf_{self.cmf_period}": cmf,
            f"mfi_{self.mfi_period}": mfi,
            "mfi_centered": mfi - 50.0,
            "vwap_dev_20": vwap_dev,
        }

    def _regime_features(self, df: pd.DataFrame) -> dict:
        adx_series = adx(df, period=self.adx_period)
        ema50 = df["close"].ewm(span=50, adjust=False).mean()
        feats = {
            f"adx_{self.adx_period}": adx_series,
            "ema_50_slope_norm": ema50.diff() / ema50.replace(0, np.nan),
        }
        if self.include_hurst:
            feats["hurst_100"] = rolling_hurst(df["close"], window=self.hurst_window)
        return feats

    def _calendar_features(self, df: pd.DataFrame) -> dict:
        idx = df.index
        if not isinstance(idx, pd.DatetimeIndex):
            return {}
        # Seno/coseno para periodicidad semanal y anual (encoding continuo,
        # preferido sobre one-hot para árboles boosted).
        dow = idx.dayofweek.astype(float)
        month = idx.month.astype(float)
        feats = {
            "dow_sin": np.sin(2 * np.pi * dow / 7.0),
            "dow_cos": np.cos(2 * np.pi * dow / 7.0),
            "month_sin": np.sin(2 * np.pi * (month - 1) / 12.0),
            "month_cos": np.cos(2 * np.pi * (month - 1) / 12.0),
        }
        # Si hay componente intradiario, añadimos también hora.
        if (idx[1] - idx[0]) < pd.Timedelta(days=1) if len(idx) > 1 else False:
            hour = idx.hour.astype(float) + idx.minute.astype(float) / 60.0
            feats["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
            feats["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
        # Convertir numpy arrays a Series alineadas con el índice.
        return {k: pd.Series(v, index=idx, name=k) for k, v in feats.items()}

    # ──────────────────────────────────────────
    # Validación
    # ──────────────────────────────────────────
    @staticmethod
    def _validate(df: pd.DataFrame) -> None:
        required = {"open", "high", "low", "close", "volume"}
        missing = required - set(df.columns.str.lower())
        if missing:
            raise ValueError(f"FeatureBuilder: faltan columnas {missing} en el DataFrame de entrada")
