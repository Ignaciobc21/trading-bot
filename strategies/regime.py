"""
regime.py — Detector de régimen de mercado.

Clasifica cada barra en uno de cuatro regímenes a partir de ADX, Hurst y
volatilidad relativa. Otras estrategias (ensemble) usan esta clasificación
para decidir si deben operar y con qué lógica (trend-following vs. mean
reversion).

Regímenes:
    TREND_UP      — ADX alto + pendiente de EMA positiva + Hurst persistente
    TREND_DOWN    — ADX alto + pendiente de EMA negativa + Hurst persistente
    MEAN_REVERT   — ADX bajo, Hurst anti-persistente
    CHOP          — resto (no operar)

El detector es vectorizado: devuelve una pd.Series alineada con el índice
del DataFrame.
"""

from __future__ import annotations

from enum import Enum

import numpy as np
import pandas as pd


class Regime(str, Enum):
    TREND_UP = "TREND_UP"
    TREND_DOWN = "TREND_DOWN"
    MEAN_REVERT = "MEAN_REVERT"
    CHOP = "CHOP"


# ──────────────────────────────────────────────
# Indicadores auxiliares
# ──────────────────────────────────────────────
def _true_range(df: pd.DataFrame) -> pd.Series:
    high = df["high"]
    low = df["low"]
    prev_close = df["close"].shift(1)
    return pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range (Wilder)."""
    tr = _true_range(df)
    return tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average Directional Index (Wilder), vectorizado."""
    up = df["high"].diff()
    down = -df["low"].diff()

    plus_dm = ((up > down) & (up > 0)).astype(float) * up.clip(lower=0.0)
    minus_dm = ((down > up) & (down > 0)).astype(float) * down.clip(lower=0.0)

    tr = _true_range(df)
    atr_ = tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    plus_di = 100.0 * plus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr_.replace(0, np.nan)
    minus_di = 100.0 * minus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr_.replace(0, np.nan)
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1 / period, adjust=False).mean()


def hurst_exponent(series: pd.Series, max_lag: int = 20) -> float:
    """
    Hurst exponent por R/S simplificado (Mandelbrot): ajuste log-log de
    desviaciones estándar de diferencias a distintos lags.

    H ≈ 0.5 → random walk
    H > 0.55 → serie persistente (trending)
    H < 0.45 → serie anti-persistente (mean-reverting)
    """
    series = np.asarray(series, dtype=float)
    if len(series) < max_lag * 2:
        return 0.5
    lags = range(2, max_lag + 1)
    tau = []
    for lag in lags:
        diff = series[lag:] - series[:-lag]
        std = np.std(diff, ddof=0)
        if std <= 0 or not np.isfinite(std):
            return 0.5
        tau.append(std)
    # Regresión log-lag vs log-tau.
    log_lags = np.log(list(lags))
    log_tau = np.log(tau)
    slope = np.polyfit(log_lags, log_tau, 1)[0]
    return float(slope)


def rolling_hurst(close: pd.Series, window: int = 100, max_lag: int = 20) -> pd.Series:
    """Hurst aplicado sobre ventanas móviles del log-precio."""
    log_price = np.log(close.replace(0, np.nan)).ffill()
    return log_price.rolling(window=window, min_periods=window).apply(
        lambda x: hurst_exponent(x, max_lag=max_lag), raw=True
    )


# ──────────────────────────────────────────────
# Detector
# ──────────────────────────────────────────────
class RegimeDetector:
    """
    Clasifica cada barra del DataFrame en un Regime.

    Args:
        adx_period:      periodo del ADX.
        adx_trending:    umbral de ADX que se considera 'trending'.
        adx_ranging:     umbral de ADX que se considera 'ranging'.
        ema_period:      EMA para la pendiente (determina signo del trend).
        hurst_window:    ventana para el cálculo rolling del exponente de Hurst.
        hurst_trend:     H > este valor se considera persistente.
        hurst_mr:        H < este valor se considera anti-persistente.
        vol_window:      ventana para la volatilidad relativa.
        vol_blowup:      si vol_rel > este valor, fuerza CHOP (evitar operar en blow-ups).
    """

    def __init__(
        self,
        adx_period: int = 14,
        adx_trending: float = 20.0,
        adx_ranging: float = 25.0,
        ema_period: int = 50,
        hurst_window: int = 100,
        hurst_trend: float = 0.52,
        hurst_mr: float = 0.48,
        vol_window: int = 63,
        vol_blowup: float = 3.0,
    ):
        self.adx_period = adx_period
        self.adx_trending = adx_trending
        self.adx_ranging = adx_ranging
        self.ema_period = ema_period
        self.hurst_window = hurst_window
        self.hurst_trend = hurst_trend
        self.hurst_mr = hurst_mr
        self.vol_window = vol_window
        self.vol_blowup = vol_blowup

    # ──────────────────────────────────────────
    # Cálculo principal
    # ──────────────────────────────────────────
    def classify(self, df: pd.DataFrame) -> pd.Series:
        """
        Devuelve una Series con el Regime por barra (aligned con df.index).
        Las barras con datos insuficientes devuelven CHOP como default seguro.
        """
        adx_series = adx(df, period=self.adx_period)
        ema = df["close"].ewm(span=self.ema_period, adjust=False).mean()
        ema_slope = ema.diff()
        hurst = rolling_hurst(df["close"], window=self.hurst_window)

        # Volatilidad relativa vs. la mediana reciente.
        ret = df["close"].pct_change()
        vol = ret.rolling(window=20, min_periods=20).std()
        vol_median = vol.rolling(window=self.vol_window, min_periods=self.vol_window).median()
        vol_rel = vol / vol_median.replace(0, np.nan)

        trending = adx_series > self.adx_trending
        ranging = adx_series < self.adx_ranging

        up_bias = ema_slope > 0
        down_bias = ema_slope < 0

        hurst_persist = hurst > self.hurst_trend
        hurst_anti = hurst < self.hurst_mr

        # Por defecto CHOP.
        regime = pd.Series(Regime.CHOP, index=df.index, dtype=object)

        trend_up_mask = trending & up_bias & (hurst_persist | hurst.isna())
        trend_down_mask = trending & down_bias & (hurst_persist | hurst.isna())
        mr_mask = ranging & (hurst_anti | hurst.isna())

        regime[trend_up_mask.fillna(False)] = Regime.TREND_UP
        regime[trend_down_mask.fillna(False)] = Regime.TREND_DOWN
        regime[mr_mask.fillna(False)] = Regime.MEAN_REVERT

        # Blow-up de volatilidad → forzar CHOP.
        blowup = (vol_rel > self.vol_blowup).fillna(False)
        regime[blowup] = Regime.CHOP
        return regime

    def diagnostics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Devuelve un DataFrame con todos los indicadores intermedios para debugging."""
        return pd.DataFrame(
            {
                "adx": adx(df, period=self.adx_period),
                "ema_slope": df["close"].ewm(span=self.ema_period, adjust=False).mean().diff(),
                "hurst": rolling_hurst(df["close"], window=self.hurst_window),
                "regime": self.classify(df),
            }
        )
