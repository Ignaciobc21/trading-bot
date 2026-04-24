"""
overlay.py — Risk overlay para el backtest / live path.

Calcula un multiplicador de tamaño de posición por barra, combinando:
  1. Vol targeting: multiplicador ∝ target_vol / realized_vol_annualized.
     Un trade en un activo muy volátil queda más pequeño, y al revés.
  2. Regime-conditioned sizing: más pequeño en CHOP / TREND_DOWN,
     full en TREND_UP / MEAN_REVERT.
  3. Confidence-scaled sizing (opcional): si el meta-labeler expone
     `predict_proba`, escalamos linealmente entre un floor (prob mínima
     aceptada) y 1.0. Trade de prob ~floor → sizing mínimo, prob alta
     → sizing máximo.

El multiplicador final es el producto (limitado arriba por `cap`) y se
pasa al BacktestEngine, que lo usa como factor sobre `position_size_pct`
en cada entrada.

La lógica es puramente vectorizada — no añade dependencia de tiempo
variable en el motor, sólo multiplica una Serie más en la entrada.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

from strategies.regime import Regime


# Escalado por régimen: cuánto del tamaño base se usa en cada clasificación.
# Valores por defecto pensados para el ensemble trend/MR:
#   - TREND_UP    : long con tendencia → full size.
#   - MEAN_REVERT : trade rápido contra-tendencia → ligera reducción.
#   - CHOP        : mercado sin dirección → penalización fuerte.
#   - TREND_DOWN  : régimen bajista → no abrimos longs.
DEFAULT_REGIME_MULTIPLIERS: Dict[Regime, float] = {
    Regime.TREND_UP: 1.00,
    Regime.MEAN_REVERT: 0.85,
    Regime.CHOP: 0.60,
    Regime.TREND_DOWN: 0.00,
}


@dataclass
class RiskConfig:
    # ── Vol targeting ──
    target_vol: float = 0.20          # Volatilidad anualizada objetivo (ej 0.20 = 20%).
    vol_window: int = 20              # Barras para realized vol rolling.
    vol_floor: float = 0.05           # Evita división por 0 / sobre-leverage en low-vol.
    vol_cap: float = 0.80             # Vol máxima aceptable; por encima sizing→0.
    vol_mult_cap: float = 1.50        # Tope del multiplicador de vol.
    vol_mult_floor: float = 0.10      # Piso (no dejes el sizing en ~0 por vol extrema).
    # ── Regime ──
    regime_multipliers: Dict[Regime, float] = None   # type: ignore[assignment]
    # ── Confidence sizing ──
    use_confidence_sizing: bool = True
    confidence_floor: float = 0.30    # Prob ≤ floor → mult = 0.
    confidence_ceiling: float = 0.70  # Prob ≥ ceiling → mult = 1.
    confidence_min_mult: float = 0.40 # Sizing mínimo cuando la prob = floor (no cerrar a 0).
    # ── Cap y floor globales ──
    overall_cap: float = 1.00         # Nunca por encima del 100% del sizing base.
    overall_floor: float = 0.00       # Por convención; un 0 deshabilita la señal.
    # ── Daily loss kill-switch (aplicado en el engine, no aquí) ──
    max_daily_loss_pct: float = 0.0   # 0 = desactivado. Ej 5.0 → para trading al -5% diario.
    # ── Periods_per_year para anualizar vol ──
    annualize_factor: Optional[float] = None

    def __post_init__(self):
        if self.regime_multipliers is None:
            self.regime_multipliers = dict(DEFAULT_REGIME_MULTIPLIERS)


# ──────────────────────────────────────────────
# Overlay
# ──────────────────────────────────────────────
class RiskOverlay:
    def __init__(self, config: Optional[RiskConfig] = None):
        self.config = config or RiskConfig()

    # ──────────────────────────────────────────
    def _annualize(self, index: pd.Index) -> float:
        if self.config.annualize_factor is not None:
            return float(self.config.annualize_factor)
        # Fallback: asume diario (252). El usuario puede forzar otro.
        if isinstance(index, pd.DatetimeIndex) and len(index) > 1:
            deltas = index.to_series().diff().dropna().dt.total_seconds()
            med = float(deltas.median()) if len(deltas) else 86400.0
            if med < 23 * 3600:
                bars_per_day = (6.5 * 3600) / med
                return float(np.sqrt(bars_per_day * 252.0))
        return float(np.sqrt(252.0))

    # ──────────────────────────────────────────
    def vol_multiplier(self, df: pd.DataFrame) -> pd.Series:
        """
        Realized vol rolling (log returns), anualizada, y multiplicador
        target_vol / realized_vol con cap + floor.
        """
        cfg = self.config
        close = df["close"].astype(float)
        log_ret = np.log(close / close.shift(1))
        rv = log_ret.rolling(cfg.vol_window).std()
        rv_annual = rv * self._annualize(df.index)
        rv_annual = rv_annual.clip(lower=cfg.vol_floor)

        mult = cfg.target_vol / rv_annual
        mult = mult.clip(lower=cfg.vol_mult_floor, upper=cfg.vol_mult_cap)
        # Vol extrema → 0 (preferimos no entrar antes que "over-reducir" y entrar).
        mult = mult.where(rv_annual <= cfg.vol_cap, other=0.0)
        return mult.fillna(cfg.vol_mult_floor).rename("vol_mult")

    # ──────────────────────────────────────────
    def regime_multiplier(self, regime_series: pd.Series) -> pd.Series:
        """
        Aplica la tabla `regime_multipliers` fila a fila.
        `regime_series` debe ser una pd.Series de enums Regime.
        """
        cfg = self.config
        # `map` sobre el enum directamente.
        return regime_series.map(cfg.regime_multipliers).fillna(0.0).rename("regime_mult")

    # ──────────────────────────────────────────
    def confidence_multiplier(self, proba: pd.Series) -> pd.Series:
        """
        Linear scale: prob ≤ floor → confidence_min_mult, prob ≥ ceiling → 1.0,
        interpolación lineal en medio. Si use_confidence_sizing=False, devuelve 1.
        """
        cfg = self.config
        if not cfg.use_confidence_sizing:
            return pd.Series(1.0, index=proba.index, name="conf_mult")
        span = max(cfg.confidence_ceiling - cfg.confidence_floor, 1e-6)
        x = (proba - cfg.confidence_floor) / span
        x = x.clip(lower=0.0, upper=1.0)
        m = cfg.confidence_min_mult + (1.0 - cfg.confidence_min_mult) * x
        return m.rename("conf_mult")

    # ──────────────────────────────────────────
    def size_multiplier(
        self,
        df: pd.DataFrame,
        regime: Optional[pd.Series] = None,
        proba: Optional[pd.Series] = None,
    ) -> pd.Series:
        """
        Devuelve un multiplicador de sizing en [overall_floor, overall_cap]
        por barra. Sólo se aplica a entradas BUY — el motor lo lee en la
        entrada y lo multiplica por `position_size_pct`.
        """
        cfg = self.config
        idx = df.index
        mult = self.vol_multiplier(df).reindex(idx)
        if regime is not None:
            mult = mult * self.regime_multiplier(regime).reindex(idx).fillna(0.0)
        if proba is not None:
            conf = self.confidence_multiplier(proba).reindex(idx).fillna(cfg.confidence_min_mult)
            mult = mult * conf
        mult = mult.clip(lower=cfg.overall_floor, upper=cfg.overall_cap)
        return mult.fillna(cfg.overall_floor).rename("size_mult")
