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
        """
        Devuelve el factor multiplicativo para pasar de desviación típica
        por-barra a anualizada. Ejemplos:
          - Diario: sqrt(252) ≈ 15.87
          - Horario intradía (6.5h sesión): sqrt(6.5 * 252) ≈ 40.4

        Si el usuario fija `annualize_factor` en el config, lo respetamos.
        En caso contrario, inferimos la frecuencia por la mediana del
        espaciamiento entre bares. Usa 252 días de trading/año como base.
        """
        if self.config.annualize_factor is not None:
            return float(self.config.annualize_factor)
        if isinstance(index, pd.DatetimeIndex) and len(index) > 1:
            deltas = index.to_series().diff().dropna().dt.total_seconds()
            med = float(deltas.median()) if len(deltas) else 86400.0
            # Si el paso mediano es < 23h lo tratamos como intradía.
            if med < 23 * 3600:
                bars_per_day = (6.5 * 3600) / med
                return float(np.sqrt(bars_per_day * 252.0))
        return float(np.sqrt(252.0))

    # ──────────────────────────────────────────
    def vol_multiplier(self, df: pd.DataFrame) -> pd.Series:
        """
        Componente de **vol targeting**. Para cada barra calcula:

            mult_i = target_vol / realized_vol_annualized_i

        * `realized_vol_annualized` = std rolling de log-returns × sqrt(factor).
        * Si la volatilidad realizada es menor que `vol_floor`, la clippeamos
          para no generar multiplicadores absurdos (p.ej. 500x) en periodos
          muy planos — ese régimen casi nunca dura, y no queremos over-leverage.
        * Si supera `vol_cap` la consideramos "blow-up" y devolvemos mult=0
          (preferimos perder la entrada a operar en condiciones extremas).
        * Limitamos el output final a [vol_mult_floor, vol_mult_cap]. Cap>1
          permite leverage intencional en activos de vol baja.

        Nota: hay una pequeña dependencia de look-ahead (rolling().std usa
        n barras incluyendo la actual). Para uso live, el motor ya aplica
        la señal del bar t-1 al open del bar t, así que no hay problema.
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
        Componente de **regime sizing**. Lookup directo en la tabla
        `regime_multipliers`:

            TREND_UP    → 1.00  (full sizing con viento a favor)
            MEAN_REVERT → 0.85  (trade táctico, ligeramente reducido)
            CHOP        → 0.60  (mercado ruidoso, exposición moderada)
            TREND_DOWN  → 0.00  (no abrir longs en tendencia bajista)

        `regime_series` se espera como una Serie de enums `Regime` o sus
        strings equivalentes (enum → str hash en el dict funciona igual).
        Valores no mapeados → 0 (conservador).
        """
        cfg = self.config
        return regime_series.map(cfg.regime_multipliers).fillna(0.0).rename("regime_mult")

    # ──────────────────────────────────────────
    def confidence_multiplier(self, proba: pd.Series) -> pd.Series:
        """
        Componente de **confidence sizing**. Mapea la probabilidad del
        meta-modelo a un multiplicador lineal:

            proba ≤ confidence_floor   → confidence_min_mult
            proba ≥ confidence_ceiling → 1.0
            proba en medio             → interpolación lineal.

        Defaults: floor=0.30, ceiling=0.70, min_mult=0.40. Es decir: los
        BUYs con confianza baja (~0.30) se ejecutan al 40% del tamaño
        base y los de confianza alta (~0.70) al 100%. En la práctica,
        el meta-modelo suele producir probas entre 0.30 y 0.65.

        Si `use_confidence_sizing=False`, devuelve una Serie de 1.0
        (desactiva la componente sin tocar las otras dos).
        """
        cfg = self.config
        if not cfg.use_confidence_sizing:
            return pd.Series(1.0, index=proba.index, name="conf_mult")
        span = max(cfg.confidence_ceiling - cfg.confidence_floor, 1e-6)
        x = (proba - cfg.confidence_floor) / span
        x = x.clip(lower=0.0, upper=1.0)
        # Lineal entre confidence_min_mult (en x=0) y 1.0 (en x=1).
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
        Producto de los tres componentes anteriores, clippeado a
        [overall_floor, overall_cap].

        El motor sólo lee esta serie en el bar anterior a una entrada BUY.
        En bares de HOLD/SELL el valor es irrelevante — lo mantenemos
        fillna(overall_floor) por simplicidad; si el motor por error lee
        un bar HOLD, el multiplicador será 0 y no se abrirá nada.

        Args:
            df     : OHLCV (necesario para vol_multiplier).
            regime : Serie con el régimen por barra. Si None, se omite
                     el componente de regime sizing.
            proba  : Serie con la proba del meta-modelo (NaN en bares
                     sin BUY). Si None, se omite confidence sizing.
        """
        cfg = self.config
        idx = df.index
        mult = self.vol_multiplier(df).reindex(idx)
        if regime is not None:
            mult = mult * self.regime_multiplier(regime).reindex(idx).fillna(0.0)
        if proba is not None:
            # Bares sin proba (NaN) → usamos confidence_min_mult para no
            # multiplicar por 0 y "matar" señales válidas fuera del meta-labeler.
            conf = self.confidence_multiplier(proba).reindex(idx).fillna(cfg.confidence_min_mult)
            mult = mult * conf
        mult = mult.clip(lower=cfg.overall_floor, upper=cfg.overall_cap)
        return mult.fillna(cfg.overall_floor).rename("size_mult")
