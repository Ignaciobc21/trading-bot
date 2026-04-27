"""
ml/drift.py — Fase K: detección de concept drift.

¿Qué es "concept drift"?
========================

Un modelo de ML asume que los datos en producción tienen la **misma
distribución** que los datos con los que se entrenó. En trading, esta
asunción es falsa por diseño: el mercado cambia (régimen, volatilidad,
flujos, correlaciones). El modelo se "degrada" silenciosamente.

Síntomas:
    - La AUC OOS del último mes cae por debajo de la AUC CV histórica.
    - Las features tienen distribuciones diferentes (p. ej. la mediana
      del RSI ha shifteado, la volatilidad realizada ha duplicado, etc.).
    - El win rate en vivo no se parece al del backtest.

Enfoques implementados
======================

Este módulo ofrece tres detectores complementarios. Cada uno captura
un aspecto distinto del drift:

1. **KS-test (Kolmogorov-Smirnov)** — test no paramétrico que compara
   dos distribuciones empíricas. Si `p_value < alpha` (default 0.01),
   rechazamos la hipótesis de que las dos muestras vienen de la misma
   distribución. Cuenta cuántas features están "rotas".

2. **PSI (Population Stability Index)** — métrica estándar en la
   industria financiera para medir cuánto ha shifteado una distribución.
   Se calcula sobre bins fijos y produce un número:
     - PSI < 0.1  : estable.
     - 0.1-0.25  : atención, empieza a haber drift.
     - > 0.25    : drift fuerte, reentrenar.

   Ventaja sobre KS: PSI es menos sensible al tamaño muestral. KS con
   10 000 muestras dispara a la mínima diferencia; PSI sigue siendo
   interpretable.

3. **Rolling AUC / hit-rate** — métrica de performance directa. Se usa
   un ground truth *retrospectivo*: sobre los últimos N trades cerrados,
   ¿el modelo sigue siendo mejor que aleatorio (AUC > 0.52)? Si cae
   durante X barras consecutivas, alerta.

No usamos un solo detector porque los falsos positivos son molestos:
un reentreno innecesario añade coste computacional y puede introducir
sesgo de overfitting al último chunk de datos. Requerir que DOS señales
disparen a la vez reduce falsos positivos a cambio de más latencia.

Uso típico
==========

>>> from ml.drift import DriftDetector, DriftConfig
>>> det = DriftDetector(DriftConfig())
>>> # Cargar la distribución de referencia del training
>>> det.fit_reference(X_train_features)
>>> # En cada iter del live, llamar con las features recientes
>>> report = det.check(X_recent, recent_trade_outcomes=...)
>>> if report.should_retrain:
>>>     trigger_retrain()

Referencias
===========
- Quiñonero-Candela et al., "Dataset Shift in Machine Learning" (2009).
- Jeffrey Yau, "Drift in ML systems" (2021).
- Siddiqi, "Intelligent Credit Scoring" — PSI thresholds estándar.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════
#  Configuración.
# ════════════════════════════════════════════════════════════════════
@dataclass
class DriftConfig:
    """
    Parámetros del detector.

    Se ha elegido cada default tras revisar literatura y casos típicos
    del trading-bot. Todos los thresholds son ajustables por CLI.
    """

    # ── KS-test ──────────────────────────────────────────────────────
    ks_alpha: float = 0.01
    """Nivel de significancia para rechazar H0 de misma distribución.
    Más bajo = más conservador (menos falsos positivos)."""

    ks_feature_frac_threshold: float = 0.30
    """Fracción mínima de features con `p < ks_alpha` para considerar
    drift sobre el KS-test. 30 % es un threshold empírico equilibrado —
    en datasets muy amplios (>100 features) bajarlo a 0.2 es razonable."""

    # ── PSI ──────────────────────────────────────────────────────────
    psi_n_bins: int = 10
    """Nº de bins para discretizar features continuas. 10 es el default
    estándar (Siddiqi 2006)."""

    psi_warning: float = 0.10
    """PSI > 0.10 → warning (cambio moderado). Sólo logueamos."""

    psi_strong: float = 0.25
    """PSI > 0.25 → drift fuerte. Dispara reentreno si además se
    supera `psi_feature_frac_threshold`."""

    psi_feature_frac_threshold: float = 0.20
    """Fracción mínima de features con `PSI > psi_strong` para
    considerar drift fuerte general. 20 % es suficiente — mejor
    detectar varios features degradados que uno solo outlier."""

    # ── AUC rolling ──────────────────────────────────────────────────
    auc_window_trades: int = 30
    """Nº mínimo de trades cerrados recientes para calcular AUC rolling.
    Por debajo de 30 el AUC es muy ruidoso."""

    auc_floor: float = 0.52
    """AUC mínima aceptable. Bajo esto, el modelo ya no aporta edge
    sobre azar. 0.52 es conservador (0.5 = random, 0.55 = bueno en
    trading)."""

    auc_trigger_consec_checks: int = 2
    """Nº de chequeos consecutivos bajo `auc_floor` para disparar
    reentreno. Evita que un mal día contamine la decisión."""

    # ── Política de disparo global ───────────────────────────────────
    require_multiple_signals: bool = True
    """Si True, requiere al menos 2 de {ks_drift, psi_drift, auc_drift}
    para disparar reentreno. Reduce falsos positivos; también aumenta
    latencia de detección."""


# ════════════════════════════════════════════════════════════════════
#  Output: reporte de drift.
# ════════════════════════════════════════════════════════════════════
@dataclass
class DriftReport:
    """
    Resultado de un chequeo de drift.

    Diseñado para ser JSON-serializable (solo tipos primitivos + listas)
    para que pueda escribirse en el `state.json` del LiveRunner y
    visualizarse en el dashboard.
    """
    timestamp: str = ""
    n_ref: int = 0
    n_recent: int = 0

    # KS-test
    ks_drift: bool = False
    ks_feature_frac: float = 0.0
    ks_worst_features: List[Tuple[str, float]] = field(default_factory=list)

    # PSI
    psi_drift: bool = False
    psi_feature_frac: float = 0.0
    psi_worst_features: List[Tuple[str, float]] = field(default_factory=list)

    # AUC
    auc_drift: bool = False
    auc_rolling: Optional[float] = None
    auc_consec_below: int = 0

    # Política global
    should_retrain: bool = False
    reason: str = ""

    def to_dict(self) -> Dict[str, object]:
        return {
            "timestamp": self.timestamp,
            "n_ref": self.n_ref,
            "n_recent": self.n_recent,
            "ks_drift": self.ks_drift,
            "ks_feature_frac": float(self.ks_feature_frac),
            "ks_worst": list(self.ks_worst_features),
            "psi_drift": self.psi_drift,
            "psi_feature_frac": float(self.psi_feature_frac),
            "psi_worst": list(self.psi_worst_features),
            "auc_drift": self.auc_drift,
            "auc_rolling": self.auc_rolling,
            "auc_consec_below": self.auc_consec_below,
            "should_retrain": self.should_retrain,
            "reason": self.reason,
        }


# ════════════════════════════════════════════════════════════════════
#  Funciones auxiliares: KS y PSI.
# ════════════════════════════════════════════════════════════════════
def _ks_test(a: np.ndarray, b: np.ndarray) -> float:
    """
    Calcula el p-value del KS-test de dos muestras. Usa `scipy.stats`.

    Devuelve 1.0 si alguna de las dos muestras está vacía (trivialmente
    "misma distribución": no podemos decir nada).
    """
    # scipy es una dependencia de sklearn → ya está instalado. Import
    # diferido por si en algún entorno no estuviese.
    from scipy.stats import ks_2samp

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if len(a) < 10 or len(b) < 10:
        return 1.0  # muy pocas muestras → no rechazar H0.
    try:
        _stat, p = ks_2samp(a, b, alternative="two-sided")
        return float(p)
    except Exception as exc:
        logger.debug("KS-test falló: %s", exc)
        return 1.0


def _psi(reference: np.ndarray, current: np.ndarray, n_bins: int = 10) -> float:
    """
    Population Stability Index.

    Formula:
        PSI = Σ (p_current - p_ref) * ln(p_current / p_ref)

    Los bins se eligen sobre los quantiles de `reference` (enfoque
    "expected-buckets"), para evitar que outliers del current creen
    bins degenerados. Se aplica un epsilon (1e-6) para evitar log(0).

    Devuelve +inf si ref está vacío o sólo tiene valores constantes
    (no tiene sentido medir PSI de una variable degenerada).
    """
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    reference = reference[~np.isnan(reference)]
    current = current[~np.isnan(current)]
    if len(reference) < 20 or len(current) < 20:
        return 0.0  # no suficientes datos → conservador: "sin drift".
    # Edge case: ref es constante → quantiles degenerados; reportamos 0.
    if np.unique(reference).size < 2:
        return 0.0

    # Bins por quantile sobre la referencia. Esto asegura bins con ~igual
    # población en ref, más robusto a outliers que bins equiespaciados.
    quantiles = np.quantile(reference, np.linspace(0, 1, n_bins + 1))
    # Dedup extremos por si hay ties (ej. valores binarios).
    quantiles = np.unique(quantiles)
    if len(quantiles) < 3:
        return 0.0

    # Bins [-inf, q1, q2, ..., +inf] para cubrir fuera-de-rango.
    edges = np.concatenate(([-np.inf], quantiles[1:-1], [np.inf]))

    ref_hist, _ = np.histogram(reference, bins=edges)
    cur_hist, _ = np.histogram(current, bins=edges)

    # Probabilidades por bin. epsilon para evitar 0 en log.
    eps = 1e-6
    p_ref = ref_hist / max(ref_hist.sum(), 1)
    p_cur = cur_hist / max(cur_hist.sum(), 1)
    p_ref = np.where(p_ref == 0, eps, p_ref)
    p_cur = np.where(p_cur == 0, eps, p_cur)

    return float(np.sum((p_cur - p_ref) * np.log(p_cur / p_ref)))


# ════════════════════════════════════════════════════════════════════
#  Detector principal.
# ════════════════════════════════════════════════════════════════════
class DriftDetector:
    """
    Encapsula el estado de referencia (training data) y los chequeos
    sucesivos contra la ventana live.

    Uso:
        det = DriftDetector(config)
        det.fit_reference(X_train, feature_cols)
        report = det.check(X_recent, recent_trade_outcomes)
    """

    def __init__(self, config: Optional[DriftConfig] = None) -> None:
        self.cfg = config or DriftConfig()
        # Buffer de la distribución de referencia: { col_name -> np.ndarray }.
        # Se guarda por columna (no el DataFrame completo) porque el live
        # puede construir features nuevas con pocas columnas y no queremos
        # penalizar por columnas que no existían.
        self._reference: Dict[str, np.ndarray] = {}
        # Contador de chequeos AUC consecutivos bajo floor. Persiste
        # entre checks — se resetea cuando AUC vuelve arriba.
        self._auc_consec_below: int = 0

    # ----- Referencia ----------------------------------------------------
    def fit_reference(
        self,
        X_reference: pd.DataFrame,
        feature_cols: Optional[List[str]] = None,
    ) -> None:
        """
        Almacena la distribución de referencia por feature.

        Guardamos los arrays limpios (sin NaN) para no tener que filtrar
        en cada check posterior.
        """
        if feature_cols is None:
            feature_cols = list(X_reference.columns)
        self._reference = {}
        for col in feature_cols:
            if col not in X_reference.columns:
                continue
            arr = X_reference[col].to_numpy(dtype=float, copy=False)
            arr = arr[~np.isnan(arr)]
            # Ignoramos columnas con muy pocos datos válidos (p. ej.
            # sentiment news_*, que en histórico es NaN casi siempre).
            if len(arr) < 20:
                continue
            self._reference[col] = arr
        logger.info(
            "Drift reference inicializada con %d features (descartadas %d por escasez de datos).",
            len(self._reference),
            len(feature_cols) - len(self._reference),
        )

    # ----- Chequeo principal --------------------------------------------
    def check(
        self,
        X_recent: pd.DataFrame,
        recent_trade_outcomes: Optional[pd.Series] = None,
        recent_trade_probas: Optional[pd.Series] = None,
    ) -> DriftReport:
        """
        Ejecuta los tres tests y devuelve un reporte.

        Args:
            X_recent : features de la ventana reciente (viva). Columnas
                       deben coincidir con las de `fit_reference` (las
                       no reconocidas se ignoran).
            recent_trade_outcomes : Serie 0/1 con los outcomes reales
                       de los trades cerrados recientemente (True=win,
                       False=loss). Usado para AUC rolling.
            recent_trade_probas : probabilidades predichas por el modelo
                       para esos mismos trades. Debe alinearse con
                       `recent_trade_outcomes`.
        """
        ts = pd.Timestamp.utcnow().isoformat()
        n_ref = sum(len(v) for v in self._reference.values())
        n_recent = len(X_recent) if X_recent is not None else 0

        # ── KS por feature ──
        ks_pvalues: Dict[str, float] = {}
        for col, ref_arr in self._reference.items():
            if col not in X_recent.columns:
                continue
            recent_arr = X_recent[col].to_numpy(dtype=float, copy=False)
            p = _ks_test(ref_arr, recent_arr)
            ks_pvalues[col] = p

        if ks_pvalues:
            n_failed = sum(1 for p in ks_pvalues.values() if p < self.cfg.ks_alpha)
            ks_frac = n_failed / len(ks_pvalues)
        else:
            ks_frac = 0.0
        ks_drift = ks_frac >= self.cfg.ks_feature_frac_threshold
        # Top 5 worst (p más bajos).
        ks_worst = sorted(ks_pvalues.items(), key=lambda kv: kv[1])[:5]

        # ── PSI por feature ──
        psi_values: Dict[str, float] = {}
        for col, ref_arr in self._reference.items():
            if col not in X_recent.columns:
                continue
            recent_arr = X_recent[col].to_numpy(dtype=float, copy=False)
            psi_values[col] = _psi(ref_arr, recent_arr, n_bins=self.cfg.psi_n_bins)

        if psi_values:
            n_strong = sum(1 for v in psi_values.values() if v > self.cfg.psi_strong)
            psi_frac = n_strong / len(psi_values)
        else:
            psi_frac = 0.0
        psi_drift = psi_frac >= self.cfg.psi_feature_frac_threshold
        psi_worst = sorted(psi_values.items(), key=lambda kv: kv[1], reverse=True)[:5]

        # ── AUC rolling sobre outcomes reales ──
        auc_rolling: Optional[float] = None
        auc_drift = False
        if (
            recent_trade_outcomes is not None
            and recent_trade_probas is not None
            and len(recent_trade_outcomes) >= self.cfg.auc_window_trades
        ):
            auc_rolling = _rolling_auc(recent_trade_outcomes, recent_trade_probas)
            if auc_rolling is not None and auc_rolling < self.cfg.auc_floor:
                self._auc_consec_below += 1
                if self._auc_consec_below >= self.cfg.auc_trigger_consec_checks:
                    auc_drift = True
            else:
                # Si subió por encima del floor, reseteamos el contador.
                self._auc_consec_below = 0

        # ── Política global ──
        signals = [ks_drift, psi_drift, auc_drift]
        n_fired = sum(signals)
        if self.cfg.require_multiple_signals:
            should_retrain = n_fired >= 2
        else:
            should_retrain = n_fired >= 1

        reason_parts = []
        if ks_drift:
            reason_parts.append(f"KS {ks_frac*100:.0f}% features degradadas")
        if psi_drift:
            reason_parts.append(f"PSI {psi_frac*100:.0f}% features PSI>{self.cfg.psi_strong}")
        if auc_drift:
            reason_parts.append(
                f"AUC {auc_rolling:.3f} < {self.cfg.auc_floor} x{self._auc_consec_below}"
            )
        reason = " & ".join(reason_parts) if reason_parts else "OK"

        return DriftReport(
            timestamp=ts,
            n_ref=n_ref,
            n_recent=n_recent,
            ks_drift=ks_drift,
            ks_feature_frac=float(ks_frac),
            ks_worst_features=[(c, float(p)) for c, p in ks_worst],
            psi_drift=psi_drift,
            psi_feature_frac=float(psi_frac),
            psi_worst_features=[(c, float(v)) for c, v in psi_worst],
            auc_drift=auc_drift,
            auc_rolling=auc_rolling,
            auc_consec_below=int(self._auc_consec_below),
            should_retrain=should_retrain,
            reason=reason,
        )

    # ----- Debug / utilidades -------------------------------------------
    def reset_auc_counter(self) -> None:
        """Usar tras un reentreno para empezar de cero con el contador AUC."""
        self._auc_consec_below = 0


# ════════════════════════════════════════════════════════════════════
#  Rolling AUC helper.
# ════════════════════════════════════════════════════════════════════
def _rolling_auc(outcomes: pd.Series, probas: pd.Series) -> Optional[float]:
    """
    AUC de los trades cerrados recientes.

    Devuelve None si hay una sola clase (todo win o todo loss) — en ese
    caso el AUC no está definido; mejor no decir nada que inventar.
    """
    from sklearn.metrics import roc_auc_score

    y = pd.Series(outcomes).astype(int).to_numpy()
    p = pd.Series(probas).astype(float).to_numpy()
    if len(y) != len(p) or len(y) == 0:
        return None
    if len(np.unique(y)) < 2:
        return None
    try:
        return float(roc_auc_score(y, p))
    except Exception as exc:
        logger.debug("rolling AUC falló: %s", exc)
        return None
