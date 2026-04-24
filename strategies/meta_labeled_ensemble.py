"""
meta_labeled_ensemble.py — Ensemble + filtro ML por meta-labeling.

Ejecuta `EnsembleStrategy` (trend + mean-reversion con detector de régimen)
y filtra cada BUY consultando un `MetaLabelerInferencer` entrenado en P4.
Si la probabilidad del modelo para esa señal < threshold, se transforma
en HOLD.

Los SELL nunca se filtran (queremos poder salir incluso si el modelo no
tiene confianza).
"""

from __future__ import annotations

from typing import Optional, Union

import pandas as pd

from features import FeatureBuilder
from ml.meta_labeler import (
    MetaLabelerInferencer,
    RegimeSplitInferencer,
    load_meta_labeler,
)
from strategies.base import BaseStrategy, StrategySignal, Action
from strategies.ensemble import EnsembleStrategy, build_default_ensemble


AnyInferencer = Union[MetaLabelerInferencer, RegimeSplitInferencer]


class MetaLabeledEnsembleStrategy(BaseStrategy):
    def __init__(
        self,
        inferencer: AnyInferencer,
        base: Optional[EnsembleStrategy] = None,
        feature_builder: Optional[FeatureBuilder] = None,
        threshold_override: Optional[float] = None,
    ):
        self.inferencer = inferencer
        self.base = base or build_default_ensemble()
        self.feature_builder = feature_builder or FeatureBuilder()
        self._is_split = isinstance(inferencer, RegimeSplitInferencer)
        # Solo aplicable al modelo global. En modo split cada sub-modelo
        # lleva su propio threshold calibrado por fold.
        self.threshold = (
            threshold_override
            if threshold_override is not None
            else (inferencer.threshold if not self._is_split else float("nan"))
        )
        tag = "split" if self._is_split else f"thr={self.threshold:.2f}"
        self.name = f"MetaLabeled[{self.base.name}] {tag}"

    # ──────────────────────────────────────────
    def _feature_cols(self) -> list[str]:
        """Conjunto de features necesarias para la inferencia, en modo
        global o split."""
        if self._is_split:
            cols: list[str] = []
            seen = set()
            for inf in self.inferencer.sub.values():
                for c in inf.feature_cols:
                    if c not in seen:
                        cols.append(c)
                        seen.add(c)
            return cols
        return self.inferencer.feature_cols

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        if self._is_split:
            base_actions, sources = self.base.generate_signals_with_source(df)
        else:
            base_actions = self.base.generate_signals(df)
            sources = None
        features = self.feature_builder.build(df)

        # Candidates = todas las barras con BUY.
        buy_mask = base_actions == Action.BUY
        buy_idx = df.index[buy_mask]
        feats_buy = features.loc[buy_idx]
        valid_mask = feats_buy[self._feature_cols()].notna().all(axis=1)
        valid_idx = feats_buy.index[valid_mask]

        if len(valid_idx) > 0:
            if self._is_split:
                keep = self.inferencer.should_take_by_source(
                    feats_buy.loc[valid_idx], sources
                )
                rejected = valid_idx[~keep.to_numpy()]
            else:
                proba = self.inferencer.predict_proba(feats_buy.loc[valid_idx])
                keep = proba >= self.threshold
                rejected = valid_idx[~keep.to_numpy()]
        else:
            rejected = pd.Index([])

        # Las barras con features incompletas se rechazan (no abrimos a ciegas).
        rejected = rejected.union(feats_buy.index[~valid_mask])

        out = base_actions.copy()
        if len(rejected) > 0:
            out.loc[rejected] = Action.HOLD
        return out

    # ──────────────────────────────────────────
    def predict_proba_series(self, df: pd.DataFrame) -> pd.Series:
        """
        Devuelve la probabilidad de éxito estimada por el meta-modelo en cada
        barra donde el ensemble dispara BUY. En el resto de barras el valor
        es NaN. Pensado para el RiskOverlay (confidence-scaled sizing).
        """
        if self._is_split:
            base_actions, sources = self.base.generate_signals_with_source(df)
        else:
            base_actions = self.base.generate_signals(df)
            sources = None
        features = self.feature_builder.build(df)

        out = pd.Series(index=df.index, dtype=float)
        buy_mask = base_actions == Action.BUY
        buy_idx = df.index[buy_mask]
        if len(buy_idx) == 0:
            return out

        feats_buy = features.loc[buy_idx]
        valid_mask = feats_buy[self._feature_cols()].notna().all(axis=1)
        valid_idx = feats_buy.index[valid_mask]
        if len(valid_idx) == 0:
            return out

        X_valid = feats_buy.loc[valid_idx]
        if self._is_split:
            assert sources is not None
            proba = self.inferencer.predict_proba_by_source(X_valid, sources)
        else:
            proba = self.inferencer.predict_proba(X_valid)
        out.loc[valid_idx] = pd.Series(proba, index=valid_idx).values
        return out

    # ──────────────────────────────────────────
    def generate_signal(self, df: pd.DataFrame) -> StrategySignal:
        actions = self.generate_signals(df)
        action = actions.iloc[-1]
        price = float(df["close"].iloc[-1])
        if action == Action.BUY:
            return StrategySignal(Action.BUY, confidence=float(self.threshold),
                                  price=price, reason="Meta-label OK")
        if action == Action.SELL:
            return StrategySignal(Action.SELL, confidence=0.7, price=price,
                                  reason="Ensemble exit")
        return StrategySignal(Action.HOLD, confidence=0.0, price=price, reason="Filtrado por ML o sin señal")


# ──────────────────────────────────────────────
def build_meta_labeled_ensemble_from_file(model_path: str) -> MetaLabeledEnsembleStrategy:
    infer = load_meta_labeler(model_path)
    return MetaLabeledEnsembleStrategy(inferencer=infer)
