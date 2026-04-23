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

from typing import Optional

import pandas as pd

from features import FeatureBuilder
from ml.meta_labeler import MetaLabelerInferencer, load_meta_labeler
from strategies.base import BaseStrategy, StrategySignal, Action
from strategies.ensemble import EnsembleStrategy, build_default_ensemble


class MetaLabeledEnsembleStrategy(BaseStrategy):
    def __init__(
        self,
        inferencer: MetaLabelerInferencer,
        base: Optional[EnsembleStrategy] = None,
        feature_builder: Optional[FeatureBuilder] = None,
        threshold_override: Optional[float] = None,
    ):
        self.inferencer = inferencer
        self.base = base or build_default_ensemble()
        self.feature_builder = feature_builder or FeatureBuilder()
        self.threshold = (
            threshold_override if threshold_override is not None else inferencer.threshold
        )
        self.name = f"MetaLabeled[{self.base.name}] thr={self.threshold:.2f}"

    # ──────────────────────────────────────────
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        base_actions = self.base.generate_signals(df)
        features = self.feature_builder.build(df)

        # Candidates = todas las barras con BUY.
        buy_mask = base_actions == Action.BUY
        buy_idx = df.index[buy_mask]
        # Las features del modelo pueden tener NaNs en los primeros bars;
        # sólo evaluamos las que tienen todas las columnas disponibles.
        feats_buy = features.loc[buy_idx]
        valid_mask = feats_buy[self.inferencer.feature_cols].notna().all(axis=1)
        valid_idx = feats_buy.index[valid_mask]

        if len(valid_idx) > 0:
            proba = self.inferencer.predict_proba(feats_buy.loc[valid_idx])
            keep = proba >= self.threshold
            rejected = valid_idx[~keep.to_numpy()]
        else:
            rejected = pd.Index([])

        # Las barras candidatas con features incompletas también se rechazan
        # (no queremos abrir a ciegas).
        rejected = rejected.union(feats_buy.index[~valid_mask])

        out = base_actions.copy()
        if len(rejected) > 0:
            out.loc[rejected] = Action.HOLD
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
