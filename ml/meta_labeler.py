"""
meta_labeler.py — Entrenador LightGBM para meta-labeling con walk-forward.

Flujo:
    1. Cargar datos OHLCV del símbolo.
    2. Construir features (features.FeatureBuilder).
    3. Generar señales con una estrategia base (por defecto EnsembleStrategy).
    4. Construir meta-labels (labels.meta_labels).
    5. Alinear features ↔ meta-labels.
    6. Walk-forward k-fold con "embargo" al cambio de fold para reducir
       data leakage.
    7. LightGBM binary classifier. Reportar AUC, precision, recall, f1,
       base_rate, uplift, y feature importance.
    8. Reentrenar sobre todo el histórico y persistir a disco.

El modelo se guarda como dict con: model, feature_cols, threshold (elegido
por maximizar F1 en el último fold OOS), tb_config, metrics.

El inferencer (`load_meta_labeler`) devuelve un objeto con métodos
`predict_proba(features_row)` y `should_take(features_row)`.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional, Tuple

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_fscore_support,
    accuracy_score,
)

from features import FeatureBuilder
from labels import TripleBarrierConfig, build_meta_labels
from strategies.base import BaseStrategy
from strategies.ensemble import build_default_ensemble


# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────
@dataclass
class MetaLabelerConfig:
    n_splits: int = 5              # folds para walk-forward
    embargo_bars: int = 5          # barras de "margen" entre train y test
    min_samples: int = 50          # nº mínimo de meta-labels para entrenar
    random_state: int = 42
    # Hiperparámetros LightGBM. Conservadores por defecto — el dataset es
    # pequeño (cientos de eventos típicamente).
    lgb_params: dict = field(
        default_factory=lambda: {
            "objective": "binary",
            "metric": "auc",
            "learning_rate": 0.05,
            "num_leaves": 31,
            "min_data_in_leaf": 10,
            "feature_fraction": 0.85,
            "bagging_fraction": 0.85,
            "bagging_freq": 5,
            "n_estimators": 300,
            "verbosity": -1,
        }
    )
    # Configuración de las etiquetas.
    tp_mult: float = 2.0
    sl_mult: float = 1.0
    max_bars: int = 10
    atr_period: int = 14


# ──────────────────────────────────────────────
# Trainer
# ──────────────────────────────────────────────
class MetaLabelerTrainer:
    def __init__(
        self,
        config: Optional[MetaLabelerConfig] = None,
        strategy: Optional[BaseStrategy] = None,
        feature_builder: Optional[FeatureBuilder] = None,
    ):
        self.config = config or MetaLabelerConfig()
        self.strategy = strategy or build_default_ensemble()
        self.feature_builder = feature_builder or FeatureBuilder()

    # ──────────────────────────────────────────
    def build_dataset(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
        """
        Devuelve (X, y, meta_df). X e y están alineados por timestamp de
        entrada de cada señal BUY de la estrategia base.
        """
        signals = self.strategy.generate_signals(df)
        features = self.feature_builder.build(df)
        tb_cfg = TripleBarrierConfig(
            tp_mult=self.config.tp_mult,
            sl_mult=self.config.sl_mult,
            max_bars=self.config.max_bars,
            atr_period=self.config.atr_period,
        )
        _, meta_df = build_meta_labels(df, signals, tb_config=tb_cfg)
        # Alineación estricta: para cada señal BUY tomamos el vector de
        # features de esa barra (ya disponible en el cierre).
        X = features.loc[meta_df.index].dropna()
        y = meta_df.loc[X.index, "meta"].astype(int)
        meta_df = meta_df.loc[X.index]
        return X, y, meta_df

    # ──────────────────────────────────────────
    def walk_forward_fit(self, X: pd.DataFrame, y: pd.Series) -> dict:
        """Walk-forward expanding window con embargo. Devuelve métricas OOS
        por fold y el modelo final re-entrenado sobre todo el conjunto."""
        n = len(X)
        if n < self.config.min_samples:
            raise ValueError(
                f"Muestras insuficientes para entrenar: {n} < {self.config.min_samples}. "
                f"Aumenta el histórico o relaja el régimen de la estrategia."
            )

        fold_size = n // (self.config.n_splits + 1)
        embargo = self.config.embargo_bars
        fold_metrics = []

        for i in range(1, self.config.n_splits + 1):
            train_end = fold_size * i
            test_start = train_end + embargo
            test_end = min(fold_size * (i + 1), n)
            if test_start >= test_end:
                continue

            X_train = X.iloc[:train_end]
            y_train = y.iloc[:train_end]
            X_test = X.iloc[test_start:test_end]
            y_test = y.iloc[test_start:test_end]

            # Requiere que haya al menos un representante de cada clase.
            if y_train.nunique() < 2 or len(X_train) < 20 or len(X_test) < 5:
                continue

            model = lgb.LGBMClassifier(**self.config.lgb_params, random_state=self.config.random_state)
            model.fit(X_train, y_train)
            proba = model.predict_proba(X_test)[:, 1]
            base_rate = float(y_test.mean()) if len(y_test) else float("nan")
            try:
                auc = float(roc_auc_score(y_test, proba)) if y_test.nunique() > 1 else float("nan")
            except ValueError:
                auc = float("nan")

            # Elegimos threshold que maximiza F1 en el set de test del fold.
            best_f1, best_thr = -1.0, 0.5
            for thr in np.linspace(0.3, 0.8, 26):
                pred = (proba >= thr).astype(int)
                prec, rec, f1, _ = precision_recall_fscore_support(
                    y_test, pred, average="binary", zero_division=0
                )
                if f1 > best_f1:
                    best_f1, best_thr = float(f1), float(thr)

            pred = (proba >= best_thr).astype(int)
            prec, rec, f1, _ = precision_recall_fscore_support(
                y_test, pred, average="binary", zero_division=0
            )
            acc = float(accuracy_score(y_test, pred))
            fold_metrics.append({
                "fold": i,
                "n_train": int(len(X_train)),
                "n_test": int(len(X_test)),
                "base_rate": base_rate,
                "auc": auc,
                "threshold": best_thr,
                "precision": float(prec),
                "recall": float(rec),
                "f1": float(f1),
                "accuracy": acc,
                "uplift_vs_base": float(prec - base_rate) if base_rate > 0 else float("nan"),
            })

        # Modelo final sobre todo el histórico.
        final_model = lgb.LGBMClassifier(**self.config.lgb_params, random_state=self.config.random_state)
        final_model.fit(X, y)
        threshold = float(np.median([m["threshold"] for m in fold_metrics])) if fold_metrics else 0.5
        importances = dict(
            sorted(
                zip(X.columns.tolist(), final_model.feature_importances_.astype(int).tolist()),
                key=lambda kv: kv[1],
                reverse=True,
            )
        )
        return {
            "model": final_model,
            "feature_cols": X.columns.tolist(),
            "threshold": threshold,
            "fold_metrics": fold_metrics,
            "feature_importance": importances,
        }

    # ──────────────────────────────────────────
    def train(self, df: pd.DataFrame) -> dict:
        X, y, meta_df = self.build_dataset(df)
        result = self.walk_forward_fit(X, y)
        result["config"] = asdict(self.config)
        result["n_samples"] = int(len(X))
        result["base_rate_global"] = float(y.mean()) if len(y) else float("nan")
        return result


# ──────────────────────────────────────────────
# Persistencia e inferencia
# ──────────────────────────────────────────────
def save_meta_labeler(result: dict, path: str) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(result, p)
    return p


class MetaLabelerInferencer:
    """Wrapper ligero alrededor de un modelo entrenado, pensado para usarse
    en live / backtest."""

    def __init__(self, payload: dict):
        self.model = payload["model"]
        self.feature_cols: List[str] = payload["feature_cols"]
        self.threshold: float = float(payload.get("threshold", 0.5))

    def predict_proba(self, X: pd.DataFrame) -> pd.Series:
        sub = X[self.feature_cols].copy()
        proba = self.model.predict_proba(sub)[:, 1]
        return pd.Series(proba, index=X.index, name="meta_proba")

    def should_take(self, X: pd.DataFrame) -> pd.Series:
        return (self.predict_proba(X) >= self.threshold).rename("meta_keep")


def load_meta_labeler(path: str) -> MetaLabelerInferencer:
    payload = joblib.load(path)
    return MetaLabelerInferencer(payload)
