"""
meta_labeler.py — Entrenador LightGBM para meta-labeling con walk-forward.

Flujo:
    1. Cargar datos OHLCV del símbolo.
    2. Construir features (features.FeatureBuilder).
    3. Generar señales con una estrategia base (por defecto EnsembleStrategy).
    4. Construir meta-labels (labels.meta_labels). Cada fila trae label
       binario (win/loss) Y el retorno realizado del trade hipotético.
    5. Alinear features ↔ meta-labels.
    6. Walk-forward k-fold con "embargo" al cambio de fold para reducir
       data leakage (los próximos N bares al borde train/test se descartan,
       evita que el modelo vea información de trades aún "vivos" al crossear
       del set de training al de test).
    7. LightGBM binary classifier. Reportar AUC, precision, recall, f1,
       base_rate, uplift, y feature importance.
    8. Reentrenar sobre todo el histórico y persistir a disco.

Selección de threshold (D):
    El threshold de la probabilidad (τ) — aquel por encima del cual una
    señal del ensemble se toma como buena — se elige de dos formas según
    `threshold_objective`:

      * "sharpe" (default): maximiza el Sharpe simulado de los
        trades realizados con los retornos realizados de cada trade
        hipotético (columna `ret` en meta_df). Más orientado a P&L.
      * "f1" (legacy): maximiza el F1 de clasificación. Orientado a
        precisión/recall del clasificador; puede no correlar con P&L
        (un F1 alto con entradas pequeñas no garantiza Sharpe alto).

Método de CV (F):
    `cv_method` selecciona cómo se generan los splits internos:

      * "walk_forward" (default): expanding window con embargo. Simple
        y respeta la flecha del tiempo, pero los folds tienen tamaño
        desigual y la varianza CV es alta.
      * "purged_kfold" (López de Prado): K folds disjuntos. Cada sample
        sirve como test exactamente una vez. Train = todo lo demás
        MENOS la zona de purge (samples cuyo horizonte de label se
        solapa con el test) MENOS la zona de embargo (los siguientes
        `embargo_bars` después del test). Métricas CV mucho más
        estables y honestas, a costa de mezclar futuro/pasado en train.

    Para deploy final se recomienda walk_forward (consistente con cómo
    operará el bot en producción). purged_kfold es más útil para
    auditar overfitting y comparar variantes de la pipeline.

El modelo se guarda como dict con: model, feature_cols, threshold (elegido
según `threshold_objective`), tb_config, metrics por fold.

El inferencer (`load_meta_labeler`) devuelve un objeto con métodos
`predict_proba(features_row)` y `should_take(features_row)`.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

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
from strategies.ensemble import build_default_ensemble, SOURCE_TREND, SOURCE_MR


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
    # ── Selección del threshold de probabilidad ──
    # "sharpe" (default): escoge τ que maximiza el Sharpe de los trades
    #     ganadores según la columna `ret` (retorno realizado).
    # "f1"            : comportamiento legacy. Maximiza F1 clasificatorio.
    threshold_objective: str = "sharpe"
    # Rango de candidatos τ explorado en la búsqueda lineal.
    threshold_min: float = 0.30
    threshold_max: float = 0.80
    threshold_n: int = 26
    # Número mínimo de trades que debe quedar por encima de τ en el set de
    # test de un fold para considerar ese τ válido. Evita Sharpe espurios
    # calculados sobre 2-3 trades. Si ningún τ alcanza el mínimo, se cae al
    # τ que deje más trades (cola del rango de búsqueda).
    min_trades_threshold: int = 10

    # ── Método de validación cruzada (F) ────────────────────────────────
    # "walk_forward" (default, comportamiento histórico): expanding window.
    #     Train = [0, train_end), test = [train_end+embargo, fold_end).
    #     Bias optimista: el test siempre es "futuro" pero la última fold
    #     se valida con muchos más datos que la primera (varianza alta).
    # "purged_kfold" (López de Prado): K folds disjuntas, cada una hace de
    #     test una vez. Train = todas las demás MENOS los samples cuyo
    #     horizonte de label se solapa con el test (purge) y MENOS los
    #     primeros `embargo_bars` después del test (embargo). Más honesto:
    #     todos los folds tienen el mismo tamaño y se eliminan los samples
    #     "infectados" por leakage temporal.
    cv_method: str = "walk_forward"

    # Horizonte de label usado por purged_kfold para el "purge". Por defecto
    # = max_bars (el timeout del triple-barrier), porque cada label puede
    # depender hasta `max_bars` barras hacia adelante. Si lo dejas en None,
    # se inicializa en `max_bars`.
    purge_bars: Optional[int] = None


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
    # Selección de threshold (ver docstring del módulo)
    # ──────────────────────────────────────────
    def _candidate_thresholds(self) -> np.ndarray:
        """Rango lineal de τ candidatos. Parametrizado en el config."""
        return np.linspace(
            self.config.threshold_min,
            self.config.threshold_max,
            self.config.threshold_n,
        )

    @staticmethod
    def _sharpe_of_trades(rets: np.ndarray) -> float:
        """
        Sharpe-like ratio sobre retornos de trades discretos.

        Fórmula usada:
            S = mean(r) / std(r) * sqrt(N)

        El factor sqrt(N) premia los thresholds con más trades válidos
        (más "confianza estadística"), y penaliza los que dejan muy pocas
        entradas aunque el mean/std sea bueno — esto es justo lo contrario
        de lo que F1 hace, y aquí es lo que queremos: un edge consistente
        en muchos trades es preferible a uno espectacular en 3.
        """
        if len(rets) == 0:
            return float("-inf")
        std = float(np.std(rets, ddof=1)) if len(rets) > 1 else 0.0
        if std <= 1e-12:
            # Sin varianza → ignoramos (posible look-ahead/degenerate).
            return float("-inf")
        return float(np.mean(rets) / std * np.sqrt(len(rets)))

    def _select_threshold(
        self,
        y_test: pd.Series,
        proba: np.ndarray,
        ret_test: Optional[pd.Series],
        objective: str,
    ) -> Tuple[float, float]:
        """
        Devuelve (best_threshold, best_score) según el criterio elegido.

        objective="f1": busca τ que maximiza F1 (precisión/recall de la
            clase 1). Score = F1.
        objective="sharpe": busca τ que maximiza el Sharpe sobre `ret_test`
            restringido al subconjunto `proba >= τ`. Score = Sharpe·sqrt(n).
            Si `ret_test` es None, se cae a F1 como salvaguarda.
        """
        cands = self._candidate_thresholds()

        # Fallback de seguridad: si no nos han pasado retornos realizados,
        # no podemos calcular Sharpe. Usamos F1.
        if objective == "sharpe" and ret_test is None:
            objective = "f1"

        if objective == "sharpe":
            ret_arr = ret_test.to_numpy().astype(float)
            best_score = float("-inf")
            best_thr = float(self.config.threshold_min)
            best_n = 0
            for thr in cands:
                mask = proba >= thr
                n = int(mask.sum())
                if n < self.config.min_trades_threshold:
                    continue
                score = self._sharpe_of_trades(ret_arr[mask])
                if score > best_score:
                    best_score, best_thr, best_n = score, float(thr), n
            # Si ningún τ tenía suficientes trades, caemos al más permisivo.
            if best_n == 0:
                best_thr = float(self.config.threshold_min)
                best_score = float("nan")
            return best_thr, best_score

        # objective == "f1"
        best_f1, best_thr = -1.0, 0.5
        for thr in cands:
            pred = (proba >= thr).astype(int)
            _, _, f1, _ = precision_recall_fscore_support(
                y_test, pred, average="binary", zero_division=0
            )
            if f1 > best_f1:
                best_f1, best_thr = float(f1), float(thr)
        return best_thr, best_f1

    # ──────────────────────────────────────────
    def build_dataset(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
        """
        Devuelve (X, y, meta_df). X e y están alineados por timestamp de
        entrada de cada señal BUY de la estrategia base.

        Si la estrategia implementa `generate_signals_with_source(df)` (p.ej.
        el `EnsembleStrategy`), `meta_df` incluye además una columna `source`
        con el origen de cada BUY, útil para modelos per-régimen (P4.2).
        """
        if hasattr(self.strategy, "generate_signals_with_source"):
            signals, sources = self.strategy.generate_signals_with_source(df)
        else:
            signals = self.strategy.generate_signals(df)
            sources = None
        features = self.feature_builder.build(df)
        tb_cfg = TripleBarrierConfig(
            tp_mult=self.config.tp_mult,
            sl_mult=self.config.sl_mult,
            max_bars=self.config.max_bars,
            atr_period=self.config.atr_period,
        )
        _, meta_df = build_meta_labels(df, signals, tb_config=tb_cfg, sources=sources)
        # Alineación estricta: para cada señal BUY tomamos el vector de
        # features de esa barra (ya disponible en el cierre).
        X = features.loc[meta_df.index].dropna()
        y = meta_df.loc[X.index, "meta"].astype(int)
        meta_df = meta_df.loc[X.index]
        return X, y, meta_df

    # ──────────────────────────────────────────
    def build_dataset_multi(
        self, dfs: Dict[str, pd.DataFrame]
    ) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
        """
        Construye un dataset combinado sobre varios símbolos. Cada entrada de
        `dfs` es {symbol: ohlcv_df}. Se etiqueta cada fila con la columna
        `symbol` para trazabilidad, y se ordenan **globalmente por timestamp**
        para que el walk-forward CV siga respetando la causalidad temporal
        entre símbolos.

        Args:
            dfs: dict símbolo → OHLCV DataFrame.

        Returns:
            (X, y, meta_df) igual que `build_dataset`, pero con la columna
            `symbol` en `meta_df`. Las features (`X`) sólo contienen valores
            numéricos (no incluimos one-hot del ticker por diseño — queremos
            un modelo que generalice a tickers fuera de muestra).
        """
        xs, ys, metas = [], [], []
        for sym, df in dfs.items():
            if df is None or df.empty:
                continue
            try:
                X_s, y_s, meta_s = self.build_dataset(df)
            except Exception as exc:  # noqa: BLE001
                # La estrategia puede no generar señales en algún ticker.
                print(f"[basket] {sym}: skip ({exc})")
                continue
            if len(X_s) == 0:
                print(f"[basket] {sym}: 0 muestras")
                continue
            meta_s = meta_s.copy()
            meta_s["symbol"] = sym
            xs.append(X_s)
            ys.append(y_s)
            metas.append(meta_s)
            print(f"[basket] {sym}: {len(X_s)} muestras  base_rate={y_s.mean():.3f}")

        if not xs:
            raise ValueError("Ningún símbolo produjo muestras válidas en el basket.")

        X = pd.concat(xs, axis=0)
        y = pd.concat(ys, axis=0)
        meta = pd.concat(metas, axis=0)

        # Ordenación global por timestamp → respeta causalidad en walk-forward.
        order = np.argsort(X.index.to_numpy(), kind="stable")
        X = X.iloc[order]
        y = y.iloc[order]
        meta = meta.iloc[order]
        return X, y, meta

    # ──────────────────────────────────────────
    def _cv_splits(self, n: int) -> Iterator[Tuple[np.ndarray, np.ndarray, int]]:
        """
        Generador de splits (train_idx, test_idx, fold_id).

        Soporta dos métodos según `self.config.cv_method`:

        ── walk_forward (default, expanding window con embargo) ────────────
            Para cada fold i ∈ [1..k]:
                train = [0, fold_size*i)
                test  = [fold_size*i + embargo, fold_size*(i+1))
            Mantiene la "flecha del tiempo": train siempre es pasado, test
            siempre futuro. Sesgo: el último fold tiene mucho más train que
            el primero, y la varianza de la métrica entre folds es alta.

        ── purged_kfold (López de Prado, más honesto) ──────────────────────
            Particiona [0,n) en K bloques contiguos disjuntos. Para cada uno:
                test  = el bloque
                train = todos los demás índices, EXCEPTO:
                  · purge      : los samples cuyo horizonte de label
                                 (max_bars hacia adelante) podría solaparse
                                 con el test → leakage. Se eliminan los
                                 últimos `purge_bars` antes del test.
                  · embargo    : los `embargo_bars` justo después del test,
                                 evita que la cola del trade durante el
                                 test contamine al train que viene a
                                 continuación.

            Resultado: cada sample sirve como test exactamente una vez, y
            los folds tienen el mismo tamaño. La varianza CV baja y la
            métrica es más representativa.

        Devuelve tuples (train_idx, test_idx, fold_id) con índices
        posicionales (no etiquetas) ordenados sobre el dataset que ya
        viene ordenado por timestamp global.
        """
        method = self.config.cv_method
        k = self.config.n_splits
        embargo = max(0, int(self.config.embargo_bars))
        # `purge_bars` por defecto = `max_bars` del triple-barrier; es el
        # horizonte máximo que mira cada label hacia adelante.
        horizon = (
            self.config.purge_bars
            if self.config.purge_bars is not None
            else self.config.max_bars
        )
        horizon = max(0, int(horizon))

        if method == "walk_forward":
            fold_size = n // (k + 1)
            for i in range(1, k + 1):
                train_end = fold_size * i
                test_start = train_end + embargo
                test_end = min(fold_size * (i + 1), n)
                if test_start >= test_end:
                    continue
                train_idx = np.arange(0, train_end, dtype=int)
                test_idx = np.arange(test_start, test_end, dtype=int)
                yield train_idx, test_idx, i

        elif method == "purged_kfold":
            # Tamaño base por fold (los últimos absorben los restos).
            fold_size = max(1, n // k)
            for i in range(k):
                test_lo = i * fold_size
                test_hi = (i + 1) * fold_size if i < k - 1 else n
                if test_hi <= test_lo:
                    continue
                test_idx = np.arange(test_lo, test_hi, dtype=int)
                # Zona contaminada que se elimina del train:
                #   - hasta `horizon` barras antes del test (overlap label).
                #   - hasta `embargo` barras después del test.
                purge_lo = max(0, test_lo - horizon - 1)
                purge_hi = min(n, test_hi + embargo)
                mask = np.ones(n, dtype=bool)
                mask[purge_lo:purge_hi] = False
                train_idx = np.where(mask)[0]
                yield train_idx, test_idx, i + 1

        else:
            raise ValueError(
                f"cv_method desconocido: {method!r}. Usa 'walk_forward' o 'purged_kfold'."
            )

    # ──────────────────────────────────────────
    def walk_forward_fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        ret: Optional[pd.Series] = None,
    ) -> dict:
        """
        Entrena con cross-validation según `self.config.cv_method`
        ('walk_forward' o 'purged_kfold').

        El nombre histórico de este método se mantiene por compatibilidad,
        aunque ahora soporta ambos métodos. Internamente delega los splits
        a `_cv_splits()`.

        Args:
            X    : features (n × k).
            y    : labels binarios (win=1 / loss=0), alineados con X.
            ret  : retornos realizados por trade (alineados con X). Si se
                   facilita, habilita la selección de threshold por Sharpe
                   (threshold_objective="sharpe"). Si es None, se cae a F1.

        Devuelve: dict con el modelo final (re-entrenado sobre TODO el
        histórico), feature_cols, threshold elegido, métricas por fold,
        cv_method utilizado y feature importance.
        """
        n = len(X)
        if n < self.config.min_samples:
            raise ValueError(
                f"Muestras insuficientes para entrenar: {n} < {self.config.min_samples}. "
                f"Aumenta el histórico o relaja el régimen de la estrategia."
            )

        fold_metrics = []
        objective = self.config.threshold_objective
        cv_method = self.config.cv_method

        for train_idx, test_idx, fold_id in self._cv_splits(n):
            X_train = X.iloc[train_idx]
            y_train = y.iloc[train_idx]
            X_test = X.iloc[test_idx]
            y_test = y.iloc[test_idx]
            ret_test = ret.iloc[test_idx] if ret is not None else None

            # Requiere que haya al menos un representante de cada clase.
            if y_train.nunique() < 2 or len(X_train) < 20 or len(X_test) < 5:
                continue

            i = fold_id  # alias usado por las métricas más abajo

            model = lgb.LGBMClassifier(**self.config.lgb_params, random_state=self.config.random_state)
            model.fit(X_train, y_train)
            proba = model.predict_proba(X_test)[:, 1]
            base_rate = float(y_test.mean()) if len(y_test) else float("nan")
            try:
                auc = float(roc_auc_score(y_test, proba)) if y_test.nunique() > 1 else float("nan")
            except ValueError:
                auc = float("nan")

            # Selección del threshold — sharpe (nuevo default) o f1 (legacy).
            best_thr, best_score = self._select_threshold(
                y_test=y_test, proba=proba, ret_test=ret_test, objective=objective,
            )

            # Métricas descriptivas SIEMPRE se reportan con el threshold
            # elegido, sea cual sea el criterio. Así podemos comparar F1 y
            # Sharpe del mismo fold.
            pred = (proba >= best_thr).astype(int)
            prec, rec, f1, _ = precision_recall_fscore_support(
                y_test, pred, average="binary", zero_division=0
            )
            acc = float(accuracy_score(y_test, pred))

            # Retorno medio de los trades aceptados por el modelo en este fold.
            if ret_test is not None:
                kept = ret_test[pred == 1]
                mean_ret = float(kept.mean()) if len(kept) else float("nan")
                trades_kept = int(len(kept))
            else:
                mean_ret = float("nan")
                trades_kept = int((pred == 1).sum())

            fold_metrics.append({
                "fold": i,
                "n_train": int(len(X_train)),
                "n_test": int(len(X_test)),
                "base_rate": base_rate,
                "auc": auc,
                "threshold": best_thr,
                "objective_score": float(best_score),
                "precision": float(prec),
                "recall": float(rec),
                "f1": float(f1),
                "accuracy": acc,
                "uplift_vs_base": float(prec - base_rate) if base_rate > 0 else float("nan"),
                "trades_kept": trades_kept,
                "mean_ret_kept": mean_ret,
            })

        # Modelo final sobre todo el histórico. Usamos la MEDIANA de los
        # thresholds elegidos por fold — robusta a outliers en folds pequeños.
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
            "threshold_objective": objective,
            "cv_method": cv_method,
            "fold_metrics": fold_metrics,
            "feature_importance": importances,
        }

    # ──────────────────────────────────────────
    @staticmethod
    def _extract_ret(meta_df: pd.DataFrame, index: pd.Index) -> Optional[pd.Series]:
        """Serie `ret` alineada con `index`, o None si la columna no existe."""
        if "ret" not in meta_df.columns:
            return None
        return meta_df.loc[index, "ret"].astype(float)

    def train(self, df: pd.DataFrame) -> dict:
        X, y, meta_df = self.build_dataset(df)
        ret = self._extract_ret(meta_df, X.index)
        result = self.walk_forward_fit(X, y, ret=ret)
        result["config"] = asdict(self.config)
        result["n_samples"] = int(len(X))
        result["base_rate_global"] = float(y.mean()) if len(y) else float("nan")
        result["train_symbols"] = None
        return result

    def train_multi(self, dfs: Dict[str, pd.DataFrame]) -> dict:
        """Entrena sobre un basket de símbolos. `dfs` = {symbol: ohlcv_df}."""
        X, y, meta_df = self.build_dataset_multi(dfs)
        ret = self._extract_ret(meta_df, X.index)
        result = self.walk_forward_fit(X, y, ret=ret)
        result["config"] = asdict(self.config)
        result["n_samples"] = int(len(X))
        result["base_rate_global"] = float(y.mean()) if len(y) else float("nan")
        result["train_symbols"] = list(dfs.keys())
        result["samples_per_symbol"] = (
            meta_df["symbol"].value_counts().to_dict()
            if "symbol" in meta_df.columns else {}
        )
        return result

    # ──────────────────────────────────────────
    def train_multi_regime_split(
        self, dfs: Dict[str, pd.DataFrame]
    ) -> dict:
        """
        Entrena DOS modelos LightGBM separados, uno por tipo de señal del
        ensemble (trend-following vs mean-reversion). Devuelve un payload
        compatible con `save_meta_labeler` y con `RegimeSplitInferencer`
        para uso en backtest/live.

        La lógica: los BUYs producidos por Donchian (tendencia) y por
        RSI(2) (mean-reversion) tienen microestructuras y drivers muy
        distintos. Un único modelo tiene que aprender a identificar de qué
        tipo es cada señal antes de predecir su éxito — ineficiente. Con
        dos modelos especializados cada uno aprende patrones más limpios.

        Requiere que `self.strategy` exponga `generate_signals_with_source`.
        """
        if not hasattr(self.strategy, "generate_signals_with_source"):
            raise ValueError(
                "La estrategia no expone 'generate_signals_with_source'. "
                "Usa EnsembleStrategy para el entrenamiento per-régimen."
            )
        X, y, meta_df = self.build_dataset_multi(dfs)
        if "source" not in meta_df.columns:
            raise ValueError(
                "Dataset sin columna 'source' — revisa build_dataset/meta_labels."
            )

        per_regime: Dict[str, dict] = {}
        for regime_name, src_label in [("trend", SOURCE_TREND), ("mean_revert", SOURCE_MR)]:
            mask = (meta_df["source"] == src_label).to_numpy()
            if mask.sum() == 0:
                print(f"[regime-split] {regime_name}: 0 muestras — se omite.")
                continue
            X_r = X.iloc[mask]
            y_r = y.iloc[mask]
            # `ret` per-régimen: sólo los retornos realizados de las señales
            # que vienen de ese sub-estrategia (trend vs MR). Importante que
            # los sub-thresholds se elijan con los retornos reales de su
            # propio tipo de señal.
            ret_r = self._extract_ret(meta_df, X_r.index)
            print(
                f"[regime-split] {regime_name}: {len(X_r)} muestras  "
                f"base_rate={y_r.mean():.3f}"
            )
            try:
                res = self.walk_forward_fit(X_r, y_r, ret=ret_r)
            except ValueError as exc:
                print(f"[regime-split] {regime_name}: skip fit ({exc})")
                continue
            res["config"] = asdict(self.config)
            res["n_samples"] = int(len(X_r))
            res["base_rate_global"] = float(y_r.mean()) if len(y_r) else float("nan")
            per_regime[regime_name] = res

        if not per_regime:
            raise ValueError("Ningún régimen produjo suficientes muestras para entrenar.")

        return {
            "kind": "regime_split",
            "regimes": per_regime,
            "train_symbols": list(dfs.keys()),
            "samples_per_symbol": (
                meta_df["symbol"].value_counts().to_dict()
                if "symbol" in meta_df.columns else {}
            ),
            "samples_per_regime": (
                meta_df["source"].value_counts().to_dict()
                if "source" in meta_df.columns else {}
            ),
            "config": asdict(self.config),
        }


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


class RegimeSplitInferencer:
    """
    Wrapper para payloads `kind="regime_split"` que contienen dos modelos:
    uno para señales trend-follower y otro para mean-reversion. En
    inferencia se llama al modelo correcto según el origen (`source`) de
    cada BUY del ensemble.
    """

    def __init__(self, payload: dict):
        regimes = payload.get("regimes", {})
        self.sub: Dict[str, MetaLabelerInferencer] = {
            name: MetaLabelerInferencer(res) for name, res in regimes.items()
        }
        self.payload = payload

    @property
    def regimes(self) -> List[str]:
        return list(self.sub.keys())

    def threshold_for(self, regime: str) -> float:
        return self.sub[regime].threshold if regime in self.sub else 0.5

    def should_take_by_source(
        self, X: pd.DataFrame, sources: pd.Series
    ) -> pd.Series:
        """
        Devuelve una Serie booleana del mismo tamaño que `X`/`sources`:
        True si el trade debe ejecutarse según el modelo apropiado al
        origen de cada señal. Para filas sin origen reconocido, False.
        """
        keep = pd.Series(False, index=X.index, name="meta_keep")
        sources = sources.reindex(X.index).fillna("")
        for regime_name, inf in self.sub.items():
            src_label = SOURCE_TREND if regime_name == "trend" else SOURCE_MR
            mask = (sources == src_label).to_numpy()
            if not mask.any():
                continue
            sub_X = X.iloc[mask]
            sub_keep = inf.should_take(sub_X)
            keep.loc[sub_X.index] = sub_keep.values
        return keep

    def predict_proba_by_source(
        self, X: pd.DataFrame, sources: pd.Series
    ) -> pd.Series:
        """
        Como `predict_proba` pero despachando al sub-modelo correcto por
        source. Filas sin source reconocido devuelven NaN.
        """
        out = pd.Series(np.nan, index=X.index, name="meta_proba")
        sources = sources.reindex(X.index).fillna("")
        for regime_name, inf in self.sub.items():
            src_label = SOURCE_TREND if regime_name == "trend" else SOURCE_MR
            mask = (sources == src_label).to_numpy()
            if not mask.any():
                continue
            sub_X = X.iloc[mask]
            sub_p = inf.predict_proba(sub_X)
            out.loc[sub_X.index] = sub_p.values
        return out


def load_meta_labeler(path: str):
    """
    Carga un payload serializado. Devuelve:
      - `RegimeSplitInferencer` si `kind == "regime_split"` (P4.2).
      - `MetaLabelerInferencer` en caso contrario (P4 / P4.1, global).
    """
    payload = joblib.load(path)
    if isinstance(payload, dict) and payload.get("kind") == "regime_split":
        return RegimeSplitInferencer(payload)
    return MetaLabelerInferencer(payload)
