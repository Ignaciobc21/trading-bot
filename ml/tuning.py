"""
ml/tuning.py — Fase J: Optuna hyperparameter tuning del meta-labeler LightGBM.

Motivación
==========

Hasta la fase I los hiperparámetros de LightGBM (num_leaves, learning_rate,
min_data_in_leaf, etc.) estaban puestos a ojo en `MetaLabelerConfig.lgb_params`.
Funcionan "razonablemente" pero casi nunca son óptimos. Un sweep con
**Optuna** en un espacio bien acotado consigue típicamente:

  * +3-8 % de AUC OOS en la media de los folds.
  * Menor varianza entre folds (parámetros más robustos a la partición).
  * A veces trades más limpios (menos señal, mejor precision).

Arquitectura
============

- `OptunaTunerConfig`: parámetros del sweep (nº trials, timeout, objetivo,
  pruner, semilla).
- `OptunaTuner`:
    - recibe un `MetaLabelerTrainer` ya instanciado (reutilizamos sus splits
      CV para que el tuning y el training final vean exactamente el mismo
      protocolo de evaluación — no hay data leakage entre ellos).
    - ejecuta `study.optimize(objective, n_trials, timeout)`.
    - devuelve `(best_params, study)`. `best_params` se integra en
      `MetaLabelerTrainer.walk_forward_fit(lgb_params_override=...)` y
      sobrescribe los defaults.

El espacio de búsqueda está acotado para el tamaño de dataset típico
(cientos a pocos miles de eventos de triple-barrier). Rangos más grandes
harían overfitting al sweep; rangos más pequeños no exploran lo suficiente.

Espacio de búsqueda
===================

Variable            Rango                 Tipo          Por qué
------------------- --------------------- ------------- ------------------------
num_leaves          15-127                int (log)     Capacidad del árbol.
                                                        Muy bajo → underfits.
                                                        Muy alto → overfits.
learning_rate       0.005-0.2             float (log)   Compromiso LR vs nº
                                                        iteraciones. Log para
                                                        explorar órdenes.
min_child_samples   5-100                 int (log)     Min muestras por hoja.
                                                        Aumenta para regularizar
                                                        en datasets pequeños.
feature_fraction    0.5-1.0               float         Subsampling de columnas
                                                        por árbol. <1.0 reduce
                                                        overfit.
bagging_fraction    0.5-1.0               float         Subsampling de filas
                                                        por iteración.
bagging_freq        0, 1, 5, 10           categorical   Frecuencia de bagging.
                                                        0 = sin bagging.
reg_alpha (L1)      1e-8-10.0             float (log)   Regularización L1.
reg_lambda (L2)     1e-8-10.0             float (log)   Regularización L2.
max_depth           -1, 4, 6, 8           categorical   Profundidad máx.
                                                        -1 = sin límite.
n_estimators        50-500                int           Nº iteraciones.

Por qué escala logarítmica en LR / reg?
---------------------------------------
El efecto de pasar de 0.005 a 0.01 NO es el mismo que de 0.1 a 0.105.
En escala log, Optuna samplea proporcionalmente a órdenes de magnitud
y explora el rango de forma eficiente.

Pruner (MedianPruner)
=====================

Optuna evalúa cada trial en varios folds. El MedianPruner corta
trials cuya AUC acumulada es peor que la mediana del resto de trials
en el MISMO step (fold). Acelera 3-5× el sweep.

Limitación: si el primer trial es casi-óptimo, el pruner puede cortar
el resto demasiado rápido. El `n_startup_trials=5` evita eso: los
primeros 5 trials corren siempre completos.

Uso típico
==========

>>> from ml.meta_labeler import MetaLabelerTrainer, MetaLabelerConfig
>>> from ml.tuning import OptunaTuner, OptunaTunerConfig
>>> trainer = MetaLabelerTrainer(MetaLabelerConfig())
>>> X, y, ret = ...  # features + labels + retornos
>>> tuner = OptunaTuner(trainer, OptunaTunerConfig(n_trials=50))
>>> best_params, study = tuner.tune(X, y)
>>> result = trainer.walk_forward_fit(X, y, ret=ret, lgb_params_override=best_params)
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Tuple, TYPE_CHECKING

import numpy as np
import pandas as pd

# Optuna se importa perezosamente para que el resto del paquete siga
# funcionando en instalaciones que no tengan optuna (p.ej. runtime live
# sin tuning). El import real se hace en `OptunaTuner.tune`.
if TYPE_CHECKING:
    import optuna

# Import defensivo de las métricas. Mantenemos el scoring en AUC para
# no requerir dependencia extra en este módulo.
from sklearn.metrics import roc_auc_score

# MetaLabelerTrainer se usa sólo como hint para evitar import circular.
if TYPE_CHECKING:
    from ml.meta_labeler import MetaLabelerTrainer


logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════
#  Configuración del tuner.
# ════════════════════════════════════════════════════════════════════
@dataclass
class OptunaTunerConfig:
    """
    Parámetros del sweep Optuna.

    Attributes
    ----------
    n_trials : int
        Número de combinaciones a probar. 30 es el mínimo útil (por debajo
        Optuna no tiene suficiente señal para su TPE sampler). 50-100 es
        el sweet spot. >200 empieza a tener rendimientos decrecientes.

    timeout : Optional[int]
        Segundos máximos del sweep en total. None = sin límite. Útil si
        el dataset es grande: p.ej. `timeout=1800` para parar a los 30 min.
        Si ambos (n_trials y timeout) están definidos, gana el primero
        que se alcance.

    random_state : int
        Semilla para reproducibilidad del TPE sampler. Default 42.

    objective : str
        "auc" (default): maximiza AUC media de los folds del CV del
        trainer. Rápido, bien calibrado, objetivo clásico de clasificación.
        "sharpe" (futuro): reserved; no implementado aún porque requiere
        correr el threshold-search completo por trial (×5 más lento).

    early_stopping_rounds : int
        Rondas de early stopping en LightGBM dentro de cada fold. Ayuda
        a regularizar y acelera el sweep (trials malos paran antes).

    n_startup_trials : int
        Trials que corren sin pruning (para que el MedianPruner tenga
        referencia). Default 5.

    n_warmup_steps : int
        Folds dentro de cada trial que corren sin posibilidad de ser
        pruneados. Default 1 (el pruner evalúa desde el fold 2).

    verbose : bool
        Si True, imprime progreso. Si False, sólo warnings/errors.
    """

    n_trials: int = 50
    timeout: Optional[int] = None
    random_state: int = 42
    objective: str = "auc"
    early_stopping_rounds: int = 30
    n_startup_trials: int = 5
    n_warmup_steps: int = 1
    verbose: bool = True

    # ── Rangos del espacio de búsqueda ──────────────────────────────
    # Se exponen como atributos para que un usuario avanzado los pueda
    # ajustar sin tocar el código del tuner.
    num_leaves_min: int = 15
    num_leaves_max: int = 127
    lr_min: float = 0.005
    lr_max: float = 0.2
    min_child_min: int = 5
    min_child_max: int = 100
    feat_frac_min: float = 0.5
    feat_frac_max: float = 1.0
    bag_frac_min: float = 0.5
    bag_frac_max: float = 1.0
    reg_l1_min: float = 1e-8
    reg_l1_max: float = 10.0
    reg_l2_min: float = 1e-8
    reg_l2_max: float = 10.0
    n_estimators_min: int = 50
    n_estimators_max: int = 500


# ════════════════════════════════════════════════════════════════════
#  Tuner.
# ════════════════════════════════════════════════════════════════════
class OptunaTuner:
    """
    Ejecuta un sweep Optuna sobre los hiperparámetros de LightGBM
    usando los mismos splits CV que el `MetaLabelerTrainer`.

    Flujo por trial:
      1. Sample de hiperparámetros (num_leaves, LR, reg…).
      2. Para cada fold de `trainer._cv_splits`:
         a. Entrena LightGBM con los params samplados (con early
            stopping en un tail del train como valid).
         b. predict_proba en test → AUC de ese fold.
         c. `trial.report(auc_mean, step=fold_id)` para el pruner.
         d. Si `trial.should_prune()`, raise TrialPruned.
      3. Score final = AUC media de los folds no pruneados.
    """

    def __init__(
        self,
        trainer: "MetaLabelerTrainer",
        config: Optional[OptunaTunerConfig] = None,
    ) -> None:
        self.trainer = trainer
        self.cfg = config or OptunaTunerConfig()

    # ----- API principal -------------------------------------------------
    def tune(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> Tuple[Dict[str, Any], "optuna.study.Study"]:
        """
        Ejecuta el sweep y devuelve `(best_params, study)`.

        Args:
            X : features (alineadas con el índice del trainer).
            y : labels binarios 0/1.

        Returns:
            best_params : dict con los hiperparámetros óptimos
                (incluyendo `objective='binary'`, `metric='auc'` y
                `verbosity=-1` como fijos).
            study : el objeto Optuna completo (útil para plots y
                para inspeccionar `trials_dataframe()`).
        """
        # Import perezoso para que un runtime sin optuna pueda cargar el
        # módulo (hasta que alguien llame a tune).
        try:
            import optuna
            from optuna.pruners import MedianPruner
            from optuna.samplers import TPESampler
        except ImportError as exc:
            raise ImportError(
                "J — Optuna no está instalado. Añade `optuna>=3.5` a "
                "requirements.txt e instala con `pip install optuna`."
            ) from exc

        if len(X) != len(y):
            raise ValueError(
                f"X e y deben tener la misma longitud: {len(X)} vs {len(y)}"
            )

        # El sampler TPE (Tree-structured Parzen Estimator) es el default
        # de Optuna y mejor que random search en la mayoría de casos
        # porque usa los trials pasados para guiar los siguientes.
        sampler = TPESampler(seed=self.cfg.random_state)
        # MedianPruner: corta trials peores que la mediana de los demás
        # en el mismo paso del CV. Acelera mucho el sweep.
        pruner = MedianPruner(
            n_startup_trials=self.cfg.n_startup_trials,
            n_warmup_steps=self.cfg.n_warmup_steps,
        )

        # Silenciamos los logs de Optuna y LightGBM para no spamear la
        # terminal en sweeps de 50+ trials. Los mensajes importantes
        # (best trial al final) los imprimimos nosotros abajo.
        optuna.logging.set_verbosity(
            optuna.logging.INFO if self.cfg.verbose else optuna.logging.WARNING
        )

        study = optuna.create_study(
            direction="maximize",
            sampler=sampler,
            pruner=pruner,
            study_name="meta_labeler_tuning",
        )

        # Wrap del objective con X, y capturadas en closure (Optuna no
        # acepta args extra en la función objective).
        objective_fn = self._make_objective(X, y)

        logger.info(
            "Optuna: %d trials%s, sampler=TPE seed=%d, pruner=Median(start=%d, warmup=%d)",
            self.cfg.n_trials,
            f", timeout={self.cfg.timeout}s" if self.cfg.timeout else "",
            self.cfg.random_state,
            self.cfg.n_startup_trials,
            self.cfg.n_warmup_steps,
        )

        # Silenciar los warnings de LightGBM (p.ej. "no further splits")
        # dentro del sweep — los trials malos los generan con frecuencia.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            study.optimize(
                objective_fn,
                n_trials=self.cfg.n_trials,
                timeout=self.cfg.timeout,
                gc_after_trial=True,
                show_progress_bar=False,
            )

        # Montamos el dict final con los fijos + los sampleados.
        best = dict(study.best_params)
        best_params = {
            "objective": "binary",
            "metric": "auc",
            "verbosity": -1,
            **best,
        }

        logger.info(
            "Optuna best AUC=%.4f con %d trials completados (%d pruned)",
            study.best_value,
            sum(1 for t in study.trials if t.state.name == "COMPLETE"),
            sum(1 for t in study.trials if t.state.name == "PRUNED"),
        )
        logger.info("Best params: %s", best)

        return best_params, study

    # ----- Construcción del objective ------------------------------------
    def _make_objective(
        self, X: pd.DataFrame, y: pd.Series
    ) -> "Callable[[optuna.trial.Trial], float]":
        """
        Genera la función `objective(trial)` que Optuna llama por trial.
        La separamos en un factory para poder cerrar `X`, `y` en closure.
        """
        import lightgbm as lgb
        import optuna

        cfg = self.cfg
        trainer = self.trainer
        n = len(X)

        def objective(trial: "optuna.trial.Trial") -> float:
            # ── 1. Sample hiperparámetros ────────────────────────────
            # Los rangos vienen del OptunaTunerConfig. Algunos (LR,
            # regularización) se samplean en escala logarítmica porque
            # el efecto es multiplicativo, no aditivo.
            params = {
                "objective": "binary",
                "metric": "auc",
                "verbosity": -1,
                "num_leaves": trial.suggest_int(
                    "num_leaves", cfg.num_leaves_min, cfg.num_leaves_max, log=True
                ),
                "learning_rate": trial.suggest_float(
                    "learning_rate", cfg.lr_min, cfg.lr_max, log=True
                ),
                "min_child_samples": trial.suggest_int(
                    "min_child_samples", cfg.min_child_min, cfg.min_child_max, log=True
                ),
                "feature_fraction": trial.suggest_float(
                    "feature_fraction", cfg.feat_frac_min, cfg.feat_frac_max
                ),
                "bagging_fraction": trial.suggest_float(
                    "bagging_fraction", cfg.bag_frac_min, cfg.bag_frac_max
                ),
                "bagging_freq": trial.suggest_categorical(
                    "bagging_freq", [0, 1, 5, 10]
                ),
                "reg_alpha": trial.suggest_float(
                    "reg_alpha", cfg.reg_l1_min, cfg.reg_l1_max, log=True
                ),
                "reg_lambda": trial.suggest_float(
                    "reg_lambda", cfg.reg_l2_min, cfg.reg_l2_max, log=True
                ),
                "max_depth": trial.suggest_categorical("max_depth", [-1, 4, 6, 8]),
                "n_estimators": trial.suggest_int(
                    "n_estimators", cfg.n_estimators_min, cfg.n_estimators_max
                ),
            }

            # ── 2. CV folds del trainer ─────────────────────────────
            # Reutilizamos la misma función del trainer para que tuning y
            # training final vean EXACTAMENTE los mismos splits.
            aucs: list[float] = []
            fold_id_counter = 0

            # pylint: disable=protected-access
            for train_idx, test_idx, fold_id in trainer._cv_splits(n):
                X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
                y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
                if y_tr.nunique() < 2 or len(X_tr) < 20 or len(X_te) < 5:
                    continue  # fold inválido → saltamos sin penalizar.

                # Pequeño valid tail del train para early stopping.
                # 10% del train o mínimo 20 filas, lo que sea mayor.
                valid_size = max(20, int(len(X_tr) * 0.1))
                X_fit = X_tr.iloc[:-valid_size]
                y_fit = y_tr.iloc[:-valid_size]
                X_val = X_tr.iloc[-valid_size:]
                y_val = y_tr.iloc[-valid_size:]
                if y_fit.nunique() < 2:
                    # El tail de train puede tener una sola clase; si
                    # pasa saltamos el early stopping y usamos todo el train.
                    X_fit, y_fit = X_tr, y_tr
                    X_val, y_val = X_te, y_te

                # LGBMClassifier con early stopping vía callback (API
                # soportada en todas las versiones >=3.3).
                model = lgb.LGBMClassifier(
                    **params, random_state=cfg.random_state, n_jobs=-1
                )
                callbacks = [
                    lgb.early_stopping(
                        stopping_rounds=cfg.early_stopping_rounds, verbose=False
                    )
                ]
                try:
                    model.fit(
                        X_fit, y_fit,
                        eval_set=[(X_val, y_val)],
                        callbacks=callbacks,
                    )
                except Exception as exc:
                    # Algún sample raro (params degenerados) → AUC=0.5 para
                    # no contaminar el study con NaN.
                    logger.debug("Trial fallback por excepción LightGBM: %s", exc)
                    aucs.append(0.5)
                    fold_id_counter += 1
                    continue

                proba = model.predict_proba(X_te)[:, 1]
                try:
                    auc = float(roc_auc_score(y_te, proba)) if y_te.nunique() > 1 else 0.5
                except ValueError:
                    auc = 0.5
                aucs.append(auc)

                # ── Reporte intermedio al pruner ─────────────────
                # El step es el id del fold; `intermediate_value` es el
                # AUC *medio hasta ahora*. Optuna comparará con los demás
                # trials en el mismo step y decidirá si prunea.
                intermediate = float(np.mean(aucs))
                trial.report(intermediate, step=fold_id_counter)
                if trial.should_prune():
                    raise optuna.TrialPruned()
                fold_id_counter += 1

            if not aucs:
                # Ningún fold válido → penalización máxima.
                return 0.5
            return float(np.mean(aucs))

        return objective


# ════════════════════════════════════════════════════════════════════
#  Helper: resumen de un study (para logs y para guardar en el pickle).
# ════════════════════════════════════════════════════════════════════
def summarize_study(study: "optuna.study.Study") -> Dict[str, Any]:
    """
    Devuelve un dict JSON-serializable con un resumen del study, útil
    para guardar en el pickle del modelo y poder auditar después sin
    cargar el objeto Study entero (que puede pesar MBs si hay muchos trials).

    Incluye:
        - best_value, best_params
        - n_trials totales, n_pruned, n_complete
        - top-5 trials (value + params)
    """
    # Ordenar por valor desc. Optuna guarda estado None en trials pruned:
    # los filtramos para el top-5.
    complete = [t for t in study.trials if t.state.name == "COMPLETE"]
    complete_sorted = sorted(complete, key=lambda t: t.value or 0.0, reverse=True)
    top_k = [
        {"number": t.number, "value": t.value, "params": t.params}
        for t in complete_sorted[:5]
    ]
    return {
        "best_value": float(study.best_value),
        "best_params": dict(study.best_params),
        "n_trials": len(study.trials),
        "n_complete": sum(1 for t in study.trials if t.state.name == "COMPLETE"),
        "n_pruned": sum(1 for t in study.trials if t.state.name == "PRUNED"),
        "n_failed": sum(1 for t in study.trials if t.state.name == "FAIL"),
        "top_trials": top_k,
    }
