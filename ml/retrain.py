"""
ml/retrain.py — Fase K: orchestrator de auto-retrain del meta-labeler.

Responsabilidad
===============

Cuando el `DriftDetector` dice "should_retrain=True", este módulo se
encarga de:

1. **Cooldown**: no reentrenar más de una vez cada N días para evitar
   thrashing (reentrenos seguidos por mala suerte estadística).
2. **Lock**: asegurar que no hay dos reentrenos simultáneos (pueden
   pisarse el pickle).
3. **Reentreno seguro en segundo plano**: thread daemon que entrena
   un modelo nuevo sin parar el bot.
4. **Validación pre-swap**: el modelo nuevo sólo reemplaza al anterior
   si su AUC CV media > `min_auc_to_promote`. Si no, se descarta y
   el bot sigue con el modelo antiguo.
5. **Atomic swap**: `model.pkl` se reemplaza por renombre atómico
   (`os.replace`). El LiveRunner debe recargar el modelo al detectar
   el cambio (hot-reload).
6. **Backups**: cada pickle antiguo se conserva como `model.pkl.bak-<ts>`
   para que un rollback manual sea posible si el modelo nuevo empeora.

Diseño
======

El orchestrator NO ejecuta el LiveRunner. Sólo entrena el nuevo modelo
y lo pone en disco. Es el LiveRunner quien, a través del `ReloadWatcher`,
detecta el cambio de mtime del pickle y hace hot-reload.

Separar ambas responsabilidades permite:
    - Testear el reentreno aisladamente.
    - Reutilizar el orchestrator en un proceso cron externo si el
      usuario prefiere ese esquema.

Estado
======

El estado persiste en un JSON junto al modelo (`<model>.retrain.json`):

    {
      "last_retrain_at": "2026-04-26T14:30:00",
      "last_retrain_auc": 0.6321,
      "last_retrain_reason": "PSI drift 25% features > 0.25",
      "retrain_count": 3,
      "failed_count": 0
    }

Esto permite al dashboard mostrar cuándo fue el último reentreno y
al orchestrator aplicar el cooldown sin tocar el pickle principal.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════
#  Config y estado persistente.
# ════════════════════════════════════════════════════════════════════
@dataclass
class RetrainConfig:
    """
    Parámetros del orchestrator.

    Attributes
    ----------
    cooldown_days : float
        Días mínimos entre reentrenos. Default 7 — en un bot diario,
        reentrenar más seguido suele ser overkill y contaminar con
        overfit al último chunk de datos.

    min_auc_to_promote : float
        AUC CV media mínima para promocionar un modelo nuevo. Si el
        sweep devuelve un modelo peor (p. ej. por inestabilidad del
        dataset durante un crash), se conserva el anterior. 0.52 es
        el mismo floor del DriftDetector.

    retrain_period : str
        Período Yahoo (p. ej. "5y") que se pide para el retrain.
        Default "5y" — suficiente para cubrir distintos regímenes.

    retrain_symbols : List[str]
        Símbolos del basket de retrain. Si vacío, se usa `live_symbol`.

    train_args : Dict
        Argumentos adicionales pasados a `run_train_fn` — permite que
        el call-site inyecte `regime_split`, `include_sentiment`,
        `tune_hp`, etc. sin que el orchestrator los conozca.

    backup_old_pickle : bool
        Si True, renombra el pickle anterior a `.bak-<ts>` antes de
        promocionar el nuevo. Default True.

    wait_result : bool
        Si True, `trigger(...)` bloquea hasta que el retrain termina.
        Si False, lanza un daemon thread y retorna inmediatamente.
        Default False — para no parar el bot.
    """

    cooldown_days: float = 7.0
    min_auc_to_promote: float = 0.52
    retrain_period: str = "5y"
    retrain_symbols: List[str] = field(default_factory=list)
    train_args: Dict[str, object] = field(default_factory=dict)
    backup_old_pickle: bool = True
    wait_result: bool = False


# ════════════════════════════════════════════════════════════════════
#  Estado persistente (lee/escribe JSON al lado del pickle).
# ════════════════════════════════════════════════════════════════════
class _RetrainState:
    """
    Estado persistente del orchestrator. Se guarda como
    `<model_path>.retrain.json` y el orchestrator lo usa para:
        - Saber cuándo fue el último retrain (cooldown).
        - Contar retrains exitosos y fallidos.

    Si el archivo no existe, se crea vacío al primer save().
    """

    def __init__(self, model_path: Path) -> None:
        self.path = model_path.with_suffix(model_path.suffix + ".retrain.json")
        self.data: Dict[str, object] = {}
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    self.data = json.load(f)
            except Exception as exc:
                logger.warning("Retrain state corrupto en %s: %s", self.path, exc)
                self.data = {}

    def save(self) -> None:
        """Escritura atómica para no corromper si el proceso muere."""
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2, default=str)
        os.replace(tmp, self.path)

    def last_retrain_ts(self) -> Optional[pd.Timestamp]:
        raw = self.data.get("last_retrain_at")
        if not raw:
            return None
        try:
            return pd.Timestamp(raw)
        except Exception:
            return None


# ════════════════════════════════════════════════════════════════════
#  Orquestador.
# ════════════════════════════════════════════════════════════════════
# Tipo del callable de reentreno: recibe path de salida y kwargs;
# devuelve dict con (al menos) "auc" y "n_samples".
RetrainFn = Callable[..., Dict[str, object]]


class RetrainOrchestrator:
    """
    Coordina el reentreno automático.

    Uso:
        orch = RetrainOrchestrator(
            model_path="models/meta.pkl",
            retrain_fn=run_train_and_return_result,
            config=RetrainConfig(cooldown_days=7),
        )
        # En cada iter del live, después del drift check:
        if report.should_retrain:
            orch.trigger(reason=report.reason)
    """

    def __init__(
        self,
        model_path: str,
        retrain_fn: RetrainFn,
        config: Optional[RetrainConfig] = None,
    ) -> None:
        self.model_path = Path(model_path).expanduser().resolve()
        self.retrain_fn = retrain_fn
        self.cfg = config or RetrainConfig()
        self.state = _RetrainState(self.model_path)
        # Un único lock por instancia: asegura que sólo corre un
        # thread de retrain a la vez.
        self._lock = threading.Lock()
        self._retrain_thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    def is_in_cooldown(self) -> bool:
        """
        ¿Está el orchestrator en período de enfriamiento?

        Compara `last_retrain_at` con now y devuelve True si la
        diferencia es menor a `cooldown_days`.
        """
        last = self.state.last_retrain_ts()
        if last is None:
            return False
        now = pd.Timestamp.utcnow().tz_localize(None)
        last = last.tz_localize(None) if last.tzinfo is not None else last
        delta_days = (now - last).total_seconds() / 86400.0
        return delta_days < self.cfg.cooldown_days

    def is_retraining(self) -> bool:
        """¿Hay un retrain en curso?"""
        return self._retrain_thread is not None and self._retrain_thread.is_alive()

    # ------------------------------------------------------------------
    def trigger(self, reason: str) -> bool:
        """
        Lanza un retrain si no estamos en cooldown y no hay otro en curso.

        Devuelve True si se inició (o ejecutó, en wait_result=True) un
        retrain; False si se omitió por cooldown o lock tomado.
        """
        if self.is_in_cooldown():
            logger.info(
                "Retrain omitido: en cooldown (última ejecución hace < %.1f días).",
                self.cfg.cooldown_days,
            )
            return False
        if self.is_retraining():
            logger.info("Retrain omitido: ya hay otro en curso.")
            return False

        def _worker():
            with self._lock:
                try:
                    self._do_retrain(reason)
                except Exception as exc:
                    logger.exception("Retrain falló: %s", exc)
                    self.state.data["failed_count"] = int(self.state.data.get("failed_count", 0)) + 1
                    self.state.data["last_failure_at"] = pd.Timestamp.utcnow().isoformat()
                    self.state.data["last_failure_error"] = str(exc)
                    self.state.save()

        self._retrain_thread = threading.Thread(
            target=_worker, name="retrain-worker", daemon=True
        )
        self._retrain_thread.start()
        if self.cfg.wait_result:
            self._retrain_thread.join()
        return True

    # ------------------------------------------------------------------
    def _do_retrain(self, reason: str) -> None:
        """
        Implementa el pipeline completo de retrain:
            1. Escribir a path temporal.
            2. Validar AUC CV >= min_auc_to_promote.
            3. Backup del antiguo.
            4. Atomic swap.
            5. Actualizar state.
        """
        logger.info("═" * 50)
        logger.info("RETRAIN disparado: %s", reason)
        logger.info("═" * 50)

        t0 = time.time()
        tmp_path = self.model_path.with_suffix(self.model_path.suffix + ".new")

        # Pasamos al retrain_fn tanto el path de salida como los kwargs
        # que el call-site quiera inyectar (p.ej. regime_split, tune_hp…).
        kwargs = dict(self.cfg.train_args)
        kwargs.setdefault("out_path", str(tmp_path))
        if self.cfg.retrain_symbols:
            kwargs.setdefault("symbols", list(self.cfg.retrain_symbols))
        kwargs.setdefault("period", self.cfg.retrain_period)

        result = self.retrain_fn(**kwargs)
        auc = _extract_auc(result)
        elapsed = time.time() - t0
        logger.info(
            "Retrain completado en %.1fs — AUC CV media reportada: %s",
            elapsed,
            f"{auc:.4f}" if auc is not None else "NaN",
        )

        # ── Validación ─────────────────────────────────────────────
        if auc is None or auc < self.cfg.min_auc_to_promote:
            logger.warning(
                "AUC %s no supera umbral %.2f — descartamos el modelo nuevo. "
                "El bot sigue con el modelo antiguo.",
                f"{auc:.4f}" if auc is not None else "NaN",
                self.cfg.min_auc_to_promote,
            )
            # Borramos el tmp para no dejar basura.
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass
            self.state.data["failed_count"] = int(self.state.data.get("failed_count", 0)) + 1
            self.state.data["last_failure_at"] = pd.Timestamp.utcnow().isoformat()
            self.state.data["last_failure_reason"] = "auc_below_promote_threshold"
            self.state.data["last_failure_auc"] = auc
            self.state.save()
            return

        # ── Backup + swap ──────────────────────────────────────────
        if self.cfg.backup_old_pickle and self.model_path.exists():
            ts = pd.Timestamp.utcnow().strftime("%Y%m%dT%H%M%S")
            bak = self.model_path.with_suffix(self.model_path.suffix + f".bak-{ts}")
            try:
                self.model_path.replace(bak)
                logger.info("Backup antiguo → %s", bak)
            except Exception as exc:
                logger.warning("Backup falló (%s) — continuamos sin backup.", exc)

        # Swap atómico.
        try:
            tmp_path.replace(self.model_path)
        except Exception as exc:
            logger.error("Swap atómico falló: %s", exc)
            raise

        # ── Estado ────────────────────────────────────────────────
        self.state.data["last_retrain_at"] = pd.Timestamp.utcnow().isoformat()
        self.state.data["last_retrain_auc"] = auc
        self.state.data["last_retrain_reason"] = reason
        self.state.data["retrain_count"] = int(self.state.data.get("retrain_count", 0)) + 1
        self.state.data["last_retrain_elapsed_sec"] = elapsed
        self.state.save()
        logger.info(
            "Retrain OK. Nuevo modelo en %s. retrain_count=%s",
            self.model_path,
            self.state.data["retrain_count"],
        )


# ════════════════════════════════════════════════════════════════════
#  Helpers.
# ════════════════════════════════════════════════════════════════════
def _extract_auc(result: object) -> Optional[float]:
    """
    Intenta extraer la AUC CV media de un payload de retrain.

    Maneja el formato del `MetaLabelerTrainer`:
        - Modelo global: result["fold_metrics"] = [ {auc: .}, ... ]
        - Regime-split: result["regimes"]["trend"|"mean_revert"]["fold_metrics"]
    """
    if not isinstance(result, dict):
        return None

    def _mean_auc(metrics: list) -> Optional[float]:
        if not metrics:
            return None
        aucs = [m.get("auc") for m in metrics if m.get("auc") is not None]
        aucs = [float(a) for a in aucs if not (a is None or (isinstance(a, float) and np.isnan(a)))]
        return float(np.mean(aucs)) if aucs else None

    if result.get("kind") == "regime_split":
        aucs = []
        for name, sub in result.get("regimes", {}).items():
            a = _mean_auc(sub.get("fold_metrics", []))
            if a is not None:
                aucs.append(a)
        return float(np.mean(aucs)) if aucs else None

    return _mean_auc(result.get("fold_metrics", []))


# ════════════════════════════════════════════════════════════════════
#  Hot-reload helper para el LiveRunner.
# ════════════════════════════════════════════════════════════════════
class ModelReloadWatcher:
    """
    Watcher barato que detecta si el pickle ha sido reemplazado
    (mtime cambió) y devuelve True la primera vez que lo ve.

    El LiveRunner lo consulta en cada iteración. Si detecta reload,
    reconstruye la estrategia con el nuevo modelo.

    No usamos `watchdog` / inotify para no añadir deps; polling por
    mtime es suficiente porque el LiveRunner itera cada minuto/hora.
    """

    def __init__(self, model_path: str) -> None:
        self.path = Path(model_path).expanduser().resolve()
        self._last_mtime: Optional[float] = None
        self._init_mtime()

    def _init_mtime(self) -> None:
        try:
            self._last_mtime = self.path.stat().st_mtime if self.path.exists() else None
        except Exception:
            self._last_mtime = None

    def check(self) -> bool:
        """
        Devuelve True si el archivo ha cambiado desde la última llamada.

        Efecto colateral: actualiza el mtime almacenado. Por diseño,
        después de un reload exitoso, la siguiente llamada devolverá
        False hasta que haya otro cambio.
        """
        try:
            mtime = self.path.stat().st_mtime if self.path.exists() else None
        except Exception:
            mtime = None
        if mtime is None:
            return False
        if self._last_mtime is None:
            self._last_mtime = mtime
            return True
        if mtime != self._last_mtime:
            self._last_mtime = mtime
            return True
        return False
