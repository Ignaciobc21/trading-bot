"""
live_runner.py — Bucle live / paper trading genérico.

Ejecuta cualquier `BaseStrategy` (p.ej. `MetaLabeledEnsembleStrategy`) contra
un broker real (Alpaca) o en modo `dry_run=True` (sin enviar órdenes, sólo
logging). Opcionalmente aplica el **risk overlay** (P6) para escalar el
tamaño de las entradas según vol targeting + regime + confidence del
meta-modelo.

Objetivos de diseño:
    - Misma "lógica de señal" que el backtest: el live consume la serie
      BUY/SELL/HOLD del mismo `generate_signals` que se usó para probar
      la estrategia. Cualquier drift entre live y backtest se debería
      a ejecución (slippage, horarios), no a la señal.
    - Modo "dry-run": se puede arrancar sin credenciales de Alpaca para
      validar la señal y el sizing. Nunca envía órdenes reales.
    - Stoppable limpio: Ctrl+C corta el loop y permite cerrar conexiones.

Flujo de cada iteración (`run_once`):
    1. Descargar las N últimas barras (Alpaca o Yahoo fallback).
    2. Recalcular features + señal. Devuelve una `StrategySignal` para la
       última barra cerrada.
    3. Consultar posiciones abiertas en el broker.
    4. Si BUY y no hay posición abierta → calcular tamaño (risk overlay
       si está activo) y abrir posición.
    5. Si SELL y hay posición abierta → cerrar posición.
    6. Si hay posición abierta → chequear SL/TP manuales con precio vivo.

El bucle `run()` lo repite cada `sleep_seconds` o hasta `max_iters`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

from strategies.base import BaseStrategy, StrategySignal, Action
from utils.logger import get_logger

logger = get_logger(__name__)


# Mapa timeframe (Alpaca naming) → segundos de sleep entre iteraciones.
# Usado como fallback si el usuario no pasa sleep_seconds explícito.
_TF_SLEEP = {
    "1Min": 60,
    "5Min": 300,
    "15Min": 900,
    "1Hour": 3600,
    "4Hour": 14400,
    "1Day": 86400,
}


@dataclass
class LiveConfig:
    """
    Parámetros de ejecución del live runner.
    """
    symbol: str = "AAPL"
    timeframe: str = "1Hour"             # Granularidad de barras.
    history_bars: int = 300              # Barras históricas a descargar por iteración (warmup features).
    base_position_size_pct: float = 2.0  # % del capital por operación (sizing base).
    max_open_positions: int = 1          # Cap de posiciones simultáneas (single-symbol default).
    sleep_seconds: Optional[int] = None  # Si None, usa _TF_SLEEP[timeframe].
    dry_run: bool = False                # Si True, no envía órdenes a Alpaca.
    stop_loss_pct: float = 0.0           # 0 = desactivado. Porcentaje SL intra-barra.
    take_profit_pct: float = 0.0         # 0 = desactivado. Porcentaje TP intra-barra.
    max_iters: Optional[int] = None      # Número máximo de iteraciones (None=infinito).
    # Origen de datos: "alpaca" usa el api del Broker. "yahoo" cae a DataFetcher.fetch_yahoo
    # (útil si no tienes suscripción de market data en Alpaca).
    data_source: str = "alpaca"
    yahoo_period: str = "6mo"            # Sólo si data_source="yahoo".
    yahoo_interval: str = "1h"
    # Notificaciones Telegram.
    send_telegram: bool = True
    # Dashboard (H): si se especifica, el runner escribe un snapshot
    # JSON tras cada iteración con estado actual + histórico para que
    # `dashboard/app.py` pueda monitorizarlo en vivo.
    state_path: Optional[str] = None
    state_history_size: int = 200        # nº máximo de decisiones a guardar en el snapshot.
    # Opcional: filtro de horario de mercado (09:30-16:00 NY). Evita que el
    # bot intente operar con quotes rancios en horario cerrado.
    only_market_hours: bool = False
    # ── K: Auto-retrain / drift detection ─────────────────────────
    # Si `drift_monitor` no es None, el LiveRunner ejecutará un chequeo
    # de drift cada `drift_check_every_iters` iteraciones (no en cada
    # barra para no gastar CPU). Si el monitor dispara should_retrain,
    # se llama a `retrain_orchestrator.trigger(...)`. Ambas piezas son
    # opcionales — si no se pasan, el bucle es idéntico al histórico.
    drift_check_every_iters: int = 20
    # Path del pickle a vigilar con `ModelReloadWatcher` para hot-reload
    # después de un retrain. Si None, no se hace hot-reload.
    model_reload_path: Optional[str] = None


@dataclass
class LiveDecision:
    """Snapshot de la decisión tomada en una iteración. Útil para tests."""
    timestamp: pd.Timestamp
    action: str                   # "BUY" | "SELL" | "HOLD" | "SKIP"
    reason: str
    price: float
    size_multiplier: float = 1.0
    proba: Optional[float] = None
    executed: bool = False
    extras: Dict[str, Any] = field(default_factory=dict)


# ──────────────────────────────────────────────
# Data source helpers
# ──────────────────────────────────────────────
def _fetch_bars_alpaca(api, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
    """
    Descarga las últimas `limit` barras desde Alpaca.

    Mapea timeframes "Alpaca-style" (p.ej. "1Hour") al enum `TimeFrame` de
    alpaca_trade_api. Si el símbolo es de crypto (BTC-USD) el endpoint de
    `get_bars` sigue funcionando siempre que la cuenta tenga acceso cripto.
    """
    from alpaca_trade_api.rest import TimeFrame
    from datetime import datetime, timedelta, timezone

    tf_map = {
        "1Min": TimeFrame.Minute,
        "5Min": TimeFrame(5, "Min"),
        "15Min": TimeFrame(15, "Min"),
        "1Hour": TimeFrame.Hour,
        "1Day": TimeFrame.Day,
    }
    tf = tf_map.get(timeframe, TimeFrame.Hour)

    # Ventana temporal "generosa" para asegurarnos de tener `limit` barras
    # en horarios con huecos (weekends, early close, etc). Alpaca trunca
    # el historial si pedimos más de lo que hay.
    end = datetime.now(timezone.utc)
    if timeframe.endswith("Min"):
        start = end - timedelta(days=max(5, limit // 78))
    elif timeframe == "1Hour":
        start = end - timedelta(days=max(15, limit // 7))
    else:
        start = end - timedelta(days=max(365, limit * 2))

    bars = api.get_bars(
        symbol, tf, start=start.isoformat(), end=end.isoformat()
    ).df
    if bars.empty:
        return bars
    bars.columns = [c.lower() for c in bars.columns]
    # Nos quedamos con las columnas OHLCV canónicas y con las últimas `limit` filas.
    cols = [c for c in ["open", "high", "low", "close", "volume"] if c in bars.columns]
    return bars[cols].tail(limit)


def _fetch_bars_yahoo(symbol: str, period: str, interval: str, limit: int) -> pd.DataFrame:
    """Fallback vía Yahoo Finance — datos con hasta 15-20 min de delay."""
    from data.fetcher import DataFetcher
    df = DataFetcher.fetch_yahoo(ticker=symbol, period=period, interval=interval)
    if df.empty:
        return df
    cols = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
    return df[cols].tail(limit)


# ──────────────────────────────────────────────
# Helpers de sizing
# ──────────────────────────────────────────────
def _size_multiplier_last_bar(
    strategy: BaseStrategy,
    df: pd.DataFrame,
    overlay,                         # risk.overlay.RiskOverlay | None
) -> tuple[float, Optional[float]]:
    """
    Calcula el size multiplier del overlay para la ÚLTIMA barra.

    Devuelve (size_mult, proba). Si overlay=None, devuelve (1.0, None).

    La idea: reusamos exactamente el mismo pipeline que el backtest
    (`overlay.size_multiplier(df, regime, proba)`) y nos quedamos con el
    último valor de la serie — que es la barra sobre la que estamos
    decidiendo ahora. Esto mantiene paridad live↔backtest.
    """
    if overlay is None:
        return 1.0, None

    # Régimen por barra — la estrategia ensemble expone la serie vía su
    # detector. Las estrategias que no lo tengan simplemente no contribuyen
    # al regime-sizing (overlay lo ignora con fillna(0)→1 por fillback).
    regime_series = None
    base = getattr(strategy, "base", strategy)
    detector = getattr(base, "detector", None)
    if detector is not None and hasattr(detector, "classify"):
        regime_series = detector.classify(df)

    # Probabilidades del meta-modelo (sólo meta_ensemble las expone).
    proba_series = None
    if hasattr(strategy, "predict_proba_series"):
        proba_series = strategy.predict_proba_series(df)

    mult_series = overlay.size_multiplier(
        df=df, regime=regime_series, proba=proba_series
    )
    last_mult = float(mult_series.iloc[-1])
    last_proba = float(proba_series.iloc[-1]) if proba_series is not None and pd.notna(proba_series.iloc[-1]) else None
    return last_mult, last_proba


# ──────────────────────────────────────────────
# Runner
# ──────────────────────────────────────────────
class LiveRunner:
    """
    Motor live/paper configurable. Tras construirlo, llamar a `run()` o
    `run_once()`.

    Colaboradores:
      - strategy  : implementa `generate_signal(df)` → StrategySignal
      - broker    : opcional; si None y dry_run=True, no conecta con Alpaca.
      - storage   : opcional; persiste trades si se pasa.
      - risk_mgr  : opcional; validación previa al open_position.
      - overlay   : opcional; `risk.overlay.RiskOverlay`. Si está activo,
                    se usa para escalar el sizing del BUY.
    """

    def __init__(
        self,
        strategy: BaseStrategy,
        config: Optional[LiveConfig] = None,
        broker=None,
        storage=None,
        risk_mgr=None,
        overlay=None,
        telegram_fn: Optional[Callable[[str], None]] = None,
        drift_detector=None,
        retrain_orchestrator=None,
    ):
        self.strategy = strategy
        self.config = config or LiveConfig()
        self.broker = broker
        self.storage = storage
        self.risk = risk_mgr
        self.overlay = overlay
        self.telegram_fn = telegram_fn
        # K: detector opcional de drift + orquestador opcional de retrain.
        # Si alguno es None, su rama se salta (retro-compat).
        self.drift_detector = drift_detector
        self.retrain_orchestrator = retrain_orchestrator

        # Estado interno mínimo — el "source of truth" siguen siendo el
        # broker y la base de datos. Este cache es sólo para evitar
        # llamadas redundantes al API dentro de una misma iteración.
        self._open_positions_cache: List[Dict] = []
        self._iter_count: int = 0
        # Histórico de decisiones para el dashboard (deque circular limitado
        # por `state_history_size`). Se popula tras cada iteración.
        self._decision_history: List[Dict[str, Any]] = []
        # K: último reporte de drift para el snapshot. Se refresca sólo
        # en los iters en los que se ejecuta el chequeo.
        self._last_drift_report: Optional[Dict[str, Any]] = None
        # K: watcher de hot-reload del modelo (si el orchestrator lo
        # reemplaza, el LiveRunner recarga la estrategia sin reiniciar).
        self._reload_watcher = None
        if self.config.model_reload_path:
            from ml.retrain import ModelReloadWatcher
            self._reload_watcher = ModelReloadWatcher(self.config.model_reload_path)

        # Validaciones básicas.
        if self.config.dry_run:
            logger.warning("LIVE RUNNER EN MODO DRY-RUN — no se enviarán órdenes reales.")
        if self.config.state_path:
            logger.info("State snapshot habilitado — escritura cada iter en %s", self.config.state_path)

    # ──────────────────────────────────────────
    def _notify(self, msg: str) -> None:
        """Manda una notificación si hay callback configurado (Telegram)."""
        if self.telegram_fn and self.config.send_telegram:
            try:
                self.telegram_fn(msg)
            except Exception as exc:
                logger.warning("Telegram notify falló: %s", exc)

    # ──────────────────────────────────────────
    def _fetch_bars(self) -> pd.DataFrame:
        """Obtiene el histórico reciente según `data_source`."""
        cfg = self.config
        if cfg.data_source == "alpaca":
            if self.broker is None or getattr(self.broker, "api", None) is None:
                raise RuntimeError("data_source=alpaca requiere un Broker inicializado.")
            return _fetch_bars_alpaca(
                self.broker.api, cfg.symbol, cfg.timeframe, cfg.history_bars
            )
        if cfg.data_source == "yahoo":
            return _fetch_bars_yahoo(
                cfg.symbol, cfg.yahoo_period, cfg.yahoo_interval, cfg.history_bars
            )
        raise ValueError(f"data_source desconocido: {cfg.data_source}")

    # ──────────────────────────────────────────
    def _refresh_positions(self) -> List[Dict]:
        """
        Posiciones abiertas del símbolo objetivo.

        En dry-run (sin broker) asumimos siempre 0 posiciones. Esta
        simplificación es suficiente para validar la señal; en un dry-run
        real con "paper" ficticio habría que simular persistencia.
        """
        if self.broker is None:
            self._open_positions_cache = []
            return []
        try:
            positions = self.broker.get_positions()
        except Exception as exc:
            logger.warning("No pude consultar posiciones: %s", exc)
            positions = []
        # Filtramos por símbolo — el broker devuelve TODAS las del account.
        self._open_positions_cache = [
            p for p in positions if p.get("symbol") == self.config.symbol
        ]
        return self._open_positions_cache

    # ──────────────────────────────────────────
    def _compute_size_qty(self, price: float, size_mult: float) -> float:
        """
        Convierte el % del capital (y el multiplier del overlay) en una
        cantidad de shares aproximada.

        En dry-run usamos un capital nominal de 10 000 $ (INITIAL_CAPITAL).
        En live sano consultamos el balance real del broker.
        """
        cfg = self.config
        capital = 10_000.0
        if self.broker is not None and not cfg.dry_run:
            try:
                capital = float(self.broker.get_balance().get("free", capital))
            except Exception:
                pass
        dollars = capital * (cfg.base_position_size_pct / 100.0) * float(size_mult)
        qty = dollars / max(price, 1e-6)
        # Redondeo a 4 decimales — cripto fraccional; acciones US acepta
        # fraccional en Alpaca.
        return round(qty, 4)

    # ──────────────────────────────────────────
    def _execute_buy(
        self, signal: StrategySignal, size_mult: float, df: pd.DataFrame
    ) -> bool:
        """Abre posición. Devuelve True si se ejecutó (o se habría ejecutado)."""
        cfg = self.config
        price = float(signal.price)
        qty = self._compute_size_qty(price, size_mult)
        if qty <= 0:
            logger.warning("Tamaño calculado <= 0 (size_mult=%.3f) — SKIP BUY", size_mult)
            return False

        # Tope de posiciones simultáneas.
        if len(self._open_positions_cache) >= cfg.max_open_positions:
            logger.info(
                "Ya hay %d posición(es) abierta(s) en %s — SKIP BUY",
                len(self._open_positions_cache), cfg.symbol,
            )
            return False

        if cfg.dry_run or self.broker is None:
            logger.info(
                "[DRY-RUN] BUY %s qty=%.4f @ ~%.2f  mult=%.2f",
                cfg.symbol, qty, price, size_mult,
            )
            return True

        try:
            self.broker.place_market_order(cfg.symbol, "buy", qty)
        except Exception as exc:
            logger.error("Fallo colocando orden BUY: %s", exc)
            return False
        self._notify(f"BUY {cfg.symbol} qty={qty:.4f} @ ~${price:.2f}  mult={size_mult:.2f}")
        return True

    # ──────────────────────────────────────────
    def _execute_sell(self, signal: StrategySignal) -> bool:
        """Cierra la posición entera del símbolo. Devuelve True si se ejecutó."""
        cfg = self.config
        if not self._open_positions_cache:
            return False
        if cfg.dry_run or self.broker is None:
            logger.info("[DRY-RUN] SELL %s (cerrar posición)", cfg.symbol)
            return True
        try:
            self.broker.close_position(cfg.symbol)
        except Exception as exc:
            logger.error("Fallo cerrando posición: %s", exc)
            return False
        self._notify(f"SELL {cfg.symbol} (cierre por señal)")
        return True

    # ──────────────────────────────────────────
    def _check_sl_tp(self, df: pd.DataFrame) -> bool:
        """
        Chequeo manual de SL/TP sobre la última barra. Devuelve True si se
        cerró posición.

        Nota: idealmente esto se deja al broker (bracket orders). Lo hacemos
        aquí para (a) mantener paridad con el backtest, (b) poder usar
        SL/TP dinámicos basados en ATR en futuras iteraciones.
        """
        cfg = self.config
        if not self._open_positions_cache:
            return False
        if cfg.stop_loss_pct <= 0 and cfg.take_profit_pct <= 0:
            return False

        last_close = float(df["close"].iloc[-1])
        for pos in self._open_positions_cache:
            entry = float(pos.get("avg_entry_price", last_close))
            change_pct = (last_close - entry) / max(entry, 1e-6) * 100.0
            hit_sl = cfg.stop_loss_pct > 0 and change_pct <= -cfg.stop_loss_pct
            hit_tp = cfg.take_profit_pct > 0 and change_pct >= cfg.take_profit_pct
            if hit_sl or hit_tp:
                reason = "SL" if hit_sl else "TP"
                logger.info("Cierre por %s (%.2f%%) @ %.2f", reason, change_pct, last_close)
                if cfg.dry_run or self.broker is None:
                    logger.info("[DRY-RUN] CLOSE %s por %s", cfg.symbol, reason)
                else:
                    try:
                        self.broker.close_position(cfg.symbol)
                    except Exception as exc:
                        logger.error("Fallo cerrando por %s: %s", reason, exc)
                        continue
                self._notify(f"{reason} hit en {cfg.symbol} @ ${last_close:.2f}")
                return True
        return False

    # ──────────────────────────────────────────
    def run_once(self) -> LiveDecision:
        """
        Una iteración completa. Aislada para facilitar test y dry-run.
        """
        cfg = self.config
        self._iter_count += 1

        # K: hot-reload del modelo si el orchestrator lo reemplazó.
        # Se hace al principio del iter — así esta iter YA usa el modelo
        # nuevo sin reiniciar el proceso.
        if self._reload_watcher is not None and self._reload_watcher.check():
            self._reload_strategy_inplace()

        # 1. Descargar histórico reciente.
        df = self._fetch_bars()
        if df is not None and not df.empty:
            # Anotar el símbolo en el DataFrame: las features de
            # sentimiento (B) lo leen vía `df.attrs["symbol"]` para
            # decidir qué noticias buscar. No afecta a nada más.
            df.attrs["symbol"] = cfg.symbol
        if df is None or df.empty:
            logger.warning("Sin datos recientes — SKIP.")
            return LiveDecision(
                timestamp=pd.Timestamp.utcnow(), action="SKIP",
                reason="no_data", price=float("nan"),
            )

        last_ts = df.index[-1] if len(df.index) else pd.Timestamp.utcnow()
        last_price = float(df["close"].iloc[-1])

        # 2. Señal sobre el dataframe completo. Usamos `generate_signals`
        # y tomamos la última barra; así en el path live reutilizamos
        # exactamente el mismo código que en el backtest (sin usar
        # `generate_signal`, que puede tener variaciones históricas).
        try:
            actions = self.strategy.generate_signals(df)
            last_action = actions.iloc[-1]
            reason = f"{self.strategy.name} @ {last_ts}"
        except Exception as exc:
            logger.error("Error generando señal: %s", exc)
            return LiveDecision(
                timestamp=last_ts, action="SKIP",
                reason=f"signal_error: {exc}", price=last_price,
            )

        signal = StrategySignal(
            action=last_action, confidence=0.5, price=last_price, reason=reason,
        )

        # 3. Posiciones vivas y (si hay) chequeo de SL/TP primero —
        #    queremos cerrar por stop antes de interpretar una nueva señal.
        self._refresh_positions()
        closed_by_sl_tp = self._check_sl_tp(df)
        if closed_by_sl_tp:
            self._refresh_positions()

        # 4. Sizing: si la señal final es BUY, aplicamos el overlay.
        size_mult, proba = 1.0, None
        if last_action == Action.BUY:
            try:
                size_mult, proba = _size_multiplier_last_bar(
                    self.strategy, df, self.overlay,
                )
            except Exception as exc:
                logger.warning("Overlay falló — sizing base: %s", exc)

        # 5. Ejecutar acción.
        executed = False
        if last_action == Action.BUY:
            if size_mult <= 0:
                logger.info("Overlay pide mult=0 (vol/regime bloquea) — SKIP BUY")
                action_str = "HOLD"
            else:
                executed = self._execute_buy(signal, size_mult, df)
                action_str = "BUY"
        elif last_action == Action.SELL:
            executed = self._execute_sell(signal)
            action_str = "SELL"
        else:
            action_str = "HOLD"

        decision = LiveDecision(
            timestamp=last_ts, action=action_str, reason=reason,
            price=last_price, size_multiplier=size_mult, proba=proba,
            executed=executed,
            extras={"n_open_positions": len(self._open_positions_cache)},
        )

        # K: chequeo de drift cada N iteraciones (no cada barra, es
        # relativamente costoso: KS + PSI sobre decenas de features).
        if (
            self.drift_detector is not None
            and self._iter_count > 0
            and self._iter_count % max(self.config.drift_check_every_iters, 1) == 0
        ):
            try:
                self._run_drift_check(df)
            except Exception as exc:
                logger.warning("Drift check falló: %s", exc)

        # Snapshot para el dashboard (H). Se escribe siempre que se haya
        # configurado `state_path`, sin importar si la iter cerró trade o no.
        if self.config.state_path:
            try:
                self._write_state_snapshot(decision, df)
            except Exception as exc:
                # Una escritura de snapshot fallida no debe romper el bot.
                logger.warning("Snapshot dashboard falló: %s", exc)

        return decision

    # ──────────────────────────────────────────
    # K: Drift check + hot-reload del modelo
    # ──────────────────────────────────────────
    def _run_drift_check(self, df: pd.DataFrame) -> None:
        """
        Compara la distribución de las features recientes contra la de
        referencia (training) y, si hay drift, llama al orchestrator.

        El DataFrame `df` no trae features aún — las calculamos con el
        mismo FeatureBuilder que usa la estrategia (si está disponible).
        Si no, caemos a las columnas OHLCV puras (detecta cambios de
        régimen de precio pero no de indicadores derivados).
        """
        if df is None or df.empty:
            return

        # Intentamos obtener las features ya calculadas. Las estrategias
        # meta_* exponen `inferencer.feature_builder`.
        fb = None
        infer = getattr(self.strategy, "inferencer", None)
        if infer is not None:
            fb = getattr(infer, "feature_builder", None)

        if fb is not None:
            try:
                features = fb.build(df, symbol=self.config.symbol)
            except Exception as exc:
                logger.debug("drift: FeatureBuilder falló (%s) — usando OHLCV.", exc)
                features = df
        else:
            features = df

        # Sólo columnas numéricas.
        features = features.select_dtypes(include=[float, int])
        if features.empty:
            return

        report = self.drift_detector.check(features)
        self._last_drift_report = report.to_dict()
        logger.info(
            "Drift check #%d — KS %.0f%% PSI %.0f%% AUC=%s should_retrain=%s (%s)",
            self._iter_count,
            report.ks_feature_frac * 100,
            report.psi_feature_frac * 100,
            f"{report.auc_rolling:.3f}" if report.auc_rolling is not None else "n/a",
            report.should_retrain,
            report.reason,
        )

        if report.should_retrain and self.retrain_orchestrator is not None:
            started = self.retrain_orchestrator.trigger(reason=report.reason)
            if started:
                self._notify(f"[trading-bot] RETRAIN lanzado — motivo: {report.reason}")

    def _reload_strategy_inplace(self) -> None:
        """
        Recarga el modelo del meta-labeler sin reconstruir la estrategia
        entera. Se invoca cuando el `ModelReloadWatcher` detecta un
        swap del pickle.

        Usa el mismo loader que el cold-start del main.py para garantizar
        que la semántica es idéntica (global vs regime-split, sentiment,
        threshold, feature_cols).
        """
        path = self.config.model_reload_path
        if not path:
            return
        logger.info("Hot-reload: detecto cambio de %s, recargando estrategia.", path)
        try:
            # Import local para evitar ciclos. Reutilizamos el builder
            # del main.py — garantiza que hot-reload y cold-start usan
            # el mismo path de construcción (regime-split, sentiment,
            # threshold, feature_cols).
            from strategies.meta_labeled_ensemble import (
                build_meta_labeled_ensemble_from_file,
            )
            new_strategy = build_meta_labeled_ensemble_from_file(path)
            self.strategy = new_strategy
            # Reiniciamos el contador de AUC bajo floor — damos al nuevo
            # modelo el beneficio de la duda hasta tener histórico propio.
            if self.drift_detector is not None:
                self.drift_detector.reset_auc_counter()
            logger.info("Hot-reload OK — estrategia ahora usa %s.", path)
        except Exception as exc:
            logger.exception(
                "Hot-reload falló (%s). Bot sigue con el modelo anterior.", exc
            )

    # ──────────────────────────────────────────
    def _write_state_snapshot(self, decision: LiveDecision, df: pd.DataFrame) -> None:
        """
        Escribe un snapshot JSON con el estado actual del bot.

        El dashboard (`dashboard/app.py`) lee este archivo y lo refresca
        cada pocos segundos. El formato es estable y backward-compatible:
        se añaden campos nuevos al final, nunca se renombran ni se
        eliminan los existentes.

        Diseño: se escribe a un fichero temporal y se hace `os.replace`
        atómico, para evitar que el dashboard lea un JSON parcial si
        coincide con el momento de escritura.
        """
        import json
        import os
        from pathlib import Path

        cfg = self.config
        # Anexar la decisión actual al histórico circular.
        hist_entry = {
            "iter": self._iter_count,
            "timestamp": decision.timestamp.isoformat() if hasattr(decision.timestamp, "isoformat") else str(decision.timestamp),
            "action": decision.action,
            "price": decision.price,
            "size_multiplier": decision.size_multiplier,
            "proba": decision.proba,
            "executed": decision.executed,
            "reason": decision.reason,
        }
        self._decision_history.append(hist_entry)
        if len(self._decision_history) > cfg.state_history_size:
            # Mantener sólo los últimos N para que el JSON no crezca sin
            # límite en runs largos.
            self._decision_history = self._decision_history[-cfg.state_history_size:]

        # Información del modelo (si la estrategia es meta_ensemble).
        model_info: Dict[str, Any] = {}
        infer = getattr(self.strategy, "inferencer", None)
        if infer is not None:
            model_info["threshold"] = getattr(self.strategy, "threshold", None)
            # `_is_split` es True para regime-split; sirve para que el
            # dashboard muestre etiqueta "split" vs "global".
            model_info["regime_split"] = getattr(self.strategy, "_is_split", False)

        # Sentiment "spot" (último valor de F&G y último news_sent_24h_mean
        # si están disponibles en el DataFrame).
        sent_info: Dict[str, Any] = {}
        if df is not None and not df.empty:
            for col in ("fng_value", "fng_class_idx", "news_sent_24h_mean", "news_sent_24h_count"):
                if col in df.columns:
                    val = df[col].dropna()
                    if len(val) > 0:
                        sent_info[col] = float(val.iloc[-1])

        snapshot = {
            "schema_version": 1,
            "updated_at": pd.Timestamp.utcnow().isoformat(),
            "symbol": cfg.symbol,
            "timeframe": cfg.timeframe,
            "strategy": self.strategy.name,
            "dry_run": cfg.dry_run,
            "data_source": cfg.data_source,
            "iter_count": self._iter_count,
            "max_iters": cfg.max_iters,
            # Snapshot de la decisión más reciente.
            "last_decision": hist_entry,
            # Posiciones abiertas conocidas por el cache (no son la
            # source of truth — el broker lo es — pero suelen coincidir).
            "open_positions": [
                {k: v for k, v in p.items() if k in {"symbol", "qty", "avg_entry_price", "side", "unrealized_pl"}}
                for p in self._open_positions_cache
            ],
            "model": model_info,
            "sentiment_spot": sent_info,
            "history": self._decision_history,
            # K: snapshot del último chequeo de drift para el dashboard.
            # Puede ser None si el monitor no está configurado o aún
            # no ha corrido por primera vez.
            "drift": self._last_drift_report,
            # K: si el orchestrator está enchufado, exponemos su estado
            # persistente (retrain_count, last_retrain_at, etc.) para
            # que el dashboard muestre cuándo fue el último retrain.
            "retrain_state": (
                dict(self.retrain_orchestrator.state.data)
                if self.retrain_orchestrator is not None else None
            ),
        }

        path = Path(cfg.state_path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(snapshot, f, indent=2, default=str)
        os.replace(tmp, path)

    # ──────────────────────────────────────────
    def run(self) -> None:
        """
        Bucle infinito (o hasta `max_iters`). Ctrl+C lo para limpiamente.
        """
        cfg = self.config
        sleep_s = cfg.sleep_seconds or _TF_SLEEP.get(cfg.timeframe, 3600)
        logger.info(
            "LIVE START — symbol=%s tf=%s sleep=%ds dry_run=%s data=%s",
            cfg.symbol, cfg.timeframe, sleep_s, cfg.dry_run, cfg.data_source,
        )

        try:
            while True:
                if cfg.max_iters is not None and self._iter_count >= cfg.max_iters:
                    logger.info("max_iters alcanzado (%d) — FIN.", cfg.max_iters)
                    break
                decision = self.run_once()
                logger.info(
                    "iter=%d  action=%s  price=%.2f  mult=%.2f  proba=%s  exec=%s",
                    self._iter_count, decision.action, decision.price,
                    decision.size_multiplier,
                    f"{decision.proba:.3f}" if decision.proba is not None else "—",
                    decision.executed,
                )
                time.sleep(sleep_s)
        except KeyboardInterrupt:
            logger.info("Bot detenido por el usuario (Ctrl+C).")
            self._notify("Trading bot detenido por el usuario.")
