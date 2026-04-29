"""
portfolio_live_runner.py — Multi-symbol live/paper trading runner.

Extiende la lógica del `LiveRunner` single-symbol para operar N símbolos
en un único proceso, con:

  - **Capital compartido**: un solo pool de cash (broker account).
  - **Max concurrent positions**: cap global configurable.
  - **Correlation guard**: rechaza entradas cuya correlación rolling
    con las posiciones abiertas supera un umbral (reusa la lógica del
    `PortfolioBacktestEngine`).
  - **Snapshot unificado**: un solo JSON con estado de todos los símbolos
    para el dashboard.

Diseño:
    En cada iteración (`run_once`), el runner:
      1. Descarga barras para TODOS los símbolos.
      2. Genera señales por símbolo.
      3. Procesa SELLs primero (libera slots y capital).
      4. Procesa BUYs con filtros de cap + correlación.
      5. Chequea SL/TP por símbolo.
      6. Escribe snapshot unificado.

    Esto es funcionalmente equivalente al portfolio backtest pero en
    tiempo real, con la misma paridad señal live↔backtest que el runner
    single-symbol.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from execution.live_runner import (
    LiveConfig,
    LiveDecision,
    _fetch_bars_alpaca,
    _fetch_bars_yahoo,
    _size_multiplier_last_bar,
    _TF_SLEEP,
)
from strategies.base import BaseStrategy, StrategySignal, Action
from utils.logger import get_logger

logger = get_logger(__name__)


# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────
@dataclass
class PortfolioLiveConfig:
    """Parámetros del runner multi-símbolo."""

    symbols: List[str] = field(default_factory=lambda: ["AAPL"])
    timeframe: str = "1Day"
    history_bars: int = 300

    # ── Portfolio limits ──
    max_positions: int = 4               # Cap global de posiciones simultáneas.
    position_size_pct: float = 20.0      # % del equity total por entrada.
    corr_threshold: float = 0.75         # Correlación máxima media con posiciones abiertas.
    corr_window: int = 60                # Ventana rolling para correlación (en barras).
    corr_min_obs: int = 30               # Mín observaciones antes de aplicar guard.

    # ── Execution ──
    sleep_seconds: Optional[int] = None
    dry_run: bool = False
    stop_loss_pct: float = 0.0
    take_profit_pct: float = 0.0
    max_iters: Optional[int] = None
    data_source: str = "alpaca"
    yahoo_period: str = "6mo"
    yahoo_interval: str = "1h"
    alpaca_feed: str = "iex"
    send_telegram: bool = True
    state_path: Optional[str] = None
    state_history_size: int = 200
    only_market_hours: bool = False

    # ── K: drift (delegamos al LiveRunner per-symbol si es necesario) ──
    drift_check_every_iters: int = 20
    model_reload_path: Optional[str] = None


# ──────────────────────────────────────────────
# Runner
# ──────────────────────────────────────────────
class PortfolioLiveRunner:
    """
    Motor live multi-símbolo. Itera sobre N tickers por ciclo con
    un pool de capital compartido y correlation guard.
    """

    def __init__(
        self,
        strategy: BaseStrategy,
        config: Optional[PortfolioLiveConfig] = None,
        broker=None,
        overlay=None,
        telegram_fn: Optional[Callable[[str], None]] = None,
    ):
        self.strategy = strategy
        self.config = config or PortfolioLiveConfig()
        self.broker = broker
        self.overlay = overlay
        self.telegram_fn = telegram_fn

        self._iter_count: int = 0
        # Cache de posiciones abiertas del broker (todas, no filtradas).
        self._all_positions: List[Dict] = []
        # Histórico de decisiones para el snapshot del dashboard.
        self._decision_history: List[Dict[str, Any]] = []
        # Rolling close prices por símbolo para correlation guard.
        # Dict[symbol] → deque de (timestamp, close_price).
        self._close_history: Dict[str, deque] = {
            sym: deque(maxlen=self.config.corr_window + 10)
            for sym in self.config.symbols
        }

    # ──────────────────────────────────────────
    def _notify(self, msg: str) -> None:
        if self.telegram_fn and self.config.send_telegram:
            try:
                self.telegram_fn(msg)
            except Exception as exc:
                logger.warning("Telegram notify falló: %s", exc)

    # ──────────────────────────────────────────
    def _fetch_bars_for_symbol(self, symbol: str) -> pd.DataFrame:
        """Descarga barras para un símbolo concreto."""
        cfg = self.config
        if cfg.data_source == "alpaca":
            if self.broker is None or getattr(self.broker, "api", None) is None:
                raise RuntimeError("data_source=alpaca requiere un Broker inicializado.")
            return _fetch_bars_alpaca(
                self.broker.api, symbol, cfg.timeframe, cfg.history_bars,
                feed=cfg.alpaca_feed,
            )
        if cfg.data_source == "yahoo":
            return _fetch_bars_yahoo(
                symbol, cfg.yahoo_period, cfg.yahoo_interval, cfg.history_bars
            )
        raise ValueError(f"data_source desconocido: {cfg.data_source}")

    # ──────────────────────────────────────────
    def _refresh_positions(self) -> List[Dict]:
        """Todas las posiciones abiertas del broker."""
        if self.broker is None:
            self._all_positions = []
            return []
        try:
            self._all_positions = self.broker.get_positions()
        except Exception as exc:
            logger.warning("No pude consultar posiciones: %s", exc)
            self._all_positions = []
        return self._all_positions

    def _positions_for(self, symbol: str) -> List[Dict]:
        """Posiciones abiertas de un símbolo específico."""
        return [p for p in self._all_positions if p.get("symbol") == symbol]

    def _open_symbols(self) -> List[str]:
        """Símbolos con posición abierta."""
        return list({p.get("symbol") for p in self._all_positions})

    def _n_open(self) -> int:
        """Número de símbolos con posición abierta."""
        return len(self._open_symbols())

    # ──────────────────────────────────────────
    def _get_capital(self) -> float:
        """Capital disponible (free cash) del broker."""
        if self.broker is None or self.config.dry_run:
            return 10_000.0
        try:
            return float(self.broker.get_balance().get("free", 10_000.0))
        except Exception:
            return 10_000.0

    def _get_equity(self) -> float:
        """Equity total (portfolio value) del broker."""
        if self.broker is None or self.config.dry_run:
            return 10_000.0
        try:
            return float(self.broker.get_balance().get("total", 10_000.0))
        except Exception:
            return 10_000.0

    # ──────────────────────────────────────────
    def _compute_qty(self, price: float, size_mult: float) -> float:
        """Calcula cantidad de shares basándose en equity total y position_size_pct."""
        equity = self._get_equity()
        dollars = equity * (self.config.position_size_pct / 100.0) * float(size_mult)
        # Cap por cash disponible (no apalancamos).
        cash = self._get_capital()
        dollars = min(dollars, cash)
        qty = dollars / max(price, 1e-6)
        return round(qty, 4)

    # ──────────────────────────────────────────
    def _check_correlation(self, candidate: str) -> bool:
        """
        Correlation guard: devuelve True si el candidato PASA el filtro
        (baja correlación con posiciones abiertas → se puede abrir).

        Reusa la misma lógica que portfolio_engine.py: correlación media
        en valor absoluto de log-returns rolling.
        """
        cfg = self.config
        if cfg.corr_threshold is None or cfg.corr_threshold > 1.0:
            return True  # Guard desactivado.

        open_syms = self._open_symbols()
        if not open_syms:
            return True  # Sin posiciones abiertas, cualquiera pasa.

        # Necesitamos suficientes observaciones.
        needed = [candidate] + open_syms
        min_len = cfg.corr_min_obs
        for sym in needed:
            if len(self._close_history.get(sym, [])) < min_len:
                return True  # Sin datos suficientes, dejamos pasar.

        try:
            # Construir panel de closes.
            panels = {}
            for sym in needed:
                hist = list(self._close_history[sym])
                ts = [h[0] for h in hist]
                px = [h[1] for h in hist]
                panels[sym] = pd.Series(px, index=ts, name=sym)

            close_panel = pd.DataFrame(panels)
            # Log-returns.
            log_ret = np.log(close_panel / close_panel.shift(1)).dropna()

            if len(log_ret) < min_len:
                return True

            # Últimas corr_window observaciones.
            window = log_ret.tail(cfg.corr_window)
            if len(window) < min_len:
                return True

            corr_matrix = window.corr()
            corr_with_open = corr_matrix.loc[candidate, open_syms]
            avg_corr = corr_with_open.abs().mean()

            if avg_corr > cfg.corr_threshold:
                logger.info(
                    "CORR GUARD: %s rechazado — corr media %.2f > umbral %.2f",
                    candidate, avg_corr, cfg.corr_threshold,
                )
                return False
            return True
        except Exception as exc:
            logger.debug("Correlation guard falló (%s) — dejamos pasar.", exc)
            return True

    # ──────────────────────────────────────────
    def _execute_buy(self, symbol: str, price: float, size_mult: float) -> bool:
        """Abre posición en un símbolo."""
        cfg = self.config
        qty = self._compute_qty(price, size_mult)
        if qty <= 0:
            logger.warning("[%s] Tamaño <= 0 (mult=%.3f) — SKIP BUY", symbol, size_mult)
            return False

        if cfg.dry_run or self.broker is None:
            logger.info("[DRY-RUN] BUY %s qty=%.4f @ ~%.2f mult=%.2f", symbol, qty, price, size_mult)
            return True

        try:
            self.broker.place_market_order(symbol, "buy", qty)
        except Exception as exc:
            logger.error("[%s] Fallo BUY: %s", symbol, exc)
            return False
        self._notify(f"🟢 BUY {symbol} qty={qty:.4f} @ ~${price:.2f}  mult={size_mult:.2f}")
        return True

    # ──────────────────────────────────────────
    def _execute_sell(self, symbol: str) -> bool:
        """Cierra la posición de un símbolo."""
        cfg = self.config
        if cfg.dry_run or self.broker is None:
            logger.info("[DRY-RUN] SELL %s (cerrar posición)", symbol)
            return True
        try:
            self.broker.close_position(symbol)
        except Exception as exc:
            logger.error("[%s] Fallo SELL: %s", symbol, exc)
            return False
        self._notify(f"🔴 SELL {symbol} (cierre por señal)")
        return True

    # ──────────────────────────────────────────
    def _check_sl_tp(self, symbol: str, df: pd.DataFrame) -> bool:
        """Chequeo SL/TP para un símbolo. Devuelve True si se cerró."""
        cfg = self.config
        positions = self._positions_for(symbol)
        if not positions:
            return False
        if cfg.stop_loss_pct <= 0 and cfg.take_profit_pct <= 0:
            return False

        last_close = float(df["close"].iloc[-1])
        for pos in positions:
            entry = float(pos.get("avg_entry_price", last_close))
            change_pct = (last_close - entry) / max(entry, 1e-6) * 100.0
            hit_sl = cfg.stop_loss_pct > 0 and change_pct <= -cfg.stop_loss_pct
            hit_tp = cfg.take_profit_pct > 0 and change_pct >= cfg.take_profit_pct
            if hit_sl or hit_tp:
                reason = "SL" if hit_sl else "TP"
                logger.info("[%s] Cierre por %s (%.2f%%) @ %.2f", symbol, reason, change_pct, last_close)
                if cfg.dry_run or self.broker is None:
                    logger.info("[DRY-RUN] CLOSE %s por %s", symbol, reason)
                else:
                    try:
                        self.broker.close_position(symbol)
                    except Exception as exc:
                        logger.error("[%s] Fallo cerrando por %s: %s", symbol, reason, exc)
                        continue
                self._notify(f"⚠️ {reason} hit en {symbol} @ ${last_close:.2f} ({change_pct:+.2f}%)")
                return True
        return False

    # ──────────────────────────────────────────
    def run_once(self) -> List[LiveDecision]:
        """
        Una iteración completa sobre TODOS los símbolos.

        Orden: SELLs primero → luego BUYs (con filtros de cap + corr).
        Devuelve lista de decisiones (una por símbolo).
        """
        cfg = self.config
        self._iter_count += 1
        decisions: List[LiveDecision] = []

        # 1. Descargar barras y generar señales para todos los símbolos.
        bars_by_sym: Dict[str, pd.DataFrame] = {}
        signals_by_sym: Dict[str, Action] = {}
        prices_by_sym: Dict[str, float] = {}

        for symbol in cfg.symbols:
            try:
                df = self._fetch_bars_for_symbol(symbol)
                if df is not None and not df.empty:
                    df.attrs["symbol"] = symbol
            except Exception as exc:
                logger.warning("[%s] Error descargando datos: %s", symbol, exc)
                df = None

            if df is None or df.empty:
                logger.warning("[%s] Sin datos — SKIP.", symbol)
                decisions.append(LiveDecision(
                    timestamp=pd.Timestamp.utcnow(), action="SKIP",
                    reason="no_data", price=float("nan"),
                ))
                continue

            bars_by_sym[symbol] = df
            last_price = float(df["close"].iloc[-1])
            prices_by_sym[symbol] = last_price

            # Actualizar historial de closes para correlation guard.
            last_ts = df.index[-1] if len(df.index) else pd.Timestamp.utcnow()
            self._close_history[symbol].append((last_ts, last_price))

            try:
                actions = self.strategy.generate_signals(df)
                last_action = actions.iloc[-1]
                signals_by_sym[symbol] = last_action
            except Exception as exc:
                logger.error("[%s] Error generando señal: %s", symbol, exc)
                decisions.append(LiveDecision(
                    timestamp=last_ts, action="SKIP",
                    reason=f"signal_error: {exc}", price=last_price,
                ))

        # 2. Refrescar posiciones del broker.
        self._refresh_positions()

        # 3. Procesar SELLs primero (libera slots y capital).
        for symbol in cfg.symbols:
            if symbol not in signals_by_sym or symbol not in bars_by_sym:
                continue
            df = bars_by_sym[symbol]

            # SL/TP primero.
            closed = self._check_sl_tp(symbol, df)
            if closed:
                self._refresh_positions()

            action = signals_by_sym[symbol]
            last_ts = df.index[-1] if len(df.index) else pd.Timestamp.utcnow()
            price = prices_by_sym[symbol]

            if action == Action.SELL:
                has_position = bool(self._positions_for(symbol))
                executed = False
                if has_position:
                    executed = self._execute_sell(symbol)
                    if executed:
                        self._refresh_positions()
                decisions.append(LiveDecision(
                    timestamp=last_ts, action="SELL", price=price,
                    reason=f"{self.strategy.name} @ {last_ts}",
                    executed=executed,
                ))

        # 4. Procesar BUYs con filtros globales.
        buy_candidates = []
        for symbol in sorted(cfg.symbols):  # Orden estable (lexicográfico).
            if symbol not in signals_by_sym or symbol not in bars_by_sym:
                continue
            action = signals_by_sym[symbol]
            if action != Action.BUY:
                continue
            # Ya tenemos posición abierta en este símbolo?
            if self._positions_for(symbol):
                continue
            buy_candidates.append(symbol)

        for symbol in buy_candidates:
            df = bars_by_sym[symbol]
            last_ts = df.index[-1] if len(df.index) else pd.Timestamp.utcnow()
            price = prices_by_sym[symbol]

            # ── Cap de posiciones ──
            if self._n_open() >= cfg.max_positions:
                logger.info(
                    "[%s] Cap posiciones (%d/%d) — SKIP BUY",
                    symbol, self._n_open(), cfg.max_positions,
                )
                decisions.append(LiveDecision(
                    timestamp=last_ts, action="HOLD", price=price,
                    reason=f"cap_positions ({self._n_open()}/{cfg.max_positions})",
                ))
                continue

            # ── Correlation guard ──
            if not self._check_correlation(symbol):
                decisions.append(LiveDecision(
                    timestamp=last_ts, action="HOLD", price=price,
                    reason=f"corr_guard (threshold={cfg.corr_threshold})",
                ))
                continue

            # ── Sizing (risk overlay si está activo) ──
            size_mult, proba = 1.0, None
            try:
                size_mult, proba = _size_multiplier_last_bar(
                    self.strategy, df, self.overlay,
                )
            except Exception as exc:
                logger.warning("[%s] Overlay falló — sizing base: %s", symbol, exc)

            if size_mult <= 0:
                logger.info("[%s] Overlay pide mult=0 — SKIP BUY", symbol)
                decisions.append(LiveDecision(
                    timestamp=last_ts, action="HOLD", price=price,
                    reason="overlay_blocked",
                    size_multiplier=0.0, proba=proba,
                ))
                continue

            # ── Ejecutar ──
            executed = self._execute_buy(symbol, price, size_mult)
            if executed:
                self._refresh_positions()

            decisions.append(LiveDecision(
                timestamp=last_ts, action="BUY", price=price,
                reason=f"{self.strategy.name} @ {last_ts}",
                size_multiplier=size_mult, proba=proba,
                executed=executed,
            ))

        # 5. HOLDs — símbolos sin señal BUY ni SELL.
        for symbol in cfg.symbols:
            if symbol not in signals_by_sym or symbol not in bars_by_sym:
                continue
            action = signals_by_sym[symbol]
            if action not in (Action.BUY, Action.SELL):
                df = bars_by_sym[symbol]
                last_ts = df.index[-1] if len(df.index) else pd.Timestamp.utcnow()
                decisions.append(LiveDecision(
                    timestamp=last_ts, action="HOLD", price=prices_by_sym[symbol],
                    reason=f"{self.strategy.name} @ {last_ts}",
                ))

        # 6. Snapshot para dashboard.
        if cfg.state_path:
            try:
                self._write_state_snapshot(decisions, bars_by_sym)
            except Exception as exc:
                logger.warning("Snapshot dashboard falló: %s", exc)

        return decisions

    # ──────────────────────────────────────────
    def _write_state_snapshot(
        self, decisions: List[LiveDecision], bars_by_sym: Dict[str, pd.DataFrame]
    ) -> None:
        """Escribe snapshot JSON unificado para el dashboard."""
        import json
        import os
        from pathlib import Path

        cfg = self.config

        # Construir entradas del histórico.
        iter_entries = []
        for d in decisions:
            entry = {
                "iter": self._iter_count,
                "timestamp": d.timestamp.isoformat() if hasattr(d.timestamp, "isoformat") else str(d.timestamp),
                "action": d.action,
                "price": d.price,
                "size_multiplier": d.size_multiplier,
                "proba": d.proba,
                "executed": d.executed,
                "reason": d.reason,
            }
            iter_entries.append(entry)
            self._decision_history.append(entry)

        # Limitar histórico.
        if len(self._decision_history) > cfg.state_history_size:
            self._decision_history = self._decision_history[-cfg.state_history_size:]

        snapshot = {
            "schema_version": 2,
            "updated_at": pd.Timestamp.utcnow().isoformat(),
            "mode": "portfolio_live",
            "symbols": cfg.symbols,
            "timeframe": cfg.timeframe,
            "strategy": self.strategy.name,
            "dry_run": cfg.dry_run,
            "data_source": cfg.data_source,
            "iter_count": self._iter_count,
            "max_iters": cfg.max_iters,
            "max_positions": cfg.max_positions,
            "corr_threshold": cfg.corr_threshold,
            "n_open_positions": self._n_open(),
            "open_symbols": self._open_symbols(),
            "last_decisions": iter_entries,
            "open_positions": [
                {k: v for k, v in p.items()
                 if k in {"symbol", "qty", "avg_entry_price", "side", "unrealized_pl"}}
                for p in self._all_positions
                if p.get("symbol") in cfg.symbols
            ],
            "history": self._decision_history,
        }

        path = Path(cfg.state_path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(snapshot, f, indent=2, default=str)
        os.replace(tmp, path)

    # ──────────────────────────────────────────
    def run(self) -> None:
        """Bucle infinito (o hasta max_iters). Ctrl+C lo para limpiamente."""
        cfg = self.config
        sleep_s = cfg.sleep_seconds or _TF_SLEEP.get(cfg.timeframe, 3600)
        logger.info(
            "PORTFOLIO LIVE START — symbols=%s tf=%s sleep=%ds dry_run=%s "
            "max_positions=%d corr≤%.2f data=%s",
            cfg.symbols, cfg.timeframe, sleep_s, cfg.dry_run,
            cfg.max_positions, cfg.corr_threshold, cfg.data_source,
        )

        self._notify(
            f"🤖 <b>Portfolio Trading Bot iniciado</b>\n"
            f"Symbols: {', '.join(cfg.symbols)} ({len(cfg.symbols)})\n"
            f"TF: {cfg.timeframe}  Max pos: {cfg.max_positions}\n"
            f"Strategy: {self.strategy.name}"
        )

        try:
            while True:
                if cfg.max_iters is not None and self._iter_count >= cfg.max_iters:
                    logger.info("max_iters alcanzado (%d) — FIN.", cfg.max_iters)
                    break
                decisions = self.run_once()
                # Log resumen de la iteración.
                actions_summary = {d.action: 0 for d in decisions}
                for d in decisions:
                    actions_summary[d.action] = actions_summary.get(d.action, 0) + 1
                logger.info(
                    "iter=%d  open=%d/%d  %s",
                    self._iter_count, self._n_open(), cfg.max_positions,
                    "  ".join(f"{k}={v}" for k, v in sorted(actions_summary.items())),
                )
                time.sleep(sleep_s)
        except KeyboardInterrupt:
            logger.info("Portfolio bot detenido por el usuario (Ctrl+C).")
            self._notify("🛑 Portfolio Trading Bot detenido por el usuario.")
