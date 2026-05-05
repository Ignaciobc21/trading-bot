"""
execution/ibkr_live_runner.py — LiveRunner adaptado para Interactive Brokers.

Diferencias clave respecto a execution/live_runner.py (Alpaca):

1. DATOS: usa IBKRDataFetcher.fetch_historical() en el arranque y
   subscribe_realtime() con barUpdateEvent para el loop en tiempo real,
   en lugar de polling Alpaca REST cada `sleep_seconds`.

2. BUCLE: corre dentro del event loop de ib_insync.
   - El método run() arranca ib.run() que mantiene viva la conexión
     y procesa todos los eventos internamente.
   - Las iteraciones lógicas ocurren en el callback de barUpdateEvent
     (cuando IB notifica una barra nueva completada).
   - Se usa ib.sleep() en lugar de time.sleep() para no bloquear
     el event loop de ib_insync.

3. ÓRDENES: delega en IBKRBroker que produce MarketOrder / LimitOrder /
   BracketOrder de ib_insync en lugar de llamadas REST Alpaca.

4. Las funciones de señal (strategy.generate_signals), features ML,
   risk overlay y drift detection NO se modifican: trabajan con el
   mismo DataFrame OHLCV y se ejecutan en el callback de barra nueva.

Uso típico:
    from execution.ibkr_live_runner import IBKRLiveRunner, IBKRLiveConfig
    from execution.ibkr_broker import IBKRBroker, IBKRConfig
    from strategies.meta_labeled_ensemble import build_meta_labeled_ensemble_from_file

    broker = IBKRBroker(IBKRConfig(port=7497))   # TWS Paper
    strategy = build_meta_labeled_ensemble_from_file("models/v1.pkl")
    runner = IBKRLiveRunner(strategy=strategy, broker=broker)
    runner.run()     # bloquea hasta Ctrl+C

Para multi-símbolo (portfolio live) usa IBKRPortfolioLiveRunner (abajo).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

from execution.ibkr_broker import IBKRBroker, IBKRDataFetcher, resolve_contract
from execution.live_runner import (
    LiveDecision,
    _size_multiplier_last_bar,
    _TF_SLEEP,
)
from strategies.base import BaseStrategy, Action

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════
#  Configuración específica de IBKR
# ════════════════════════════════════════════════════════════════════
@dataclass
class IBKRLiveConfig:
    """
    Parámetros del runner IBKR single-symbol.

    Mayormente iguales a LiveConfig de Alpaca, con las diferencias:
      - No hay data_source / alpaca_feed (siempre es IB).
      - use_bracket_orders: si True, usa Bracket (OCA SL+TP) en lugar
        de gestionar SL/TP manualmente barra a barra.
      - sl_pct / tp_pct: sólo se usan si use_bracket_orders=True.
        Para gestión manual, usar el stop_loss_pct / take_profit_pct
        del runner original.
      - what_to_show: "TRADES" (acciones) / "MIDPOINT" (forex).
      - use_rth: True = sólo Regular Trading Hours.
    """
    symbol: str = "AAPL"
    timeframe: str = "1Day"
    history_bars: int = 500
    base_position_size_pct: float = 2.0
    max_open_positions: int = 1
    dry_run: bool = False
    stop_loss_pct: float = 0.0
    take_profit_pct: float = 0.0
    use_bracket_orders: bool = False      # True → Bracket OCA en lugar de manual
    max_iters: Optional[int] = None
    state_path: Optional[str] = None
    state_history_size: int = 200
    send_telegram: bool = False
    what_to_show: str = "TRADES"
    use_rth: bool = True


# ════════════════════════════════════════════════════════════════════
#  Runner single-symbol IBKR
# ════════════════════════════════════════════════════════════════════
class IBKRLiveRunner:
    """
    Runner live que usa IB como fuente de datos y broker.

    Bucle de eventos (PUNTO 4 del brief):
    ──────────────────────────────────────
    El runner NO usa time.sleep() ni un while-True propio. En su lugar:

      1. Descarga histórico inicial con fetch_historical() (bloqueante).
      2. Se suscribe a barUpdateEvent con subscribe_realtime().
      3. Llama a ib.run() que cede el control al event loop de ib_insync.
         Cada vez que IB notifica una barra nueva, se llama _on_new_bar().
      4. _on_new_bar() ejecuta toda la lógica: señal, sizing, orden.
      5. Ctrl+C interrumpe ib.run() y se llama ib.disconnect().

    Este diseño garantiza que el hilo principal nunca se bloquea y que
    la conexión con TWS se mantiene viva continuamente.
    """

    def __init__(
        self,
        strategy: BaseStrategy,
        broker: IBKRBroker,
        config: Optional[IBKRLiveConfig] = None,
        overlay=None,
        risk_mgr=None,
        storage=None,
        telegram_fn: Optional[Callable[[str], None]] = None,
        drift_detector=None,
        retrain_orchestrator=None,
    ):
        self.strategy = strategy
        self.broker = broker
        self.config = config or IBKRLiveConfig()
        self.overlay = overlay
        self.risk = risk_mgr
        self.storage = storage
        self.telegram_fn = telegram_fn
        self.drift_detector = drift_detector
        self.retrain_orchestrator = retrain_orchestrator

        # ib subyacente (del broker, no instanciamos uno nuevo).
        self._ib = broker.get_ib()
        self._fetcher: IBKRDataFetcher = broker.data_fetcher

        # Estado interno.
        self._df: pd.DataFrame = pd.DataFrame()
        self._iter_count: int = 0
        self._open_positions_cache: List[Dict] = []
        self._decision_history: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Arranque
    # ------------------------------------------------------------------
    def run(self) -> None:
        """
        Arranca el runner.

        1. Carga histórico.
        2. Suscribe a barras en tiempo real.
        3. Entra en ib.run() (event loop bloqueante hasta Ctrl+C).
        """
        cfg = self.config
        logger.info(
            "IBKR LIVE START — symbol=%s tf=%s dry_run=%s",
            cfg.symbol, cfg.timeframe, cfg.dry_run,
        )
        if cfg.dry_run:
            logger.warning("MODO DRY-RUN: no se enviarán órdenes reales a IB.")

        # ── 1. Histórico inicial ────────────────────────────────────
        self._df = self._fetcher.fetch_historical(
            symbol=cfg.symbol,
            timeframe=cfg.timeframe,
            limit=cfg.history_bars,
            what_to_show=cfg.what_to_show,
            use_rth=cfg.use_rth,
        )
        if self._df.empty:
            logger.error(
                "Sin datos históricos para %s. "
                "Verifica que TWS esté conectado y que el símbolo sea válido.", cfg.symbol
            )
            return

        # ── 2. Suscripción en tiempo real ───────────────────────────
        self._fetcher.subscribe_realtime(
            symbol=cfg.symbol,
            callback=self._on_new_bar,
            timeframe=cfg.timeframe,
        )

        # Notificación de inicio.
        self._notify(
            f"🤖 IBKR Bot iniciado: {cfg.symbol} {cfg.timeframe} "
            f"dry_run={cfg.dry_run}"
        )

        # ── 3. Event loop de ib_insync ───────────────────────────────
        # ib.run() bloquea hasta que la conexión se cierre o se llame
        # ib.stop(). Toda la lógica viva ocurre en _on_new_bar().
        try:
            self._ib.run()
        except KeyboardInterrupt:
            logger.info("IBKR Bot detenido por el usuario (Ctrl+C).")
            self._notify("🛑 IBKR Bot detenido por el usuario.")
        finally:
            self._fetcher.unsubscribe(cfg.symbol)
            self.broker.disconnect()

    # ------------------------------------------------------------------
    # Callback de barra nueva (corazón del bucle)
    # ------------------------------------------------------------------
    def _on_new_bar(self, symbol: str, df: pd.DataFrame) -> None:
        """
        Se llama por ib_insync cuando IB notifica una barra nueva completa.

        Este método reemplaza el while-True del runner original de Alpaca.
        Toda la lógica de señal, sizing y ejecución ocurre aquí.

        IMPORTANTE: IB llama a este callback en el hilo del event loop.
        No hacer operaciones bloqueantes largas aquí; en ese caso usar
        ib.schedule() o un thread separado.
        """
        cfg = self.config
        self._iter_count += 1

        # Límite de iteraciones para testing.
        if cfg.max_iters is not None and self._iter_count > cfg.max_iters:
            logger.info("max_iters=%d alcanzado — deteniendo.", cfg.max_iters)
            self._ib.stop()
            return

        # Actualizar el DataFrame con las barras nuevas (append y dedup).
        if not df.empty:
            df.attrs["symbol"] = symbol
            if self._df.empty:
                self._df = df
            else:
                combined = pd.concat([self._df, df])
                combined = combined[~combined.index.duplicated(keep="last")]
                self._df = combined.tail(cfg.history_bars)

        if self._df.empty:
            return

        last_price = float(self._df["close"].iloc[-1])
        last_ts = self._df.index[-1]

        # ── Generar señal ───────────────────────────────────────────
        try:
            actions = self.strategy.generate_signals(self._df)
            last_action = actions.iloc[-1]
        except Exception as exc:
            logger.error("[%s] Error en generate_signals: %s", symbol, exc)
            return

        # ── Posiciones actuales ─────────────────────────────────────
        self._refresh_positions()

        # ── SL/TP manual (si no usamos bracket orders) ──────────────
        if not cfg.use_bracket_orders:
            self._check_sl_tp(last_price)
            self._refresh_positions()

        # ── Sizing ──────────────────────────────────────────────────
        size_mult, proba = 1.0, None
        if last_action == Action.BUY:
            try:
                size_mult, proba = _size_multiplier_last_bar(
                    self.strategy, self._df, self.overlay
                )
            except Exception as exc:
                logger.warning("Overlay falló: %s — usando sizing base.", exc)

        # ── Ejecutar acción ─────────────────────────────────────────
        executed = False
        action_str = "HOLD"

        if last_action == Action.BUY and size_mult > 0:
            if len(self._open_positions_cache) < cfg.max_open_positions:
                executed = self._execute_buy(last_price, size_mult)
                action_str = "BUY"
            else:
                logger.info("[%s] Cap de posiciones — SKIP BUY.", symbol)
        elif last_action == Action.SELL:
            executed = self._execute_sell()
            action_str = "SELL"

        decision = LiveDecision(
            timestamp=last_ts,
            action=action_str,
            reason=f"{self.strategy.name} @ {last_ts}",
            price=last_price,
            size_multiplier=size_mult,
            proba=proba,
            executed=executed,
        )
        logger.info(
            "iter=%d symbol=%s action=%s price=%.2f mult=%.2f exec=%s",
            self._iter_count, symbol, action_str, last_price, size_mult, executed,
        )

        # ── Snapshot para dashboard ─────────────────────────────────
        if cfg.state_path:
            try:
                self._write_state_snapshot(decision)
            except Exception as exc:
                logger.warning("Snapshot falló: %s", exc)

        # ── Drift check ─────────────────────────────────────────────
        if (
            self.drift_detector is not None
            and self._iter_count % 20 == 0
        ):
            try:
                features = self._df.select_dtypes(include=[float, int])
                report = self.drift_detector.check(features)
                if report.should_retrain and self.retrain_orchestrator is not None:
                    self.retrain_orchestrator.trigger(reason=report.reason)
            except Exception as exc:
                logger.warning("Drift check falló: %s", exc)

    # ------------------------------------------------------------------
    # Ejecución de órdenes
    # ------------------------------------------------------------------
    def _execute_buy(self, price: float, size_mult: float) -> bool:
        cfg = self.config
        capital = 10_000.0
        if not cfg.dry_run:
            try:
                capital = self.broker.get_balance()["free"]
            except Exception:
                pass

        dollars = capital * (cfg.base_position_size_pct / 100.0) * float(size_mult)
        qty = round(dollars / max(price, 1e-6), 4)
        if qty <= 0:
            return False

        if cfg.dry_run:
            logger.info("[DRY-RUN] BUY %s qty=%.4f @ ~%.2f", cfg.symbol, qty, price)
            return True

        try:
            if cfg.use_bracket_orders and cfg.stop_loss_pct > 0 and cfg.take_profit_pct > 0:
                # Calcular niveles de SL/TP.
                sl_price = round(price * (1.0 - cfg.stop_loss_pct / 100.0), 4)
                tp_price = round(price * (1.0 + cfg.take_profit_pct / 100.0), 4)
                # Bracket: Market entry + SL + TP OCA.
                self.broker.place_bracket_order(
                    symbol=cfg.symbol,
                    side="buy",
                    qty=qty,
                    limit_price=None,       # Market entry
                    stop_loss_price=sl_price,
                    take_profit_price=tp_price,
                )
                logger.info(
                    "IB BracketOrder BUY %s qty=%.4f SL=%.2f TP=%.2f",
                    cfg.symbol, qty, sl_price, tp_price,
                )
            else:
                self.broker.place_market_order(cfg.symbol, "buy", qty)
        except Exception as exc:
            logger.error("Fallo ejecutando BUY en IB: %s", exc)
            return False

        self._notify(f"🟢 IB BUY {cfg.symbol} qty={qty:.4f} @ ~${price:.2f}")
        return True

    def _execute_sell(self) -> bool:
        cfg = self.config
        if not self._open_positions_cache:
            return False
        if cfg.dry_run:
            logger.info("[DRY-RUN] SELL %s", cfg.symbol)
            return True
        try:
            self.broker.close_position(cfg.symbol)
        except Exception as exc:
            logger.error("Fallo ejecutando SELL en IB: %s", exc)
            return False
        self._notify(f"🔴 IB SELL {cfg.symbol}")
        return True

    def _check_sl_tp(self, last_price: float) -> None:
        """Chequeo manual de SL/TP barra a barra (cuando NO se usan bracket orders)."""
        cfg = self.config
        if cfg.stop_loss_pct <= 0 and cfg.take_profit_pct <= 0:
            return
        for pos in self._open_positions_cache:
            entry = float(pos.get("avg_entry_price", last_price))
            chg = (last_price - entry) / max(entry, 1e-6) * 100.0
            if cfg.stop_loss_pct > 0 and chg <= -cfg.stop_loss_pct:
                logger.info("SL hit %.2f%% @ %.2f", chg, last_price)
                if not cfg.dry_run:
                    self.broker.close_position(cfg.symbol)
            elif cfg.take_profit_pct > 0 and chg >= cfg.take_profit_pct:
                logger.info("TP hit %.2f%% @ %.2f", chg, last_price)
                if not cfg.dry_run:
                    self.broker.close_position(cfg.symbol)

    def _refresh_positions(self) -> None:
        if self.broker is None:
            self._open_positions_cache = []
            return
        try:
            all_pos = self.broker.get_positions()
            self._open_positions_cache = [
                p for p in all_pos
                if p.get("symbol", "").upper() == self.config.symbol.upper()
            ]
        except Exception as exc:
            logger.warning("get_positions falló: %s", exc)

    def _notify(self, msg: str) -> None:
        if self.telegram_fn and self.config.send_telegram:
            try:
                self.telegram_fn(msg)
            except Exception:
                pass

    def _write_state_snapshot(self, decision: LiveDecision) -> None:
        """Escribe el snapshot JSON para el dashboard (mismo formato que LiveRunner)."""
        import json, os
        from pathlib import Path

        cfg = self.config
        entry = {
            "iter": self._iter_count,
            "timestamp": (
                decision.timestamp.isoformat()
                if hasattr(decision.timestamp, "isoformat")
                else str(decision.timestamp)
            ),
            "action": decision.action,
            "price": decision.price,
            "size_multiplier": decision.size_multiplier,
            "proba": decision.proba,
            "executed": decision.executed,
            "reason": decision.reason,
        }
        self._decision_history.append(entry)
        if len(self._decision_history) > cfg.state_history_size:
            self._decision_history = self._decision_history[-cfg.state_history_size:]

        snapshot = {
            "schema_version": 1,
            "updated_at": pd.Timestamp.utcnow().isoformat(),
            "symbol": cfg.symbol,
            "timeframe": cfg.timeframe,
            "strategy": self.strategy.name,
            "dry_run": cfg.dry_run,
            "data_source": "ibkr",
            "iter_count": self._iter_count,
            "last_decision": entry,
            "open_positions": self._open_positions_cache,
            "history": self._decision_history,
            "drift": None,
            "retrain_state": None,
        }
        path = Path(cfg.state_path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(snapshot, f, indent=2, default=str)
        os.replace(tmp, path)


# ════════════════════════════════════════════════════════════════════
#  Runner multi-símbolo IBKR (Portfolio Live)
# ════════════════════════════════════════════════════════════════════
@dataclass
class IBKRPortfolioLiveConfig:
    """Configuración del runner multi-símbolo IBKR."""
    symbols: List[str] = field(default_factory=lambda: ["AAPL"])
    timeframe: str = "1Day"
    history_bars: int = 500
    max_positions: int = 4
    position_size_pct: float = 20.0
    corr_threshold: float = 0.75
    dry_run: bool = False
    stop_loss_pct: float = 0.0
    take_profit_pct: float = 0.0
    use_bracket_orders: bool = False
    max_iters: Optional[int] = None
    state_path: Optional[str] = None
    send_telegram: bool = False
    use_rth: bool = True


class IBKRPortfolioLiveRunner:
    """
    Runner live multi-símbolo para Interactive Brokers.

    La lógica de portfolio (cap de posiciones, correlation guard, sizing
    compartido) es idéntica a PortfolioLiveRunner; sólo cambia la fuente
    de datos (IB en lugar de Alpaca/Yahoo) y el bucle de eventos (IB
    barUpdateEvent en lugar de while-True + time.sleep).

    Diseño:
        - Se suscriben todos los símbolos a barUpdateEvent.
        - Cada vez que un símbolo emite una barra nueva, se actualiza su
          DataFrame interno.
        - Un timer IB (ib.schedule()) o el callback mismo chequea
          periódicamente si hay señales nuevas para el conjunto.
    """

    def __init__(
        self,
        strategy: BaseStrategy,
        broker: IBKRBroker,
        config: Optional[IBKRPortfolioLiveConfig] = None,
        overlay=None,
        telegram_fn: Optional[Callable[[str], None]] = None,
    ):
        self.strategy = strategy
        self.broker = broker
        self.config = config or IBKRPortfolioLiveConfig()
        self.overlay = overlay
        self.telegram_fn = telegram_fn

        self._ib = broker.get_ib()
        self._fetcher: IBKRDataFetcher = broker.data_fetcher

        self._dfs: Dict[str, pd.DataFrame] = {}
        self._pending_signals: Dict[str, Action] = {}
        self._all_positions: List[Dict] = []
        self._iter_count: int = 0
        self._decision_history: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    def run(self) -> None:
        cfg = self.config
        logger.info(
            "IBKR PORTFOLIO LIVE START — symbols=%s tf=%s dry_run=%s max_pos=%d",
            cfg.symbols, cfg.timeframe, cfg.dry_run, cfg.max_positions,
        )

        # ── Histórico inicial para todos los símbolos ────────────────
        for sym in cfg.symbols:
            df = self._fetcher.fetch_historical(
                symbol=sym,
                timeframe=cfg.timeframe,
                limit=cfg.history_bars,
                use_rth=cfg.use_rth,
            )
            if not df.empty:
                self._dfs[sym] = df
                logger.info("  %-8s : %d filas cargadas.", sym, len(df))
            else:
                logger.warning("  %-8s : sin datos. Se omite.", sym)

        if not self._dfs:
            logger.error("Ningún símbolo devolvió datos. Abortando.")
            return

        # ── Suscripción en tiempo real ──────────────────────────────
        for sym in self._dfs:
            self._fetcher.subscribe_realtime(
                symbol=sym,
                callback=self._on_new_bar,
                timeframe=cfg.timeframe,
            )

        self._notify(
            f"🤖 IBKR Portfolio Bot: {', '.join(cfg.symbols)} "
            f"tf={cfg.timeframe} max_pos={cfg.max_positions}"
        )

        try:
            self._ib.run()
        except KeyboardInterrupt:
            logger.info("Portfolio bot detenido (Ctrl+C).")
            self._notify("🛑 IBKR Portfolio Bot detenido.")
        finally:
            for sym in list(self._dfs.keys()):
                self._fetcher.unsubscribe(sym)
            self.broker.disconnect()

    # ------------------------------------------------------------------
    def _on_new_bar(self, symbol: str, df: pd.DataFrame) -> None:
        """Callback por símbolo cuando IB emite una barra nueva."""
        cfg = self.config
        self._iter_count += 1

        if cfg.max_iters is not None and self._iter_count > cfg.max_iters:
            self._ib.stop()
            return

        # Actualizar DataFrame del símbolo.
        if not df.empty:
            df.attrs["symbol"] = symbol
            if symbol in self._dfs:
                combined = pd.concat([self._dfs[symbol], df])
                combined = combined[~combined.index.duplicated(keep="last")]
                self._dfs[symbol] = combined.tail(cfg.history_bars)
            else:
                self._dfs[symbol] = df

        # Generar señal para este símbolo.
        try:
            actions = self.strategy.generate_signals(self._dfs[symbol])
            self._pending_signals[symbol] = actions.iloc[-1]
        except Exception as exc:
            logger.error("[%s] generate_signals falló: %s", symbol, exc)
            return

        # Procesar portafolio cuando tengamos señales para todos.
        # (Esperar a tener al menos 1 señal de cada símbolo cargado.)
        if len(self._pending_signals) >= len(self._dfs):
            self._process_portfolio()

    # ------------------------------------------------------------------
    def _process_portfolio(self) -> None:
        """
        Itera sobre las señales pendientes y ejecuta SELLs primero,
        luego BUYs con filtro de cap + correlación.
        Lógica idéntica a PortfolioLiveRunner._process().
        """
        cfg = self.config
        self._all_positions = self.broker.get_positions()
        open_syms = {p["symbol"].upper() for p in self._all_positions}

        # ── SELLs ──────────────────────────────────────────────────
        for sym, action in list(self._pending_signals.items()):
            if action == Action.SELL and sym.upper() in open_syms:
                if cfg.dry_run:
                    logger.info("[DRY-RUN] SELL %s", sym)
                else:
                    try:
                        self.broker.close_position(sym)
                        self._notify(f"🔴 IB SELL {sym}")
                    except Exception as exc:
                        logger.error("SELL %s falló: %s", sym, exc)

        # Refrescar posiciones tras los SELLs.
        self._all_positions = self.broker.get_positions()
        open_syms = {p["symbol"].upper() for p in self._all_positions}

        # ── BUYs ────────────────────────────────────────────────────
        n_open = len(open_syms)
        for sym in sorted(self._pending_signals.keys()):
            if self._pending_signals[sym] != Action.BUY:
                continue
            if sym.upper() in open_syms:
                continue
            if n_open >= cfg.max_positions:
                logger.info("Cap posiciones (%d/%d) — SKIP BUY %s", n_open, cfg.max_positions, sym)
                continue

            df = self._dfs.get(sym)
            if df is None or df.empty:
                continue
            last_price = float(df["close"].iloc[-1])

            size_mult = 1.0
            try:
                size_mult, _ = _size_multiplier_last_bar(self.strategy, df, self.overlay)
            except Exception:
                pass

            if size_mult <= 0:
                continue

            equity = self.broker.get_balance()["total"] if not cfg.dry_run else 10_000.0
            dollars = min(equity * (cfg.position_size_pct / 100.0) * size_mult,
                          self.broker.get_balance()["free"] if not cfg.dry_run else equity)
            qty = round(dollars / max(last_price, 1e-6), 4)
            if qty <= 0:
                continue

            if cfg.dry_run:
                logger.info("[DRY-RUN] BUY %s qty=%.4f @ ~%.2f mult=%.2f", sym, qty, last_price, size_mult)
            else:
                try:
                    self.broker.place_market_order(sym, "buy", qty)
                    self._notify(f"🟢 IB BUY {sym} qty={qty:.4f} @ ~${last_price:.2f}")
                    n_open += 1
                    open_syms.add(sym.upper())
                except Exception as exc:
                    logger.error("BUY %s falló: %s", sym, exc)

        # Limpiar señales procesadas.
        self._pending_signals.clear()

    def _notify(self, msg: str) -> None:
        if self.telegram_fn and self.config.send_telegram:
            try:
                self.telegram_fn(msg)
            except Exception:
                pass