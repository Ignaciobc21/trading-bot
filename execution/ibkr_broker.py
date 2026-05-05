"""
execution/ibkr_broker.py — Adaptador Interactive Brokers (ib_insync).

Reemplaza execution/broker.py (Alpaca) con una implementación completa
para IB TWS / IB Gateway. Mantiene la MISMA interfaz pública que Broker
(get_balance, place_market_order, close_position, get_positions, etc.)
para que el resto del código (LiveRunner, OrderManager) NO necesite
modificarse.

Cambios respecto al broker Alpaca original:
  1. Fuente de datos: ib.reqHistoricalData + ib.barUpdateEvent (en lugar
     del REST de Alpaca).
  2. Mapeo de activos: símbolo str → contrato IB (Stock / Crypto / Forex /
     Future) con Smart Routing.
  3. Órdenes: MarketOrder / LimitOrder de ib_insync, con soporte para
     Bracket (OCA group con Stop + Take Profit adjuntos).
  4. Bucle: todo corre dentro del event loop de ib_insync; usa ib.sleep()
     en lugar de time.sleep() para no bloquear.
  5. Error handling: listener para códigos TWS (1100 disconnected, 2104
     data farm, 502 no connection, etc.).

Requisitos:
    pip install ib_insync

Conexión típica a TWS Paper:
    host="127.0.0.1", port=7497, clientId=1  (Paper Trading TWS)
    host="127.0.0.1", port=4002, clientId=1  (IB Gateway Paper)
    host="127.0.0.1", port=7496, clientId=1  (TWS Live)
    host="127.0.0.1", port=4001, clientId=1  (IB Gateway Live)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════
#  Detección de ib_insync (import opcional para no romper el bot
#  si no está instalado — en ese caso IBKRBroker lanzará ImportError
#  con mensaje claro al instanciarse, no al importar el módulo).
# ════════════════════════════════════════════════════════════════════
try:
    from ib_insync import (
        IB,
        Stock,
        Crypto,
        Forex,
        Future,
        ContFuture,
        Contract,
        MarketOrder,
        LimitOrder,
        StopOrder,
        Order,
        Trade,
        BarData,
        util,
    )
    _IB_AVAILABLE = True
except ImportError:
    _IB_AVAILABLE = False


# ════════════════════════════════════════════════════════════════════
#  Configuración de conexión.
# ════════════════════════════════════════════════════════════════════
@dataclass
class IBKRConfig:
    """
    Parámetros de conexión a TWS / IB Gateway.

    Attributes
    ----------
    host : str
        IP de la máquina que corre TWS/Gateway. "127.0.0.1" para local.
    port : int
        7497 = TWS Paper | 4002 = Gateway Paper
        7496 = TWS Live  | 4001 = Gateway Live
    client_id : int
        ID único por conexión simultánea. Si tienes varios bots, usa
        IDs distintos (1, 2, 3…). TWS soporta hasta 32 conexiones.
    timeout : float
        Segundos de espera para que los requests IB respondan.
    readonly : bool
        Si True, el cliente no puede enviar órdenes. Útil para monitoreo.
    account : str
        Cuenta IB. Si vacío, se usa la primera cuenta disponible.
        Formato: "DU1234567" (paper) o "U1234567" (live).
    """
    host: str = "127.0.0.1"
    port: int = 7497           # TWS Paper por defecto
    client_id: int = 1
    timeout: float = 20.0
    readonly: bool = False
    account: str = ""          # "" → primera cuenta disponible


# ════════════════════════════════════════════════════════════════════
#  Clasificador de símbolos → tipo de contrato IB.
# ════════════════════════════════════════════════════════════════════
# Prefijos / sufijos que indican que NO es una equity US estándar.
_CRYPTO_SYMBOLS = {
    "BTC-USD", "ETH-USD", "LTC-USD", "BCH-USD",
    "BTC",    "ETH",    "LTC",    "BCH",
    "XRP-USD","ADA-USD","SOL-USD","DOT-USD",
}
_FOREX_PAIRS = {
    "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD",
    "USDCHF", "NZDUSD", "EURGBP", "EURJPY",
}
# Sufijos que Yahoo usa para mercados no-US (ignoramos en el ticker).
_YAHOO_SUFFIXES = (".L", ".PA", ".DE", ".AS", ".MI", ".HK", ".TO", ".AX")


def resolve_contract(symbol: str, currency: str = "USD", exchange: str = "SMART") -> "Contract":
    """
    Convierte un símbolo string al contrato ib_insync correcto.

    Lógica de detección:
      - "BTC-USD", "ETH" → Crypto (exchange="PAXOS")
      - "EURUSD", "GBPUSD" → Forex
      - Otros → Stock(exchange="SMART", currency="USD")

    Args:
        symbol   : Ticker al estilo Yahoo ("AAPL", "BTC-USD", "MSFT").
        currency : Moneda del contrato (default "USD").
        exchange : Exchange de enrutamiento (default "SMART").

    Returns:
        Contrato ib_insync listo para usar en reqHistoricalData / placeOrder.
    """
    if not _IB_AVAILABLE:
        raise ImportError(
            "ib_insync no está instalado. Ejecuta: pip install ib_insync"
        )

    # Limpiar sufijos Yahoo (AAPL.L → AAPL).
    clean = symbol.upper()
    for sfx in _YAHOO_SUFFIXES:
        if clean.endswith(sfx.upper()):
            clean = clean[: -len(sfx)]
            break

    # ── Crypto ──────────────────────────────────────────────────────
    if clean in _CRYPTO_SYMBOLS or clean.endswith("-USD"):
        base = clean.split("-")[0]          # "BTC-USD" → "BTC"
        # IB Crypto: Crypto("BTC", "USD", "PAXOS")
        return Crypto(base, "USD", "PAXOS")

    # ── Forex ────────────────────────────────────────────────────────
    if clean in _FOREX_PAIRS or (len(clean) == 6 and clean.isalpha()):
        pair = clean[:3]
        quote_ccy = clean[3:]
        return Forex(pair, currency=quote_ccy, localSymbol=clean)

    # ── Acciones / ETFs US (SMART) ───────────────────────────────────
    return Stock(clean, exchange, currency)


# ════════════════════════════════════════════════════════════════════
#  Mapeador de timeframe (Alpaca → duración IB + bar_size IB).
# ════════════════════════════════════════════════════════════════════
_TF_TO_IB: Dict[str, Tuple[str, str]] = {
    "1Min":  ("2 D",   "1 min"),
    "2Min":  ("2 D",   "2 mins"),
    "5Min":  ("5 D",   "5 mins"),
    "15Min": ("10 D",  "15 mins"),
    "30Min": ("20 D",  "30 mins"),
    "1Hour": ("60 D",  "1 hour"),
    "4Hour": ("120 D", "4 hours"),
    "1Day":  ("5 Y",   "1 day"),
    "1d":    ("5 Y",   "1 day"),
    "1h":    ("60 D",  "1 hour"),
}


def timeframe_to_ib(timeframe: str) -> Tuple[str, str]:
    """
    Convierte un timeframe Alpaca-style a (duration_str, bar_size_str) de IB.

    Returns:
        ("5 Y", "1 day") para "1Day", etc.

    Raises:
        ValueError si el timeframe no está mapeado.
    """
    result = _TF_TO_IB.get(timeframe)
    if result is None:
        raise ValueError(
            f"Timeframe '{timeframe}' no tiene mapeo a IB. "
            f"Disponibles: {list(_TF_TO_IB.keys())}"
        )
    return result


# ════════════════════════════════════════════════════════════════════
#  DataFetcher IBKR — sustituye data/fetcher.py para el modo live.
# ════════════════════════════════════════════════════════════════════
class IBKRDataFetcher:
    """
    Fetcher de barras históricas y en tiempo real usando IB.

    Diferencia clave respecto al DataFetcher de Alpaca/Yahoo:
      - fetch_historical() → ib.reqHistoricalData() (bloqueante, una vez).
      - subscribe_realtime() → ib.reqRealTimeBars() + barUpdateEvent callback.
        IB emite una barra cada 5 segundos en tiempo real (RTBars).
        Para barras de 1min/1d se usa reqHistoricalData con keepUpToDate=True.

    IBKR tiene límites de pacing: máx 50 requests por 10 min para histórico.
    No hagas polling agresivo; el bot llama a fetch_historical() una vez al
    inicio del timeframe y luego escucha eventos.
    """

    def __init__(self, ib: "IB"):
        self._ib = ib
        self._bar_subscriptions: Dict[str, Any] = {}

    def fetch_historical(
        self,
        symbol: str,
        timeframe: str = "1Day",
        limit: int = 500,
        end_datetime: str = "",
        what_to_show: str = "TRADES",
        use_rth: bool = True,
    ) -> pd.DataFrame:
        """
        Descarga barras históricas. Equivale a DataFetcher.fetch_yahoo().

        Args:
            symbol      : Ticker ("AAPL", "BTC-USD", etc.).
            timeframe   : Timeframe Alpaca-style ("1Day", "1Hour", "5Min").
            limit       : Número aproximado de barras deseadas (ajusta duration).
            end_datetime: Fecha final IB format ("20240101 23:59:59 US/Eastern").
                          "" = ahora.
            what_to_show: "TRADES" para equities/ETFs. "MIDPOINT" para Forex.
                          "BID_ASK" disponible también.
            use_rth     : True = sólo Regular Trading Hours. False = 24h.

        Returns:
            DataFrame OHLCV con DatetimeIndex UTC, columnas en minúsculas.
        """
        contract = resolve_contract(symbol)
        duration_str, bar_size = timeframe_to_ib(timeframe)

        # Para Crypto/Forex, IB no usa RTH.
        is_crypto = isinstance(contract, Crypto)
        is_forex = isinstance(contract, Forex)
        rth = 0 if (is_crypto or is_forex) else (1 if use_rth else 0)
        show = "MIDPOINT" if is_forex else what_to_show

        logger.info(
            "IB reqHistoricalData: %s duration=%s bar=%s rth=%s",
            symbol, duration_str, bar_size, rth,
        )

        bars = self._ib.reqHistoricalData(
            contract=contract,
            endDateTime=end_datetime,
            durationStr=duration_str,
            barSizeSetting=bar_size,
            whatToShow=show,
            useRTH=rth,
            formatDate=1,
            keepUpToDate=False,
        )

        if not bars:
            logger.warning("IB devolvió 0 barras para %s", symbol)
            return pd.DataFrame()

        df = util.df(bars)
        df.columns = [c.lower() for c in df.columns]
        # La columna de tiempo en IB se llama 'date'; renombrarla a índice.
        if "date" in df.columns:
            df = df.set_index("date")
        df.index = pd.to_datetime(df.index, utc=True)
        df.index.name = None
        # Aseguramos las columnas OHLCV estándar.
        df = df.rename(columns={"average": "vwap", "barcount": "barcount"})
        for col in ("open", "high", "low", "close", "volume"):
            if col not in df.columns:
                logger.warning("Columna '%s' no encontrada en respuesta IB para %s", col, symbol)
        keep = [c for c in ("open", "high", "low", "close", "volume") if c in df.columns]
        df = df[keep].tail(limit)
        df.attrs["symbol"] = symbol
        logger.info("IB histórico %s: %d filas [%s → %s]", symbol, len(df), df.index[0], df.index[-1])
        return df

    def subscribe_realtime(
        self,
        symbol: str,
        callback: Callable[[str, pd.DataFrame], None],
        timeframe: str = "1Day",
    ) -> None:
        """
        Suscribe a actualizaciones de barras en tiempo real.

        IB ofrece dos mecanismos:
          a) reqRealTimeBars: barras de 5 seg, siempre disponibles.
          b) reqHistoricalData con keepUpToDate=True: recibe la barra
             al completarse cada período (1min, 1h, 1d…). Preferido
             para timeframes > 5s porque está alineado con el cierre
             de la vela.

        Este método usa (b) via barUpdateEvent. El callback recibe el
        símbolo y el DataFrame actualizado.

        Args:
            symbol    : Ticker.
            callback  : fn(symbol: str, df: pd.DataFrame) → llamada al
                        completarse cada nueva barra.
            timeframe : Timeframe Alpaca-style.
        """
        if symbol in self._bar_subscriptions:
            logger.debug("Ya suscrito a %s, ignorando.", symbol)
            return

        contract = resolve_contract(symbol)
        duration_str, bar_size = timeframe_to_ib(timeframe)
        is_forex = isinstance(contract, Forex)
        show = "MIDPOINT" if is_forex else "TRADES"
        rth = 0 if isinstance(contract, (Crypto, Forex)) else 1

        logger.info("IB subscribe realtime: %s bar=%s", symbol, bar_size)

        bars_obj = self._ib.reqHistoricalData(
            contract=contract,
            endDateTime="",
            durationStr=duration_str,
            barSizeSetting=bar_size,
            whatToShow=show,
            useRTH=rth,
            formatDate=1,
            keepUpToDate=True,          # ← activa el streaming
        )
        self._bar_subscriptions[symbol] = bars_obj

        # barUpdateEvent se dispara cuando llega una barra nueva o se
        # actualiza la barra actual (IB actualiza la última barra durante
        # su construcción).
        def _on_bar_update(bars: List[BarData], has_new_bar: bool) -> None:
            if not has_new_bar:
                return
            try:
                df = util.df(bars)
                df.columns = [c.lower() for c in df.columns]
                if "date" in df.columns:
                    df = df.set_index("date")
                df.index = pd.to_datetime(df.index, utc=True)
                keep = [c for c in ("open", "high", "low", "close", "volume") if c in df.columns]
                df = df[keep]
                df.attrs["symbol"] = symbol
                callback(symbol, df)
            except Exception as exc:
                logger.error("Error en callback de barra [%s]: %s", symbol, exc)

        bars_obj.updateEvent += _on_bar_update

    def unsubscribe(self, symbol: str) -> None:
        """Cancela la suscripción de barras en tiempo real para un símbolo."""
        bars_obj = self._bar_subscriptions.pop(symbol, None)
        if bars_obj is not None:
            self._ib.cancelHistoricalData(bars_obj)
            logger.info("IB unsuscribed realtime: %s", symbol)


# ════════════════════════════════════════════════════════════════════
#  IBKRBroker — interfaz pública idéntica a Broker (Alpaca).
# ════════════════════════════════════════════════════════════════════
class IBKRBroker:
    """
    Wrapper de ib_insync con la misma API que execution/broker.py (Alpaca).

    Métodos mantenidos (drop-in replacement):
        get_balance()           → {"free": cash, "used": ..., "total": ...}
        place_market_order()    → dict con id, status, symbol, side, qty
        place_limit_order()     → dict con id, status, symbol
        cancel_order()          → None
        get_open_orders()       → List[dict]
        get_order_status()      → dict
        get_ticker()            → {"last": ..., "bid": ..., "ask": ...}
        get_positions()         → List[dict]
        close_position()        → None

    Métodos NUEVOS (específicos de IB):
        place_bracket_order()   → Bracket con Stop Loss + Take Profit.
        get_ib()                → acceso al objeto IB subyacente.
    """

    def __init__(self, config: Optional[IBKRConfig] = None):
        if not _IB_AVAILABLE:
            raise ImportError(
                "ib_insync no está instalado.\n"
                "Instálalo con: pip install ib_insync\n"
                "Documentación: https://ib-insync.readthedocs.io/"
            )
        self.config = config or IBKRConfig()
        self._ib = IB()
        self._connect()
        self._setup_error_handler()
        self._account = self._resolve_account()
        self.data_fetcher = IBKRDataFetcher(self._ib)
        logger.info(
            "IBKRBroker conectado a %s:%d  cuenta=%s",
            self.config.host, self.config.port, self._account,
        )

    # ------------------------------------------------------------------
    # Conexión
    # ------------------------------------------------------------------
    def _connect(self) -> None:
        """Conecta a TWS / IB Gateway con reintentos básicos."""
        cfg = self.config
        logger.info(
            "Conectando a IB en %s:%d (clientId=%d)…",
            cfg.host, cfg.port, cfg.client_id,
        )
        self._ib.connect(
            host=cfg.host,
            port=cfg.port,
            clientId=cfg.client_id,
            timeout=cfg.timeout,
            readonly=cfg.readonly,
        )
        if not self._ib.isConnected():
            raise ConnectionError(
                f"No se pudo conectar a IB en {cfg.host}:{cfg.port}. "
                "Verifica que TWS o IB Gateway esté abierto y que la API esté habilitada "
                "(TWS → Edit → Global Configuration → API → Settings → Enable ActiveX and Socket Clients)."
            )

    def _resolve_account(self) -> str:
        """Devuelve la cuenta activa (config.account o la primera disponible)."""
        if self.config.account:
            return self.config.account
        accounts = self._ib.managedAccounts()
        if not accounts:
            raise RuntimeError("IB no devolvió ninguna cuenta administrada.")
        account = accounts[0]
        logger.info("Cuenta IB auto-detectada: %s", account)
        return account

    # ------------------------------------------------------------------
    # Error handler (PUNTO 5 del brief)
    # ------------------------------------------------------------------
    def _setup_error_handler(self) -> None:
        """
        Registra un listener para los mensajes de error de la API TWS/Gateway.

        Códigos relevantes:
          1100  — TWS desconectado de los servidores IB. Perderemos los
                  datos de mercado en tiempo real. El bot debe pausar.
          1101  — TWS re-conectado; suscripciones de datos perdidas. Hay
                  que re-suscribirse.
          1102  — TWS re-conectado; suscripciones de datos recuperadas.
          2100  — API client disconnected. clientId ya en uso.
          2103  — Market data farm connection is broken.
          2104  — Market data farm connection is OK.
          2106  — HMDS data farm connected.
          2107  — HMDS data farm inactive (sin datos históricos).
          2108  — Market data farm connection is inactive.
          502   — Couldn't connect to TWS (no hay nadie escuchando).
          504   — Not connected.
          10197 — No hay datos disponibles para el contrato/período pedido.
        """
        def on_error(
            req_id: int,
            error_code: int,
            error_string: str,
            contract: Optional["Contract"] = None,
        ) -> None:
            # Mensajes puramente informativos (no son errores reales).
            INFO_CODES = {2104, 2106, 2158, 2103, 2107, 2108}
            if error_code in INFO_CODES:
                logger.debug("[IB API INFO %d] %s", error_code, error_string)
                return

            # Desconexión de TWS.
            if error_code == 1100:
                logger.error(
                    "[IB API] TWS desconectado de los servidores IB (código 1100). "
                    "El bot no recibirá datos de mercado. "
                    "Esperando reconexión automática…"
                )
                return

            # Re-conexión con pérdida de suscripciones.
            if error_code == 1101:
                logger.warning(
                    "[IB API] TWS re-conectado (código 1101). "
                    "Las suscripciones de datos de mercado se han PERDIDO. "
                    "Re-suscribir en el próximo ciclo."
                )
                return

            # Re-conexión sin pérdida de suscripciones.
            if error_code == 1102:
                logger.info(
                    "[IB API] TWS re-conectado (código 1102). "
                    "Las suscripciones de datos se han recuperado automáticamente."
                )
                return

            # clientId duplicado.
            if error_code == 2100:
                logger.error(
                    "[IB API] Client ID %d ya está en uso (código 2100). "
                    "Cambia IBKRConfig.client_id a un valor único.", req_id
                )
                return

            # Sin conexión a TWS.
            if error_code in (502, 504):
                logger.critical(
                    "[IB API] Sin conexión a TWS/Gateway (código %d): %s. "
                    "Verifica que TWS esté abierto.", error_code, error_string
                )
                return

            # Sin datos disponibles para el contrato (aviso, no error fatal).
            if error_code == 10197:
                sym = contract.symbol if contract else "desconocido"
                logger.warning(
                    "[IB API] Sin datos para '%s' (código 10197): %s", sym, error_string
                )
                return

            # Mensajes de riesgo / margen (313, 321, 322) → loguear como WARNING.
            if 300 <= error_code < 400:
                logger.warning("[IB API RISK %d] %s", error_code, error_string)
                return

            # Resto de errores.
            logger.error(
                "[IB API ERROR reqId=%d code=%d] %s", req_id, error_code, error_string
            )

        self._ib.errorEvent += on_error

    # ------------------------------------------------------------------
    # API pública compatible con execution/broker.py
    # ------------------------------------------------------------------
    def get_balance(self, currency: str = "USD") -> Dict[str, float]:
        """
        Devuelve el balance de la cuenta.

        Mapeo IB → estructura del bot:
          NetLiquidation   → total
          TotalCashValue   → free  (efectivo disponible)
          GrossPositionValue → used (valor de posiciones abiertas)
        """
        vals = self._ib.accountValues(self._account)
        _map: Dict[str, float] = {}
        for av in vals:
            if av.currency == currency:
                _map[av.tag] = _safe_float(av.value)

        total = _map.get("NetLiquidation", 0.0)
        free  = _map.get("TotalCashValue", 0.0)
        used  = _map.get("GrossPositionValue", 0.0)
        logger.info("IB balance: free=%.2f used=%.2f total=%.2f %s", free, used, total, currency)
        return {"free": free, "used": used, "total": total}

    # ------------------------------------------------------------------
    def place_market_order(
        self, symbol: str, side: str, qty: float
    ) -> Dict[str, Any]:
        """
        Coloca una orden de mercado en IB.

        IB requiere:
          - Contrato cualificado (reqContractDetails primero para acciones).
          - qty como float; IB acepta fraccional en algunos activos.
          - side: "buy" o "sell" → IB usa "BUY" / "SELL" en mayúsculas.

        ¿Por qué IB puede RECHAZAR una orden de mercado?
          - Fuera de RTH: equities no aceptan Market fuera de sesión.
            Solución: usar LimitOrder al bid/ask o añadir tif="OPG".
          - Contrato no cualificado: necesita conId. Lo resolvemos con
            _qualify().
          - Cuenta en paper y símbolo no disponible en paper data.
        """
        contract = self._qualify(resolve_contract(symbol))
        action = side.upper()  # IB usa "BUY" / "SELL"
        order = MarketOrder(action, qty)
        order.account = self._account
        trade = self._ib.placeOrder(contract, order)
        self._ib.sleep(0.5)  # dar tiempo al TWS para procesar
        logger.info(
            "IB MarketOrder %s %s qty=%.4f — orderId=%s status=%s",
            action, symbol, qty, trade.order.orderId, trade.orderStatus.status,
        )
        return _trade_to_dict(trade)

    # ------------------------------------------------------------------
    def place_limit_order(
        self, symbol: str, side: str, qty: float, price: float
    ) -> Dict[str, Any]:
        """
        Coloca una orden límite en IB.

        LimitOrder es la orden más segura en IB porque funciona dentro
        Y fuera de RTH (con tif="GTC"). Market orders fuera de sesión
        son rechazadas en acciones.
        """
        contract = self._qualify(resolve_contract(symbol))
        action = side.upper()
        order = LimitOrder(action, qty, price)
        order.account = self._account
        order.tif = "GTC"            # Good Till Cancelled
        trade = self._ib.placeOrder(contract, order)
        self._ib.sleep(0.5)
        logger.info(
            "IB LimitOrder %s %s qty=%.4f @ %.4f — orderId=%s",
            action, symbol, qty, price, trade.order.orderId,
        )
        return _trade_to_dict(trade)

    # ------------------------------------------------------------------
    def place_bracket_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        limit_price: Optional[float],
        stop_loss_price: float,
        take_profit_price: float,
    ) -> List[Dict[str, Any]]:
        """
        Coloca una orden Bracket: entrada + Stop Loss + Take Profit en un
        único grupo OCA (One-Cancels-All). Cuando se dispara SL o TP, la
        otra se cancela automáticamente.

        IB implementa esto con tres órdenes vinculadas por parentId. La
        entrada puede ser Market o Limit.

        Args:
            symbol             : Ticker.
            side               : "buy" / "sell" (dirección de la entrada).
            qty                : Cantidad.
            limit_price        : None → MarketOrder de entrada. Float → LimitOrder.
            stop_loss_price    : Precio del stop (al contrario de la entrada).
            take_profit_price  : Precio del take profit.

        Returns:
            Lista de 3 dicts (entrada, take_profit, stop_loss).

        ¿Por qué 3 órdenes y no 1?
            IB no tiene un tipo de orden "bracket" unificado; lo construye
            el SDK como 3 órdenes enlazadas por parentId y con tif="GTC".
            Si la entrada se llena, las 2 salidas quedan activas como OCA.
        """
        contract = self._qualify(resolve_contract(symbol))
        action = side.upper()
        exit_action = "SELL" if action == "BUY" else "BUY"

        # ── Orden de entrada ────────────────────────────────────────
        if limit_price is not None:
            parent = LimitOrder(action, qty, limit_price)
        else:
            parent = MarketOrder(action, qty)
        parent.account = self._account
        parent.transmit = False          # No transmitir hasta tener las 3.

        # ── Take Profit ─────────────────────────────────────────────
        tp = LimitOrder(exit_action, qty, take_profit_price)
        tp.account = self._account
        tp.tif = "GTC"
        tp.parentId = parent.orderId     # se asigna después por IB; usamos bracket()
        tp.transmit = False

        # ── Stop Loss ───────────────────────────────────────────────
        sl = StopOrder(exit_action, qty, stop_loss_price)
        sl.account = self._account
        sl.tif = "GTC"
        sl.parentId = parent.orderId
        sl.transmit = True               # La última orden transmite todo el grupo.

        # ib_insync.IB.bracketOrder() gestiona automáticamente los parentIds
        # y el campo transmit. Lo usamos en su lugar:
        bracket = self._ib.bracketOrder(
            action=action,
            quantity=qty,
            limitPrice=limit_price or 0,
            takeProfitPrice=take_profit_price,
            stopLossPrice=stop_loss_price,
        )
        # bracket devuelve (parent, takeProfit, stopLoss).
        trades = []
        for order in bracket:
            order.account = self._account
            trade = self._ib.placeOrder(contract, order)
            trades.append(trade)
        self._ib.sleep(1)
        logger.info(
            "IB Bracket %s %s qty=%.4f SL=%.2f TP=%.2f — %d órdenes creadas",
            action, symbol, qty, stop_loss_price, take_profit_price, len(trades),
        )
        return [_trade_to_dict(t) for t in trades]

    # ------------------------------------------------------------------
    def cancel_order(self, order_id: str, symbol: str = "") -> None:
        """Cancela una orden abierta por su orderId."""
        trades = self._ib.openTrades()
        for t in trades:
            if str(t.order.orderId) == str(order_id):
                self._ib.cancelOrder(t.order)
                logger.info("IB cancelOrder orderId=%s", order_id)
                return
        logger.warning("IB cancelOrder: orderId=%s no encontrado en openTrades", order_id)

    # ------------------------------------------------------------------
    def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict]:
        """Devuelve las órdenes abiertas, opcionalmente filtradas por símbolo."""
        trades = self._ib.openTrades()
        result = []
        for t in trades:
            sym = t.contract.symbol if t.contract else ""
            if symbol and sym.upper() != symbol.upper():
                continue
            result.append({
                "id": str(t.order.orderId),
                "symbol": sym,
                "side": t.order.action,
                "qty": t.order.totalQuantity,
                "type": t.order.orderType,
                "status": t.orderStatus.status,
            })
        return result

    # ------------------------------------------------------------------
    def get_order_status(self, order_id: str, symbol: str = "") -> Dict[str, Any]:
        """Consulta el estado de una orden por orderId."""
        trades = self._ib.trades()
        for t in trades:
            if str(t.order.orderId) == str(order_id):
                return {
                    "id": str(t.order.orderId),
                    "status": t.orderStatus.status,
                    "filled_avg_price": t.orderStatus.avgFillPrice,
                    "filled": t.orderStatus.filled,
                    "remaining": t.orderStatus.remaining,
                }
        logger.warning("IB get_order_status: orderId=%s no encontrado", order_id)
        return {"id": order_id, "status": "UNKNOWN", "filled_avg_price": None}

    # ------------------------------------------------------------------
    def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """Obtiene el último precio del símbolo (bid/ask/last) vía snapshot."""
        contract = self._qualify(resolve_contract(symbol))
        # reqMktData con snapshot=True → una sola consulta, no stream.
        ticker = self._ib.reqMktData(contract, "", snapshot=True, regulatorySnapshot=False)
        self._ib.sleep(2)          # dar tiempo al snapshot de llegar
        self._ib.cancelMktData(contract)
        last = ticker.last or ticker.close or float("nan")
        bid = ticker.bid or float("nan")
        ask = ticker.ask or float("nan")
        return {"last": float(last), "bid": float(bid), "ask": float(ask)}

    # ------------------------------------------------------------------
    def get_positions(self) -> List[Dict]:
        """Devuelve las posiciones abiertas de la cuenta."""
        positions = self._ib.positions(self._account)
        result = []
        for p in positions:
            sym = p.contract.symbol if p.contract else ""
            avg_cost = p.avgCost or 0.0
            qty = p.position or 0.0
            # IB no da unrealized_pl directamente en positions();
            # se puede consultar con portfolioItems() que sí lo incluye.
            result.append({
                "symbol": sym,
                "qty": float(qty),
                "avg_entry_price": float(avg_cost),
                "current_price": float("nan"),    # necesitaría tick data
                "unrealized_pl": float("nan"),
            })
        return result

    # ------------------------------------------------------------------
    def close_position(self, symbol: str) -> None:
        """Cierra toda la posición de un símbolo con una Market order."""
        positions = self._ib.positions(self._account)
        for p in positions:
            sym = p.contract.symbol if p.contract else ""
            if sym.upper() != symbol.upper():
                continue
            qty = float(p.position or 0.0)
            if qty == 0:
                continue
            side = "SELL" if qty > 0 else "BUY"
            contract = self._qualify(p.contract)
            order = MarketOrder(side, abs(qty))
            order.account = self._account
            self._ib.placeOrder(contract, order)
            self._ib.sleep(0.5)
            logger.info(
                "IB close_position: %s qty=%.4f vía MarketOrder %s", symbol, abs(qty), side
            )
            return
        logger.warning("IB close_position: no hay posición abierta para %s", symbol)

    # ------------------------------------------------------------------
    # Utilidades internas
    # ------------------------------------------------------------------
    def _qualify(self, contract: "Contract") -> "Contract":
        """
        Cualifica el contrato contra los servidores IB.

        IB requiere que el contrato tenga un conId válido antes de colocar
        órdenes. reqContractDetails() devuelve los detalles y rellena el
        conId automáticamente.

        ¿Por qué IB rechaza órdenes sin contrato cualificado?
            TWS valida el conId en el mensaje de la orden. Sin él, devuelve
            el error "No security definition has been found for the request".
        """
        try:
            qualified = self._ib.qualifyContracts(contract)
            return qualified[0] if qualified else contract
        except Exception as exc:
            logger.warning("Qualify contrato falló (%s) — usando sin cualificar.", exc)
            return contract

    def get_ib(self) -> "IB":
        """Acceso al objeto IB subyacente para uso avanzado."""
        return self._ib

    def disconnect(self) -> None:
        """Desconecta limpiamente de TWS/Gateway."""
        if self._ib.isConnected():
            self._ib.disconnect()
            logger.info("IB desconectado.")


# ════════════════════════════════════════════════════════════════════
#  Helpers privados
# ════════════════════════════════════════════════════════════════════
def _safe_float(value: Any) -> float:
    """Convierte un valor a float de forma segura; devuelve 0.0 si falla."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _trade_to_dict(trade: "Trade") -> Dict[str, Any]:
    """Convierte un objeto Trade de ib_insync al dict que espera el bot."""
    contract = trade.contract
    order = trade.order
    status = trade.orderStatus
    return {
        "id": str(order.orderId),
        "status": status.status,
        "symbol": contract.symbol if contract else "",
        "side": order.action,
        "qty": float(order.totalQuantity),
        "filled_avg_price": float(status.avgFillPrice) if status.avgFillPrice else None,
    }