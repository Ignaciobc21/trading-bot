"""
execution/costs.py — Modelo de costes de ejecución (fase I).

Motivación
==========

Hasta la fase H los backtests descontaban costes con dos parámetros
constantes: `commission_pct` (p.ej. 0.1 %) y `slippage_pct` (0.05 %).
Esto es cómodo y trazable pero optimista:

  1. **Spread bid-ask**: los mercados reales tienen un tick mínimo
     (spread entre el mejor bid y ask). Apertura / cierre → spread
     más ancho; medio día → spread más estrecho. No es constante.
  2. **Market impact**: una orden que mueve un % grande del ADV
     (Average Daily Volume) empuja el precio en tu contra. Un modelo
     realista lo aproxima con una raíz cuadrada del ratio
     `order_notional / adv_notional`. Almgren-Chriss propone una
     dependencia lineal en vol y raíz en participación.
  3. **Comisión** en Alpaca retail está cerca de cero en US equities
     (pero ETFs+options+crypto pagan). En bps es lo más transparente.

Arquitectura
============

Dos implementaciones de la interfaz abstracta `CostModel`:

    - `FlatCostModel`  : comisión + slippage constantes (default, retro-
      compatible). Lo que había antes en `BacktestEngine`.
    - `RealisticCostModel` : spread variable por hora del día +
      market-impact raíz cuadrática + comisión en bps.

El motor sólo llama al método `apply(...)` del modelo; internamente
recibe `mid` (open de la barra), `side` (BUY/SELL), `notional` y el
timestamp, y devuelve un `CostBreakdown` con:

    - `fill_price`        : precio efectivo que paga/recibe (ya con
                            slippage direccional aplicado).
    - `commission_paid`   : dinero descontado por comisión.
    - `slippage_paid`     : dinero "pagado" al mercado vía spread+impact
                            (diferencia entre mid y fill_price × qty).
    - `total_cost_bps`    : suma en bps sobre el notional (para auditoría).

Tradeoffs asumidos
==================

- **Daily bars**: en datos diarios no sabemos a qué hora se ejecutó la
  orden. Asumimos "media sesión" (factor horario = 1.0). Si en el futuro
  pasamos a 1m/5m, el factor horario sí se calcula por timestamp.
- **ADV**: se aproxima con el volumen medio de 20 barras. En intraday
  habría que normalizar por franja del día (la apertura tiene 3x el
  volumen de medio día), pero para backtests daily es suficientemente
  cercano.
- **Sin borrow costs**: somos long-only. Cuando añadamos shorts, añadir
  un parámetro `short_borrow_bps_per_day`.
- **Sin market-maker rebates** / maker-taker: asumimos taker (cruzamos
  el spread) tanto en entradas como en salidas. Es lo más conservador y
  fiel a un bot retail que usa market orders.

Uso típico
==========

>>> from execution.costs import RealisticCostModel, CostParams
>>> cm = RealisticCostModel(CostParams(commission_bps=1.0, spread_bps=4.0,
...                                    impact_coef=0.1))
>>> cm.apply(mid=100.0, side="BUY", notional=2000.0, bar=bar_row, adv_notional=2e6)
CostBreakdown(fill_price=100.0408, commission_paid=0.2, slippage_paid=0.816, total_cost_bps=5.08)

El motor (`BacktestEngine` / `PortfolioBacktestEngine`) instancia el
modelo una vez y lo pasa como parámetro `cost_model`. Si se deja a
`None`, se crea un `FlatCostModel` con los valores clásicos para no
romper backtests antiguos.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


# ════════════════════════════════════════════════════════════════════
#  Desglose de costes devuelto por `CostModel.apply`.
# ════════════════════════════════════════════════════════════════════
@dataclass
class CostBreakdown:
    """
    Desglose por trade de los costes cargados por el modelo.

    Todos los valores son en unidades monetarias salvo `total_cost_bps`,
    que es el coste total expresado en basis points (1 bp = 0.01 %)
    sobre el notional bruto. Útil para auditoría:

        - BUY  → notional = mid * quantity                (antes de slippage)
        - SELL → notional = mid * quantity                (antes de slippage)

    `fill_price` es el precio al que el motor materializa el fill:
        BUY  : fill_price > mid  (pagamos el ask ampliado por impact)
        SELL : fill_price < mid  (recibimos el bid recortado por impact)
    """

    fill_price: float          # precio efectivo (ya con slippage direccional).
    commission_paid: float     # dinero por comisión ($).
    slippage_paid: float       # dinero por spread + impact ($).
    total_cost_bps: float      # coste total en bps del notional.


# ════════════════════════════════════════════════════════════════════
#  Parámetros del modelo realista.
# ════════════════════════════════════════════════════════════════════
@dataclass
class CostParams:
    """
    Parámetros del cost model realista.

    Attributes
    ----------
    commission_bps : float
        Comisión en basis points sobre el notional ejecutado.
        Default 1 bp (≈ Alpaca US equities retail). IBKR Pro cobra
        ~0.35 bp (pero con mínimos por ticket).

    spread_bps : float
        Spread bid-ask base en bps. El ESTUDIO "TAQ" de equities US
        grandes (SPY, AAPL) da 1-3 bp; small caps 20-50 bp; crypto
        en exchanges centralizados 5-30 bp. Este valor se MULTIPLICA
        por el factor horario y, si se pasa, por el factor de liquidez
        (ratio low-volume tickers).

    impact_coef : float
        Coeficiente `k` del modelo de market impact:
            impact_bps = k * 10000 * sqrt( notional / adv_notional )
        Con k=0.1, una orden del 1% del ADV → ~10 bps extra.
        10% → ~32 bps. Almgren-Chriss usa ~0.1-0.3 en equities US.

    hour_open_mult : float
        Multiplicador del spread durante la primera hora de sesión
        (09:30–10:30 US East). Apertura = más volatilidad, spreads
        más anchos. Default 1.5x.

    hour_close_mult : float
        Multiplicador del spread durante la última hora (15:00–16:00
        US East). Rebalanceo institucional → spread algo más ancho.
        Default 1.3x.

    hour_mid_mult : float
        Multiplicador del spread en medio día (10:30–15:00 US East).
        Es el periodo más líquido. Default 1.0x (baseline).

    overnight_mult : float
        Multiplicador para datos que no caen dentro del horario de
        sesión US (crypto 24/7, asiáticos, o extended hours).
        Default 1.2x.

    daily_bar_mult : float
        Factor a aplicar cuando no podemos inferir la hora (datos
        daily). Default 1.0x (asumimos "media sesión" en promedio).
    """

    # ── Costes base ───────────────────────────────────────────────
    commission_bps: float = 1.0       # 0.01 %
    spread_bps: float = 4.0           # 0.04 %
    impact_coef: float = 0.10

    # ── Multiplicadores horarios del spread ───────────────────────
    hour_open_mult: float = 1.5
    hour_close_mult: float = 1.3
    hour_mid_mult: float = 1.0
    overnight_mult: float = 1.2
    daily_bar_mult: float = 1.0

    # ── Flag opcional para desactivar impact (debug) ──────────────
    include_impact: bool = True


# ════════════════════════════════════════════════════════════════════
#  Interfaz abstracta del CostModel.
# ════════════════════════════════════════════════════════════════════
class CostModel(ABC):
    """
    Contrato que deben cumplir todos los cost models.

    El motor de backtest llama a `apply(...)` cada vez que abre o cierra
    una posición, pasando:
        mid          = precio "justo" (open de la barra de ejecución).
        side         = "BUY" o "SELL".
        notional     = $ brutos del trade (antes de slippage).
        bar_time     = timestamp de la barra (para hora del día).
        adv_notional = ADV en $ (volumen medio 20d × precio medio).
                       None → el modelo no aplica market impact.

    Devuelve un `CostBreakdown`. El motor sólo necesita leer
    `fill_price` y `commission_paid`; el resto es para logs/audit.
    """

    name: str = "base"

    @abstractmethod
    def apply(
        self,
        mid: float,
        side: str,                # "BUY" | "SELL"
        notional: float,
        bar_time: Optional[pd.Timestamp] = None,
        adv_notional: Optional[float] = None,
        is_daily_bar: bool = True,
    ) -> CostBreakdown:
        ...

    @abstractmethod
    def commission_rate(self) -> float:
        """
        Fracción de comisión aplicada al notional. Se usa en cierres
        por stop (SL/TP/MAX_HOLDING) donde el precio de ejecución es
        el trigger exacto (sin slippage) y sólo cargamos comisión.

        Ejemplo: 0.001 = 0.1 % = 10 bps.
        """
        ...

    # Hook opcional por si un modelo necesita inicializarse con datos
    # de contexto (p.ej. pre-calcular ADV por símbolo). Default no-op.
    def prepare(self, df: pd.DataFrame) -> None:
        """Permite al modelo pre-computar estadísticos del DF (ADV, etc.)."""
        return None


# ════════════════════════════════════════════════════════════════════
#  Implementación 1: FlatCostModel (retrocompatible, el de siempre).
# ════════════════════════════════════════════════════════════════════
class FlatCostModel(CostModel):
    """
    Modelo de costes constante — el mismo que tenía el motor antes
    de la fase I. Útil para backtests rápidos y para comprobar que los
    números no cambian al migrar al nuevo sistema.

    - `commission_pct`: porcentaje sobre el notional (0.1 = 0.1 %).
    - `slippage_pct`  : porcentaje sobre el mid (0.05 = 0.05 %).

    La conversión interna es a bps multiplicando por 100. Esto mantiene
    la compatibilidad con los flags antiguos `--commission-pct` y
    `--slippage-pct` del CLI sin tocar nada.
    """

    name = "flat"

    def __init__(self, commission_pct: float = 0.1, slippage_pct: float = 0.05):
        # Guardamos en fracciones (0.001 = 0.1 %) para aritmética directa.
        self.commission = commission_pct / 100.0
        self.slippage = slippage_pct / 100.0

    def apply(
        self,
        mid: float,
        side: str,
        notional: float,
        bar_time: Optional[pd.Timestamp] = None,
        adv_notional: Optional[float] = None,
        is_daily_bar: bool = True,
    ) -> CostBreakdown:
        # Slippage direccional: BUY paga más, SELL recibe menos.
        sign = +1.0 if side.upper() == "BUY" else -1.0
        fill_price = mid * (1.0 + sign * self.slippage)

        # Slippage "pagado" en $ = |fill - mid| * qty. Aproximamos qty con
        # notional / mid (antes de slippage) para mantener reporting en bps
        # consistente con el motor.
        qty_est = notional / mid if mid > 0 else 0.0
        slippage_paid = abs(fill_price - mid) * qty_est
        commission_paid = notional * self.commission

        total_bps = (
            ((commission_paid + slippage_paid) / notional) * 1e4
            if notional > 0
            else 0.0
        )
        return CostBreakdown(
            fill_price=fill_price,
            commission_paid=commission_paid,
            slippage_paid=slippage_paid,
            total_cost_bps=total_bps,
        )

    def commission_rate(self) -> float:
        """Fracción de comisión (retro-compat con self.commission)."""
        return self.commission


# ════════════════════════════════════════════════════════════════════
#  Implementación 2: RealisticCostModel.
# ════════════════════════════════════════════════════════════════════
class RealisticCostModel(CostModel):
    """
    Cost model con spread variable por hora del día + market impact.

    Pasos al ejecutar `apply(...)`:

    1. **Factor horario** del spread:
         · si `bar_time` no cae en sesión US (crypto, Asia, extended) →
           `overnight_mult`.
         · si datos daily (`is_daily_bar=True`) → `daily_bar_mult`.
         · si intradía dentro de sesión:
             09:30–10:30 → `hour_open_mult`
             10:30–15:00 → `hour_mid_mult`
             15:00–16:00 → `hour_close_mult`
         Multiplica al `spread_bps` base.

    2. **Market impact** (si `adv_notional` disponible y > 0):
         impact_bps = impact_coef * 10000 * sqrt(notional / adv_notional)
         Es la forma "raíz cuadrada" clásica (Almgren-Chriss, Kyle).
         Para participation=1% del ADV y coef=0.1 → ≈10 bps.

    3. **Slippage total** en bps = spread_bps_effective/2 + impact_bps.
         El `/2` es porque al cruzar el spread sólo pagas medio spread
         (del mid al ask o del mid al bid).

    4. **Fill price** = mid * (1 ± slippage_total).

    5. **Comisión** = notional * (commission_bps/1e4).

    6. **Total coste bps** = commission_bps + slippage_total.
    """

    name = "realistic"

    def __init__(self, params: Optional[CostParams] = None):
        self.p = params or CostParams()
        # Cache opcional de ADV por símbolo (poblada por `prepare`).
        self._adv_cache: dict[str, float] = {}

    # ----- Factor horario -------------------------------------------------
    def _hour_factor(self, bar_time: Optional[pd.Timestamp], is_daily_bar: bool) -> float:
        """Devuelve el multiplicador del spread según la hora / tipo de barra."""
        p = self.p
        # Bar diaria → usamos multiplicador promedio (no sabemos la hora).
        if is_daily_bar or bar_time is None:
            return p.daily_bar_mult

        # Normalizamos a hora UTC. El horario US regular es 14:30–21:00 UTC
        # (09:30–16:00 NY) fuera de DST, y 13:30–20:00 UTC en DST. Para no
        # depender de tzinfo concreto, usamos una aproximación robusta:
        # si la hora UTC cae entre 13:30 y 21:00, es sesión US.
        try:
            if bar_time.tzinfo is None:
                # Si no hay tz, asumimos UTC para el cálculo (no tenemos
                # mejor información). Mejor ser consistente que extrapolar.
                t = bar_time.tz_localize("UTC")
            else:
                t = bar_time.tz_convert("UTC")
        except Exception:
            # tz inválido → fallback a daily.
            return p.daily_bar_mult

        hour = t.hour + t.minute / 60.0
        # US Equities: 13:30–21:00 UTC (aprox, ignorando DST; con DST sería
        # 12:30–20:00). Un rango amplio 13:00–21:00 es tolerante con ambos.
        if 13.0 <= hour < 14.5:       # primera hora
            return p.hour_open_mult
        if 14.5 <= hour < 20.0:       # medio día
            return p.hour_mid_mult
        if 20.0 <= hour < 21.0:       # última hora
            return p.hour_close_mult
        return p.overnight_mult

    # ----- Market impact --------------------------------------------------
    def _impact_bps(self, notional: float, adv_notional: Optional[float]) -> float:
        """Impacto en bps según regla raíz cuadrática simplificada."""
        if not self.p.include_impact or adv_notional is None or adv_notional <= 0:
            return 0.0
        participation = max(0.0, notional / adv_notional)
        # sqrt(participation) * coef * 10_000 → resultado en bps.
        return self.p.impact_coef * 1e4 * float(np.sqrt(participation))

    # ----- API principal --------------------------------------------------
    def apply(
        self,
        mid: float,
        side: str,
        notional: float,
        bar_time: Optional[pd.Timestamp] = None,
        adv_notional: Optional[float] = None,
        is_daily_bar: bool = True,
    ) -> CostBreakdown:
        p = self.p

        # 1) Spread efectivo (en bps) según hora.
        hour_factor = self._hour_factor(bar_time, is_daily_bar)
        spread_bps_eff = p.spread_bps * hour_factor

        # 2) Impact (bps).
        impact_bps = self._impact_bps(notional, adv_notional)

        # 3) Slippage total (bps). Cruzamos medio spread + impact.
        slippage_bps = spread_bps_eff * 0.5 + impact_bps

        # 4) Fill price: BUY paga slip positivo; SELL recibe slip negativo.
        sign = +1.0 if side.upper() == "BUY" else -1.0
        fill_price = mid * (1.0 + sign * slippage_bps / 1e4)

        # 5) Comisión y slippage pagados en $.
        commission_paid = notional * (p.commission_bps / 1e4)
        qty_est = notional / mid if mid > 0 else 0.0
        slippage_paid = abs(fill_price - mid) * qty_est

        total_bps = p.commission_bps + slippage_bps
        return CostBreakdown(
            fill_price=fill_price,
            commission_paid=commission_paid,
            slippage_paid=slippage_paid,
            total_cost_bps=total_bps,
        )

    def commission_rate(self) -> float:
        """Fracción de comisión (bps / 1e4)."""
        return self.p.commission_bps / 1e4


# ════════════════════════════════════════════════════════════════════
#  Helper: ADV (Average Daily Volume en $) de un DataFrame.
# ════════════════════════════════════════════════════════════════════
def compute_adv_notional(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """
    Calcula el ADV (average daily volume) en términos monetarios.

        adv_notional[t] = mean_{k=t-w..t-1}(close[k] * volume[k])

    Se usa `close * volume` como proxy del notional diario del símbolo.
    La ventana por defecto es 20 barras, lo estándar en literatura
    (≈1 mes bursátil). La primera ventana es NaN y el motor debe
    tratarlo como "no disponible → sin impact".

    Importante: usamos los valores HASTA `t-1` (shift), no hasta `t`,
    para que el cálculo sea honesto en el backtest (no se conoce el
    close del día t cuando se ejecuta al open del día t).
    """
    if "close" not in df.columns or "volume" not in df.columns:
        return pd.Series(dtype=float, index=df.index)
    dollar_volume = (df["close"] * df["volume"]).astype(float)
    # Ventana rolling, excluyendo la barra actual (shift(1)).
    return dollar_volume.rolling(window=window, min_periods=max(5, window // 2)).mean().shift(1)


# ════════════════════════════════════════════════════════════════════
#  Factoría: construir un CostModel desde flags del CLI.
# ════════════════════════════════════════════════════════════════════
def build_cost_model(
    kind: str = "flat",
    commission_pct: float = 0.1,
    slippage_pct: float = 0.05,
    commission_bps: Optional[float] = None,
    spread_bps: Optional[float] = None,
    impact_coef: Optional[float] = None,
) -> CostModel:
    """
    Devuelve el `CostModel` adecuado según `kind`.

    Args:
        kind: "flat" (retro-compatible) o "realistic".
        commission_pct, slippage_pct: usados si kind="flat".
        commission_bps, spread_bps, impact_coef: overrides opcionales
            para el modelo realista (todos aplican `CostParams` defaults
            si se pasan None).

    Razonamiento de los defaults:
        · commission_bps=1  → Alpaca / IBKR Pro retail (aprox).
        · spread_bps=4      → mid entre equities líquidos (1-3) y algo
                              más defensivo para portfolios mixtos.
        · impact_coef=0.1   → literatura Almgren-Chriss para equities US.
    """
    kind = (kind or "flat").lower()
    if kind == "realistic":
        params = CostParams(
            commission_bps=commission_bps if commission_bps is not None else 1.0,
            spread_bps=spread_bps if spread_bps is not None else 4.0,
            impact_coef=impact_coef if impact_coef is not None else 0.1,
        )
        return RealisticCostModel(params)

    # Default / "flat"
    return FlatCostModel(commission_pct=commission_pct, slippage_pct=slippage_pct)
