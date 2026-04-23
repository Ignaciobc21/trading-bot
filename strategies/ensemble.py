"""
ensemble.py — Estrategia híbrida: router por régimen.

Combina un detector de régimen con varias sub-estrategias:
    - Sub-estrategia de trend-following  → se ejecuta en régimen TREND_UP.
    - Sub-estrategia de mean-reversion   → se ejecuta en régimen MEAN_REVERT.
    - Cualquier otro régimen (CHOP, TREND_DOWN) → HOLD.

Las señales de las sub-estrategias se filtran de forma que sólo producen
BUY cuando el régimen lo permite. Los SELL siempre pasan (queremos poder
salir aunque el régimen cambie mientras estamos dentro de una posición).

Esta clase hereda de BaseStrategy e implementa `generate_signals` para que
funcione directamente con el BacktestEngine vectorizado.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from strategies.base import BaseStrategy, StrategySignal, Action
from strategies.regime import Regime, RegimeDetector


class EnsembleStrategy(BaseStrategy):
    """Router por régimen sobre una sub-estrategia de trend y otra de MR."""

    def __init__(
        self,
        trend_strategy: BaseStrategy,
        mean_revert_strategy: BaseStrategy,
        detector: Optional[RegimeDetector] = None,
        allow_trend_down: bool = False,
        allow_mr_in_chop: bool = True,
        allow_trend_in_chop: bool = True,
        trend_chop_ema_period: int = 50,
    ):
        self.trend_strategy = trend_strategy
        self.mean_revert_strategy = mean_revert_strategy
        self.detector = detector or RegimeDetector()
        self.allow_trend_down = allow_trend_down
        self.allow_mr_in_chop = allow_mr_in_chop
        self.allow_trend_in_chop = allow_trend_in_chop
        self.trend_chop_ema_period = trend_chop_ema_period
        self.name = (
            f"Ensemble[{trend_strategy.name} | {mean_revert_strategy.name}]"
        )
        # Cache del último diagnóstico para inspección externa.
        self._last_regime: Optional[pd.Series] = None

    # ──────────────────────────────────────────
    # Señales vectorizadas
    # ──────────────────────────────────────────
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        regime = self.detector.classify(df)
        self._last_regime = regime

        trend_actions = self.trend_strategy.generate_signals(df)
        mr_actions = self.mean_revert_strategy.generate_signals(df)

        in_trend_up = regime == Regime.TREND_UP
        in_mr = regime == Regime.MEAN_REVERT
        in_chop = regime == Regime.CHOP
        in_trend_down = regime == Regime.TREND_DOWN

        actions = pd.Series(Action.HOLD, index=df.index, name="action", dtype=object)

        # EMA auxiliar: habilita que la sub-estrategia trend opere en CHOP cuando
        # la pendiente de la EMA es positiva (captura tendencias suaves que no
        # llegan a ADX>20).
        ema_rising = (
            df["close"].ewm(span=self.trend_chop_ema_period, adjust=False).mean().diff() > 0
        ).fillna(False)

        # ── BUY: sólo cuando el régimen permite operar la lógica correspondiente.
        trend_chop_mask = (in_chop & ema_rising) if self.allow_trend_in_chop else pd.Series(False, index=df.index)
        trend_allowed = in_trend_up | trend_chop_mask
        trend_buy = (trend_actions == Action.BUY) & trend_allowed

        mr_allowed = in_mr | (in_chop if self.allow_mr_in_chop else pd.Series(False, index=df.index))
        mr_buy = (mr_actions == Action.BUY) & mr_allowed

        # ── SELL primero (se aplican salidas de ambas sub-estrategias). Luego
        # BUY: si un mismo bar tiene señal de entrada y de salida, preferimos
        # entrar (la salida llegará en el siguiente cruce).
        trend_sell = trend_actions == Action.SELL
        mr_sell = mr_actions == Action.SELL
        actions[trend_sell | mr_sell] = Action.SELL

        # Régimen TREND_DOWN: cierre forzoso si lo pide el usuario.
        if not self.allow_trend_down:
            actions[in_trend_down] = Action.SELL

        # BUY tiene prioridad en caso de empate.
        actions[trend_buy] = Action.BUY
        actions[mr_buy] = Action.BUY

        return actions

    # ──────────────────────────────────────────
    # Señal puntual (modo live)
    # ──────────────────────────────────────────
    def generate_signal(self, df: pd.DataFrame) -> StrategySignal:
        action = self.generate_signals(df).iloc[-1]
        regime = self._last_regime.iloc[-1] if self._last_regime is not None else Regime.CHOP
        price = float(df["close"].iloc[-1])
        if action == Action.BUY:
            src = self.trend_strategy.name if regime == Regime.TREND_UP else self.mean_revert_strategy.name
            return StrategySignal(Action.BUY, confidence=0.75, price=price,
                                  reason=f"Ensemble BUY vía '{src}' en {regime.value}")
        if action == Action.SELL:
            return StrategySignal(Action.SELL, confidence=0.75, price=price,
                                  reason=f"Ensemble SELL (regime={regime.value if hasattr(regime,'value') else regime})")
        return StrategySignal(Action.HOLD, confidence=0.0, price=price,
                              reason=f"HOLD — regime={regime.value if hasattr(regime,'value') else regime}")

    # ──────────────────────────────────────────
    # Inspección
    # ──────────────────────────────────────────
    def regime_series(self) -> Optional[pd.Series]:
        """Devuelve la serie de régimen calculada en la última llamada a generate_signals()."""
        return self._last_regime


# ──────────────────────────────────────────────
# Factory con defaults razonables
# ──────────────────────────────────────────────
def build_default_ensemble(
    detector: Optional[RegimeDetector] = None,
    allow_trend_down: bool = False,
) -> EnsembleStrategy:
    """Construye un ensemble con Donchian trend + Connors RSI(2) MR."""
    from strategies.donchian_trend import DonchianTrendStrategy
    from strategies.rsi2_mean_reversion import RSI2MeanReversionStrategy

    return EnsembleStrategy(
        trend_strategy=DonchianTrendStrategy(entry_lookback=20, exit_lookback=10),
        mean_revert_strategy=RSI2MeanReversionStrategy(),
        detector=detector,
        allow_trend_down=allow_trend_down,
    )
