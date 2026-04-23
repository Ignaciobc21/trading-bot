"""
meta_labels.py — Construcción de meta-etiquetas para filtrado ML de señales.

El concepto (López de Prado) es:
    - La estrategia rule-based produce señales BUY/SELL/HOLD.
    - Para cada BUY, queremos predecir **si ese trade concreto habría sido
      ganador** aplicando triple-barrier desde ese bar.
    - El modelo no reemplaza la estrategia; sólo la filtra. Acepta el BUY
      con prob > threshold, lo rechaza si no.

Esto reduce el espacio de aprendizaje: el modelo no tiene que elegir qué
hacer cada barra, solo validar eventos muy concretos. Mucho más fácil de
aprender con datos limitados y mucho más robusto out-of-sample.
"""

from __future__ import annotations

from typing import Tuple

import pandas as pd

from labels.triple_barrier import TripleBarrierConfig, triple_barrier_labels
from strategies.base import Action


def build_meta_labels(
    df: pd.DataFrame,
    signals: pd.Series,
    tb_config: TripleBarrierConfig | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Construye el dataset de meta-etiquetas a partir de un DataFrame OHLCV y
    una serie de acciones (BUY/SELL/HOLD) emitida por la estrategia base.

    Args:
        df:      OHLCV con índice datetime.
        signals: Series indexada como `df` con valores Action (BUY/SELL/HOLD).
        tb_config: configuración de la triple-barrier.

    Returns:
        (tb_df, meta_df):
            tb_df   : DataFrame completo con la salida de `triple_barrier_labels`.
            meta_df : subset filtrado por señales BUY con columnas:
                      - meta : 1 si el trade fue ganador (+1), 0 si no (-1 o 0).
                      - ret  : retorno nominal del trade hipotético.
                      - entry_bar, exit_bar, entry_price, exit_price.
                      El índice conserva el timestamp de la barra de la señal.
    """
    cfg = tb_config or TripleBarrierConfig()
    tb = triple_barrier_labels(df, config=cfg)

    buy_mask = signals == Action.BUY
    meta = tb.loc[buy_mask].copy()
    meta["meta"] = (meta["label"] == 1).astype(int)
    meta["entry_bar"] = meta.index
    # Convertir el offset numérico de exit_bar a datetime si es posible.
    if len(df):
        idx_array = df.index
        exit_ts = idx_array[meta["exit_bar"].clip(0, len(idx_array) - 1)]
        meta["exit_ts"] = exit_ts
    return tb, meta[["meta", "ret", "entry_price", "exit_price", "exit_bar", "exit_ts"]]
