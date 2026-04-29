"""
triple_barrier.py — Método de las tres barreras (López de Prado).

Para cada barra `t` definimos una ventana de observación de `max_bars`
barras hacia adelante. Dentro de la ventana, el primer evento que
ocurra entre:

    - barrera superior : close >= entry_price * (1 + tp_mult * ATR_t / entry_price)
    - barrera inferior : close <= entry_price * (1 - sl_mult * ATR_t / entry_price)
    - barrera temporal : t + max_bars

Determina la etiqueta:

    +1 si toca primero la barrera superior,
    -1 si toca primero la barrera inferior,
     0 si toca primero la barrera temporal (timeout).

Esta etiqueta es **equivalente al resultado de un trade largo con stops y
take-profit basados en volatilidad**, que es precisamente lo que hará el
bot en live. Por eso es el label "correcto" para entrenar un filtro ML.

Uso:
    >>> cfg = TripleBarrierConfig(tp_mult=2.0, sl_mult=1.0, max_bars=10)
    >>> out = triple_barrier_labels(df, config=cfg)
    >>> out[['label', 'ret', 'exit_bar']].head()
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from strategies.regime import atr


@dataclass
class TripleBarrierConfig:
    tp_mult: float = 2.0   # take-profit en múltiplos de ATR
    sl_mult: float = 1.0   # stop-loss en múltiplos de ATR (positivo → se resta)
    max_bars: int = 10     # barrera temporal en número de barras
    atr_period: int = 14
    # Si True se usa la barra siguiente como entrada (consistente con el motor).
    entry_on_next_open: bool = True


def triple_barrier_labels(df: pd.DataFrame, config: TripleBarrierConfig | None = None) -> pd.DataFrame:
    """
    Genera etiquetas triple-barrier para cada barra del DataFrame.

    Args:
        df: DataFrame con columnas open, high, low, close. Índice datetime.
        config: TripleBarrierConfig, opcional.

    Returns:
        DataFrame con columnas:
            - entry_price  : precio de entrada usado (open de t+1 o close de t).
            - atr          : ATR en la barra de entrada.
            - tp_price     : nivel absoluto de la barrera superior.
            - sl_price     : nivel absoluto de la barrera inferior.
            - exit_bar     : índice relativo (offset) de la barra donde se cerró.
            - exit_price   : precio de cierre efectivo.
            - ret          : retorno del trade (entrada→salida).
            - label        : -1 / 0 / +1.
    """
    cfg = config or TripleBarrierConfig()
    df = df.copy()

    atr_series = atr(df, period=cfg.atr_period)

    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    close = df["close"].to_numpy()
    open_ = df["open"].to_numpy()
    atr_arr = atr_series.to_numpy()

    n = len(df)
    entry_price = np.full(n, np.nan)
    tp_price = np.full(n, np.nan)
    sl_price = np.full(n, np.nan)
    exit_bar = np.full(n, -1, dtype=np.int64)
    exit_price = np.full(n, np.nan)
    label = np.zeros(n, dtype=np.int8)

    # Bucle principal: para cada barra `i` simulamos el trade hipotético
    # "abierto en t o t+1 y cerrado cuando toque alguna de las 3 barreras".
    # Es un bucle secuencial (no vectorizable trivialmente porque la salida
    # de cada trade depende del camino de precios), pero es O(n * max_bars)
    # que para max_bars ≈ 10 y n ≈ 1300 son ~13k iteraciones — trivial.
    for i in range(n):
        # Barra donde "entramos" al trade hipotético. Por defecto
        # entry_on_next_open=True → mimetiza el engine real que ejecuta
        # la orden al open de la siguiente barra (evita look-ahead).
        start = i + 1 if cfg.entry_on_next_open else i
        if start >= n:
            break
        # Si el ATR aún no está calculado (lookback inicial) no etiquetamos.
        # Sin ATR no podemos dimensionar las barreras de forma correcta.
        if np.isnan(atr_arr[i]) or atr_arr[i] <= 0:
            continue

        ep = open_[start] if cfg.entry_on_next_open else close[i]
        a = atr_arr[i]
        # Barreras en unidades absolutas (USD) usando el ATR como "tick"
        # de volatilidad. El mismo tp_mult adapta la barrera a la vol local.
        tp = ep + cfg.tp_mult * a
        sl = ep - cfg.sl_mult * a

        entry_price[i] = ep
        tp_price[i] = tp
        sl_price[i] = sl

        # Barrera temporal: forzamos salida tras max_bars barras incluso si
        # ni TP ni SL han sido tocados. Label=0 en ese caso (timeout).
        end = min(start + cfg.max_bars, n - 1)
        outcome_label = 0
        outcome_bar = end
        outcome_price = close[end]
        for j in range(start, end + 1):
            # Orden de chequeo dentro de UNA MISMA barra:
            #   1º SL (prioridad) → si low toca el stop, asumimos que
            #      cerró en SL. Esto es pesimista (el high puede haber
            #      tocado antes el TP), pero sin datos intra-barra no
            #      podemos saberlo. Es la convención de López de Prado.
            #   2º TP → si high toca take-profit, cerramos en TP.
            # Si ambos se "cumplen" en la misma barra, nos quedamos con
            # el SL (la lectura pesimista hace el modelo más robusto).
            if low[j] <= sl:
                outcome_label = -1
                outcome_bar = j
                outcome_price = sl
                break
            if high[j] >= tp:
                outcome_label = 1
                outcome_bar = j
                outcome_price = tp
                break
        exit_bar[i] = outcome_bar
        exit_price[i] = outcome_price
        label[i] = outcome_label

    # Retorno realizado por cada trade hipotético. Usado en P5 (selección
    # de threshold por Sharpe) y también por quien quiera reportar P&L.
    ret = (exit_price - entry_price) / entry_price

    return pd.DataFrame(
        {
            "entry_price": entry_price,
            "atr": atr_arr,
            "tp_price": tp_price,
            "sl_price": sl_price,
            "exit_bar": exit_bar,
            "exit_price": exit_price,
            "ret": ret,
            "label": label,
        },
        index=df.index,
    )
