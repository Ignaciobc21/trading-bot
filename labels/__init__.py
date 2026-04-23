"""
labels — Etiquetado de barras para aprendizaje supervisado.

Módulos:
    triple_barrier : etiquetas ±1/0 tipo López de Prado (TP, SL, timeout).
    meta_labels    : meta-etiquetas binarias para filtrar señales de una estrategia rule-based.
"""

from labels.triple_barrier import (  # noqa: F401
    TripleBarrierConfig,
    triple_barrier_labels,
)
from labels.meta_labels import build_meta_labels  # noqa: F401
