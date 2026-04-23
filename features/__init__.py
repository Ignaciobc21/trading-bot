"""
features — pipeline de ingeniería de características para ML.

El `FeatureBuilder` produce un DataFrame tabular amplio con una fila por
barra y columnas por feature, listo para entrenar modelos supervisados en
fase P4 (meta-labeling, triple-barrier, etc.).

Convención:
- Todas las features calculadas en el cierre de la barra `t` usan solo
  información disponible en ese momento (no look-ahead).
- Los nombres de columna son snake_case.
- Las NaN iniciales por ventanas rolling se conservan — el caller decide
  si dropna() o imputar.

Versionado:
- `FEATURE_VERSION` se incrementa cuando el conjunto de features cambia,
  invalidando la cache.
"""

from features.builder import FeatureBuilder, FEATURE_VERSION  # noqa: F401
from features.cache import FeatureCache  # noqa: F401
