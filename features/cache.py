"""
cache.py — Cache en disco de DataFrames de features.

Usa parquet si `pyarrow`/`fastparquet` está disponible (más rápido, más
pequeño) y se cae a pickle comprimido si no. La clave de cache es
`{symbol}_{interval}_{period}_{feature_version}` para invalidar
automáticamente cuando se cambia el set de features.
"""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from typing import Optional

import pandas as pd

from features.builder import FEATURE_VERSION

logger = logging.getLogger(__name__)


def _has_parquet() -> bool:
    try:
        import pyarrow  # noqa: F401
        return True
    except ImportError:
        pass
    try:
        import fastparquet  # noqa: F401
        return True
    except ImportError:
        return False


class FeatureCache:
    """
    Cache en disco para DataFrames de features, indexados por (symbol,
    interval, period, feature_version).

    Args:
        base_dir: raíz de la cache. Por defecto, `~/.cache/trading-bot/features`.
    """

    def __init__(self, base_dir: Optional[str] = None):
        if base_dir is None:
            base_dir = os.path.join(str(Path.home()), ".cache", "trading-bot", "features")
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._format = "parquet" if _has_parquet() else "pickle"

    # ──────────────────────────────────────────
    def _key(self, symbol: str, interval: str, period: str, extra: str = "") -> str:
        raw = f"{symbol}|{interval}|{period}|{FEATURE_VERSION}|{extra}"
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]

    def _path(self, symbol: str, interval: str, period: str, extra: str = "") -> Path:
        suffix = "parquet" if self._format == "parquet" else "pkl.gz"
        name = f"{symbol}_{interval}_{period}_{FEATURE_VERSION}_{self._key(symbol, interval, period, extra)}.{suffix}"
        # Sanitizar el nombre por si el symbol contiene "/", "=", etc.
        safe = name.replace("/", "_").replace("=", "_")
        return self.base_dir / safe

    # ──────────────────────────────────────────
    def load(
        self, symbol: str, interval: str, period: str, extra: str = ""
    ) -> Optional[pd.DataFrame]:
        path = self._path(symbol, interval, period, extra)
        if not path.exists():
            return None
        try:
            if self._format == "parquet":
                return pd.read_parquet(path)
            return pd.read_pickle(path, compression="gzip")
        except Exception as exc:  # cache corrupta → la ignoramos.
            logger.warning("FeatureCache: no se pudo leer %s (%s) — se ignora.", path, exc)
            return None

    def save(
        self,
        df: pd.DataFrame,
        symbol: str,
        interval: str,
        period: str,
        extra: str = "",
    ) -> Path:
        path = self._path(symbol, interval, period, extra)
        if self._format == "parquet":
            df.to_parquet(path)
        else:
            df.to_pickle(path, compression="gzip")
        logger.info("FeatureCache: guardado %s (%s, %d filas × %d cols)",
                    path.name, self._format, len(df), df.shape[1])
        return path

    # ──────────────────────────────────────────
    def get_or_build(
        self,
        symbol: str,
        interval: str,
        period: str,
        build_fn,
        extra: str = "",
    ) -> pd.DataFrame:
        cached = self.load(symbol, interval, period, extra)
        if cached is not None:
            logger.info("FeatureCache: hit (%s %s %s)", symbol, interval, period)
            return cached
        logger.info("FeatureCache: miss (%s %s %s) — construyendo", symbol, interval, period)
        df = build_fn()
        self.save(df, symbol, interval, period, extra)
        return df
