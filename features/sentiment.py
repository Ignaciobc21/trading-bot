"""
sentiment.py — Construcción de features de sentimiento por barra (B).

Diseño:
    Este módulo añade al pipeline de ML features derivadas de fuentes de
    sentimiento externas. El meta-labeler las consume como cualquier otra
    columna numérica: si están NaN para una barra, LightGBM las maneja
    nativamente (sin tener que imputar valores artificiales).

Fuentes implementadas (sin necesidad de API keys):
    1. **VADER sobre noticias de yfinance**
       - VADER (Valence Aware Dictionary for Sentiment Reasoning) es un
         scorer basado en lexicón muy ligero (~5 MB), sin GPU, sin modelo
         que descargar.  Devuelve un score "compound" en [-1, +1] que es
         lo bastante bueno para titulares financieros en inglés.
       - yfinance expone `Ticker(sym).news` con noticias **recientes**
         (últimos días/semanas). Sin historia profunda → en backtests de
         varios años las primeras barras tendrán NaN; en live es cuando
         más aporta.

    2. **Crypto Fear & Greed Index** (alternative.me)
       - Endpoint público, sin auth: https://api.alternative.me/fng/
       - Granularidad diaria, historia desde 2018. Útil para BTC-USD y
         como proxy general de risk-on / risk-off para acciones.
       - Para tickers no-crypto se usa también: el F&G refleja apetito al
         riesgo del mercado en general, así que correlaciona con la
         direccionalidad de toda la renta variable.

Fuentes NO implementadas (puntos de extensión):
    - **FinBERT** (HuggingFace). Modelo BERT pre-entrenado para finanzas;
      ~500 MB, requiere `transformers` y `torch`. Mejor que VADER en
      titulares ambiguos. Para activarlo bastaría con un nuevo
      `_score_with_finbert` y un flag de configuración.
    - **Alpaca News REST API**. Tiene historia de varios años pero
      requiere las credenciales del broker. Implementación sería un
      `AlpacaNewsFetcher` que rellena el mismo campo que
      `_fetch_yfinance_news`.
    - **Reddit mention velocity** (PRAW). Necesita una app de Reddit
      gratis (~2 min en https://www.reddit.com/prefs/apps). Útil para
      detectar saturación retail (squeezes en small-caps).

Cache en disco:
    - Se persisten las consultas de yfinance y Fear&Greed bajo
      `~/.cache/trading-bot/sentiment/<symbol>.parquet` y
      `~/.cache/trading-bot/sentiment/_fng.parquet`.
    - TTL implícito: el caller decide cuándo refrescar borrando el
      archivo. Para backtests de 5y la cache es suficiente; para live
      conviene refrescar la del símbolo en cada iteración (lo hace el
      LiveRunner).

Features producidas (todas alineadas al índice del DataFrame OHLCV):
    - news_sent_24h_mean   : media del compound de noticias en últimas 24h.
    - news_sent_24h_count  : nº de noticias en últimas 24h.
    - news_sent_3d_mean    : media del compound últimas 72h (suaviza ruido).
    - news_sent_decay      : exp-weighted con half-life ~24h.
    - news_pos_ratio_3d    : fracción de noticias con compound > 0.05 en 3d.
    - news_neg_ratio_3d    : fracción de noticias con compound < -0.05 en 3d.
    - fng_value            : Fear & Greed Index 0-100 (interpolado al timestamp
                             de la barra, NaN para fechas previas a 2018).
    - fng_class_idx        : 0=Extreme Fear ... 4=Extreme Greed (categórico
                             como entero).

Las barras sin cobertura devuelven NaN en cualquiera de estas columnas.
LightGBM trata NaN como una rama explícita del árbol — no rompe nada y
permite al modelo aprender que "no hay noticias" puede ser información en
sí misma.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ──────────────────────────────────────────────
# Constantes / paths
# ──────────────────────────────────────────────
# Cache en disco. ~/.cache/ es el sitio estándar XDG en Linux/macOS y
# funciona bien también en Windows (usa el HOME del usuario).
_CACHE_DIR = Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache"))) / "trading-bot" / "sentiment"

# Endpoints públicos (sin auth).
_FNG_URL = "https://api.alternative.me/fng/?limit=0&format=json"

# Umbrales VADER para clasificar polaridad. Convención del autor de VADER:
# compound > 0.05 → positivo, < -0.05 → negativo, en medio → neutro.
_VADER_POS_THR = 0.05
_VADER_NEG_THR = -0.05

# Mapeo de la clasificación textual del Fear & Greed a un índice ordinal.
# Útil para usarla como feature numérica además del valor 0-100.
_FNG_CLASS_TO_IDX = {
    "Extreme Fear": 0,
    "Fear": 1,
    "Neutral": 2,
    "Greed": 3,
    "Extreme Greed": 4,
}


# ──────────────────────────────────────────────
# Utilidades
# ──────────────────────────────────────────────
def _ensure_cache_dir() -> None:
    """Crea el directorio de cache si no existe (idempotente)."""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _strip_html(text: str) -> str:
    """
    Elimina etiquetas HTML básicas y entidades comunes. yfinance devuelve
    descripciones con `<a href=...>` y `&nbsp;` que rompen un poco el
    scoring de VADER (no entiende HTML, pero las etiquetas hacen ruido).
    """
    if not text:
        return ""
    # Sustitución muy simple — para textos cortos es suficiente y evita
    # añadir BeautifulSoup como dependencia.
    import re
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.replace("&nbsp;", " ").replace("&amp;", "&")
    return re.sub(r"\s+", " ", text).strip()


# ──────────────────────────────────────────────
# Lazy loader de VADER
# ──────────────────────────────────────────────
class _VaderHolder:
    """Singleton perezoso para no importar VADER si nadie lo usa."""

    _instance = None

    @classmethod
    def get(cls):
        if cls._instance is None:
            try:
                # Import dentro del método para que el módulo siga
                # importable aunque el paquete no esté instalado (en ese
                # caso, las features de news devuelven NaN).
                from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
                cls._instance = SentimentIntensityAnalyzer()
            except Exception:  # noqa: BLE001
                cls._instance = False  # marcador de "no disponible".
        return cls._instance


def _vader_available() -> bool:
    """True si VADER está cargado y funcional."""
    return bool(_VaderHolder.get())


def _score_text(text: str) -> Optional[float]:
    """
    Score VADER 'compound' ∈ [-1, +1] para un texto. None si VADER no está
    instalado o el texto está vacío. Para titulares y descripciones cortas
    funciona razonablemente bien sin tuning.
    """
    if not text:
        return None
    s = _VaderHolder.get()
    if not s:
        return None
    return float(s.polarity_scores(text)["compound"])


# ──────────────────────────────────────────────
# News fetcher (yfinance) + cache
# ──────────────────────────────────────────────
def _news_cache_path(symbol: str) -> Path:
    """Path del parquet de noticias cacheadas para `symbol`."""
    return _CACHE_DIR / f"news_{symbol.upper().replace('-', '_')}.parquet"


def _load_news_cache(symbol: str) -> pd.DataFrame:
    """Carga la cache de noticias del símbolo (vacía si no existe)."""
    path = _news_cache_path(symbol)
    if not path.exists():
        return pd.DataFrame(columns=["ts", "title", "summary", "compound"])
    try:
        return pd.read_parquet(path)
    except Exception:  # noqa: BLE001
        return pd.DataFrame(columns=["ts", "title", "summary", "compound"])


def _save_news_cache(symbol: str, df: pd.DataFrame) -> None:
    """Persiste el DataFrame (deduplicado y ordenado) a parquet."""
    _ensure_cache_dir()
    if df.empty:
        return
    df = df.drop_duplicates(subset=["ts", "title"]).sort_values("ts")
    df.to_parquet(_news_cache_path(symbol), index=False)


def fetch_news_yf(symbol: str, refresh: bool = True) -> pd.DataFrame:
    """
    Descarga noticias recientes para `symbol` desde yfinance, las puntúa
    con VADER y las guarda incrementalmente en cache.

    Args:
        symbol  : ticker (ej. "AAPL"). Para crypto-like ("BTC-USD") yfinance
                  también lo acepta y devuelve su feed de noticias.
        refresh : si True, intenta golpear yfinance y mergear con la cache.
                  Si False, devuelve sólo lo que ya está cacheado.

    Returns:
        DataFrame con columnas: ts (UTC tz-aware), title, summary, compound.
        Vacío si no hay datos disponibles.
    """
    cached = _load_news_cache(symbol)
    if not refresh:
        return cached

    fresh_rows: List[Dict] = []
    try:
        import yfinance as yf
        t = yf.Ticker(symbol)
        # yfinance ha cambiado el shape varias veces. En 2024+ devuelve
        # una lista de dicts con clave 'content' embebida; la ruta robusta
        # es introspectar y caer a defaults sensatos.
        items = t.news or []
    except Exception:  # noqa: BLE001
        items = []

    for it in items:
        # Normalizar entre versiones del SDK.
        content = it.get("content") if isinstance(it, dict) else None
        if isinstance(content, dict):
            title = content.get("title") or ""
            summary = _strip_html(content.get("summary") or content.get("description") or "")
            ts_raw = content.get("pubDate") or content.get("displayTime")
        else:
            title = (it.get("title") or "") if isinstance(it, dict) else ""
            summary = (it.get("summary") or it.get("description") or "") if isinstance(it, dict) else ""
            ts_raw = it.get("providerPublishTime") if isinstance(it, dict) else None

        # Parseo del timestamp en formato unificado UTC.
        ts = pd.NaT
        if isinstance(ts_raw, (int, float)):
            ts = pd.to_datetime(int(ts_raw), unit="s", utc=True)
        elif isinstance(ts_raw, str) and ts_raw:
            try:
                ts = pd.to_datetime(ts_raw, utc=True)
            except Exception:  # noqa: BLE001
                ts = pd.NaT

        if pd.isna(ts) or not title:
            continue

        # Score concatenando título + resumen (titulares solos pierden
        # contexto; el summary añade ~50-150 palabras).
        text = (title + ". " + summary).strip(". ")
        score = _score_text(text)
        if score is None:
            continue

        fresh_rows.append({
            "ts": ts,
            "title": title,
            "summary": summary[:500],  # capamos para no inflar la cache.
            "compound": score,
        })

    fresh = pd.DataFrame(fresh_rows)
    if cached.empty and fresh.empty:
        return cached
    merged = pd.concat([cached, fresh], ignore_index=True) if not cached.empty else fresh
    _save_news_cache(symbol, merged)
    return merged


# ──────────────────────────────────────────────
# Fear & Greed Index (crypto, alternative.me)
# ──────────────────────────────────────────────
_FNG_CACHE_PATH = _CACHE_DIR / "_fng.parquet"


def fetch_fear_greed(refresh: bool = True) -> pd.DataFrame:
    """
    Descarga la serie histórica completa del Fear & Greed Index.

    El endpoint `?limit=0` devuelve toda la historia disponible (desde
    2018-02). La granularidad es diaria.

    Args:
        refresh: si True, intenta refrescar; si la red falla devuelve la
                 cache.

    Returns:
        DataFrame con columnas: date (UTC tz-aware), value (int 0-100),
        class_idx (int 0..4). Ordenado ascendente. Vacío si no hay datos.
    """
    cached = pd.DataFrame(columns=["date", "value", "class_idx"])
    if _FNG_CACHE_PATH.exists():
        try:
            cached = pd.read_parquet(_FNG_CACHE_PATH)
        except Exception:  # noqa: BLE001
            pass

    if not refresh and not cached.empty:
        return cached

    try:
        import requests
        resp = requests.get(_FNG_URL, timeout=10)
        resp.raise_for_status()
        data = resp.json().get("data") or []
    except Exception:  # noqa: BLE001
        # Sin red: nos quedamos con lo que haya en cache.
        return cached

    rows = []
    for d in data:
        try:
            ts = pd.to_datetime(int(d["timestamp"]), unit="s", utc=True).normalize()
            value = int(d["value"])
            cls = _FNG_CLASS_TO_IDX.get(d.get("value_classification", ""), 2)
            rows.append({"date": ts, "value": value, "class_idx": cls})
        except Exception:  # noqa: BLE001
            continue

    fresh = pd.DataFrame(rows).sort_values("date") if rows else pd.DataFrame(columns=cached.columns)
    if fresh.empty:
        return cached

    _ensure_cache_dir()
    fresh.to_parquet(_FNG_CACHE_PATH, index=False)
    return fresh


# ──────────────────────────────────────────────
# Aggregator: ventanas rolling sobre noticias
# ──────────────────────────────────────────────
def _aggregate_news_to_index(
    news: pd.DataFrame,
    target_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    Agrega un DataFrame de noticias (ts, compound) a las barras del índice
    `target_index`. Para cada barra t produce ventanas [t-24h, t),
    [t-72h, t), y un decay exp-weighted con half-life 24h.

    El cálculo se hace por timestamp y NO mira al futuro: cada agregado
    sobre la barra t usa noticias estrictamente anteriores a t. Esto evita
    leakage (no usamos la noticia que se publicó la misma barra que se
    está prediciendo).

    Returns:
        DataFrame con índice = target_index y columnas:
            news_sent_24h_mean, news_sent_24h_count,
            news_sent_3d_mean, news_sent_decay,
            news_pos_ratio_3d, news_neg_ratio_3d.
    """
    cols = [
        "news_sent_24h_mean", "news_sent_24h_count",
        "news_sent_3d_mean", "news_sent_decay",
        "news_pos_ratio_3d", "news_neg_ratio_3d",
    ]
    empty = pd.DataFrame(np.nan, index=target_index, columns=cols)
    if news is None or news.empty:
        return empty

    # Asegurar tz UTC en ambos índices para poder comparar timestamps.
    news = news.dropna(subset=["ts", "compound"]).copy()
    news["ts"] = pd.to_datetime(news["ts"], utc=True)
    news = news.sort_values("ts")

    if target_index.tz is None:
        target_idx_utc = pd.DatetimeIndex(target_index).tz_localize("UTC")
    else:
        target_idx_utc = target_index.tz_convert("UTC")

    # Vectorización con numpy: para cada barra hacemos searchsorted sobre
    # los timestamps ya ordenados de las noticias y agregamos slices.
    # Convertimos timestamps a enteros (ns desde epoch) para evitar
    # mezcla tz-aware / tz-naive en numpy — la comparación numérica es
    # equivalente y mucho más rápida.
    news_ts_ns = news["ts"].astype("int64").to_numpy()  # ns since epoch UTC
    news_sc = news["compound"].to_numpy()

    one_day_ns = int(pd.Timedelta(days=1).value)      # 86_400_000_000_000
    three_days_ns = int(pd.Timedelta(days=3).value)

    target_ns = pd.DatetimeIndex(target_idx_utc).asi8  # int64 ns array

    out: Dict[str, List[float]] = {c: [] for c in cols}
    for t_ns in target_ns:
        # Slice [t-24h, t) — strict less-than en t para no usar futuro.
        lo24 = np.searchsorted(news_ts_ns, t_ns - one_day_ns, side="left")
        hi = np.searchsorted(news_ts_ns, t_ns, side="left")
        slice24 = news_sc[lo24:hi]
        # Slice [t-3d, t)
        lo3 = np.searchsorted(news_ts_ns, t_ns - three_days_ns, side="left")
        slice3 = news_sc[lo3:hi]

        if slice24.size:
            out["news_sent_24h_mean"].append(float(np.mean(slice24)))
            out["news_sent_24h_count"].append(int(slice24.size))
        else:
            out["news_sent_24h_mean"].append(np.nan)
            out["news_sent_24h_count"].append(0)

        if slice3.size:
            out["news_sent_3d_mean"].append(float(np.mean(slice3)))
            out["news_pos_ratio_3d"].append(float(np.mean(slice3 > _VADER_POS_THR)))
            out["news_neg_ratio_3d"].append(float(np.mean(slice3 < _VADER_NEG_THR)))
            # Decay exp-weighted: half-life de 1 día → λ = ln(2)/86400 s.
            # ages_sec = (t - ts_i) en segundos; ns→s ÷ 1e9.
            ages_sec = (t_ns - news_ts_ns[lo3:hi]).astype(np.float64) / 1e9
            weights = np.exp(-np.log(2.0) * ages_sec / 86400.0)
            wsum = weights.sum()
            decay = float(np.sum(slice3 * weights) / wsum) if wsum > 0 else np.nan
            out["news_sent_decay"].append(decay)
        else:
            out["news_sent_3d_mean"].append(np.nan)
            out["news_pos_ratio_3d"].append(np.nan)
            out["news_neg_ratio_3d"].append(np.nan)
            out["news_sent_decay"].append(np.nan)

    df_out = pd.DataFrame(out, index=target_index)
    # `news_sent_24h_count` es un entero "técnico" (nº noticias),
    # interesante como feature por sí mismo (mide intensidad mediática).
    return df_out


def _align_fng_to_index(
    fng: pd.DataFrame, target_index: pd.DatetimeIndex
) -> pd.DataFrame:
    """
    Alinea el Fear&Greed (granularidad diaria) al índice target. Para cada
    barra t usa el valor del F&G del día anterior (shift de 1 día) — así
    no metemos información del día actual antes de cerrarlo.
    """
    cols = ["fng_value", "fng_class_idx"]
    if fng is None or fng.empty:
        return pd.DataFrame(np.nan, index=target_index, columns=cols)

    fng = fng.copy()
    # Forzamos resolución 'ns' en ambos lados; pd.merge_asof exige dtype
    # idéntico en las claves de join.
    fng["date"] = pd.to_datetime(fng["date"], utc=True).dt.normalize().astype("datetime64[ns, UTC]")
    fng = fng.sort_values("date")

    if target_index.tz is None:
        idx_utc = pd.DatetimeIndex(target_index).tz_localize("UTC")
    else:
        idx_utc = target_index.tz_convert("UTC")
    idx_utc = pd.DatetimeIndex(idx_utc).astype("datetime64[ns, UTC]")

    # Por cada barra, encontrar el último valor publicado **antes** de la
    # barra (estricto). pd.merge_asof es perfecto para esto.
    target_df = pd.DataFrame({"_ts": idx_utc}).sort_values("_ts")
    fng_reset = fng.rename(columns={"date": "_ts"}).sort_values("_ts")
    merged = pd.merge_asof(
        target_df, fng_reset, on="_ts", direction="backward", allow_exact_matches=False,
    )
    # Restaurar el índice original (puede no ser monotónico tras el sort).
    merged.index = target_df.index
    merged = merged.sort_index()
    out = merged[["value", "class_idx"]].rename(
        columns={"value": "fng_value", "class_idx": "fng_class_idx"}
    )
    out.index = target_index
    return out


# ──────────────────────────────────────────────
# Builder principal
# ──────────────────────────────────────────────
@dataclass
class SentimentConfig:
    """Configuración del builder de sentimiento."""

    # Si False, ni siquiera intenta golpear yfinance — útil en backtests
    # ofline donde sólo queremos cache pre-existente.
    fetch_news: bool = True
    # Si False, omite el Fear & Greed (más rápido si no se quiere).
    fetch_fng: bool = True
    # Si True, fuerza re-descarga ignorando la cache de noticias.
    refresh_news: bool = True
    # Igual para F&G. La serie diaria es estable, refrescar 1×día basta.
    refresh_fng: bool = True


@dataclass
class SentimentFeatureBuilder:
    """
    Construye features de sentimiento para un símbolo dado.

    Uso típico desde `FeatureBuilder`:
        sb = SentimentFeatureBuilder()
        sent = sb.build(df.index, symbol="AAPL")
        feats = pd.concat([base_feats, sent], axis=1)

    Si VADER no está instalado o la red falla, las columnas devuelven NaN
    y el meta-labeler las trata como features con poca señal.
    """

    config: SentimentConfig = field(default_factory=SentimentConfig)

    def build(self, target_index: pd.DatetimeIndex, symbol: str) -> pd.DataFrame:
        """
        Devuelve un DataFrame indexado por `target_index` con todas las
        columnas de sentimiento. Los timestamps de salida coinciden uno a
        uno con los de entrada — sin shifts adicionales aplicados aquí
        (el shift de "no usar el futuro" ya se aplica internamente).
        """
        if symbol is None:
            # Sin símbolo no hay forma de ir a buscar noticias; devolvemos
            # un DataFrame vacío con las columnas esperadas (todo NaN).
            return self._empty(target_index)

        # ── 1. Noticias VADER ──────────────────────────────────────────
        if self.config.fetch_news and _vader_available():
            news_df = fetch_news_yf(symbol, refresh=self.config.refresh_news)
        else:
            news_df = pd.DataFrame()
        news_feats = _aggregate_news_to_index(news_df, target_index)

        # ── 2. Fear & Greed ────────────────────────────────────────────
        if self.config.fetch_fng:
            fng_df = fetch_fear_greed(refresh=self.config.refresh_fng)
        else:
            fng_df = pd.DataFrame()
        fng_feats = _align_fng_to_index(fng_df, target_index)

        return pd.concat([news_feats, fng_feats], axis=1)

    @staticmethod
    def _empty(target_index: pd.DatetimeIndex) -> pd.DataFrame:
        cols = [
            "news_sent_24h_mean", "news_sent_24h_count",
            "news_sent_3d_mean", "news_sent_decay",
            "news_pos_ratio_3d", "news_neg_ratio_3d",
            "fng_value", "fng_class_idx",
        ]
        return pd.DataFrame(np.nan, index=target_index, columns=cols)
