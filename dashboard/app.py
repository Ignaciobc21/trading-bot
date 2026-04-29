"""
dashboard/app.py — Dashboard Streamlit (H).

Tres pestañas:

1. **Live Monitor**: lee el `state.json` que escribe `LiveRunner` cada
   iteración (configurable con `--state-path` en `--mode live`) y muestra
   en tiempo real:
     - última acción / precio / proba / multiplicador del overlay,
     - histórico de las últimas N decisiones (tabla),
     - estado de kill-switch / dry-run / data source,
     - sentiment "spot" si el modelo lo expone (Fear & Greed, news_sent).
   Auto-refresh cada `refresh_seconds` segundos.

2. **Backtest Review**: abre un pickle generado con `--save-result`
   y explora:
     - equity curve interactiva (Plotly),
     - tabla de trades (filtrable por símbolo / razón de salida),
     - desglose por símbolo (en runs `kind=portfolio`),
     - resumen textual.

3. **Model Inspection**: abre un pickle de meta-modelo (`models/*.pkl`)
   y muestra:
     - feature importance (barplot),
     - threshold seleccionado, n_features, símbolos del basket,
     - tipo de modelo (global vs regime-split).

Cómo se invoca (desde `main.py`):
    python main.py --mode dashboard \
        --state-path /tmp/live_state.json \
        --save-result /tmp/portfolio_result.pkl \
        --model models/basket25_split_sent.pkl

Los paths configurados se pasan al app vía `argv` después del `--`. El
usuario puede cambiarlos en runtime desde la sidebar.

Dependencias: streamlit + plotly (requirements.txt).
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import streamlit as st


# ──────────────────────────────────────────────
# Args (paths por defecto que vienen de --mode dashboard)
# ──────────────────────────────────────────────
def _parse_argv() -> argparse.Namespace:
    """
    Parsear los argumentos posicionales que `main.py --mode dashboard`
    empuja después del `--`. Todos son opcionales.
    """
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--state-path", default=None)
    p.add_argument("--result-path", default=None)
    p.add_argument("--model-path", default=None)
    args, _ = p.parse_known_args(sys.argv[1:])
    return args


_ARGS = _parse_argv()


# ──────────────────────────────────────────────
# Carga (con cache para no leer ficheros en cada rerun)
# ──────────────────────────────────────────────
@st.cache_data(ttl=3, show_spinner=False)
def _load_state(path: str, mtime: float) -> Optional[Dict[str, Any]]:
    """
    Lee el snapshot JSON del LiveRunner. La key `mtime` se incluye en la
    firma para que cualquier modificación del archivo invalide el cache.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


@st.cache_data(ttl=10, show_spinner=False)
def _load_result(path: str, mtime: float) -> Optional[Dict[str, Any]]:
    """Carga un pickle de BacktestResult / PortfolioResult."""
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except (FileNotFoundError, pickle.UnpicklingError, EOFError):
        return None


@st.cache_data(ttl=30, show_spinner=False)
def _load_model(path: str, mtime: float) -> Optional[Dict[str, Any]]:
    """
    Carga un pickle de meta-modelo y devuelve su payload. Soporta:
      - payload `{"model": ..., "feature_cols": ..., "threshold": ...}`
        (entrenamiento global).
      - payload `{"trend": {...}, "mr": {...}}` (regime-split).
    """
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except (FileNotFoundError, pickle.UnpicklingError, EOFError):
        return None


# ──────────────────────────────────────────────
# UI helpers
# ──────────────────────────────────────────────
def _file_mtime(path: Optional[str]) -> float:
    """mtime del fichero (0 si no existe). Sirve como cache key."""
    if not path:
        return 0.0
    p = Path(path).expanduser()
    return p.stat().st_mtime if p.exists() else 0.0


def _kpi(col, label: str, value: str, help_text: Optional[str] = None) -> None:
    """Wrapper de `st.metric` con estilo consistente."""
    col.metric(label, value, help=help_text)


# ──────────────────────────────────────────────
# Tab: Live Monitor
# ──────────────────────────────────────────────
def render_live_monitor(state_path: Optional[str], refresh_seconds: int) -> None:
    st.subheader("Live Monitor")

    # Path: el usuario puede sobreescribir el del CLI desde la sidebar.
    if not state_path:
        st.info(
            "No hay `state-path` configurado. Lanza el runner con "
            "`--state-path /ruta/state.json` y vuelve aquí."
        )
        return

    mtime = _file_mtime(state_path)
    state = _load_state(state_path, mtime)
    if state is None:
        st.warning(f"No se encuentra `{state_path}` o el JSON está corrupto.")
        return

    # KPIs principales en una fila.
    last = state.get("last_decision", {})
    cols = st.columns(5)
    _kpi(cols[0], "Símbolo", state.get("symbol", "—"))
    _kpi(cols[1], "Última acción", last.get("action", "—"))
    price = last.get("price")
    _kpi(cols[2], "Último precio", f"${price:.2f}" if isinstance(price, (int, float)) else "—")
    proba = last.get("proba")
    _kpi(cols[3], "Proba meta", f"{proba:.3f}" if isinstance(proba, (int, float)) else "—")
    mult = last.get("size_multiplier")
    _kpi(cols[4], "Size mult", f"{mult:.2f}" if isinstance(mult, (int, float)) else "—")

    # Línea de estado.
    cols2 = st.columns(4)
    _kpi(cols2[0], "Iteración", str(state.get("iter_count", "—")))
    _kpi(cols2[1], "Dry-run", "Sí" if state.get("dry_run") else "No")
    _kpi(cols2[2], "Data source", state.get("data_source", "—"))
    _kpi(cols2[3], "Actualizado", state.get("updated_at", "—")[:19])

    # Sentiment spot — sólo si el modelo lo expone.
    sent = state.get("sentiment_spot") or {}
    if sent:
        st.markdown("**Sentiment spot**")
        scols = st.columns(len(sent))
        for i, (k, v) in enumerate(sent.items()):
            _kpi(scols[i], k, f"{v:.2f}" if isinstance(v, (int, float)) else str(v))

    # Posiciones abiertas.
    positions = state.get("open_positions") or []
    st.markdown(f"**Posiciones abiertas** ({len(positions)})")
    if positions:
        st.dataframe(pd.DataFrame(positions), use_container_width=True, hide_index=True)
    else:
        st.caption("Sin posiciones abiertas.")

    # Historial de decisiones — gráfica + tabla.
    history = state.get("history") or []
    if history:
        df_hist = pd.DataFrame(history)
        df_hist["timestamp"] = pd.to_datetime(df_hist["timestamp"], errors="coerce")

        # Curva de precio + marcas BUY/SELL si las hay.
        try:
            import plotly.graph_objects as go
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_hist["timestamp"], y=df_hist["price"],
                mode="lines+markers", name="Precio",
                line={"width": 2},
            ))
            buys = df_hist[df_hist["action"] == "BUY"]
            sells = df_hist[df_hist["action"] == "SELL"]
            if not buys.empty:
                fig.add_trace(go.Scatter(
                    x=buys["timestamp"], y=buys["price"], mode="markers",
                    marker={"symbol": "triangle-up", "color": "green", "size": 12},
                    name="BUY",
                ))
            if not sells.empty:
                fig.add_trace(go.Scatter(
                    x=sells["timestamp"], y=sells["price"], mode="markers",
                    marker={"symbol": "triangle-down", "color": "red", "size": 12},
                    name="SELL",
                ))
            fig.update_layout(
                height=340, margin={"t": 20, "b": 20, "l": 20, "r": 20},
                xaxis_title=None, yaxis_title="Precio",
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as exc:  # plotly puede fallar en envs muy minimal
            st.warning(f"Plotly no disponible: {exc}")

        st.markdown("**Historial reciente**")
        st.dataframe(
            df_hist.sort_values("timestamp", ascending=False).head(50),
            use_container_width=True, hide_index=True,
        )

    # ── K: Drift + auto-retrain status ──
    drift = state.get("drift") or {}
    retrain = state.get("retrain_state") or {}
    if drift or retrain:
        st.markdown("**Drift monitor (K)**")
        dcols = st.columns(5)
        if drift:
            _kpi(dcols[0], "KS %", f"{drift.get('ks_feature_frac', 0)*100:.1f}%")
            _kpi(dcols[1], "PSI %", f"{drift.get('psi_feature_frac', 0)*100:.1f}%")
            auc = drift.get("auc_rolling")
            _kpi(dcols[2], "AUC rolling", f"{auc:.3f}" if isinstance(auc, (int, float)) else "—")
            _kpi(dcols[3], "Should retrain", "SÍ" if drift.get("should_retrain") else "NO")
            _kpi(dcols[4], "Reason", drift.get("reason", "—"))

            # Top features con peor PSI / peor p-value KS.
            worst_psi = drift.get("psi_worst") or []
            worst_ks = drift.get("ks_worst") or []
            if worst_psi:
                st.caption("Top-5 features con mayor PSI (≥0.25 = drift):")
                st.dataframe(
                    pd.DataFrame(worst_psi, columns=["feature", "psi"]).round(3),
                    use_container_width=True, hide_index=True,
                )
            if worst_ks:
                st.caption("Top-5 features con p-value KS más bajo (drift distribucional):")
                st.dataframe(
                    pd.DataFrame(worst_ks, columns=["feature", "p_value"]).round(4),
                    use_container_width=True, hide_index=True,
                )

        if retrain:
            st.markdown("**Retrain orchestrator (K)**")
            rcols = st.columns(4)
            _kpi(rcols[0], "Retrains count", str(retrain.get("retrain_count", 0)))
            _kpi(rcols[1], "Fallidos", str(retrain.get("failed_count", 0)))
            _kpi(rcols[2], "Último retrain",
                 retrain.get("last_retrain_at", "—")[:19] if retrain.get("last_retrain_at") else "—")
            auc_last = retrain.get("last_retrain_auc")
            _kpi(rcols[3], "AUC último",
                 f"{auc_last:.3f}" if isinstance(auc_last, (int, float)) else "—")
            if retrain.get("last_retrain_reason"):
                st.caption(f"Motivo: {retrain['last_retrain_reason']}")

    # Auto-refresh: rerun después de `refresh_seconds`. Sólo se llama
    # cuando esta función es el "render activo" (la página actual la elige
    # por sidebar radio en `main`). Si pusiéramos esto dentro de
    # `st.tabs`, todos los tabs se ejecutarían en cada rerun y el
    # `st.rerun()` aquí abortaría la pintada de los demás tabs.
    if refresh_seconds > 0:
        import time
        time.sleep(refresh_seconds)
        st.rerun()


# ──────────────────────────────────────────────
# Tab: Backtest Review
# ──────────────────────────────────────────────
def render_backtest_review(result_path: Optional[str]) -> None:
    st.subheader("Backtest Review")

    if not result_path:
        st.info(
            "No hay `result-path` configurado. Lanza un backtest con "
            "`--save-result /ruta/result.pkl` y vuelve aquí."
        )
        return

    mtime = _file_mtime(result_path)
    payload = _load_result(result_path, mtime)
    if payload is None:
        st.warning(f"No se encuentra `{result_path}` o el pickle está corrupto.")
        return

    kind = payload.get("kind", "backtest")
    symbol = payload.get("symbol", "—")
    saved_at = payload.get("saved_at", "—")
    result = payload["result"]

    st.caption(f"**kind**: {kind}  |  **symbol(s)**: {symbol}  |  **saved**: {saved_at[:19]}")

    # KPIs principales — comunes a Backtest y Portfolio.
    cols = st.columns(5)
    _kpi(cols[0], "Sharpe", f"{result.sharpe_ratio:.2f}")
    _kpi(cols[1], "Retorno", f"{result.total_return_pct:+.2f}%")
    _kpi(cols[2], "Max DD", f"{result.max_drawdown_pct:.2f}%")
    _kpi(cols[3], "Trades", str(result.total_trades))
    _kpi(cols[4], "Win Rate", f"{result.win_rate:.1f}%")

    cols2 = st.columns(5)
    _kpi(cols2[0], "Profit Factor", f"{result.profit_factor:.2f}")
    _kpi(cols2[1], "Expectancy", f"{result.expectancy_pct:+.2f}%")
    _kpi(cols2[2], "Exposure", f"{result.exposure_pct:.1f}%")
    _kpi(cols2[3], "Calmar", f"{result.calmar_ratio:.2f}")
    _kpi(cols2[4], "Sortino", f"{result.sortino_ratio:.2f}")

    # Equity curve.
    if result.equity_curve is not None:
        try:
            import plotly.graph_objects as go
            ec = result.equity_curve.dropna()
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=ec.index, y=ec.values, mode="lines", name="Equity",
                line={"width": 2},
            ))
            fig.update_layout(
                height=380, margin={"t": 20, "b": 20, "l": 20, "r": 20},
                xaxis_title=None, yaxis_title="Equity ($)",
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as exc:
            st.warning(f"Plotly no disponible: {exc}")

    # Desglose por símbolo (sólo en portfolio runs).
    per_sym = getattr(result, "per_symbol", None)
    if per_sym:
        rows = []
        for sym, res in per_sym.items():
            rows.append({
                "symbol": sym,
                "trades": res.total_trades,
                "win_rate": f"{res.win_rate:.1f}%",
                "profit_factor": f"{res.profit_factor:.2f}",
                "return_pct": f"{res.total_return_pct:+.2f}%",
            })
        st.markdown("**Desglose por símbolo**")
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # Trades.
    if result.trades:
        df_trades = pd.DataFrame(result.trades)
        st.markdown(f"**Trades ({len(df_trades)})**")
        # Filtros opcionales.
        if "symbol" in df_trades.columns:
            symbols = sorted(df_trades["symbol"].dropna().unique().tolist())
            sel = st.multiselect("Filtrar por símbolo", symbols, default=symbols)
            df_trades = df_trades[df_trades["symbol"].isin(sel)]
        if "exit_reason" in df_trades.columns:
            reasons = sorted(df_trades["exit_reason"].dropna().unique().tolist())
            sel_r = st.multiselect("Filtrar por motivo de salida", reasons, default=reasons)
            df_trades = df_trades[df_trades["exit_reason"].isin(sel_r)]
        st.dataframe(df_trades, use_container_width=True, hide_index=True)

    # Resumen textual (cae bien al final como ground truth).
    with st.expander("Resumen completo (texto)"):
        st.text(result.summary())


# ──────────────────────────────────────────────
# Tab: Model Inspection
# ──────────────────────────────────────────────
def render_model_inspection(model_path: Optional[str]) -> None:
    st.subheader("Model Inspection")

    if not model_path:
        st.info("No hay `model-path` configurado. Pasa `--model models/...pkl` o cambia abajo.")
        return

    mtime = _file_mtime(model_path)
    payload = _load_model(model_path, mtime)
    if payload is None:
        st.warning(f"No se encuentra `{model_path}` o el pickle está corrupto.")
        return

    is_split = isinstance(payload, dict) and {"trend", "mr"}.issubset(payload.keys())
    st.caption(f"**Tipo**: {'regime-split (trend + mean-revert)' if is_split else 'global'}")

    # Selección de sub-modelo en split.
    if is_split:
        sub_choice = st.radio(
            "Sub-modelo", ["trend", "mr"], horizontal=True,
            help="Modelos del régimen TREND_UP/TREND_DOWN vs CHOP/MEAN_REVERT.",
        )
        sub = payload[sub_choice]
    else:
        sub = payload

    # Threshold y nº de features.
    threshold = sub.get("threshold")
    feature_cols = sub.get("feature_cols", []) or []
    cols = st.columns(3)
    _kpi(cols[0], "Threshold", f"{threshold:.3f}" if isinstance(threshold, (int, float)) else "—")
    _kpi(cols[1], "n_features", str(len(feature_cols)))
    _kpi(cols[2], "Symbols (basket)", str(len(sub.get("symbols", []) or [])))

    # Feature importance.
    model = sub.get("model")
    if model is not None and feature_cols:
        try:
            importance = model.feature_importance(importance_type="split")
        except Exception:
            importance = getattr(model, "feature_importances_", None)
        if importance is not None and len(importance) == len(feature_cols):
            df_imp = pd.DataFrame({
                "feature": feature_cols,
                "importance": importance,
            }).sort_values("importance", ascending=False).head(30)

            try:
                import plotly.express as px
                fig = px.bar(df_imp, x="importance", y="feature", orientation="h", height=600)
                fig.update_layout(yaxis={"categoryorder": "total ascending"},
                                   margin={"t": 10, "b": 10, "l": 10, "r": 10})
                st.plotly_chart(fig, use_container_width=True)
            except Exception:
                st.dataframe(df_imp, use_container_width=True, hide_index=True)

    # Metadata varia (si la guardó el trainer).
    with st.expander("Metadata cruda"):
        meta = {k: v for k, v in sub.items() if k not in {"model", "feature_cols"}}
        # Recortar listas largas para no inundar pantalla.
        for k, v in meta.items():
            if isinstance(v, list) and len(v) > 50:
                meta[k] = f"<list of {len(v)} items>"
        st.json(meta, expanded=False)


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main() -> None:
    st.set_page_config(
        page_title="Trading Bot Dashboard",
        page_icon="📈",
        layout="wide",
    )
    st.title("Trading Bot — Dashboard")

    # Sidebar: paths configurables (defaults vienen del CLI).
    with st.sidebar:
        st.header("Configuración")
        state_path = st.text_input("Live state JSON", value=_ARGS.state_path or "")
        result_path = st.text_input("Backtest result pickle", value=_ARGS.result_path or "")
        model_path = st.text_input("Model pickle", value=_ARGS.model_path or "")
        refresh_seconds = st.slider(
            "Auto-refresh Live (segundos)", min_value=0, max_value=60, value=5, step=1,
            help="0 = desactivado.",
        )
        st.caption(
            "**Tip**: `--state-path` para Live Monitor, `--save-result` para "
            "Backtest Review, `--model` para Model Inspection."
        )

    # Usamos un radio en sidebar en vez de st.tabs porque la pestaña
    # Live Monitor incluye un `st.rerun()` periódico que abortaría las
    # otras pestañas si fuesen `st.tabs` (todas se ejecutan en cada
    # rerun). Con un radio sólo se llama la función de la página activa.
    page = st.sidebar.radio(
        "Página",
        ["Live Monitor", "Backtest Review", "Model Inspection"],
        index=0,
    )
    if page == "Live Monitor":
        render_live_monitor(state_path or None, refresh_seconds)
    elif page == "Backtest Review":
        render_backtest_review(result_path or None)
    else:
        render_model_inspection(model_path or None)


if __name__ == "__main__":
    main()
