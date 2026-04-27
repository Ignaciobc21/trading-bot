# trading-bot

Motor de trading algorítmico modular con **pipeline ML completo**
(meta-labeling LightGBM + regime split + threshold tuning por Sharpe +
Optuna), **risk overlay**, **portfolio multi-símbolo**, **cost model
realista**, **dashboard Streamlit** y **auto-retrain con detección de
concept drift**. Listo para paper trading en Alpaca con monitorización
end-to-end.

> **Estado actual**: 16 PRs entregados cubriendo las fases
> B → C → D → E → F → G → H → I → J → K. Ver sección *Roadmap implementado*.

---

## Tabla de contenidos

1. [Quick start](#quick-start)
2. [Modos de ejecución](#modos-de-ejecución)
3. [Flujo de un trade](#flujo-de-un-trade)
4. [Estructura del repo](#estructura-del-repo)
5. [Roadmap implementado](#roadmap-implementado)
6. [Ejemplos rápidos](#ejemplos-rápidos)
7. [Dashboard](#dashboard)
8. [Troubleshooting](#troubleshooting)

---

## Quick start

```bash
# 1) Clona y entra
git clone https://github.com/Ignaciobc21/trading-bot.git
cd trading-bot

# 2) Entorno virtual
python -m venv .venv
# Linux / macOS:
. .venv/bin/activate
# Windows Git Bash:
# . .venv/Scripts/activate

# 3) Dependencias
pip install -r requirements.txt

# 4) Sanity check: backtest de 1 año sobre AAPL con la estrategia base
python main.py --mode backtest --symbol AAPL --period 1y --strategy mfi_rsi

# 5) Ver la ayuda completa del CLI (agrupada por modo)
python main.py --help
```

Hay un fichero [`commands.txt`](commands.txt) con **19 comandos listos para
copiar/pegar** cubriendo todos los modos (backtest, portfolio, train, live,
dashboard, features) y stress-tests típicos.

---

## Modos de ejecución

El CLI expone 6 modos a través de `--mode`:

| Modo | Qué hace |
|---|---|
| `backtest`  | Backtest single-symbol con cualquier estrategia (mfi_rsi, ensemble, meta_ensemble…). |
| `portfolio` | Backtest multi-símbolo con cap de posiciones y correlation guard. |
| `train`     | Entrena el meta-labeler LightGBM (triple-barrier → CV → threshold → pickle). |
| `features`  | Vuelca el DataFrame de features a parquet/csv (útil para EDA). |
| `live`      | Live / paper trading en Alpaca (o `--dry-run` con Yahoo sin credenciales). |
| `dashboard` | UI Streamlit con 3 páginas (Live Monitor / Backtest Review / Model Inspection). |

### Estrategias disponibles (`--strategy`)

| Nombre | Descripción |
|---|---|
| `rsi`            | RSI-2 Connors puro. |
| `mfi_rsi`        | MFI + RSI reversal (base original). |
| `donchian`       | Donchian channel breakout (trend-following). |
| `rsi2_mr`        | RSI2 mean-reversion con SMA200 filter. |
| `ensemble`       | Donchian trend + RSI2 mean-rev + detector de régimen. |
| `meta_ensemble`  | **RECOMENDADA** — ensemble + LightGBM meta-filter. Requiere `--model`. |

---

## Flujo de un trade

```
Yahoo / Alpaca data   →  FeatureBuilder        →  Strategy.next()
(1m — 1d, hasta 5y)      (80+ feats técnicas     (Donchian + RSI2 +
                          + sentiment VADER)      Regime detector)
                                                         ↓
                                                    signal BUY?
                                                         ↓
                                           Meta-labeler LightGBM
                                           (regime split: trend/mean-rev)
                                                         ↓
                                                 threshold Sharpe-tuned
                                                         ↓
                                           Risk overlay (vol × regime
                                           × confidence × kill-switch)
                                                         ↓
                                           Cost model realista
                                           (commission + spread + impact ADV)
                                                         ↓
                                           Broker (Alpaca paper / dry-run)
                                                         ↓
                                           state.json snapshot  →  Dashboard
                                                         ↓
                                           Drift detector (KS + PSI + AUC)
                                                         ↓ si drift
                                           Retrain orchestrator
                                           (cooldown + lock + validate + atomic swap)
```

---

## Estructura del repo

```
trading-bot/
├── main.py                    # CLI único (6 modos, 60+ flags agrupados)
├── commands.txt               # Cheatsheet listo para copy-paste
├── requirements.txt
│
├── config/
│   ├── settings.py            # parámetros por defecto (risk, timeframes…)
│   └── secrets.env            # ALPACA_API_KEY / ALPACA_SECRET_KEY (no commit)
│
├── data/                      # ingesta Alpaca / yfinance + storage SQL
├── features/
│   ├── builder.py             # 80+ features técnicas
│   ├── sentiment.py           # VADER + crypto Fear & Greed (fase B)
│   └── cache.py               # cache parquet en ~/.cache/trading-bot/
│
├── strategies/
│   ├── regime.py              # detector trend / mean-revert / chop
│   ├── donchian_trend.py      # trend-following
│   ├── rsi2_mean_reversion.py # mean-revert
│   ├── ensemble.py            # combina ambas según régimen
│   └── meta_labeled_ensemble.py  # ensemble + filtro ML
│
├── labels/
│   └── triple_barrier.py      # labels ML (TP antes que SL en N barras)
│
├── ml/
│   ├── meta_labeler.py        # entrenamiento LightGBM + CV + threshold
│   ├── tuning.py              # Optuna TPE + MedianPruner (fase J)
│   ├── drift.py               # KS + PSI + AUC rolling (fase K)
│   └── retrain.py             # orchestrator con cooldown + atomic swap (fase K)
│
├── risk/
│   └── overlay.py             # vol target × regime × confidence × kill-switch
│
├── execution/
│   ├── live_runner.py         # loop live paper/real + hot-reload pickle
│   └── costs.py               # cost model flat y realistic (fase I)
│
├── backtesting/
│   ├── engine.py              # motor single-symbol
│   └── portfolio_engine.py    # motor multi-símbolo con corr guard (fase G)
│
└── dashboard/
    └── app.py                 # Streamlit 3 páginas (fase H)
```

---

## Roadmap implementado

16 PRs apilados (cada uno sobre el anterior). Todos merged / disponibles en
la rama más reciente `devin/1777260000-k-auto-retrain`.

### Capa de datos & features
| PR | Fase | Qué aporta |
|---|---|---|
| [#1](https://github.com/Ignaciobc21/trading-bot/pull/1)  | base | MFI+RSI strategy + backtest engine |
| [#3](https://github.com/Ignaciobc21/trading-bot/pull/3)  | P3   | 80+ features técnicas + cache parquet |
| [#11](https://github.com/Ignaciobc21/trading-bot/pull/11)| B    | Sentiment VADER + crypto Fear & Greed |

### Capa de etiquetado & ML
| PR | Fase | Qué aporta |
|---|---|---|
| [#4](https://github.com/Ignaciobc21/trading-bot/pull/4)   | P4    | Triple-barrier + LightGBM meta-labeler |
| [#5](https://github.com/Ignaciobc21/trading-bot/pull/5)   | P4.1  | Basket training (anti-overfit) |
| [#6](https://github.com/Ignaciobc21/trading-bot/pull/6)   | P4.2  | Regime split (2 modelos: trend / mean-rev) |
| [#8](https://github.com/Ignaciobc21/trading-bot/pull/8)   | D     | Threshold tuning por Sharpe |
| [#10](https://github.com/Ignaciobc21/trading-bot/pull/10) | F     | Purged k-fold CV (López de Prado) |
| [#15](https://github.com/Ignaciobc21/trading-bot/pull/15) | J     | Optuna tuning (TPE + MedianPruner) |

### Capa de estrategias
| PR | Fase | Qué aporta |
|---|---|---|
| [#2](https://github.com/Ignaciobc21/trading-bot/pull/2) | P1+P2 | Régimen detector + Ensemble |

### Capa de riesgo & ejecución
| PR | Fase | Qué aporta |
|---|---|---|
| [#7](https://github.com/Ignaciobc21/trading-bot/pull/7)  | P6 | Risk overlay (vol + regime + confidence + kill-switch) |
| [#9](https://github.com/Ignaciobc21/trading-bot/pull/9)  | E  | Live runner (Alpaca paper + dry-run) |
| [#14](https://github.com/Ignaciobc21/trading-bot/pull/14)| I  | Cost model realista (spread + impact + commission bps) |

### Capa de portfolio & monitorización
| PR | Fase | Qué aporta |
|---|---|---|
| [#12](https://github.com/Ignaciobc21/trading-bot/pull/12)| G | Portfolio backtest + correlation guard |
| [#13](https://github.com/Ignaciobc21/trading-bot/pull/13)| H | Dashboard Streamlit (3 páginas) |
| [#16](https://github.com/Ignaciobc21/trading-bot/pull/16)| K | Auto-retrain + drift detection (KS + PSI + AUC) |

---

## Ejemplos rápidos

### Backtest RECOMENDADO (meta-ensemble + risk overlay + costes reales)

```bash
python main.py --mode backtest --strategy meta_ensemble \
  --model models/basket25_split_sent.pkl \
  --symbol NFLX --period 5y \
  --risk-overlay --target-vol 0.20 \
  --cost-model realistic --commission-bps 1.0 --spread-bps 4.0 \
  --save-result /tmp/nflx_bt.pkl
```

### Portfolio 7 símbolos con correlation guard

```bash
python main.py --mode portfolio --strategy meta_ensemble \
  --model models/basket25_split_sent.pkl \
  --symbols NFLX,DIS,KO,BTC-USD,INTC,PFE,NKE \
  --max-positions 4 --corr-threshold 0.75 --period 5y \
  --cost-model realistic --save-result /tmp/portfolio.pkl
```

### Entrenar un modelo completo (basket-25 + regime split + Optuna + sentiment)

```bash
python main.py --mode train \
  --symbols SPY,QQQ,IWM,AAPL,MSFT,NVDA,GOOGL,META,AMZN,TSLA,JPM,BAC,GS,WMT,COST,MCD,UNH,JNJ,LLY,XOM,CVX,CAT,BA,AMD,AVGO \
  --period 5y --regime-split --threshold-objective sharpe \
  --cv-method purged_kfold --include-sentiment \
  --tune-hp --tune-trials 50 \
  --model models/basket25_split_sent.pkl
```

### Paper / dry-run con auto-retrain y monitor de drift

```bash
python main.py --mode live --strategy meta_ensemble \
  --model models/basket25_split_sent.pkl \
  --symbol AAPL --live-timeframe 1Day --data-source yahoo --dry-run \
  --risk-overlay --target-vol 0.20 \
  --auto-retrain --retrain-cooldown-days 7 \
  --drift-check-every-iters 20 \
  --state-path /tmp/live.json
```

Más ejemplos en [`commands.txt`](commands.txt).

---

## Dashboard

Streamlit con 3 páginas. Se conecta a los ficheros generados por los otros
modos:

- **Live Monitor** — lee `--state-path` que escribe el `LiveRunner` cada
  iteración. Muestra PnL, posición, señal, régimen, confidence,
  decisiones del risk overlay, drift (PSI / KS / AUC) y estado del retrain
  orchestrator.
- **Backtest Review** — carga el pickle de `--save-result` y muestra equity
  curve, trades, Sharpe, max DD, análisis por régimen, cost breakdown.
- **Model Inspection** — abre el `.pkl` del meta-labeler y muestra feature
  importance, AUC por fold, threshold search, histograma de probabilidades,
  resumen del Optuna study.

```bash
python main.py --mode dashboard \
  --state-path /tmp/live.json \
  --save-result /tmp/portfolio.pkl \
  --model models/basket25_split_sent.pkl
```

---

## Troubleshooting

### `UnicodeEncodeError: 'charmap' codec can't encode character` en Windows

Ya resuelto — `main.py` reconfigura `stdout`/`stderr` a UTF-8 al arrancar.
Si aún ves el error, ejecuta en PowerShell:

```powershell
$env:PYTHONIOENCODING="utf-8"
```

### `ValueError: Length of values (N) does not match length of index (M)`

Suele aparecer cuando el período es demasiado corto y algunas estrategias
no tienen suficientes barras para arrancar. Prueba con `--period 1y` o más.

### El modelo no predice — `meta_ensemble` requiere `--model`

```
--strategy meta_ensemble requiere --model /ruta/al/modelo.pkl
```

Entrena uno primero con `--mode train` o descárgate uno pre-entrenado.

### Cache de features viejo

```bash
rm -rf ~/.cache/trading-bot/features/
```

---

## Disclaimer

Este software es educacional y para investigación. No es asesoramiento
financiero. El autor no se responsabiliza de pérdidas derivadas de su uso
con dinero real. Prueba siempre en `--dry-run` / paper antes de operar
live.
