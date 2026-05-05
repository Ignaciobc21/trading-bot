"""
main_ibkr.py — Punto de entrada del Trading Bot para Interactive Brokers.

Sustituye el bloque `--mode live` de main.py con soporte IBKR completo.
El resto de modos (backtest, train, portfolio, features, dashboard) no
cambian: siguen usando Yahoo Finance para datos históricos.

Uso:
    # Single symbol – paper trading TWS (puerto 7497)
    python main_ibkr.py --mode live --symbol AAPL --timeframe 1Day \\
        --strategy meta_ensemble --model models/v1.pkl \\
        --ibkr-port 7497 --dry-run

    # Multi-símbolo con correlation guard
    python main_ibkr.py --mode live \\
        --symbols AAPL,MSFT,NVDA,NFLX \\
        --strategy meta_ensemble --model models/v1.pkl \\
        --ibkr-port 7497 --max-positions 3 --dry-run

    # Con bracket orders (SL + TP nativos en IB)
    python main_ibkr.py --mode live --symbol NVDA --timeframe 1Hour \\
        --strategy ensemble --ibkr-port 7497 \\
        --stop-loss-pct 2.0 --take-profit-pct 4.0 --use-bracket-orders

    # Live trading real (TWS Live, puerto 7496)
    python main_ibkr.py --mode live --symbol AAPL \\
        --strategy meta_ensemble --model models/v1.pkl \\
        --ibkr-port 7496 --ibkr-account U1234567

Notas:
  - El modo 'backtest', 'train', 'portfolio', 'features' y 'dashboard'
    se delegan al main.py original (no usan IB).
  - Requiere ib_insync: pip install ib_insync
  - TWS / IB Gateway debe estar abierto con la API habilitada.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

# ── Re-usar la lógica de logging del bot original ──
from utils.logger import get_logger

logger = get_logger("main_ibkr")


# ════════════════════════════════════════════════════════════════════
#  Runner principal IBKR (single y multi-symbol)
# ════════════════════════════════════════════════════════════════════
def run_ibkr_live(
    symbol: str,
    timeframe: str,
    strategy_name: str,
    model_path: Optional[str],
    ibkr_host: str,
    ibkr_port: int,
    ibkr_client_id: int,
    ibkr_account: str,
    dry_run: bool,
    stop_loss_pct: float,
    take_profit_pct: float,
    use_bracket_orders: bool,
    base_position_size_pct: float,
    risk_overlay: bool,
    target_vol: float,
    max_iters: Optional[int],
    state_path: Optional[str],
    use_rth: bool,
    history_bars: int,
) -> None:
    """
    Arranca el runner IBKR single-symbol.

    Cambios respecto a run_live() de main.py:
      1. IBKRBroker en lugar de Broker (Alpaca).
      2. IBKRLiveRunner en lugar de LiveRunner.
      3. El bucle corre en ib.run() (evento IB) en lugar de while-True.
      4. Datos via reqHistoricalData + barUpdateEvent en lugar de Alpaca REST.
    """
    from execution.ibkr_broker import IBKRBroker, IBKRConfig
    from execution.ibkr_live_runner import IBKRLiveRunner, IBKRLiveConfig

    logger.info("═" * 55)
    logger.info(
        "  IBKR LIVE — %s %s  strategy=%s  dry_run=%s",
        symbol, timeframe, strategy_name, dry_run,
    )
    logger.info("  TWS/Gateway: %s:%d  clientId=%d  account=%s",
                ibkr_host, ibkr_port, ibkr_client_id, ibkr_account or "auto")
    logger.info("═" * 55)

    # ── 1. Estrategia ────────────────────────────────────────────────
    strategy = _build_strategy(strategy_name, model_path)

    # ── 2. Broker IBKR ──────────────────────────────────────────────
    ibkr_cfg = IBKRConfig(
        host=ibkr_host,
        port=ibkr_port,
        client_id=ibkr_client_id,
        account=ibkr_account,
    )
    try:
        broker = IBKRBroker(ibkr_cfg)
    except Exception as exc:
        logger.error(
            "No se pudo conectar a IB en %s:%d — %s\n"
            "Soluciones:\n"
            "  1. Abre TWS o IB Gateway.\n"
            "  2. Habilita la API: TWS → Edit → Global Configuration → "
            "API → Settings → Enable ActiveX and Socket Clients.\n"
            "  3. Verifica el puerto: 7497 (TWS Paper) / 4002 (Gateway Paper) "
            "/ 7496 (TWS Live) / 4001 (Gateway Live).",
            ibkr_host, ibkr_port, exc,
        )
        sys.exit(1)

    # ── 3. Risk overlay (opcional) ───────────────────────────────────
    overlay = None
    if risk_overlay:
        from risk.overlay import RiskConfig, RiskOverlay
        overlay = RiskOverlay(RiskConfig(target_vol=target_vol))
        logger.info("Risk overlay ON — target_vol=%.2f", target_vol)

    # ── 4. Telegram (opcional) ───────────────────────────────────────
    try:
        from utils.helpers import send_telegram_message
        telegram_fn = send_telegram_message
    except Exception:
        telegram_fn = None

    # ── 5. Config del runner ─────────────────────────────────────────
    runner_cfg = IBKRLiveConfig(
        symbol=symbol,
        timeframe=timeframe,
        history_bars=history_bars,
        base_position_size_pct=base_position_size_pct,
        dry_run=dry_run,
        stop_loss_pct=stop_loss_pct,
        take_profit_pct=take_profit_pct,
        use_bracket_orders=use_bracket_orders,
        max_iters=max_iters,
        state_path=state_path,
        use_rth=use_rth,
    )

    # ── 6. Arranque ──────────────────────────────────────────────────
    runner = IBKRLiveRunner(
        strategy=strategy,
        broker=broker,
        config=runner_cfg,
        overlay=overlay,
        telegram_fn=telegram_fn,
    )
    runner.run()


# ════════════════════════════════════════════════════════════════════
def run_ibkr_live_portfolio(
    symbols: List[str],
    timeframe: str,
    strategy_name: str,
    model_path: Optional[str],
    ibkr_host: str,
    ibkr_port: int,
    ibkr_client_id: int,
    ibkr_account: str,
    dry_run: bool,
    stop_loss_pct: float,
    take_profit_pct: float,
    use_bracket_orders: bool,
    position_size_pct: float,
    max_positions: int,
    corr_threshold: float,
    risk_overlay: bool,
    target_vol: float,
    max_iters: Optional[int],
    state_path: Optional[str],
    use_rth: bool,
    history_bars: int,
) -> None:
    """Arranca el runner IBKR multi-símbolo (portfolio live)."""
    from execution.ibkr_broker import IBKRBroker, IBKRConfig
    from execution.ibkr_live_runner import IBKRPortfolioLiveRunner, IBKRPortfolioLiveConfig

    logger.info("═" * 55)
    logger.info(
        "  IBKR PORTFOLIO LIVE — %d símbolos  strategy=%s  dry_run=%s",
        len(symbols), strategy_name, dry_run,
    )
    logger.info("  Símbolos: %s", ", ".join(symbols))
    logger.info("  TWS/Gateway: %s:%d  clientId=%d", ibkr_host, ibkr_port, ibkr_client_id)
    logger.info("═" * 55)

    strategy = _build_strategy(strategy_name, model_path)

    ibkr_cfg = IBKRConfig(
        host=ibkr_host, port=ibkr_port,
        client_id=ibkr_client_id, account=ibkr_account,
    )
    try:
        broker = IBKRBroker(ibkr_cfg)
    except Exception as exc:
        logger.error("Conexión IB fallida: %s", exc)
        sys.exit(1)

    overlay = None
    if risk_overlay:
        from risk.overlay import RiskConfig, RiskOverlay
        overlay = RiskOverlay(RiskConfig(target_vol=target_vol))

    try:
        from utils.helpers import send_telegram_message
        telegram_fn = send_telegram_message
    except Exception:
        telegram_fn = None

    runner_cfg = IBKRPortfolioLiveConfig(
        symbols=symbols,
        timeframe=timeframe,
        history_bars=history_bars,
        max_positions=max_positions,
        position_size_pct=position_size_pct,
        corr_threshold=corr_threshold,
        dry_run=dry_run,
        stop_loss_pct=stop_loss_pct,
        take_profit_pct=take_profit_pct,
        use_bracket_orders=use_bracket_orders,
        max_iters=max_iters,
        state_path=state_path,
        use_rth=use_rth,
    )

    runner = IBKRPortfolioLiveRunner(
        strategy=strategy,
        broker=broker,
        config=runner_cfg,
        overlay=overlay,
        telegram_fn=telegram_fn,
    )
    runner.run()


# ════════════════════════════════════════════════════════════════════
#  Factoría de estrategias (idéntica a main.py para no duplicar)
# ════════════════════════════════════════════════════════════════════
def _build_strategy(strategy_name: str, model_path: Optional[str]):
    if strategy_name == "meta_ensemble":
        if not model_path:
            logger.error("--strategy meta_ensemble requiere --model")
            sys.exit(1)
        from strategies.meta_labeled_ensemble import build_meta_labeled_ensemble_from_file
        return build_meta_labeled_ensemble_from_file(model_path)
    elif strategy_name == "ensemble":
        from strategies.ensemble import build_default_ensemble
        return build_default_ensemble()
    elif strategy_name == "donchian":
        from strategies.donchian_trend import DonchianTrendStrategy
        return DonchianTrendStrategy()
    elif strategy_name == "rsi2_mr":
        from strategies.rsi2_mean_reversion import RSI2MeanReversionStrategy
        return RSI2MeanReversionStrategy()
    elif strategy_name == "mfi_rsi":
        from strategies.mfi_rsi_strategy import MfiRsiStrategy
        return MfiRsiStrategy()
    else:
        from strategies.rsi_strategy import RSIStrategy
        return RSIStrategy()


# ════════════════════════════════════════════════════════════════════
#  CLI
# ════════════════════════════════════════════════════════════════════
def main() -> None:
    parser = argparse.ArgumentParser(
        prog="trading-bot-ibkr",
        description=(
            "Trading Bot con Interactive Brokers (ib_insync).\n"
            "\n"
            "Para backtest / train / features / dashboard, usa main.py (Yahoo Finance).\n"
            "Este script gestiona exclusivamente el modo --mode live con IB.\n"
            "\n"
            "PUERTOS TWS/GATEWAY:\n"
            "  7497  TWS Paper Trading\n"
            "  4002  IB Gateway Paper Trading\n"
            "  7496  TWS Live Trading\n"
            "  4001  IB Gateway Live Trading\n"
            "\n"
            "EJEMPLOS:\n"
            "  # Single symbol, paper, meta_ensemble\n"
            "  python main_ibkr.py --symbol AAPL --strategy meta_ensemble \\\n"
            "      --model models/v1.pkl --ibkr-port 7497 --dry-run\n"
            "\n"
            "  # Portfolio, bracket orders SL/TP nativos en IB\n"
            "  python main_ibkr.py --symbols AAPL,MSFT,NVDA \\\n"
            "      --strategy ensemble --ibkr-port 7497 \\\n"
            "      --stop-loss-pct 2.0 --take-profit-pct 4.0 \\\n"
            "      --use-bracket-orders --dry-run\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    g = parser.add_argument_group("estrategia y símbolo")
    g.add_argument("--symbol",   default="AAPL",   help="Ticker single-symbol.")
    g.add_argument("--symbols",  default=None,      help="Tickers CSV para portfolio live.")
    g.add_argument("--strategy",
                   choices=["rsi", "mfi_rsi", "donchian", "rsi2_mr", "ensemble", "meta_ensemble"],
                   default="ensemble")
    g.add_argument("--model",    default=None,      help="Path al .pkl del meta-labeler.")
    g.add_argument("--timeframe",default="1Day",
                   help="Granularidad IB-style: 1Min, 5Min, 15Min, 1Hour, 4Hour, 1Day.")
    g.add_argument("--history-bars", type=int, default=500,
                   help="Barras de histórico a cargar al inicio (default 500).")

    g2 = parser.add_argument_group("conexión Interactive Brokers")
    g2.add_argument("--ibkr-host",      default="127.0.0.1")
    g2.add_argument("--ibkr-port",      type=int, default=7497,
                    help="7497=TWS Paper | 4002=Gateway Paper | 7496=TWS Live | 4001=Gateway Live")
    g2.add_argument("--ibkr-client-id", type=int, default=1,
                    help="ID único por conexión simultánea (default 1).")
    g2.add_argument("--ibkr-account",   default="",
                    help="Cuenta IB (ej: DU1234567). Vacío = auto-detect.")
    g2.add_argument("--no-rth", action="store_true",
                    help="Incluir barras fuera de Regular Trading Hours.")

    g3 = parser.add_argument_group("ejecución")
    g3.add_argument("--dry-run",         action="store_true")
    g3.add_argument("--stop-loss-pct",   type=float, default=0.0)
    g3.add_argument("--take-profit-pct", type=float, default=0.0)
    g3.add_argument("--use-bracket-orders", action="store_true",
                    help="Usa Bracket OCA (SL+TP nativos en IB) en lugar de gestión manual.")
    g3.add_argument("--position-size-pct", type=float, default=2.0)
    g3.add_argument("--max-positions",   type=int,   default=4)
    g3.add_argument("--corr-threshold",  type=float, default=0.75)
    g3.add_argument("--max-iters",       type=int,   default=None)
    g3.add_argument("--state-path",      default=None)

    g4 = parser.add_argument_group("risk overlay")
    g4.add_argument("--risk-overlay",  action="store_true")
    g4.add_argument("--target-vol",    type=float, default=0.20)

    args = parser.parse_args()

    symbols = None
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]

    common = dict(
        timeframe=args.timeframe,
        strategy_name=args.strategy,
        model_path=args.model,
        ibkr_host=args.ibkr_host,
        ibkr_port=args.ibkr_port,
        ibkr_client_id=args.ibkr_client_id,
        ibkr_account=args.ibkr_account,
        dry_run=args.dry_run,
        stop_loss_pct=args.stop_loss_pct,
        take_profit_pct=args.take_profit_pct,
        use_bracket_orders=args.use_bracket_orders,
        risk_overlay=args.risk_overlay,
        target_vol=args.target_vol,
        max_iters=args.max_iters,
        state_path=args.state_path,
        use_rth=not args.no_rth,
        history_bars=args.history_bars,
    )

    if symbols and len(symbols) >= 2:
        run_ibkr_live_portfolio(
            symbols=symbols,
            position_size_pct=args.position_size_pct,
            max_positions=args.max_positions,
            corr_threshold=args.corr_threshold,
            **common,
        )
    else:
        sym = (symbols[0] if symbols else args.symbol)
        run_ibkr_live(
            symbol=sym,
            base_position_size_pct=args.position_size_pct,
            **common,
        )


if __name__ == "__main__":
    main()