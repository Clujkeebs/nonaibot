"""
24/7 Trading Bot — Entry Point

Start with:  python main.py
Kill with:   Ctrl-C  (or set KILL_SWITCH=1 in env)

The bot starts the TradingEngine, wires up the TaskManager (APScheduler),
immediately runs an initial regime update + crypto scan, then blocks on
a heartbeat loop that keeps the process alive and logs every 5 minutes.
"""
import os
import signal
import sys
import time

from utils.logger import log, setup_logging

# Logger must be set up before any other import that uses it
setup_logging()

import config  # noqa: E402  (after logging)
from engine import TradingEngine  # noqa: E402
from scheduler.task_manager import TaskManager  # noqa: E402
from utils.alerts import alert_error  # noqa: E402


def _banner() -> None:
    log.info("=" * 60)
    log.info("  24/7 AUTOMATED TRADING BOT")
    log.info("  Paper mode : {}", config.ALPACA_PAPER)
    log.info("  Strategies : TrendFollow MeanRev VolBreak Rotation Crypto")
    log.info("  Risk       : {:.0%} per trade | {:.0%} daily halt | {:.0%} weekly halt",
             config.RISK_PER_TRADE_PCT, config.DAILY_LOSS_LIMIT_PCT, config.WEEKLY_LOSS_LIMIT_PCT)
    log.info("  AI layer   : {}", config.ENABLE_AI_LAYER)
    log.info("=" * 60)


def main() -> None:
    _banner()

    engine  = TradingEngine()
    manager = TaskManager(engine)

    # Graceful shutdown
    _running = [True]

    def _shutdown(sig, frame):
        log.info("Shutdown signal received — stopping gracefully...")
        _running[0] = False
        manager.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    # Start all scheduled jobs
    manager.start()

    # Immediate warm-up: regime + first crypto scan
    try:
        log.info("Warm-up: updating regime...")
        engine.update_regime()
        log.info("Warm-up: initial crypto scan...")
        engine.run_crypto_strategies()
    except Exception as e:
        log.error("Warm-up error: {}", e)
        alert_error(str(e), "startup warm-up")

    # Main keep-alive loop
    heartbeat_interval = 300  # 5 min
    last_hb = 0.0

    log.info("Bot is running. Press Ctrl-C to stop.")

    while _running[0]:
        now = time.time()
        if now - last_hb >= heartbeat_interval:
            try:
                engine.portfolio_heartbeat()
            except Exception as e:
                log.warning("Heartbeat error: {}", e)
            last_hb = now
        time.sleep(10)


if __name__ == "__main__":
    main()
