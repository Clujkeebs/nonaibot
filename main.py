"""
main.py — Autonomous Alpaca Trading Bot run loop.

Start:  python main.py
Stop:   SIGTERM or Ctrl-C (graceful shutdown)

Loop cadence (30-second ticks):
  Every tick   : config reload, circuit-breaker check, exit checker
  Every N min  : crypto scan (24/7), equity scan (market hours only)
  Pre-market   : regime update, screener (once/day), audit report (once/day)
  4 AM ET      : daily reset (circuit-breaker + cooldowns)

No AI in trading logic — every decision is deterministic and rule-based.
"""
from __future__ import annotations

import logging
import os
import signal
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional

import pytz

# ── Bootstrap logging before any other imports ────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("main")

# ── Project imports ───────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from core.config import BotConfig
from core.broker import BrokerClient
from core.clock import MarketClock
from core.state import SQLiteState
from data.fetcher import DataFetcher
from data.indicators import atr as calc_atr
from execution.assets import AssetCache
from execution.executor import Executor
from risk.sizer import PositionSizer
from risk.circuit_breaker import CircuitBreaker, HaltLevel
from screener.screener import Screener
from strategy.regime import RegimeFilter
from strategy.trend import TrendStrategy
from strategy.mean_reversion import MeanReversionStrategy
from strategy.crypto_momentum import CryptoMomentumStrategy
from audit.report import AuditReport
from audit.delivery import send_report, send_alert
import server

ET = pytz.timezone("America/New_York")

# ── Constants ─────────────────────────────────────────────────────────────────
TICK_SECONDS = 30
EQUITY_SCAN_INTERVAL = 300   # 5 min during market hours
CRYPTO_SCAN_INTERVAL = 180   # 3 min, 24/7
REGIME_UPDATE_INTERVAL = 900 # 15 min
AUDIT_HOUR_ET = 8            # 8 AM ET
DAILY_RESET_HOUR_ET = 4      # 4 AM ET


def _banner(cfg: BotConfig) -> None:
    mode = "LIVE" if not cfg.paper else "PAPER"
    logger.info("=" * 60)
    logger.info("  AUTONOMOUS ALPACA TRADING BOT  [%s MODE]", mode)
    logger.info("  Core equities : %s", cfg.core_equities)
    logger.info("  Core crypto   : %s", cfg.core_crypto)
    logger.info("  Risk/trade    : %.1f%%  |  Floor: $%.2f", cfg.risk_per_trade_pct * 100, cfg.kill_switch_floor)
    logger.info("  Circuit       : soft=%.0f%%  hard=%.0f%%", cfg.soft_halt_pct * 100, cfg.hard_halt_pct * 100)
    logger.info("=" * 60)


class Bot:
    def __init__(self) -> None:
        self._cfg = BotConfig()
        errors = self._cfg.validate()
        if errors:
            for e in errors:
                logger.error("Config error: %s", e)
            sys.exit(1)

        self._broker = BrokerClient(self._cfg)
        self._clock = MarketClock(config=self._cfg, broker=self._broker)
        self._state = SQLiteState(self._cfg.db_path)
        self._fetcher = DataFetcher(self._broker, self._cfg)
        self._assets = AssetCache(self._broker)
        self._regime = RegimeFilter(self._cfg)
        self._strategies = {
            "trend": TrendStrategy(),
            "mean_reversion": MeanReversionStrategy(),
            "crypto_momentum": CryptoMomentumStrategy(),
        }
        self._sizer = PositionSizer(self._cfg)
        self._circuit = CircuitBreaker(self._cfg)
        self._executor = Executor(
            broker=self._broker,
            assets=self._assets,
            state=self._state,
            config=self._cfg,
        )
        self._screener = Screener(
            fetcher=self._fetcher,
            assets=self._assets,
            state=self._state,
            config=self._cfg,
        )
        self._audit = AuditReport(self._cfg, self._state)

        # Timestamps for interval tracking
        self._last_equity_scan: float = 0.0
        self._last_crypto_scan: float = 0.0
        self._last_regime_update: float = 0.0
        self._audit_done_today: bool = False
        self._screener_done_today: bool = False
        self._last_daily_reset_date: Optional[str] = None

        # Running flag for graceful shutdown
        self._running = True

        # Position tracking loaded from DB
        self._position_highs: Dict[str, float] = {}

    def _load_state(self) -> None:
        """Load persisted state from DB on startup."""
        highs = self._state.get_position_highs()
        self._position_highs.update(highs)
        logger.info("Loaded %d position high-water marks from DB", len(highs))

        # Reload dynamic watchlist into config
        self._cfg.dynamic_equities = self._state.get_dynamic_symbols("equity")
        self._cfg.dynamic_crypto = self._state.get_dynamic_symbols("crypto")
        logger.info(
            "Loaded dynamic watchlist: %d equities, %d crypto",
            len(self._cfg.dynamic_equities),
            len(self._cfg.dynamic_crypto),
        )

    def _get_account(self) -> Optional[Dict]:
        try:
            acct = self._broker.get_account()
            return {
                "equity": float(acct.equity),
                "buying_power": float(acct.buying_power),
                "cash": float(acct.cash),
            }
        except Exception as e:
            logger.warning("Could not fetch account: %s", e)
            return None

    def _get_open_positions(self) -> Dict:
        try:
            positions = self._broker.get_all_positions()
            result = {}
            for p in positions:
                result[p.symbol] = {
                    "qty": float(p.qty),
                    "market_value": float(p.market_value),
                    "avg_price": float(p.avg_entry_price),
                    "unrealized_pl": float(p.unrealized_pl),
                    "unrealized_plpc": float(p.unrealized_plpc),
                }
            return result
        except Exception as e:
            logger.warning("Could not fetch positions: %s", e)
            return {}

    # ── Position reconciliation (adopt manual trades) ──────────────────────────

    def _reconcile_positions(self, open_positions: Dict) -> None:
        """
        Adopt any open position the bot isn't already tracking — for example a
        trade you place by hand in Alpaca (buy $10 of a new stock).

        For each untracked position we:
          - add the symbol to the dynamic watchlist (so it's actively watched and
            protected from being dropped by the screener while held)
          - record an entry-age and high-water mark so the bot fully manages its
            exits (ATR stop, trailing stop, take profit, time stop)

        Detection: a position with no entry-age record in the DB is one the bot
        didn't open itself. Adoption time is used as the time-stop clock start
        (we don't have the original fill timestamp from the positions endpoint).
        """
        if not open_positions:
            return

        tracked = self._state.get_position_ages()  # symbols the bot already manages
        core_set = set(self._cfg.core_equities + self._cfg.core_crypto)
        now_et = datetime.now(ET)

        for sym, pos in open_positions.items():
            if sym in tracked:
                continue  # already managed — nothing to do

            is_crypto = "/" in sym
            qty = float(pos.get("qty", 0))
            avg_price = float(pos.get("avg_price", 0))
            seed_high = avg_price if avg_price > 0 else float(pos.get("market_value", 0))

            # Record entry-age (adoption time) and seed the high-water mark
            self._state.save_position_age(sym, now_et, strategy="manual")
            self._position_highs[sym] = seed_high
            self._state.save_position_high(sym, seed_high)

            # Add to dynamic watchlist unless it's already a core/watchlist symbol
            if sym not in core_set:
                if is_crypto and sym not in self._cfg.dynamic_crypto:
                    self._state.add_dynamic_symbol(sym, "crypto", "manual position adopted")
                    self._cfg.dynamic_crypto.append(sym)
                elif not is_crypto and sym not in self._cfg.dynamic_equities:
                    self._state.add_dynamic_symbol(sym, "equity", "manual position adopted")
                    self._cfg.dynamic_equities.append(sym)

            logger.info(
                "ADOPTED untracked position %s qty=%.6f @ $%.4f — now watching & managing exits",
                sym, qty, avg_price,
            )
            send_alert(
                f"ADOPTED {sym} qty={qty:.6f} @ ${avg_price:.4f} — added to watchlist, "
                f"now managing exits (ATR / trailing / take-profit / time stop)",
                self._cfg,
                level="INFO",
            )

    # ── Exit checker ──────────────────────────────────────────────────────────

    def _check_exits(self, open_positions: Dict, equity: float) -> None:
        """
        Check all open positions for exit conditions:
          1. ATR stop (hard floor below entry)
          2. Trailing stop (arms after trailing_arm_pct gain)
          3. Take profit (take_profit_ratio × initial risk)
          4. Strategy exit signal
          5. Time stop (equity: 10 days; crypto: 5 days)
        """
        if not open_positions:
            return

        all_symbols = list(open_positions.keys())
        equities = [s for s in all_symbols if "/" not in s]
        cryptos = [s for s in all_symbols if "/" in s]

        bars_cache: Dict = {}
        if equities:
            try:
                bars_cache.update(self._fetcher.get_stock_bars(equities))
            except Exception as e:
                logger.warning("Exit check: could not fetch equity bars: %s", e)
        if cryptos:
            try:
                bars_cache.update(self._fetcher.get_crypto_bars(cryptos))
            except Exception as e:
                logger.warning("Exit check: could not fetch crypto bars: %s", e)

        position_ages = self._state.get_position_ages()  # {sym: {opened: datetime, strategy: str}}
        now_et = datetime.now(ET)

        for sym, pos in open_positions.items():
            df = bars_cache.get(sym)
            if df is None or len(df) < 5:
                continue

            qty = float(pos["qty"])
            avg_price = float(pos["avg_price"])
            cur_price = float(df["close"].iloc[-1])
            is_crypto = "/" in sym

            if avg_price <= 0 or qty <= 0:
                continue

            pnl_pct = (cur_price - avg_price) / avg_price

            # Update high-water mark
            prev_high = self._position_highs.get(sym, avg_price)
            new_high = max(prev_high, cur_price)
            if new_high != prev_high:
                self._position_highs[sym] = new_high
                self._state.save_position_high(sym, new_high)
            peak = self._position_highs.get(sym, avg_price)

            # ATR stop
            atr_s = calc_atr(df["high"], df["low"], df["close"], self._cfg.atr_period)
            atr_val = float(atr_s.iloc[-1]) if len(atr_s) > 0 else 0.0
            stop_price = avg_price - self._cfg.atr_stop_mult * atr_val
            exit_reason = ""

            if atr_val > 0 and cur_price <= stop_price:
                exit_reason = "atr_stop"
                logger.info(
                    "EXIT %s atr_stop: price=%.4f stop=%.4f atr=%.4f",
                    sym, cur_price, stop_price, atr_val,
                )

            # Trailing stop (arms after threshold gain)
            if not exit_reason and pnl_pct >= self._cfg.trailing_arm_pct:
                trail_stop = peak * (1 - self._cfg.trailing_giveback_pct)
                if cur_price < trail_stop:
                    exit_reason = "trailing_stop"
                    logger.info(
                        "EXIT %s trailing_stop: price=%.4f peak=%.4f trail=%.4f",
                        sym, cur_price, peak, trail_stop,
                    )

            # Take profit
            if not exit_reason and atr_val > 0 and avg_price > 0:
                risk_pct = self._cfg.atr_stop_mult * atr_val / avg_price
                tp_pct = self._cfg.take_profit_ratio * risk_pct
                if pnl_pct >= tp_pct:
                    exit_reason = "take_profit"
                    logger.info(
                        "EXIT %s take_profit: pnl_pct=%.2f%% tp_pct=%.2f%%",
                        sym, pnl_pct * 100, tp_pct * 100,
                    )

            # Time stop
            if not exit_reason:
                age_rec = position_ages.get(sym)
                if age_rec:
                    entry_dt = age_rec["opened"]
                    if entry_dt.tzinfo is None:
                        entry_dt = ET.localize(entry_dt)
                    max_days = 5 if is_crypto else 10
                    held = (now_et - entry_dt).days
                    if held >= max_days:
                        exit_reason = "time_stop"
                        logger.info("EXIT %s time_stop: held %d days", sym, held)

            # Strategy exit
            if not exit_reason:
                strat_key = "crypto_momentum" if is_crypto else (
                    "trend" if pnl_pct >= 0 else "mean_reversion"
                )
                strat = self._strategies.get(strat_key)
                if strat and strat.check_exit(sym, avg_price, df, self._cfg):
                    exit_reason = f"strategy_{strat_key}"
                    logger.info("EXIT %s strategy signal from %s", sym, strat_key)

            if exit_reason:
                self._do_exit(sym, qty, is_crypto, exit_reason, cur_price, pnl_pct)

    def _do_exit(
        self,
        sym: str,
        qty: float,
        is_crypto: bool,
        reason: str,
        cur_price: float,
        pnl_pct: float,
    ) -> None:
        ok = self._executor.sell(
            symbol=sym,
            qty=qty,
            strategy="exit",
            price=cur_price,
            is_crypto=is_crypto,
            reason=reason,
        )
        if ok:
            self._position_highs.pop(sym, None)
            self._state.save_position_high(sym, 0.0)
            cooldown_hours = self._cfg.cooldown_crypto_hours if is_crypto else self._cfg.cooldown_equity_hours
            cd_until = datetime.now(ET) + timedelta(hours=cooldown_hours)
            self._state.save_cooldown(sym, cd_until, reason=reason)
            pnl_str = f"{pnl_pct*100:+.2f}%"
            logger.info("EXIT OK %s reason=%s pnl=%s", sym, reason, pnl_str)
            send_alert(
                f"EXIT {sym} reason={reason} pnl={pnl_str} @ ${cur_price:.4f}",
                self._cfg,
                level="INFO",
            )

    # ── Entry scanner ─────────────────────────────────────────────────────────

    def _run_scan(self, symbols: list, is_crypto: bool, equity: float, buying_power: float,
                  open_positions: Dict) -> None:
        """Run all strategies over the given symbol set and execute entries."""
        if not symbols:
            return

        halt_level = self._circuit.level
        if halt_level in (HaltLevel.HARD_HALT, HaltLevel.KILLED):
            logger.info("Scan skipped — circuit breaker: %s", halt_level.name)
            return

        try:
            if is_crypto:
                bars = self._fetcher.get_crypto_bars(symbols)
            else:
                bars = self._fetcher.get_stock_bars(symbols)
        except Exception as e:
            logger.warning("Bar fetch failed for scan: %s", e)
            return

        cooldowns = self._state.get_cooldowns()
        now_et = datetime.now(ET)

        for strat_name, strat in self._strategies.items():
            if is_crypto and strat_name != "crypto_momentum":
                continue
            if not is_crypto and strat_name == "crypto_momentum":
                continue

            # Regime weight gate for new entries
            if halt_level == HaltLevel.SOFT_HALT and not is_crypto:
                continue

            try:
                signals = strat.generate_signals(bars, self._cfg)
            except Exception as e:
                logger.warning("Signal generation error %s: %s", strat_name, e)
                continue

            for sig in signals:
                sym = sig.symbol

                # Already in position
                if sym in open_positions:
                    continue

                # Cooldown check (get_cooldowns returns {sym: datetime} already filtered)
                if sym in cooldowns:
                    continue

                # Max concurrent positions
                if len(open_positions) >= self._cfg.max_concurrent_positions:
                    break

                # Regime weight (scale down signal strength for off-regime)
                regime_scale = self._regime.strategy_weight(strat_name)

                # Position sizing
                approved, reason, qty = self._sizer.check_signal(
                    signal=sig,
                    equity=equity,
                    buying_power=buying_power,
                    open_positions=open_positions,
                    regime_scale=regime_scale,
                )
                if not approved:
                    logger.debug("Signal rejected %s %s: %s", strat_name, sym, reason)
                    continue

                ok = self._executor.buy(
                    symbol=sym,
                    qty=qty,
                    strategy=strat_name,
                    price=sig.price,
                    is_crypto=is_crypto,
                    reason=f"sig={sig.reason or strat_name}",
                )
                if ok:
                    self._position_highs[sym] = sig.price
                    self._state.save_position_high(sym, sig.price)
                    self._state.save_position_age(sym, now_et, strategy=strat_name)
                    send_alert(
                        f"BUY {sym} qty={qty:.4f} @ ${sig.price:.4f} "
                        f"strategy={strat_name} reason={sig.reason or ''}",
                        self._cfg,
                        level="INFO",
                    )

    # ── Regime update ─────────────────────────────────────────────────────────

    def _update_regime(self) -> None:
        try:
            spy_bars = self._fetcher.get_stock_bars(["SPY"])
            spy_df = spy_bars.get("SPY")
            if spy_df is not None and len(spy_df) >= 60:
                self._regime.update(spy_df)
                logger.info("Regime: %s", self._regime.regime)
        except Exception as e:
            logger.warning("Regime update failed: %s", e)

    # ── Screener ──────────────────────────────────────────────────────────────

    def _run_screener(self, open_positions: Dict) -> None:
        if self._state.get_screener_ran_today():
            return
        logger.info("Running screener...")
        try:
            added, dropped = self._screener.run(held_symbols=set(open_positions.keys()))
            if added or dropped:
                logger.info("Screener: +%s -%s", added, dropped)
            # Refresh dynamic watchlist in config
            self._cfg.dynamic_equities = self._state.get_dynamic_symbols("equity")
            self._cfg.dynamic_crypto = self._state.get_dynamic_symbols("crypto")
            self._state.mark_screener_ran()
        except Exception as e:
            logger.warning("Screener error: %s", e)

    # ── Daily audit ───────────────────────────────────────────────────────────

    def _run_audit(self, account: Optional[Dict], open_positions: Dict) -> None:
        # DB-backed guard so a restart after delivery doesn't resend the report
        if self._audit_done_today or self._state.get_audit_ran_today():
            return
        logger.info("Generating daily audit report...")
        try:
            report_data = self._audit.generate(
                account=account,
                open_positions=open_positions,
                circuit_status=self._circuit.level.name,
                circuit_reason=self._circuit.halt_reason(),
            )
            text = self._audit.format_text(report_data)
            self._audit.write_to_file(text)
            send_report(text, self._cfg)
            self._state.mark_audit_ran()
            self._audit_done_today = True
        except Exception as e:
            logger.warning("Audit error: %s", e)

    # ── Daily reset ───────────────────────────────────────────────────────────

    def _maybe_daily_reset(self) -> None:
        now_et = datetime.now(ET)
        today_str = now_et.strftime("%Y-%m-%d")
        if (
            now_et.hour == DAILY_RESET_HOUR_ET
            and self._last_daily_reset_date != today_str
        ):
            logger.info("Daily reset at 4 AM ET")
            self._circuit.reset_daily()
            self._audit_done_today = False
            self._screener_done_today = False
            self._last_daily_reset_date = today_str

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self) -> None:
        _banner(self._cfg)
        self._load_state()
        server.start_server(port=int(os.environ.get("PORT", "8080")))

        # Warm-up: fetch initial account state
        account = self._get_account()
        equity = float(account["equity"]) if account else 0.0
        self._state.save_equity_snapshot(equity)
        self._circuit.check_equity(equity)

        logger.info("Bot started. equity=%.2f  paper=%s", equity, self._cfg.paper)
        send_alert(
            f"Bot started. equity=${equity:.2f} mode={'PAPER' if self._cfg.paper else 'LIVE'}",
            self._cfg,
            level="INFO",
        )

        # Warm-up: adopt any positions placed while the bot was offline, then regime
        try:
            self._reconcile_positions(self._get_open_positions())
        except Exception as e:
            logger.warning("Startup reconciliation failed: %s", e)
        self._update_regime()

        while self._running:
            tick_start = time.monotonic()

            try:
                self._tick()
            except Exception as e:
                logger.error("Unhandled tick error: %s", e, exc_info=True)
                self._state.log_error("main_loop", str(e))

            elapsed = time.monotonic() - tick_start
            sleep_time = max(0, TICK_SECONDS - elapsed)
            time.sleep(sleep_time)

        logger.info("Bot stopped.")

    def _tick(self) -> None:
        now = time.monotonic()
        now_et = datetime.now(ET)

        # Hot-reload config
        try:
            self._cfg.reload()
        except Exception as e:
            logger.warning("Config reload error: %s", e)

        # Fetch live account state
        account = self._get_account()
        if account is None:
            logger.warning("Tick skipped — could not fetch account")
            return

        equity = float(account["equity"])
        buying_power = float(account["buying_power"])

        # Circuit breaker check
        self._circuit.check_equity(equity)
        if self._circuit.level == HaltLevel.KILLED:
            logger.warning("Bot is KILLED. No scans. Monitoring only.")
            self._update_server_status(account, {})
            return

        # Open positions
        open_positions = self._get_open_positions()

        # Adopt any manually-placed trades into the managed watchlist
        self._reconcile_positions(open_positions)

        # Exit checker (every tick)
        self._check_exits(open_positions, equity)

        # Daily reset at 4 AM ET
        self._maybe_daily_reset()

        # Equity snapshot (for PnL tracking)
        self._state.save_equity_snapshot(equity)

        # ── Pre-market window (7–9:30 AM ET, weekdays): regime + screener + audit ─
        is_weekday = now_et.weekday() < 5
        in_premarket_window = (7 <= now_et.hour < 9) or (now_et.hour == 9 and now_et.minute < 30)
        is_premarket = is_weekday and in_premarket_window
        if is_premarket:
            if now - self._last_regime_update >= REGIME_UPDATE_INTERVAL:
                self._update_regime()
                self._last_regime_update = now
            self._run_screener(open_positions)
            if now_et.hour == AUDIT_HOUR_ET:
                self._run_audit(account, open_positions)

        # ── Regime update during market hours ─────────────────────────────────
        market_open = False
        try:
            market_open = self._clock.is_market_open()
        except Exception as e:
            logger.warning("Clock error: %s", e)

        if market_open and now - self._last_regime_update >= REGIME_UPDATE_INTERVAL:
            self._update_regime()
            self._last_regime_update = now

        # ── Crypto scan (24/7) ────────────────────────────────────────────────
        if now - self._last_crypto_scan >= CRYPTO_SCAN_INTERVAL:
            crypto_symbols = self._cfg.all_crypto
            if crypto_symbols:
                self._run_scan(
                    symbols=crypto_symbols,
                    is_crypto=True,
                    equity=equity,
                    buying_power=buying_power,
                    open_positions=open_positions,
                )
            self._last_crypto_scan = now

        # ── Equity scan (market hours only) ──────────────────────────────────
        if market_open and now - self._last_equity_scan >= EQUITY_SCAN_INTERVAL:
            equity_symbols = self._cfg.all_equities
            if equity_symbols:
                self._run_scan(
                    symbols=equity_symbols,
                    is_crypto=False,
                    equity=equity,
                    buying_power=buying_power,
                    open_positions=open_positions,
                )
            self._last_equity_scan = now

        self._update_server_status(account, open_positions)

    def _update_server_status(self, account: Optional[Dict], open_positions: Dict) -> None:
        server.update_status({
            "bot": "running" if self._running else "stopped",
            "equity": account.get("equity") if account else None,
            "buying_power": account.get("buying_power") if account else None,
            "open_positions": len(open_positions),
            "risk_status": self._circuit.level.name,
            "risk_reason": self._circuit.halt_reason(),
            "paper_mode": self._cfg.paper,
            "last_equity_scan": datetime.fromtimestamp(self._last_equity_scan, tz=ET).isoformat()
            if self._last_equity_scan > 0 else None,
            "last_crypto_scan": datetime.fromtimestamp(self._last_crypto_scan, tz=ET).isoformat()
            if self._last_crypto_scan > 0 else None,
            "regime": self._regime.regime,
        })

    def shutdown(self) -> None:
        logger.info("Shutting down gracefully...")
        self._running = False


def main() -> None:
    bot = Bot()

    def _handle_signal(sig, frame):
        logger.info("Signal %s received — shutting down", sig)
        bot.shutdown()

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    bot.run()


if __name__ == "__main__":
    main()
