"""
TradingEngine — orchestrates all components into a unified trading loop.

Flow per equity cycle (every 5 min, market hours):
  1. Fetch latest bars for all equity symbols
  2. Run enabled strategies → raw signals
  3. Apply regime filter weights
  4. Apply optional AI rank multiplier
  5. Deduplicate (one signal per symbol, highest strength wins)
  6. Check circuit breakers
  7. For each signal: RiskManager.check_signal() → OrderEngine.buy/sell

Flow per crypto cycle (every 5 min, 24/7):
  Same but using 15-minute bars and crypto symbols only.

Exit checks (every 2 min):
  For every open position, ask its originating strategy check_exit().
  If exit → OrderEngine.close_partial() with full qty (with retries & fill verify).
  Uses 15-minute bars for crypto, daily bars for equity.

Circuit breaker check (every 1 min):
  Pull daily/weekly P&L from PortfolioTracker.
  Feed to CircuitBreaker.check_daily_loss() / check_weekly_loss().
  If full halt → close_all_positions().
"""
from __future__ import annotations

import math
import time
from datetime import datetime, timedelta, time as dtime
from typing import Dict, List, Optional

import pytz

import config
from data.market_data import MarketDataClient
from data.universe import Universe
from execution.order_engine import OrderEngine
from portfolio.tracker import PortfolioTracker
from risk.circuit_breaker import CircuitBreaker, HaltLevel
from risk.risk_manager import RiskManager
from strategies.base import Signal
from strategies.crypto_momentum import CryptoMomentum
from strategies.mean_reversion import MeanReversion
from strategies.opening_range_breakout import OpeningRangeBreakout
from strategies.regime_filter import RegimeFilter
from strategies.sector_rotation import SectorRotation
from strategies.trend_following import TrendFollowing
from strategies.volatility_breakout import VolatilityBreakout
from utils.alerts import alert_daily_summary
from utils.logger import log

# Optional AI components
from ai.trade_journal import TradeJournal
from ai.position_sizer import AIPositionSizer
from ai.strategy_allocator import AIStrategyAllocator
from ai.exit_optimizer import AIExitOptimizer
from ai.sentiment import SentimentAnalyzer
from ai.scorecard import TechnicalScorecard

from alpaca.data.timeframe import TimeFrame

ET = pytz.timezone(config.TIMEZONE)


class TradingEngine:
    def __init__(self) -> None:
        log.info("=" * 60)
        log.info("Initialising TradingEngine (paper={})", config.ALPACA_PAPER)

        self._universe  = Universe()
        self._data      = MarketDataClient()
        self._portfolio = PortfolioTracker()
        self._breaker   = CircuitBreaker()
        self._regime    = RegimeFilter()
        self._risk      = RiskManager(self._regime)
        self._exec      = OrderEngine(portfolio=self._portfolio)

        # Strategy registry
        self._equity_strategies = []
        self._crypto_strategies = []

        if config.ENABLE_TREND_FOLLOWING:
            self._equity_strategies.append(TrendFollowing())
        if config.ENABLE_MEAN_REVERSION:
            self._equity_strategies.append(MeanReversion())
        if config.ENABLE_VOLATILITY_BREAKOUT:
            self._equity_strategies.append(VolatilityBreakout())
        if config.ENABLE_SECTOR_ROTATION:
            self._equity_strategies.append(SectorRotation())
        if config.ENABLE_CRYPTO_MOMENTUM:
            self._crypto_strategies.append(CryptoMomentum())
        if config.ENABLE_OPENING_RANGE:
            self._equity_strategies.append(OpeningRangeBreakout())

        # Map symbol → strategy name for exit tracking
        self._position_strategy: Dict[str, str] = {}
        # Track high-water marks for trailing stops: symbol → highest seen close
        self._position_high: Dict[str, float] = {}
        # Track position entry times for time-based exits
        self._position_opened: Dict[str, datetime] = {}
        # Cooldown after stop-outs: symbol → datetime when re-entry is allowed.
        # Persisted to SQLite so it survives Railway redeploys (otherwise the
        # bot would re-buy stopped-out symbols on every push).
        self._cooldown_until: Dict[str, datetime] = {}
        self._init_cooldown_table()
        self._load_cooldowns()
        self._seed_position_strategies()
        # VIX data for dynamic risk scaling
        self._vix_level: float = 20.0  # default neutral VIX

        # Optional AI layer
        self._ai_ranker = None
        self._ai_regime = None
        self._ai_journal = TradeJournal()  # always-on — records all trades
        self._ai_position_sizer = None
        self._ai_strategy_alloc = None
        self._ai_exit_optimizer = None
        self._ai_sentiment = None
        self._ai_scorecard = None
        # Set far in the past so the first _refresh_ai_models() in __init__ actually runs
        self._last_ai_refresh = datetime(2000, 1, 1, tzinfo=ET)
        if config.ENABLE_AI_LAYER:
            self._init_ai()

        # Inject AI exit optimizer into all strategies (needed for adaptive stops)
        for strat in self._equity_strategies + self._crypto_strategies:
            if hasattr(strat, "attach_exit_optimizer"):
                strat.attach_exit_optimizer(self._ai_exit_optimizer)
                log.debug("Injected AI exit optimizer into {}", strat.name)

        # Update regime immediately
        self.update_regime()

        log.info(
            "TradingEngine ready — {} equity strategies, {} crypto strategies",
            len(self._equity_strategies), len(self._crypto_strategies),
        )

    # ── Scheduled jobs ────────────────────────────────────────────────────────

    def run_equity_strategies(self) -> None:
        if not self._is_market_hours():
            return
        if self._breaker.is_halted():
            log.info("Equity strategies skipped — circuit breaker active ({})", self._breaker.get_level())
            return

        log.info("Running equity strategies...")
        bars = self._data.get_stock_bars(
            self._universe.equities,
            timeframe=TimeFrame.Day,
            lookback_days=config.BARS_LOOKBACK_DAYS,
        )
        if not bars:
            log.warning("No equity bars received")
            return

        all_signals: List[Signal] = []
        for strat in self._equity_strategies:
            if not strat.enabled:
                continue
            weight = self._regime.strategy_weight(strat.name)
            # Apply AI strategy allocator multiplier
            if config.ENABLE_AI_LAYER and config.ENABLE_AI_STRATEGY_ALLOC and self._ai_strategy_alloc:
                alloc_w = self._ai_strategy_alloc.get_weight(strat.name)
                if alloc_w > 0:
                    weight *= alloc_w * len(self._equity_strategies)
            if weight <= 0:
                continue
            try:
                sigs = strat.generate_signals(bars)
                for s in sigs:
                    if s.side == "buy":
                        s.strength *= weight
                all_signals.extend(sigs)
            except Exception as e:
                log.error("Strategy {} error: {}", strat.name, e)

        self._execute_signals(all_signals)

    def run_opening_range(self) -> None:
        """Run the ORB strategy on 1-minute bars during market hours."""
        if not self._is_market_hours():
            return
        if self._breaker.is_halted():
            return
        if not config.ENABLE_OPENING_RANGE:
            return

        log.info("Running ORB strategy (1-min bars)...")
        bars = self._data.get_stock_bars(
            self._universe.equities,
            timeframe=TimeFrame.Minute,
            lookback_days=5,
        )
        if not bars:
            log.warning("No 5-min bars for ORB")
            return

        orb_strat = self._get_strategy("opening_range_breakout")
        if orb_strat is None:
            return

        weight = self._regime.strategy_weight(orb_strat.name)
        # Apply AI strategy allocator multiplier
        if config.ENABLE_AI_LAYER and config.ENABLE_AI_STRATEGY_ALLOC and self._ai_strategy_alloc:
            alloc_w = self._ai_strategy_alloc.get_weight(orb_strat.name)
            if alloc_w > 0:
                weight *= alloc_w * len(self._equity_strategies)
        if weight <= 0:
            return

        try:
            sigs = orb_strat.generate_signals(bars)
            for s in sigs:
                if s.side == "buy":
                    s.strength *= weight
            self._execute_signals(sigs)
        except Exception as e:
            log.error("ORB strategy error: {}", e)

    def run_crypto_strategies(self) -> None:
        if self._breaker.full_halt_active():
            return

        log.info("Running crypto strategies...")
        bars = self._data.get_crypto_bars(
            self._universe.crypto,
            timeframe=TimeFrame.Hour,
            lookback_days=config.CRYPTO_BARS_LOOKBACK,
        )
        log.info("Crypto bars received for {} symbols: {}", len(bars), list(bars.keys()))
        if not bars:
            log.warning("No crypto bars returned — skipping")
            return

        all_signals: List[Signal] = []
        for strat in self._crypto_strategies:
            weight = self._regime.strategy_weight(strat.name)
            # Apply AI strategy allocator multiplier
            if config.ENABLE_AI_LAYER and config.ENABLE_AI_STRATEGY_ALLOC and self._ai_strategy_alloc:
                alloc_w = self._ai_strategy_alloc.get_weight(strat.name)
                if alloc_w > 0:
                    weight *= alloc_w * len(self._crypto_strategies)
            if weight <= 0:
                continue
            try:
                sigs = strat.generate_signals(bars)
                for s in sigs:
                    if s.side == "buy":
                        s.strength *= weight
                all_signals.extend(sigs)
            except Exception as e:
                log.error("Crypto strategy {} error: {}", strat.name, e)

        self._execute_signals(all_signals)

    def run_sector_rotation(self) -> None:
        if self._breaker.is_halted():
            return
        log.info("Running weekly sector rotation...")
        bars = self._data.get_stock_bars(
            self._universe.equities,
            timeframe=TimeFrame.Day,
            lookback_days=60,
        )
        if not bars:
            return
        rotation = SectorRotation()
        signals  = rotation.generate_signals(bars)
        self._execute_signals(signals)

    def check_all_exits(self) -> None:
        positions = self._portfolio.get_open_positions()
        if not positions:
            return

        for sym, pos in positions.items():
            is_crypto = self._universe.is_crypto(sym)

            # Skip equity checks outside market hours
            if not is_crypto and not self._is_market_hours():
                continue

            strat_name = self._position_strategy.get(sym)
            strat = self._get_strategy(strat_name)
            if strat is None:
                continue

            try:
                # Fetch recent bars — use intraday bars for ORB strategy
                if is_crypto:
                    bars_dict = self._data.get_crypto_bars([sym], TimeFrame.Hour, lookback_days=30)
                elif strat_name == "opening_range_breakout":
                    bars_dict = self._data.get_stock_bars([sym], TimeFrame.Minute, lookback_days=5)
                else:
                    bars_dict = self._data.get_stock_bars([sym], TimeFrame.Day, lookback_days=40)

                df = bars_dict.get(sym)
                if df is None:
                    continue

                qty = pos.get("qty", 0)
                cur_price = float(df["close"].iloc[-1]) if len(df) > 0 else pos["avg_price"]
                entry_price = pos.get("avg_price", 0)

                # Compute ATR early — needed for both hard stop ATR check and trailing stop
                atr = self._atr_from_df(df)

                # Update high-water mark for trailing stop
                prev_high = self._position_high.get(sym, entry_price)
                new_high  = max(prev_high, cur_price)
                self._position_high[sym] = new_high

                exit_reason: Optional[str] = None

                # 1. Hard % stop — absolute floor on losses (crypto gets more room)
                pnl_pct = 0.0
                if entry_price > 0:
                    pnl_pct = (cur_price - entry_price) / entry_price
                    hard_limit = config.HARD_STOP_PCT_CRYPTO if is_crypto else config.HARD_STOP_PCT
                    if pnl_pct <= -hard_limit:
                        log.warning("{} HARD STOP — down {:.1%} from entry {:.4f} (limit {:.1%})",
                                    sym, pnl_pct, entry_price, hard_limit)
                        exit_reason = "hard_stop"

                # 1b. ATR-based stop — fires independently (not elif of hard stop)
                # Guard against NaN atr
                if exit_reason is None and atr > 0 and math.isfinite(atr) and entry_price > 0:
                    if cur_price < entry_price - config.ATR_STOP_MULT * atr:
                        log.warning("{} ATR STOP — down {:.1%} from entry ({}×ATR)",
                                    sym, pnl_pct, config.ATR_STOP_MULT)
                        exit_reason = "atr_stop"

                # 2. Time stop — kill stale positions that haven't produced
                if exit_reason is None and entry_price > 0:
                    pos_opened = self._position_opened.get(sym)
                    if pos_opened:
                        try:
                            age = datetime.now(ET) - pos_opened
                            pnl_pct_stale = (cur_price - entry_price) / entry_price if entry_price > 0 else 0
                            # Use AI-optimized time stop if available
                            time_stop_days = config.TIME_STOP_DAYS
                            if config.ENABLE_AI_LAYER and config.ENABLE_AI_EXIT_OPT and self._ai_exit_optimizer:
                                try:
                                    time_stop_days = self._ai_exit_optimizer.adjust_time_stop_days(
                                        strat_name, config.TIME_STOP_DAYS
                                    )
                                except Exception:
                                    pass
                            # Time stop: close if held too long AND not showing enough profit
                            if age.days >= time_stop_days and pnl_pct_stale < 0.02:
                                log.info("{} TIME STOP — {} days old with {:.1%} pnl (limit {}d)",
                                         sym, age.days, pnl_pct_stale, time_stop_days)
                                exit_reason = "time_stop"
                            # Also exit on time stop if down significantly (stop-out alternative)
                            elif age.days >= max(3, time_stop_days // 2) and pnl_pct_stale < -0.015:
                                log.info("{} EARLY TIME STOP — {} days old with {:.1%} loss",
                                         sym, age.days, pnl_pct_stale)
                                exit_reason = "early_time_stop"
                        except Exception:
                            pass

                # 3. Trailing stop — once up TRAIL_ARM_PCT, lock in gains
                if exit_reason is None and entry_price > 0:
                    pnl_pct = (new_high - entry_price) / entry_price
                    if pnl_pct >= config.TRAIL_ARM_PCT:
                        # Dynamic trailing stop: use tighter giveback in high-vol environments
                        # Base giveback on ATR as % of price, scaled by current regime
                        if atr > 0 and math.isfinite(atr) and entry_price > 0:
                            # ATR-based giveback: volatile markets = wider giveback, calm = tighter
                            atr_giveback = 2.0 * atr / new_high  # 2× ATR as % of peak
                            # Scale between TRAIL_GIVEBACK_PCT (5%) and 2× ATR giveback
                            trail_pct = min(0.20, max(config.TRAIL_GIVEBACK_PCT, atr_giveback))
                            # Additional VIX scaling: if VIX > 25, widen the trail slightly
                            if self._vix_level > 25:
                                trail_pct = min(0.25, trail_pct * 1.25)
                        else:
                            trail_pct = config.TRAIL_GIVEBACK_PCT
                        trail_stop = new_high * (1 - trail_pct)
                        if cur_price <= trail_stop:
                            log.warning(
                                "{} TRAILING STOP — high={:.4f} cur={:.4f} stop={:.4f} (gains locked)",
                                sym, new_high, cur_price, trail_stop,
                            )
                            exit_reason = "trailing_stop"

                # 4. Strategy exit
                if exit_reason is None and strat.check_exit(sym, entry_price, df):
                    log.info("Exit signal for {} from {}", sym, strat_name)
                    exit_reason = "strategy"

                # 4b. AI take-profit: use AI exit optimizer's tp_ratio
                if exit_reason is None and config.ENABLE_AI_LAYER and config.ENABLE_AI_EXIT_OPT and self._ai_exit_optimizer:
                    tp_ratio = self._ai_exit_optimizer.adjust_tp_ratio(strat_name or "unknown", default_ratio=2.0)
                    if tp_ratio > 0 and entry_price > 0 and atr > 0 and math.isfinite(atr):
                        stop_dist = config.ATR_STOP_MULT * atr
                        tp_target = entry_price + tp_ratio * stop_dist
                        if cur_price >= tp_target:
                            log.info("{} AI TAKE-PROFIT — up {:.1%} (target {:.4f} hit)",
                                     sym, (cur_price - entry_price) / entry_price, tp_target)
                            exit_reason = "ai_take_profit"

                if exit_reason is not None:
                    # Ensure qty is positive for close_partial
                    close_qty = abs(float(qty)) if qty is not None else abs(float(pos.get("qty", 0)))
                    self._exec.close_partial(sym, close_qty, cur_price, is_crypto)
                    # Log exit to AI trade journal AFTER successful close
                    try:
                        self._ai_journal.log_exit(
                            symbol=sym,
                            exit_price=cur_price,
                            exit_reason=exit_reason,
                            entry_price=entry_price,
                            entry_atr=self._atr_from_df(df),
                        )
                    except Exception:
                        pass
                    self._position_strategy.pop(sym, None)
                    self._position_high.pop(sym, None)
                    self._position_opened.pop(sym, None)

                    # Set cooldown after STOP exits so we don't re-buy a falling
                    # knife. Strategy exits (RSI fade, EMA cross) don't get
                    # cooldown — those are normal rotations.
                    if exit_reason in ("hard_stop", "trailing_stop"):
                        cooldown_h = config.COOLDOWN_HOURS_CRYPTO if is_crypto else config.COOLDOWN_HOURS_EQUITY
                        until = datetime.now(ET) + timedelta(hours=cooldown_h)
                        self._cooldown_until[sym] = until
                        self._save_cooldown(sym, until, exit_reason)
                        log.info("{} cooldown {}h until {} (persisted)", sym, cooldown_h,
                                 until.strftime("%Y-%m-%d %H:%M ET"))
                    else:
                        # Strategy exit — fine to refill the slot
                        if is_crypto:
                            self.run_crypto_strategies()
                        elif self._is_market_hours():
                            self.run_equity_strategies()
            except Exception as e:
                log.error("check_all_exits({}) error: {}", sym, e)

    def check_circuit_breakers(self) -> None:
        try:
            state = self._portfolio.get_account_state()
            pv    = state["portfolio_value"]
            daily = self._portfolio.daily_pnl()
            weekly = self._portfolio.weekly_pnl()

            self._breaker.check_daily_loss(daily, pv)
            self._breaker.check_weekly_loss(weekly, pv)

            if self._breaker.full_halt_active():
                log.warning("Full halt active — closing all positions")
                self._exec.close_all_positions()
        except Exception as e:
            log.error("check_circuit_breakers error: {}", e)

    def update_regime(self) -> None:
        try:
            bars = self._data.get_stock_bars(["SPY"], TimeFrame.Day, lookback_days=400)
            spy  = bars.get("SPY")
            self._regime.update(spy)

            if config.ENABLE_AI_LAYER and self._ai_regime:
                state = self._ai_regime.predict(spy) if spy is not None else None
                if state is not None:
                    log.info("AI regime state: {}", state)

            # Fetch VIX for dynamic risk scaling
            self._update_vix()
        except Exception as e:
            log.error("update_regime error: {}", e)

    def daily_open(self) -> None:
        log.info("── Daily open tasks ──")
        self._breaker.reset()
        self.update_regime()
        state = self._portfolio.get_account_state()
        log.info("Account: portfolio=${:.2f} BP=${:.2f}", state["portfolio_value"], state["buying_power"])

    def daily_close(self) -> None:
        log.info("── Daily close tasks ──")
        self._portfolio.snapshot()
        state     = self._portfolio.get_account_state()
        positions = self._portfolio.get_open_positions()
        pnl       = self._portfolio.daily_pnl()
        alert_daily_summary(state["portfolio_value"], pnl, len(positions))
        log.info(self._portfolio.summary())

    def portfolio_heartbeat(self) -> None:
        try:
            log.info(self._portfolio.summary())
        except Exception as e:
            log.warning("portfolio_heartbeat error: {}", e)

    def manage_open_positions(self) -> None:
        """
        Active position management — runs every 30 min.

        Scale-out: if unrealized loss > 1.5 × ATR × qty → close 50% of position.
        Pyramid:   if unrealized PnL > 0 AND originating strategy still fires a
                   buy signal AND position is below 1.5× original size → add 50%.
        """
        positions = self._portfolio.get_open_positions()
        if not positions:
            return

        log.info("Position management scan — {} open positions", len(positions))
        state = self._portfolio.get_account_state()

        for sym, pos in list(positions.items()):
            is_crypto = self._universe.is_crypto(sym)

            if not is_crypto and not self._is_market_hours():
                continue

            try:
                if is_crypto:
                    bars_dict = self._data.get_crypto_bars([sym], TimeFrame.Hour, lookback_days=30)
                else:
                    bars_dict = self._data.get_stock_bars([sym], TimeFrame.Day, lookback_days=40)

                df = bars_dict.get(sym)
                if df is None or len(df) < 10:
                    continue

                unrealized_pl = pos.get("unrealized_pl", 0.0)
                qty           = pos.get("qty", 0.0)
                avg_price     = pos.get("avg_price", 0.0)
                market_value  = pos.get("market_value", 0.0)

                atr = self._atr_from_df(df)

                # ── Scale-out: cut losers ─────────────────────────────────────
                if atr > 0 and unrealized_pl < -(1.5 * atr * qty):
                    scale_qty = qty * 0.5
                    scale_qty = round(scale_qty, 4) if is_crypto else math.floor(scale_qty)
                    if scale_qty > 0:
                        log.warning(
                            "Scale-out {}: uPnL={:.2f} < -1.5×ATR×qty ({:.2f}) — closing 50%",
                            sym, unrealized_pl, -(1.5 * atr * qty),
                        )
                        # Use current price (cur_price from df), not avg_price
                        cur_p = float(df["close"].iloc[-1]) if len(df) > 0 else avg_price
                        self._exec.close_partial(sym, scale_qty, cur_p, is_crypto)
                        continue  # Skip pyramid check this iteration after scale-out

                # ── Pyramid: add to winners ───────────────────────────────────
                elif (
                    unrealized_pl > 0
                    and self._breaker.trading_allowed()
                    and market_value < state["portfolio_value"] * config.MAX_POSITION_PCT * 1.45
                ):
                    strat_name = self._position_strategy.get(sym)
                    strat = self._get_strategy(strat_name)
                    if strat is None:
                        continue

                    # AI-guided pyramid: check scorecard and sentiment
                    ai_confidence = 1.0
                    if self._ai_scorecard:
                        ai_confidence *= self._ai_scorecard.get_multiplier(sym)
                    if self._ai_sentiment:
                        headlines = self._ai_sentiment.get_recent_headlines(sym, hours=48)
                        if headlines:
                            ai_confidence *= self._ai_sentiment.sentiment_multiplier(sym, headlines)

                    # Only pyramid if AI confidence is high enough
                    if ai_confidence < 0.9:
                        log.debug("Pyramid skip {}: AI confidence={:.2f}", sym, ai_confidence)
                        continue

                    try:
                        fresh_signals = strat.generate_signals({sym: df})
                    except Exception:
                        continue

                    buy_sigs = [s for s in fresh_signals if s.symbol == sym and s.side == "buy"]
                    if not buy_sigs:
                        continue

                    sig       = buy_sigs[0]
                    add_value = min(
                        market_value * 0.5,
                        state["portfolio_value"] * config.MAX_POSITION_PCT * 0.5,
                    )
                    # AI-boosted pyramid: scale up when confidence is high
                    if ai_confidence > 1.2:
                        add_value = min(add_value * 1.25, state["portfolio_value"] * config.MAX_POSITION_PCT * 0.625)
                    add_qty = round(add_value / sig.price, 4) if is_crypto else math.floor(add_value / sig.price)

                    if add_qty > 0 and add_value <= state["buying_power"] * 0.9:
                        log.info(
                            "Pyramid {}: uPnL={:.2f} AI_conf={:.2f} — adding {} units",
                            sym, unrealized_pl, ai_confidence, add_qty,
                        )
                        success = self._exec.buy(
                            symbol=sym,
                            qty=add_qty,
                            strategy=sig.strategy + "_pyramid",
                            price=sig.price,
                            is_crypto=is_crypto,
                        )
                        if success:
                            state = self._portfolio.get_account_state()

            except Exception as e:
                log.error("manage_open_positions({}) error: {}", sym, e)

    def theme_rebalance(self) -> None:
        """
        Weekly theme rebalance.

        Themes overweight by >5% of portfolio → trim weakest position first.
        Themes underweight → noted; organic strategy signals fill them naturally.
        """
        if self._breaker.is_halted():
            return

        positions = self._portfolio.get_open_positions()
        if not positions:
            return

        state = self._portfolio.get_account_state()
        pv = state["portfolio_value"]
        if pv <= 0:
            return

        theme_exp: Dict[str, float] = {}
        for sym, pos in positions.items():
            theme = self._universe.theme_for_symbol(sym)
            theme_exp[theme] = theme_exp.get(theme, 0.0) + pos["market_value"]

        n_themes = len(theme_exp)
        if n_themes == 0:
            return

        target_pct = 1.0 / n_themes
        log.info("Theme rebalance — {} themes, target={:.1%} each", n_themes, target_pct)

        for theme, exposure in theme_exp.items():
            current_pct = exposure / pv
            overweight  = current_pct - target_pct

            if overweight > 0.05:
                trim_value = overweight * pv
                theme_pos  = sorted(
                    [(s, p) for s, p in positions.items()
                     if self._universe.theme_for_symbol(s) == theme],
                    key=lambda x: x[1]["unrealized_pl"],
                )
                trimmed = 0.0
                for sym, pos in theme_pos:
                    if trimmed >= trim_value:
                        break
                    is_crypto   = self._universe.is_crypto(sym)
                    close_value = min(pos["market_value"], trim_value - trimmed)
                    close_pct   = close_value / pos["market_value"] if pos["market_value"] > 0 else 1.0

                    # Get current price for the close
                    try:
                        if is_crypto:
                            bars_dict = self._data.get_crypto_bars([sym], TimeFrame.Hour, 5)
                        else:
                            bars_dict = self._data.get_stock_bars([sym], TimeFrame.Day, 5)
                        df_sym = bars_dict.get(sym)
                        close_price = float(df_sym["close"].iloc[-1]) if df_sym is not None and len(df_sym) > 0 else pos["avg_price"]
                    except Exception:
                        close_price = pos["avg_price"]

                    if close_pct >= 0.90:
                        log.info("Rebalance: close {} ({} overweight {:.1%})", sym, theme, overweight)
                        self._exec.close_partial(sym, pos["qty"], close_price, is_crypto)
                        self._position_strategy.pop(sym, None)
                        self._position_high.pop(sym, None)
                        self._position_opened.pop(sym, None)
                    else:
                        close_qty = pos["qty"] * close_pct
                        close_qty = round(close_qty, 4) if is_crypto else math.floor(close_qty)
                        if close_qty > 0:
                            log.info("Rebalance: trim {} {} ({} OW {:.1%})", sym, close_qty, theme, overweight)
                            self._exec.close_partial(sym, close_qty, close_price, is_crypto)

                    trimmed += close_value

            elif overweight < -0.05:
                log.info("Theme {} underweight {:.1%} — strategies will fill organically", theme, -overweight)

    # ── Signal execution ──────────────────────────────────────────────────────

    def _execute_signals(self, signals: List[Signal]) -> None:
        if not signals:
            return

        # Deduplicate — one signal per symbol, highest strength wins.
        # Confluence tracks how many strategies generated a BUY, but
        # sell signals still compete on strength (sector rotation exits
        # must be able to override buy signals).
        best: Dict[str, Signal] = {}
        confluence: Dict[str, int] = {}
        for sig in signals:
            if sig.side == "buy":
                confluence[sig.symbol] = confluence.get(sig.symbol, 0) + 1
            if sig.symbol not in best or sig.strength > best[sig.symbol].strength:
                best[sig.symbol] = sig

        state     = self._portfolio.get_account_state()
        positions = self._portfolio.get_open_positions()

        for sym, sig in sorted(best.items(), key=lambda x: x[1].strength, reverse=True):
            if self._breaker.is_halted() and sig.side == "buy":
                break

            # Cooldown check — skip buys for symbols recently stopped out
            if sig.side == "buy":
                cooldown = self._cooldown_until.get(sym)
                if cooldown and datetime.now(ET) < cooldown:
                    log.info("Signal SKIPPED ({}) — cooldown until {}",
                             sym, cooldown.strftime("%H:%M ET"))
                    continue
                elif cooldown:
                    # Cooldown expired — clean up
                    self._cooldown_until.pop(sym, None)

            # Apply AI rank multiplier
            if config.ENABLE_AI_LAYER and self._ai_ranker and sig.side == "buy":
                try:
                    bars_dict = (
                        self._data.get_crypto_bars([sym], TimeFrame.Hour, 30)
                        if sig.is_crypto
                        else self._data.get_stock_bars([sym], TimeFrame.Day, 30)
                    )
                    df = bars_dict.get(sym)
                    if df is not None:
                        mult = self._ai_ranker.rank_multiplier(sym, df)
                        sig.strength = min(1.0, sig.strength * mult)
                except Exception:
                    pass

            # Apply technical scorecard multiplier (free AI — no external APIs)
            if config.ENABLE_AI_LAYER and self._ai_scorecard and sig.side == "buy":
                try:
                    bars_dict = (
                        self._data.get_crypto_bars([sym], TimeFrame.Hour, 30)
                        if sig.is_crypto
                        else self._data.get_stock_bars([sym], TimeFrame.Day, 30)
                    )
                    df = bars_dict.get(sym)
                    if df is not None:
                        score_mult = self._ai_scorecard.get_multiplier(sym)
                        sig.strength = min(1.0, sig.strength * score_mult)
                        if score_mult != 1.0:
                            log.debug("Scorecard {}: score={:.0f} mult={:.2f}",
                                     sym, self._ai_scorecard.get_score(sym), score_mult)
                except Exception:
                    pass

            # Apply sentiment multiplier (free AI)
            if config.ENABLE_AI_LAYER and self._ai_sentiment and sig.side == "buy":
                try:
                    headlines = self._ai_sentiment.get_recent_headlines(sym, hours=48)
                    if headlines:
                        sent_mult = self._ai_sentiment.sentiment_multiplier(sym, headlines)
                        sig.strength = min(1.0, sig.strength * sent_mult)
                        log.debug("Sentiment {}: score={:.2f} mult={:.2f}",
                                 sym, self._ai_sentiment.get_sentiment(sym), sent_mult)
                except Exception:
                    pass

            if sig.side == "sell":
                # Sell signal (e.g. sector rotation exit) — only act if we hold it
                if sym in positions:
                    pos = positions[sym]
                    is_crypto = self._universe.is_crypto(sym)
                    # Log exit to AI trade journal
                    try:
                        self._ai_journal.log_exit(
                            symbol=sym,
                            exit_price=sig.price,
                            exit_reason="strategy_sell_signal",
                            entry_price=pos.get("avg_price", 0),
                        )
                    except Exception:
                        pass
                    self._exec.close_partial(sym, pos["qty"], sig.price, is_crypto)
                    self._position_strategy.pop(sym, None)
                    self._position_high.pop(sym, None)
                    self._position_opened.pop(sym, None)
                continue

            # Buy signal — apply AI position sizer multiplier
            ai_mult = 1.0
            if config.ENABLE_AI_LAYER and config.ENABLE_AI_POSITION_SIZER and self._ai_position_sizer:
                try:
                    regime = self._regime.regime or "unknown"
                    ai_mult = self._ai_position_sizer.size_multiplier(sig.strategy, regime)
                except Exception:
                    pass

            # VIX-based risk scaling: reduce exposure when VIX is high
            vix_scale = self._vix_scale()

            # Respect VIX_RISK_SCALE config flag — don't scale positions if disabled
            if not config.VIX_RISK_SCALE:
                vix_scale = 1.0

            # Run through risk manager
            approved, reason, qty = self._risk.check_signal(
                sig,
                state["portfolio_value"],
                state["buying_power"],
                positions,
                self._portfolio.daily_pnl(),
                confluence.get(sym, 1),
                ai_mult * vix_scale,
            )
            if not approved:
                log.info("Signal REJECTED ({}) — {}: {}", sym, sig.strategy, reason)
                continue

            success = self._exec.buy(
                symbol=sym,
                qty=qty,
                strategy=sig.strategy,
                price=sig.price,
                is_crypto=sig.is_crypto,
            )
            if success:
                self._position_strategy[sym] = sig.strategy
                self._position_opened[sym] = datetime.now(ET)
                # Log entry to AI trade journal
                try:
                    self._ai_journal.log_entry(
                        symbol=sym,
                        strategy=sig.strategy,
                        sector=self._universe.theme_for_symbol(sym),
                        entry_price=sig.price,
                        entry_atr=sig.atr,
                        signal_strength=sig.strength,
                        regime=self._regime.regime or "unknown",
                        realized_vol=sig.atr / sig.price if sig.price > 0 else 0,
                        confluence=confluence.get(sym, 1),
                    )
                except Exception:
                    pass
                # Refresh state after each order
                state     = self._portfolio.get_account_state()
                positions = self._portfolio.get_open_positions()

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _is_market_hours(self) -> bool:
        now = datetime.now(ET)
        if now.weekday() >= 5:
            return False
        return dtime(9, 35) <= now.time() <= dtime(15, 55)

    def _seed_position_strategies(self) -> None:
        """
        On startup, assign a default strategy to any existing position that
        isn't already tracked. Without this, pre-existing or cross-restart
        positions are invisible to check_all_exits and never get stop-loss
        or momentum-exit checks applied.
        """
        try:
            positions = self._portfolio.get_open_positions()
            for sym in positions:
                if sym not in self._position_strategy:
                    default = "crypto_momentum" if self._universe.is_crypto(sym) else "trend_following"
                    self._position_strategy[sym] = default
                    log.info("Seeded exit tracking: {} → {}", sym, default)
                # Don't seed _position_high here — let check_all_exits set a real
                # high-water mark from live price data. Seeding with avg_price (entry)
                # would cause the trailing stop to immediately arm on any profitable
                # position held overnight, which is wrong.
        except Exception as e:
            log.warning("_seed_position_strategies error: {}", e)

    # ── Cooldown persistence (survives Railway redeploys) ────────────────────

    def _init_cooldown_table(self) -> None:
        try:
            import sqlite3
            with sqlite3.connect(config.DB_PATH) as c:
                c.execute("""
                    CREATE TABLE IF NOT EXISTS symbol_cooldowns (
                        symbol     TEXT PRIMARY KEY,
                        until_iso  TEXT NOT NULL,
                        reason     TEXT
                    )
                """)
        except Exception as e:
            log.warning("_init_cooldown_table error: {}", e)

    def _load_cooldowns(self) -> None:
        try:
            import sqlite3
            now = datetime.now(ET)
            with sqlite3.connect(config.DB_PATH) as c:
                c.row_factory = sqlite3.Row
                rows = c.execute("SELECT symbol, until_iso FROM symbol_cooldowns").fetchall()
                for row in rows:
                    try:
                        until = datetime.fromisoformat(row["until_iso"])
                        if until > now:
                            self._cooldown_until[row["symbol"]] = until
                        else:
                            c.execute("DELETE FROM symbol_cooldowns WHERE symbol=?", (row["symbol"],))
                    except Exception:
                        pass
            if self._cooldown_until:
                log.info("Loaded {} active cooldowns from DB", len(self._cooldown_until))
        except Exception as e:
            log.warning("_load_cooldowns error: {}", e)

    def _save_cooldown(self, symbol: str, until: datetime, reason: str) -> None:
        try:
            import sqlite3
            with sqlite3.connect(config.DB_PATH) as c:
                c.execute("""
                    INSERT OR REPLACE INTO symbol_cooldowns (symbol, until_iso, reason)
                    VALUES (?, ?, ?)
                """, (symbol, until.isoformat(), reason))
        except Exception as e:
            log.warning("_save_cooldown error: {}", e)

    def _update_vix(self) -> None:
        """Fetch VIX level for dynamic risk scaling."""
        try:
            vix_bars = self._data.get_stock_bars(["VIXY"], TimeFrame.Day, lookback_days=5)
            vix_df = vix_bars.get("VIXY")
            if vix_df is not None and len(vix_df) > 0:
                self._vix_level = float(vix_df["close"].iloc[-1])
                log.debug("VIX level: %.2f", self._vix_level)
            else:
                # Fallback: derive from SPY realized vol
                spy_bars = self._data.get_stock_bars(["SPY"], TimeFrame.Day, lookback_days=20)
                spy_df = spy_bars.get("SPY")
                if spy_df is not None:
                    ret_std = float(spy_df["close"].pct_change().tail(20).std())
                    self._vix_level = max(12.0, ret_std * 100 * (20 ** 0.5))  # annualise
                    log.debug("VIX fallback (SPY-derived): %.2f", self._vix_level)
                else:
                    log.debug("VIX unavailable - using default %.2f", self._vix_level)
        except Exception:
            pass  # keep previous VIX level

    def _vix_scale(self) -> float:
        """Return a position scale factor based on VIX. High VIX = reduce risk."""
        if not config.VIX_RISK_SCALE:
            return 1.0
        vix = self._vix_level
        if vix >= 30:
            return 0.5
        elif vix >= 25:
            return 0.7
        elif vix >= 20:
            return 0.85
        elif vix >= 15:
            return 1.0
        else:
            return 1.15

    def _atr_from_df(self, df) -> float:
        """14-period ATR from an OHLCV dataframe."""
        try:
            import pandas_ta as ta
            atr_s = ta.atr(df["high"], df["low"], df["close"], length=14)
            if atr_s is not None and len(atr_s) > 0:
                val = atr_s.iloc[-1]
                return float(val) if val == val else 0.0
        except Exception:
            pass
        try:
            return float((df["high"] - df["low"]).abs().tail(14).mean())
        except Exception:
            return 0.0

    def _get_strategy(self, name: Optional[str]):
        if not name:
            return None
        all_strats = self._equity_strategies + self._crypto_strategies
        for s in all_strats:
            if s.name == name:
                return s
        return None

    def _init_ai(self) -> None:
        try:
            from ai.asset_ranker import LocalAssetRanker
            ranker = LocalAssetRanker()
            if ranker.is_available():
                if not ranker.load():
                    log.info("AI: training asset ranker on startup...")
                    bars = self._data.get_stock_bars(
                        self._universe.equities, TimeFrame.Day, 504
                    )
                    if bars:
                        ranker.fit(bars)
                self._ai_ranker = ranker
                log.info("AI asset ranker active")
            else:
                log.info("AI: scikit-learn not installed — ranker disabled")
        except Exception as e:
            log.warning("AI ranker init error: {}", e)

        # ── AI Technical Scorecard ─────────────────────────────────────────
        try:
            self._ai_scorecard = TechnicalScorecard()
            log.info("AI technical scorecard active")
        except Exception as e:
            log.warning("AI scorecard init error: {}", e)

        # ── AI Sentiment Analyzer ───────────────────────────────────────────
        if config.ENABLE_AI_LAYER:
            try:
                self._ai_sentiment = SentimentAnalyzer()
                self._ai_sentiment.load()
                log.info("AI sentiment analyzer active")
            except Exception as e:
                log.warning("AI sentiment init error: {}", e)

        try:
            from ai.regime_detector import LocalRegimeDetector
            detector = LocalRegimeDetector()
            if detector.is_available():
                if not detector.load():
                    log.info("AI: training HMM regime detector on startup...")
                    spy_bars = self._data.get_stock_bars(["SPY"], TimeFrame.Day, 600)
                    spy = spy_bars.get("SPY")
                    if spy is not None:
                        detector.fit(spy)
                self._ai_regime = detector
                log.info("AI regime detector active")
            else:
                log.info("AI: hmmlearn not installed — regime detector disabled")
        except Exception as e:
            log.warning("AI regime detector init error: {}", e)

        # ── AI Position Sizer ───────────────────────────────────────────
        if config.ENABLE_AI_POSITION_SIZER:
            try:
                self._ai_position_sizer = AIPositionSizer(self._ai_journal)
                log.info("AI position sizer active")
            except Exception as e:
                log.warning("AI position sizer init error: {}", e)

        # ── AI Strategy Allocator ───────────────────────────────────────
        if config.ENABLE_AI_STRATEGY_ALLOC:
            try:
                self._ai_strategy_alloc = AIStrategyAllocator(self._ai_journal)
                log.info("AI strategy allocator active")
            except Exception as e:
                log.warning("AI strategy allocator init error: {}", e)

        # ── AI Exit Optimizer ───────────────────────────────────────────
        if config.ENABLE_AI_EXIT_OPT:
            try:
                self._ai_exit_optimizer = AIExitOptimizer(self._ai_journal)
                log.info("AI exit optimizer active")
            except Exception as e:
                log.warning("AI exit optimizer init error: {}", e)

    # ── AI model refresh ────────────────────────────────────────────────

    def _refresh_ai_models(self) -> None:
        """
        Refresh all AI models that learn from trade outcomes.
        Called at startup and periodically by the scheduler.
        """
        now = datetime.now(ET)
        if (now - self._last_ai_refresh).total_seconds() < config.AI_REFRESH_MINUTES * 60:
            return
        self._last_ai_refresh = now

        regime = self._regime.regime or "unknown"
        all_strats = [s.name for s in self._equity_strategies + self._crypto_strategies]
        equity_strats = [s.name for s in self._equity_strategies]
        crypto_strats = [s.name for s in self._crypto_strategies]

        if self._ai_position_sizer:
            try:
                self._ai_position_sizer.refresh(all_strats, regime)
            except Exception as e:
                log.warning("AI position sizer refresh error: {}", e)

        if self._ai_strategy_alloc:
            try:
                self._ai_strategy_alloc.refresh(all_strats)
            except Exception as e:
                log.warning("AI strategy allocator refresh error: {}", e)

        if self._ai_exit_optimizer:
            try:
                self._ai_exit_optimizer.refresh(all_strats)
            except Exception as e:
                log.warning("AI exit optimizer refresh error: {}", e)

        # ── AI Technical Scorecard refresh ─────────────────────────────────
        if self._ai_scorecard:
            try:
                bars = self._data.get_stock_bars(
                    self._universe.equities, TimeFrame.Day, 200
                )
                if bars:
                    self._ai_scorecard.update_scores(bars)
                    top = self._ai_scorecard.top_n(3, min_score=50.0)
                    if top:
                        log.info("AI Scorecard top: {}", top)
            except Exception as e:
                log.warning("AI scorecard refresh error: {}", e)

        # ── AI Sentiment refresh — parallel fetching for speed ──────────────
        if self._ai_sentiment:
            try:
                import concurrent.futures
                top_syms = self._universe.equities[:10]
                news_by_symbol: Dict[str, List[str]] = {}

                def _fetch_headlines(sym: str):
                    try:
                        return sym, self._ai_sentiment.get_recent_headlines(sym, hours=48)
                    except Exception:
                        return sym, []

                with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                    futures = {executor.submit(_fetch_headlines, sym): sym for sym in top_syms}
                    for fut in concurrent.futures.as_completed(futures, timeout=20):
                        sym, headlines = fut.result()
                        if headlines:
                            news_by_symbol[sym] = headlines

                if news_by_symbol:
                    self._ai_sentiment.refresh(news_by_symbol)
                    log.debug("AI sentiment refreshed {} symbols", len(news_by_symbol))
            except Exception as e:
                log.warning("AI sentiment refresh error: {}", e)
