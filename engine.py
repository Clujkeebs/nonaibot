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
from datetime import datetime, time as dtime
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
from strategies.regime_filter import RegimeFilter
from strategies.sector_rotation import SectorRotation
from strategies.trend_following import TrendFollowing
from strategies.volatility_breakout import VolatilityBreakout
from utils.alerts import alert_daily_summary
from utils.logger import log

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
        self._exec      = OrderEngine()

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

        # Map symbol → strategy name for exit tracking
        self._position_strategy: Dict[str, str] = {}
        self._seed_position_strategies()

        # Optional AI layer
        self._ai_ranker = None
        self._ai_regime = None
        if config.ENABLE_AI_LAYER:
            self._init_ai()

        # Update regime immediately so we don't start in UNKNOWN for an hour
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

    def run_crypto_strategies(self) -> None:
        if self._breaker.full_halt_active():
            return

        log.info("Running crypto strategies...")
        bars = self._data.get_crypto_bars(
            self._universe.crypto,
            timeframe=TimeFrame.Minute15,
            lookback_days=config.CRYPTO_BARS_LOOKBACK,
        )
        log.info("Crypto bars received for {} symbols: {}", len(bars), list(bars.keys()))
        if not bars:
            log.warning("No crypto bars returned — skipping")
            return

        all_signals: List[Signal] = []
        for strat in self._crypto_strategies:
            weight = self._regime.strategy_weight(strat.name)
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
                # Fetch recent bars
                if is_crypto:
                    bars_dict = self._data.get_crypto_bars([sym], TimeFrame.Minute15, lookback_days=10)
                else:
                    bars_dict = self._data.get_stock_bars([sym], TimeFrame.Day, lookback_days=10)

                df = bars_dict.get(sym)
                if df is None:
                    continue

                if strat.check_exit(sym, pos["avg_price"], df):
                    log.info("Exit signal for {} from {}", sym, strat_name)
                    qty = pos.get("qty", 0)
                    cur_price = float(df["close"].iloc[-1]) if len(df) > 0 else pos["avg_price"]
                    self._exec.close_partial(sym, abs(qty), cur_price, is_crypto)
                    self._position_strategy.pop(sym, None)
                    # Freed a slot — scan immediately for a replacement
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
                    bars_dict = self._data.get_crypto_bars([sym], TimeFrame.Minute15, lookback_days=20)
                else:
                    bars_dict = self._data.get_stock_bars([sym], TimeFrame.Day, lookback_days=20)

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
                        self._exec.close_partial(sym, scale_qty, avg_price, is_crypto)

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
                    add_qty = round(add_value / sig.price, 4) if is_crypto else math.floor(add_value / sig.price)

                    if add_qty > 0 and add_value <= state["buying_power"] * 0.9:
                        log.info(
                            "Pyramid {}: uPnL={:.2f} strategy still bullish — adding {} units",
                            sym, unrealized_pl, add_qty,
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

                    if close_pct >= 0.90:
                        log.info("Rebalance: close {} ({} overweight {:.1%})", sym, theme, overweight)
                        self._exec.close_partial(sym, pos["qty"], pos["avg_price"], is_crypto)
                        self._position_strategy.pop(sym, None)
                    else:
                        close_qty = pos["qty"] * close_pct
                        close_qty = round(close_qty, 4) if is_crypto else math.floor(close_qty)
                        if close_qty > 0:
                            log.info("Rebalance: trim {} {} ({} OW {:.1%})", sym, close_qty, theme, overweight)
                            self._exec.close_partial(sym, close_qty, pos["avg_price"], is_crypto)

                    trimmed += close_value

            elif overweight < -0.05:
                log.info("Theme {} underweight {:.1%} — strategies will fill organically", theme, -overweight)

    # ── Signal execution ──────────────────────────────────────────────────────

    def _execute_signals(self, signals: List[Signal]) -> None:
        if not signals:
            return

        # Deduplicate — one signal per symbol, highest strength wins
        best: Dict[str, Signal] = {}
        for sig in signals:
            if sig.symbol not in best or sig.strength > best[sig.symbol].strength:
                best[sig.symbol] = sig

        state     = self._portfolio.get_account_state()
        positions = self._portfolio.get_open_positions()

        for sym, sig in sorted(best.items(), key=lambda x: x[1].strength, reverse=True):
            if self._breaker.is_halted() and sig.side == "buy":
                break

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

            if sig.side == "sell":
                # Sell signal (e.g. sector rotation exit) — only act if we hold it
                if sym in positions:
                    pos = positions[sym]
                    is_crypto = self._universe.is_crypto(sym)
                    self._exec.close_partial(sym, pos["qty"], sig.price, is_crypto)
                    self._position_strategy.pop(sym, None)
                continue

            # Buy signal — run through risk manager
            approved, reason, qty = self._risk.check_signal(
                sig,
                state["portfolio_value"],
                state["buying_power"],
                positions,
                self._portfolio.daily_pnl(),
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
        except Exception as e:
            log.warning("_seed_position_strategies error: {}", e)

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
