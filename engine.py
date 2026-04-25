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

Flow per crypto cycle (every 15 min, 24/7):
  Same but using hourly bars and crypto symbols only.

Exit checks (every 2 min):
  For every open position, ask its originating strategy check_exit().
  If exit → OrderEngine.close_position().

Circuit breaker check (every 1 min):
  Pull daily/weekly P&L from PortfolioTracker.
  Feed to CircuitBreaker.check_daily_loss() / check_weekly_loss().
  If full halt → close_all_positions().
"""
from __future__ import annotations

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

        # Optional AI layer
        self._ai_ranker = None
        if config.ENABLE_AI_LAYER:
            self._init_ai()

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
            timeframe=TimeFrame.Hour,
            lookback_days=config.CRYPTO_BARS_LOOKBACK,
        )
        if not bars:
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
                    bars_dict = self._data.get_crypto_bars([sym], TimeFrame.Hour, lookback_days=10)
                else:
                    bars_dict = self._data.get_stock_bars([sym], TimeFrame.Day, lookback_days=10)

                df = bars_dict.get(sym)
                if df is None:
                    continue

                if strat.check_exit(sym, pos["avg_price"], df):
                    log.info("Exit signal for {} from {}", sym, strat_name)
                    self._exec.close_position(sym)
                    self._position_strategy.pop(sym, None)
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
            bars = self._data.get_stock_bars(["SPY"], TimeFrame.Day, lookback_days=250)
            spy  = bars.get("SPY")
            self._regime.update(spy)

            if config.ENABLE_AI_LAYER and self._ai_ranker:
                state = self._ai_ranker.predict(spy) if spy is not None else None
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
                    self._exec.close_position(sym)
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
                log.debug("Signal REJECTED ({}) — {}: {}", sym, sig.strategy, reason)
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
            log.warning("AI init error: {}", e)
