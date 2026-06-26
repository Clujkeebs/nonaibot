"""
BacktestRunner — simple event-driven backtest for strategy sanity-checking.

Usage:
  python -m backtest.runner --start 2024-01-01 --end 2024-12-31 --symbols NLR,SMH,LLY

NOT a production backtester — no slippage modelling, no partial fills, no bid/ask spread.
Purpose: sanity-check that strategy parameters produce reasonable signals on history.

Metrics reported:
  Total return, # trades, win rate, average win/loss, max drawdown, Sharpe ratio

The backtest uses the SAME indicator code as the live bot — if it works in backtest,
the indicators are at least producing the right signal direction.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import pytz

# Add project root to path when run as a module
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.config import BotConfig
from data.indicators import atr as calc_atr, ema as calc_ema, rsi as calc_rsi
from data.indicators import adx as calc_adx, bollinger
from strategy.trend import TrendStrategy
from strategy.mean_reversion import MeanReversionStrategy
from strategy.crypto_momentum import CryptoMomentumStrategy

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    symbol: str
    strategy: str
    entry_date: str
    entry_price: float
    exit_date: Optional[str] = None
    exit_price: Optional[float] = None
    qty: float = 0.0
    pnl: float = 0.0
    pnl_pct: float = 0.0
    exit_reason: str = ""


@dataclass
class BacktestResult:
    start_date: str
    end_date: str
    starting_capital: float
    ending_capital: float = 0.0
    trades: List[Trade] = field(default_factory=list)

    @property
    def total_return_pct(self) -> float:
        if self.starting_capital <= 0:
            return 0.0
        return (self.ending_capital - self.starting_capital) / self.starting_capital * 100

    @property
    def num_trades(self) -> int:
        return len([t for t in self.trades if t.exit_date is not None])

    @property
    def win_rate(self) -> float:
        closed = [t for t in self.trades if t.exit_date is not None]
        if not closed:
            return 0.0
        wins = sum(1 for t in closed if t.pnl > 0)
        return wins / len(closed) * 100

    @property
    def avg_win(self) -> float:
        wins = [t.pnl_pct for t in self.trades if t.exit_date and t.pnl > 0]
        return sum(wins) / len(wins) * 100 if wins else 0.0

    @property
    def avg_loss(self) -> float:
        losses = [t.pnl_pct for t in self.trades if t.exit_date and t.pnl < 0]
        return sum(losses) / len(losses) * 100 if losses else 0.0

    @property
    def max_drawdown_pct(self) -> float:
        """Calculate max drawdown from equity curve implied by trades."""
        if not self.trades:
            return 0.0
        eq = self.starting_capital
        hwm = eq
        max_dd = 0.0
        for t in sorted(self.trades, key=lambda x: x.exit_date or ""):
            if t.exit_date:
                eq += t.pnl
                if eq > hwm:
                    hwm = eq
                dd = (hwm - eq) / hwm if hwm > 0 else 0
                max_dd = max(max_dd, dd)
        return max_dd * 100

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "  BACKTEST RESULTS",
            f"  Period: {self.start_date} → {self.end_date}",
            f"  Capital: ${self.starting_capital:.2f} → ${self.ending_capital:.2f}",
            "=" * 60,
            f"  Total Return:  {self.total_return_pct:>+8.2f}%",
            f"  # Trades:      {self.num_trades:>8}",
            f"  Win Rate:      {self.win_rate:>8.1f}%",
            f"  Avg Win:       {self.avg_win:>+8.2f}%",
            f"  Avg Loss:      {self.avg_loss:>+8.2f}%",
            f"  Max Drawdown:  {self.max_drawdown_pct:>8.2f}%",
            "=" * 60,
        ]
        if self.trades:
            lines.append("\n  TRADE LOG:")
            for t in self.trades[:50]:  # show first 50 trades
                status = "OPEN" if not t.exit_date else f"EXIT({t.exit_reason})"
                pnl_str = f"${t.pnl:>+8.2f} ({t.pnl_pct*100:+.1f}%)" if t.exit_date else "open"
                lines.append(
                    f"  {t.entry_date[:10]} BUY  {t.symbol:<10} @ ${t.entry_price:.4f} "
                    f"→ {status} {pnl_str}"
                )
        return "\n".join(lines)


class BacktestRunner:
    """
    Simple bar-by-bar backtest engine.

    For each bar:
      1. Run strategy.generate_signals() on data up to current bar
      2. If signal=BUY and no position: open position at next bar's open
      3. If position open: check strategy exit + ATR stop
      4. Track equity and trades
    """

    def __init__(self, config: BotConfig) -> None:
        self._cfg = config
        self._strategies = {
            "trend": TrendStrategy(),
            "mean_reversion": MeanReversionStrategy(),
            "crypto_momentum": CryptoMomentumStrategy(),
        }

    def run(
        self,
        all_bars: Dict[str, pd.DataFrame],
        starting_capital: float = 100.0,
        risk_per_trade: float = 0.01,
    ) -> BacktestResult:
        """
        Run backtest across all symbols.
        all_bars: {symbol: full history DataFrame}
        """
        if not all_bars:
            return BacktestResult("", "", starting_capital, starting_capital)

        # Determine date range from the data
        all_dates = sorted(set(
            d.date().isoformat()
            for df in all_bars.values()
            for d in df.index.to_pydatetime()
        ))
        if not all_dates:
            return BacktestResult("", "", starting_capital, starting_capital)

        result = BacktestResult(
            start_date=all_dates[0],
            end_date=all_dates[-1],
            starting_capital=starting_capital,
        )

        equity = starting_capital
        positions: Dict[str, Trade] = {}  # symbol → open trade
        position_highs: Dict[str, float] = {}  # symbol → high-water mark for trailing stop

        # Walk forward bar by bar (using index position)
        # Find minimum bars needed (slow EMA period + 20)
        min_bars = max(self._cfg.trend_slow_ema, self._cfg.mr_bb_period) + 20

        # Get common bars across all symbols
        first_symbol = next(iter(all_bars))
        all_bar_dates = all_bars[first_symbol].index.tolist()

        for i in range(min_bars, len(all_bar_dates)):
            current_date = all_bar_dates[i]

            # Slice data up to current bar (no lookahead)
            current_bars = {sym: df.iloc[: i + 1] for sym, df in all_bars.items()}

            # ── Exit check for open positions ──────────────────────────────
            for sym in list(positions.keys()):
                trade = positions[sym]
                df = current_bars.get(sym)
                if df is None or len(df) < 2:
                    continue

                cur_price = float(df["close"].iloc[-1])
                is_crypto = "/" in sym

                # Update high-water mark
                prev_high = position_highs.get(sym, trade.entry_price)
                position_highs[sym] = max(prev_high, cur_price)
                peak = position_highs[sym]

                entry_price = trade.entry_price
                pnl_pct = (cur_price - entry_price) / entry_price if entry_price > 0 else 0

                # ATR stop
                atr_s = calc_atr(df["high"], df["low"], df["close"], self._cfg.atr_period)
                atr_val = float(atr_s.iloc[-1]) if len(atr_s) > 0 else 0
                stop_price = entry_price - self._cfg.atr_stop_mult * atr_val
                exit_reason = ""

                if atr_val > 0 and cur_price < stop_price:
                    exit_reason = "atr_stop"
                elif pnl_pct >= self._cfg.trailing_arm_pct:
                    trail_stop = peak * (1 - self._cfg.trailing_giveback_pct)
                    if cur_price < trail_stop:
                        exit_reason = "trailing_stop"
                elif pnl_pct >= self._cfg.take_profit_ratio * (self._cfg.atr_stop_mult * atr_val / entry_price if entry_price > 0 else 0.04):
                    exit_reason = "take_profit"
                else:
                    # Strategy exit
                    strat_name = trade.strategy
                    strat = self._strategies.get(strat_name)
                    if strat and strat.check_exit(sym, entry_price, df, self._cfg):
                        exit_reason = "strategy"

                if exit_reason:
                    trade.exit_date = str(current_date.date())
                    trade.exit_price = cur_price
                    trade.pnl = (cur_price - entry_price) * trade.qty
                    trade.pnl_pct = pnl_pct
                    trade.exit_reason = exit_reason
                    equity += trade.pnl
                    result.trades.append(trade)
                    del positions[sym]
                    position_highs.pop(sym, None)

            # ── Entry signals (only if not already in position) ─────────────
            if len(positions) < self._cfg.max_concurrent_positions:
                for strat_name, strat in self._strategies.items():
                    try:
                        signals = strat.generate_signals(current_bars, self._cfg)
                        for sig in signals:
                            sym = sig.symbol
                            if sym in positions or sym not in current_bars:
                                continue
                            if sig.stop_distance <= 0:
                                continue
                            # Size: 1% risk / stop distance
                            risk_dollars = equity * risk_per_trade
                            qty = risk_dollars / sig.stop_distance
                            if "/" not in sym:
                                import math
                                qty = math.floor(qty)
                            else:
                                qty = round(qty, 6)
                            if qty <= 0:
                                continue
                            notional = qty * sig.price
                            if notional > equity * 0.10:
                                qty = (equity * 0.10) / sig.price
                            # Open position
                            positions[sym] = Trade(
                                symbol=sym,
                                strategy=strat_name,
                                entry_date=str(current_date.date()),
                                entry_price=sig.price,
                                qty=qty,
                            )
                            position_highs[sym] = sig.price
                            break  # one signal per bar per strategy
                    except Exception as e:
                        logger.debug("Backtest signal error %s %s: %s", strat_name, current_date, e)

        # Close any open positions at last price
        for sym, trade in positions.items():
            df = all_bars.get(sym)
            if df is not None and len(df) > 0:
                cur_price = float(df["close"].iloc[-1])
                trade.exit_date = all_dates[-1]
                trade.exit_price = cur_price
                trade.pnl = (cur_price - trade.entry_price) * trade.qty
                trade.pnl_pct = (cur_price - trade.entry_price) / trade.entry_price if trade.entry_price > 0 else 0
                trade.exit_reason = "end_of_period"
                equity += trade.pnl
                result.trades.append(trade)

        result.ending_capital = equity
        return result


def _load_bars_from_alpaca(
    symbols: List[str], start: str, end: str, config: BotConfig
) -> Dict[str, pd.DataFrame]:
    """Fetch historical bars for backtest from Alpaca. Requires valid credentials."""
    from core.broker import BrokerClient
    from data.fetcher import DataFetcher
    from alpaca.data.timeframe import TimeFrame

    broker = BrokerClient(config)
    fetcher = DataFetcher(broker, config)

    start_dt = datetime.strptime(start, "%Y-%m-%d")
    end_dt = datetime.strptime(end, "%Y-%m-%d")
    days = (end_dt - start_dt).days + 30  # extra buffer for indicators

    crypto = [s for s in symbols if "/" in s]
    equities = [s for s in symbols if "/" not in s]

    result = {}
    if equities:
        bars = fetcher.get_stock_bars(equities, TimeFrame.Day, lookback_days=days)
        result.update(bars)
    if crypto:
        bars = fetcher.get_crypto_bars(crypto, TimeFrame.Hour, lookback_days=days)
        result.update(bars)

    # Filter to date range
    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts = pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)
    filtered = {}
    for sym, df in result.items():
        try:
            idx = df.index
            if idx.tz is None:
                idx = idx.tz_localize("UTC")
            mask = (idx >= start_ts) & (idx <= end_ts)
            if mask.sum() > 10:
                filtered[sym] = df[mask]
        except Exception:
            filtered[sym] = df

    return filtered


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(description="Trading bot backtest")
    parser.add_argument("--start", default="2024-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default="2024-12-31", help="End date (YYYY-MM-DD)")
    parser.add_argument(
        "--symbols",
        default="NLR,SMH,LLY,SOL/USD,BTC/USD",
        help="Comma-separated symbols",
    )
    parser.add_argument("--capital", type=float, default=100.0, help="Starting capital")
    parser.add_argument("--risk", type=float, default=0.01, help="Risk per trade (0.01=1%%)")
    parser.add_argument("--output", default=None, help="Write results to JSON file")
    args = parser.parse_args()

    config = BotConfig()
    errors = config.validate()
    if errors:
        print("Config errors:", errors)
        sys.exit(1)

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    print(f"Loading bars for {symbols} from {args.start} to {args.end}...")

    all_bars = _load_bars_from_alpaca(symbols, args.start, args.end, config)
    if not all_bars:
        print("No bars loaded. Check credentials and symbol names.")
        sys.exit(1)

    print(f"Loaded {len(all_bars)} symbols. Running backtest...")
    runner = BacktestRunner(config)
    result = runner.run(all_bars, starting_capital=args.capital, risk_per_trade=args.risk)

    print(result.summary())

    if args.output:
        with open(args.output, "w") as f:
            json.dump(
                {
                    "start": result.start_date,
                    "end": result.end_date,
                    "starting_capital": result.starting_capital,
                    "ending_capital": result.ending_capital,
                    "total_return_pct": result.total_return_pct,
                    "num_trades": result.num_trades,
                    "win_rate": result.win_rate,
                    "max_drawdown_pct": result.max_drawdown_pct,
                },
                f, indent=2,
            )
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
