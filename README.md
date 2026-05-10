# NoAiBot — 24/7 Algorithmic Trading Bot

A high-performance, AI-enhanced automated trading engine for Alpaca Markets. Trades equities and crypto around the clock with intelligent risk management, free local AI improvements, and multi-strategy execution.

## Features

### Multi-Strategy Trading
- **Trend Following** — Dual EMA crossover + ADX trend filter
- **Mean Reversion** — Bollinger Band + RSI oversold filter
- **Volatility Breakout** — Keltner Channel expansion strategy
- **Sector Rotation** — Weekly thematic momentum scoring
- **Crypto Momentum** — 24/7 hourly RSI/SMA trend strategy
- **Opening Range Breakout** — Intraday 30-min range breakout (equities)

### Free Local AI Layer (No External APIs)
- **Asset Ranker** — sklearn GBM predicts next-bar returns from price features
- **Regime Detector** — HMM identifies bull/bear/transition market states
- **Position Sizer** — Adaptive sizing based on strategy+regime performance
- **Strategy Allocator** — Dynamic weight allocation toward hot strategies
- **Exit Optimizer** — Learns optimal stop/take-profit from trade history
- **Technical Scorecard** — Composite 0-100 score from RSI/MACD/BB/volume/trend
- **Sentiment Analyzer** — Keyword + TF-IDF classifier for news headlines

### Risk Management
- ATR-based adaptive stops (not fixed %)
- Trailing stops with volatility-adjusted giveback
- Correlation caps (AI tech + crypto equities grouped)
- Sector exposure limits
- Circuit breakers (daily/weekly loss limits)
- Cooldown after stop-outs to prevent re-buying falling knives

### Execution
- Paper trading by default (ALPACA_PAPER=true)
- Market orders for equities, limit orders for crypto buys
- Auto-retry with fill verification
- Fractional shares support

## Quick Start

```bash
# 1. Clone
git clone https://github.com/noaibot/noaibot.git
cd noaibot

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure
cp .env.example .env
# Edit .env with your Alpaca API keys

# 4. Run
python main.py
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ALPACA_API_KEY` | required | Alpaca API key |
| `ALPACA_SECRET_KEY` | required | Alpaca secret key |
| `ALPACA_PAPER` | `true` | Paper trading mode |
| `ENABLE_AI_LAYER` | `true` | Enable local AI components |
| `ENABLE_AI_POSITION_SIZER` | `true` | Adaptive position sizing |
| `ENABLE_AI_STRATEGY_ALLOC` | `true` | Dynamic strategy weights |
| `ENABLE_AI_EXIT_OPT` | `true` | ML-driven exit optimization |
| `MAX_POSITION_PCT` | `0.08` | Max 8% per symbol |
| `RISK_PER_TRADE_PCT` | `0.015` | 1.5% risk per trade |
| `DAILY_LOSS_LIMIT_PCT` | `0.02` | 2% daily halt |
| `SLACK_WEBHOOK_URL` | `` | Slack alerts (optional) |

## Architecture

```
main.py              → Entry point, graceful shutdown
engine.py            → Trading engine, signal execution, exit logic
config.py            → All environment configuration
strategies/          → 6 strategy implementations
risk/                → RiskManager, CircuitBreaker
execution/           → OrderEngine with retry logic
portfolio/           → PortfolioTracker, P&L calculation
data/                → MarketDataClient, Universe
ai/                  → 7 free AI modules (all local, no external APIs)
scheduler/           → TaskManager (APScheduler jobs)
utils/               → Logging, alerts
```

## Disclaimer

Trading stocks and digital assets involves significant risk. This bot is for educational and informational purposes only. Use at your own risk. Past performance does not guarantee future results.