# Autonomous Alpaca Trading Bot — $100 Challenge

A fully deterministic, rule-based trading bot that trades equities and crypto on Alpaca Markets. No AI/LLM in any trading decision — every signal is explainable, auditable, and driven by classic technical indicators.

Deploys to Railway as an always-on worker process with SQLite state persistence.

## Design Principles

- **No blow-up first** — kill switch at $85, hard halt at 10% drawdown, soft halt at 5%
- **No AI in trading logic** — every decision is a deterministic rule with a documented reason
- **No hardcoded secrets** — all credentials via environment variables only
- **Idempotent orders** — MD5-hashed client_order_id prevents duplicate orders on restart
- **Hot-reloadable config** — change strategy params or watchlist without redeploying

## Strategies

| Strategy | Timeframe | When Active |
|----------|-----------|-------------|
| **Trend** | Daily bars | Bull regime (EMA20 > EMA50, ADX > 20) |
| **Mean Reversion** | Daily bars | Ranging regime (RSI < 35, price ≤ lower BB) |
| **Crypto Momentum** | Hourly bars | 24/7 (EMA12/26 cross, volume spike, RSI 40-70) |

A **regime filter** (SPY vs 50/200 MA + ADX) weights strategies appropriately for current market conditions.

## Risk Controls

- **ATR stop**: hard floor at entry − 2×ATR
- **Trailing stop**: arms at 8% gain, gives back 5% from peak
- **Take profit**: 2× initial risk
- **Time stop**: equity 10 days, crypto 5 days
- **Edge gate**: only enter if expected gain > round-trip costs (crypto ≥ 0.8%, equity ≥ 0.4%)
- **Cooldowns**: equity 24h, crypto 6h after exit
- **Circuit breakers**: soft halt (5% drawdown) → hard halt (10%) → killed ($85 floor)

## Quick Start

```bash
git clone <repo>
cd nonaibot
pip install -r requirements.txt

cp .env.example .env
# Edit .env with your Alpaca API keys and TRADING_MODE

python main.py
```

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `APCA_API_KEY_ID` | yes | — | Alpaca API key |
| `APCA_API_SECRET_KEY` | yes | — | Alpaca secret key |
| `APCA_API_BASE_URL` | yes | paper URL | Alpaca base URL |
| `TRADING_MODE` | no | `paper` | `paper` or `live` |
| `DB_PATH` | no | `bot.db` | SQLite file path (use `/data/bot.db` on Railway) |
| `PORT` | no | `8080` | HTTP health server port |
| `SLACK_WEBHOOK_URL` | no | — | Slack webhook for alerts/reports |
| `DISCORD_WEBHOOK_URL` | no | — | Discord webhook |
| `TELEGRAM_BOT_TOKEN` | no | — | Telegram bot token |
| `TELEGRAM_CHAT_ID` | no | — | Telegram chat ID |

## HTTP Endpoints

| Endpoint | Purpose |
|----------|---------|
| `GET /health` | Railway health check — returns `{"status": "ok"}` |
| `GET /status` | Full bot state as JSON (equity, positions, circuit status) |

## Backtest

```bash
python -m backtest.runner --start 2024-01-01 --end 2024-12-31 --symbols NLR,SMH,LLY,SOL/USD
```

Uses the same indicator code as the live bot. Not a production backtester (no slippage, no spread), but validates that strategy parameters produce reasonable signals on historical data.

## Project Structure

```
config/           YAML config files (hot-reloaded each tick)
  settings.yaml   Timing, schedule, data, logging
  watchlist.yaml  Core + dynamic symbol lists and screener candidates
  strategy.yaml   Indicator parameters and edge gate thresholds
  risk.yaml       Position sizing, circuit breakers, cooldowns

core/
  config.py       BotConfig — loads all YAML + env vars
  broker.py       BrokerClient — Alpaca SDK wrapper with rate limiter
  clock.py        MarketClock — market hours with 60s cache
  state.py        SQLiteState — WAL-mode SQLite persistence

data/
  fetcher.py      DataFetcher — bar data with TTL cache
  indicators.py   Pure functions: EMA, RSI, ATR, ADX, Bollinger

strategy/
  base.py         Signal dataclass + BaseStrategy ABC
  regime.py       RegimeFilter — bull/bear/ranging classification
  trend.py        TrendStrategy — EMA crossover + ADX
  mean_reversion.py  MeanReversionStrategy — RSI + Bollinger
  crypto_momentum.py CryptoMomentumStrategy — hourly EMA + volume

risk/
  sizer.py        PositionSizer — ATR-based Kelly sizing + edge gate
  circuit_breaker.py CircuitBreaker — halt levels and daily reset

execution/
  assets.py       AssetCache — fractionable flag lookup
  executor.py     Executor — order placement with idempotency

screener/
  screener.py     Screener — dynamic watchlist management

audit/
  report.py       AuditReport — daily pre-market status report
  delivery.py     Webhook delivery (Slack, Discord, Telegram)

backtest/
  runner.py       BacktestRunner — bar-by-bar event simulation

server.py         Lightweight HTTP health/status server
main.py           Run loop — orchestrates all components
```

## Railway Deployment

1. Push to GitHub
2. Connect repo to Railway
3. Add environment variables (Alpaca keys, `TRADING_MODE=paper` to start)
4. Add a Railway Volume mounted at `/data`, set `DB_PATH=/data/bot.db`
5. Railway uses `Procfile` to start: `python main.py`
6. Health checks hit `GET /health` on port 8080

Switch to live trading only after validating with paper mode:
```
TRADING_MODE=live
APCA_API_BASE_URL=https://api.alpaca.markets
```

## Constraints (Alpaca, June 2026)

- PDT rule retired June 4, 2026 — no daytrade restrictions
- Fractional equity orders: must be market + DAY + notional (not qty)
- Stops tracked in code — no resting stop orders for fractional positions
- Crypto fee: 0.25% taker / 0.15% maker
- Rate limit: 200 req/min (free tier) — enforced by token bucket

## Disclaimer

Trading involves significant financial risk. This bot is a technical experiment, not financial advice. Start in paper mode. Only switch to live trading with money you can afford to lose.
