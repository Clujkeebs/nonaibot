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

## Account-Size Awareness (Capital Tiers)

The bot scales how it sizes and diversifies to how much money is in the account —
a $100 account should not trade like a $100k one. Each loop tick it re-reads live
equity and selects a **capital tier** that overrides the sizing parameters:

| Tier | Equity | Max/position | Concurrent | Risk/trade | Edge gate (eq/crypto) | Watchlist slots |
|------|--------|--------------|------------|------------|------------------------|-----------------|
| micro | ≤ $1k | 35% | 3 | 2.0% | 0.60% / 1.20% | 2 + 1 |
| small | ≤ $10k | 20% | 6 | 1.5% | 0.40% / 0.90% | 4 + 2 |
| mid | ≤ $100k | 12% | 10 | 1.0% | 0.30% / 0.70% | 6 + 3 |
| large | > $100k | 8% | 15 | 0.75% | 0.25% / 0.60% | 10 + 4 |

The logic: a small account **concentrates** into a few meaningful positions and
demands a wider edge — tiny $5 positions get eaten alive by per-trade costs. A
large account **spreads** across many names with a small cap on each and can accept
a thinner edge, since costs are a smaller percentage at size. The bot retiers itself
automatically as it grows or draws down, logs every tier change, and reports the
active tier in `/status` and the daily audit.

Tiers are fully config-driven in `config/risk.yaml` (`capital_tiers:`) and hot-reload
like everything else. An explicit `RISK_PER_TRADE_PCT` env var pins risk-per-trade
and is never overridden by a tier; a hard 5% ceiling caps it regardless.

## Manual Trades Are Auto-Adopted

If you place a trade by hand in Alpaca — say you buy $10 of a new stock — the bot
detects the untracked position on its next tick and **adopts** it:

- adds the symbol to the dynamic watchlist so it's actively watched
- records an entry time and high-water mark so the bot manages its exits
  (ATR stop, trailing stop, take profit, time stop) just like its own trades
- protects it from the screener — a held symbol is never rotated out of the
  watchlist until the position is closed

Adoption is idempotent (it only happens once per position) and runs both at
startup and on every loop tick, so positions opened while the bot was offline
are picked up immediately when it restarts.

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
| `TRADING_MODE` | no | `live` | `live` (real money) or `paper` (dry-run) |
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
3. Add environment variables: **live** Alpaca keys + `TRADING_MODE=live`
4. Add a Railway Volume mounted at `/data`, set `DB_PATH=/data/bot.db`
5. Railway uses `Procfile` to start: `python main.py`
6. Health checks hit `GET /health` on port 8080

This runs a **funded live account** (real money). The kill switch ($85 floor) and
drawdown halts (5% soft / 10% hard) are the safety rails. To dry-run instead, set
`TRADING_MODE=paper` and use paper keys.

## Constraints (Alpaca, June 2026)

- PDT rule retired June 4, 2026 — no daytrade restrictions
- Fractional equity orders: must be market + DAY + notional (not qty)
- Stops tracked in code — no resting stop orders for fractional positions
- Crypto fee: 0.25% taker / 0.15% maker
- Rate limit: 200 req/min (free tier) — enforced by token bucket

## Disclaimer

Trading involves significant financial risk. This bot is a technical experiment, not financial advice. Start in paper mode. Only switch to live trading with money you can afford to lose.
