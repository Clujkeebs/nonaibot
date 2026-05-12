"""
Central configuration — all settings loaded from environment variables.
Paper trading is the default; set ALPACA_PAPER=false for live.
"""
import os
from typing import Dict, List

from dotenv import load_dotenv

load_dotenv()


def _bool(key: str, default: str = "true") -> bool:
    return os.getenv(key, default).lower() == "true"


def _float(key: str, default: str) -> float:
    return float(os.getenv(key, default))


def _int(key: str, default: str) -> int:
    return int(os.getenv(key, default))


# ── Alpaca ───────────────────────────────────────────────────────────────────
ALPACA_API_KEY: str = os.environ["ALPACA_API_KEY"]
ALPACA_SECRET_KEY: str = os.environ["ALPACA_SECRET_KEY"]
ALPACA_PAPER: bool = _bool("ALPACA_PAPER", "true")

# ── Risk ─────────────────────────────────────────────────────────────────────
MAX_POSITION_PCT: float     = _float("MAX_POSITION_PCT",     "0.05")   # 5% per symbol (was 8 — reduced to prevent over-concentration)
MAX_SECTOR_PCT: float       = _float("MAX_SECTOR_PCT",       "0.25")   # 25% per sector (was 30)
MAX_CORRELATED_PCT: float   = _float("MAX_CORRELATED_PCT",   "0.20")   # 20% cap on correlated positions (was 25)
MAX_CRYPTO_PCT: float       = _float("MAX_CRYPTO_PCT",       "1.0")    # no hard crypto cap
RISK_PER_TRADE_PCT: float   = _float("RISK_PER_TRADE_PCT",   "0.01")   # 1.0% risk per trade (was 1.5 — more conservative)
HARD_STOP_PCT: float        = _float("HARD_STOP_PCT",        "0.025")  # 2.5% hard stop equity (was 3 — too late)
HARD_STOP_PCT_CRYPTO: float = _float("HARD_STOP_PCT_CRYPTO", "0.04")   # 4% hard stop crypto (was 6)
TRAIL_ARM_PCT: float        = _float("TRAIL_ARM_PCT",        "0.08")   # arm trail after +8% from entry
TRAIL_GIVEBACK_PCT: float   = _float("TRAIL_GIVEBACK_PCT",   "0.05")   # give back 5% from peak
COOLDOWN_HOURS_EQUITY: int  = _int("COOLDOWN_HOURS_EQUITY",  "24")     # no re-entry for 24h after stop
COOLDOWN_HOURS_CRYPTO: int  = _int("COOLDOWN_HOURS_CRYPTO",  "6")      # crypto recovers faster
ATR_STOP_MULT: float        = _float("ATR_STOP_MULT",        "1.5")    # 1.5× ATR stop
DAILY_LOSS_LIMIT_PCT: float = _float("DAILY_LOSS_LIMIT_PCT", "0.02")   # 2% daily halt (was 3 — was firing too late)
WEEKLY_LOSS_LIMIT_PCT: float = _float("WEEKLY_LOSS_LIMIT_PCT","0.05")  # 5% weekly halt (was 7)
MAX_OPEN_POSITIONS: int     = _int("MAX_OPEN_POSITIONS",     "10")    # was 40 — quality over quantity
MIN_ATR_PCT: float          = _float("MIN_ATR_PCT",          "0.005")  # skip illiquid symbols
PORTFOLIO_HEAT_MAX: float   = max(0.35, _float("PORTFOLIO_HEAT_MAX",   "0.55"))  # max total risk-on (floor 0.35, was 0.70)
TIME_STOP_DAYS: int         = _int("TIME_STOP_DAYS",         "7")      # was 15 — cut flat/losing positions in 1 week
CONFLUENCE_BONUS: float     = _float("CONFLUENCE_BONUS",     "0.25")   # extra size when 2+ strats agree
VIX_RISK_SCALE: bool        = _bool("VIX_RISK_SCALE",        "true")   # scale positions by VIX level

# ── Strategy toggles ─────────────────────────────────────────────────────────
ENABLE_TREND_FOLLOWING:     bool = _bool("ENABLE_TREND_FOLLOWING",     "true")
ENABLE_MEAN_REVERSION:      bool = _bool("ENABLE_MEAN_REVERSION",      "true")
ENABLE_VOLATILITY_BREAKOUT: bool = _bool("ENABLE_VOLATILITY_BREAKOUT", "true")
ENABLE_SECTOR_ROTATION:     bool = _bool("ENABLE_SECTOR_ROTATION",     "true")
ENABLE_CRYPTO_MOMENTUM:     bool = _bool("ENABLE_CRYPTO_MOMENTUM",     "true")
ENABLE_OPENING_RANGE:       bool = _bool("ENABLE_OPENING_RANGE",       "true")
ENABLE_AI_LAYER:            bool = _bool("ENABLE_AI_LAYER",            "true")
ENABLE_AI_POSITION_SIZER:   bool = _bool("ENABLE_AI_POSITION_SIZER",   "true")  # adaptive size per strategy+regime
ENABLE_AI_STRATEGY_ALLOC:   bool = _bool("ENABLE_AI_STRATEGY_ALLOC",   "true")  # dynamic strategy weights
ENABLE_AI_EXIT_OPT:         bool = _bool("ENABLE_AI_EXIT_OPT",         "true")  # ML-driven stop/tp optimization
AI_REFRESH_MINUTES:         int   = _int("AI_REFRESH_MINUTES",          "30")    # how often AI models refresh

# ── Trading hours ─────────────────────────────────────────────────────────────
MARKET_OPEN_HOUR:  int = _int("MARKET_OPEN_HOUR",  "9")
MARKET_OPEN_MIN:   int = _int("MARKET_OPEN_MIN",   "35")   # 09:35 ET — 5 min warmup after open
MARKET_CLOSE_HOUR: int = _int("MARKET_CLOSE_HOUR", "15")
MARKET_CLOSE_MIN:  int = _int("MARKET_CLOSE_MIN",  "55")   # 15:55 ET — 5 min cooldown before close
EQUITY_STRATEGY_INTERVAL_MIN: int = _int("EQUITY_STRATEGY_INTERVAL_MIN", "3")   # was 5 — faster adaptation
CRYPTO_STRATEGY_INTERVAL_MIN: int = _int("CRYPTO_STRATEGY_INTERVAL_MIN", "5")   # 24/7 crypto scan interval
EXIT_CHECK_INTERVAL_MIN: int     = _int("EXIT_CHECK_INTERVAL_MIN",     "2")   # exit check frequency

# ── Thematic universe (23 equity symbols, focused & high-quality) ───────────
AI_TECH: List[str] = [
    "NVDA", "MSFT", "AAPL", "META", "AMZN", "GOOGL",
    "TSLA", "AVGO", "AMD",  "PLTR", "CRWD",
    "MU",   "ALGN",                          # sleeper picks: AI memory + medtech dip
]

GROWTH_ETFS: List[str] = [
    "QQQ", "SPY", "SMH", "NLR", "LLY",
]

CRYPTO_EQUITIES: List[str] = [
    "COIN", "MSTR", "MARA",
]

CORE_MACRO: List[str] = ["GLD", "TLT"]

CRYPTO_SYMBOLS: List[str] = [
    "BTC/USD", "ETH/USD", "SOL/USD",
    "AVAX/USD", "DOT/USD", "LINK/USD",
    "DOGE/USD", "AAVE/USD", "UNI/USD",
    "LTC/USD", "XRP/USD", "ADA/USD",
    "SHIB/USD", "BCH/USD",
]

# All equity symbols (deduped) — 23 high-conviction names
ALL_EQUITY_SYMBOLS: List[str] = sorted(set(
    AI_TECH + GROWTH_ETFS + CRYPTO_EQUITIES + CORE_MACRO
))

# Sector tags (used for exposure tracking)
SECTOR_MAP: Dict[str, str] = {
    **{s: "ai_tech"        for s in AI_TECH},
    **{s: "growth_etf"     for s in GROWTH_ETFS},
    **{s: "crypto_equity"  for s in CRYPTO_EQUITIES},
    **{s: "core_macro"     for s in CORE_MACRO},
    **{s.replace("/", ""): "crypto" for s in CRYPTO_SYMBOLS},
}

# ── Execution ────────────────────────────────────────────────────────────────
ORDER_RETRY_LIMIT: int       = _int("ORDER_RETRY_LIMIT",   "3")
ORDER_FILL_TIMEOUT: int      = _int("ORDER_FILL_TIMEOUT",  "60")   # seconds
SLIPPAGE_LIMIT_PCT: float    = _float("SLIPPAGE_LIMIT_PCT","0.003") # 0.3%
MIN_NOTIONAL: float          = _float("MIN_NOTIONAL",      "100")   # $100 min equity order
MIN_NOTIONAL_CRYPTO: float   = _float("MIN_NOTIONAL_CRYPTO","25")   # $25 min crypto order

# ── Data ─────────────────────────────────────────────────────────────────────
BARS_LOOKBACK_DAYS: int      = _int("BARS_LOOKBACK_DAYS", "350")  # needs 210+ trading days for SMA200
CRYPTO_BARS_LOOKBACK: int    = _int("CRYPTO_BARS_LOOKBACK","30")  # 15-min bars, 30 days = ~384 bars

# ── Infrastructure ───────────────────────────────────────────────────────────
DB_PATH: str          = os.getenv("DB_PATH",    "trading_bot.db")
LOG_LEVEL: str        = os.getenv("LOG_LEVEL",  "INFO")
LOG_FILE: str         = os.getenv("LOG_FILE",   "logs/bot.log")

# ── Alerts ───────────────────────────────────────────────────────────────────
SLACK_WEBHOOK_URL: str = os.getenv("SLACK_WEBHOOK_URL", "")
ALERT_EMAIL: str       = os.getenv("ALERT_EMAIL", "")
ALERT_ON_TRADE: bool   = _bool("ALERT_ON_TRADE", "true")
ALERT_ON_ERROR: bool   = _bool("ALERT_ON_ERROR", "true")
ALERT_ON_CIRCUIT: bool = _bool("ALERT_ON_CIRCUIT", "true")

# ── Timezone ─────────────────────────────────────────────────────────────────
TIMEZONE: str = "America/New_York"
