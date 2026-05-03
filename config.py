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
MAX_POSITION_PCT: float     = _float("MAX_POSITION_PCT",     "0.10")   # 10% per symbol
MAX_SECTOR_PCT: float       = _float("MAX_SECTOR_PCT",       "0.25")   # 25% per sector
MAX_CRYPTO_PCT: float       = _float("MAX_CRYPTO_PCT",       "1.0")    # no hard crypto cap
RISK_PER_TRADE_PCT: float   = _float("RISK_PER_TRADE_PCT",   "0.02")   # 2% risk per trade
ATR_STOP_MULT: float        = _float("ATR_STOP_MULT",        "1.5")    # 1.5× ATR stop
DAILY_LOSS_LIMIT_PCT: float = _float("DAILY_LOSS_LIMIT_PCT", "0.03")   # 3% daily halt
WEEKLY_LOSS_LIMIT_PCT: float = _float("WEEKLY_LOSS_LIMIT_PCT","0.07")  # 7% weekly halt
MAX_OPEN_POSITIONS: int     = _int("MAX_OPEN_POSITIONS",     "40")
MIN_ATR_PCT: float          = _float("MIN_ATR_PCT",          "0.005")  # skip illiquid symbols
PORTFOLIO_HEAT_MAX: float   = max(0.50, _float("PORTFOLIO_HEAT_MAX",   "0.80"))  # max total risk-on (floor 0.50)

# ── Strategy toggles ─────────────────────────────────────────────────────────
ENABLE_TREND_FOLLOWING:     bool = _bool("ENABLE_TREND_FOLLOWING",     "true")
ENABLE_MEAN_REVERSION:      bool = _bool("ENABLE_MEAN_REVERSION",      "true")
ENABLE_VOLATILITY_BREAKOUT: bool = _bool("ENABLE_VOLATILITY_BREAKOUT", "true")
ENABLE_SECTOR_ROTATION:     bool = _bool("ENABLE_SECTOR_ROTATION",     "true")
ENABLE_CRYPTO_MOMENTUM:     bool = _bool("ENABLE_CRYPTO_MOMENTUM",     "true")
ENABLE_AI_LAYER:            bool = _bool("ENABLE_AI_LAYER",            "false")

# ── Thematic universe ─────────────────────────────────────────────────────────
AI_TECH: List[str] = [
    "NVDA", "AMD", "MSFT", "GOOGL", "META", "AMZN", "AAPL",
    "TSLA", "PLTR", "AI",  "IONQ", "RGTI", "SOUN", "BBAI",
    "SMCI", "AVGO", "QCOM", "ARM",  "TSM",  "CRWD", "SNOW",
]

CLEAN_ENERGY: List[str] = [
    "ENPH", "FSLR", "SEDG", "RUN", "NEE", "BE",
    "PLUG", "BLDP", "FCEL", "NOVA", "ARRY",
]

GROWTH_ETFS: List[str] = [
    "QQQ", "ARKK", "ARKG", "ARKW", "BOTZ",
    "HERO", "ICLN", "QCLN", "CIBR",
]

CRYPTO_EQUITIES: List[str] = [
    "MSTR", "COIN", "MARA", "RIOT", "CLSK", "HUT",
]

CORE_MACRO: List[str] = ["SPY", "IWM", "GLD", "TLT", "DIA"]

CRYPTO_SYMBOLS: List[str] = [
    "BTC/USD", "ETH/USD", "SOL/USD",
    "AVAX/USD", "DOT/USD", "LINK/USD",
    "DOGE/USD", "AAVE/USD", "UNI/USD",
    "LTC/USD", "XRP/USD", "ADA/USD",
    "SHIB/USD", "BCH/USD",
]

# All equity symbols (deduped)
ALL_EQUITY_SYMBOLS: List[str] = sorted(set(
    AI_TECH + CLEAN_ENERGY + GROWTH_ETFS + CRYPTO_EQUITIES + CORE_MACRO
))

# Sector tags (used for exposure tracking)
SECTOR_MAP: Dict[str, str] = {
    **{s: "ai_tech"        for s in AI_TECH},
    **{s: "clean_energy"   for s in CLEAN_ENERGY},
    **{s: "growth_etf"     for s in GROWTH_ETFS},
    **{s: "crypto_equity"  for s in CRYPTO_EQUITIES},
    **{s: "core_macro"     for s in CORE_MACRO},
    **{s.replace("/", ""): "crypto" for s in CRYPTO_SYMBOLS},
}

# ── Execution ────────────────────────────────────────────────────────────────
ORDER_RETRY_LIMIT: int       = _int("ORDER_RETRY_LIMIT",   "3")
ORDER_FILL_TIMEOUT: int      = _int("ORDER_FILL_TIMEOUT",  "60")   # seconds
SLIPPAGE_LIMIT_PCT: float    = _float("SLIPPAGE_LIMIT_PCT","0.003") # 0.3%
MIN_NOTIONAL: float          = _float("MIN_NOTIONAL",      "100")   # $100 min order

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
