"""
BotConfig — loads config/*.yaml and overlays environment variables.

Environment variable precedence (highest to lowest):
  1. Env vars (APCA_API_KEY_ID, TRADING_MODE, DB_PATH, LOG_LEVEL, ...)
  2. config/*.yaml files
  3. Built-in defaults

Call config.reload() at the top of each main loop iteration to pick up
hot-reloaded watchlist/strategy/risk changes without redeploy.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from dotenv import load_dotenv

load_dotenv()

_CONFIG_DIR = Path(__file__).parent.parent / "config"


def _read_yaml(name: str) -> Dict[str, Any]:
    path = _CONFIG_DIR / name
    if not path.exists():
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _env(key: str, default: Any = None) -> Optional[str]:
    return os.environ.get(key, default)


def _bool_env(key: str, default: bool) -> bool:
    v = os.environ.get(key)
    if v is None:
        return default
    return v.lower() in ("1", "true", "yes", "on")


def _float_env(key: str, default: float) -> float:
    v = os.environ.get(key)
    return float(v) if v is not None else default


def _int_env(key: str, default: int) -> int:
    v = os.environ.get(key)
    return int(v) if v is not None else default


class BotConfig:
    """
    Single source of truth for all configuration.
    Thread-safe for reads (attributes are plain Python values after load).
    """

    def __init__(self) -> None:
        self._load()

    def _load(self) -> None:
        settings = _read_yaml("settings.yaml")
        watchlist = _read_yaml("watchlist.yaml")
        strategy = _read_yaml("strategy.yaml")
        risk = _read_yaml("risk.yaml")

        # ── Alpaca credentials (REQUIRED env vars) ──────────────────────────
        self.api_key: str = (
            _env("APCA_API_KEY_ID") or _env("ALPACA_API_KEY") or ""
        )
        self.secret_key: str = (
            _env("APCA_API_SECRET_KEY") or _env("ALPACA_SECRET_KEY") or ""
        )

        # ── Trading mode ─────────────────────────────────────────────────────
        # TRADING_MODE=live (default) or paper. This bot runs a funded LIVE
        # account, so live is the default; set TRADING_MODE=paper to dry-run.
        raw_mode = _env("TRADING_MODE", "live").lower()
        self.paper: bool = raw_mode == "paper"

        # Override base URLs for paper vs live
        if self.paper:
            self.base_url: str = _env("APCA_API_BASE_URL", "https://paper-api.alpaca.markets")
        else:
            self.base_url = _env("APCA_API_BASE_URL", "https://api.alpaca.markets")

        # ── DB ───────────────────────────────────────────────────────────────
        db_cfg = settings.get("db", {})
        self.db_path: str = _env("DB_PATH", db_cfg.get("path", "bot.db"))

        # ── Logging ──────────────────────────────────────────────────────────
        log_cfg = settings.get("logging", {})
        self.log_level: str = _env("LOG_LEVEL", log_cfg.get("level", "INFO"))
        self.log_file: str = _env("LOG_FILE", log_cfg.get("file", "logs/bot.log"))

        # ── Server ───────────────────────────────────────────────────────────
        srv_cfg = settings.get("server", {})
        self.server_port: int = _int_env("PORT", srv_cfg.get("port", 8080))
        self.server_host: str = srv_cfg.get("host", "0.0.0.0")

        # ── Loop timing ──────────────────────────────────────────────────────
        loop = settings.get("loop", {})
        self.equity_interval_min: int = loop.get("equity_interval_min", 5)
        self.crypto_interval_min: int = loop.get("crypto_interval_min", 5)
        self.exit_check_interval_min: int = loop.get("exit_check_interval_min", 2)
        self.regime_update_interval_min: int = loop.get("regime_update_interval_min", 30)
        self.circuit_check_interval_sec: int = loop.get("circuit_check_interval_sec", 60)
        self.heartbeat_interval_sec: int = loop.get("heartbeat_interval_sec", 300)

        # ── Schedule ─────────────────────────────────────────────────────────
        sched = settings.get("schedule", {})
        self.screener_hour_et: int = sched.get("screener_hour_et", 8)
        self.audit_hour_et: int = sched.get("audit_hour_et", 8)
        self.daily_reset_hour_et: int = sched.get("daily_reset_hour_et", 4)

        # ── Market hours ─────────────────────────────────────────────────────
        mkt = settings.get("market", {})
        self.equity_open_hour: int = mkt.get("equity_open_hour", 9)
        self.equity_open_min: int = mkt.get("equity_open_min", 35)
        self.equity_close_hour: int = mkt.get("equity_close_hour", 15)
        self.equity_close_min: int = mkt.get("equity_close_min", 55)
        self.timezone: str = _env("TIMEZONE", mkt.get("timezone", "America/New_York"))

        # ── Data ─────────────────────────────────────────────────────────────
        data_cfg = settings.get("data", {})
        self.equity_lookback_days: int = data_cfg.get("equity_lookback_days", 252)
        self.crypto_lookback_days: int = data_cfg.get("crypto_lookback_days", 45)
        self.cache_ttl_seconds: int = data_cfg.get("cache_ttl_seconds", 60)
        self.iex_feed: bool = data_cfg.get("iex_feed", True)

        # ── Notifications ────────────────────────────────────────────────────
        notif = settings.get("notifications", {})
        self.slack_webhook: str = _env("SLACK_WEBHOOK_URL", "")
        self.discord_webhook: str = _env("DISCORD_WEBHOOK_URL", "")
        self.telegram_token: str = _env("TELEGRAM_BOT_TOKEN", "")
        self.telegram_chat: str = _env("TELEGRAM_CHAT_ID", "")
        self.notify_on_trade: bool = notif.get("on_trade", True)
        self.notify_on_error: bool = notif.get("on_error", True)
        self.notify_on_circuit: bool = notif.get("on_circuit_breaker", True)
        self.notify_on_audit: bool = notif.get("on_audit", True)
        self.notify_on_kill: bool = notif.get("on_kill_switch", True)

        # ── Watchlist ────────────────────────────────────────────────────────
        core_wl = watchlist.get("core", {})
        self.core_equities: List[str] = core_wl.get("equities", ["NLR", "SMH", "LLY"])
        self.core_crypto: List[str] = core_wl.get("crypto", ["SOL/USD", "BTC/USD", "ETH/USD"])

        dyn = watchlist.get("dynamic", {})
        self.max_dynamic_slots: int = dyn.get("max_slots", 5)
        self.max_dynamic_crypto_slots: int = dyn.get("max_crypto_slots", 2)
        self.dynamic_equities: List[str] = dyn.get("equities", [])
        self.dynamic_crypto: List[str] = dyn.get("crypto", [])
        self.veto_symbols: List[str] = watchlist.get("veto", [])

        self.screener_equity_pool: List[str] = watchlist.get("screener_pool", {}).get("equities", [])
        self.screener_crypto_pool: List[str] = watchlist.get("screener_pool", {}).get("crypto", [])

        scr_cfg = watchlist.get("screener", {})
        eq_scr = scr_cfg.get("equity", {})
        self.scr_min_dollar_vol: float = eq_scr.get("min_avg_dollar_volume", 50_000_000)
        self.scr_min_price: float = eq_scr.get("min_price", 5.0)
        self.scr_require_fractionable: bool = eq_scr.get("require_fractionable", True)
        self.scr_min_return_20d: float = eq_scr.get("min_return_20d", 0.02)
        self.scr_min_atr_pct: float = eq_scr.get("min_atr_pct", 0.005)
        self.scr_max_atr_pct: float = eq_scr.get("max_atr_pct", 0.10)
        self.scr_top_n: int = eq_scr.get("top_n", 5)

        cr_scr = scr_cfg.get("crypto", {})
        self.scr_crypto_min_return_7d: float = cr_scr.get("min_return_7d", 0.0)
        self.scr_crypto_top_n: int = cr_scr.get("top_n", 2)

        # ── Strategy ─────────────────────────────────────────────────────────
        reg = strategy.get("regime", {})
        self.regime_spy_fast_ma: int = reg.get("spy_fast_ma", 50)
        self.regime_spy_slow_ma: int = reg.get("spy_slow_ma", 200)
        self.regime_adx_trending: float = reg.get("adx_trending_threshold", 20.0)

        tr = strategy.get("trend", {})
        self.trend_enabled: bool = tr.get("enabled", True)
        self.trend_fast_ema: int = tr.get("fast_ema", 20)
        self.trend_slow_ema: int = tr.get("slow_ema", 50)
        self.trend_adx_period: int = tr.get("adx_period", 14)
        self.trend_adx_threshold: float = tr.get("adx_threshold", 20.0)
        self.trend_min_slope_pct: float = tr.get("min_slope_pct", 0.0005)

        mr = strategy.get("mean_reversion", {})
        self.mr_enabled: bool = mr.get("enabled", True)
        self.mr_rsi_period: int = mr.get("rsi_period", 14)
        self.mr_rsi_oversold: float = mr.get("rsi_oversold", 35.0)
        self.mr_rsi_overbought: float = mr.get("rsi_overbought", 65.0)
        self.mr_bb_period: int = mr.get("bb_period", 20)
        self.mr_bb_std: float = mr.get("bb_std", 2.0)
        self.mr_require_bb_lower: bool = mr.get("require_bb_lower_touch", True)

        cm = strategy.get("crypto_momentum", {})
        self.crypto_enabled: bool = cm.get("enabled", True)
        self.crypto_fast_ema: int = cm.get("fast_ema", 12)
        self.crypto_slow_ema: int = cm.get("slow_ema", 26)
        self.crypto_rsi_period: int = cm.get("rsi_period", 14)
        self.crypto_rsi_min: float = cm.get("rsi_min", 40.0)
        self.crypto_rsi_max: float = cm.get("rsi_max", 70.0)
        self.crypto_vol_ratio_min: float = cm.get("volume_ratio_min", 1.3)

        eg = strategy.get("edge_gate", {})
        self.edge_crypto_min_pct: float = eg.get("crypto_min_edge_pct", 0.008)
        self.edge_equity_min_pct: float = eg.get("equity_min_edge_pct", 0.004)

        # ── Risk ─────────────────────────────────────────────────────────────
        siz = risk.get("sizing", {})
        self.risk_per_trade_pct: float = _float_env("RISK_PER_TRADE_PCT", siz.get("risk_per_trade_pct", 0.01))
        self.max_position_pct: float = siz.get("max_position_pct", 0.10)
        self.max_concurrent_positions: int = siz.get("max_concurrent_positions", 8)
        self.max_crypto_allocation_pct: float = siz.get("max_crypto_allocation_pct", 0.30)
        self.portfolio_heat_max: float = siz.get("portfolio_heat_max", 0.60)
        self.min_notional: float = siz.get("min_notional", 1.00)
        self.confluence_bonus_pct: float = siz.get("confluence_bonus_pct", 0.25)

        # Capital tiers — sizing/diversification that scales with account equity.
        # Applied each tick via apply_capital_tier(). An explicit RISK_PER_TRADE_PCT
        # env var pins risk-per-trade and is never overridden by a tier.
        self.capital_tiers: List[dict] = risk.get("capital_tiers", [])
        self._risk_env_pinned: bool = "RISK_PER_TRADE_PCT" in os.environ
        self.active_tier_name: str = "base"

        stops = risk.get("stops", {})
        self.atr_period: int = stops.get("atr_period", 14)
        self.atr_stop_mult: float = stops.get("atr_stop_mult", 2.0)
        self.take_profit_ratio: float = stops.get("take_profit_ratio", 2.0)
        self.trailing_arm_pct: float = stops.get("trailing_arm_pct", 0.08)
        self.trailing_giveback_pct: float = stops.get("trailing_giveback_pct", 0.05)
        self.time_stop_days: int = stops.get("time_stop_days", 10)
        self.time_stop_min_pnl: float = stops.get("time_stop_min_pnl_pct", 0.02)
        self.early_time_stop_days: int = stops.get("early_time_stop_days", 5)
        self.early_time_stop_loss: float = stops.get("early_time_stop_loss_pct", 0.015)

        cool = risk.get("cooldown", {})
        self.cooldown_equity_hours: int = cool.get("equity_hours", 24)
        self.cooldown_crypto_hours: int = cool.get("crypto_hours", 6)

        cb = risk.get("circuit_breakers", {})
        self.soft_halt_pct: float = cb.get("soft_halt_drawdown_pct", 0.05)
        self.hard_halt_pct: float = cb.get("hard_halt_drawdown_pct", 0.10)
        self.daily_loss_limit_pct: float = cb.get("daily_loss_limit_pct", 0.03)
        self.kill_switch_floor: float = cb.get("kill_switch_floor", 85.0)
        self.kill_switch_enabled: bool = cb.get("kill_switch_enabled", True)
        self.kill_switch_override: bool = cb.get("kill_switch_override", False)

        diag = risk.get("diagnostics", {})
        self.max_consecutive_rejections: int = diag.get("max_consecutive_order_rejections", 5)
        self.error_rate_window_min: int = diag.get("error_rate_window_minutes", 60)
        self.max_errors_per_window: int = diag.get("max_errors_per_window", 10)
        self.stale_data_threshold_min: int = diag.get("stale_data_threshold_minutes", 30)

    def reload(self) -> None:
        """Re-read all YAML files. Called at the top of each main loop iteration."""
        self._load()

    # Tier keys → (config attribute, cast). Only these are overlaid from a tier.
    _TIER_KEYS = {
        "max_concurrent_positions": ("max_concurrent_positions", int),
        "max_position_pct": ("max_position_pct", float),
        "portfolio_heat_max": ("portfolio_heat_max", float),
        "risk_per_trade_pct": ("risk_per_trade_pct", float),
        "max_crypto_allocation_pct": ("max_crypto_allocation_pct", float),
        "max_dynamic_slots": ("max_dynamic_slots", int),
        "max_dynamic_crypto_slots": ("max_dynamic_crypto_slots", int),
        "edge_equity_min_pct": ("edge_equity_min_pct", float),
        "edge_crypto_min_pct": ("edge_crypto_min_pct", float),
    }

    def apply_capital_tier(self, equity: float) -> str:
        """
        Scale sizing/diversification parameters to the current account size.

        Call this each tick AFTER reload() (which restores the flat YAML defaults)
        so the tier overlay always sits on fresh base values. Tiers are checked in
        order; the first whose `up_to` is >= equity wins, with the final (up_to:
        null) tier as the catch-all. Keys a tier omits keep their flat defaults.

        Returns the active tier name (e.g. "micro", "large", or "base" if no tiers
        are configured).
        """
        if not self.capital_tiers or equity <= 0:
            self.active_tier_name = "base"
            return self.active_tier_name

        tier = None
        for t in self.capital_tiers:
            up_to = t.get("up_to")
            if up_to is None or equity <= float(up_to):
                tier = t
                break
        if tier is None:
            tier = self.capital_tiers[-1]

        for key, (attr, cast) in self._TIER_KEYS.items():
            if key not in tier or tier[key] is None:
                continue
            # An explicit RISK_PER_TRADE_PCT env var always wins over the tier.
            if attr == "risk_per_trade_pct" and self._risk_env_pinned:
                continue
            setattr(self, attr, cast(tier[key]))

        # Hard safety ceiling — never let a tier push risk-per-trade above 5%.
        if self.risk_per_trade_pct > 0.05:
            self.risk_per_trade_pct = 0.05

        self.active_tier_name = tier.get("name", "unknown")
        return self.active_tier_name

    @property
    def all_equities(self) -> List[str]:
        """All equity symbols currently being watched (core + dynamic, deduped)."""
        return list(dict.fromkeys(self.core_equities + self.dynamic_equities))

    @property
    def all_crypto(self) -> List[str]:
        """All crypto symbols currently being watched (core + dynamic, deduped)."""
        return list(dict.fromkeys(self.core_crypto + self.dynamic_crypto))

    def validate(self) -> List[str]:
        """Return list of validation errors. Empty list = ok."""
        errors = []
        if not self.api_key:
            errors.append("APCA_API_KEY_ID env var not set")
        if not self.secret_key:
            errors.append("APCA_API_SECRET_KEY env var not set")
        if self.risk_per_trade_pct > 0.05:
            errors.append(f"risk_per_trade_pct={self.risk_per_trade_pct} > 5% (dangerously high for $100 account)")
        if self.hard_halt_pct <= self.soft_halt_pct:
            errors.append("hard_halt_drawdown_pct must be greater than soft_halt_drawdown_pct")
        return errors
