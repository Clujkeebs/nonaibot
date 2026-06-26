"""
AuditReport — generates the daily pre-market status report.

The report covers:
  - Current equity and buying power
  - Day/week/total PnL (realized + unrealized)
  - Open positions with entry date, current value, unrealized PnL
  - Trades executed in last 24h with the rule that triggered each
  - Screener add/drops since last report
  - Errors and throttled components
  - Current risk state (NORMAL/SOFT_HALT/HARD_HALT/KILLED) and recovery instructions
  - One-line health verdict

Written to logs/audit_YYYYMMDD.txt and optionally delivered via webhooks.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import pytz

from core.config import BotConfig
from core.state import SQLiteState

logger = logging.getLogger(__name__)


class AuditReport:
    def __init__(self, config: BotConfig, state: SQLiteState) -> None:
        self._cfg = config
        self._state = state
        self._et = pytz.timezone(config.timezone)

    def generate(
        self,
        account: Optional[Dict],
        open_positions: Optional[Dict],
        circuit_status: str,
        circuit_reason: str,
    ) -> Dict[str, Any]:
        """
        Compile all report data into a dictionary.
        account: dict with keys equity, buying_power (from Alpaca account)
        open_positions: dict {symbol: {qty, market_value, unrealized_pl, avg_price}}
        circuit_status: e.g. "NORMAL", "SOFT_HALT"
        circuit_reason: human-readable halt reason
        """
        now = datetime.now(self._et)

        equity = float(account.get("equity", 0)) if account else 0
        buying_power = float(account.get("buying_power", 0)) if account else 0

        recent_trades = self._state.get_recent_trades(hours=24)
        screener_log = self._state.get_screener_log(days=1)
        errors = self._state.get_recent_errors(hours=24)
        equity_history = self._state.get_equity_history(days=7)

        # PnL calculations
        daily_realized = self._state.get_daily_realized_pnl()
        weekly_realized = self._state.get_weekly_realized_pnl()

        # Unrealized PnL from open positions
        unrealized_total = 0.0
        if open_positions:
            unrealized_total = sum(
                float(p.get("unrealized_pl", 0)) for p in open_positions.values()
            )

        daily_total_pnl = daily_realized + unrealized_total
        opening_equity = self._state.get_opening_equity()
        total_pnl = equity - (opening_equity if opening_equity > 0 else equity)

        # Determine health verdict
        verdict, recovery = self._health_verdict(
            circuit_status, circuit_reason, equity, daily_total_pnl, len(errors)
        )

        return {
            "generated_at": now.isoformat(),
            "account": {
                "equity": equity,
                "buying_power": buying_power,
                "open_positions_count": len(open_positions) if open_positions else 0,
            },
            "pnl": {
                "daily_realized": daily_realized,
                "daily_unrealized": unrealized_total,
                "daily_total": daily_total_pnl,
                "weekly_realized": weekly_realized,
                "total": total_pnl,
            },
            "positions": [
                {
                    "symbol": sym,
                    "qty": float(p.get("qty", 0)),
                    "market_value": float(p.get("market_value", 0)),
                    "avg_entry": float(p.get("avg_price", 0)),
                    "unrealized_pl": float(p.get("unrealized_pl", 0)),
                    "unrealized_pct": (
                        float(p.get("unrealized_plpc", 0)) * 100
                        if "unrealized_plpc" in p
                        else 0
                    ),
                }
                for sym, p in (open_positions or {}).items()
            ],
            "trades_24h": recent_trades,
            "screener_changes": screener_log,
            "errors_24h": errors,
            "risk": {
                "status": circuit_status,
                "reason": circuit_reason,
            },
            "health": {
                "verdict": verdict,
                "recovery": recovery,
            },
        }

    def format_text(self, report: Dict[str, Any]) -> str:
        """Format report dict as human-readable text for delivery."""
        lines: List[str] = []
        lines.append("=" * 60)
        lines.append("  DAILY TRADING BOT AUDIT REPORT")
        lines.append(f"  Generated: {report['generated_at']}")
        lines.append("=" * 60)

        # Health verdict (most important — top of report)
        verdict = report["health"]["verdict"]
        recovery = report["health"]["recovery"]
        lines.append(f"\n  HEALTH: {verdict}")
        if recovery:
            lines.append(f"  RECOVERY: {recovery}")

        # Account
        acc = report["account"]
        lines.append(f"\n── ACCOUNT ────────────────────────────────────────────")
        lines.append(f"  Equity:          ${acc['equity']:>10.2f}")
        lines.append(f"  Buying Power:    ${acc['buying_power']:>10.2f}")
        lines.append(f"  Open Positions:  {acc['open_positions_count']}")

        # PnL
        pnl = report["pnl"]
        lines.append(f"\n── P&L ────────────────────────────────────────────────")
        lines.append(f"  Today Realized:  ${pnl['daily_realized']:>+10.2f}")
        lines.append(f"  Today Unrealized:${pnl['daily_unrealized']:>+10.2f}")
        lines.append(f"  Today Total:     ${pnl['daily_total']:>+10.2f}")
        lines.append(f"  This Week:       ${pnl['weekly_realized']:>+10.2f}")
        lines.append(f"  All Time:        ${pnl['total']:>+10.2f}")

        # Positions
        if report["positions"]:
            lines.append(f"\n── OPEN POSITIONS ─────────────────────────────────────")
            lines.append(f"  {'Symbol':<12} {'Value':>8} {'Entry':>8} {'Unr P&L':>10} {'%':>7}")
            lines.append(f"  {'-'*12} {'-'*8} {'-'*8} {'-'*10} {'-'*7}")
            for p in report["positions"]:
                lines.append(
                    f"  {p['symbol']:<12} ${p['market_value']:>7.2f} "
                    f"${p['avg_entry']:>7.4f} ${p['unrealized_pl']:>+9.2f} "
                    f"{p['unrealized_pct']:>+6.1f}%"
                )

        # Trades last 24h
        trades = report["trades_24h"]
        if trades:
            lines.append(f"\n── TRADES (last 24h) ──────────────────────────────────")
            for t in trades[:20]:  # cap at 20 to keep report readable
                lines.append(
                    f"  {t.get('created_at','')[:16]} {t.get('side','?').upper():4} "
                    f"{t.get('symbol','?'):12} qty={t.get('qty',0):.4f} "
                    f"strategy={t.get('strategy','?')} reason={t.get('reason','')[:40]}"
                )
        else:
            lines.append(f"\n── TRADES (last 24h) — none ────────────────────────────")

        # Screener changes
        screener = report["screener_changes"]
        if screener:
            lines.append(f"\n── SCREENER CHANGES ───────────────────────────────────")
            for s in screener:
                lines.append(
                    f"  {s.get('logged_at','')[:16]} {s.get('action','?').upper():6} "
                    f"{s.get('symbol','?'):12} {s.get('metric','')}"
                )
        else:
            lines.append(f"\n── SCREENER — no changes in last 24h ─────────────────")

        # Errors
        errors = report["errors_24h"]
        if errors:
            lines.append(f"\n── ERRORS (last 24h, up to 10) ──────────────────────")
            for e in errors[:10]:
                lines.append(
                    f"  {e.get('logged_at','')[:16]} [{e.get('component','?')}] "
                    f"{e.get('message','')[:80]}"
                )

        # Risk state
        risk = report["risk"]
        lines.append(f"\n── RISK STATE ─────────────────────────────────────────")
        lines.append(f"  Status: {risk['status']}")
        if risk["reason"]:
            lines.append(f"  Reason: {risk['reason']}")

        lines.append("\n" + "=" * 60)
        return "\n".join(lines)

    def _health_verdict(
        self,
        circuit_status: str,
        circuit_reason: str,
        equity: float,
        daily_pnl: float,
        error_count: int,
    ) -> tuple[str, str]:
        """Return (verdict_string, recovery_instructions)."""
        if circuit_status == "KILLED":
            return (
                f"🔴 KILLED — {circuit_reason}",
                "Set kill_switch_override: true in config/risk.yaml and push to re-enable",
            )
        if circuit_status == "HARD_HALT":
            return (
                f"🟠 HARD HALT — {circuit_reason}",
                "Auto-resets at next daily open (4 AM ET). No action needed unless config change required.",
            )
        if circuit_status == "SOFT_HALT":
            return (
                f"🟡 SOFT HALT — {circuit_reason}",
                "New entries paused. Exits managed normally. Auto-resets at daily open.",
            )
        if daily_pnl < -5.0:
            return (
                f"🟡 CAUTION — daily P&L ${daily_pnl:+.2f}",
                "",
            )
        if error_count >= 10:
            return (
                f"🟡 CAUTION — {error_count} errors in last 24h",
                "Check logs for recurring errors. Consider adjusting API rate or strategy config.",
            )
        return (
            f"🟢 HEALTHY — equity ${equity:.2f}, today ${daily_pnl:+.2f}",
            "",
        )

    def write_to_file(self, text: str) -> str:
        """Write the report to a dated file in logs/. Returns the file path."""
        import os
        from pathlib import Path

        today = datetime.now(self._et).strftime("%Y%m%d")
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        path = log_dir / f"audit_{today}.txt"
        try:
            with open(path, "w") as f:
                f.write(text)
            logger.info("Audit report written to %s", path)
        except Exception as e:
            logger.error("Failed to write audit report: %s", e)
        return str(path)
