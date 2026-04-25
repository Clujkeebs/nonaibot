"""
Portfolio Tracker — live view of account, positions, P&L.

Pulls fresh data from Alpaca on every call; also maintains a local SQLite
ledger for daily/weekly P&L calculation (Alpaca resets unrealized_pl daily).

Exposes the exact dict shape required by RiskManager.
"""
from __future__ import annotations

import sqlite3
from datetime import date, datetime, timedelta
from typing import Dict, Optional, Tuple

import pytz

from alpaca.trading.client import TradingClient
from alpaca.trading.models import Position

import config
from utils.logger import log

ET = pytz.timezone(config.TIMEZONE)

# Alpaca positions API returns "BTCUSD"; data API uses "BTC/USD".
# Normalize on the way out so the rest of the codebase sees slash format.
_CRYPTO_SLASH: dict = {s.replace("/", ""): s for s in config.CRYPTO_SYMBOLS}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS portfolio_snapshots (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    snapshot_date  TEXT,
    portfolio_value REAL,
    cash            REAL,
    daily_pnl       REAL,
    created_at      TEXT
);
CREATE TABLE IF NOT EXISTS position_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol      TEXT,
    qty         REAL,
    avg_price   REAL,
    market_value REAL,
    unrealized_pl REAL,
    strategy    TEXT,
    opened_at   TEXT,
    updated_at  TEXT
);
"""


class PortfolioTracker:
    def __init__(self) -> None:
        self._client = TradingClient(
            api_key=config.ALPACA_API_KEY,
            secret_key=config.ALPACA_SECRET_KEY,
            paper=config.ALPACA_PAPER,
        )
        self._db = config.DB_PATH
        self._init_db()
        log.info("PortfolioTracker initialised")

    # ── Account ───────────────────────────────────────────────────────────────

    def get_account_state(self) -> Dict:
        """Return dict with portfolio_value, buying_power, cash, equity."""
        try:
            acc = self._client.get_account()
            return {
                "portfolio_value": float(acc.portfolio_value or 0),
                "buying_power":    float(acc.buying_power    or 0),
                "cash":            float(acc.cash            or 0),
                "equity":          float(acc.equity          or 0),
                "day_trade_count": int(acc.daytrade_count    or 0),
            }
        except Exception as e:
            log.error("get_account_state failed: {}", e)
            return {
                "portfolio_value": 0.0,
                "buying_power": 0.0,
                "cash": 0.0,
                "equity": 0.0,
                "day_trade_count": 0,
            }

    # ── Positions ─────────────────────────────────────────────────────────────

    def get_open_positions(self) -> Dict[str, dict]:
        """
        Return {symbol: {qty, avg_price, market_value, unrealized_pl, side}}
        Shape matches what RiskManager.check_signal() expects.
        """
        try:
            positions: list[Position] = self._client.get_all_positions()
            result = {}
            for p in positions:
                sym = _CRYPTO_SLASH.get(p.symbol, p.symbol)
                result[sym] = {
                    "qty":           float(p.qty),
                    "avg_price":     float(p.avg_entry_price),
                    "market_value":  float(p.market_value or 0),
                    "unrealized_pl": float(p.unrealized_pl or 0),
                    "side":          p.side.value if hasattr(p.side, "value") else str(p.side),
                }
            return result
        except Exception as e:
            log.error("get_open_positions failed: {}", e)
            return {}

    def position_for(self, symbol: str) -> Optional[dict]:
        positions = self.get_open_positions()
        return positions.get(symbol)

    # ── P&L ───────────────────────────────────────────────────────────────────

    def daily_pnl(self) -> float:
        """Sum of unrealized + realized P&L since today's open."""
        try:
            positions = self.get_open_positions()
            unrealized = sum(p["unrealized_pl"] for p in positions.values())
            realized   = self._realized_pnl_today()
            return unrealized + realized
        except Exception as e:
            log.error("daily_pnl error: {}", e)
            return 0.0

    def weekly_pnl(self) -> float:
        """P&L since Monday 00:00."""
        try:
            today = date.today()
            monday = today - timedelta(days=today.weekday())
            return self._pnl_since(monday.isoformat())
        except Exception as e:
            log.error("weekly_pnl error: {}", e)
            return 0.0

    def snapshot(self) -> None:
        """Persist today's portfolio value to SQLite (called at day close)."""
        state = self.get_account_state()
        today = date.today().isoformat()
        now   = datetime.now(ET).isoformat()
        with self._conn() as c:
            c.execute(
                """
                INSERT INTO portfolio_snapshots
                  (snapshot_date, portfolio_value, cash, daily_pnl, created_at)
                VALUES (?,?,?,?,?)
                """,
                (today, state["portfolio_value"], state["cash"], self.daily_pnl(), now),
            )
        log.info(
            "Portfolio snapshot: value={:.2f} cash={:.2f} daily_pnl={:.2f}",
            state["portfolio_value"], state["cash"], self.daily_pnl(),
        )

    def log_trade(
        self,
        symbol: str,
        qty: float,
        price: float,
        side: str,
        strategy: str,
        realized_pl: float = 0.0,
    ) -> None:
        now = datetime.now(ET).isoformat()
        with self._conn() as c:
            c.execute(
                """
                INSERT INTO position_log
                  (symbol, qty, avg_price, market_value, unrealized_pl, strategy, opened_at, updated_at)
                VALUES (?,?,?,?,?,?,?,?)
                """,
                (symbol, qty, price, qty * price, realized_pl, strategy, now, now),
            )

    # ── Summary helpers ───────────────────────────────────────────────────────

    def summary(self) -> str:
        state     = self.get_account_state()
        positions = self.get_open_positions()
        pnl       = self.daily_pnl()
        sign      = "+" if pnl >= 0 else ""
        lines = [
            f"Portfolio=${state['portfolio_value']:,.2f} | "
            f"Cash=${state['cash']:,.2f} | "
            f"BP=${state['buying_power']:,.2f} | "
            f"DailyPnL={sign}{pnl:,.2f}",
            f"Open positions ({len(positions)}):",
        ]
        for sym, p in positions.items():
            pl_sign = "+" if p["unrealized_pl"] >= 0 else ""
            lines.append(
                f"  {sym:12s}  qty={p['qty']:.4f}  "
                f"val=${p['market_value']:,.2f}  "
                f"PnL={pl_sign}{p['unrealized_pl']:,.2f}"
            )
        return "\n".join(lines)

    # ── SQLite ────────────────────────────────────────────────────────────────

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._conn() as c:
            for stmt in _SCHEMA.strip().split(";"):
                if stmt.strip():
                    c.execute(stmt)

    def _realized_pnl_today(self) -> float:
        today = date.today().isoformat()
        with self._conn() as c:
            row = c.execute(
                """
                SELECT COALESCE(SUM(unrealized_pl), 0) AS pnl
                FROM position_log
                WHERE DATE(opened_at) = ?
                """,
                (today,),
            ).fetchone()
        return float(row["pnl"]) if row else 0.0

    def _pnl_since(self, iso_date: str) -> float:
        with self._conn() as c:
            row = c.execute(
                """
                SELECT COALESCE(SUM(unrealized_pl), 0) AS pnl
                FROM position_log
                WHERE DATE(opened_at) >= ?
                """,
                (iso_date,),
            ).fetchone()
        return float(row["pnl"]) if row else 0.0
