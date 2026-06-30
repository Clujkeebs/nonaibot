"""
SQLiteState — all persistent state lives here.

Tables:
  orders          — every order submitted (trade ledger)
  equity_curve    — daily equity snapshots (for audit + backtest)
  cooldowns       — per-symbol re-entry cooldowns (survives Railway restarts)
  position_ages   — per-symbol entry timestamps (for time-stop tracking)
  position_meta   — per-symbol strategy + high-water mark (survives restarts)
  dynamic_watchlist — screener-managed symbols
  screener_log    — screener add/drop history
  daily_log       — daily PnL + reset tracking
  error_log       — API/strategy errors for self-diagnostics

Design: one SQLite file, WAL mode for concurrent reads.
No foreign keys to keep inserts simple (it's a single-writer bot).
"""
from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import date, datetime, timedelta
from typing import Dict, Generator, List, Optional

import pytz


class SQLiteState:
    def __init__(self, db_path: str, timezone: str = "America/New_York") -> None:
        self._path = db_path
        self._et = pytz.timezone(timezone)
        self._lock = threading.Lock()
        self._init_schema()

    @contextmanager
    def _conn(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(self._path, check_same_thread=False, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=OFF")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # ── Schema ─────────────────────────────────────────────────────────────────

    def _init_schema(self) -> None:
        with self._conn() as c:
            c.executescript("""
                CREATE TABLE IF NOT EXISTS orders (
                    id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    order_id     TEXT UNIQUE,
                    client_order_id TEXT,
                    symbol       TEXT NOT NULL,
                    side         TEXT NOT NULL,
                    qty          REAL,
                    notional     REAL,
                    price        REAL,
                    strategy     TEXT,
                    reason       TEXT,
                    status       TEXT DEFAULT 'pending',
                    filled_qty   REAL DEFAULT 0,
                    filled_price REAL DEFAULT 0,
                    created_at   TEXT NOT NULL,
                    updated_at   TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS equity_curve (
                    id            INTEGER PRIMARY KEY AUTOINCREMENT,
                    snapshot_date TEXT NOT NULL,
                    snapshot_time TEXT NOT NULL,
                    equity        REAL NOT NULL,
                    buying_power  REAL,
                    open_positions INTEGER,
                    daily_pnl     REAL,
                    created_at    TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS cooldowns (
                    symbol    TEXT PRIMARY KEY,
                    until_iso TEXT NOT NULL,
                    reason    TEXT
                );

                CREATE TABLE IF NOT EXISTS position_ages (
                    symbol     TEXT PRIMARY KEY,
                    opened_iso TEXT NOT NULL,
                    strategy   TEXT
                );

                CREATE TABLE IF NOT EXISTS position_highs (
                    symbol    TEXT PRIMARY KEY,
                    high_price REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS dynamic_watchlist (
                    symbol     TEXT PRIMARY KEY,
                    asset_type TEXT NOT NULL,
                    added_at   TEXT NOT NULL,
                    reason     TEXT
                );

                CREATE TABLE IF NOT EXISTS screener_log (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    action     TEXT NOT NULL,
                    symbol     TEXT NOT NULL,
                    asset_type TEXT,
                    metric     TEXT,
                    value      REAL,
                    logged_at  TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS daily_log (
                    log_date      TEXT PRIMARY KEY,
                    opening_equity REAL,
                    realized_pnl   REAL DEFAULT 0,
                    screener_run   INTEGER DEFAULT 0,
                    audit_run      INTEGER DEFAULT 0,
                    circuit_resets INTEGER DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS error_log (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    component  TEXT,
                    message    TEXT,
                    logged_at  TEXT NOT NULL
                );
            """)

    def _now_iso(self) -> str:
        return datetime.now(self._et).isoformat()

    def _today(self) -> str:
        return datetime.now(self._et).date().isoformat()

    # ── Orders / trade ledger ──────────────────────────────────────────────────

    def save_order(
        self,
        order_id: str,
        client_order_id: str,
        symbol: str,
        side: str,
        qty: Optional[float],
        notional: Optional[float],
        price: float,
        strategy: str,
        reason: str = "",
    ) -> None:
        now = self._now_iso()
        with self._conn() as c:
            c.execute("""
                INSERT OR IGNORE INTO orders
                  (order_id, client_order_id, symbol, side, qty, notional, price,
                   strategy, reason, status, created_at, updated_at)
                VALUES (?,?,?,?,?,?,?,?,?,'pending',?,?)
            """, (order_id, client_order_id, symbol, side, qty, notional, price,
                  strategy, reason, now, now))

    def update_order(self, order_id: str, status: str, filled_qty: float, filled_price: float) -> None:
        with self._conn() as c:
            c.execute("""
                UPDATE orders SET status=?, filled_qty=?, filled_price=?, updated_at=?
                WHERE order_id=?
            """, (status, filled_qty, filled_price, self._now_iso(), order_id))

    def get_recent_trades(self, hours: int = 24) -> List[dict]:
        cutoff = (datetime.now(self._et) - timedelta(hours=hours)).isoformat()
        with self._conn() as c:
            rows = c.execute("""
                SELECT * FROM orders
                WHERE created_at >= ? AND status IN ('filled', 'partially_filled')
                ORDER BY created_at DESC
            """, (cutoff,)).fetchall()
        return [dict(r) for r in rows]

    def idempotency_check(self, client_order_id: str) -> bool:
        """Return True if this client_order_id already exists (duplicate)."""
        with self._conn() as c:
            row = c.execute(
                "SELECT id FROM orders WHERE client_order_id=?", (client_order_id,)
            ).fetchone()
        return row is not None

    # ── Equity curve ───────────────────────────────────────────────────────────

    def save_equity_snapshot(
        self,
        equity: float,
        buying_power: float = 0.0,
        open_positions: int = 0,
        daily_pnl: float = 0.0,
    ) -> None:
        now = self._now_iso()
        today = self._today()
        with self._conn() as c:
            c.execute("""
                INSERT INTO equity_curve
                  (snapshot_date, snapshot_time, equity, buying_power, open_positions, daily_pnl, created_at)
                VALUES (?,?,?,?,?,?,?)
            """, (today, now, equity, buying_power, open_positions, daily_pnl, now))

    def get_equity_history(self, days: int = 30) -> List[dict]:
        cutoff = (datetime.now(self._et) - timedelta(days=days)).date().isoformat()
        with self._conn() as c:
            rows = c.execute("""
                SELECT snapshot_date, equity, daily_pnl FROM equity_curve
                WHERE snapshot_date >= ?
                ORDER BY snapshot_date ASC
            """, (cutoff,)).fetchall()
        return [dict(r) for r in rows]

    def get_high_water_mark(self) -> float:
        """Return highest recorded equity value (for drawdown calculation)."""
        with self._conn() as c:
            row = c.execute("SELECT MAX(equity) as hwm FROM equity_curve").fetchone()
        return float(row["hwm"] or 0)

    def get_opening_equity(self) -> float:
        """Return today's opening equity (first snapshot of today)."""
        today = self._today()
        with self._conn() as c:
            row = c.execute("""
                SELECT equity FROM equity_curve
                WHERE snapshot_date=? ORDER BY snapshot_time ASC LIMIT 1
            """, (today,)).fetchone()
        return float(row["equity"]) if row else 0.0

    # ── Cooldowns ──────────────────────────────────────────────────────────────

    def save_cooldown(self, symbol: str, until: datetime, reason: str) -> None:
        with self._conn() as c:
            c.execute("""
                INSERT OR REPLACE INTO cooldowns (symbol, until_iso, reason)
                VALUES (?,?,?)
            """, (symbol, until.isoformat(), reason))

    def get_cooldowns(self) -> Dict[str, datetime]:
        """Return active (not-yet-expired) cooldowns as {symbol: until_datetime}."""
        now = datetime.now(self._et)
        result: Dict[str, datetime] = {}
        with self._conn() as c:
            rows = c.execute("SELECT symbol, until_iso FROM cooldowns").fetchall()
            for row in rows:
                try:
                    until = datetime.fromisoformat(row["until_iso"])
                    if until.tzinfo is None:
                        until = self._et.localize(until)
                    if until > now:
                        result[row["symbol"]] = until
                    else:
                        c.execute("DELETE FROM cooldowns WHERE symbol=?", (row["symbol"],))
                except Exception:
                    pass
        return result

    def clear_cooldown(self, symbol: str) -> None:
        with self._conn() as c:
            c.execute("DELETE FROM cooldowns WHERE symbol=?", (symbol,))

    # ── Position ages ──────────────────────────────────────────────────────────

    def save_position_age(self, symbol: str, opened: datetime, strategy: str = "") -> None:
        with self._conn() as c:
            c.execute("""
                INSERT OR REPLACE INTO position_ages (symbol, opened_iso, strategy)
                VALUES (?,?,?)
            """, (symbol, opened.isoformat(), strategy))

    def get_position_ages(self) -> Dict[str, dict]:
        """Return {symbol: {opened: datetime, strategy: str}}."""
        with self._conn() as c:
            rows = c.execute("SELECT symbol, opened_iso, strategy FROM position_ages").fetchall()
        result: Dict[str, dict] = {}
        for row in rows:
            try:
                opened = datetime.fromisoformat(row["opened_iso"])
                if opened.tzinfo is None:
                    opened = self._et.localize(opened)
                result[row["symbol"]] = {"opened": opened, "strategy": row["strategy"] or ""}
            except Exception:
                pass
        return result

    def clear_position_age(self, symbol: str) -> None:
        with self._conn() as c:
            c.execute("DELETE FROM position_ages WHERE symbol=?", (symbol,))

    # ── Position high-water marks ──────────────────────────────────────────────

    def save_position_high(self, symbol: str, high_price: float) -> None:
        with self._conn() as c:
            c.execute("""
                INSERT OR REPLACE INTO position_highs (symbol, high_price)
                VALUES (?,?)
            """, (symbol, high_price))

    def get_position_highs(self) -> Dict[str, float]:
        with self._conn() as c:
            rows = c.execute("SELECT symbol, high_price FROM position_highs").fetchall()
        return {row["symbol"]: float(row["high_price"]) for row in rows}

    def clear_position_high(self, symbol: str) -> None:
        with self._conn() as c:
            c.execute("DELETE FROM position_highs WHERE symbol=?", (symbol,))

    # ── Dynamic watchlist ──────────────────────────────────────────────────────

    def get_dynamic_symbols(self, asset_type: str = "equity") -> List[str]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT symbol FROM dynamic_watchlist WHERE asset_type=?", (asset_type,)
            ).fetchall()
        return [row["symbol"] for row in rows]

    def add_dynamic_symbol(self, symbol: str, asset_type: str, reason: str) -> None:
        now = self._now_iso()
        with self._conn() as c:
            c.execute("""
                INSERT OR IGNORE INTO dynamic_watchlist (symbol, asset_type, added_at, reason)
                VALUES (?,?,?,?)
            """, (symbol, asset_type, now, reason))
        self._log_screener("add", symbol, asset_type, reason)

    def remove_dynamic_symbol(self, symbol: str, reason: str = "") -> None:
        asset_type = ""
        with self._conn() as c:
            row = c.execute(
                "SELECT asset_type FROM dynamic_watchlist WHERE symbol=?", (symbol,)
            ).fetchone()
            if row:
                asset_type = row["asset_type"]
            c.execute("DELETE FROM dynamic_watchlist WHERE symbol=?", (symbol,))
        self._log_screener("remove", symbol, asset_type, reason)

    def _log_screener(self, action: str, symbol: str, asset_type: str, metric: str, value: float = 0.0) -> None:
        with self._conn() as c:
            c.execute("""
                INSERT INTO screener_log (action, symbol, asset_type, metric, value, logged_at)
                VALUES (?,?,?,?,?,?)
            """, (action, symbol, asset_type, metric, value, self._now_iso()))

    def get_screener_log(self, days: int = 1) -> List[dict]:
        cutoff = (datetime.now(self._et) - timedelta(days=days)).isoformat()
        with self._conn() as c:
            rows = c.execute(
                "SELECT * FROM screener_log WHERE logged_at >= ? ORDER BY logged_at DESC",
                (cutoff,)
            ).fetchall()
        return [dict(r) for r in rows]

    # ── Daily log ──────────────────────────────────────────────────────────────

    def _ensure_daily_log(self, today: str, opening_equity: float = 0.0) -> None:
        with self._conn() as c:
            c.execute("""
                INSERT OR IGNORE INTO daily_log (log_date, opening_equity)
                VALUES (?,?)
            """, (today, opening_equity))

    def get_screener_ran_today(self) -> bool:
        today = self._today()
        with self._conn() as c:
            row = c.execute(
                "SELECT screener_run FROM daily_log WHERE log_date=?", (today,)
            ).fetchone()
        return bool(row and row["screener_run"])

    def mark_screener_ran(self) -> None:
        today = self._today()
        self._ensure_daily_log(today)
        with self._conn() as c:
            c.execute(
                "UPDATE daily_log SET screener_run=1 WHERE log_date=?", (today,)
            )

    def get_audit_ran_today(self) -> bool:
        today = self._today()
        with self._conn() as c:
            row = c.execute(
                "SELECT audit_run FROM daily_log WHERE log_date=?", (today,)
            ).fetchone()
        return bool(row and row["audit_run"])

    def mark_audit_ran(self) -> None:
        today = self._today()
        self._ensure_daily_log(today)
        with self._conn() as c:
            c.execute(
                "UPDATE daily_log SET audit_run=1 WHERE log_date=?", (today,)
            )

    def set_opening_equity(self, equity: float) -> None:
        today = self._today()
        self._ensure_daily_log(today, equity)
        with self._conn() as c:
            c.execute(
                "UPDATE daily_log SET opening_equity=? WHERE log_date=? AND opening_equity=0",
                (equity, today)
            )

    def add_realized_pnl(self, pnl: float) -> None:
        today = self._today()
        self._ensure_daily_log(today)
        with self._conn() as c:
            c.execute(
                "UPDATE daily_log SET realized_pnl = realized_pnl + ? WHERE log_date=?",
                (pnl, today)
            )

    def get_daily_realized_pnl(self) -> float:
        today = self._today()
        with self._conn() as c:
            row = c.execute(
                "SELECT realized_pnl FROM daily_log WHERE log_date=?", (today,)
            ).fetchone()
        return float(row["realized_pnl"]) if row else 0.0

    def get_weekly_realized_pnl(self) -> float:
        cutoff = (datetime.now(self._et) - timedelta(days=7)).date().isoformat()
        with self._conn() as c:
            row = c.execute(
                "SELECT SUM(realized_pnl) as total FROM daily_log WHERE log_date >= ?",
                (cutoff,)
            ).fetchone()
        return float(row["total"] or 0)

    # ── Error log ──────────────────────────────────────────────────────────────

    def log_error(self, component: str, message: str) -> None:
        with self._conn() as c:
            c.execute("""
                INSERT INTO error_log (component, message, logged_at)
                VALUES (?,?,?)
            """, (component, message[:2000], self._now_iso()))

    def get_recent_errors(self, hours: int = 24) -> List[dict]:
        cutoff = (datetime.now(self._et) - timedelta(hours=hours)).isoformat()
        with self._conn() as c:
            rows = c.execute("""
                SELECT * FROM error_log WHERE logged_at >= ?
                ORDER BY logged_at DESC LIMIT 100
            """, (cutoff,)).fetchall()
        return [dict(r) for r in rows]

    def error_count_recent(self, component: str, minutes: int = 60) -> int:
        cutoff = (datetime.now(self._et) - timedelta(minutes=minutes)).isoformat()
        with self._conn() as c:
            row = c.execute("""
                SELECT COUNT(*) as n FROM error_log
                WHERE component=? AND logged_at >= ?
            """, (component, cutoff)).fetchone()
        return int(row["n"] if row else 0)
