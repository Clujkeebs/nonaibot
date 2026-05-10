"""
AI Trade Journal — records every trade outcome for online learning.

Creates a SQLite table `ai_trades` with features captured at entry time
and outcomes logged at exit time. This is the foundation for all AI
components to learn from actual trading history.

Entry features recorded:
  - strategy name, symbol, sector
  - signal strength, entry price, entry ATR
  - regime (bull/bear, vol level)
  - confluence count, day of week
  - VIX proxy (realized vol)

Outcome features (updated on exit):
  - exit price, exit reason (stop, target, strategy, time)
  - return (%), R-multiple (return / ATR), holding days
  - win/loss flag

All AI models retrain from this journal periodically.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime
from typing import Dict, List, Optional

import pytz

import config
from utils.logger import log

ET = pytz.timezone(config.TIMEZONE)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS ai_trades (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol          TEXT    NOT NULL,
    strategy        TEXT    NOT NULL,
    sector          TEXT,
    entry_price     REAL    NOT NULL,
    entry_atr       REAL,
    signal_strength REAL,
    regime          TEXT,
    realized_vol    REAL,
    confluence      INTEGER DEFAULT 1,
    day_of_week     INTEGER,
    entry_time      TEXT    NOT NULL,

    -- Filled on exit
    exit_price      REAL,
    exit_time       TEXT,
    exit_reason     TEXT,
    return_pct      REAL,
    r_multiple      REAL,
    holding_days    REAL,
    is_win          INTEGER,

    created_at      TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_ai_trades_strategy ON ai_trades(strategy);
CREATE INDEX IF NOT EXISTS idx_ai_trades_entry_time ON ai_trades(entry_time);
CREATE INDEX IF NOT EXISTS idx_ai_trades_regime ON ai_trades(regime);
"""


class TradeJournal:
    """Records trade entries and exits for AI learning."""

    def __init__(self) -> None:
        self._db = config.DB_PATH
        self._init_db()
        log.info("AI TradeJournal initialised")

    def log_entry(
        self,
        symbol: str,
        strategy: str,
        sector: str,
        entry_price: float,
        entry_atr: float,
        signal_strength: float,
        regime: str,
        realized_vol: float,
        confluence: int = 1,
    ) -> int:
        """
        Record a new trade entry. Returns the row ID for later update.
        """
        now = datetime.now(ET)
        try:
            with self._conn() as c:
                c.execute(
                    """INSERT INTO ai_trades
                       (symbol, strategy, sector, entry_price, entry_atr,
                        signal_strength, regime, realized_vol, confluence,
                        day_of_week, entry_time, created_at)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        symbol, strategy, sector, entry_price, entry_atr,
                        signal_strength, regime, realized_vol, confluence,
                        now.weekday(), now.isoformat(), now.isoformat(),
                    ),
                )
                row_id = c.lastrowid
                log.debug("AI journal: entry {} #{} — {} @ {:.4f}", strategy, row_id, symbol, entry_price)
                return row_id or 0
        except Exception as e:
            log.warning("TradeJournal.log_entry error: {}", e)
            return 0

    def log_exit(
        self,
        symbol: str,
        exit_price: float,
        exit_reason: str,
        entry_price: Optional[float] = None,
        entry_atr: Optional[float] = None,
    ) -> None:
        """
        Update the most recent open trade for this symbol with exit info.
        Multiple entries for the same symbol use the most recent one.
        """
        try:
            now = datetime.now(ET)
            with self._conn() as c:
                # Find the most recent open trade for this symbol
                row = c.execute(
                    """SELECT id, entry_price, entry_atr, entry_time
                       FROM ai_trades
                       WHERE symbol=? AND exit_price IS NULL
                       ORDER BY id DESC LIMIT 1""",
                    (symbol,),
                ).fetchone()

                if row is None:
                    log.debug("AI journal: no open trade found for {}", symbol)
                    return

                trade_id = row["id"]
                ep = entry_price or float(row["entry_price"])
                ea = entry_atr or float(row["entry_atr"] or 0.0)

                ret_pct = (exit_price - ep) / ep if ep > 0 else 0.0
                r_mult = ret_pct / (ea / ep) if ea > 0 and ep > 0 else 0.0

                try:
                    entry_dt = datetime.fromisoformat(row["entry_time"])
                    holding = (now - entry_dt).total_seconds() / 86400.0
                except Exception:
                    holding = 0.0

                is_win = 1 if ret_pct > 0 else 0

                c.execute(
                    """UPDATE ai_trades
                       SET exit_price=?, exit_time=?, exit_reason=?,
                           return_pct=?, r_multiple=?, holding_days=?, is_win=?
                       WHERE id=?""",
                    (exit_price, now.isoformat(), exit_reason,
                     ret_pct, r_mult, holding, is_win, trade_id),
                )
                log.debug(
                    "AI journal: exit {} — {:.2%} R={:.2f} win={}",
                    symbol, ret_pct, r_mult, bool(is_win),
                )
        except Exception as e:
            log.warning("TradeJournal.log_exit error: {}", e)

    def get_trades(
        self,
        strategy: Optional[str] = None,
        min_date: Optional[str] = None,
        closed_only: bool = True,
        limit: int = 500,
    ) -> List[dict]:
        """Retrieve trades for model training."""
        try:
            with self._conn() as c:
                query = "SELECT * FROM ai_trades WHERE 1=1"
                params: list = []
                if strategy:
                    query += " AND strategy=?"
                    params.append(strategy)
                if min_date:
                    query += " AND entry_time >= ?"
                    params.append(min_date)
                if closed_only:
                    query += " AND exit_price IS NOT NULL"
                query += " ORDER BY id DESC LIMIT ?"
                params.append(limit)
                rows = c.execute(query, params).fetchall()
                return [dict(r) for r in rows]
        except Exception as e:
            log.warning("TradeJournal.get_trades error: {}", e)
            return []

    def recent_win_rate(self, strategy: Optional[str] = None, n: int = 20) -> float:
        """Return win rate over last N closed trades."""
        trades = self.get_trades(strategy=strategy, closed_only=True, limit=n)
        if not trades:
            return 0.5  # neutral prior
        wins = sum(1 for t in trades if t.get("is_win"))
        return wins / len(trades)

    def recent_avg_r(self, strategy: Optional[str] = None, n: int = 20) -> float:
        """Return average R-multiple over last N closed trades."""
        trades = self.get_trades(strategy=strategy, closed_only=True, limit=n)
        if not trades:
            return 0.0
        vals = [t.get("r_multiple", 0) or 0 for t in trades]
        return sum(vals) / len(vals) if vals else 0.0

    def total_trades(self, strategy: Optional[str] = None) -> int:
        """Count closed trades."""
        try:
            with self._conn() as c:
                query = "SELECT COUNT(*) FROM ai_trades WHERE exit_price IS NOT NULL"
                params: list = []
                if strategy:
                    query += " AND strategy=?"
                    params.append(strategy)
                row = c.execute(query, params).fetchone()
                return int(row[0]) if row else 0
        except Exception:
            return 0

    # ── SQLite ────────────────────────────────────────────────────────────────

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        try:
            with self._conn() as c:
                for stmt in _SCHEMA.strip().split(";"):
                    if stmt.strip():
                        c.execute(stmt)
        except Exception as e:
            log.warning("TradeJournal._init_db error: {}", e)
