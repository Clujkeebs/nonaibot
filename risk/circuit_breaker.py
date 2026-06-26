"""
CircuitBreaker — account-level safety valves.

Levels (escalating):
  NORMAL      — all clear, trade freely
  SOFT_HALT   — drawdown from equity high exceeded soft threshold
                → pause NEW entries, continue managing exits and stops
  HARD_HALT   — drawdown exceeded hard threshold OR daily loss exceeded
                → flatten all positions, stop all trading until next session
  KILLED      — equity fell below absolute floor ($85 default)
                → permanent halt until human re-enables via config flag

Self-correction:
  - SOFT_HALT resets at daily open (new day, fresh start)
  - HARD_HALT resets at daily open (circuit resets with the day)
  - KILLED never auto-resets — requires human intervention
"""
from __future__ import annotations

import logging
from enum import Enum, auto

from core.config import BotConfig

logger = logging.getLogger(__name__)


class HaltLevel(Enum):
    NORMAL = auto()
    SOFT_HALT = auto()
    HARD_HALT = auto()
    KILLED = auto()


class CircuitBreaker:
    def __init__(self, config: BotConfig) -> None:
        self._cfg = config
        self._level: HaltLevel = HaltLevel.NORMAL
        self._halt_reason: str = ""
        self._high_water_mark: float = 0.0
        self._consecutive_losses: int = 0

    # ── Public query interface ─────────────────────────────────────────────────

    @property
    def level(self) -> HaltLevel:
        return self._level

    def is_normal(self) -> bool:
        return self._level == HaltLevel.NORMAL

    def new_entries_allowed(self) -> bool:
        """False during any halt. Exits still run in SOFT_HALT."""
        return self._level == HaltLevel.NORMAL

    def exits_allowed(self) -> bool:
        """Always true — we must be able to exit positions even when halted."""
        return True

    def trading_allowed(self) -> bool:
        return self.new_entries_allowed()

    def is_halted(self) -> bool:
        return self._level != HaltLevel.NORMAL

    def full_halt_active(self) -> bool:
        return self._level in (HaltLevel.HARD_HALT, HaltLevel.KILLED)

    def halt_reason(self) -> str:
        return self._halt_reason

    def status_string(self) -> str:
        level_names = {
            HaltLevel.NORMAL: "NORMAL",
            HaltLevel.SOFT_HALT: "SOFT_HALT",
            HaltLevel.HARD_HALT: "HARD_HALT",
            HaltLevel.KILLED: "KILLED",
        }
        s = level_names[self._level]
        if self._halt_reason:
            s += f" ({self._halt_reason})"
        return s

    # ── Check methods — call these from the main loop ──────────────────────────

    def check_equity(self, current_equity: float) -> HaltLevel:
        """
        Main check — call with current portfolio value.
        Updates high-water mark and checks all thresholds.
        Returns the current halt level.
        """
        cfg = self._cfg

        if current_equity > self._high_water_mark:
            self._high_water_mark = current_equity

        # Kill switch (most severe — check first)
        if cfg.kill_switch_enabled and not cfg.kill_switch_override:
            if current_equity < cfg.kill_switch_floor:
                self._escalate(
                    HaltLevel.KILLED,
                    f"equity ${current_equity:.2f} < floor ${cfg.kill_switch_floor:.2f}. "
                    f"Recovery: set kill_switch_override: true in config/risk.yaml",
                )
                return self._level

        # Drawdown from high-water mark
        if self._high_water_mark > 0:
            drawdown = (self._high_water_mark - current_equity) / self._high_water_mark
            if drawdown >= cfg.hard_halt_pct:
                self._escalate(
                    HaltLevel.HARD_HALT,
                    f"drawdown {drawdown:.1%} >= hard_halt {cfg.hard_halt_pct:.1%} "
                    f"(hwm=${self._high_water_mark:.2f}). "
                    f"Recovery: auto-resets at next daily open",
                )
            elif drawdown >= cfg.soft_halt_pct:
                if self._level == HaltLevel.NORMAL:
                    self._escalate(
                        HaltLevel.SOFT_HALT,
                        f"drawdown {drawdown:.1%} >= soft_halt {cfg.soft_halt_pct:.1%} "
                        f"(hwm=${self._high_water_mark:.2f}). New entries paused.",
                    )
            elif self._level == HaltLevel.SOFT_HALT:
                # Drawdown recovered below soft threshold
                logger.info(
                    "CircuitBreaker: drawdown recovered to %.1f%% — resuming NORMAL", drawdown * 100
                )
                self._level = HaltLevel.NORMAL
                self._halt_reason = ""

        return self._level

    def check_daily_loss(self, daily_pnl: float, equity: float) -> HaltLevel:
        """Check if today's loss (realized + unrealized) exceeds the daily limit."""
        if equity <= 0 or daily_pnl >= 0:
            return self._level

        cfg = self._cfg
        loss_pct = abs(daily_pnl) / equity
        if loss_pct >= cfg.daily_loss_limit_pct:
            self._escalate(
                HaltLevel.HARD_HALT,
                f"daily loss {loss_pct:.1%} >= limit {cfg.daily_loss_limit_pct:.1%} "
                f"(P&L=${daily_pnl:+.2f}). Recovery: auto-resets tomorrow",
            )
        return self._level

    def record_loss(self) -> None:
        self._consecutive_losses += 1

    def record_win(self) -> None:
        self._consecutive_losses = 0

    def check_consecutive_losses(self, max_losses: int = 3) -> bool:
        """Trigger SOFT_HALT if consecutive losses exceed threshold."""
        if self._consecutive_losses >= max_losses:
            self._escalate(
                HaltLevel.SOFT_HALT,
                f"{self._consecutive_losses} consecutive losses — pausing new entries. "
                f"Recovery: auto-resets at next daily open",
            )
            return True
        return False

    def reset_daily(self) -> None:
        """
        Called at daily open (4 AM ET by default).
        SOFT_HALT and HARD_HALT reset. KILLED does NOT reset (requires human action).
        """
        if self._level in (HaltLevel.SOFT_HALT, HaltLevel.HARD_HALT):
            logger.info("CircuitBreaker: daily reset — resuming from %s", self._level.name)
            self._level = HaltLevel.NORMAL
            self._halt_reason = ""
        self._consecutive_losses = 0

    def set_high_water_mark(self, equity: float) -> None:
        """Initialize high-water mark from DB on startup."""
        if equity > self._high_water_mark:
            self._high_water_mark = equity

    def _escalate(self, level: HaltLevel, reason: str) -> None:
        if level.value <= self._level.value:
            return
        prev = self._level.name
        self._level = level
        self._halt_reason = reason
        logger.warning("CircuitBreaker: %s → %s — %s", prev, level.name, reason)
