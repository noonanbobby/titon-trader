"""Automated drawdown circuit breakers for Project Titan.

Tracks daily, weekly, monthly, and total P&L against the session high-water
mark and triggers increasingly restrictive trading controls when drawdown
thresholds are breached.  Circuit breaker state is persisted to PostgreSQL so
it survives application restarts.

The recovery ladder allows the system to earn its way back to full capacity
after a drawdown event through a graduated process of consecutive winners.

Usage::

    from src.risk.circuit_breakers import CircuitBreaker

    cb = CircuitBreaker(risk_config=config, db_pool=pool)
    await cb.load_state()
    level = await cb.update_pnl(realized_pnl=150.0, unrealized_pnl=-30.0,
                                net_liquidation=148_500.0)
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from src.utils.logging import get_logger

if TYPE_CHECKING:
    import structlog

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class BreakerLevel(StrEnum):
    """Circuit breaker severity levels, ordered from least to most severe."""

    NORMAL = "NORMAL"
    CAUTION = "CAUTION"
    WARNING = "WARNING"
    HALT = "HALT"
    EMERGENCY = "EMERGENCY"


# Ordered list for severity comparison
_LEVEL_ORDER: list[str] = [
    BreakerLevel.NORMAL,
    BreakerLevel.CAUTION,
    BreakerLevel.WARNING,
    BreakerLevel.HALT,
    BreakerLevel.EMERGENCY,
]

# Default strategies allowed during WARNING level (overridden by config)
_DEFAULT_WARNING_STRATEGIES: list[str] = ["bull_put_spread", "iron_condor"]


# ---------------------------------------------------------------------------
# CircuitBreaker
# ---------------------------------------------------------------------------


class CircuitBreaker:
    """Automated drawdown circuit breaker with persistent state.

    Monitors P&L relative to the high-water mark and triggers escalating
    restrictions when drawdown thresholds are breached.  After a breaker
    fires, the system must pass through a recovery ladder before returning
    to full trading capacity.

    Args:
        risk_config: Dictionary loaded from ``config/risk_limits.yaml``.
        db_pool: Optional asyncpg connection pool for state persistence.
            When ``None``, state is held in memory only (useful for tests).
    """

    def __init__(
        self,
        risk_config: dict[str, Any],
        db_pool: Any | None = None,
    ) -> None:
        self._log: structlog.stdlib.BoundLogger = get_logger("risk.circuit_breakers")
        self._db_pool = db_pool
        self._config = risk_config

        # Parse circuit breaker levels from config
        cb_config = self._config.get("circuit_breakers", {})
        self._levels_config: list[dict[str, Any]] = cb_config.get("levels", [])
        self._daily_loss_limit: float = cb_config.get("daily_loss_limit", -3000.0)
        self._weekly_loss_limit: float = cb_config.get("weekly_loss_limit", -7500.0)
        self._monthly_loss_limit: float = cb_config.get("monthly_loss_limit", -15000.0)

        # Parse recovery ladder from config
        recovery_config = self._config.get("recovery", {})
        self._recovery_stages: list[dict[str, Any]] = recovery_config.get("stages", [])
        self._reset_on_loss: bool = recovery_config.get("reset_on_loss", True)
        self._min_recovery_days: int = recovery_config.get("min_recovery_days", 5)

        # Build threshold lookup: level_name -> drawdown_pct
        self._thresholds: dict[str, float] = {}
        self._size_multipliers: dict[str, float] = {}
        self._allowed_strategies_by_level: dict[str, list[str] | None] = {}
        for level_def in self._levels_config:
            name = level_def["name"]
            self._thresholds[name] = level_def.get("drawdown_pct", 0.0)
            self._size_multipliers[name] = level_def.get("size_multiplier", 1.0)
            self._allowed_strategies_by_level[name] = level_def.get(
                "allowed_strategies", None
            )

        # ── Mutable state (persisted to PostgreSQL) ──
        self.current_level: str = BreakerLevel.NORMAL
        self.high_water_mark: float = 0.0
        self.daily_pnl: float = 0.0
        self.weekly_pnl: float = 0.0
        self.monthly_pnl: float = 0.0
        self.total_drawdown_pct: float = 0.0
        self.recovery_stage: int = 0  # 0 = normal, 1-3 = recovery stages
        self.consecutive_winners: int = 0
        self._recovery_pnl_accumulated: float = 0.0
        self._last_triggered_at: datetime | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def update_pnl(
        self,
        realized_pnl: float,
        unrealized_pnl: float,
        net_liquidation: float,
    ) -> str:
        """Update P&L tracking and evaluate circuit breaker thresholds.

        Recalculates drawdown from the high-water mark, checks all dollar
        and percentage thresholds, triggers level changes as needed, and
        persists the updated state to PostgreSQL.

        Args:
            realized_pnl: Today's realized P&L in dollars.
            unrealized_pnl: Current unrealized P&L across all positions.
            net_liquidation: Current net liquidation value of the account.

        Returns:
            The new circuit breaker level as a string (e.g. ``"NORMAL"``).
        """
        total_pnl = realized_pnl + unrealized_pnl

        # Update daily P&L tracking
        self.daily_pnl = total_pnl

        # Update high-water mark if NLV is a new high
        if net_liquidation > self.high_water_mark:
            self.high_water_mark = net_liquidation
            self._log.debug(
                "high_water_mark_updated",
                high_water_mark=self.high_water_mark,
            )

        # Calculate drawdown from HWM
        if self.high_water_mark > 0:
            self.total_drawdown_pct = (
                self.high_water_mark - net_liquidation
            ) / self.high_water_mark
        else:
            self.total_drawdown_pct = 0.0

        # Ensure drawdown is non-negative (NLV above HWM means no drawdown)
        self.total_drawdown_pct = max(0.0, self.total_drawdown_pct)

        previous_level = self.current_level

        # Evaluate percentage-based thresholds (highest severity first)
        new_level = BreakerLevel.NORMAL
        for level_name in reversed(_LEVEL_ORDER):
            threshold = self._thresholds.get(level_name, 0.0)
            if threshold > 0.0 and self.total_drawdown_pct >= threshold:
                new_level = level_name
                break

        # Evaluate dollar-based limits (independent of percentage drawdown)
        dollar_level = self._check_dollar_limits()
        if _severity(dollar_level) > _severity(new_level):
            new_level = dollar_level

        # Only escalate, never de-escalate automatically
        # (de-escalation happens through the recovery ladder)
        if _severity(new_level) > _severity(self.current_level):
            self._log.warning(
                "circuit_breaker_triggered",
                previous_level=self.current_level,
                new_level=new_level,
                drawdown_pct=round(self.total_drawdown_pct, 4),
                daily_pnl=round(self.daily_pnl, 2),
                weekly_pnl=round(self.weekly_pnl, 2),
                monthly_pnl=round(self.monthly_pnl, 2),
                net_liquidation=round(net_liquidation, 2),
                high_water_mark=round(self.high_water_mark, 2),
            )
            self.current_level = new_level
            self._last_triggered_at = datetime.now(UTC)

            # Enter recovery when escalating beyond NORMAL
            if self.recovery_stage == 0 and new_level != BreakerLevel.NORMAL:
                self.recovery_stage = 1
                self.consecutive_winners = 0
                self._recovery_pnl_accumulated = 0.0
                self._log.info(
                    "recovery_ladder_entered",
                    stage=self.recovery_stage,
                )
        elif new_level != previous_level:
            self._log.info(
                "circuit_breaker_level_unchanged",
                current_level=self.current_level,
                evaluated_level=new_level,
                reason="no_escalation_allowed",
            )

        await self._persist_state()
        return self.current_level

    def is_trading_allowed(self) -> tuple[bool, str, float]:
        """Check whether trading is currently permitted.

        Returns:
            A tuple of ``(allowed, reason, size_multiplier)``.
            - ``allowed``: ``True`` if new trades may be opened.
            - ``reason``: Human-readable explanation of the current state.
            - ``size_multiplier``: Fraction of normal position sizing to use.
        """
        size_multiplier = self.get_size_multiplier()

        if self.current_level == BreakerLevel.NORMAL:
            return True, "Normal operations", size_multiplier

        if self.current_level == BreakerLevel.CAUTION:
            return (
                True,
                f"CAUTION: 2% drawdown — position sizes reduced "
                f"to {size_multiplier:.0%}",
                size_multiplier,
            )

        if self.current_level == BreakerLevel.WARNING:
            strategies = self.get_allowed_strategies()
            return (
                True,
                f"WARNING: 5% drawdown — restricted to {strategies}, "
                f"sizes at {size_multiplier:.0%}",
                size_multiplier,
            )

        if self.current_level == BreakerLevel.HALT:
            return (
                False,
                "HALT: 10% drawdown — no new trades permitted",
                0.0,
            )

        # EMERGENCY
        return (
            False,
            "EMERGENCY: 15% drawdown — close all positions, trading suspended",
            0.0,
        )

    def get_allowed_strategies(self) -> list[str] | None:
        """Return the list of strategies allowed at the current breaker level.

        Returns:
            ``None`` when all strategies are allowed (NORMAL, CAUTION levels).
            A restricted list of strategy names for WARNING level.
            An empty list for HALT and EMERGENCY levels (no trading).
        """
        if self.current_level in (BreakerLevel.NORMAL, BreakerLevel.CAUTION):
            return None  # All strategies allowed

        if self.current_level == BreakerLevel.WARNING:
            configured = self._allowed_strategies_by_level.get(BreakerLevel.WARNING)
            return configured if configured else _DEFAULT_WARNING_STRATEGIES

        # HALT and EMERGENCY — no strategies allowed
        return []

    def get_size_multiplier(self) -> float:
        """Calculate the effective position size multiplier.

        Combines the circuit breaker level multiplier with the recovery
        stage multiplier.  The final multiplier is the product of both.

        Returns:
            A float between 0.0 and 1.0 representing the fraction of
            normal position sizing to use.
        """
        # Base multiplier from breaker level
        level_multiplier = self._size_multipliers.get(self.current_level, 1.0)

        # Recovery stage multiplier
        recovery_multiplier = 1.0
        if self.recovery_stage > 0:
            for stage_def in self._recovery_stages:
                if stage_def.get("stage") == self.recovery_stage:
                    recovery_multiplier = stage_def.get("size_pct", 1.0)
                    break

        # Use the more restrictive of the two
        return min(level_multiplier, recovery_multiplier)

    async def record_trade_result(self, is_winner: bool, pnl: float) -> None:
        """Record the outcome of a completed trade for recovery tracking.

        Tracks consecutive winners and accumulated profit to determine
        whether the system should advance through the recovery ladder.

        Args:
            is_winner: ``True`` if the trade was profitable.
            pnl: Realized P&L of the trade in dollars.
        """
        if is_winner:
            self.consecutive_winners += 1
            self._recovery_pnl_accumulated += pnl
            self._log.info(
                "trade_result_winner",
                consecutive_winners=self.consecutive_winners,
                pnl=round(pnl, 2),
                recovery_stage=self.recovery_stage,
                recovery_pnl_accumulated=round(self._recovery_pnl_accumulated, 2),
            )
        else:
            self._log.info(
                "trade_result_loser",
                pnl=round(pnl, 2),
                recovery_stage=self.recovery_stage,
                consecutive_winners_before_reset=self.consecutive_winners,
            )
            if self._reset_on_loss and self.recovery_stage > 0:
                self.recovery_stage = 1
                self.consecutive_winners = 0
                self._recovery_pnl_accumulated = 0.0
                self._log.warning(
                    "recovery_reset_on_loss",
                    new_stage=self.recovery_stage,
                )
            else:
                self.consecutive_winners = 0

        # Check if we can advance to the next recovery stage
        if self.recovery_stage > 0 and is_winner:
            await self._check_recovery_advance()

        # Update weekly and monthly accumulators
        self.weekly_pnl += pnl
        self.monthly_pnl += pnl

        await self._persist_state()

    async def reset(self, level: str = "NORMAL") -> None:
        """Manually reset the circuit breaker to a specified level.

        This is an administrative action intended for manual intervention
        after a drawdown event has been reviewed.

        Args:
            level: The target circuit breaker level to reset to.
        """
        previous_level = self.current_level
        self.current_level = level
        self.recovery_stage = 0
        self.consecutive_winners = 0
        self._recovery_pnl_accumulated = 0.0
        self.daily_pnl = 0.0

        self._log.warning(
            "circuit_breaker_manual_reset",
            previous_level=previous_level,
            new_level=level,
        )

        await self._persist_state()

    async def load_state(self) -> None:
        """Load persisted circuit breaker state from PostgreSQL.

        Called at application startup to restore state from the last
        known checkpoint.  If no state exists or the database is
        unavailable, defaults are used.
        """
        if self._db_pool is None:
            self._log.info("circuit_breaker_load_state_skipped", reason="no_db_pool")
            return

        try:
            async with self._db_pool.acquire() as conn:
                row = await conn.fetchrow(
                    """
                    SELECT level, triggered_at, daily_pnl, weekly_pnl,
                           monthly_pnl, total_drawdown_pct, high_water_mark,
                           recovery_stage, consecutive_winners
                    FROM circuit_breaker_state
                    ORDER BY updated_at DESC
                    LIMIT 1
                    """
                )

                if row is not None:
                    self.current_level = row["level"]
                    self._last_triggered_at = row["triggered_at"]
                    self.daily_pnl = float(row["daily_pnl"] or 0.0)
                    self.weekly_pnl = float(row["weekly_pnl"] or 0.0)
                    self.monthly_pnl = float(row["monthly_pnl"] or 0.0)
                    self.total_drawdown_pct = float(row["total_drawdown_pct"] or 0.0)
                    self.high_water_mark = float(row["high_water_mark"] or 0.0)
                    self.recovery_stage = int(row["recovery_stage"] or 0)
                    self.consecutive_winners = int(row["consecutive_winners"] or 0)

                    self._log.info(
                        "circuit_breaker_state_loaded",
                        level=self.current_level,
                        high_water_mark=self.high_water_mark,
                        drawdown_pct=round(self.total_drawdown_pct, 4),
                        recovery_stage=self.recovery_stage,
                        consecutive_winners=self.consecutive_winners,
                    )
                else:
                    self._log.info(
                        "circuit_breaker_no_persisted_state",
                        reason="starting_fresh",
                    )
        except Exception:
            self._log.exception("circuit_breaker_load_state_failed")

    # ------------------------------------------------------------------
    # Periodic reset helpers
    # ------------------------------------------------------------------

    def reset_daily_pnl(self) -> None:
        """Reset the daily P&L accumulator.

        Should be called at the start of each trading day.
        """
        self._log.info(
            "daily_pnl_reset",
            previous_daily_pnl=round(self.daily_pnl, 2),
        )
        self.daily_pnl = 0.0

    def reset_weekly_pnl(self) -> None:
        """Reset the weekly P&L accumulator.

        Should be called at the start of each trading week.
        """
        self._log.info(
            "weekly_pnl_reset",
            previous_weekly_pnl=round(self.weekly_pnl, 2),
        )
        self.weekly_pnl = 0.0

    def reset_monthly_pnl(self) -> None:
        """Reset the monthly P&L accumulator.

        Should be called at the start of each calendar month.
        """
        self._log.info(
            "monthly_pnl_reset",
            previous_monthly_pnl=round(self.monthly_pnl, 2),
        )
        self.monthly_pnl = 0.0

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    def _check_dollar_limits(self) -> str:
        """Evaluate dollar-based P&L limits and return the implied level.

        Returns:
            The circuit breaker level implied by dollar-based limits.
        """
        if self.daily_pnl <= self._daily_loss_limit:
            self._log.warning(
                "daily_loss_limit_breached",
                daily_pnl=round(self.daily_pnl, 2),
                limit=self._daily_loss_limit,
            )
            return BreakerLevel.HALT

        if self.weekly_pnl <= self._weekly_loss_limit:
            self._log.warning(
                "weekly_loss_limit_breached",
                weekly_pnl=round(self.weekly_pnl, 2),
                limit=self._weekly_loss_limit,
            )
            return BreakerLevel.HALT

        if self.monthly_pnl <= self._monthly_loss_limit:
            self._log.warning(
                "monthly_loss_limit_breached",
                monthly_pnl=round(self.monthly_pnl, 2),
                limit=self._monthly_loss_limit,
            )
            return BreakerLevel.HALT

        return BreakerLevel.NORMAL

    async def _check_recovery_advance(self) -> None:
        """Check if the recovery ladder criteria are met to advance a stage.

        Examines consecutive winners and cumulative profit against the
        current stage's advancement criteria.
        """
        if self.recovery_stage <= 0:
            return

        current_stage_def: dict[str, Any] | None = None
        for stage_def in self._recovery_stages:
            if stage_def.get("stage") == self.recovery_stage:
                current_stage_def = stage_def
                break

        if current_stage_def is None:
            return

        advance_criteria = current_stage_def.get("advance_criteria")
        if advance_criteria is None:
            # Final stage -- no further advancement possible
            self._log.info(
                "recovery_final_stage",
                stage=self.recovery_stage,
            )
            # At final stage with criteria met, clear recovery
            if self.recovery_stage >= len(self._recovery_stages):
                self.recovery_stage = 0
                self.current_level = BreakerLevel.NORMAL
                self._log.info("recovery_complete", level=self.current_level)
            return

        required_winners = advance_criteria.get("consecutive_winners", 3)
        required_profit = advance_criteria.get("min_profit", 500.0)

        if (
            self.consecutive_winners >= required_winners
            and self._recovery_pnl_accumulated >= required_profit
        ):
            next_stage = self.recovery_stage + 1
            max_stage = len(self._recovery_stages)

            if next_stage > max_stage:
                # Recovery complete -- return to normal
                self._log.info(
                    "recovery_complete",
                    previous_stage=self.recovery_stage,
                )
                self.recovery_stage = 0
                self.current_level = BreakerLevel.NORMAL
                self.consecutive_winners = 0
                self._recovery_pnl_accumulated = 0.0
            else:
                self._log.info(
                    "recovery_stage_advanced",
                    previous_stage=self.recovery_stage,
                    new_stage=next_stage,
                    consecutive_winners=self.consecutive_winners,
                    recovery_pnl=round(self._recovery_pnl_accumulated, 2),
                )
                self.recovery_stage = next_stage
                self.consecutive_winners = 0
                self._recovery_pnl_accumulated = 0.0

                # De-escalate breaker level as recovery progresses
                if next_stage >= max_stage:
                    self.current_level = BreakerLevel.NORMAL
                elif self.current_level == BreakerLevel.HALT:
                    self.current_level = BreakerLevel.WARNING
                elif self.current_level == BreakerLevel.WARNING:
                    self.current_level = BreakerLevel.CAUTION

    async def _persist_state(self) -> None:
        """Save the current circuit breaker state to PostgreSQL.

        Inserts a new row into the ``circuit_breaker_state`` table.  If
        the database pool is not available, this is a no-op.
        """
        if self._db_pool is None:
            return

        try:
            async with self._db_pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO circuit_breaker_state
                        (level, triggered_at, daily_pnl, weekly_pnl,
                         monthly_pnl, total_drawdown_pct, high_water_mark,
                         recovery_stage, consecutive_winners, updated_at)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, NOW())
                    """,
                    self.current_level,
                    self._last_triggered_at,
                    self.daily_pnl,
                    self.weekly_pnl,
                    self.monthly_pnl,
                    self.total_drawdown_pct,
                    self.high_water_mark,
                    self.recovery_stage,
                    self.consecutive_winners,
                )
                self._log.debug(
                    "circuit_breaker_state_persisted",
                    level=self.current_level,
                    recovery_stage=self.recovery_stage,
                )
        except Exception:
            self._log.exception("circuit_breaker_persist_failed")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _severity(level: str) -> int:
    """Return a numeric severity for ordering circuit breaker levels.

    Args:
        level: Circuit breaker level name.

    Returns:
        Integer where higher values mean more severe.
    """
    try:
        return _LEVEL_ORDER.index(level)
    except ValueError:
        return 0
