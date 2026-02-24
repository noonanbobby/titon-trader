"""Comprehensive unit tests for src/risk/circuit_breakers.py.

Tests cover:
    - BreakerLevel severity ordering
    - Drawdown percentage threshold evaluation
    - Escalation-only behaviour (never auto-de-escalates)
    - Dollar-based daily/weekly/monthly loss limits
    - Recovery ladder advancement and loss-reset logic
    - min_recovery_days enforcement in is_trading_allowed and _check_recovery_advance
    - Size multiplier calculation (min of level and recovery multipliers)
    - Allowed strategy filtering per breaker level
    - High-water mark tracking
    - State persistence (load_state / _persist_state) with mocked db_pool
    - Manual reset clears all recovery state
    - Daily/weekly/monthly PnL reset helpers

Important implementation detail:
    update_pnl() sets ``daily_pnl = realized_pnl + unrealized_pnl``.  Dollar
    limits (daily -$3K, weekly -$7.5K, monthly -$15K) are checked independently
    of percentage drawdown.  To test percentage thresholds in isolation, pass
    ``realized_pnl=0, unrealized_pnl=0`` and vary only the ``net_liquidation``
    value -- this avoids inadvertently tripping dollar-based HALT thresholds.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.risk.circuit_breakers import (
    _LEVEL_ORDER,
    BreakerLevel,
    CircuitBreaker,
    _severity,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

ACCOUNT_SIZE = 150_000.0


@pytest.fixture()
def risk_config() -> dict[str, Any]:
    """Return a risk configuration dict mirroring config/risk_limits.yaml."""
    return {
        "circuit_breakers": {
            "levels": [
                {
                    "name": "NORMAL",
                    "drawdown_pct": 0.00,
                    "size_multiplier": 1.0,
                },
                {
                    "name": "CAUTION",
                    "drawdown_pct": 0.02,
                    "size_multiplier": 0.50,
                },
                {
                    "name": "WARNING",
                    "drawdown_pct": 0.05,
                    "size_multiplier": 0.25,
                    "allowed_strategies": ["bull_put_spread", "iron_condor"],
                },
                {
                    "name": "HALT",
                    "drawdown_pct": 0.10,
                },
                {
                    "name": "EMERGENCY",
                    "drawdown_pct": 0.15,
                },
            ],
            "daily_loss_limit": -3000.0,
            "weekly_loss_limit": -7500.0,
            "monthly_loss_limit": -15000.0,
        },
        "recovery": {
            "stages": [
                {
                    "stage": 1,
                    "size_pct": 0.50,
                    "advance_criteria": {
                        "consecutive_winners": 3,
                        "min_profit": 500.0,
                    },
                },
                {
                    "stage": 2,
                    "size_pct": 0.75,
                    "advance_criteria": {
                        "consecutive_winners": 3,
                        "min_profit": 750.0,
                    },
                },
                {
                    "stage": 3,
                    "size_pct": 1.00,
                    "advance_criteria": None,
                },
            ],
            "reset_on_loss": True,
            "min_recovery_days": 5,
        },
    }


@pytest.fixture()
def cb(risk_config: dict[str, Any]) -> CircuitBreaker:
    """Return a CircuitBreaker with no db_pool (in-memory only)."""
    breaker = CircuitBreaker(risk_config=risk_config, db_pool=None)
    breaker.high_water_mark = ACCOUNT_SIZE
    return breaker


def _mock_db_pool(row: dict[str, Any] | None = None) -> MagicMock:
    """Build a mock asyncpg pool that returns *row* from fetchrow.

    The pool exposes ``acquire()`` as an async context manager whose
    connection supports ``fetchrow()`` and ``execute()``.
    """
    conn = AsyncMock()
    conn.fetchrow = AsyncMock(return_value=row)
    conn.execute = AsyncMock()

    pool = MagicMock()

    @asynccontextmanager
    async def _acquire():
        yield conn

    pool.acquire = _acquire
    pool._mock_conn = conn  # stash for assertions
    return pool


# ===================================================================
# 1. BreakerLevel ordering / _severity()
# ===================================================================


class TestBreakerLevelSeverity:
    """_severity() must return monotonically increasing ints."""

    def test_severity_ordering(self) -> None:
        assert _severity(BreakerLevel.NORMAL) == 0
        assert _severity(BreakerLevel.CAUTION) == 1
        assert _severity(BreakerLevel.WARNING) == 2
        assert _severity(BreakerLevel.HALT) == 3
        assert _severity(BreakerLevel.EMERGENCY) == 4

    def test_severity_monotonically_increases(self) -> None:
        severities = [_severity(lv) for lv in _LEVEL_ORDER]
        for i in range(len(severities) - 1):
            assert severities[i] < severities[i + 1]

    def test_unknown_level_returns_zero(self) -> None:
        assert _severity("BOGUS") == 0
        assert _severity("") == 0

    def test_level_order_matches_enum_members(self) -> None:
        for member in BreakerLevel:
            assert member in _LEVEL_ORDER


# ===================================================================
# 2. Threshold evaluation at correct drawdown percentages
# ===================================================================


class TestThresholdEvaluation:
    """update_pnl must trigger correct levels based on drawdown pct.

    NOTE: To isolate percentage thresholds from dollar-based limits we
    pass ``realized_pnl=0, unrealized_pnl=0`` and control drawdown
    entirely through the ``net_liquidation`` parameter.
    """

    async def test_normal_no_drawdown(self, cb: CircuitBreaker) -> None:
        level = await cb.update_pnl(0.0, 0.0, ACCOUNT_SIZE)
        assert level == BreakerLevel.NORMAL

    async def test_caution_at_2pct(self, cb: CircuitBreaker) -> None:
        nlv = ACCOUNT_SIZE * 0.98  # exactly 2% drawdown
        level = await cb.update_pnl(0.0, 0.0, nlv)
        assert level == BreakerLevel.CAUTION

    async def test_warning_at_5pct(self, cb: CircuitBreaker) -> None:
        nlv = ACCOUNT_SIZE * 0.95  # 5% drawdown
        level = await cb.update_pnl(0.0, 0.0, nlv)
        assert level == BreakerLevel.WARNING

    async def test_halt_at_10pct(self, cb: CircuitBreaker) -> None:
        nlv = ACCOUNT_SIZE * 0.90  # 10% drawdown
        level = await cb.update_pnl(0.0, 0.0, nlv)
        assert level == BreakerLevel.HALT

    async def test_emergency_at_15pct(self, cb: CircuitBreaker) -> None:
        nlv = ACCOUNT_SIZE * 0.85  # 15% drawdown
        level = await cb.update_pnl(0.0, 0.0, nlv)
        assert level == BreakerLevel.EMERGENCY

    async def test_below_caution_stays_normal(self, cb: CircuitBreaker) -> None:
        nlv = ACCOUNT_SIZE * 0.99  # 1% drawdown -- below 2% threshold
        level = await cb.update_pnl(0.0, 0.0, nlv)
        assert level == BreakerLevel.NORMAL

    async def test_between_caution_and_warning(self, cb: CircuitBreaker) -> None:
        nlv = ACCOUNT_SIZE * 0.96  # 4% drawdown -- past CAUTION, below WARNING
        level = await cb.update_pnl(0.0, 0.0, nlv)
        assert level == BreakerLevel.CAUTION

    async def test_above_hwm_no_drawdown(self, cb: CircuitBreaker) -> None:
        """NLV above HWM should mean zero drawdown and NORMAL."""
        nlv = ACCOUNT_SIZE + 5000.0
        level = await cb.update_pnl(0.0, 0.0, nlv)
        assert level == BreakerLevel.NORMAL
        assert cb.total_drawdown_pct == 0.0

    async def test_just_below_threshold_stays_lower(self, cb: CircuitBreaker) -> None:
        """1.99% drawdown (just below 2%) should remain NORMAL."""
        nlv = ACCOUNT_SIZE * (1.0 - 0.0199)
        level = await cb.update_pnl(0.0, 0.0, nlv)
        assert level == BreakerLevel.NORMAL


# ===================================================================
# 3. Escalation only -- never auto-de-escalates
# ===================================================================


class TestEscalationOnly:
    """Circuit breaker must only escalate, never de-escalate automatically."""

    async def _trigger_to_level(self, cb: CircuitBreaker, target: str) -> None:
        """Push the breaker to *target* via percentage drawdown.

        Uses 0 realized/unrealized to avoid dollar-limit interference.
        """
        pct_by_level = {
            BreakerLevel.CAUTION: 0.02,
            BreakerLevel.WARNING: 0.05,
            BreakerLevel.HALT: 0.10,
            BreakerLevel.EMERGENCY: 0.15,
        }
        dd = pct_by_level[target]
        nlv = ACCOUNT_SIZE * (1.0 - dd)
        await cb.update_pnl(0.0, 0.0, nlv)

    async def test_escalate_normal_to_warning(self, cb: CircuitBreaker) -> None:
        await self._trigger_to_level(cb, BreakerLevel.WARNING)
        assert cb.current_level == BreakerLevel.WARNING

    async def test_no_deescalation_on_recovery(self, cb: CircuitBreaker) -> None:
        """After hitting WARNING, an improvement to CAUTION range stays WARNING."""
        await self._trigger_to_level(cb, BreakerLevel.WARNING)
        assert cb.current_level == BreakerLevel.WARNING

        # NLV recovers to CAUTION range -- should NOT de-escalate
        level = await cb.update_pnl(0.0, 0.0, ACCOUNT_SIZE * 0.98)
        assert level == BreakerLevel.WARNING

    async def test_no_deescalation_to_normal(self, cb: CircuitBreaker) -> None:
        """Even full recovery to zero drawdown doesn't auto-de-escalate."""
        await self._trigger_to_level(cb, BreakerLevel.WARNING)

        # Full recovery -- level stays WARNING
        level = await cb.update_pnl(0.0, 0.0, ACCOUNT_SIZE)
        assert level == BreakerLevel.WARNING

    async def test_further_escalation_is_allowed(self, cb: CircuitBreaker) -> None:
        """Escalation from WARNING to HALT must work."""
        await self._trigger_to_level(cb, BreakerLevel.WARNING)
        assert cb.current_level == BreakerLevel.WARNING

        await self._trigger_to_level(cb, BreakerLevel.HALT)
        assert cb.current_level == BreakerLevel.HALT

    async def test_emergency_is_terminal_without_manual_reset(
        self, cb: CircuitBreaker
    ) -> None:
        """EMERGENCY cannot be auto-de-escalated to anything lower."""
        await self._trigger_to_level(cb, BreakerLevel.EMERGENCY)
        assert cb.current_level == BreakerLevel.EMERGENCY

        level = await cb.update_pnl(0.0, 0.0, ACCOUNT_SIZE)
        assert level == BreakerLevel.EMERGENCY

    async def test_escalate_caution_to_emergency_directly(
        self, cb: CircuitBreaker
    ) -> None:
        """A single large drop can skip intermediate levels."""
        await self._trigger_to_level(cb, BreakerLevel.CAUTION)
        assert cb.current_level == BreakerLevel.CAUTION

        await self._trigger_to_level(cb, BreakerLevel.EMERGENCY)
        assert cb.current_level == BreakerLevel.EMERGENCY


# ===================================================================
# 4. Dollar limit checks trigger HALT
# ===================================================================


class TestDollarLimits:
    """Dollar-based daily/weekly/monthly limits independently trigger HALT."""

    async def test_daily_loss_limit_triggers_halt(self, cb: CircuitBreaker) -> None:
        # daily_pnl is set to realized + unrealized in update_pnl
        # -1500 + -1500 = -3000, exactly at the limit (<= -3000)
        level = await cb.update_pnl(-1500.0, -1500.0, ACCOUNT_SIZE)
        assert level == BreakerLevel.HALT

    async def test_daily_loss_below_limit_triggers_halt(
        self, cb: CircuitBreaker
    ) -> None:
        level = await cb.update_pnl(-2000.0, -1500.0, ACCOUNT_SIZE)
        assert level == BreakerLevel.HALT

    async def test_daily_loss_just_above_limit_no_halt(
        self, cb: CircuitBreaker
    ) -> None:
        """$-2999 daily P&L is above the -$3000 limit -- no HALT."""
        level = await cb.update_pnl(-1499.0, -1500.0, ACCOUNT_SIZE)
        assert level == BreakerLevel.NORMAL

    async def test_weekly_loss_limit_triggers_halt(self, cb: CircuitBreaker) -> None:
        cb.weekly_pnl = -7500.0
        level = await cb.update_pnl(0.0, 0.0, ACCOUNT_SIZE)
        assert level == BreakerLevel.HALT

    async def test_monthly_loss_limit_triggers_halt(self, cb: CircuitBreaker) -> None:
        cb.monthly_pnl = -15000.0
        level = await cb.update_pnl(0.0, 0.0, ACCOUNT_SIZE)
        assert level == BreakerLevel.HALT

    async def test_dollar_limit_escalates_above_percentage_level(
        self, cb: CircuitBreaker
    ) -> None:
        """Dollar limit yielding HALT should beat a percentage-only CAUTION."""
        # 2% drawdown -> CAUTION by pct, but daily_pnl breach -> HALT
        nlv = ACCOUNT_SIZE * 0.98
        level = await cb.update_pnl(-2000.0, -1500.0, nlv)
        assert level == BreakerLevel.HALT

    async def test_no_dollar_breach_stays_normal(self, cb: CircuitBreaker) -> None:
        level = await cb.update_pnl(-500.0, 0.0, ACCOUNT_SIZE)
        assert level == BreakerLevel.NORMAL


# ===================================================================
# 5. Recovery ladder -- consecutive winners advance, losers reset
# ===================================================================


class TestRecoveryLadder:
    """Recovery ladder advancement and reset-on-loss mechanics."""

    async def _trigger_warning(self, cb: CircuitBreaker) -> None:
        """Push circuit breaker to WARNING via 5% NLV drawdown.

        Uses 0 realized/unrealized to avoid dollar-limit interference.
        """
        await cb.update_pnl(0.0, 0.0, ACCOUNT_SIZE * 0.95)
        assert cb.current_level == BreakerLevel.WARNING
        assert cb.recovery_stage == 1

    async def test_recovery_enters_on_escalation(self, cb: CircuitBreaker) -> None:
        await self._trigger_warning(cb)
        assert cb.recovery_stage == 1
        assert cb.consecutive_winners == 0

    async def test_consecutive_winners_increment(self, cb: CircuitBreaker) -> None:
        await self._trigger_warning(cb)
        # Bypass min_recovery_days by setting trigger far in the past
        cb._last_triggered_at = datetime.now(UTC) - timedelta(days=30)

        await cb.record_trade_result(is_winner=True, pnl=100.0)
        assert cb.consecutive_winners == 1
        assert cb.recovery_stage == 1

        await cb.record_trade_result(is_winner=True, pnl=100.0)
        assert cb.consecutive_winners == 2
        assert cb.recovery_stage == 1

    async def test_advance_from_stage1_to_stage2(self, cb: CircuitBreaker) -> None:
        await self._trigger_warning(cb)
        cb._last_triggered_at = datetime.now(UTC) - timedelta(days=30)

        # 3 winners with >= $500 total profit should advance to stage 2
        await cb.record_trade_result(is_winner=True, pnl=200.0)
        await cb.record_trade_result(is_winner=True, pnl=200.0)
        await cb.record_trade_result(is_winner=True, pnl=200.0)

        assert cb.recovery_stage == 2
        # After advancing, consecutive_winners resets
        assert cb.consecutive_winners == 0
        assert cb._recovery_pnl_accumulated == 0.0

    async def test_advance_from_stage2_to_stage3(self, cb: CircuitBreaker) -> None:
        await self._trigger_warning(cb)
        cb._last_triggered_at = datetime.now(UTC) - timedelta(days=30)

        # Advance stage 1 -> 2
        for _ in range(3):
            await cb.record_trade_result(is_winner=True, pnl=200.0)
        assert cb.recovery_stage == 2

        # Advance stage 2 -> 3 (needs >= $750 profit)
        # Stage 3 is the final stage (max_stage=3), and the code checks
        # next_stage > max_stage to decide if recovery is complete.
        # next_stage = 2 + 1 = 3, max_stage = 3, so 3 > 3 is false,
        # meaning we advance TO stage 3 (with de-escalation) rather
        # than completing recovery.
        await cb.record_trade_result(is_winner=True, pnl=300.0)
        await cb.record_trade_result(is_winner=True, pnl=300.0)
        await cb.record_trade_result(is_winner=True, pnl=300.0)

        assert cb.recovery_stage == 3
        # De-escalation: at stage 3 (final), level set to NORMAL
        assert cb.current_level == BreakerLevel.NORMAL

    async def test_recovery_complete_from_stage3(self, cb: CircuitBreaker) -> None:
        """Stage 3 has advance_criteria=None, so no further advancement.

        But if recovery_stage >= len(stages), recovery is cleared.
        Stage 3 == len(stages) == 3, so the >= condition holds and
        recovery completes.
        """
        cb.current_level = BreakerLevel.CAUTION
        cb.recovery_stage = 3
        cb.consecutive_winners = 5
        cb._recovery_pnl_accumulated = 2000.0
        cb._last_triggered_at = datetime.now(UTC) - timedelta(days=30)

        await cb._check_recovery_advance()

        # Stage 3 is the final stage, advance_criteria is None
        # recovery_stage (3) >= len(stages) (3) -> recovery complete
        assert cb.recovery_stage == 0
        assert cb.current_level == BreakerLevel.NORMAL

    async def test_loss_resets_to_stage1(self, cb: CircuitBreaker) -> None:
        await self._trigger_warning(cb)
        cb._last_triggered_at = datetime.now(UTC) - timedelta(days=30)

        # Advance to stage 2
        for _ in range(3):
            await cb.record_trade_result(is_winner=True, pnl=200.0)
        assert cb.recovery_stage == 2

        # A loss resets back to stage 1
        await cb.record_trade_result(is_winner=False, pnl=-300.0)
        assert cb.recovery_stage == 1
        assert cb.consecutive_winners == 0
        assert cb._recovery_pnl_accumulated == 0.0

    async def test_loss_at_stage1_stays_at_stage1(self, cb: CircuitBreaker) -> None:
        await self._trigger_warning(cb)
        cb._last_triggered_at = datetime.now(UTC) - timedelta(days=30)

        await cb.record_trade_result(is_winner=True, pnl=100.0)
        await cb.record_trade_result(is_winner=False, pnl=-100.0)
        assert cb.recovery_stage == 1
        assert cb.consecutive_winners == 0

    async def test_insufficient_profit_blocks_advance(self, cb: CircuitBreaker) -> None:
        """3 winners but total profit below $500 should NOT advance."""
        await self._trigger_warning(cb)
        cb._last_triggered_at = datetime.now(UTC) - timedelta(days=30)

        # 3 winners with only $450 total (< $500 required)
        await cb.record_trade_result(is_winner=True, pnl=150.0)
        await cb.record_trade_result(is_winner=True, pnl=150.0)
        await cb.record_trade_result(is_winner=True, pnl=150.0)

        assert cb.recovery_stage == 1  # Still at stage 1

    async def test_trade_result_accumulates_weekly_monthly(
        self, cb: CircuitBreaker
    ) -> None:
        """record_trade_result adds pnl to weekly and monthly accumulators."""
        cb.weekly_pnl = 0.0
        cb.monthly_pnl = 0.0

        await cb.record_trade_result(is_winner=True, pnl=500.0)
        assert cb.weekly_pnl == 500.0
        assert cb.monthly_pnl == 500.0

        await cb.record_trade_result(is_winner=False, pnl=-200.0)
        assert cb.weekly_pnl == 300.0
        assert cb.monthly_pnl == 300.0


# ===================================================================
# 6. min_recovery_days enforcement
# ===================================================================


class TestMinRecoveryDays:
    """min_recovery_days blocks is_trading_allowed and recovery advance."""

    async def test_is_trading_allowed_blocked_during_cooling(
        self, cb: CircuitBreaker
    ) -> None:
        """HALT within cooling period should block trading."""
        cb.current_level = BreakerLevel.HALT
        cb._last_triggered_at = datetime.now(UTC) - timedelta(days=2)

        allowed, reason, mult = cb.is_trading_allowed()
        assert allowed is False
        assert "Min recovery period" in reason
        assert "day(s) remaining" in reason
        assert mult == 0.0

    async def test_is_trading_allowed_after_cooling_period(
        self, cb: CircuitBreaker
    ) -> None:
        """HALT after cooling period should use normal HALT logic (still blocked)."""
        cb.current_level = BreakerLevel.HALT
        cb._last_triggered_at = datetime.now(UTC) - timedelta(days=10)

        allowed, reason, mult = cb.is_trading_allowed()
        # HALT still blocks trading (it's the level itself, not cooling)
        assert allowed is False
        assert "HALT" in reason

    async def test_emergency_blocked_during_cooling(self, cb: CircuitBreaker) -> None:
        cb.current_level = BreakerLevel.EMERGENCY
        cb._last_triggered_at = datetime.now(UTC) - timedelta(days=1)

        allowed, reason, mult = cb.is_trading_allowed()
        assert allowed is False
        assert "Min recovery period" in reason
        assert mult == 0.0

    async def test_caution_not_affected_by_cooling(self, cb: CircuitBreaker) -> None:
        """CAUTION level should not trigger min_recovery_days check."""
        cb.current_level = BreakerLevel.CAUTION
        cb._last_triggered_at = datetime.now(UTC) - timedelta(seconds=10)

        allowed, reason, mult = cb.is_trading_allowed()
        assert allowed is True
        assert "CAUTION" in reason

    async def test_warning_not_affected_by_cooling(self, cb: CircuitBreaker) -> None:
        """WARNING level should not trigger min_recovery_days check."""
        cb.current_level = BreakerLevel.WARNING
        cb._last_triggered_at = datetime.now(UTC) - timedelta(seconds=10)

        allowed, reason, mult = cb.is_trading_allowed()
        assert allowed is True
        assert "WARNING" in reason

    async def test_recovery_advance_blocked_during_cooling(
        self, cb: CircuitBreaker
    ) -> None:
        """_check_recovery_advance should block when cooling period not met."""
        cb.current_level = BreakerLevel.WARNING
        cb.recovery_stage = 1
        cb.consecutive_winners = 5
        cb._recovery_pnl_accumulated = 1000.0
        cb._last_triggered_at = datetime.now(UTC) - timedelta(days=2)

        await cb._check_recovery_advance()
        # Should NOT have advanced despite meeting winner/profit criteria
        assert cb.recovery_stage == 1

    async def test_recovery_advance_allowed_after_cooling(
        self, cb: CircuitBreaker
    ) -> None:
        """_check_recovery_advance should work after cooling period."""
        cb.current_level = BreakerLevel.WARNING
        cb.recovery_stage = 1
        cb.consecutive_winners = 3
        cb._recovery_pnl_accumulated = 600.0
        cb._last_triggered_at = datetime.now(UTC) - timedelta(days=30)

        await cb._check_recovery_advance()
        assert cb.recovery_stage == 2

    async def test_min_recovery_days_zero_disables_check(
        self, risk_config: dict[str, Any]
    ) -> None:
        """When min_recovery_days is 0, cooling period should not apply."""
        risk_config["recovery"]["min_recovery_days"] = 0
        breaker = CircuitBreaker(risk_config=risk_config, db_pool=None)
        breaker.high_water_mark = ACCOUNT_SIZE
        breaker.current_level = BreakerLevel.HALT
        breaker._last_triggered_at = datetime.now(UTC)

        allowed, reason, mult = breaker.is_trading_allowed()
        # With min_recovery_days=0, should fall through to the HALT logic
        assert allowed is False
        assert "HALT" in reason
        assert "Min recovery" not in reason

    async def test_halt_no_triggered_at_skips_cooling(self, cb: CircuitBreaker) -> None:
        """HALT with _last_triggered_at=None skips cooling check."""
        cb.current_level = BreakerLevel.HALT
        cb._last_triggered_at = None

        allowed, reason, mult = cb.is_trading_allowed()
        assert allowed is False
        assert "HALT" in reason
        assert "Min recovery" not in reason


# ===================================================================
# 7. Size multiplier calculation
# ===================================================================


class TestSizeMultiplier:
    """get_size_multiplier returns min of level and recovery multipliers."""

    def test_normal_no_recovery(self, cb: CircuitBreaker) -> None:
        cb.current_level = BreakerLevel.NORMAL
        cb.recovery_stage = 0
        assert cb.get_size_multiplier() == 1.0

    def test_caution_no_recovery(self, cb: CircuitBreaker) -> None:
        cb.current_level = BreakerLevel.CAUTION
        cb.recovery_stage = 0
        assert cb.get_size_multiplier() == 0.50

    def test_warning_no_recovery(self, cb: CircuitBreaker) -> None:
        cb.current_level = BreakerLevel.WARNING
        cb.recovery_stage = 0
        assert cb.get_size_multiplier() == 0.25

    def test_normal_recovery_stage1(self, cb: CircuitBreaker) -> None:
        """Level=NORMAL but in recovery stage 1 -> min(1.0, 0.5) = 0.5."""
        cb.current_level = BreakerLevel.NORMAL
        cb.recovery_stage = 1
        assert cb.get_size_multiplier() == 0.50

    def test_normal_recovery_stage2(self, cb: CircuitBreaker) -> None:
        """Level=NORMAL but in recovery stage 2 -> min(1.0, 0.75) = 0.75."""
        cb.current_level = BreakerLevel.NORMAL
        cb.recovery_stage = 2
        assert cb.get_size_multiplier() == 0.75

    def test_normal_recovery_stage3(self, cb: CircuitBreaker) -> None:
        """Level=NORMAL and recovery stage 3 -> min(1.0, 1.0) = 1.0."""
        cb.current_level = BreakerLevel.NORMAL
        cb.recovery_stage = 3
        assert cb.get_size_multiplier() == 1.0

    def test_caution_recovery_stage1(self, cb: CircuitBreaker) -> None:
        """Level=CAUTION(0.5) and recovery stage 1(0.5) -> min = 0.5."""
        cb.current_level = BreakerLevel.CAUTION
        cb.recovery_stage = 1
        assert cb.get_size_multiplier() == 0.50

    def test_warning_recovery_stage2(self, cb: CircuitBreaker) -> None:
        """Level=WARNING(0.25) and recovery stage 2(0.75) -> min = 0.25."""
        cb.current_level = BreakerLevel.WARNING
        cb.recovery_stage = 2
        assert cb.get_size_multiplier() == 0.25

    def test_halt_returns_default_1(self, cb: CircuitBreaker) -> None:
        """HALT has no size_multiplier in config -> defaults to 1.0.

        In practice HALT blocks trading so the multiplier is academic.
        The function itself returns 1.0 since HALT has no config entry.
        """
        cb.current_level = BreakerLevel.HALT
        cb.recovery_stage = 0
        assert cb.get_size_multiplier() == 1.0

    def test_unknown_recovery_stage_defaults_to_1(self, cb: CircuitBreaker) -> None:
        """An unrecognised recovery stage should leave multiplier at 1.0."""
        cb.current_level = BreakerLevel.NORMAL
        cb.recovery_stage = 99  # Not a real stage
        assert cb.get_size_multiplier() == 1.0


# ===================================================================
# 8. Allowed strategies per level
# ===================================================================


class TestAllowedStrategies:
    """get_allowed_strategies returns correct lists for each level."""

    def test_normal_all_allowed(self, cb: CircuitBreaker) -> None:
        cb.current_level = BreakerLevel.NORMAL
        assert cb.get_allowed_strategies() is None

    def test_caution_all_allowed(self, cb: CircuitBreaker) -> None:
        cb.current_level = BreakerLevel.CAUTION
        assert cb.get_allowed_strategies() is None

    def test_warning_restricted(self, cb: CircuitBreaker) -> None:
        cb.current_level = BreakerLevel.WARNING
        strategies = cb.get_allowed_strategies()
        assert strategies == ["bull_put_spread", "iron_condor"]

    def test_halt_empty(self, cb: CircuitBreaker) -> None:
        cb.current_level = BreakerLevel.HALT
        assert cb.get_allowed_strategies() == []

    def test_emergency_empty(self, cb: CircuitBreaker) -> None:
        cb.current_level = BreakerLevel.EMERGENCY
        assert cb.get_allowed_strategies() == []

    def test_warning_falls_back_to_defaults(self, risk_config: dict[str, Any]) -> None:
        """When WARNING has no configured strategies, use defaults."""
        # Remove allowed_strategies from WARNING level config
        for level_def in risk_config["circuit_breakers"]["levels"]:
            if level_def["name"] == "WARNING":
                level_def.pop("allowed_strategies", None)

        breaker = CircuitBreaker(risk_config=risk_config, db_pool=None)
        breaker.current_level = BreakerLevel.WARNING
        strategies = breaker.get_allowed_strategies()
        assert strategies == ["bull_put_spread", "iron_condor"]


# ===================================================================
# 9. High-water mark tracking
# ===================================================================


class TestHighWaterMark:
    """HWM only updates when NLV increases."""

    async def test_hwm_increases_with_nlv(self, cb: CircuitBreaker) -> None:
        assert cb.high_water_mark == ACCOUNT_SIZE

        new_nlv = ACCOUNT_SIZE + 5000.0
        await cb.update_pnl(0.0, 0.0, new_nlv)
        assert cb.high_water_mark == new_nlv

    async def test_hwm_unchanged_when_nlv_drops(self, cb: CircuitBreaker) -> None:
        await cb.update_pnl(0.0, 0.0, ACCOUNT_SIZE - 5000.0)
        assert cb.high_water_mark == ACCOUNT_SIZE

    async def test_hwm_tracks_new_highs_across_updates(
        self, cb: CircuitBreaker
    ) -> None:
        await cb.update_pnl(0.0, 0.0, ACCOUNT_SIZE + 2000.0)
        assert cb.high_water_mark == ACCOUNT_SIZE + 2000.0

        await cb.update_pnl(0.0, 0.0, ACCOUNT_SIZE + 3000.0)
        assert cb.high_water_mark == ACCOUNT_SIZE + 3000.0

        # Drop -- HWM stays
        await cb.update_pnl(0.0, 0.0, ACCOUNT_SIZE + 1000.0)
        assert cb.high_water_mark == ACCOUNT_SIZE + 3000.0

    async def test_hwm_starts_at_zero_before_first_update(
        self, risk_config: dict[str, Any]
    ) -> None:
        breaker = CircuitBreaker(risk_config=risk_config, db_pool=None)
        assert breaker.high_water_mark == 0.0

        # First update sets the HWM
        await breaker.update_pnl(0.0, 0.0, ACCOUNT_SIZE)
        assert breaker.high_water_mark == ACCOUNT_SIZE

    async def test_drawdown_calculated_from_hwm(self, cb: CircuitBreaker) -> None:
        nlv = ACCOUNT_SIZE - 3000.0  # $3K below HWM
        await cb.update_pnl(0.0, 0.0, nlv)
        expected_drawdown = 3000.0 / ACCOUNT_SIZE
        assert abs(cb.total_drawdown_pct - expected_drawdown) < 1e-10

    async def test_drawdown_clamped_to_non_negative(self, cb: CircuitBreaker) -> None:
        """When NLV is above HWM, drawdown must be 0 not negative."""
        await cb.update_pnl(0.0, 0.0, ACCOUNT_SIZE + 10_000.0)
        assert cb.total_drawdown_pct == 0.0


# ===================================================================
# 10. State persistence -- load_state / _persist_state
# ===================================================================


class TestStatePersistence:
    """Mocked DB pool verifies SQL interaction for load and persist."""

    async def test_load_state_restores_all_fields(
        self, risk_config: dict[str, Any]
    ) -> None:
        triggered_at = datetime(2026, 2, 20, 14, 30, 0, tzinfo=UTC)
        row = {
            "level": "WARNING",
            "triggered_at": triggered_at,
            "daily_pnl": -1500.0,
            "weekly_pnl": -4000.0,
            "monthly_pnl": -8000.0,
            "total_drawdown_pct": 0.05,
            "high_water_mark": 155000.0,
            "recovery_stage": 2,
            "consecutive_winners": 1,
        }
        pool = _mock_db_pool(row=row)
        breaker = CircuitBreaker(risk_config=risk_config, db_pool=pool)

        await breaker.load_state()

        assert breaker.current_level == BreakerLevel.WARNING
        assert breaker._last_triggered_at == triggered_at
        assert breaker.daily_pnl == -1500.0
        assert breaker.weekly_pnl == -4000.0
        assert breaker.monthly_pnl == -8000.0
        assert breaker.total_drawdown_pct == 0.05
        assert breaker.high_water_mark == 155000.0
        assert breaker.recovery_stage == 2
        assert breaker.consecutive_winners == 1

    async def test_load_state_no_rows_stays_default(
        self, risk_config: dict[str, Any]
    ) -> None:
        pool = _mock_db_pool(row=None)
        breaker = CircuitBreaker(risk_config=risk_config, db_pool=pool)

        await breaker.load_state()
        # No row found -> stays at defaults
        assert breaker.current_level == BreakerLevel.NORMAL
        assert breaker.high_water_mark == 0.0

    async def test_load_state_no_db_defaults_to_halt(
        self, risk_config: dict[str, Any]
    ) -> None:
        breaker = CircuitBreaker(risk_config=risk_config, db_pool=None)

        await breaker.load_state()
        assert breaker.current_level == BreakerLevel.HALT

    async def test_load_state_db_error_defaults_to_halt(
        self, risk_config: dict[str, Any]
    ) -> None:
        conn = AsyncMock()
        conn.fetchrow = AsyncMock(side_effect=Exception("DB down"))

        pool = MagicMock()

        @asynccontextmanager
        async def _acquire():
            yield conn

        pool.acquire = _acquire
        breaker = CircuitBreaker(risk_config=risk_config, db_pool=pool)

        await breaker.load_state()
        assert breaker.current_level == BreakerLevel.HALT

    async def test_load_state_invalid_level_defaults_to_halt(
        self, risk_config: dict[str, Any]
    ) -> None:
        row = {
            "level": "INVALID_LEVEL",
            "triggered_at": None,
            "daily_pnl": 0.0,
            "weekly_pnl": 0.0,
            "monthly_pnl": 0.0,
            "total_drawdown_pct": 0.0,
            "high_water_mark": 150000.0,
            "recovery_stage": 0,
            "consecutive_winners": 0,
        }
        pool = _mock_db_pool(row=row)
        breaker = CircuitBreaker(risk_config=risk_config, db_pool=pool)

        await breaker.load_state()
        assert breaker.current_level == BreakerLevel.HALT

    async def test_persist_state_calls_execute(
        self, risk_config: dict[str, Any]
    ) -> None:
        pool = _mock_db_pool()
        breaker = CircuitBreaker(risk_config=risk_config, db_pool=pool)
        breaker.high_water_mark = ACCOUNT_SIZE
        breaker.current_level = BreakerLevel.CAUTION
        breaker.daily_pnl = -1000.0

        await breaker._persist_state()

        # The mock_conn was used inside the context manager
        conn = pool._mock_conn
        conn.execute.assert_called_once()
        call_args = conn.execute.call_args
        # First positional arg is the SQL string
        assert "INSERT INTO circuit_breaker_state" in call_args[0][0]

    async def test_persist_state_noop_without_db(self, cb: CircuitBreaker) -> None:
        """_persist_state with no db_pool is a silent no-op."""
        # This should not raise
        await cb._persist_state()

    async def test_persist_state_db_error_does_not_raise(
        self, risk_config: dict[str, Any]
    ) -> None:
        """If the DB throws during persist, exception is logged but not raised."""
        conn = AsyncMock()
        conn.execute = AsyncMock(side_effect=Exception("DB write failed"))

        pool = MagicMock()

        @asynccontextmanager
        async def _acquire():
            yield conn

        pool.acquire = _acquire
        breaker = CircuitBreaker(risk_config=risk_config, db_pool=pool)

        # Should not raise
        await breaker._persist_state()

    async def test_load_state_restores_all_valid_levels(
        self, risk_config: dict[str, Any]
    ) -> None:
        """Every BreakerLevel value should be accepted by load_state."""
        for level in BreakerLevel:
            row = {
                "level": level.value,
                "triggered_at": None,
                "daily_pnl": 0.0,
                "weekly_pnl": 0.0,
                "monthly_pnl": 0.0,
                "total_drawdown_pct": 0.0,
                "high_water_mark": 150000.0,
                "recovery_stage": 0,
                "consecutive_winners": 0,
            }
            pool = _mock_db_pool(row=row)
            breaker = CircuitBreaker(risk_config=risk_config, db_pool=pool)
            await breaker.load_state()
            assert breaker.current_level == level.value


# ===================================================================
# 11. Manual reset
# ===================================================================


class TestManualReset:
    """Manual reset() clears all recovery state."""

    async def test_reset_to_normal(self, cb: CircuitBreaker) -> None:
        cb.current_level = BreakerLevel.HALT
        cb.recovery_stage = 2
        cb.consecutive_winners = 5
        cb._recovery_pnl_accumulated = 2000.0
        cb.daily_pnl = -5000.0

        await cb.reset(level="NORMAL")

        assert cb.current_level == BreakerLevel.NORMAL
        assert cb.recovery_stage == 0
        assert cb.consecutive_winners == 0
        assert cb._recovery_pnl_accumulated == 0.0
        assert cb.daily_pnl == 0.0

    async def test_reset_to_caution(self, cb: CircuitBreaker) -> None:
        cb.current_level = BreakerLevel.EMERGENCY
        cb.recovery_stage = 1
        cb.consecutive_winners = 2

        await cb.reset(level="CAUTION")
        assert cb.current_level == BreakerLevel.CAUTION
        assert cb.recovery_stage == 0

    async def test_reset_default_is_normal(self, cb: CircuitBreaker) -> None:
        cb.current_level = BreakerLevel.WARNING
        await cb.reset()
        assert cb.current_level == BreakerLevel.NORMAL

    async def test_reset_preserves_hwm(self, cb: CircuitBreaker) -> None:
        """Reset should NOT touch the high-water mark."""
        cb.high_water_mark = 160000.0
        await cb.reset()
        assert cb.high_water_mark == 160000.0

    async def test_reset_preserves_weekly_monthly(self, cb: CircuitBreaker) -> None:
        """Reset clears daily_pnl but weekly/monthly are not touched."""
        cb.weekly_pnl = -3000.0
        cb.monthly_pnl = -7000.0
        await cb.reset()
        assert cb.weekly_pnl == -3000.0
        assert cb.monthly_pnl == -7000.0

    async def test_reset_then_update_allows_normal(self, cb: CircuitBreaker) -> None:
        """After manual reset to NORMAL, update_pnl with no drawdown stays NORMAL."""
        cb.current_level = BreakerLevel.EMERGENCY
        await cb.reset(level="NORMAL")

        level = await cb.update_pnl(0.0, 0.0, ACCOUNT_SIZE)
        assert level == BreakerLevel.NORMAL


# ===================================================================
# 12. PnL reset helpers
# ===================================================================


class TestPnlResetHelpers:
    """reset_daily_pnl / reset_weekly_pnl / reset_monthly_pnl."""

    def test_reset_daily_pnl(self, cb: CircuitBreaker) -> None:
        cb.daily_pnl = -2500.0
        cb.reset_daily_pnl()
        assert cb.daily_pnl == 0.0

    def test_reset_weekly_pnl(self, cb: CircuitBreaker) -> None:
        cb.weekly_pnl = -5000.0
        cb.reset_weekly_pnl()
        assert cb.weekly_pnl == 0.0

    def test_reset_monthly_pnl(self, cb: CircuitBreaker) -> None:
        cb.monthly_pnl = -12000.0
        cb.reset_monthly_pnl()
        assert cb.monthly_pnl == 0.0

    def test_reset_daily_idempotent(self, cb: CircuitBreaker) -> None:
        cb.daily_pnl = 0.0
        cb.reset_daily_pnl()
        assert cb.daily_pnl == 0.0

    def test_resets_are_independent(self, cb: CircuitBreaker) -> None:
        cb.daily_pnl = -100.0
        cb.weekly_pnl = -200.0
        cb.monthly_pnl = -300.0

        cb.reset_daily_pnl()
        assert cb.daily_pnl == 0.0
        assert cb.weekly_pnl == -200.0
        assert cb.monthly_pnl == -300.0

    def test_reset_positive_pnl(self, cb: CircuitBreaker) -> None:
        """Even positive P&L is zeroed out on reset."""
        cb.daily_pnl = 5000.0
        cb.reset_daily_pnl()
        assert cb.daily_pnl == 0.0


# ===================================================================
# Additional integration-style tests
# ===================================================================


class TestIntegration:
    """End-to-end scenarios combining multiple subsystems."""

    async def test_full_lifecycle_trigger_recover_reset(
        self, cb: CircuitBreaker
    ) -> None:
        """Simulate: normal -> warning -> recovery stage advance -> normal."""
        # Start normal
        assert cb.current_level == BreakerLevel.NORMAL

        # Trigger WARNING via 5% drawdown (zero pnl to avoid dollar limits)
        await cb.update_pnl(0.0, 0.0, ACCOUNT_SIZE * 0.95)
        assert cb.current_level == BreakerLevel.WARNING
        assert cb.recovery_stage == 1

        # Verify trading is allowed but restricted
        allowed, reason, mult = cb.is_trading_allowed()
        assert allowed is True
        assert "WARNING" in reason

        # Bypass cooling period
        cb._last_triggered_at = datetime.now(UTC) - timedelta(days=30)

        # Win 3 trades with enough profit -> stage 2
        for _ in range(3):
            await cb.record_trade_result(is_winner=True, pnl=200.0)
        assert cb.recovery_stage == 2

        # Win 3 more -> advance to stage 3 (final, NORMAL)
        for _ in range(3):
            await cb.record_trade_result(is_winner=True, pnl=300.0)
        assert cb.recovery_stage == 3
        assert cb.current_level == BreakerLevel.NORMAL

    async def test_escalation_during_recovery(self, cb: CircuitBreaker) -> None:
        """Further drawdown during recovery should escalate, not regress."""
        # Trigger WARNING (zero pnl to avoid dollar limits)
        await cb.update_pnl(0.0, 0.0, ACCOUNT_SIZE * 0.95)
        assert cb.current_level == BreakerLevel.WARNING

        # Further loss to HALT (zero pnl, just NLV-based)
        await cb.update_pnl(0.0, 0.0, ACCOUNT_SIZE * 0.90)
        assert cb.current_level == BreakerLevel.HALT
        # Recovery stage remains 1 (was already in recovery)
        assert cb.recovery_stage == 1

    async def test_is_trading_allowed_all_levels(self, cb: CircuitBreaker) -> None:
        """Verify is_trading_allowed tuple structure for all levels."""
        for level in BreakerLevel:
            cb.current_level = level
            cb._last_triggered_at = datetime.now(UTC) - timedelta(days=30)
            allowed, reason, mult = cb.is_trading_allowed()
            assert isinstance(allowed, bool)
            assert isinstance(reason, str)
            assert isinstance(mult, float)

            if level in (BreakerLevel.HALT, BreakerLevel.EMERGENCY):
                assert allowed is False
            elif level in (BreakerLevel.NORMAL, BreakerLevel.CAUTION):
                assert allowed is True

    async def test_concurrent_update_pnl_is_safe(self, cb: CircuitBreaker) -> None:
        """Concurrent calls to update_pnl should not corrupt state."""

        async def _update(nlv: float) -> str:
            return await cb.update_pnl(0.0, 0.0, nlv)

        # Fire multiple concurrent updates
        results = await asyncio.gather(
            _update(ACCOUNT_SIZE),
            _update(ACCOUNT_SIZE * 0.99),
            _update(ACCOUNT_SIZE * 0.98),
        )
        # All should return valid BreakerLevel strings
        for r in results:
            assert r in {lv.value for lv in BreakerLevel}

    async def test_recovery_with_reset_on_loss_disabled(
        self, risk_config: dict[str, Any]
    ) -> None:
        """When reset_on_loss=False, a loss only resets consecutive_winners."""
        risk_config["recovery"]["reset_on_loss"] = False
        breaker = CircuitBreaker(risk_config=risk_config, db_pool=None)
        breaker.high_water_mark = ACCOUNT_SIZE

        # Trigger WARNING to enter recovery (zero pnl)
        await breaker.update_pnl(0.0, 0.0, ACCOUNT_SIZE * 0.95)
        assert breaker.recovery_stage == 1
        breaker._last_triggered_at = datetime.now(UTC) - timedelta(days=30)

        # Advance to stage 2
        for _ in range(3):
            await breaker.record_trade_result(is_winner=True, pnl=200.0)
        assert breaker.recovery_stage == 2

        # A loss should NOT reset to stage 1 (reset_on_loss=False)
        await breaker.record_trade_result(is_winner=False, pnl=-300.0)
        assert breaker.recovery_stage == 2
        assert breaker.consecutive_winners == 0

    async def test_drawdown_exactly_at_threshold_boundary(
        self, cb: CircuitBreaker
    ) -> None:
        """Drawdown exactly equal to a threshold should trigger that level."""
        # Exactly 2.0% drawdown, zero pnl
        nlv = ACCOUNT_SIZE * (1.0 - 0.02)
        level = await cb.update_pnl(0.0, 0.0, nlv)
        assert level == BreakerLevel.CAUTION

    async def test_hwm_zero_means_no_drawdown(
        self, risk_config: dict[str, Any]
    ) -> None:
        """When HWM is zero (initial state), drawdown should be 0%."""
        breaker = CircuitBreaker(risk_config=risk_config, db_pool=None)
        breaker.high_water_mark = 0.0

        # First call sets HWM and calculates 0 drawdown
        level = await breaker.update_pnl(0.0, 0.0, 100.0)
        assert breaker.high_water_mark == 100.0
        assert breaker.total_drawdown_pct == 0.0
        assert level == BreakerLevel.NORMAL

    async def test_dollar_and_pct_combined_takes_worse(
        self, cb: CircuitBreaker
    ) -> None:
        """When both pct (CAUTION) and dollar (HALT) fire, HALT wins."""
        nlv = ACCOUNT_SIZE * 0.98  # 2% -> CAUTION
        # But daily pnl = -4000 -> HALT
        level = await cb.update_pnl(-2000.0, -2000.0, nlv)
        assert level == BreakerLevel.HALT

    async def test_recovery_de_escalation_from_halt(self, cb: CircuitBreaker) -> None:
        """Recovery ladder de-escalates breaker level as stages advance."""
        # Set up: at HALT, in recovery stage 1
        cb.current_level = BreakerLevel.HALT
        cb.recovery_stage = 1
        cb.consecutive_winners = 3
        cb._recovery_pnl_accumulated = 600.0
        cb._last_triggered_at = datetime.now(UTC) - timedelta(days=30)

        await cb._check_recovery_advance()

        # Stage advanced from 1 to 2, HALT de-escalated to WARNING
        assert cb.recovery_stage == 2
        assert cb.current_level == BreakerLevel.WARNING

    async def test_empty_config_uses_defaults(self) -> None:
        """CircuitBreaker with empty config should not crash."""
        breaker = CircuitBreaker(risk_config={}, db_pool=None)
        assert breaker.current_level == BreakerLevel.NORMAL
        assert breaker._daily_loss_limit == -3000.0
        assert breaker._weekly_loss_limit == -7500.0
        assert breaker._monthly_loss_limit == -15000.0
