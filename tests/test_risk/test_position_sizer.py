"""Comprehensive unit tests for src/risk/position_sizer.py.

Tests cover:
- Quarter-Kelly fraction calculation (divided by 4, capped at 0.25)
- Position sizing via math.floor (never rounds up)
- Per-trade risk caps (percentage and absolute dollar)
- Zero and edge cases (zero win prob, zero avg_loss, zero equity, etc.)
- Regime adjustment multipliers for all regime types
- Circuit breaker level multipliers (NORMAL through EMERGENCY)
- Recovery stage multipliers (stages 0-3)
- Combined regime + circuit breaker adjustments
- Max contracts cap enforcement
- PositionSize pydantic model validation
"""

from __future__ import annotations

import math
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from src.risk.position_sizer import (
    _CB_LEVEL_MULTIPLIERS,
    _DEFAULT_REGIME_FACTOR,
    _KELLY_FRACTION_MULTIPLIER,
    _MAX_KELLY_FRACTION,
    _RECOVERY_STAGE_MULTIPLIERS,
    _REGIME_FACTORS,
    PositionSize,
    PositionSizer,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_settings() -> MagicMock:
    """Create a mock Settings object with default trading parameters."""
    settings = MagicMock()
    settings.trading.per_trade_risk_pct = 0.02
    return settings


@pytest.fixture()
def default_risk_config() -> dict:
    """Minimal risk config matching the production risk_limits.yaml structure."""
    return {
        "per_trade": {
            "max_risk_pct": 0.02,
            "max_risk_dollars": 3000.0,
        },
        "circuit_breakers": {
            "levels": [
                {"name": "NORMAL"},
                {"name": "CAUTION", "size_multiplier": 0.50},
                {"name": "WARNING", "size_multiplier": 0.25},
                {"name": "HALT"},
                {"name": "EMERGENCY"},
            ],
        },
        "recovery": {
            "stages": [
                {"stage": 1, "size_pct": 0.50},
                {"stage": 2, "size_pct": 0.75},
                {"stage": 3, "size_pct": 1.00},
            ],
        },
    }


@pytest.fixture()
def sizer(mock_settings: MagicMock, default_risk_config: dict) -> PositionSizer:
    """Create a PositionSizer instance with mocked dependencies."""
    with patch("src.risk.position_sizer.get_logger") as mock_get_logger:
        mock_get_logger.return_value = MagicMock()
        return PositionSizer(settings=mock_settings, risk_config=default_risk_config)


# ===========================================================================
# 1. Quarter-Kelly Calculation Tests
# ===========================================================================


class TestQuarterKelly:
    """Verify Kelly fraction is divided by 4 and capped at 0.25."""

    def test_basic_quarter_kelly(self, sizer: PositionSizer) -> None:
        """Standard case: 72% win rate, 1.5:1 odds.

        Full Kelly = (0.72 * 1.5 - 0.28) / 1.5 = 0.5333...
        Quarter Kelly = 0.5333... * 0.25 = 0.1333...
        """
        kelly = sizer.calculate_kelly_fraction(
            win_probability=0.72,
            avg_win=4500.0,
            avg_loss=3000.0,
        )
        # Full Kelly: (0.72 * 1.5 - 0.28) / 1.5 = 0.5333...
        full_kelly = (0.72 * 1.5 - 0.28) / 1.5
        expected = full_kelly * _KELLY_FRACTION_MULTIPLIER
        assert abs(kelly - expected) < 1e-9
        assert abs(kelly - 0.1333) < 0.001

    def test_quarter_kelly_multiplier_is_025(self) -> None:
        """Confirm the constant is exactly 0.25 (quarter-Kelly)."""
        assert _KELLY_FRACTION_MULTIPLIER == 0.25

    def test_kelly_cap_at_025(self, sizer: PositionSizer) -> None:
        """Even with extreme edge, quarter-Kelly is capped at 0.25.

        With 99% win rate and 10:1 odds, full Kelly ~ 0.98.
        Quarter-Kelly = 0.245. But if full Kelly > 1.0, cap applies.
        """
        # 99% win rate, 100:1 odds => full Kelly approaches ~0.99
        # Quarter-Kelly = 0.99 * 0.25 = 0.2475 (under cap, but test the cap constant)
        assert _MAX_KELLY_FRACTION == 0.25

        # Force a scenario where quarter-Kelly would exceed 0.25
        # Full Kelly > 1.0 needed. That requires p > (q/b) + ... basically p * b >> q.
        # p=0.99, b=100 => full_kelly = (0.99*100 - 0.01)/100 = 0.9899
        # quarter = 0.9899 * 0.25 = 0.2475 (under 0.25, so capped already)
        # Let's use p=0.999, b=1000 => full = (999 - 0.001)/1000 = 0.998999
        # quarter = 0.998999 * 0.25 = 0.2497 (still under)
        # The cap only fires if full_kelly * 0.25 > 0.25 => full_kelly > 1.0
        # f = (p*b - q)/b. For f > 1: p*b - q > b => p*b - b > q => b(p-1) > q
        # Since p < 1, b(p-1) is negative. So f < 1 always when p < 1.
        # Therefore the cap can only fire when win_probability >= 1.0 (edge case).
        kelly = sizer.calculate_kelly_fraction(
            win_probability=0.99,
            avg_win=100_000.0,
            avg_loss=1.0,
        )
        assert kelly <= _MAX_KELLY_FRACTION

    def test_kelly_with_equal_win_loss(self, sizer: PositionSizer) -> None:
        """When avg_win equals avg_loss (b=1), Kelly simplifies to p - q = 2p - 1.

        For p=0.60: full_kelly = (0.60 * 1 - 0.40) / 1 = 0.20
        Quarter-Kelly = 0.20 * 0.25 = 0.05
        """
        kelly = sizer.calculate_kelly_fraction(
            win_probability=0.60,
            avg_win=500.0,
            avg_loss=500.0,
        )
        expected = (0.60 - 0.40) * _KELLY_FRACTION_MULTIPLIER
        assert abs(kelly - expected) < 1e-9
        assert abs(kelly - 0.05) < 1e-9

    def test_kelly_with_low_edge(self, sizer: PositionSizer) -> None:
        """Barely profitable: 52% win rate, 1:1 odds.

        Full Kelly = 0.04, Quarter Kelly = 0.01.
        """
        kelly = sizer.calculate_kelly_fraction(
            win_probability=0.52,
            avg_win=100.0,
            avg_loss=100.0,
        )
        expected = (0.52 - 0.48) * _KELLY_FRACTION_MULTIPLIER  # 0.04 * 0.25 = 0.01
        assert abs(kelly - expected) < 1e-9
        assert abs(kelly - 0.01) < 1e-9

    def test_kelly_negative_edge_returns_zero(self, sizer: PositionSizer) -> None:
        """When expected value is negative, Kelly should return 0.

        p=0.40, b=1 => full_kelly = (0.40 - 0.60)/1 = -0.20
        """
        kelly = sizer.calculate_kelly_fraction(
            win_probability=0.40,
            avg_win=100.0,
            avg_loss=100.0,
        )
        assert kelly == 0.0

    def test_kelly_breakeven_returns_zero(self, sizer: PositionSizer) -> None:
        """At exact breakeven (p=0.50, b=1), Kelly is zero."""
        kelly = sizer.calculate_kelly_fraction(
            win_probability=0.50,
            avg_win=100.0,
            avg_loss=100.0,
        )
        assert kelly == 0.0

    def test_kelly_high_odds_low_probability(self, sizer: PositionSizer) -> None:
        """Low win rate but high payoff ratio can still yield positive Kelly.

        p=0.30, avg_win=1000, avg_loss=200 => b=5
        full_kelly = (0.30 * 5 - 0.70) / 5 = (1.5 - 0.7) / 5 = 0.16
        quarter = 0.04
        """
        kelly = sizer.calculate_kelly_fraction(
            win_probability=0.30,
            avg_win=1000.0,
            avg_loss=200.0,
        )
        expected = ((0.30 * 5.0 - 0.70) / 5.0) * _KELLY_FRACTION_MULTIPLIER
        assert abs(kelly - expected) < 1e-9
        assert abs(kelly - 0.04) < 1e-9

    def test_kelly_formula_matches_spec(self, sizer: PositionSizer) -> None:
        """Verify the formula matches the CLAUDE.md specification exactly.

        Kelly = (p * b - q) / b, with quarter-Kelly = Kelly / 4.
        """
        p = 0.65
        q = 1.0 - p
        avg_win = 600.0
        avg_loss = 400.0
        b = avg_win / avg_loss  # 1.5

        full_kelly = (p * b - q) / b
        expected_quarter = full_kelly * 0.25

        actual = sizer.calculate_kelly_fraction(p, avg_win, avg_loss)
        assert abs(actual - expected_quarter) < 1e-9


# ===========================================================================
# 2. Zero and Edge Cases
# ===========================================================================


class TestZeroAndEdgeCases:
    """Test boundary conditions and degenerate inputs."""

    def test_zero_win_probability(self, sizer: PositionSizer) -> None:
        """Zero win probability should return Kelly = 0."""
        kelly = sizer.calculate_kelly_fraction(
            win_probability=0.0,
            avg_win=500.0,
            avg_loss=500.0,
        )
        assert kelly == 0.0

    def test_negative_win_probability(self, sizer: PositionSizer) -> None:
        """Negative win probability should return Kelly = 0."""
        kelly = sizer.calculate_kelly_fraction(
            win_probability=-0.1,
            avg_win=500.0,
            avg_loss=500.0,
        )
        assert kelly == 0.0

    def test_win_probability_of_one(self, sizer: PositionSizer) -> None:
        """Win probability of 1.0 is unrealistic but handled gracefully.

        p=1.0, q=0.0, b=2 => full_kelly = (1.0*2 - 0)/2 = 1.0
        quarter = 0.25 (capped at _MAX_KELLY_FRACTION)
        """
        kelly = sizer.calculate_kelly_fraction(
            win_probability=1.0,
            avg_win=500.0,
            avg_loss=250.0,
        )
        # The code checks win_probability >= 1.0 as an edge case
        # but continues gracefully. Verify it returns a non-negative value.
        assert kelly >= 0.0
        assert kelly <= _MAX_KELLY_FRACTION

    def test_zero_avg_win(self, sizer: PositionSizer) -> None:
        """Zero average win should return Kelly = 0."""
        kelly = sizer.calculate_kelly_fraction(
            win_probability=0.65,
            avg_win=0.0,
            avg_loss=500.0,
        )
        assert kelly == 0.0

    def test_zero_avg_loss(self, sizer: PositionSizer) -> None:
        """Zero average loss should return Kelly = 0 (avoids division by zero)."""
        kelly = sizer.calculate_kelly_fraction(
            win_probability=0.65,
            avg_win=500.0,
            avg_loss=0.0,
        )
        assert kelly == 0.0

    def test_negative_avg_win(self, sizer: PositionSizer) -> None:
        """Negative avg_win should return Kelly = 0."""
        kelly = sizer.calculate_kelly_fraction(
            win_probability=0.65,
            avg_win=-100.0,
            avg_loss=500.0,
        )
        assert kelly == 0.0

    def test_negative_avg_loss(self, sizer: PositionSizer) -> None:
        """Negative avg_loss should return Kelly = 0."""
        kelly = sizer.calculate_kelly_fraction(
            win_probability=0.65,
            avg_win=500.0,
            avg_loss=-100.0,
        )
        assert kelly == 0.0

    def test_zero_account_equity(self, sizer: PositionSizer) -> None:
        """Zero account equity should return zero contracts."""
        result = sizer.calculate_position_size(
            account_equity=0.0,
            max_loss_per_contract=500.0,
            win_probability=0.65,
            avg_win=500.0,
            avg_loss=300.0,
            regime="low_vol_trend",
        )
        assert result.contracts == 0
        assert result.dollar_risk == 0.0

    def test_negative_account_equity(self, sizer: PositionSizer) -> None:
        """Negative account equity should return zero contracts."""
        result = sizer.calculate_position_size(
            account_equity=-10_000.0,
            max_loss_per_contract=500.0,
            win_probability=0.65,
            avg_win=500.0,
            avg_loss=300.0,
            regime="low_vol_trend",
        )
        assert result.contracts == 0
        assert result.dollar_risk == 0.0

    def test_zero_max_loss_per_contract(self, sizer: PositionSizer) -> None:
        """Zero max loss per contract should return zero contracts."""
        result = sizer.calculate_position_size(
            account_equity=150_000.0,
            max_loss_per_contract=0.0,
            win_probability=0.65,
            avg_win=500.0,
            avg_loss=300.0,
            regime="low_vol_trend",
        )
        assert result.contracts == 0
        assert result.dollar_risk == 0.0

    def test_negative_max_loss_per_contract(self, sizer: PositionSizer) -> None:
        """Negative max loss per contract should return zero contracts."""
        result = sizer.calculate_position_size(
            account_equity=150_000.0,
            max_loss_per_contract=-500.0,
            win_probability=0.65,
            avg_win=500.0,
            avg_loss=300.0,
            regime="low_vol_trend",
        )
        assert result.contracts == 0
        assert result.dollar_risk == 0.0

    def test_negative_kelly_returns_zero_contracts(self, sizer: PositionSizer) -> None:
        """When Kelly fraction is negative (no edge), zero contracts."""
        result = sizer.calculate_position_size(
            account_equity=150_000.0,
            max_loss_per_contract=500.0,
            win_probability=0.30,  # Bad win rate
            avg_win=100.0,  # Small wins
            avg_loss=500.0,  # Big losses
            regime="low_vol_trend",
        )
        assert result.contracts == 0
        assert result.dollar_risk == 0.0
        assert result.kelly_fraction == 0.0

    def test_very_small_account(self, sizer: PositionSizer) -> None:
        """Very small account: risk budget may be too small for even 1 contract."""
        result = sizer.calculate_position_size(
            account_equity=1_000.0,  # Tiny account
            max_loss_per_contract=500.0,  # $500 per contract
            win_probability=0.65,
            avg_win=500.0,
            avg_loss=300.0,
            regime="low_vol_trend",
        )
        # $1000 * 0.02 = $20 max risk, can't afford a $500 contract
        # But Kelly fraction also factors in: Kelly ~0.0425
        # dollar_risk = 1000 * 0.0425 = $42.50, capped at $20 by max_risk_pct
        # $20 / $500 = 0.04, math.floor = 0 contracts
        assert result.contracts == 0


# ===========================================================================
# 3. Position Sizing (math.floor, risk caps)
# ===========================================================================


class TestPositionSizing:
    """Verify contract count uses math.floor and respects risk limits."""

    def test_floor_not_round(self, sizer: PositionSizer) -> None:
        """Position sizing must use math.floor, never round up.

        If dollar_risk / max_loss = 4.99, result must be 4, not 5.
        """
        # Use values that produce a fractional contract count.
        # With $150K, kelly ~0.1333, dollar_risk = $20K, capped at $3K
        # $3K / $700 per contract = 4.28 => floor = 4
        result = sizer.calculate_position_size(
            account_equity=150_000.0,
            max_loss_per_contract=700.0,
            win_probability=0.72,
            avg_win=4500.0,
            avg_loss=3000.0,
            regime="low_vol_trend",
        )
        assert result.contracts == 4
        assert isinstance(result.contracts, int)

    def test_floor_behavior_explicit(self) -> None:
        """Confirm math.floor always rounds down for positive values."""
        assert math.floor(4.99) == 4
        assert math.floor(4.01) == 4
        assert math.floor(4.0) == 4
        assert math.floor(0.99) == 0
        assert math.floor(0.01) == 0

    def test_per_trade_risk_pct_cap(self, sizer: PositionSizer) -> None:
        """Dollar risk is capped at account_equity * max_risk_pct (2% = $3000)."""
        # High Kelly would suggest more risk, but 2% cap limits it.
        result = sizer.calculate_position_size(
            account_equity=150_000.0,
            max_loss_per_contract=500.0,
            win_probability=0.72,
            avg_win=4500.0,
            avg_loss=3000.0,
            regime="low_vol_trend",
        )
        # Kelly ~0.1333, uncapped dollar_risk = $20K
        # Capped at min($3K by pct, $3K by dollars) = $3K
        # $3K / $500 = 6 contracts
        assert result.contracts == 6
        assert result.dollar_risk == 3000.0

    def test_absolute_dollar_cap(
        self, mock_settings: MagicMock, default_risk_config: dict
    ) -> None:
        """Dollar risk capped at max_risk_dollars from config."""
        # Lower the dollar cap to $1500
        default_risk_config["per_trade"]["max_risk_dollars"] = 1500.0
        with patch("src.risk.position_sizer.get_logger") as mock_get_logger:
            mock_get_logger.return_value = MagicMock()
            sizer = PositionSizer(
                settings=mock_settings, risk_config=default_risk_config
            )

        result = sizer.calculate_position_size(
            account_equity=150_000.0,
            max_loss_per_contract=500.0,
            win_probability=0.72,
            avg_win=4500.0,
            avg_loss=3000.0,
            regime="low_vol_trend",
        )
        # Capped at $1500, so $1500 / $500 = 3 contracts
        assert result.contracts == 3
        assert result.dollar_risk == 1500.0

    def test_risk_pct_dominates_when_lower(
        self, mock_settings: MagicMock, default_risk_config: dict
    ) -> None:
        """When percentage cap is lower than dollar cap, percentage wins."""
        default_risk_config["per_trade"]["max_risk_pct"] = 0.01  # 1% = $1500
        default_risk_config["per_trade"]["max_risk_dollars"] = 5000.0  # Higher
        with patch("src.risk.position_sizer.get_logger") as mock_get_logger:
            mock_get_logger.return_value = MagicMock()
            sizer = PositionSizer(
                settings=mock_settings, risk_config=default_risk_config
            )

        result = sizer.calculate_position_size(
            account_equity=150_000.0,
            max_loss_per_contract=500.0,
            win_probability=0.72,
            avg_win=4500.0,
            avg_loss=3000.0,
            regime="low_vol_trend",
        )
        # 1% of $150K = $1500, $1500 / $500 = 3 contracts
        assert result.contracts == 3
        assert result.dollar_risk == 1500.0

    def test_spread_too_wide_for_budget(self, sizer: PositionSizer) -> None:
        """When max_loss_per_contract exceeds entire risk budget, 0 contracts.

        The dangerous min-1 override was removed. The sizer correctly returns 0.
        """
        result = sizer.calculate_position_size(
            account_equity=150_000.0,
            max_loss_per_contract=5000.0,  # Way more than $3K budget
            win_probability=0.72,
            avg_win=4500.0,
            avg_loss=3000.0,
            regime="low_vol_trend",
        )
        assert result.contracts == 0
        assert result.dollar_risk == 0.0

    def test_dollar_risk_consistent_with_contracts(self, sizer: PositionSizer) -> None:
        """dollar_risk in result must equal contracts * max_loss_per_contract."""
        result = sizer.calculate_position_size(
            account_equity=150_000.0,
            max_loss_per_contract=500.0,
            win_probability=0.65,
            avg_win=600.0,
            avg_loss=400.0,
            regime="low_vol_trend",
        )
        assert result.dollar_risk == result.contracts * result.risk_per_contract

    def test_exact_division_contracts(self, sizer: PositionSizer) -> None:
        """When dollar_risk / max_loss divides evenly, exact contract count."""
        # $3000 / $500 = exactly 6
        result = sizer.calculate_position_size(
            account_equity=150_000.0,
            max_loss_per_contract=500.0,
            win_probability=0.72,
            avg_win=4500.0,
            avg_loss=3000.0,
            regime="low_vol_trend",
        )
        assert result.contracts == 6

    def test_bull_call_spread_sizing_from_spec(self, sizer: PositionSizer) -> None:
        """From spec: $5 spread, $150K account, 2% risk = 6 contracts max.

        With a strong Kelly (high enough to hit the $3K cap), we get
        $3K / $500 = 6 contracts.
        """
        result = sizer.calculate_position_size(
            account_equity=150_000.0,
            max_loss_per_contract=500.0,  # $5 spread * 100 multiplier
            win_probability=0.72,
            avg_win=4500.0,
            avg_loss=3000.0,
            regime="low_vol_trend",
        )
        assert result.contracts == 6

    def test_iron_condor_sizing_from_spec(self, sizer: PositionSizer) -> None:
        """Iron condor: $10 wings - $3 credit = $700 max loss.

        $3000 / $700 = 4.28 => floor = 4 contracts.
        """
        result = sizer.calculate_position_size(
            account_equity=150_000.0,
            max_loss_per_contract=700.0,
            win_probability=0.72,
            avg_win=4500.0,
            avg_loss=3000.0,
            regime="low_vol_trend",
        )
        assert result.contracts == 4


# ===========================================================================
# 4. Regime Adjustment Tests
# ===========================================================================


class TestRegimeAdjustments:
    """Verify regime multipliers are applied correctly."""

    def test_low_vol_trend_full_size(self, sizer: PositionSizer) -> None:
        """low_vol_trend regime allows full position sizing (1.0x)."""
        factor = sizer.apply_regime_adjustment(1.0, "low_vol_trend")
        assert factor == 1.0

    def test_range_bound_75pct(self, sizer: PositionSizer) -> None:
        """range_bound regime reduces to 75%."""
        factor = sizer.apply_regime_adjustment(1.0, "range_bound")
        assert factor == 0.75

    def test_high_vol_trend_50pct(self, sizer: PositionSizer) -> None:
        """high_vol_trend regime reduces to 50%."""
        factor = sizer.apply_regime_adjustment(1.0, "high_vol_trend")
        assert factor == 0.50

    def test_crisis_25pct(self, sizer: PositionSizer) -> None:
        """crisis regime reduces to 25%."""
        factor = sizer.apply_regime_adjustment(1.0, "crisis")
        assert factor == 0.25

    def test_unknown_regime_uses_default(self, sizer: PositionSizer) -> None:
        """Unknown regime string falls back to default factor (0.50)."""
        factor = sizer.apply_regime_adjustment(1.0, "completely_unknown_regime")
        assert factor == _DEFAULT_REGIME_FACTOR
        assert factor == 0.50

    def test_empty_regime_string(self, sizer: PositionSizer) -> None:
        """Empty string regime falls back to default."""
        factor = sizer.apply_regime_adjustment(1.0, "")
        assert factor == _DEFAULT_REGIME_FACTOR

    def test_regime_factors_match_constants(self) -> None:
        """Verify the regime factor constants are correct per spec."""
        assert _REGIME_FACTORS["low_vol_trend"] == 1.00
        assert _REGIME_FACTORS["range_bound"] == 0.75
        assert _REGIME_FACTORS["high_vol_trend"] == 0.50
        assert _REGIME_FACTORS["crisis"] == 0.25
        assert _DEFAULT_REGIME_FACTOR == 0.50

    def test_regime_adjustment_scales_base_value(self, sizer: PositionSizer) -> None:
        """Regime factor multiplies the base value, not just returns the factor."""
        base = 5000.0
        adjusted = sizer.apply_regime_adjustment(base, "crisis")
        assert adjusted == base * 0.25
        assert adjusted == 1250.0

    def test_crisis_regime_reduces_contracts(self, sizer: PositionSizer) -> None:
        """Crisis regime should meaningfully reduce contract count."""
        normal_result = sizer.calculate_position_size(
            account_equity=150_000.0,
            max_loss_per_contract=500.0,
            win_probability=0.72,
            avg_win=4500.0,
            avg_loss=3000.0,
            regime="low_vol_trend",
        )
        crisis_result = sizer.calculate_position_size(
            account_equity=150_000.0,
            max_loss_per_contract=500.0,
            win_probability=0.72,
            avg_win=4500.0,
            avg_loss=3000.0,
            regime="crisis",
        )
        assert crisis_result.contracts < normal_result.contracts
        assert crisis_result.regime_factor == 0.25
        assert normal_result.regime_factor == 1.0

    def test_all_regimes_reduce_or_maintain(self, sizer: PositionSizer) -> None:
        """Every regime factor should be between 0 and 1 (inclusive)."""
        for regime_name, factor in _REGIME_FACTORS.items():
            assert 0.0 <= factor <= 1.0, (
                f"Regime {regime_name} has invalid factor {factor}"
            )


# ===========================================================================
# 5. Circuit Breaker Adjustment Tests
# ===========================================================================


class TestCircuitBreakerAdjustments:
    """Verify circuit breaker and recovery stage multipliers."""

    def test_normal_level_full_trading(self, sizer: PositionSizer) -> None:
        """NORMAL circuit breaker allows full sizing."""
        factor = sizer.apply_circuit_breaker_adjustment(1.0, "NORMAL", recovery_stage=0)
        assert factor == 1.0

    def test_caution_level_half_size(self, sizer: PositionSizer) -> None:
        """CAUTION reduces to 50%."""
        factor = sizer.apply_circuit_breaker_adjustment(
            1.0, "CAUTION", recovery_stage=0
        )
        assert factor == 0.50

    def test_warning_level_quarter_size(self, sizer: PositionSizer) -> None:
        """WARNING reduces to 25%."""
        factor = sizer.apply_circuit_breaker_adjustment(
            1.0, "WARNING", recovery_stage=0
        )
        assert factor == 0.25

    def test_halt_level_zero_trading(self, sizer: PositionSizer) -> None:
        """HALT completely stops new trades (0.0)."""
        factor = sizer.apply_circuit_breaker_adjustment(1.0, "HALT", recovery_stage=0)
        assert factor == 0.0

    def test_emergency_level_zero_trading(self, sizer: PositionSizer) -> None:
        """EMERGENCY completely stops new trades (0.0)."""
        factor = sizer.apply_circuit_breaker_adjustment(
            1.0, "EMERGENCY", recovery_stage=0
        )
        assert factor == 0.0

    def test_halt_overrides_recovery_stage(self, sizer: PositionSizer) -> None:
        """When CB level is HALT, recovery stage does not matter."""
        for stage in range(4):
            factor = sizer.apply_circuit_breaker_adjustment(
                1.0, "HALT", recovery_stage=stage
            )
            assert factor == 0.0, (
                f"HALT should be 0.0 regardless of recovery stage {stage}"
            )

    def test_emergency_overrides_recovery_stage(self, sizer: PositionSizer) -> None:
        """When CB level is EMERGENCY, recovery stage does not matter."""
        for stage in range(4):
            factor = sizer.apply_circuit_breaker_adjustment(
                1.0, "EMERGENCY", recovery_stage=stage
            )
            assert factor == 0.0, f"EMERGENCY should be 0.0 at recovery stage {stage}"

    def test_recovery_stage_1_half_size(self, sizer: PositionSizer) -> None:
        """Recovery stage 1 trades at 50% of the CB-level size."""
        factor = sizer.apply_circuit_breaker_adjustment(1.0, "NORMAL", recovery_stage=1)
        # NORMAL CB = 1.0, recovery stage 1 = 0.50 => combined = 0.50
        assert factor == 0.50

    def test_recovery_stage_2_three_quarter_size(self, sizer: PositionSizer) -> None:
        """Recovery stage 2 trades at 75% of the CB-level size."""
        factor = sizer.apply_circuit_breaker_adjustment(1.0, "NORMAL", recovery_stage=2)
        assert factor == 0.75

    def test_recovery_stage_3_full_size(self, sizer: PositionSizer) -> None:
        """Recovery stage 3 restores full sizing."""
        factor = sizer.apply_circuit_breaker_adjustment(1.0, "NORMAL", recovery_stage=3)
        assert factor == 1.0

    def test_combined_caution_and_recovery(self, sizer: PositionSizer) -> None:
        """CAUTION (0.50) with recovery stage 1 (0.50) = 0.25 combined."""
        factor = sizer.apply_circuit_breaker_adjustment(
            1.0, "CAUTION", recovery_stage=1
        )
        assert factor == 0.50 * 0.50
        assert factor == 0.25

    def test_combined_warning_and_recovery(self, sizer: PositionSizer) -> None:
        """WARNING (0.25) with recovery stage 2 (0.75) = 0.1875."""
        factor = sizer.apply_circuit_breaker_adjustment(
            1.0, "WARNING", recovery_stage=2
        )
        assert abs(factor - 0.25 * 0.75) < 1e-9
        assert abs(factor - 0.1875) < 1e-9

    def test_unknown_cb_level_defaults_to_zero(self, sizer: PositionSizer) -> None:
        """Unknown circuit breaker level defaults to 0.0 (safe default)."""
        factor = sizer.apply_circuit_breaker_adjustment(
            1.0, "UNKNOWN_LEVEL", recovery_stage=0
        )
        assert factor == 0.0

    def test_unknown_recovery_stage_defaults(self, sizer: PositionSizer) -> None:
        """Unknown recovery stage falls back to 0.50 (conservative)."""
        factor = sizer.apply_circuit_breaker_adjustment(
            1.0,
            "NORMAL",
            recovery_stage=99,  # Non-existent stage
        )
        # Default recovery multiplier is 0.50
        assert factor == 0.50

    def test_cb_level_constants_are_correct(self) -> None:
        """Verify the CB level constant dictionary matches spec."""
        assert _CB_LEVEL_MULTIPLIERS["NORMAL"] == 1.0
        assert _CB_LEVEL_MULTIPLIERS["CAUTION"] == 0.50
        assert _CB_LEVEL_MULTIPLIERS["WARNING"] == 0.25
        assert _CB_LEVEL_MULTIPLIERS["HALT"] == 0.0
        assert _CB_LEVEL_MULTIPLIERS["EMERGENCY"] == 0.0

    def test_recovery_stage_constants_are_correct(self) -> None:
        """Verify recovery stage constants match spec."""
        assert _RECOVERY_STAGE_MULTIPLIERS[0] == 1.0
        assert _RECOVERY_STAGE_MULTIPLIERS[1] == 0.50
        assert _RECOVERY_STAGE_MULTIPLIERS[2] == 0.75
        assert _RECOVERY_STAGE_MULTIPLIERS[3] == 1.0

    def test_halt_produces_zero_contracts(self, sizer: PositionSizer) -> None:
        """Full position sizing pipeline with HALT produces zero contracts."""
        result = sizer.calculate_position_size(
            account_equity=150_000.0,
            max_loss_per_contract=500.0,
            win_probability=0.72,
            avg_win=4500.0,
            avg_loss=3000.0,
            regime="low_vol_trend",
            circuit_breaker_level="HALT",
        )
        assert result.contracts == 0
        assert result.dollar_risk == 0.0
        assert result.cb_factor == 0.0


# ===========================================================================
# 6. Full Pipeline Integration Tests
# ===========================================================================


class TestFullPipeline:
    """End-to-end tests of the full position sizing pipeline."""

    def test_standard_trade(self, sizer: PositionSizer) -> None:
        """Standard bull call spread: $150K account, normal conditions."""
        result = sizer.calculate_position_size(
            account_equity=150_000.0,
            max_loss_per_contract=500.0,
            win_probability=0.72,
            avg_win=4500.0,
            avg_loss=3000.0,
            regime="low_vol_trend",
            circuit_breaker_level="NORMAL",
            recovery_stage=0,
        )
        assert result.contracts == 6
        assert result.dollar_risk == 3000.0
        assert result.regime_factor == 1.0
        assert result.cb_factor == 1.0
        assert result.risk_per_contract == 500.0

    def test_crisis_regime_with_normal_cb(self, sizer: PositionSizer) -> None:
        """Crisis regime but normal CB: 25% sizing.

        $3000 * 0.25 (crisis) = $750
        $750 / $500 = 1.5 => floor = 1 contract
        """
        result = sizer.calculate_position_size(
            account_equity=150_000.0,
            max_loss_per_contract=500.0,
            win_probability=0.72,
            avg_win=4500.0,
            avg_loss=3000.0,
            regime="crisis",
            circuit_breaker_level="NORMAL",
            recovery_stage=0,
        )
        assert result.contracts == 1
        assert result.dollar_risk == 500.0
        assert result.regime_factor == 0.25

    def test_normal_regime_with_caution_cb(self, sizer: PositionSizer) -> None:
        """Normal regime but CAUTION CB: 50% sizing.

        $3000 * 1.0 (regime) * 0.5 (CB) = $1500
        $1500 / $500 = 3 contracts
        """
        result = sizer.calculate_position_size(
            account_equity=150_000.0,
            max_loss_per_contract=500.0,
            win_probability=0.72,
            avg_win=4500.0,
            avg_loss=3000.0,
            regime="low_vol_trend",
            circuit_breaker_level="CAUTION",
            recovery_stage=0,
        )
        assert result.contracts == 3
        assert result.dollar_risk == 1500.0

    def test_crisis_plus_caution_stacked(self, sizer: PositionSizer) -> None:
        """Crisis regime (0.25) + CAUTION CB (0.50) = 12.5% of budget.

        $3000 * 0.25 * 0.50 = $375
        $375 / $500 = 0.75 => floor = 0 contracts
        """
        result = sizer.calculate_position_size(
            account_equity=150_000.0,
            max_loss_per_contract=500.0,
            win_probability=0.72,
            avg_win=4500.0,
            avg_loss=3000.0,
            regime="crisis",
            circuit_breaker_level="CAUTION",
            recovery_stage=0,
        )
        assert result.contracts == 0
        assert result.dollar_risk == 0.0

    def test_recovery_stage_reduces_contracts(self, sizer: PositionSizer) -> None:
        """Recovery stage 1 at NORMAL CB: 50% sizing.

        $3000 * 1.0 (regime) * (1.0 CB * 0.50 recovery) = $1500
        $1500 / $500 = 3 contracts
        """
        result = sizer.calculate_position_size(
            account_equity=150_000.0,
            max_loss_per_contract=500.0,
            win_probability=0.72,
            avg_win=4500.0,
            avg_loss=3000.0,
            regime="low_vol_trend",
            circuit_breaker_level="NORMAL",
            recovery_stage=1,
        )
        assert result.contracts == 3
        assert result.dollar_risk == 1500.0

    def test_kelly_driven_sizing_below_cap(self, sizer: PositionSizer) -> None:
        """When Kelly fraction produces risk below the cap, Kelly drives sizing.

        p=0.55, avg_win=300, avg_loss=250 => b=1.2
        full_kelly = (0.55*1.2 - 0.45)/1.2 = (0.66 - 0.45)/1.2 = 0.175
        quarter_kelly = 0.175 * 0.25 = 0.04375
        dollar_risk = 150_000 * 0.04375 = $6562.50
        Capped at $3000 (percentage or dollar cap)
        """
        result = sizer.calculate_position_size(
            account_equity=150_000.0,
            max_loss_per_contract=500.0,
            win_probability=0.55,
            avg_win=300.0,
            avg_loss=250.0,
            regime="low_vol_trend",
        )
        # Kelly = 0.04375, dollar_risk = $6562.50, capped at $3000
        # $3000 / $500 = 6
        assert result.contracts == 6

    def test_small_kelly_uncapped(self, sizer: PositionSizer) -> None:
        """Very small Kelly produces risk below all caps: Kelly drives sizing.

        p=0.52, avg_win=100, avg_loss=100 => b=1.0
        full_kelly = 0.04, quarter_kelly = 0.01
        dollar_risk = 150_000 * 0.01 = $1500 (below $3K cap)
        $1500 / $500 = 3 contracts
        """
        result = sizer.calculate_position_size(
            account_equity=150_000.0,
            max_loss_per_contract=500.0,
            win_probability=0.52,
            avg_win=100.0,
            avg_loss=100.0,
            regime="low_vol_trend",
        )
        assert result.contracts == 3
        assert result.dollar_risk == 1500.0

    def test_large_account_still_capped(self, sizer: PositionSizer) -> None:
        """Even with a large account, absolute dollar cap ($3000) applies."""
        result = sizer.calculate_position_size(
            account_equity=1_000_000.0,
            max_loss_per_contract=500.0,
            win_probability=0.72,
            avg_win=4500.0,
            avg_loss=3000.0,
            regime="low_vol_trend",
        )
        # 2% of $1M = $20K, but capped at $3000 absolute
        assert result.contracts == 6
        assert result.dollar_risk == 3000.0


# ===========================================================================
# 7. PositionSize Pydantic Model Tests
# ===========================================================================


class TestPositionSizeModel:
    """Verify the PositionSize pydantic model constraints."""

    def test_valid_position_size(self) -> None:
        """Valid PositionSize creation."""
        ps = PositionSize(
            contracts=5,
            dollar_risk=2500.0,
            kelly_fraction=0.05,
            regime_factor=1.0,
            cb_factor=1.0,
            risk_per_contract=500.0,
        )
        assert ps.contracts == 5
        assert ps.dollar_risk == 2500.0

    def test_zero_contracts_valid(self) -> None:
        """Zero contracts is a valid result (no trade)."""
        ps = PositionSize(
            contracts=0,
            dollar_risk=0.0,
            kelly_fraction=0.0,
            regime_factor=0.0,
            cb_factor=0.0,
            risk_per_contract=500.0,
        )
        assert ps.contracts == 0

    def test_negative_contracts_rejected(self) -> None:
        """Negative contracts should be rejected by pydantic validation."""
        with pytest.raises(ValidationError):
            PositionSize(
                contracts=-1,
                dollar_risk=0.0,
                kelly_fraction=0.0,
                regime_factor=0.0,
                cb_factor=0.0,
                risk_per_contract=500.0,
            )

    def test_negative_dollar_risk_rejected(self) -> None:
        """Negative dollar_risk should be rejected."""
        with pytest.raises(ValidationError):
            PositionSize(
                contracts=0,
                dollar_risk=-100.0,
                kelly_fraction=0.0,
                regime_factor=0.0,
                cb_factor=0.0,
                risk_per_contract=500.0,
            )

    def test_regime_factor_out_of_range_rejected(self) -> None:
        """Regime factor > 1.0 should be rejected."""
        with pytest.raises(ValidationError):
            PositionSize(
                contracts=0,
                dollar_risk=0.0,
                kelly_fraction=0.0,
                regime_factor=1.5,  # Invalid: le=1.0
                cb_factor=1.0,
                risk_per_contract=500.0,
            )


# ===========================================================================
# 8. Config Loading Tests
# ===========================================================================


class TestConfigLoading:
    """Verify that PositionSizer correctly loads config overrides."""

    def test_custom_recovery_stages_loaded(self, mock_settings: MagicMock) -> None:
        """Custom recovery stages from risk config override defaults."""
        config = {
            "per_trade": {"max_risk_pct": 0.02, "max_risk_dollars": 3000.0},
            "recovery": {
                "stages": [
                    {"stage": 1, "size_pct": 0.40},  # Custom: 40% instead of 50%
                    {"stage": 2, "size_pct": 0.60},  # Custom: 60% instead of 75%
                    {"stage": 3, "size_pct": 0.90},  # Custom: 90% instead of 100%
                ],
            },
        }
        with patch("src.risk.position_sizer.get_logger") as mock_get_logger:
            mock_get_logger.return_value = MagicMock()
            sizer = PositionSizer(settings=mock_settings, risk_config=config)

        # Verify custom values were loaded
        assert sizer._recovery_multipliers[1] == 0.40
        assert sizer._recovery_multipliers[2] == 0.60
        assert sizer._recovery_multipliers[3] == 0.90
        # Stage 0 should still be the default (1.00)
        assert sizer._recovery_multipliers[0] == 1.00

    def test_custom_cb_multipliers_loaded(self, mock_settings: MagicMock) -> None:
        """Custom CB level multipliers from config override defaults."""
        config = {
            "per_trade": {"max_risk_pct": 0.02, "max_risk_dollars": 3000.0},
            "circuit_breakers": {
                "levels": [
                    {"name": "CAUTION", "size_multiplier": 0.60},  # Custom
                    {"name": "WARNING", "size_multiplier": 0.30},  # Custom
                ],
            },
        }
        with patch("src.risk.position_sizer.get_logger") as mock_get_logger:
            mock_get_logger.return_value = MagicMock()
            sizer = PositionSizer(settings=mock_settings, risk_config=config)

        assert sizer._cb_multipliers["CAUTION"] == 0.60
        assert sizer._cb_multipliers["WARNING"] == 0.30
        # Others should be defaults
        assert sizer._cb_multipliers["NORMAL"] == 1.0
        assert sizer._cb_multipliers["HALT"] == 0.0

    def test_empty_risk_config(self, mock_settings: MagicMock) -> None:
        """Empty risk config falls back to defaults from settings."""
        with patch("src.risk.position_sizer.get_logger") as mock_get_logger:
            mock_get_logger.return_value = MagicMock()
            sizer = PositionSizer(settings=mock_settings, risk_config={})

        # Should use settings.trading.per_trade_risk_pct
        assert sizer._max_risk_pct == 0.02
        # Should use default $3000
        assert sizer._max_risk_dollars == 3000.0

    def test_default_max_risk_from_settings(self, mock_settings: MagicMock) -> None:
        """max_risk_pct defaults to settings.trading.per_trade_risk_pct."""
        mock_settings.trading.per_trade_risk_pct = 0.03  # 3%
        config: dict = {"per_trade": {}}  # No override
        with patch("src.risk.position_sizer.get_logger") as mock_get_logger:
            mock_get_logger.return_value = MagicMock()
            sizer = PositionSizer(settings=mock_settings, risk_config=config)

        assert sizer._max_risk_pct == 0.03


# ===========================================================================
# 9. Contracts Never Exceed Budget Tests
# ===========================================================================


class TestMaxContractsCap:
    """Verify that contract count never produces risk exceeding the budget."""

    @pytest.mark.parametrize(
        "max_loss,expected_contracts",
        [
            (100.0, 30),  # $3000 / $100 = 30
            (200.0, 15),  # $3000 / $200 = 15
            (300.0, 10),  # $3000 / $300 = 10
            (500.0, 6),  # $3000 / $500 = 6
            (700.0, 4),  # $3000 / $700 = 4.28 => 4
            (1000.0, 3),  # $3000 / $1000 = 3
            (1500.0, 2),  # $3000 / $1500 = 2
            (2000.0, 1),  # $3000 / $2000 = 1.5 => 1
            (3000.0, 1),  # $3000 / $3000 = 1
            (3500.0, 0),  # $3000 / $3500 = 0.857 => 0 (rejected)
            (5000.0, 0),  # $3000 / $5000 = 0.6 => 0 (rejected)
            (10000.0, 0),  # Far exceeds budget
        ],
    )
    def test_contracts_scale_with_max_loss(
        self, sizer: PositionSizer, max_loss: float, expected_contracts: int
    ) -> None:
        """Parametrized: verify contract count for various max_loss values."""
        result = sizer.calculate_position_size(
            account_equity=150_000.0,
            max_loss_per_contract=max_loss,
            win_probability=0.72,
            avg_win=4500.0,
            avg_loss=3000.0,
            regime="low_vol_trend",
            circuit_breaker_level="NORMAL",
            recovery_stage=0,
        )
        assert result.contracts == expected_contracts, (
            f"max_loss={max_loss}: expected {expected_contracts} contracts, "
            f"got {result.contracts}"
        )

    def test_actual_risk_never_exceeds_budget(self, sizer: PositionSizer) -> None:
        """For any contract count, actual dollar risk must not exceed budget."""
        for max_loss in [100, 200, 300, 500, 700, 1000, 1500, 2000, 3000]:
            result = sizer.calculate_position_size(
                account_equity=150_000.0,
                max_loss_per_contract=float(max_loss),
                win_probability=0.72,
                avg_win=4500.0,
                avg_loss=3000.0,
                regime="low_vol_trend",
                circuit_breaker_level="NORMAL",
                recovery_stage=0,
            )
            assert result.dollar_risk <= 3000.0, (
                f"max_loss={max_loss}: dollar_risk={result.dollar_risk} exceeds $3000"
            )

    @pytest.mark.parametrize("regime", list(_REGIME_FACTORS.keys()))
    def test_risk_capped_across_all_regimes(
        self, sizer: PositionSizer, regime: str
    ) -> None:
        """Dollar risk never exceeds $3000 in any regime."""
        result = sizer.calculate_position_size(
            account_equity=150_000.0,
            max_loss_per_contract=500.0,
            win_probability=0.72,
            avg_win=4500.0,
            avg_loss=3000.0,
            regime=regime,
        )
        assert result.dollar_risk <= 3000.0

    @pytest.mark.parametrize("cb_level", list(_CB_LEVEL_MULTIPLIERS.keys()))
    def test_risk_capped_across_all_cb_levels(
        self, sizer: PositionSizer, cb_level: str
    ) -> None:
        """Dollar risk never exceeds $3000 at any CB level."""
        result = sizer.calculate_position_size(
            account_equity=150_000.0,
            max_loss_per_contract=500.0,
            win_probability=0.72,
            avg_win=4500.0,
            avg_loss=3000.0,
            regime="low_vol_trend",
            circuit_breaker_level=cb_level,
        )
        assert result.dollar_risk <= 3000.0


# ===========================================================================
# 10. Return Value Structure Tests
# ===========================================================================


class TestReturnValueStructure:
    """Verify the PositionSize return object has correct values."""

    def test_kelly_fraction_in_result(self, sizer: PositionSizer) -> None:
        """The result contains the calculated Kelly fraction."""
        result = sizer.calculate_position_size(
            account_equity=150_000.0,
            max_loss_per_contract=500.0,
            win_probability=0.65,
            avg_win=600.0,
            avg_loss=400.0,
            regime="low_vol_trend",
        )
        assert result.kelly_fraction > 0.0
        assert result.kelly_fraction <= _MAX_KELLY_FRACTION

    def test_risk_per_contract_preserved(self, sizer: PositionSizer) -> None:
        """risk_per_contract in result matches the input max_loss_per_contract."""
        result = sizer.calculate_position_size(
            account_equity=150_000.0,
            max_loss_per_contract=750.0,
            win_probability=0.72,
            avg_win=4500.0,
            avg_loss=3000.0,
            regime="low_vol_trend",
        )
        assert result.risk_per_contract == 750.0

    def test_zero_position_preserves_risk_per_contract(
        self, sizer: PositionSizer
    ) -> None:
        """Even with zero contracts, risk_per_contract is preserved."""
        result = sizer.calculate_position_size(
            account_equity=150_000.0,
            max_loss_per_contract=5000.0,  # Too wide
            win_probability=0.72,
            avg_win=4500.0,
            avg_loss=3000.0,
            regime="low_vol_trend",
        )
        assert result.contracts == 0
        assert result.risk_per_contract == 5000.0

    def test_result_is_position_size_instance(self, sizer: PositionSizer) -> None:
        """calculate_position_size returns a PositionSize pydantic model."""
        result = sizer.calculate_position_size(
            account_equity=150_000.0,
            max_loss_per_contract=500.0,
            win_probability=0.65,
            avg_win=500.0,
            avg_loss=300.0,
            regime="low_vol_trend",
        )
        assert isinstance(result, PositionSize)
