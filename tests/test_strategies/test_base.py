"""Unit tests for src/strategies/base.py — abstract base strategy and Pydantic models.

Tests cover:
  - Pydantic model construction and validation (StrategyConfig, LegSpec, etc.)
  - Strategy instantiation via a concrete subclass
  - is_eligible() regime and IV rank gating
  - check_mechanical_exit() profit target, stop loss, and DTE limit rules
  - calculate_reward_risk_ratio() edge cases
  - filter_options_by_dte() date filtering logic
  - find_strike_by_delta() delta-proximity strike selection
  - find_option_by_delta() range-based search on OptionData objects
  - validate_bid_ask_spread() liquidity check
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any
from uuid import uuid4

import pytest
from pydantic import ValidationError

from src.strategies.base import (
    DEFAULT_CLOSE_BEFORE_EXPIRY_DTE,
    BaseStrategy,
    Direction,
    ExitReason,
    ExitSignal,
    ExitType,
    GreeksSnapshot,
    LegAction,
    LegSpec,
    OptionData,
    OptionRight,
    StrategyConfig,
    TradeRecord,
    TradeSignal,
)

# ---------------------------------------------------------------------------
# Concrete subclass for testing (BaseStrategy is abstract)
# ---------------------------------------------------------------------------


class _StubStrategy(BaseStrategy):
    """Minimal concrete subclass that satisfies all abstract methods."""

    async def check_entry(
        self,
        ticker: str,
        spot_price: float,
        iv_rank: float,
        regime: str,
        greeks: dict[str, float],
        options_chain: list[dict[str, Any]],
    ) -> TradeSignal | None:
        return None

    async def check_exit(
        self,
        trade: dict[str, Any],
        spot_price: float,
        current_pnl: float,
        current_pnl_pct: float,
        dte_remaining: int,
        greeks: dict[str, float],
    ) -> ExitSignal | None:
        return None

    def construct_legs(
        self,
        spot_price: float,
        options_chain: list[dict[str, Any]],
        **kwargs: Any,
    ) -> list[LegSpec]:
        return []

    def calculate_max_profit(
        self,
        legs: list[LegSpec],
        net_premium: float,
    ) -> float:
        return 0.0

    def calculate_max_loss(
        self,
        legs: list[LegSpec],
        net_premium: float,
    ) -> float:
        return 0.0

    async def construct_order(
        self,
        signal: TradeSignal,
        contract_factory: Any,
    ) -> Any:
        return None

    def calculate_greeks(
        self,
        legs: list[LegSpec],
        greeks: dict[str, float],
    ) -> dict[str, float]:
        return {"delta": 0.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def default_config() -> StrategyConfig:
    """Return a StrategyConfig with sensible defaults for testing."""
    return StrategyConfig(
        enabled=True,
        description="Test strategy",
        regimes=["low_vol_trend", "range_bound"],
        min_iv_rank=20.0,
        max_iv_rank=60.0,
        target_dte=45,
        dte_range=[30, 60],
        profit_target_pct=0.50,
        stop_loss_pct=1.00,
        max_positions=3,
    )


@pytest.fixture()
def strategy(default_config: StrategyConfig) -> _StubStrategy:
    """Return a concrete strategy instance backed by default_config."""
    return _StubStrategy(name="test_strategy", config=default_config)


@pytest.fixture()
def _make_option_dict():
    """Factory for building option chain dicts used by filter/find methods."""

    def _build(
        strike: float = 200.0,
        right: str = "C",
        expiry: str | date | None = None,
        delta: float = 0.50,
        open_interest: int = 1000,
        bid: float = 3.00,
        ask: float = 3.20,
    ) -> dict[str, Any]:
        if expiry is None:
            expiry = (date.today() + timedelta(days=40)).strftime("%Y%m%d")
        return {
            "strike": strike,
            "right": right,
            "expiry": expiry,
            "delta": delta,
            "gamma": 0.03,
            "theta": -0.05,
            "vega": 0.12,
            "implied_vol": 0.30,
            "bid": bid,
            "ask": ask,
            "mid_price": (bid + ask) / 2.0,
            "open_interest": open_interest,
            "volume": 500,
        }

    return _build


# ===========================================================================
# Enumeration tests
# ===========================================================================


class TestEnumerations:
    """StrEnum members should behave as strings."""

    def test_direction_values(self):
        assert Direction.LONG == "LONG"
        assert Direction.SHORT == "SHORT"

    def test_option_right_values(self):
        assert OptionRight.CALL == "C"
        assert OptionRight.PUT == "P"

    def test_leg_action_values(self):
        assert LegAction.BUY == "BUY"
        assert LegAction.SELL == "SELL"

    def test_exit_type_values(self):
        assert ExitType.PROFIT_TARGET == "PROFIT_TARGET"
        assert ExitType.STOP_LOSS == "STOP_LOSS"
        assert ExitType.TIME_DECAY == "TIME_DECAY"
        assert ExitType.MECHANICAL == "MECHANICAL"
        assert ExitType.MANUAL == "MANUAL"

    def test_exit_reason_values(self):
        assert ExitReason.DTE_LIMIT == "DTE_LIMIT"
        assert ExitReason.CIRCUIT_BREAKER == "CIRCUIT_BREAKER"
        assert ExitReason.EVENT_RISK == "EVENT_RISK"


# ===========================================================================
# StrategyConfig tests
# ===========================================================================


class TestStrategyConfig:
    """StrategyConfig Pydantic model validation and helpers."""

    def test_default_construction(self):
        """Defaults should yield a valid config."""
        cfg = StrategyConfig()
        assert cfg.enabled is True
        assert cfg.min_iv_rank == 0.0
        assert cfg.max_iv_rank == 100.0
        assert cfg.profit_target_pct == 0.50
        assert cfg.stop_loss_pct == 1.00
        assert cfg.max_positions == 3

    def test_regimes_coerced_from_string(self):
        """A single string regime should be wrapped in a list."""
        cfg = StrategyConfig(regimes="low_vol_trend")
        assert cfg.regimes == ["low_vol_trend"]

    def test_regimes_accepts_list(self):
        cfg = StrategyConfig(regimes=["range_bound", "high_vol_trend"])
        assert len(cfg.regimes) == 2

    def test_profit_target_must_be_positive(self):
        with pytest.raises(ValueError, match="Must be positive"):
            StrategyConfig(profit_target_pct=0.0)

    def test_stop_loss_must_be_positive(self):
        with pytest.raises(ValueError, match="Must be positive"):
            StrategyConfig(stop_loss_pct=-0.5)

    def test_max_positions_at_least_one(self):
        with pytest.raises(ValueError, match="max_positions must be >= 1"):
            StrategyConfig(max_positions=0)

    def test_get_primary_dte_range_simple(self):
        cfg = StrategyConfig(dte_range=[30, 60])
        assert cfg.get_primary_dte_range() == (30, 60)

    def test_get_primary_dte_range_dict_front_month(self):
        cfg = StrategyConfig(
            dte_range={"front_month": [25, 45], "back_month": [55, 90]},
        )
        assert cfg.get_primary_dte_range() == (25, 45)

    def test_get_primary_dte_range_dict_fallback(self):
        """When no front_month key exists, use the first key."""
        cfg = StrategyConfig(dte_range={"short_leg": [20, 35]})
        assert cfg.get_primary_dte_range() == (20, 35)

    def test_get_primary_target_dte_simple(self):
        cfg = StrategyConfig(target_dte=45)
        assert cfg.get_primary_target_dte() == 45

    def test_get_primary_target_dte_dict(self):
        cfg = StrategyConfig(target_dte={"front_month": 30, "back_month": 60})
        assert cfg.get_primary_target_dte() == 30

    def test_get_primary_target_dte_dict_fallback(self):
        cfg = StrategyConfig(target_dte={"only_key": 42})
        assert cfg.get_primary_target_dte() == 42


# ===========================================================================
# LegSpec tests
# ===========================================================================


class TestLegSpec:
    """LegSpec Pydantic model validation."""

    def test_valid_construction(self):
        leg = LegSpec(
            action="BUY",
            right="C",
            strike=200.0,
            expiry="20260420",
        )
        assert leg.action == "BUY"
        assert leg.strike == 200.0
        assert leg.quantity == 1
        assert leg.delta is None

    def test_expiry_non_numeric_rejected(self):
        with pytest.raises(ValueError):
            LegSpec(action="BUY", right="C", strike=200.0, expiry="2026-0ab")

    def test_expiry_bad_year_rejected(self):
        with pytest.raises(ValueError, match="year out of range"):
            LegSpec(action="BUY", right="C", strike=200.0, expiry="19990115")

    def test_expiry_bad_month_rejected(self):
        with pytest.raises(ValueError, match="month out of range"):
            LegSpec(action="BUY", right="C", strike=200.0, expiry="20261315")

    def test_expiry_bad_day_rejected(self):
        with pytest.raises(ValueError, match="day out of range"):
            LegSpec(action="BUY", right="C", strike=200.0, expiry="20260400")

    def test_frozen_model(self):
        """LegSpec should be immutable."""
        leg = LegSpec(action="BUY", right="C", strike=200.0, expiry="20260420")
        with pytest.raises(ValidationError):
            leg.strike = 210.0


# ===========================================================================
# TradeSignal tests
# ===========================================================================


class TestTradeSignal:
    """TradeSignal Pydantic model validation."""

    def _make_leg(self) -> LegSpec:
        return LegSpec(action="BUY", right="C", strike=200.0, expiry="20260420")

    def test_valid_construction(self):
        sig = TradeSignal(
            ticker="AAPL",
            strategy_name="bull_call_spread",
            direction="LONG",
            confidence=0.85,
            legs=[self._make_leg()],
            max_profit=500.0,
            max_loss=500.0,
        )
        assert sig.ticker == "AAPL"
        assert sig.confidence == 0.85
        assert sig.reward_risk_ratio == 0.0  # default

    def test_confidence_clamped(self):
        """Confidence must be in [0.0, 1.0]."""
        with pytest.raises(ValueError):
            TradeSignal(
                ticker="AAPL",
                strategy_name="test",
                direction="LONG",
                confidence=1.5,
                legs=[self._make_leg()],
                max_profit=100.0,
                max_loss=100.0,
            )

    def test_max_loss_must_be_positive(self):
        with pytest.raises(ValueError):
            TradeSignal(
                ticker="AAPL",
                strategy_name="test",
                direction="LONG",
                legs=[self._make_leg()],
                max_profit=100.0,
                max_loss=0.0,
            )

    def test_legs_cannot_be_empty(self):
        with pytest.raises(ValueError):
            TradeSignal(
                ticker="AAPL",
                strategy_name="test",
                direction="LONG",
                legs=[],
                max_profit=100.0,
                max_loss=100.0,
            )


# ===========================================================================
# OptionData tests
# ===========================================================================


class TestOptionData:
    """OptionData Pydantic model."""

    def test_frozen_model(self):
        opt = OptionData(
            strike=200.0,
            right="C",
            expiry=date(2026, 4, 20),
            delta=0.50,
            gamma=0.03,
            theta=-0.05,
            vega=0.12,
            implied_vol=0.30,
            bid=3.00,
            ask=3.20,
            mid_price=3.10,
            open_interest=1000,
        )
        with pytest.raises(ValidationError):
            opt.strike = 210.0


# ===========================================================================
# GreeksSnapshot tests
# ===========================================================================


class TestGreeksSnapshot:
    def test_defaults_to_zeros(self):
        gs = GreeksSnapshot()
        assert gs.delta == 0.0
        assert gs.gamma == 0.0
        assert gs.theta == 0.0
        assert gs.vega == 0.0
        assert gs.iv == 0.0


# ===========================================================================
# TradeRecord tests
# ===========================================================================


class TestTradeRecord:
    def test_construction(self):
        from uuid import UUID

        # TradeRecord uses UUID under TYPE_CHECKING; rebuild the model
        # with UUID available in the namespace so Pydantic can resolve it.
        TradeRecord.model_rebuild(_types_namespace={"UUID": UUID})
        tid = uuid4()
        rec = TradeRecord(
            id=tid,
            ticker="AAPL",
            strategy="bull_call_spread",
            direction="LONG",
            entry_price=5.00,
            max_profit=500.0,
            max_loss=500.0,
        )
        assert rec.id == tid
        assert rec.legs == []
        assert rec.entry_time is None


# ===========================================================================
# BaseStrategy instantiation tests
# ===========================================================================


class TestStrategyInstantiation:
    """Create a concrete subclass and verify attributes."""

    def test_name_property(self, strategy: _StubStrategy):
        assert strategy.name == "test_strategy"

    def test_config_property(
        self, strategy: _StubStrategy, default_config: StrategyConfig
    ):
        assert strategy.config is default_config

    def test_repr(self, strategy: _StubStrategy):
        r = repr(strategy)
        assert "_StubStrategy" in r
        assert "test_strategy" in r
        assert "low_vol_trend" in r

    def test_cannot_instantiate_abstract_directly(self):
        with pytest.raises(TypeError):
            BaseStrategy(name="bad", config=StrategyConfig())


# ===========================================================================
# is_eligible() tests
# ===========================================================================


class TestIsEligible:
    """Test regime and IV rank gating in is_eligible()."""

    def test_eligible_when_all_conditions_met(self, strategy: _StubStrategy):
        assert strategy.is_eligible(regime="low_vol_trend", iv_rank=40.0) is True

    def test_eligible_at_iv_rank_boundaries(self, strategy: _StubStrategy):
        """IV rank exactly at min and max should be eligible."""
        assert strategy.is_eligible(regime="low_vol_trend", iv_rank=20.0) is True
        assert strategy.is_eligible(regime="low_vol_trend", iv_rank=60.0) is True

    def test_not_eligible_disabled(self, default_config: StrategyConfig):
        default_config.enabled = False
        s = _StubStrategy(name="disabled", config=default_config)
        assert s.is_eligible(regime="low_vol_trend", iv_rank=40.0) is False

    def test_not_eligible_wrong_regime(self, strategy: _StubStrategy):
        assert strategy.is_eligible(regime="crisis", iv_rank=40.0) is False

    def test_not_eligible_iv_rank_too_low(self, strategy: _StubStrategy):
        assert strategy.is_eligible(regime="low_vol_trend", iv_rank=10.0) is False

    def test_not_eligible_iv_rank_too_high(self, strategy: _StubStrategy):
        assert strategy.is_eligible(regime="low_vol_trend", iv_rank=70.0) is False

    def test_second_regime_also_eligible(self, strategy: _StubStrategy):
        """Config has two regimes; verify the second one works."""
        assert strategy.is_eligible(regime="range_bound", iv_rank=40.0) is True


# ===========================================================================
# check_mechanical_exit() tests
# ===========================================================================


class TestCheckMechanicalExit:
    """Test profit target, stop loss, and DTE-based exit rules."""

    # --- Stop loss (highest priority) ---

    def test_stop_loss_triggers(self, strategy: _StubStrategy):
        """P&L at -100% of max loss should trigger stop loss."""
        result = strategy.check_mechanical_exit(current_pnl_pct=-1.0, dte_remaining=30)
        assert result is not None
        assert result.exit_type == ExitType.STOP_LOSS
        assert "Stop loss triggered" in result.reasoning

    def test_stop_loss_triggers_at_boundary(self, strategy: _StubStrategy):
        """P&L exactly at -stop_loss_pct (default -1.0) triggers."""
        result = strategy.check_mechanical_exit(current_pnl_pct=-1.0, dte_remaining=30)
        assert result is not None
        assert result.exit_type == ExitType.STOP_LOSS

    def test_stop_loss_triggers_beyond(self, strategy: _StubStrategy):
        """P&L worse than stop_loss_pct should still trigger."""
        result = strategy.check_mechanical_exit(current_pnl_pct=-1.5, dte_remaining=30)
        assert result is not None
        assert result.exit_type == ExitType.STOP_LOSS

    def test_stop_loss_does_not_trigger_above_threshold(self, strategy: _StubStrategy):
        result = strategy.check_mechanical_exit(current_pnl_pct=-0.5, dte_remaining=30)
        # At -0.5 with stop_loss_pct=1.0, should NOT trigger stop loss
        assert result is None

    # --- Profit target (second priority) ---

    def test_profit_target_triggers(self, strategy: _StubStrategy):
        """50% profit captured (default profit_target_pct=0.50) triggers."""
        result = strategy.check_mechanical_exit(current_pnl_pct=0.50, dte_remaining=30)
        assert result is not None
        assert result.exit_type == ExitType.PROFIT_TARGET
        assert "Profit target reached" in result.reasoning

    def test_profit_target_triggers_above(self, strategy: _StubStrategy):
        result = strategy.check_mechanical_exit(current_pnl_pct=0.80, dte_remaining=30)
        assert result is not None
        assert result.exit_type == ExitType.PROFIT_TARGET

    def test_profit_target_below_threshold(self, strategy: _StubStrategy):
        result = strategy.check_mechanical_exit(current_pnl_pct=0.40, dte_remaining=30)
        assert result is None

    # --- Time decay / DTE limit (third priority) ---

    def test_dte_exit_triggers(self, strategy: _StubStrategy):
        """Positions at or below DEFAULT_CLOSE_BEFORE_EXPIRY_DTE should exit."""
        result = strategy.check_mechanical_exit(
            current_pnl_pct=0.0,
            dte_remaining=DEFAULT_CLOSE_BEFORE_EXPIRY_DTE,
        )
        assert result is not None
        assert result.exit_type == ExitType.TIME_DECAY
        assert "Time decay exit" in result.reasoning

    def test_dte_exit_below_threshold(self, strategy: _StubStrategy):
        result = strategy.check_mechanical_exit(current_pnl_pct=0.0, dte_remaining=2)
        assert result is not None
        assert result.exit_type == ExitType.TIME_DECAY

    def test_dte_exit_above_threshold(self, strategy: _StubStrategy):
        result = strategy.check_mechanical_exit(
            current_pnl_pct=0.0,
            dte_remaining=DEFAULT_CLOSE_BEFORE_EXPIRY_DTE + 1,
        )
        assert result is None

    # --- No exit ---

    def test_no_exit_when_all_normal(self, strategy: _StubStrategy):
        result = strategy.check_mechanical_exit(current_pnl_pct=0.20, dte_remaining=25)
        assert result is None

    # --- Priority ordering ---

    def test_stop_loss_takes_priority_over_dte(self, strategy: _StubStrategy):
        """When both stop loss and DTE are triggered, stop loss wins."""
        result = strategy.check_mechanical_exit(current_pnl_pct=-1.0, dte_remaining=3)
        assert result is not None
        assert result.exit_type == ExitType.STOP_LOSS

    def test_profit_target_takes_priority_over_dte(self, strategy: _StubStrategy):
        """When both profit target and DTE are triggered, profit target wins."""
        result = strategy.check_mechanical_exit(current_pnl_pct=0.60, dte_remaining=3)
        assert result is not None
        assert result.exit_type == ExitType.PROFIT_TARGET

    # --- Custom thresholds ---

    def test_custom_profit_target(self):
        cfg = StrategyConfig(
            regimes=["low_vol_trend"],
            profit_target_pct=0.75,
            stop_loss_pct=2.0,
        )
        s = _StubStrategy(name="custom", config=cfg)
        # 60% profit should NOT trigger at 75% target
        assert s.check_mechanical_exit(current_pnl_pct=0.60, dte_remaining=30) is None
        # 75% should trigger
        result = s.check_mechanical_exit(current_pnl_pct=0.75, dte_remaining=30)
        assert result is not None
        assert result.exit_type == ExitType.PROFIT_TARGET

    def test_custom_stop_loss(self):
        cfg = StrategyConfig(
            regimes=["low_vol_trend"],
            profit_target_pct=0.50,
            stop_loss_pct=0.50,
        )
        s = _StubStrategy(name="tight_stop", config=cfg)
        # -0.5 exactly should trigger with stop_loss_pct=0.50
        result = s.check_mechanical_exit(current_pnl_pct=-0.50, dte_remaining=30)
        assert result is not None
        assert result.exit_type == ExitType.STOP_LOSS

    # --- ExitSignal placeholder fields ---

    def test_exit_signal_has_placeholder_trade_id(self, strategy: _StubStrategy):
        result = strategy.check_mechanical_exit(current_pnl_pct=-1.0, dte_remaining=30)
        assert result is not None
        assert result.trade_id == ""
        assert result.current_pnl == 0.0


# ===========================================================================
# calculate_reward_risk_ratio() tests
# ===========================================================================


class TestCalculateRewardRiskRatio:
    """Test the reward-to-risk ratio helper."""

    def test_normal_ratio(self, strategy: _StubStrategy):
        assert strategy.calculate_reward_risk_ratio(500.0, 500.0) == 1.0

    def test_high_reward(self, strategy: _StubStrategy):
        assert strategy.calculate_reward_risk_ratio(1000.0, 500.0) == 2.0

    def test_low_reward(self, strategy: _StubStrategy):
        assert strategy.calculate_reward_risk_ratio(250.0, 500.0) == 0.5

    def test_zero_max_loss_returns_zero(self, strategy: _StubStrategy):
        assert strategy.calculate_reward_risk_ratio(500.0, 0.0) == 0.0

    def test_negative_max_loss_returns_zero(self, strategy: _StubStrategy):
        assert strategy.calculate_reward_risk_ratio(500.0, -100.0) == 0.0

    def test_rounded_to_four_decimals(self, strategy: _StubStrategy):
        ratio = strategy.calculate_reward_risk_ratio(1.0, 3.0)
        assert ratio == round(1.0 / 3.0, 4)


# ===========================================================================
# filter_options_by_dte() tests
# ===========================================================================


class TestFilterOptionsByDte:
    """Test DTE-based option chain filtering."""

    def test_filters_within_range(self, strategy: _StubStrategy, _make_option_dict):
        today = date.today()
        chain = [
            _make_option_dict(expiry=(today + timedelta(days=10)).strftime("%Y%m%d")),
            _make_option_dict(expiry=(today + timedelta(days=35)).strftime("%Y%m%d")),
            _make_option_dict(expiry=(today + timedelta(days=60)).strftime("%Y%m%d")),
            _make_option_dict(expiry=(today + timedelta(days=90)).strftime("%Y%m%d")),
        ]
        result = strategy.filter_options_by_dte(chain, min_dte=30, max_dte=60)
        assert len(result) == 2
        # Verify DTE was added to each result
        for opt in result:
            assert "dte" in opt
            assert 30 <= opt["dte"] <= 60

    def test_empty_chain(self, strategy: _StubStrategy):
        result = strategy.filter_options_by_dte([], min_dte=30, max_dte=60)
        assert result == []

    def test_no_matches(self, strategy: _StubStrategy, _make_option_dict):
        today = date.today()
        chain = [
            _make_option_dict(expiry=(today + timedelta(days=5)).strftime("%Y%m%d")),
        ]
        result = strategy.filter_options_by_dte(chain, min_dte=30, max_dte=60)
        assert len(result) == 0

    def test_handles_date_objects(self, strategy: _StubStrategy, _make_option_dict):
        """The method should accept date objects in the expiry field."""
        today = date.today()
        chain = [
            _make_option_dict(expiry=today + timedelta(days=40)),
        ]
        result = strategy.filter_options_by_dte(chain, min_dte=30, max_dte=60)
        assert len(result) == 1

    def test_boundary_inclusive(self, strategy: _StubStrategy, _make_option_dict):
        today = date.today()
        chain = [
            _make_option_dict(expiry=(today + timedelta(days=30)).strftime("%Y%m%d")),
            _make_option_dict(expiry=(today + timedelta(days=60)).strftime("%Y%m%d")),
        ]
        result = strategy.filter_options_by_dte(chain, min_dte=30, max_dte=60)
        assert len(result) == 2

    def test_invalid_expiry_string_skipped(self, strategy: _StubStrategy):
        chain = [
            {"strike": 200, "expiry": "badvalue", "right": "C"},
            {"strike": 200, "expiry": "", "right": "C"},
            {"strike": 200, "right": "C"},  # missing expiry
        ]
        result = strategy.filter_options_by_dte(chain, min_dte=0, max_dte=365)
        assert len(result) == 0


# ===========================================================================
# find_strike_by_delta() tests
# ===========================================================================


class TestFindStrikeByDelta:
    """Test delta-proximity strike selection."""

    def test_finds_closest_delta(self, strategy: _StubStrategy, _make_option_dict):
        chain = [
            _make_option_dict(strike=195, delta=0.60, right="C"),
            _make_option_dict(strike=200, delta=0.50, right="C"),
            _make_option_dict(strike=205, delta=0.40, right="C"),
            _make_option_dict(strike=210, delta=0.30, right="C"),
        ]
        result = strategy.find_strike_by_delta(chain, target_delta=0.50, right="C")
        assert result is not None
        assert result["strike"] == 200

    def test_filters_by_right(self, strategy: _StubStrategy, _make_option_dict):
        chain = [
            _make_option_dict(strike=200, delta=0.50, right="C"),
            _make_option_dict(strike=200, delta=-0.50, right="P"),
        ]
        result = strategy.find_strike_by_delta(chain, target_delta=0.50, right="P")
        assert result is not None
        assert result["right"] == "P"

    def test_none_when_no_match_within_tolerance(
        self, strategy: _StubStrategy, _make_option_dict
    ):
        chain = [
            _make_option_dict(strike=200, delta=0.80, right="C"),
        ]
        result = strategy.find_strike_by_delta(
            chain,
            target_delta=0.30,
            right="C",
            tolerance=0.05,
        )
        assert result is None

    def test_uses_absolute_delta(self, strategy: _StubStrategy, _make_option_dict):
        """Put deltas are negative; search should compare absolute values."""
        chain = [
            _make_option_dict(strike=190, delta=-0.30, right="P"),
            _make_option_dict(strike=185, delta=-0.20, right="P"),
        ]
        result = strategy.find_strike_by_delta(
            chain,
            target_delta=-0.30,
            right="P",
            tolerance=0.05,
        )
        assert result is not None
        assert result["strike"] == 190

    def test_skips_nan_delta(self, strategy: _StubStrategy, _make_option_dict):
        chain = [
            _make_option_dict(strike=200, delta=float("nan"), right="C"),
            _make_option_dict(strike=205, delta=0.45, right="C"),
        ]
        result = strategy.find_strike_by_delta(chain, target_delta=0.45, right="C")
        assert result is not None
        assert result["strike"] == 205

    def test_skips_none_delta(self, strategy: _StubStrategy):
        chain = [
            {"strike": 200, "right": "C", "delta": None},
        ]
        result = strategy.find_strike_by_delta(chain, target_delta=0.50, right="C")
        assert result is None

    def test_empty_chain(self, strategy: _StubStrategy):
        result = strategy.find_strike_by_delta([], target_delta=0.50, right="C")
        assert result is None

    def test_custom_tolerance(self, strategy: _StubStrategy, _make_option_dict):
        chain = [
            _make_option_dict(strike=200, delta=0.53, right="C"),
        ]
        # Default tolerance=0.05 should match (distance=0.03 <= 0.05)
        assert (
            strategy.find_strike_by_delta(chain, target_delta=0.50, right="C")
            is not None
        )
        # Tighter tolerance should not (distance=0.03 > 0.01)
        assert (
            strategy.find_strike_by_delta(
                chain,
                target_delta=0.50,
                right="C",
                tolerance=0.01,
            )
            is None
        )


# ===========================================================================
# find_option_by_delta() — static method on OptionData objects
# ===========================================================================


class TestFindOptionByDelta:
    """Test range-based delta search on OptionData objects."""

    def _make_opt(
        self,
        strike: float,
        right: str,
        delta: float,
        oi: int = 1000,
    ) -> OptionData:
        return OptionData(
            strike=strike,
            right=right,
            expiry=date(2026, 4, 20),
            delta=delta,
            gamma=0.03,
            theta=-0.05,
            vega=0.12,
            implied_vol=0.30,
            bid=3.00,
            ask=3.20,
            mid_price=3.10,
            open_interest=oi,
        )

    def test_finds_best_within_range(self):
        options = [
            self._make_opt(195, "C", 0.60),
            self._make_opt(200, "C", 0.50),
            self._make_opt(205, "C", 0.40),
        ]
        result = BaseStrategy.find_option_by_delta(
            options,
            right="C",
            target_delta_min=0.45,
            target_delta_max=0.55,
        )
        assert result is not None
        assert result.strike == 200

    def test_returns_closest_to_midpoint(self):
        """When multiple options qualify, pick the one closest to midpoint."""
        options = [
            self._make_opt(200, "C", 0.48),
            self._make_opt(202, "C", 0.52),
        ]
        # Midpoint is 0.50; both are within range
        result = BaseStrategy.find_option_by_delta(
            options,
            right="C",
            target_delta_min=0.45,
            target_delta_max=0.55,
        )
        assert result is not None
        # 0.48 is 0.02 from midpoint, 0.52 is 0.02 from midpoint
        # Either is acceptable; check one is returned
        assert result.strike in (200, 202)

    def test_filters_by_right(self):
        options = [
            self._make_opt(200, "C", 0.50),
            self._make_opt(200, "P", -0.50),
        ]
        result = BaseStrategy.find_option_by_delta(
            options,
            right="P",
            target_delta_min=-0.55,
            target_delta_max=-0.45,
        )
        assert result is not None
        assert result.right == "P"

    def test_filters_by_open_interest(self):
        """Options with OI below DEFAULT_MIN_OPEN_INTEREST should be skipped."""
        options = [
            self._make_opt(200, "C", 0.50, oi=100),  # Below minimum
        ]
        result = BaseStrategy.find_option_by_delta(
            options,
            right="C",
            target_delta_min=0.45,
            target_delta_max=0.55,
        )
        assert result is None

    def test_none_when_no_match(self):
        options = [
            self._make_opt(200, "C", 0.80),
        ]
        result = BaseStrategy.find_option_by_delta(
            options,
            right="C",
            target_delta_min=0.45,
            target_delta_max=0.55,
        )
        assert result is None

    def test_empty_list(self):
        result = BaseStrategy.find_option_by_delta(
            [],
            right="C",
            target_delta_min=0.45,
            target_delta_max=0.55,
        )
        assert result is None


# ===========================================================================
# validate_bid_ask_spread() tests
# ===========================================================================


class TestValidateBidAskSpread:
    """Test the liquidity quality check on OptionData."""

    def _make_opt(self, bid: float, ask: float) -> OptionData:
        return OptionData(
            strike=200.0,
            right="C",
            expiry=date(2026, 4, 20),
            delta=0.50,
            gamma=0.03,
            theta=-0.05,
            vega=0.12,
            implied_vol=0.30,
            bid=bid,
            ask=ask,
            mid_price=(bid + ask) / 2.0,
            open_interest=1000,
        )

    def test_tight_spread_passes(self):
        opt = self._make_opt(bid=3.00, ask=3.10)
        # Spread = 0.10, mid = 3.05, pct = 3.28%
        assert BaseStrategy.validate_bid_ask_spread(opt) is True

    def test_wide_spread_fails(self):
        opt = self._make_opt(bid=3.00, ask=4.00)
        # Spread = 1.00, mid = 3.50, pct = 28.6%
        assert BaseStrategy.validate_bid_ask_spread(opt) is False

    def test_zero_mid_price_fails(self):
        opt = self._make_opt(bid=0.0, ask=0.0)
        # mid_price = 0 should return False
        assert BaseStrategy.validate_bid_ask_spread(opt) is False

    def test_spread_exactly_at_threshold(self):
        # DEFAULT_MAX_BID_ASK_SPREAD_PCT = 0.05
        # Need spread / mid = 0.05 exactly
        # If bid=4.75, ask=5.25 => spread=0.50, mid=5.00, pct=0.10 => fails
        # If bid=4.875, ask=5.125 => spread=0.25, mid=5.00, pct=0.05 => passes (<=)
        opt = self._make_opt(bid=4.875, ask=5.125)
        assert BaseStrategy.validate_bid_ask_spread(opt) is True
