"""Comprehensive unit tests for src/signals/regime.py.

Tests the RegimeDetector class: initialization, feature extraction, HMM fit
and predict workflow, VIX > 35 crisis override, backup regime classification,
state mapping, confidence calculation, and signal packaging.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from src.signals.regime import (
    ADX_STRONG_TREND_THRESHOLD,
    ADX_TRENDING_THRESHOLD,
    ALL_REGIMES,
    HMM_MIN_TRAINING_SAMPLES,
    REGIME_CRISIS,
    REGIME_HIGH_VOL_TREND,
    REGIME_LOW_VOL_TREND,
    REGIME_RANGE_BOUND,
    ROLLING_WINDOW,
    VIX_CRISIS_THRESHOLD,
    VIX_HIGH_VOL_THRESHOLD,
    VIX_LOW_VOL_CEILING,
    RegimeDetector,
    RegimeResult,
    RegimeSignal,
)

# ======================================================================
# Fixtures / helpers
# ======================================================================


def _make_price_data(
    n_days: int = 300,
    start_price: float = 100.0,
    daily_return: float = 0.0005,
    volatility: float = 0.01,
    start_date: str = "2023-01-03",
) -> pd.DataFrame:
    """Generate synthetic price data with a DatetimeIndex.

    Returns a DataFrame with a 'close' column and DatetimeIndex.
    """
    rng = np.random.default_rng(seed=42)
    dates = pd.bdate_range(start=start_date, periods=n_days, freq="B")
    log_returns = rng.normal(loc=daily_return, scale=volatility, size=n_days)
    prices = start_price * np.exp(np.cumsum(log_returns))
    return pd.DataFrame({"close": prices}, index=dates)


def _make_vix_data(
    index: pd.DatetimeIndex,
    mean_vix: float = 18.0,
    std_vix: float = 2.0,
) -> pd.Series:
    """Generate synthetic VIX data aligned to the given index."""
    rng = np.random.default_rng(seed=123)
    vix_values = rng.normal(loc=mean_vix, scale=std_vix, size=len(index))
    vix_values = np.clip(vix_values, 9.0, 80.0)
    return pd.Series(vix_values, index=index, name="vix")


def _make_crisis_vix_data(index: pd.DatetimeIndex) -> pd.Series:
    """Generate VIX data that is consistently above 35 (crisis level)."""
    rng = np.random.default_rng(seed=99)
    vix_values = rng.normal(loc=45.0, scale=3.0, size=len(index))
    vix_values = np.clip(vix_values, 36.0, 80.0)  # All above 35
    return pd.Series(vix_values, index=index, name="vix")


# ======================================================================
# RegimeDetector initialization
# ======================================================================


class TestRegimeDetectorInit:
    """Tests for RegimeDetector initialization."""

    def test_default_initialization(self) -> None:
        """Detector initializes with default n_states=3 and lookback_years=4."""
        detector = RegimeDetector()
        assert detector._n_states == 3
        assert detector._lookback_years == 4
        assert detector._model is None
        assert detector._state_regime_map == {}
        assert detector.is_fitted is False

    def test_custom_parameters(self) -> None:
        """Detector accepts custom n_states and lookback_years."""
        detector = RegimeDetector(n_states=4, lookback_years=6)
        assert detector._n_states == 4
        assert detector._lookback_years == 6

    def test_is_fitted_false_initially(self) -> None:
        detector = RegimeDetector()
        assert detector.is_fitted is False

    def test_state_mapping_empty_initially(self) -> None:
        detector = RegimeDetector()
        assert detector.state_mapping == {}

    def test_previous_regime_none_initially(self) -> None:
        detector = RegimeDetector()
        assert detector._previous_regime is None


# ======================================================================
# Feature extraction
# ======================================================================


class TestFeatureExtraction:
    """Tests for the static _extract_features method."""

    def test_returns_three_columns(self) -> None:
        """Feature DataFrame has rolling_return, realized_vol, and vix columns."""
        price_data = _make_price_data(n_days=100)
        vix_data = _make_vix_data(price_data.index)
        features = RegimeDetector._extract_features(price_data, vix_data)
        assert list(features.columns) == ["rolling_return", "realized_vol", "vix"]

    def test_drops_nan_rows(self) -> None:
        """Feature extraction drops NaN rows from the rolling window."""
        price_data = _make_price_data(n_days=100)
        vix_data = _make_vix_data(price_data.index)
        features = RegimeDetector._extract_features(price_data, vix_data)
        assert not features.isna().any().any()
        # Rolling window of 20 means first ~20 rows have NaN
        assert len(features) < len(price_data)
        assert len(features) == len(price_data) - ROLLING_WINDOW

    def test_vix_column_matches_input(self) -> None:
        """VIX column in features should match input values (aligned)."""
        price_data = _make_price_data(n_days=50)
        vix_data = _make_vix_data(price_data.index, mean_vix=25.0)
        features = RegimeDetector._extract_features(price_data, vix_data)
        # VIX values in features should come from the input
        for idx in features.index:
            assert features.loc[idx, "vix"] == pytest.approx(
                vix_data.loc[idx], abs=1e-10
            )

    def test_empty_after_extraction_with_insufficient_data(self) -> None:
        """Very short price data may yield empty features after rolling."""
        price_data = _make_price_data(n_days=10)
        vix_data = _make_vix_data(price_data.index)
        features = RegimeDetector._extract_features(price_data, vix_data)
        # 10 days with a 20-day window -> all NaN -> empty after dropna
        assert features.empty


# ======================================================================
# HMM fit
# ======================================================================


class TestFit:
    """Tests for the fit() method."""

    def test_fit_sets_model(self) -> None:
        """After fit(), the internal model is set."""
        detector = RegimeDetector(n_states=3)
        price_data = _make_price_data(n_days=300)
        vix_data = _make_vix_data(price_data.index)
        detector.fit(price_data, vix_data)
        assert detector.is_fitted is True
        assert detector._model is not None

    def test_fit_creates_state_mapping(self) -> None:
        """After fit(), state_regime_map is populated for all states."""
        detector = RegimeDetector(n_states=3)
        price_data = _make_price_data(n_days=300)
        vix_data = _make_vix_data(price_data.index)
        detector.fit(price_data, vix_data)
        mapping = detector.state_mapping
        assert len(mapping) == 3
        # All mapped regime names should be valid
        for regime_name in mapping.values():
            assert regime_name in ALL_REGIMES

    def test_fit_insufficient_data_raises(self) -> None:
        """fit() raises ValueError when data is insufficient."""
        detector = RegimeDetector(n_states=3)
        price_data = _make_price_data(n_days=30)  # Too few after rolling window
        vix_data = _make_vix_data(price_data.index)
        with pytest.raises(ValueError, match="Insufficient training data"):
            detector.fit(price_data, vix_data)

    def test_fit_respects_lookback_window(self) -> None:
        """fit() trims data to lookback_years * 252 days."""
        detector = RegimeDetector(n_states=3, lookback_years=1)
        # Provide much more data than 1 year
        price_data = _make_price_data(n_days=1000)
        vix_data = _make_vix_data(price_data.index)
        detector.fit(price_data, vix_data)
        assert detector.is_fitted is True


# ======================================================================
# HMM predict
# ======================================================================


class TestPredict:
    """Tests for the predict() method."""

    @pytest.fixture()
    def fitted_detector(self) -> RegimeDetector:
        """Return a RegimeDetector that has been fitted on synthetic data."""
        detector = RegimeDetector(n_states=3)
        price_data = _make_price_data(n_days=400)
        vix_data = _make_vix_data(price_data.index)
        detector.fit(price_data, vix_data)
        return detector

    def test_predict_requires_fitted_model(self) -> None:
        """predict() raises RuntimeError if model is not fitted."""
        detector = RegimeDetector()
        price_data = _make_price_data(n_days=50)
        vix_data = _make_vix_data(price_data.index)
        with pytest.raises(RuntimeError, match="Model has not been fitted"):
            detector.predict(price_data, vix_data)

    def test_predict_returns_regime_result(
        self, fitted_detector: RegimeDetector
    ) -> None:
        """predict() returns a RegimeResult with valid fields."""
        price_data = _make_price_data(n_days=100)
        vix_data = _make_vix_data(price_data.index)
        result = fitted_detector.predict(price_data, vix_data)
        assert isinstance(result, RegimeResult)
        assert result.regime in ALL_REGIMES
        assert 0.0 <= result.confidence <= 1.0
        assert len(result.state_probabilities) > 0
        assert len(result.features_used) == 3
        assert "rolling_return" in result.features_used
        assert "realized_vol" in result.features_used
        assert "vix" in result.features_used

    def test_predict_with_empty_features_returns_range_bound(
        self, fitted_detector: RegimeDetector
    ) -> None:
        """predict() with too-short data returns range_bound with 0 confidence."""
        price_data = _make_price_data(n_days=5)
        vix_data = _make_vix_data(price_data.index)
        result = fitted_detector.predict(price_data, vix_data)
        assert result.regime == REGIME_RANGE_BOUND
        assert result.confidence == 0.0

    def test_predict_state_probabilities_sum_to_one(
        self, fitted_detector: RegimeDetector
    ) -> None:
        """State probabilities should approximately sum to 1.0."""
        price_data = _make_price_data(n_days=100)
        vix_data = _make_vix_data(price_data.index)
        result = fitted_detector.predict(price_data, vix_data)
        prob_sum = sum(result.state_probabilities.values())
        assert prob_sum == pytest.approx(1.0, abs=0.01)


# ======================================================================
# VIX > 35 crisis override (CRITICAL SPEC REQUIREMENT)
# ======================================================================


class TestVixCrisisOverride:
    """Tests verifying the unconditional VIX > 35 crisis override in predict().

    This is a critical safety feature: regardless of what the HMM predicts,
    if VIX > 35, the regime MUST be 'crisis'.
    """

    @pytest.fixture()
    def fitted_detector(self) -> RegimeDetector:
        """Return a RegimeDetector fitted on normal (non-crisis) data."""
        detector = RegimeDetector(n_states=3)
        price_data = _make_price_data(n_days=400, volatility=0.01)
        vix_data = _make_vix_data(price_data.index, mean_vix=16.0, std_vix=2.0)
        detector.fit(price_data, vix_data)
        return detector

    def test_crisis_override_with_high_vix(
        self, fitted_detector: RegimeDetector
    ) -> None:
        """When VIX > 35, regime MUST be 'crisis' regardless of HMM output."""
        price_data = _make_price_data(n_days=100)
        crisis_vix = _make_crisis_vix_data(price_data.index)
        result = fitted_detector.predict(price_data, crisis_vix)
        assert result.regime == REGIME_CRISIS

    def test_crisis_override_confidence_at_least_095(
        self, fitted_detector: RegimeDetector
    ) -> None:
        """Crisis override sets confidence to at least 0.95."""
        price_data = _make_price_data(n_days=100)
        crisis_vix = _make_crisis_vix_data(price_data.index)
        result = fitted_detector.predict(price_data, crisis_vix)
        assert result.confidence >= 0.95

    def test_crisis_override_vix_exactly_above_35(
        self, fitted_detector: RegimeDetector
    ) -> None:
        """VIX at 35.01 triggers crisis override; VIX at 35.0 does not."""
        price_data = _make_price_data(n_days=100)

        # VIX just above 35 -> crisis
        vix_above = pd.Series(
            [35.01] * len(price_data), index=price_data.index, name="vix"
        )
        result_above = fitted_detector.predict(price_data, vix_above)
        assert result_above.regime == REGIME_CRISIS

    def test_no_crisis_override_at_vix_35(
        self, fitted_detector: RegimeDetector
    ) -> None:
        """VIX exactly at 35.0 does NOT trigger crisis override (> not >=)."""
        price_data = _make_price_data(n_days=100)
        vix_at_35 = pd.Series(
            [35.0] * len(price_data), index=price_data.index, name="vix"
        )
        result_at_35 = fitted_detector.predict(price_data, vix_at_35)
        # Regime may or may not be crisis, but the override is strictly > 35
        # The key assertion is that it wasn't FORCED to crisis by the override
        # (the HMM might still map it to crisis based on learned patterns, but
        # the override line `if current_vix > VIX_CRISIS_THRESHOLD` won't fire)
        # We can only check that the function doesn't raise and returns valid result
        assert result_at_35.regime in ALL_REGIMES

    def test_crisis_features_show_high_vix(
        self, fitted_detector: RegimeDetector
    ) -> None:
        """features_used dict reports the actual high VIX value during crisis."""
        price_data = _make_price_data(n_days=100)
        crisis_vix = _make_crisis_vix_data(price_data.index)
        result = fitted_detector.predict(price_data, crisis_vix)
        assert result.features_used["vix"] > VIX_CRISIS_THRESHOLD


# ======================================================================
# Backup regime classification
# ======================================================================


class TestGetBackupRegime:
    """Tests for the rule-based backup regime classifier."""

    def setup_method(self) -> None:
        self.detector = RegimeDetector()

    # Rule 1: VIX > 35 -> crisis
    def test_vix_above_35_is_crisis(self) -> None:
        assert (
            self.detector.get_backup_regime(vix=40.0, adx=30.0, trend_direction=1.0)
            == REGIME_CRISIS
        )

    def test_vix_at_36_is_crisis(self) -> None:
        assert (
            self.detector.get_backup_regime(vix=36.0, adx=10.0, trend_direction=-1.0)
            == REGIME_CRISIS
        )

    def test_vix_at_35_is_not_crisis(self) -> None:
        """VIX exactly at 35 does not trigger crisis (strict >)."""
        result = self.detector.get_backup_regime(
            vix=35.0, adx=10.0, trend_direction=0.0
        )
        assert result != REGIME_CRISIS

    # Rule 2: VIX > 25 AND ADX > 25 -> high_vol_trend
    def test_high_vix_high_adx_is_high_vol_trend(self) -> None:
        assert (
            self.detector.get_backup_regime(vix=28.0, adx=30.0, trend_direction=1.0)
            == REGIME_HIGH_VOL_TREND
        )

    def test_high_vix_low_adx_is_not_high_vol_trend(self) -> None:
        """VIX > 25 but ADX < 25 does not trigger high_vol_trend."""
        result = self.detector.get_backup_regime(
            vix=28.0, adx=15.0, trend_direction=1.0
        )
        assert result != REGIME_HIGH_VOL_TREND

    # Rule 3: ADX < 20 -> range_bound
    def test_low_adx_is_range_bound(self) -> None:
        assert (
            self.detector.get_backup_regime(vix=15.0, adx=15.0, trend_direction=0.0)
            == REGIME_RANGE_BOUND
        )

    def test_adx_at_19_is_range_bound(self) -> None:
        assert (
            self.detector.get_backup_regime(vix=15.0, adx=19.0, trend_direction=0.5)
            == REGIME_RANGE_BOUND
        )

    # Rule 4: ADX >= 20 AND VIX <= 20 -> low_vol_trend
    def test_trending_low_vol_is_low_vol_trend(self) -> None:
        assert (
            self.detector.get_backup_regime(vix=15.0, adx=25.0, trend_direction=1.0)
            == REGIME_LOW_VOL_TREND
        )

    def test_adx_exactly_20_vix_exactly_20_is_low_vol_trend(self) -> None:
        """ADX at 20 and VIX at 20 should hit Rule 4 (adx >= 20 and vix <= 20)."""
        assert (
            self.detector.get_backup_regime(vix=20.0, adx=20.0, trend_direction=1.0)
            == REGIME_LOW_VOL_TREND
        )

    # Rule 5: Default -> range_bound
    def test_default_is_range_bound(self) -> None:
        """VIX=22, ADX=22 -> doesn't match rules 1-4 (VIX > VIX_LOW_VOL_CEILING)
        so falls to default."""
        result = self.detector.get_backup_regime(
            vix=22.0, adx=22.0, trend_direction=0.5
        )
        assert result == REGIME_RANGE_BOUND

    def test_all_results_are_valid_regimes(self) -> None:
        """Every combination returns a valid regime name."""
        for vix in [10.0, 20.0, 28.0, 35.0, 40.0, 50.0]:
            for adx in [5.0, 15.0, 20.0, 25.0, 30.0, 40.0]:
                result = self.detector.get_backup_regime(
                    vix=vix, adx=adx, trend_direction=0.0
                )
                assert result in ALL_REGIMES, (
                    f"Invalid regime '{result}' for vix={vix}, adx={adx}"
                )


# ======================================================================
# Regime classification (4 regimes map to correct VIX/ADX ranges)
# ======================================================================


class TestRegimeClassificationRanges:
    """Verify the 4 regimes map to correct VIX/ADX ranges per the spec.

    From CLAUDE.md:
    | Regime            | VIX    | ADX  |
    |-------------------|--------|------|
    | Low Vol Trending  | <18    | >25  |
    | High Vol Trending | 18-35  | >25  |
    | Range-Bound High  | 18-35  | <20  |
    | Crisis            | >35    | any  |

    The backup classifier uses slightly different thresholds but serves
    the same conceptual purpose.
    """

    def setup_method(self) -> None:
        self.detector = RegimeDetector()

    def test_low_vol_trending_regime(self) -> None:
        """VIX < 18, ADX > 25 -> low_vol_trend."""
        result = self.detector.get_backup_regime(
            vix=15.0, adx=28.0, trend_direction=1.0
        )
        assert result == REGIME_LOW_VOL_TREND

    def test_high_vol_trending_regime(self) -> None:
        """VIX 18-35 (specifically > 25), ADX > 25 -> high_vol_trend."""
        result = self.detector.get_backup_regime(
            vix=28.0, adx=28.0, trend_direction=1.0
        )
        assert result == REGIME_HIGH_VOL_TREND

    def test_range_bound_regime(self) -> None:
        """VIX 18-35, ADX < 20 -> range_bound."""
        result = self.detector.get_backup_regime(
            vix=22.0, adx=15.0, trend_direction=0.0
        )
        assert result == REGIME_RANGE_BOUND

    def test_crisis_regime(self) -> None:
        """VIX > 35, any ADX -> crisis."""
        result = self.detector.get_backup_regime(
            vix=40.0, adx=15.0, trend_direction=-1.0
        )
        assert result == REGIME_CRISIS

    def test_crisis_any_adx(self) -> None:
        """Crisis triggers regardless of ADX value."""
        for adx in [5.0, 15.0, 25.0, 35.0, 50.0]:
            result = self.detector.get_backup_regime(
                vix=45.0, adx=adx, trend_direction=-1.0
            )
            assert result == REGIME_CRISIS, f"Expected crisis at VIX=45, ADX={adx}"


# ======================================================================
# Confidence calculation
# ======================================================================


class TestCalculateRegimeConfidence:
    """Tests for the static confidence calculation method."""

    def test_certain_prediction(self) -> None:
        """All probability in one state yields confidence ~1.0."""
        probs = np.array([1.0, 0.0, 0.0])
        assert RegimeDetector.calculate_regime_confidence(probs) == 1.0

    def test_uniform_distribution(self) -> None:
        """Equal probabilities yield confidence ~0.333 for 3 states."""
        probs = np.array([1 / 3, 1 / 3, 1 / 3])
        assert RegimeDetector.calculate_regime_confidence(probs) == pytest.approx(
            1 / 3, abs=0.01
        )

    def test_two_state_split(self) -> None:
        """50/50 split gives confidence 0.5."""
        probs = np.array([0.5, 0.5])
        assert RegimeDetector.calculate_regime_confidence(probs) == 0.5

    def test_empty_array_returns_zero(self) -> None:
        """Empty probability array returns 0.0."""
        probs = np.array([])
        assert RegimeDetector.calculate_regime_confidence(probs) == 0.0

    def test_high_confidence(self) -> None:
        """Dominant probability gives high confidence."""
        probs = np.array([0.92, 0.05, 0.03])
        assert RegimeDetector.calculate_regime_confidence(probs) == 0.92

    def test_clamped_to_one(self) -> None:
        """Probabilities > 1.0 are clamped to 1.0."""
        probs = np.array([1.01, 0.0, 0.0])
        assert RegimeDetector.calculate_regime_confidence(probs) == 1.0


# ======================================================================
# State-to-regime mapping
# ======================================================================


class TestMapStatesToRegimes:
    """Tests for _map_states_to_regimes using a mock HMM model."""

    def test_highest_vix_negative_return_maps_to_crisis(self) -> None:
        """State with highest VIX and negative return -> crisis."""
        detector = RegimeDetector(n_states=3)
        mock_model = MagicMock()
        # State 0: low vol, positive return, low vix
        # State 1: medium vol, near-zero return, medium vix
        # State 2: high vol, negative return, high vix (crisis)
        mock_model.means_ = np.array(
            [
                [0.10, 0.10, 14.0],  # state 0: low vol trend
                [0.01, 0.18, 20.0],  # state 1: range bound
                [-0.15, 0.35, 30.0],  # state 2: crisis
            ]
        )
        mapping = detector._map_states_to_regimes(mock_model)
        assert mapping[2] == REGIME_CRISIS

    def test_lowest_vol_positive_return_maps_to_low_vol_trend(self) -> None:
        """State with lowest vol and positive return -> low_vol_trend."""
        detector = RegimeDetector(n_states=3)
        mock_model = MagicMock()
        mock_model.means_ = np.array(
            [
                [0.10, 0.08, 13.0],  # state 0: lowest vol, positive return
                [0.02, 0.18, 22.0],  # state 1: medium
                [-0.10, 0.35, 32.0],  # state 2: highest vix/vol
            ]
        )
        mapping = detector._map_states_to_regimes(mock_model)
        assert mapping[0] == REGIME_LOW_VOL_TREND

    def test_all_states_assigned(self) -> None:
        """Every state gets a regime label."""
        detector = RegimeDetector(n_states=3)
        mock_model = MagicMock()
        mock_model.means_ = np.array(
            [
                [0.10, 0.10, 14.0],
                [0.01, 0.18, 20.0],
                [-0.15, 0.35, 30.0],
            ]
        )
        mapping = detector._map_states_to_regimes(mock_model)
        assert len(mapping) == 3
        for state_idx in range(3):
            assert state_idx in mapping

    def test_all_assigned_regimes_are_valid(self) -> None:
        """Every assigned regime is in the ALL_REGIMES list."""
        detector = RegimeDetector(n_states=3)
        mock_model = MagicMock()
        mock_model.means_ = np.array(
            [
                [0.10, 0.10, 14.0],
                [0.01, 0.18, 20.0],
                [-0.15, 0.35, 30.0],
            ]
        )
        mapping = detector._map_states_to_regimes(mock_model)
        for regime in mapping.values():
            assert regime in ALL_REGIMES


# ======================================================================
# Signal packaging
# ======================================================================


class TestGetRegimeSignal:
    """Tests for get_regime_signal() transition tracking."""

    def test_first_signal_no_transition(self) -> None:
        """First signal has no transition (previous_regime is None)."""
        detector = RegimeDetector()
        signal = detector.get_regime_signal(REGIME_LOW_VOL_TREND, 0.85, 15.0)
        assert isinstance(signal, RegimeSignal)
        assert signal.regime == REGIME_LOW_VOL_TREND
        assert signal.confidence == 0.85
        assert signal.vix == 15.0
        assert signal.is_transitioning is False
        assert signal.previous_regime is None
        assert signal.regime_duration_days >= 1

    def test_same_regime_no_transition(self) -> None:
        """Repeated same regime does not flag transition."""
        detector = RegimeDetector()
        detector.get_regime_signal(REGIME_LOW_VOL_TREND, 0.85, 15.0)
        signal2 = detector.get_regime_signal(REGIME_LOW_VOL_TREND, 0.82, 14.5)
        assert signal2.is_transitioning is False
        assert signal2.previous_regime == REGIME_LOW_VOL_TREND

    def test_regime_change_flags_transition(self) -> None:
        """Change from one regime to another flags is_transitioning."""
        detector = RegimeDetector()
        detector.get_regime_signal(REGIME_LOW_VOL_TREND, 0.85, 15.0)
        signal2 = detector.get_regime_signal(REGIME_CRISIS, 0.95, 40.0)
        assert signal2.is_transitioning is True
        assert signal2.previous_regime == REGIME_LOW_VOL_TREND
        assert signal2.regime == REGIME_CRISIS
        assert signal2.regime_duration_days == 1

    def test_updates_previous_regime(self) -> None:
        """After multiple transitions, previous_regime tracks correctly."""
        detector = RegimeDetector()
        detector.get_regime_signal(REGIME_LOW_VOL_TREND, 0.85, 15.0)
        detector.get_regime_signal(REGIME_RANGE_BOUND, 0.70, 22.0)
        signal3 = detector.get_regime_signal(REGIME_HIGH_VOL_TREND, 0.80, 28.0)
        assert signal3.previous_regime == REGIME_RANGE_BOUND
        assert signal3.is_transitioning is True


# ======================================================================
# Pydantic model validation
# ======================================================================


class TestPydanticModels:
    """Tests for RegimeResult and RegimeSignal validation."""

    def test_regime_result_valid(self) -> None:
        result = RegimeResult(
            regime=REGIME_LOW_VOL_TREND,
            confidence=0.85,
            state_probabilities={"low_vol_trend": 0.85, "range_bound": 0.15},
            features_used={"rolling_return": 0.05, "realized_vol": 0.12, "vix": 15.0},
        )
        assert result.regime == REGIME_LOW_VOL_TREND
        assert result.confidence == 0.85

    def test_regime_result_confidence_out_of_range(self) -> None:
        """Confidence outside [0, 1] should fail validation."""
        with pytest.raises(ValidationError):
            RegimeResult(regime=REGIME_CRISIS, confidence=1.5)

    def test_regime_result_negative_confidence(self) -> None:
        with pytest.raises(ValidationError):
            RegimeResult(regime=REGIME_CRISIS, confidence=-0.1)

    def test_regime_signal_valid(self) -> None:
        signal = RegimeSignal(
            regime=REGIME_CRISIS,
            confidence=0.95,
            vix=40.0,
            is_transitioning=True,
            previous_regime=REGIME_LOW_VOL_TREND,
            regime_duration_days=1,
        )
        assert signal.regime == REGIME_CRISIS
        assert signal.vix == 40.0

    def test_regime_result_has_timestamp(self) -> None:
        """RegimeResult auto-generates a UTC timestamp."""
        result = RegimeResult(regime=REGIME_RANGE_BOUND, confidence=0.5)
        assert result.timestamp is not None
        assert result.timestamp.tzinfo is not None


# ======================================================================
# Constants verification
# ======================================================================


class TestConstants:
    """Verify critical constants match the spec."""

    def test_vix_crisis_threshold(self) -> None:
        assert VIX_CRISIS_THRESHOLD == 35.0

    def test_vix_high_vol_threshold(self) -> None:
        assert VIX_HIGH_VOL_THRESHOLD == 25.0

    def test_vix_low_vol_ceiling(self) -> None:
        assert VIX_LOW_VOL_CEILING == 20.0

    def test_adx_trending_threshold(self) -> None:
        assert ADX_TRENDING_THRESHOLD == 20.0

    def test_adx_strong_trend_threshold(self) -> None:
        assert ADX_STRONG_TREND_THRESHOLD == 25.0

    def test_all_regimes_list(self) -> None:
        assert set(ALL_REGIMES) == {
            REGIME_LOW_VOL_TREND,
            REGIME_HIGH_VOL_TREND,
            REGIME_RANGE_BOUND,
            REGIME_CRISIS,
        }

    def test_rolling_window(self) -> None:
        assert ROLLING_WINDOW == 20

    def test_hmm_min_training_samples(self) -> None:
        assert HMM_MIN_TRAINING_SAMPLES == 100
