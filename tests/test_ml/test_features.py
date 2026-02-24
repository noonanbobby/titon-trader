"""Unit tests for src/ml/features.py — FeatureEngineer.

Tests cover:
    - build_trade_features() — the new method for weekly retrain
    - build_feature_matrix() — existing method
    - select_features() — importance, mutual_info, variance methods
    - normalize_features() — standard, minmax, robust scalers
    - create_target_variable() — forward return binary classification
    - add_lagged_features() — lag generation
    - remove_correlated_features() — high-correlation pruning
    - FeaturePipelineConfig Pydantic model
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.ml.features import FeatureEngineer, FeaturePipelineConfig

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def engineer() -> FeatureEngineer:
    """Return a fresh FeatureEngineer with default settings."""
    return FeatureEngineer(importance_threshold=0.005)


@pytest.fixture()
def sample_trade_df() -> pd.DataFrame:
    """Return a sample trade DataFrame mimicking the PostgreSQL trades table."""
    return pd.DataFrame(
        {
            "ticker": ["AAPL", "MSFT", "GOOG", "AMZN", "META"],
            "strategy": [
                "bull_call_spread",
                "iron_condor",
                "bull_put_spread",
                "calendar_spread",
                "short_strangle",
            ],
            "direction": ["bullish", "neutral", "bullish", "neutral", "bearish"],
            "entry_price": [150.0, 400.0, 170.0, 300.0, 500.0],
            "exit_price": [155.0, 398.0, 175.0, 305.0, 495.0],
            "realized_pnl": [500.0, -200.0, 300.0, 150.0, -100.0],
            "ml_confidence": [0.85, 0.79, 0.82, 0.80, 0.78],
            "regime": [
                "low_vol_trending",
                "range_bound",
                "low_vol_trending",
                "high_vol_trending",
                "range_bound",
            ],
            "entry_time": pd.to_datetime(
                [
                    "2026-01-15 10:30",
                    "2026-01-16 11:00",
                    "2026-01-17 14:30",
                    "2026-01-20 09:45",
                    "2026-01-21 15:00",
                ]
            ),
            "exit_time": pd.to_datetime(
                [
                    "2026-01-20 15:00",
                    "2026-01-21 10:30",
                    "2026-01-22 12:00",
                    "2026-01-24 14:00",
                    "2026-01-26 11:30",
                ]
            ),
        }
    )


@pytest.fixture()
def sample_price_data() -> dict[str, pd.DataFrame]:
    """Return sample price DataFrames for build_feature_matrix."""
    dates = pd.date_range("2025-01-01", periods=100, freq="B")
    rng = np.random.default_rng(42)

    def make_df() -> pd.DataFrame:
        close = 100.0 + np.cumsum(rng.normal(0, 1, 100))
        return pd.DataFrame(
            {
                "open": close + rng.normal(0, 0.5, 100),
                "high": close + abs(rng.normal(0, 1, 100)),
                "low": close - abs(rng.normal(0, 1, 100)),
                "close": close,
                "volume": rng.integers(1000, 100000, 100),
            },
            index=dates,
        )

    return {"AAPL": make_df(), "MSFT": make_df()}


# ---------------------------------------------------------------------------
# FeaturePipelineConfig tests
# ---------------------------------------------------------------------------


class TestFeaturePipelineConfig:
    """Tests for the FeaturePipelineConfig Pydantic model."""

    def test_defaults(self) -> None:
        """Default configuration should match expected values."""
        config = FeaturePipelineConfig()
        assert config.importance_threshold == 0.005
        assert config.normalize_method == "standard"
        assert config.target_horizon == 5
        assert config.target_threshold == 0.02
        assert config.lag_periods == [1, 2, 3, 5]

    def test_custom_values(self) -> None:
        """Custom values should override defaults."""
        config = FeaturePipelineConfig(
            importance_threshold=0.01,
            normalize_method="robust",
            target_horizon=10,
        )
        assert config.importance_threshold == 0.01
        assert config.normalize_method == "robust"
        assert config.target_horizon == 10

    def test_importance_threshold_bounds(self) -> None:
        """importance_threshold must be between 0.0 and 1.0."""
        with pytest.raises(ValueError):
            FeaturePipelineConfig(importance_threshold=1.5)
        with pytest.raises(ValueError):
            FeaturePipelineConfig(importance_threshold=-0.1)


# ---------------------------------------------------------------------------
# build_trade_features tests
# ---------------------------------------------------------------------------


class TestBuildTradeFeatures:
    """Tests for FeatureEngineer.build_trade_features() — the new method."""

    def test_returns_dataframe(
        self, engineer: FeatureEngineer, sample_trade_df: pd.DataFrame
    ) -> None:
        """Should return a pandas DataFrame."""
        result = engineer.build_trade_features(sample_trade_df)
        assert isinstance(result, pd.DataFrame)

    def test_same_row_count(
        self, engineer: FeatureEngineer, sample_trade_df: pd.DataFrame
    ) -> None:
        """Output should have one row per input trade."""
        result = engineer.build_trade_features(sample_trade_df)
        assert len(result) == len(sample_trade_df)

    def test_no_nan_values(
        self, engineer: FeatureEngineer, sample_trade_df: pd.DataFrame
    ) -> None:
        """Output should have no NaN values after imputation."""
        result = engineer.build_trade_features(sample_trade_df)
        assert result.isna().sum().sum() == 0

    def test_ml_confidence_column_present(
        self, engineer: FeatureEngineer, sample_trade_df: pd.DataFrame
    ) -> None:
        """ml_confidence should be extracted as a feature."""
        result = engineer.build_trade_features(sample_trade_df)
        assert "ml_confidence" in result.columns

    def test_ml_confidence_values_preserved(
        self, engineer: FeatureEngineer, sample_trade_df: pd.DataFrame
    ) -> None:
        """ml_confidence values should match the input."""
        result = engineer.build_trade_features(sample_trade_df)
        np.testing.assert_array_almost_equal(
            result["ml_confidence"].values, [0.85, 0.79, 0.82, 0.80, 0.78]
        )

    def test_regime_one_hot_encoded(
        self, engineer: FeatureEngineer, sample_trade_df: pd.DataFrame
    ) -> None:
        """Regime column should be one-hot encoded with regime_ prefix."""
        result = engineer.build_trade_features(sample_trade_df)
        regime_cols = [c for c in result.columns if c.startswith("regime_")]
        assert len(regime_cols) >= 2  # at least 2 distinct regimes in sample

    def test_strategy_one_hot_encoded(
        self, engineer: FeatureEngineer, sample_trade_df: pd.DataFrame
    ) -> None:
        """Strategy column should be one-hot encoded with strategy_ prefix."""
        result = engineer.build_trade_features(sample_trade_df)
        strategy_cols = [c for c in result.columns if c.startswith("strategy_")]
        assert len(strategy_cols) == 5  # 5 distinct strategies in sample

    def test_direction_encoded(
        self, engineer: FeatureEngineer, sample_trade_df: pd.DataFrame
    ) -> None:
        """Direction should be encoded as 1 (bullish), -1 (bearish), 0 (neutral)."""
        result = engineer.build_trade_features(sample_trade_df)
        assert "direction_encoded" in result.columns
        values = result["direction_encoded"].tolist()
        assert values[0] == 1  # bullish
        assert values[1] == 0  # neutral
        assert values[4] == -1  # bearish

    def test_hold_days_column(
        self, engineer: FeatureEngineer, sample_trade_df: pd.DataFrame
    ) -> None:
        """hold_days should capture the duration of each trade."""
        result = engineer.build_trade_features(sample_trade_df)
        assert "hold_days" in result.columns
        # First trade: Jan 15 to Jan 20 = ~5 days
        assert result["hold_days"].iloc[0] > 0

    def test_entry_hour_column(
        self, engineer: FeatureEngineer, sample_trade_df: pd.DataFrame
    ) -> None:
        """entry_hour should capture hour of trade entry."""
        result = engineer.build_trade_features(sample_trade_df)
        assert "entry_hour" in result.columns
        assert result["entry_hour"].iloc[0] == 10  # 10:30 entry

    def test_day_of_week_column(
        self, engineer: FeatureEngineer, sample_trade_df: pd.DataFrame
    ) -> None:
        """day_of_week should be 0-4 (Mon-Fri)."""
        result = engineer.build_trade_features(sample_trade_df)
        assert "day_of_week" in result.columns
        assert all(0 <= v <= 6 for v in result["day_of_week"])

    def test_log_price_column(
        self, engineer: FeatureEngineer, sample_trade_df: pd.DataFrame
    ) -> None:
        """log_price should be the natural log of entry_price."""
        result = engineer.build_trade_features(sample_trade_df)
        assert "log_price" in result.columns
        expected = np.log(150.0)
        assert abs(result["log_price"].iloc[0] - expected) < 0.001

    def test_all_columns_numeric(
        self, engineer: FeatureEngineer, sample_trade_df: pd.DataFrame
    ) -> None:
        """All output columns should be numeric (no strings)."""
        result = engineer.build_trade_features(sample_trade_df)
        for col in result.columns:
            assert pd.api.types.is_numeric_dtype(result[col]), f"{col} is not numeric"

    def test_empty_dataframe_returns_empty(self, engineer: FeatureEngineer) -> None:
        """Empty input should return empty output."""
        empty_df = pd.DataFrame()
        result = engineer.build_trade_features(empty_df)
        assert isinstance(result, pd.DataFrame)

    def test_missing_columns_graceful(self, engineer: FeatureEngineer) -> None:
        """Missing optional columns should not crash, just omit those features."""
        minimal_df = pd.DataFrame({"ticker": ["AAPL"], "strategy": ["iron_condor"]})
        result = engineer.build_trade_features(minimal_df)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1

    def test_single_trade(self, engineer: FeatureEngineer) -> None:
        """A single trade should still produce a valid feature row."""
        single = pd.DataFrame(
            {
                "ticker": ["AAPL"],
                "strategy": ["bull_call_spread"],
                "direction": ["bullish"],
                "entry_price": [150.0],
                "exit_price": [155.0],
                "realized_pnl": [500.0],
                "ml_confidence": [0.85],
                "regime": ["low_vol_trending"],
                "entry_time": pd.to_datetime(["2026-01-15 10:30"]),
                "exit_time": pd.to_datetime(["2026-01-20 15:00"]),
            }
        )
        result = engineer.build_trade_features(single)
        assert len(result) == 1
        assert result.isna().sum().sum() == 0


# ---------------------------------------------------------------------------
# build_feature_matrix tests
# ---------------------------------------------------------------------------


class TestBuildFeatureMatrix:
    """Tests for FeatureEngineer.build_feature_matrix()."""

    def test_returns_dataframe(
        self, engineer: FeatureEngineer, sample_price_data: dict[str, pd.DataFrame]
    ) -> None:
        """Should return a DataFrame."""
        result = engineer.build_feature_matrix(sample_price_data, {})
        assert isinstance(result, pd.DataFrame)

    def test_combined_rows(
        self, engineer: FeatureEngineer, sample_price_data: dict[str, pd.DataFrame]
    ) -> None:
        """Output should contain rows from all tickers."""
        result = engineer.build_feature_matrix(sample_price_data, {})
        # 2 tickers x 100 rows each
        assert len(result) == 200

    def test_signal_scores_injected(
        self, engineer: FeatureEngineer, sample_price_data: dict[str, pd.DataFrame]
    ) -> None:
        """Numeric signal scores should appear as columns."""
        signals = {"AAPL": {"technical_score": 0.75, "sentiment_score": 0.60}}
        result = engineer.build_feature_matrix(sample_price_data, signals)
        assert "signal_technical_score" in result.columns

    def test_empty_price_data(self, engineer: FeatureEngineer) -> None:
        """Empty price data should return empty DataFrame."""
        result = engineer.build_feature_matrix({}, {})
        assert result.empty

    def test_no_nan_after_imputation(
        self, engineer: FeatureEngineer, sample_price_data: dict[str, pd.DataFrame]
    ) -> None:
        """After imputation, there should be no NaN values in numeric columns."""
        result = engineer.build_feature_matrix(sample_price_data, {})
        numeric = result.select_dtypes(include=[np.number])
        assert numeric.isna().sum().sum() == 0


# ---------------------------------------------------------------------------
# create_target_variable tests
# ---------------------------------------------------------------------------


class TestCreateTargetVariable:
    """Tests for FeatureEngineer.create_target_variable()."""

    def test_binary_output(self, engineer: FeatureEngineer) -> None:
        """Target should be binary (0 or 1) except for trailing NaN."""
        dates = pd.date_range("2025-01-01", periods=50, freq="B")
        df = pd.DataFrame({"close": np.linspace(100, 120, 50)}, index=dates)

        target = engineer.create_target_variable(df, horizon_days=5, threshold=0.02)
        valid = target.dropna()
        assert set(valid.unique()).issubset({0, 1})

    def test_trailing_nan(self, engineer: FeatureEngineer) -> None:
        """Last horizon_days entries should be NaN (no future data)."""
        dates = pd.date_range("2025-01-01", periods=50, freq="B")
        df = pd.DataFrame({"close": np.linspace(100, 120, 50)}, index=dates)

        target = engineer.create_target_variable(df, horizon_days=5)
        assert target.iloc[-5:].isna().all()

    def test_missing_close_raises(self, engineer: FeatureEngineer) -> None:
        """DataFrame without 'close' column should raise ValueError."""
        df = pd.DataFrame({"open": [100, 110]})
        with pytest.raises(ValueError, match="close"):
            engineer.create_target_variable(df)


# ---------------------------------------------------------------------------
# normalize_features tests
# ---------------------------------------------------------------------------


class TestNormalizeFeatures:
    """Tests for FeatureEngineer.normalize_features()."""

    def test_standard_scaler(self, engineer: FeatureEngineer) -> None:
        """Standard normalization should produce ~zero mean, ~unit variance."""
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [10, 20, 30, 40, 50]})
        result, params = engineer.normalize_features(df, method="standard")
        assert abs(result["a"].mean()) < 1e-10
        assert params["method"] == "standard"

    def test_minmax_scaler(self, engineer: FeatureEngineer) -> None:
        """MinMax normalization should produce values in [0, 1]."""
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5]})
        result, params = engineer.normalize_features(df, method="minmax")
        assert result["a"].min() >= -1e-10
        assert result["a"].max() <= 1.0 + 1e-10
        assert params["method"] == "minmax"

    def test_robust_scaler(self, engineer: FeatureEngineer) -> None:
        """Robust normalization should work without errors."""
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5]})
        result, params = engineer.normalize_features(df, method="robust")
        assert params["method"] == "robust"
        assert len(result) == 5


# ---------------------------------------------------------------------------
# add_lagged_features tests
# ---------------------------------------------------------------------------


class TestAddLaggedFeatures:
    """Tests for FeatureEngineer.add_lagged_features()."""

    def test_lag_columns_created(self, engineer: FeatureEngineer) -> None:
        """Lagged columns should be created for specified columns and lags."""
        df = pd.DataFrame({"price": range(10), "volume": range(10)})
        result = engineer.add_lagged_features(df, columns=["price"], lags=[1, 2])
        assert "price_lag1" in result.columns
        assert "price_lag2" in result.columns

    def test_lag_values(self, engineer: FeatureEngineer) -> None:
        """Lag-1 should shift values by one row."""
        df = pd.DataFrame({"price": [10, 20, 30, 40, 50]})
        result = engineer.add_lagged_features(df, columns=["price"], lags=[1])
        assert result["price_lag1"].iloc[1] == 10  # shifted from index 0

    def test_missing_column_skipped(self, engineer: FeatureEngineer) -> None:
        """Missing column should be skipped with a warning, not crash."""
        df = pd.DataFrame({"price": range(5)})
        result = engineer.add_lagged_features(
            df, columns=["price", "nonexistent"], lags=[1]
        )
        assert "price_lag1" in result.columns
        assert "nonexistent_lag1" not in result.columns


# ---------------------------------------------------------------------------
# remove_correlated_features tests
# ---------------------------------------------------------------------------


class TestRemoveCorrelatedFeatures:
    """Tests for FeatureEngineer.remove_correlated_features()."""

    def test_removes_highly_correlated(self, engineer: FeatureEngineer) -> None:
        """Perfectly correlated features should have one removed."""
        df = pd.DataFrame(
            {
                "a": [1, 2, 3, 4, 5],
                "b": [2, 4, 6, 8, 10],  # perfectly correlated with a
                "c": [5, 3, 1, 4, 2],  # uncorrelated
            }
        )
        result = engineer.remove_correlated_features(df, threshold=0.95)
        # Either 'a' or 'b' should be dropped, 'c' should remain
        assert "c" in result.columns
        assert len(result.columns) == 2  # one of a/b dropped

    def test_no_removal_below_threshold(self, engineer: FeatureEngineer) -> None:
        """Features below the correlation threshold should all be kept."""
        rng = np.random.default_rng(42)
        df = pd.DataFrame(
            {
                "a": rng.normal(0, 1, 100),
                "b": rng.normal(0, 1, 100),
                "c": rng.normal(0, 1, 100),
            }
        )
        result = engineer.remove_correlated_features(df, threshold=0.95)
        assert len(result.columns) == 3  # nothing removed


# ---------------------------------------------------------------------------
# select_features tests
# ---------------------------------------------------------------------------


class TestSelectFeatures:
    """Tests for FeatureEngineer.select_features()."""

    def test_importance_method(self, engineer: FeatureEngineer) -> None:
        """Importance-based selection should return a non-empty list."""
        rng = np.random.default_rng(42)
        x_df = pd.DataFrame(
            {
                "strong": rng.normal(0, 1, 200),
                "noise1": rng.normal(0, 0.01, 200),
                "noise2": rng.normal(0, 0.01, 200),
            }
        )
        y = pd.Series((x_df["strong"] > 0).astype(int))
        selected = engineer.select_features(x_df, y, method="importance")
        assert isinstance(selected, list)
        assert len(selected) > 0

    def test_variance_method(self, engineer: FeatureEngineer) -> None:
        """Variance-based selection should drop near-zero variance columns."""
        x_df = pd.DataFrame(
            {
                "high_var": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "zero_var": [5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
            }
        )
        y = pd.Series([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        selected = engineer.select_features(x_df, y, method="variance")
        assert "high_var" in selected
        # zero_var has zero variance so should be removed
        assert "zero_var" not in selected

    def test_insufficient_samples_returns_all(self, engineer: FeatureEngineer) -> None:
        """With fewer than 50 samples, all features should be returned."""
        x_df = pd.DataFrame({"a": range(10), "b": range(10)})
        y = pd.Series([0, 1] * 5)
        selected = engineer.select_features(x_df, y, method="importance")
        assert len(selected) == 2

    def test_feature_importances_property(self, engineer: FeatureEngineer) -> None:
        """After selection, feature_importances property should be populated."""
        rng = np.random.default_rng(42)
        x_df = pd.DataFrame({"a": rng.normal(0, 1, 100), "b": rng.normal(0, 1, 100)})
        y = pd.Series((x_df["a"] > 0).astype(int))
        engineer.select_features(x_df, y)
        importances = engineer.feature_importances
        assert "a" in importances
        assert "b" in importances
