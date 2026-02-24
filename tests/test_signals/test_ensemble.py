"""Unit tests for src/signals/ensemble.py — ensemble signal generator.

Tests cover:
  - SignalInputs default construction and field validation
  - EnsembleResult construction
  - Confidence threshold gating (>= 0.78 trades, below does not)
  - Feature vector construction (build_feature_vector)
  - Fallback predict_raw weighted-average scoring
  - Isotonic calibration placement (after scoring, before threshold)
  - Signal contribution tracking with fallback weights
  - Direction bias computation
  - Full generate_signal() end-to-end flow
  - Model loading fallback when file is missing
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.signals.ensemble import (
    DEFAULT_MODEL_VERSION,
    DEFAULT_WEIGHTS,
    FEATURE_GROUP_RANGES,
    REGIME_NAMES,
    TOTAL_FEATURES,
    EnsembleResult,
    EnsembleSignalGenerator,
    SignalInputs,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def generator() -> EnsembleSignalGenerator:
    """Return a fresh EnsembleSignalGenerator with default (fallback) model."""
    return EnsembleSignalGenerator(confidence_threshold=0.78)


@pytest.fixture()
def neutral_signals() -> SignalInputs:
    """Return SignalInputs with all-neutral / zero values."""
    return SignalInputs()


@pytest.fixture()
def bullish_signals() -> SignalInputs:
    """Return SignalInputs strongly bullish on all axes."""
    return SignalInputs(
        technical_score=0.90,
        technical_features={"technical_composite": 0.90},
        sentiment_score=0.80,
        sentiment_articles=10,
        sentiment_confidence=0.85,
        flow_score=0.70,
        flow_consistency=0.80,
        flow_net_premium=5_000_000.0,
        flow_num_unusual=8,
        regime="low_vol_trend",
        regime_confidence=0.90,
        gex_score=0.50,
        gex_net_gex=500_000_000.0,
        gex_regime="positive_gamma",
        insider_score=0.60,
        insider_num_buys=5,
        insider_num_sells=0,
        insider_net_value=3_000_000.0,
        vrp_iv_rank=45.0,
        vrp_iv_percentile=50.0,
        vrp_score=0.75,
        vrp_hv_iv_ratio=0.85,
        vrp_spread=8.0,
        cross_asset_score=0.60,
        cross_asset_bias="risk_on",
        cross_asset_yield_curve_score=0.40,
        cross_asset_credit_score=0.30,
        cross_asset_vix_ts_score=0.50,
    )


# ===========================================================================
# SignalInputs construction tests
# ===========================================================================


class TestSignalInputs:
    """Verify default values and field validation on SignalInputs."""

    def test_default_construction(self):
        si = SignalInputs()
        assert si.technical_score == 0.0
        assert si.sentiment_score == 0.0
        assert si.flow_score == 0.0
        assert si.regime == "unknown"
        assert si.regime_confidence == 0.0
        assert si.gex_score == 0.0
        assert si.insider_score == 0.0
        assert si.vrp_iv_rank == 0.0
        assert si.vrp_score == 0.0
        assert si.cross_asset_score == 0.0
        assert si.cross_asset_bias == "neutral"

    def test_technical_features_defaults_to_empty_dict(self):
        si = SignalInputs()
        assert si.technical_features == {}

    def test_vrp_defaults(self):
        si = SignalInputs()
        assert si.vrp_hv_iv_ratio == 1.0
        assert si.vrp_spread == 0.0

    def test_sentiment_score_range(self):
        """Sentiment score accepts [-1, 1] range."""
        si = SignalInputs(sentiment_score=-1.0)
        assert si.sentiment_score == -1.0
        si = SignalInputs(sentiment_score=1.0)
        assert si.sentiment_score == 1.0

    def test_sentiment_score_out_of_range(self):
        with pytest.raises(ValueError):
            SignalInputs(sentiment_score=1.5)
        with pytest.raises(ValueError):
            SignalInputs(sentiment_score=-1.5)

    def test_technical_score_out_of_range(self):
        with pytest.raises(ValueError):
            SignalInputs(technical_score=-0.1)
        with pytest.raises(ValueError):
            SignalInputs(technical_score=1.1)

    def test_vrp_iv_rank_out_of_range(self):
        with pytest.raises(ValueError):
            SignalInputs(vrp_iv_rank=-1.0)
        with pytest.raises(ValueError):
            SignalInputs(vrp_iv_rank=101.0)

    def test_all_fields_populated(self, bullish_signals: SignalInputs):
        assert bullish_signals.technical_score == 0.90
        assert bullish_signals.regime == "low_vol_trend"
        assert bullish_signals.gex_regime == "positive_gamma"


# ===========================================================================
# EnsembleResult construction tests
# ===========================================================================


class TestEnsembleResult:
    """Verify EnsembleResult model construction."""

    def test_basic_construction(self, neutral_signals: SignalInputs):
        result = EnsembleResult(
            ticker="AAPL",
            confidence=0.82,
            raw_score=0.80,
            should_trade=True,
            inputs=neutral_signals,
        )
        assert result.ticker == "AAPL"
        assert result.confidence == 0.82
        assert result.should_trade is True
        assert result.direction_bias == "neutral"
        assert result.model_version == DEFAULT_MODEL_VERSION

    def test_timestamp_auto_populated(self, neutral_signals: SignalInputs):
        result = EnsembleResult(
            ticker="AAPL",
            confidence=0.50,
            raw_score=0.50,
            should_trade=False,
            inputs=neutral_signals,
        )
        assert result.timestamp is not None
        assert isinstance(result.timestamp, datetime)

    def test_signal_contributions_default_empty(self, neutral_signals: SignalInputs):
        result = EnsembleResult(
            ticker="AAPL",
            confidence=0.50,
            raw_score=0.50,
            should_trade=False,
            inputs=neutral_signals,
        )
        assert result.signal_contributions == {}


# ===========================================================================
# EnsembleSignalGenerator initialization tests
# ===========================================================================


class TestGeneratorInit:
    """Test generator construction and properties."""

    def test_default_threshold(self, generator: EnsembleSignalGenerator):
        assert generator.confidence_threshold == 0.78

    def test_custom_threshold(self):
        gen = EnsembleSignalGenerator(confidence_threshold=0.85)
        assert gen.confidence_threshold == 0.85

    def test_starts_with_fallback(self, generator: EnsembleSignalGenerator):
        assert generator.using_fallback is True

    def test_default_model_version(self, generator: EnsembleSignalGenerator):
        assert generator.model_version == DEFAULT_MODEL_VERSION


# ===========================================================================
# Confidence threshold tests
# ===========================================================================


class TestConfidenceThreshold:
    """Verify the should_trade boundary at 0.78."""

    def test_below_threshold_no_trade(self, generator: EnsembleSignalGenerator):
        assert generator.should_trade(0.77) is False

    def test_at_threshold_trades(self, generator: EnsembleSignalGenerator):
        """Score >= 0.78 should trade (spec says 'minimum', so inclusive)."""
        assert generator.should_trade(0.78) is True

    def test_above_threshold_trades(self, generator: EnsembleSignalGenerator):
        assert generator.should_trade(0.90) is True

    def test_zero_no_trade(self, generator: EnsembleSignalGenerator):
        assert generator.should_trade(0.0) is False

    def test_one_trades(self, generator: EnsembleSignalGenerator):
        assert generator.should_trade(1.0) is True

    def test_barely_below(self, generator: EnsembleSignalGenerator):
        assert generator.should_trade(0.7799) is False

    def test_custom_threshold(self):
        gen = EnsembleSignalGenerator(confidence_threshold=0.50)
        assert gen.should_trade(0.50) is True
        assert gen.should_trade(0.49) is False


# ===========================================================================
# build_feature_vector() tests
# ===========================================================================


class TestBuildFeatureVector:
    """Verify the 48-feature vector construction."""

    def test_output_shape(
        self, generator: EnsembleSignalGenerator, neutral_signals: SignalInputs
    ):
        vec = generator.build_feature_vector(neutral_signals)
        assert vec.shape == (TOTAL_FEATURES,)
        assert vec.dtype == np.float32

    def test_neutral_signals_produce_zeros_mostly(
        self,
        generator: EnsembleSignalGenerator,
        neutral_signals: SignalInputs,
    ):
        vec = generator.build_feature_vector(neutral_signals)
        # Technical features (0..19) should be 0
        assert np.all(vec[0:20] == 0.0)
        # Sentiment score should be 0
        assert vec[20] == 0.0

    def test_technical_composite_filled(self, generator: EnsembleSignalGenerator):
        signals = SignalInputs(technical_score=0.65)
        vec = generator.build_feature_vector(signals)
        # Index 19 is technical_composite; falls back to technical_score if 0
        assert vec[19] == pytest.approx(0.65, abs=1e-5)

    def test_technical_features_mapped(self, generator: EnsembleSignalGenerator):
        signals = SignalInputs(
            technical_features={"RSI_14": 65.0, "MACD_hist": 0.5, "ADX": 30.0},
        )
        vec = generator.build_feature_vector(signals)
        assert vec[0] == pytest.approx(65.0, abs=1e-5)  # RSI_14
        assert vec[1] == pytest.approx(0.5, abs=1e-5)  # MACD_hist
        assert vec[2] == pytest.approx(30.0, abs=1e-5)  # ADX

    def test_sentiment_features(self, generator: EnsembleSignalGenerator):
        signals = SignalInputs(
            sentiment_score=0.7,
            sentiment_articles=25,
            sentiment_confidence=0.9,
        )
        vec = generator.build_feature_vector(signals)
        assert vec[20] == pytest.approx(0.7, abs=1e-5)
        assert vec[21] == pytest.approx(25 / 50.0, abs=1e-5)  # normalized
        assert vec[22] == pytest.approx(0.9, abs=1e-5)

    def test_sentiment_articles_capped_at_50(self, generator: EnsembleSignalGenerator):
        signals = SignalInputs(sentiment_articles=100)
        vec = generator.build_feature_vector(signals)
        assert vec[21] == pytest.approx(1.0, abs=1e-5)

    def test_flow_features(self, generator: EnsembleSignalGenerator):
        signals = SignalInputs(
            flow_score=0.5,
            flow_net_premium=5_000_000.0,
            flow_consistency=0.8,
            flow_num_unusual=10,
        )
        vec = generator.build_feature_vector(signals)
        assert vec[23] == pytest.approx(0.5, abs=1e-5)
        assert vec[24] == pytest.approx(0.5, abs=1e-5)  # 5M / 10M
        assert vec[25] == pytest.approx(0.8, abs=1e-5)
        assert vec[26] == pytest.approx(0.5, abs=1e-5)  # 10 / 20

    def test_flow_net_premium_clipped(self, generator: EnsembleSignalGenerator):
        signals = SignalInputs(flow_net_premium=20_000_000.0)
        vec = generator.build_feature_vector(signals)
        assert vec[24] == pytest.approx(1.0, abs=1e-5)

    def test_regime_one_hot_encoding(self, generator: EnsembleSignalGenerator):
        signals = SignalInputs(regime="low_vol_trend", regime_confidence=0.85)
        vec = generator.build_feature_vector(signals)
        # low_vol_trend is index 0 in REGIME_NAMES
        assert vec[27] == 1.0
        assert vec[28] == 0.0
        assert vec[29] == 0.0
        assert vec[30] == 0.0
        assert vec[31] == pytest.approx(0.85, abs=1e-5)

    def test_regime_crisis_encoding(self, generator: EnsembleSignalGenerator):
        signals = SignalInputs(regime="crisis", regime_confidence=0.95)
        vec = generator.build_feature_vector(signals)
        assert vec[27] == 0.0  # low_vol_trend
        assert vec[28] == 0.0  # high_vol_trend
        assert vec[29] == 0.0  # range_bound
        assert vec[30] == 1.0  # crisis
        assert vec[31] == pytest.approx(0.95, abs=1e-5)

    def test_unknown_regime_all_zeros(self, generator: EnsembleSignalGenerator):
        signals = SignalInputs(regime="unknown")
        vec = generator.build_feature_vector(signals)
        assert vec[27] == 0.0
        assert vec[28] == 0.0
        assert vec[29] == 0.0
        assert vec[30] == 0.0

    def test_gex_features(self, generator: EnsembleSignalGenerator):
        signals = SignalInputs(
            gex_score=0.6,
            gex_net_gex=500_000_000.0,
            gex_regime="positive_gamma",
        )
        vec = generator.build_feature_vector(signals)
        assert vec[32] == pytest.approx(0.6, abs=1e-5)
        assert vec[33] == pytest.approx(0.5, abs=1e-5)  # 500M / 1B
        assert vec[34] == 1.0  # positive_gamma binary

    def test_gex_negative_regime(self, generator: EnsembleSignalGenerator):
        signals = SignalInputs(gex_regime="negative_gamma")
        vec = generator.build_feature_vector(signals)
        assert vec[34] == 0.0

    def test_insider_features(self, generator: EnsembleSignalGenerator):
        signals = SignalInputs(
            insider_score=0.4,
            insider_num_buys=5,
            insider_num_sells=2,
            insider_net_value=2_000_000.0,
        )
        vec = generator.build_feature_vector(signals)
        assert vec[35] == pytest.approx(0.4, abs=1e-5)
        assert vec[36] == pytest.approx(0.5, abs=1e-5)  # 5/10
        assert vec[37] == pytest.approx(0.2, abs=1e-5)  # 2/10
        assert vec[38] == pytest.approx(0.2, abs=1e-5)  # 2M/10M

    def test_vrp_features(self, generator: EnsembleSignalGenerator):
        signals = SignalInputs(
            vrp_iv_rank=50.0,
            vrp_iv_percentile=60.0,
            vrp_spread=15.0,
            vrp_hv_iv_ratio=0.9,
            vrp_score=0.7,
        )
        vec = generator.build_feature_vector(signals)
        assert vec[39] == pytest.approx(0.5, abs=1e-5)  # 50/100
        assert vec[40] == pytest.approx(0.6, abs=1e-5)  # 60/100
        assert vec[41] == pytest.approx(0.5, abs=1e-5)  # 15/30
        assert vec[42] == pytest.approx(0.3, abs=1e-5)  # 0.9/3.0
        assert vec[43] == pytest.approx(0.7, abs=1e-5)

    def test_cross_asset_features(self, generator: EnsembleSignalGenerator):
        signals = SignalInputs(
            cross_asset_score=0.5,
            cross_asset_yield_curve_score=0.3,
            cross_asset_credit_score=-0.2,
            cross_asset_vix_ts_score=0.4,
        )
        vec = generator.build_feature_vector(signals)
        assert vec[44] == pytest.approx(0.5, abs=1e-5)
        assert vec[45] == pytest.approx(0.3, abs=1e-5)
        assert vec[46] == pytest.approx(-0.2, abs=1e-5)
        assert vec[47] == pytest.approx(0.4, abs=1e-5)

    def test_feature_group_ranges_cover_all(self):
        """All indices 0..47 must be covered by FEATURE_GROUP_RANGES."""
        covered = set()
        for start, end in FEATURE_GROUP_RANGES.values():
            for i in range(start, end):
                covered.add(i)
        assert covered == set(range(TOTAL_FEATURES))


# ===========================================================================
# predict_raw() fallback tests
# ===========================================================================


class TestPredictRawFallback:
    """Test the weighted-average fallback when no trained model is loaded."""

    def test_neutral_signals_midrange_score(self, generator: EnsembleSignalGenerator):
        """All-zero signals with remapping should produce a mid-range score."""
        signals = SignalInputs()
        features = generator.build_feature_vector(signals)
        score = generator.predict_raw(features)
        # With all zeros:
        # - technical (idx 19) = 0.0 => 0.0 * 0.22 = 0.0
        # - sentiment (idx 20) = 0.0 => (0+1)/2=0.5 * 0.08 = 0.04
        # - flow (idx 23) = 0.0 => (0+1)/2=0.5 * 0.15 = 0.075
        # - regime (idx 31) = 0.0 => 0.0 * 0.18 = 0.0
        # - insider (idx 35) = 0.0 => (0+1)/2=0.5 * 0.10 = 0.05
        # - vrp (idx 43) = 0.0 => 0.0 * 0.13 = 0.0
        # - cross_asset (idx 44) = 0.0 => (0+1)/2=0.5 * 0.09 = 0.045
        # - gex (idx 32) = 0.0 => (0+1)/2=0.5 * 0.05 = 0.025
        # Total = 0.235
        expected = (
            0.22 * 0.0
            + 0.08 * 0.5
            + 0.15 * 0.5
            + 0.18 * 0.0
            + 0.10 * 0.5
            + 0.13 * 0.0
            + 0.09 * 0.5
            + 0.05 * 0.5
        )
        assert score == pytest.approx(expected, abs=1e-4)

    def test_all_max_signals_high_score(self, generator: EnsembleSignalGenerator):
        """All signals at maximum should produce a high score close to 1.0."""
        signals = SignalInputs(
            technical_score=1.0,
            sentiment_score=1.0,
            flow_score=1.0,
            regime="low_vol_trend",
            regime_confidence=1.0,
            gex_score=1.0,
            insider_score=1.0,
            vrp_score=1.0,
            cross_asset_score=1.0,
        )
        features = generator.build_feature_vector(signals)
        score = generator.predict_raw(features)
        # technical: 1.0*0.22, sentiment: (1+1)/2=1.0*0.08,
        # flow: (1+1)/2=1.0*0.15, regime: 1.0*0.18,
        # insider: (1+1)/2=1.0*0.10, vrp: 1.0*0.13,
        # cross_asset: (1+1)/2=1.0*0.09, gex: (1+1)/2=1.0*0.05
        # = sum of all weights = 1.0
        assert score == pytest.approx(1.0, abs=0.01)

    def test_score_clipped_to_unit_interval(self, generator: EnsembleSignalGenerator):
        signals = SignalInputs()
        features = generator.build_feature_vector(signals)
        score = generator.predict_raw(features)
        assert 0.0 <= score <= 1.0

    def test_fallback_flag(self, generator: EnsembleSignalGenerator):
        assert generator.using_fallback is True


# ===========================================================================
# calibrate() tests
# ===========================================================================


class TestCalibrate:
    """Test isotonic calibration placement."""

    def test_no_calibrator_passthrough(self, generator: EnsembleSignalGenerator):
        """Without a calibrator loaded, raw score passes through unchanged."""
        assert generator.calibrate(0.65) == pytest.approx(0.65, abs=1e-6)

    def test_calibrator_applied_when_present(self, generator: EnsembleSignalGenerator):
        """When a calibrator is set, it should transform the raw score."""
        mock_calibrator = MagicMock()
        mock_calibrator.predict.return_value = np.array([0.82])
        generator._calibrator = mock_calibrator

        result = generator.calibrate(0.75)
        assert result == pytest.approx(0.82, abs=1e-6)
        mock_calibrator.predict.assert_called_once()

    def test_calibration_happens_after_scoring(
        self, generator: EnsembleSignalGenerator
    ):
        """In generate_signal flow: raw score is computed first, then calibrated."""
        # This test verifies the ordering by checking the pipeline steps
        signals = SignalInputs(technical_score=0.5)
        features = generator.build_feature_vector(signals)
        raw = generator.predict_raw(features)
        # Without calibrator, calibrate is identity
        calibrated = generator.calibrate(raw)
        assert calibrated == raw

    def test_calibration_output_clipped(self, generator: EnsembleSignalGenerator):
        """Calibrator output is clipped to [0, 1]."""
        mock_calibrator = MagicMock()
        mock_calibrator.predict.return_value = np.array([1.5])
        generator._calibrator = mock_calibrator
        result = generator.calibrate(0.95)
        assert result <= 1.0

    def test_calibration_negative_clipped(self, generator: EnsembleSignalGenerator):
        mock_calibrator = MagicMock()
        mock_calibrator.predict.return_value = np.array([-0.1])
        generator._calibrator = mock_calibrator
        result = generator.calibrate(0.05)
        assert result >= 0.0


# ===========================================================================
# Signal contribution tracking tests
# ===========================================================================


class TestSignalContributions:
    """Verify individual signal weights are tracked."""

    def test_fallback_contributions_match_default_weights(
        self,
        generator: EnsembleSignalGenerator,
    ):
        signals = SignalInputs()
        features = generator.build_feature_vector(signals)
        contributions = generator.get_signal_contributions(features)

        for name, weight in DEFAULT_WEIGHTS.items():
            expected_pct = round(weight * 100.0, 2)
            assert contributions[name] == expected_pct, (
                f"Contribution for '{name}' should be {expected_pct}%, "
                f"got {contributions[name]}%"
            )

    def test_contributions_sum_to_100(self, generator: EnsembleSignalGenerator):
        signals = SignalInputs()
        features = generator.build_feature_vector(signals)
        contributions = generator.get_signal_contributions(features)
        total = sum(contributions.values())
        assert total == pytest.approx(100.0, abs=0.1)

    def test_all_signal_streams_present(self, generator: EnsembleSignalGenerator):
        signals = SignalInputs()
        features = generator.build_feature_vector(signals)
        contributions = generator.get_signal_contributions(features)
        expected_keys = set(DEFAULT_WEIGHTS.keys())
        assert set(contributions.keys()) == expected_keys

    def test_default_weights_sum_to_one(self):
        """The DEFAULT_WEIGHTS dict must sum to 1.0 for a proper probability."""
        total = sum(DEFAULT_WEIGHTS.values())
        assert total == pytest.approx(1.0, abs=1e-6)


# ===========================================================================
# Direction bias tests
# ===========================================================================


class TestDirectionBias:
    """Verify directional bias computation."""

    def test_bearish_with_default_signals(self, generator: EnsembleSignalGenerator):
        """Default technical_score=0.0 maps to -1.0, pulling bias bearish."""
        signals = SignalInputs()
        bias = generator._compute_direction_bias(signals)
        # All directional scores are 0 except technical which remaps
        # 0.0 -> (0.0 - 0.5) * 2.0 = -1.0, weighted at 0.15 => net negative
        assert bias == "bearish"

    def test_bullish_bias(
        self, generator: EnsembleSignalGenerator, bullish_signals: SignalInputs
    ):
        bias = generator._compute_direction_bias(bullish_signals)
        assert bias == "bullish"

    def test_bearish_bias(self, generator: EnsembleSignalGenerator):
        signals = SignalInputs(
            technical_score=0.1,  # maps to (0.1-0.5)*2 = -0.8
            sentiment_score=-0.8,
            flow_score=-0.7,
            insider_score=-0.5,
            cross_asset_score=-0.6,
            gex_score=-0.4,
        )
        bias = generator._compute_direction_bias(signals)
        assert bias == "bearish"

    def test_mixed_signals_neutral(self, generator: EnsembleSignalGenerator):
        """When bullish and bearish signals cancel out, result is neutral."""
        signals = SignalInputs(
            technical_score=0.5,  # neutral
            sentiment_score=0.0,
            flow_score=0.0,
            insider_score=0.0,
            cross_asset_score=0.0,
            gex_score=0.0,
        )
        bias = generator._compute_direction_bias(signals)
        assert bias == "neutral"


# ===========================================================================
# Full generate_signal() end-to-end tests
# ===========================================================================


class TestGenerateSignal:
    """End-to-end generate_signal() with fallback model."""

    @pytest.mark.asyncio
    async def test_basic_generate(self, generator: EnsembleSignalGenerator):
        signals = SignalInputs(technical_score=0.5)
        result = await generator.generate_signal("AAPL", signals)

        assert isinstance(result, EnsembleResult)
        assert result.ticker == "AAPL"
        assert 0.0 <= result.confidence <= 1.0
        assert 0.0 <= result.raw_score <= 1.0
        assert result.model_version == DEFAULT_MODEL_VERSION
        assert result.inputs is signals

    @pytest.mark.asyncio
    async def test_high_confidence_should_trade(
        self, generator: EnsembleSignalGenerator
    ):
        """With all-max signals, the fallback model should produce should_trade=True."""
        signals = SignalInputs(
            technical_score=1.0,
            sentiment_score=1.0,
            flow_score=1.0,
            regime="low_vol_trend",
            regime_confidence=1.0,
            gex_score=1.0,
            insider_score=1.0,
            vrp_score=1.0,
            cross_asset_score=1.0,
        )
        result = await generator.generate_signal("AAPL", signals)
        # Score should be ~1.0
        assert result.should_trade is True
        assert result.confidence >= 0.78

    @pytest.mark.asyncio
    async def test_low_confidence_should_not_trade(
        self, generator: EnsembleSignalGenerator
    ):
        signals = SignalInputs()  # all zeros/neutral
        result = await generator.generate_signal("AAPL", signals)
        # Score should be ~0.235 (well below 0.78)
        assert result.should_trade is False
        assert result.confidence < 0.78

    @pytest.mark.asyncio
    async def test_signal_contributions_populated(
        self, generator: EnsembleSignalGenerator
    ):
        signals = SignalInputs(technical_score=0.5)
        result = await generator.generate_signal("AAPL", signals)
        assert len(result.signal_contributions) > 0
        assert "technical" in result.signal_contributions

    @pytest.mark.asyncio
    async def test_direction_bias_populated(self, generator: EnsembleSignalGenerator):
        signals = SignalInputs()
        result = await generator.generate_signal("AAPL", signals)
        assert result.direction_bias in ("bullish", "bearish", "neutral")

    @pytest.mark.asyncio
    async def test_timestamp_is_utc(self, generator: EnsembleSignalGenerator):
        signals = SignalInputs()
        result = await generator.generate_signal("AAPL", signals)
        assert result.timestamp.tzinfo is not None


# ===========================================================================
# Model loading tests
# ===========================================================================


class TestLoadModel:
    """Test model loading logic."""

    @pytest.mark.asyncio
    async def test_missing_file_creates_fallback(
        self, generator: EnsembleSignalGenerator
    ):
        await generator.load_model("/nonexistent/path/model.json")
        assert generator.using_fallback is True
        assert generator.model_version == DEFAULT_MODEL_VERSION
        assert generator._model is None
        assert generator._calibrator is None

    @pytest.mark.asyncio
    async def test_create_default_model(self, generator: EnsembleSignalGenerator):
        generator._create_default_model()
        assert generator._model is None
        assert generator._calibrator is None
        assert generator._using_fallback is True
        assert generator._model_version == DEFAULT_MODEL_VERSION


# ===========================================================================
# Constants validation tests
# ===========================================================================


class TestConstants:
    """Verify the ensemble module's constants are correctly defined."""

    def test_total_features_matches_ranges(self):
        """TOTAL_FEATURES should equal the max end index in FEATURE_GROUP_RANGES."""
        max_end = max(end for _, end in FEATURE_GROUP_RANGES.values())
        assert max_end == TOTAL_FEATURES

    def test_feature_groups_non_overlapping(self):
        """Feature group ranges should not overlap."""
        all_indices: list[int] = []
        for start, end in FEATURE_GROUP_RANGES.values():
            for i in range(start, end):
                assert i not in all_indices, f"Index {i} is duplicated"
                all_indices.append(i)

    def test_regime_names_count(self):
        assert len(REGIME_NAMES) == 4

    def test_default_weights_keys(self):
        expected = {
            "technical",
            "regime",
            "flow",
            "vrp",
            "insider",
            "sentiment",
            "cross_asset",
            "gex",
        }
        assert set(DEFAULT_WEIGHTS.keys()) == expected
