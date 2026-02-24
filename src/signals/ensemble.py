"""Meta-learner combining all signal streams for Project Titan.

Collects calibrated outputs from all signal generators (technical, sentiment,
options flow, regime, GEX, insider, VRP, cross-asset) and combines them
through an XGBoost meta-learner with isotonic regression calibration to
produce a final confidence score between 0.0 and 1.0.

When no trained model exists on disk, a weighted-average fallback model is
created automatically so the system can trade before ML training completes.

Usage::

    from src.signals.ensemble import (
        EnsembleSignalGenerator, SignalInputs, EnsembleResult,
    )

    generator = EnsembleSignalGenerator(confidence_threshold=0.78)
    await generator.load_model("models/ensemble_xgb.json")

    result = await generator.generate_signal("AAPL", signals)
    if result.should_trade:
        print(f"Trade {result.ticker} with confidence {result.confidence:.2f}")
"""

from __future__ import annotations

import pickle
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from pydantic import BaseModel, Field
from xgboost import XGBClassifier

from src.utils.logging import get_logger
from src.utils.metrics import CONFIDENCE_SCORE, SIGNAL_SCORE

if TYPE_CHECKING:
    import structlog

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Default signal stream weights used when no trained model is available.
# These encode prior beliefs about each stream's predictive value.
DEFAULT_WEIGHTS: dict[str, float] = {
    "technical": 0.25,
    "regime": 0.20,
    "flow": 0.15,
    "vrp": 0.15,
    "sentiment": 0.10,
    "cross_asset": 0.10,
    "gex": 0.05,
}

# Feature group boundaries within the flattened feature vector.  Used to
# map XGBoost feature importances back to named signal streams.
FEATURE_GROUP_RANGES: dict[str, tuple[int, int]] = {
    "technical": (0, 20),  # indices 0..19 — 20 features
    "sentiment": (20, 23),  # indices 20..22 — 3 features
    "flow": (23, 27),  # indices 23..26 — 4 features
    "regime": (27, 32),  # indices 27..31 — 5 features (4 one-hot + confidence)
    "gex": (32, 35),  # indices 32..34 — 3 features
    "insider": (35, 39),  # indices 35..38 — 4 features
    "vrp": (39, 44),  # indices 39..43 — 5 features
    "cross_asset": (44, 48),  # indices 44..47 — 4 features
}

TOTAL_FEATURES: int = 48

# Canonical regime names for one-hot encoding.
REGIME_NAMES: list[str] = [
    "low_vol_trend",
    "high_vol_trend",
    "range_bound",
    "crisis",
]

# Directional signals and their weights for computing direction bias.
_DIRECTIONAL_WEIGHTS: dict[str, float] = {
    "sentiment": 0.20,
    "flow": 0.25,
    "insider": 0.15,
    "cross_asset": 0.15,
    "gex": 0.10,
    "technical": 0.15,
}

DEFAULT_MODEL_VERSION: str = "fallback-v1"


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class SignalInputs(BaseModel):
    """All inputs needed from signal generators.

    Every field has a sensible default so the ensemble works with partial
    data.  Missing signal streams contribute neutral (zero) values to the
    feature vector rather than blocking the pipeline.

    Attributes:
        technical_score: Aggregate technical score (0--1).
        technical_features: Raw technical feature values keyed by name.
        sentiment_score: Sentiment score from FinBERT (-1 to 1).
        sentiment_articles: Number of articles analysed.
        sentiment_confidence: Average model confidence for sentiment.
        flow_score: Net directional options flow score (-1 to 1).
        flow_consistency: Multi-day consistency score (0--1).
        flow_net_premium: Net directional premium in dollars.
        flow_num_unusual: Count of unusual activity detections.
        regime: Current market regime identifier.
        regime_confidence: Regime classification confidence (0--1).
        gex_score: Gamma exposure directional score (-1 to 1).
        gex_net_gex: Raw net GEX value (for normalization).
        gex_regime: GEX regime label (positive/negative).
        insider_score: Insider cluster score (-1 to 1).
        insider_num_buys: Number of insider purchases detected.
        insider_num_sells: Number of insider sales detected.
        insider_net_value: Net insider purchase value in dollars.
        vrp_iv_rank: IV Rank (0--100).
        vrp_iv_percentile: IV Percentile (0--100).
        vrp_score: Volatility risk premium score (0--1).
        vrp_hv_iv_ratio: Historical vol / implied vol ratio.
        vrp_spread: IV minus RV spread.
        cross_asset_score: Cross-asset composite score (-1 to 1).
        cross_asset_bias: Directional bias from cross-asset signals.
        cross_asset_yield_curve_score: Yield curve signal score (-1 to 1).
        cross_asset_credit_score: Credit spread signal score (-1 to 1).
        cross_asset_vix_ts_score: VIX term structure score (-1 to 1).
    """

    # Technical
    technical_score: float = Field(default=0.0, ge=0.0, le=1.0)
    technical_features: dict[str, float] = Field(default_factory=dict)

    # Sentiment
    sentiment_score: float = Field(default=0.0, ge=-1.0, le=1.0)
    sentiment_articles: int = Field(default=0, ge=0)
    sentiment_confidence: float = Field(default=0.0, ge=0.0, le=1.0)

    # Options Flow
    flow_score: float = Field(default=0.0, ge=-1.0, le=1.0)
    flow_consistency: float = Field(default=0.0, ge=0.0, le=1.0)
    flow_net_premium: float = Field(default=0.0)
    flow_num_unusual: int = Field(default=0, ge=0)

    # Regime
    regime: str = Field(default="unknown")
    regime_confidence: float = Field(default=0.0, ge=0.0, le=1.0)

    # GEX
    gex_score: float = Field(default=0.0, ge=-1.0, le=1.0)
    gex_net_gex: float = Field(default=0.0)
    gex_regime: str = Field(default="unknown")

    # Insider
    insider_score: float = Field(default=0.0, ge=-1.0, le=1.0)
    insider_num_buys: int = Field(default=0, ge=0)
    insider_num_sells: int = Field(default=0, ge=0)
    insider_net_value: float = Field(default=0.0)

    # VRP
    vrp_iv_rank: float = Field(default=0.0, ge=0.0, le=100.0)
    vrp_iv_percentile: float = Field(default=0.0, ge=0.0, le=100.0)
    vrp_score: float = Field(default=0.0, ge=0.0, le=1.0)
    vrp_hv_iv_ratio: float = Field(default=1.0)
    vrp_spread: float = Field(default=0.0)

    # Cross-Asset
    cross_asset_score: float = Field(default=0.0, ge=-1.0, le=1.0)
    cross_asset_bias: str = Field(default="neutral")
    cross_asset_yield_curve_score: float = Field(default=0.0, ge=-1.0, le=1.0)
    cross_asset_credit_score: float = Field(default=0.0, ge=-1.0, le=1.0)
    cross_asset_vix_ts_score: float = Field(default=0.0, ge=-1.0, le=1.0)


class EnsembleResult(BaseModel):
    """Result produced by the ensemble meta-learner.

    Attributes:
        ticker: Underlying symbol.
        confidence: Final calibrated confidence score (0--1).
        raw_score: Raw XGBoost output before calibration (0--1).
        should_trade: Whether confidence exceeds the threshold.
        signal_contributions: Mapping of signal stream name to its
            contribution percentage of the final prediction.
        direction_bias: Derived directional bias (bullish, bearish, or
            neutral) based on weighted directional signals.
        inputs: The original signal inputs that produced this result.
        model_version: Version identifier of the model used.
        timestamp: UTC timestamp when the signal was generated.
    """

    ticker: str = Field(..., min_length=1, max_length=10)
    confidence: float = Field(ge=0.0, le=1.0)
    raw_score: float = Field(ge=0.0, le=1.0)
    should_trade: bool
    signal_contributions: dict[str, float] = Field(default_factory=dict)
    direction_bias: str = Field(default="neutral")
    inputs: SignalInputs
    model_version: str = Field(default=DEFAULT_MODEL_VERSION)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


# ---------------------------------------------------------------------------
# EnsembleSignalGenerator
# ---------------------------------------------------------------------------


class EnsembleSignalGenerator:
    """Meta-learner that combines all signal streams into a single score.

    Uses an XGBoost classifier to predict the probability of a profitable
    trade from a ~48-feature vector built from technical, sentiment,
    options flow, regime, GEX, insider, VRP, and cross-asset signals.
    The raw model output is post-processed through isotonic regression
    calibration to produce a well-calibrated probability.

    When no trained model file is found on disk, a deterministic
    weighted-average fallback is used so the system can begin trading
    immediately.  The fallback is replaced once offline training produces
    a proper model.

    Args:
        confidence_threshold: Minimum calibrated score to recommend a
            trade.  Defaults to ``0.78``.
        model_path: Filesystem path to the serialized XGBoost model in
            JSON format.  Defaults to ``"models/ensemble_xgb.json"``.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.78,
        model_path: str = "models/ensemble_xgb.json",
    ) -> None:
        self._confidence_threshold: float = confidence_threshold
        self._model_path: str = model_path
        self._log: structlog.stdlib.BoundLogger = get_logger("signals.ensemble")

        # Model state — populated by load_model() or _create_default_model().
        self._model: XGBClassifier | None = None
        self._calibrator: Any | None = None
        self._model_version: str = DEFAULT_MODEL_VERSION
        self._using_fallback: bool = True

        self._log.info(
            "ensemble_generator_created",
            confidence_threshold=self._confidence_threshold,
            model_path=self._model_path,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def generate_signal(
        self,
        ticker: str,
        signals: SignalInputs,
    ) -> EnsembleResult:
        """Generate a combined confidence signal for *ticker*.

        Builds a feature vector from all signal inputs, runs it through
        the XGBoost meta-learner (or weighted-average fallback), applies
        isotonic calibration, and returns a fully populated
        :class:`EnsembleResult`.

        Args:
            ticker: Underlying symbol (e.g. ``"AAPL"``).
            signals: Aggregated inputs from all signal generators.

        Returns:
            An :class:`EnsembleResult` containing the final confidence
            score, direction bias, feature contributions, and metadata.
        """
        features = self.build_feature_vector(signals)
        raw_score = self.predict_raw(features)
        calibrated = self.calibrate(raw_score)
        trade_flag = self.should_trade(calibrated)
        contributions = self.get_signal_contributions(features)
        direction = self._compute_direction_bias(signals)

        result = EnsembleResult(
            ticker=ticker,
            confidence=round(calibrated, 6),
            raw_score=round(raw_score, 6),
            should_trade=trade_flag,
            signal_contributions=contributions,
            direction_bias=direction,
            inputs=signals,
            model_version=self._model_version,
            timestamp=datetime.now(UTC),
        )

        # Record Prometheus metrics.
        CONFIDENCE_SCORE.observe(calibrated)
        SIGNAL_SCORE.labels(signal_type="ensemble", ticker=ticker).set(calibrated)

        self._log.info(
            "ensemble_signal_generated",
            ticker=ticker,
            raw_score=round(raw_score, 4),
            calibrated=round(calibrated, 4),
            should_trade=trade_flag,
            direction_bias=direction,
            model_version=self._model_version,
            using_fallback=self._using_fallback,
        )

        return result

    def build_feature_vector(self, signals: SignalInputs) -> np.ndarray:
        """Build a numeric feature vector from all signal components.

        Extracts numeric features from each signal stream, handles
        missing data by filling with neutral values, and returns a
        1-D numpy array of length :data:`TOTAL_FEATURES`.

        Feature layout (48 total):
            * [0..19]  Technical: top 20 features (RSI_14, MACD_hist,
              ADX, BB_width, ATR_pct, trend_strength, etc.)
            * [20..22] Sentiment: score, num_articles (normalized),
              avg_confidence
            * [23..26] Flow: score, net_premium (normalized),
              consistency, num_unusual (normalized)
            * [27..31] Regime: one-hot encoded (4 columns) + confidence
            * [32..34] GEX: score, net_gex (normalized), regime binary
            * [35..38] Insider: score, num_buys (normalized),
              num_sells (normalized), net_value (normalized)
            * [39..43] VRP: iv_rank (scaled), iv_percentile (scaled),
              vrp spread, hv_iv_ratio, score
            * [44..47] Cross-Asset: score, yield_curve_score,
              credit_score, vix_ts_score

        Args:
            signals: Aggregated signal inputs.

        Returns:
            A 1-D numpy array of shape ``(TOTAL_FEATURES,)`` with
            dtype ``float32``.
        """
        features = np.zeros(TOTAL_FEATURES, dtype=np.float32)

        # -- Technical features (indices 0..19) ----------------------------
        _TECHNICAL_KEYS: list[str] = [  # noqa: N806
            "RSI_14",
            "MACD_hist",
            "ADX",
            "BB_width",
            "ATR_pct",
            "trend_strength",
            "SMA_20_50_cross",
            "SMA_50_200_cross",
            "volume_ratio",
            "OBV_slope",
            "VWAP_deviation",
            "stoch_K",
            "stoch_D",
            "CCI_20",
            "MFI_14",
            "williams_R",
            "ROC_10",
            "TRIX",
            "ultimate_osc",
            "technical_composite",
        ]
        tech_feats = signals.technical_features
        for idx, key in enumerate(_TECHNICAL_KEYS):
            features[idx] = float(tech_feats.get(key, 0.0))

        # If the technical composite was not provided, fall back to the
        # aggregate technical_score.
        if features[19] == 0.0 and signals.technical_score > 0.0:
            features[19] = signals.technical_score

        # -- Sentiment features (indices 20..22) ---------------------------
        features[20] = signals.sentiment_score
        # Normalize article count: cap at 50 and scale to [0, 1].
        features[21] = min(signals.sentiment_articles / 50.0, 1.0)
        features[22] = signals.sentiment_confidence

        # -- Options Flow features (indices 23..26) ------------------------
        features[23] = signals.flow_score
        # Normalize net premium: divide by $10M cap, clip to [-1, 1].
        premium_norm = signals.flow_net_premium / 10_000_000.0
        features[24] = float(np.clip(premium_norm, -1.0, 1.0))
        features[25] = signals.flow_consistency
        # Normalize unusual count: cap at 20 and scale to [0, 1].
        features[26] = min(signals.flow_num_unusual / 20.0, 1.0)

        # -- Regime features (indices 27..31) ------------------------------
        # One-hot encode the regime (4 columns, indices 27..30).
        regime_lower = signals.regime.lower().strip()
        for i, regime_name in enumerate(REGIME_NAMES):
            features[27 + i] = 1.0 if regime_lower == regime_name else 0.0
        features[31] = signals.regime_confidence

        # -- GEX features (indices 32..34) ---------------------------------
        features[32] = signals.gex_score
        # Normalize net GEX: divide by $1B cap, clip to [-1, 1].
        gex_norm = signals.gex_net_gex / 1_000_000_000.0
        features[33] = float(np.clip(gex_norm, -1.0, 1.0))
        # Binary: 1 if positive GEX regime, 0 otherwise.
        features[34] = 1.0 if signals.gex_regime.lower() == "positive" else 0.0

        # -- Insider features (indices 35..38) -----------------------------
        features[35] = signals.insider_score
        # Normalize counts: cap at 10 and scale to [0, 1].
        features[36] = min(signals.insider_num_buys / 10.0, 1.0)
        features[37] = min(signals.insider_num_sells / 10.0, 1.0)
        # Normalize net insider value: divide by $10M cap, clip to [-1, 1].
        insider_val_norm = signals.insider_net_value / 10_000_000.0
        features[38] = float(np.clip(insider_val_norm, -1.0, 1.0))

        # -- VRP features (indices 39..43) ---------------------------------
        # Scale IV Rank and IV Percentile from 0--100 to 0--1.
        features[39] = signals.vrp_iv_rank / 100.0
        features[40] = signals.vrp_iv_percentile / 100.0
        # VRP spread: clip to [-1, 1] (typical range for vol difference).
        features[41] = float(np.clip(signals.vrp_spread, -1.0, 1.0))
        # HV/IV ratio: clip to [0, 3] and scale to [0, 1].
        features[42] = float(np.clip(signals.vrp_hv_iv_ratio / 3.0, 0.0, 1.0))
        features[43] = signals.vrp_score

        # -- Cross-Asset features (indices 44..47) -------------------------
        features[44] = signals.cross_asset_score
        features[45] = signals.cross_asset_yield_curve_score
        features[46] = signals.cross_asset_credit_score
        features[47] = signals.cross_asset_vix_ts_score

        return features

    def predict_raw(self, features: np.ndarray) -> float:
        """Run features through the trained XGBoost model.

        If the model is using the weighted-average fallback (no trained
        model loaded), computes a score by weighting each signal stream's
        features according to :data:`DEFAULT_WEIGHTS`.

        Args:
            features: 1-D numpy array of shape ``(TOTAL_FEATURES,)``
                produced by :meth:`build_feature_vector`.

        Returns:
            Raw probability between 0.0 and 1.0.
        """
        if self._using_fallback or self._model is None:
            return self._predict_fallback(features)

        # XGBoost predict_proba returns (n_samples, n_classes).
        # We want P(class=1) — the probability of a profitable trade.
        features_2d = features.reshape(1, -1)
        probas = self._model.predict_proba(features_2d)
        # probas shape: (1, 2) for binary classification.
        raw_score = float(probas[0, 1])
        return float(np.clip(raw_score, 0.0, 1.0))

    def calibrate(self, raw_score: float) -> float:
        """Apply isotonic regression calibration to a raw model output.

        Maps the raw XGBoost probability to a calibrated probability
        using a pre-fitted isotonic regression model.  If no calibrator
        is loaded, the raw score is returned unchanged.

        Args:
            raw_score: Raw model output between 0.0 and 1.0.

        Returns:
            Calibrated probability between 0.0 and 1.0.
        """
        if self._calibrator is None:
            return raw_score

        # IsotonicRegression.predict expects an array.
        calibrated = self._calibrator.predict(np.array([raw_score]))[0]
        return float(np.clip(calibrated, 0.0, 1.0))

    def should_trade(self, calibrated_score: float) -> bool:
        """Determine whether the calibrated score exceeds the threshold.

        Args:
            calibrated_score: Calibrated confidence score (0--1).

        Returns:
            ``True`` if the score meets or exceeds the configured
            confidence threshold.
        """
        return calibrated_score >= self._confidence_threshold

    async def load_model(self, model_path: str | None = None) -> None:
        """Load a trained XGBoost model and calibrator from disk.

        If the model file does not exist, falls back to creating a
        default weighted-average model.  The calibrator pickle file is
        expected at ``<model_path>.calibrator.pkl``.

        Args:
            model_path: Optional override for the model file path.
                Defaults to the path provided at construction time.
        """
        path = Path(model_path or self._model_path)
        calibrator_path = path.with_suffix(".calibrator.pkl")

        if not path.exists():
            self._log.warning(
                "model_file_not_found",
                model_path=str(path),
                action="creating_default_fallback",
            )
            self._create_default_model()
            return

        # Load the XGBoost model.
        try:
            model = XGBClassifier()
            model.load_model(str(path))
            self._model = model
            self._using_fallback = False

            # Extract model metadata.
            n_features = getattr(model, "n_features_in_", TOTAL_FEATURES)
            self._model_version = f"xgb-{path.stem}-f{n_features}"

            self._log.info(
                "xgboost_model_loaded",
                model_path=str(path),
                n_features=n_features,
                model_version=self._model_version,
            )
        except Exception:
            self._log.exception(
                "failed_to_load_xgboost_model",
                model_path=str(path),
            )
            self._create_default_model()
            return

        # Load the isotonic calibrator if available.
        if calibrator_path.exists():
            try:
                with open(calibrator_path, "rb") as f:
                    self._calibrator = pickle.load(f)  # noqa: S301
                self._log.info(
                    "calibrator_loaded",
                    calibrator_path=str(calibrator_path),
                )
            except Exception:
                self._log.exception(
                    "failed_to_load_calibrator",
                    calibrator_path=str(calibrator_path),
                    action="proceeding_without_calibration",
                )
                self._calibrator = None
        else:
            self._log.info(
                "no_calibrator_file",
                calibrator_path=str(calibrator_path),
                action="using_raw_scores",
            )
            self._calibrator = None

    async def save_model(self, model_path: str | None = None) -> None:
        """Save the current model and calibrator to disk.

        Creates the parent directory if it does not exist.  The XGBoost
        model is saved in JSON format; the calibrator is pickled.

        Args:
            model_path: Optional override for the output file path.
                Defaults to the path provided at construction time.
        """
        path = Path(model_path or self._model_path)
        calibrator_path = path.with_suffix(".calibrator.pkl")

        # Ensure the parent directory exists.
        path.parent.mkdir(parents=True, exist_ok=True)

        if self._model is not None and not self._using_fallback:
            try:
                self._model.save_model(str(path))
                self._log.info("xgboost_model_saved", model_path=str(path))
            except Exception:
                self._log.exception(
                    "failed_to_save_xgboost_model",
                    model_path=str(path),
                )

        if self._calibrator is not None:
            try:
                with open(calibrator_path, "wb") as f:
                    pickle.dump(self._calibrator, f, protocol=pickle.HIGHEST_PROTOCOL)
                self._log.info(
                    "calibrator_saved",
                    calibrator_path=str(calibrator_path),
                )
            except Exception:
                self._log.exception(
                    "failed_to_save_calibrator",
                    calibrator_path=str(calibrator_path),
                )

    def get_signal_contributions(
        self,
        features: np.ndarray,
    ) -> dict[str, float]:
        """Extract per-stream contribution percentages.

        For a trained XGBoost model, feature importances are aggregated
        by signal group.  For the weighted-average fallback, the static
        default weights are returned.

        Args:
            features: 1-D numpy array of shape ``(TOTAL_FEATURES,)``.

        Returns:
            Dictionary mapping signal stream names (e.g. ``"technical"``,
            ``"sentiment"``) to their contribution as a percentage (values
            sum to approximately 100.0).
        """
        if self._using_fallback or self._model is None:
            return {
                name: round(weight * 100.0, 2)
                for name, weight in DEFAULT_WEIGHTS.items()
            }

        # Get per-feature importances from XGBoost (gain-based).
        try:
            importances = self._model.feature_importances_
        except AttributeError:
            self._log.warning(
                "feature_importances_unavailable",
                action="returning_default_weights",
            )
            return {
                name: round(weight * 100.0, 2)
                for name, weight in DEFAULT_WEIGHTS.items()
            }

        # Aggregate importances by signal group.
        group_importances: dict[str, float] = {}
        for group_name, (start, end) in FEATURE_GROUP_RANGES.items():
            group_slice = importances[start:end]
            group_importances[group_name] = float(np.sum(group_slice))

        # Normalize to percentages.
        total_importance = sum(group_importances.values())
        if total_importance <= 0.0:
            return {
                name: round(weight * 100.0, 2)
                for name, weight in DEFAULT_WEIGHTS.items()
            }

        contributions: dict[str, float] = {}
        for name, importance in group_importances.items():
            contributions[name] = round(
                (importance / total_importance) * 100.0,
                2,
            )

        return contributions

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def confidence_threshold(self) -> float:
        """Return the current confidence threshold for trade signals."""
        return self._confidence_threshold

    @property
    def model_version(self) -> str:
        """Return the version identifier of the loaded model."""
        return self._model_version

    @property
    def using_fallback(self) -> bool:
        """Return ``True`` if the weighted-average fallback is in use."""
        return self._using_fallback

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _create_default_model(self) -> None:
        """Create a weighted-average fallback model.

        This deterministic model does not use XGBoost at all.  Instead,
        it computes a weighted sum of each signal stream's primary score
        using :data:`DEFAULT_WEIGHTS`.  This allows the system to trade
        before ML training has completed.

        The model is marked as ``_using_fallback = True`` so that
        :meth:`predict_raw` routes to the fallback path.
        """
        self._model = None
        self._calibrator = None
        self._using_fallback = True
        self._model_version = DEFAULT_MODEL_VERSION

        self._log.info(
            "default_fallback_model_created",
            model_version=self._model_version,
            weights=DEFAULT_WEIGHTS,
        )

    def _predict_fallback(self, features: np.ndarray) -> float:
        """Compute a weighted-average score from signal stream features.

        Each signal stream's primary score feature is extracted from the
        feature vector and multiplied by its default weight.  The result
        is clipped to [0.0, 1.0].

        The mapping from feature vector index to primary score:
            * technical  -> index 19  (technical_composite / technical_score)
            * sentiment  -> index 20  (sentiment_score, remapped from [-1,1] to [0,1])
            * flow       -> index 23  (flow_score, remapped from [-1,1] to [0,1])
            * regime     -> index 31  (regime_confidence)
            * vrp        -> index 43  (vrp_score)
            * cross_asset-> index 44  (cross_asset_score, remapped from [-1,1] to [0,1])
            * gex        -> index 32  (gex_score, remapped from [-1,1] to [0,1])

        Args:
            features: 1-D numpy array of shape ``(TOTAL_FEATURES,)``.

        Returns:
            Weighted-average score between 0.0 and 1.0.
        """
        # Extract primary scores and remap [-1, 1] signals to [0, 1].
        technical_val = float(features[19])  # Already 0..1
        sentiment_val = (float(features[20]) + 1.0) / 2.0  # [-1,1] -> [0,1]
        flow_val = (float(features[23]) + 1.0) / 2.0  # [-1,1] -> [0,1]
        regime_val = float(features[31])  # Already 0..1
        vrp_val = float(features[43])  # Already 0..1
        cross_asset_val = (float(features[44]) + 1.0) / 2.0  # [-1,1] -> [0,1]
        gex_val = (float(features[32]) + 1.0) / 2.0  # [-1,1] -> [0,1]

        score = (
            DEFAULT_WEIGHTS["technical"] * technical_val
            + DEFAULT_WEIGHTS["sentiment"] * sentiment_val
            + DEFAULT_WEIGHTS["flow"] * flow_val
            + DEFAULT_WEIGHTS["regime"] * regime_val
            + DEFAULT_WEIGHTS["vrp"] * vrp_val
            + DEFAULT_WEIGHTS["cross_asset"] * cross_asset_val
            + DEFAULT_WEIGHTS["gex"] * gex_val
        )

        return float(np.clip(score, 0.0, 1.0))

    def _compute_direction_bias(self, signals: SignalInputs) -> str:
        """Derive a directional bias from weighted directional signals.

        Combines sentiment, options flow, insider, cross-asset, GEX, and
        technical scores to produce a net directional reading.  Scores
        on the [-1, 1] scale are used directly; the technical score (on
        [0, 1]) is remapped to [-1, 1] with 0.5 as neutral.

        Args:
            signals: Signal inputs with directional components.

        Returns:
            One of ``"bullish"``, ``"bearish"``, or ``"neutral"``.
        """
        # Remap technical_score from [0, 1] to [-1, 1].
        tech_directional = (signals.technical_score - 0.5) * 2.0

        directional_scores: dict[str, float] = {
            "sentiment": signals.sentiment_score,
            "flow": signals.flow_score,
            "insider": signals.insider_score,
            "cross_asset": signals.cross_asset_score,
            "gex": signals.gex_score,
            "technical": tech_directional,
        }

        weighted_sum = 0.0
        total_weight = 0.0
        for name, score in directional_scores.items():
            weight = _DIRECTIONAL_WEIGHTS.get(name, 0.0)
            weighted_sum += score * weight
            total_weight += weight

        if total_weight <= 0.0:
            return "neutral"

        net_direction = weighted_sum / total_weight

        # Thresholds for classification.
        _BULLISH_THRESHOLD: float = 0.15  # noqa: N806
        _BEARISH_THRESHOLD: float = -0.15  # noqa: N806

        if net_direction >= _BULLISH_THRESHOLD:
            return "bullish"
        elif net_direction <= _BEARISH_THRESHOLD:
            return "bearish"
        else:
            return "neutral"
