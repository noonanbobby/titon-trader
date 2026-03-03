"""HMM-based market regime detection for Project Titan.

Fits a Gaussian Hidden Markov Model to rolling price returns, realized
volatility, and VIX levels to classify the market into one of four
regimes: ``low_vol_trend``, ``high_vol_trend``, ``range_bound``, or
``crisis``.  A rule-based backup using VIX thresholds and ADX is
available when the HMM is not yet trained or produces low-confidence
predictions.

Usage::

    from src.signals.regime import RegimeDetector

    detector = RegimeDetector(n_states=3, lookback_years=4)
    detector.fit(price_data, vix_data)
    result = detector.predict(price_data, vix_data)
    print(result.regime, result.confidence)
"""

from __future__ import annotations

import math
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from pydantic import BaseModel, Field

from src.utils.logging import get_logger

if TYPE_CHECKING:
    import structlog

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TRADING_DAYS_PER_YEAR: int = 252
ROLLING_WINDOW: int = 20
ANNUALIZATION_FACTOR: float = math.sqrt(TRADING_DAYS_PER_YEAR)

# Regime name constants
REGIME_LOW_VOL_TREND: str = "low_vol_trend"
REGIME_HIGH_VOL_TREND: str = "high_vol_trend"
REGIME_RANGE_BOUND: str = "range_bound"
REGIME_CRISIS: str = "crisis"

ALL_REGIMES: list[str] = [
    REGIME_LOW_VOL_TREND,
    REGIME_HIGH_VOL_TREND,
    REGIME_RANGE_BOUND,
    REGIME_CRISIS,
]

# Backup rule thresholds
VIX_CRISIS_THRESHOLD: float = 35.0
VIX_HIGH_VOL_THRESHOLD: float = 25.0
VIX_LOW_VOL_CEILING: float = 20.0
ADX_TRENDING_THRESHOLD: float = 20.0
ADX_STRONG_TREND_THRESHOLD: float = 25.0

# HMM training parameters
HMM_N_ITER: int = 200
HMM_TOL: float = 1e-4
HMM_COVARIANCE_TYPE: str = "full"
HMM_RANDOM_STATE: int = 42
HMM_MIN_TRAINING_SAMPLES: int = 100


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class RegimeResult(BaseModel):
    """Result of a regime prediction.

    Attributes:
        regime: The predicted market regime name.
        confidence: Confidence in the prediction (0.0--1.0), derived
            from the maximum state probability.
        state_probabilities: Mapping of regime name to its posterior
            probability from the HMM.
        features_used: Key feature values used for the prediction
            (e.g. rolling return, realized vol, VIX).
        timestamp: When the prediction was generated.
    """

    regime: str
    confidence: float = Field(ge=0.0, le=1.0)
    state_probabilities: dict[str, float] = Field(default_factory=dict)
    features_used: dict[str, float] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class RegimeSignal(BaseModel):
    """Packaged regime signal for downstream consumption by the ensemble
    and strategy selector.

    Attributes:
        regime: Current market regime name.
        confidence: Confidence score for the regime classification.
        vix: Current VIX level at the time the signal was generated.
        is_transitioning: ``True`` if the regime changed from the
            previous observation.
        previous_regime: The regime before the current one, or ``None``
            if no prior observation is available.
        regime_duration_days: Number of consecutive trading days the
            market has been in the current regime.
    """

    regime: str
    confidence: float = Field(ge=0.0, le=1.0)
    vix: float
    is_transitioning: bool = False
    previous_regime: str | None = None
    regime_duration_days: int = 0


# ---------------------------------------------------------------------------
# RegimeDetector
# ---------------------------------------------------------------------------


class RegimeDetector:
    """Hidden Markov Model-based market regime detector.

    Trains a :class:`~hmmlearn.hmm.GaussianHMM` on rolling 20-day
    returns, rolling 20-day annualized realized volatility, and VIX
    levels.  The learned hidden states are mapped to human-readable
    regime labels based on their emission distributions.

    Args:
        n_states: Number of hidden states for the HMM (default 3).
            Typically 3 or 4 produce the best regime separation.
        lookback_years: Number of years of rolling historical data to
            use when fitting the model (default 4).
    """

    def __init__(self, n_states: int = 3, lookback_years: int = 4) -> None:
        self._n_states: int = n_states
        self._lookback_years: int = lookback_years
        self._model: GaussianHMM | None = None
        self._state_regime_map: dict[int, str] = {}
        self._previous_regime: str | None = None
        self._regime_start_date: datetime | None = None
        self._log: structlog.stdlib.BoundLogger = get_logger("signals.regime")

        self._log.info(
            "regime_detector_initialized",
            n_states=n_states,
            lookback_years=lookback_years,
        )

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_features(
        price_data: pd.DataFrame,
        vix_data: pd.Series,
    ) -> pd.DataFrame:
        """Extract HMM input features from price and VIX data.

        Computes three features aligned by date index:

        1. Rolling 20-day log returns (annualized).
        2. Rolling 20-day realized volatility (annualized).
        3. VIX level.

        Args:
            price_data: DataFrame with a ``"close"`` column and a
                :class:`~pandas.DatetimeIndex`.
            vix_data: Series of VIX closing levels with a
                :class:`~pandas.DatetimeIndex`.

        Returns:
            A DataFrame with columns ``["rolling_return",
            "realized_vol", "vix"]`` and rows where all three values
            are available (NaN rows dropped).
        """
        close = price_data["close"]

        # Daily log returns
        log_returns: pd.Series = np.log(close / close.shift(1))

        # Rolling 20-day cumulative return (annualized)
        rolling_return: pd.Series = log_returns.rolling(window=ROLLING_WINDOW).sum() * (
            TRADING_DAYS_PER_YEAR / ROLLING_WINDOW
        )

        # Rolling 20-day realized volatility (annualized)
        realized_vol: pd.Series = (
            log_returns.rolling(window=ROLLING_WINDOW).std() * ANNUALIZATION_FACTOR
        )

        features = pd.DataFrame(
            {
                "rolling_return": rolling_return,
                "realized_vol": realized_vol,
                "vix": vix_data,
            },
            index=close.index,
        )

        # Align and drop any rows with missing data
        features = features.dropna()
        return features

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, price_data: pd.DataFrame, vix_data: pd.Series) -> None:
        """Fit the Gaussian HMM on historical price and VIX data.

        Uses a rolling window of ``lookback_years`` years.  After
        fitting, the learned state means/covariances are inspected to
        build a mapping from HMM state indices to human-readable regime
        names via :meth:`_map_states_to_regimes`.

        Args:
            price_data: DataFrame with at minimum a ``"close"`` column
                and a :class:`~pandas.DatetimeIndex`.
            vix_data: Series of VIX closing levels with a
                :class:`~pandas.DatetimeIndex`.

        Raises:
            ValueError: If insufficient data remains after feature
                extraction.
        """
        self._log.info(
            "regime_fit_started",
            price_data_rows=len(price_data),
            vix_data_rows=len(vix_data),
            lookback_years=self._lookback_years,
        )

        features = self._extract_features(price_data, vix_data)

        # Apply lookback window
        lookback_days = self._lookback_years * TRADING_DAYS_PER_YEAR
        if len(features) > lookback_days:
            features = features.iloc[-lookback_days:]

        if len(features) < HMM_MIN_TRAINING_SAMPLES:
            msg = (
                f"Insufficient training data: {len(features)} samples "
                f"(minimum {HMM_MIN_TRAINING_SAMPLES})"
            )
            self._log.error("regime_fit_failed", reason=msg, sample_count=len(features))
            raise ValueError(msg)

        feature_matrix: np.ndarray = features[
            ["rolling_return", "realized_vol", "vix"]
        ].values

        model = GaussianHMM(
            n_components=self._n_states,
            covariance_type=HMM_COVARIANCE_TYPE,
            n_iter=HMM_N_ITER,
            tol=HMM_TOL,
            random_state=HMM_RANDOM_STATE,
        )
        model.fit(feature_matrix)

        self._model = model
        self._state_regime_map = self._map_states_to_regimes(model)

        self._log.info(
            "regime_fit_complete",
            n_states=self._n_states,
            training_samples=len(features),
            converged=model.monitor_.converged,
            n_iterations=model.monitor_.iter,
            state_mapping=self._state_regime_map,
            means={
                str(i): {
                    "return": round(float(model.means_[i, 0]), 4),
                    "vol": round(float(model.means_[i, 1]), 4),
                    "vix": round(float(model.means_[i, 2]), 4),
                }
                for i in range(self._n_states)
            },
        )

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(
        self,
        price_data: pd.DataFrame,
        vix_data: pd.Series,
    ) -> RegimeResult:
        """Predict the current market regime from recent data.

        Extracts the same features used in :meth:`fit`, runs the HMM
        forward algorithm to obtain posterior state probabilities for
        the most recent observation, and maps the most probable state
        to a regime name.

        Args:
            price_data: DataFrame with a ``"close"`` column and a
                :class:`~pandas.DatetimeIndex`.
            vix_data: Series of VIX closing levels with a
                :class:`~pandas.DatetimeIndex`.

        Returns:
            A :class:`RegimeResult` with the predicted regime,
            confidence, state probabilities, and features used.

        Raises:
            RuntimeError: If the model has not been fitted yet.
        """
        if self._model is None:
            msg = "Model has not been fitted. Call fit() before predict()."
            self._log.error("regime_predict_failed", reason=msg)
            raise RuntimeError(msg)

        features = self._extract_features(price_data, vix_data)

        if features.empty:
            self._log.warning("regime_predict_empty_features")
            return RegimeResult(
                regime=REGIME_RANGE_BOUND,
                confidence=0.0,
                state_probabilities={},
                features_used={},
            )

        feature_matrix: np.ndarray = features[
            ["rolling_return", "realized_vol", "vix"]
        ].values

        # Compute posterior state probabilities for all observations
        state_probs: np.ndarray = self._model.predict_proba(feature_matrix)

        # Use the most recent observation
        latest_probs: np.ndarray = state_probs[-1]
        predicted_state: int = int(np.argmax(latest_probs))
        regime: str = self._state_regime_map.get(predicted_state, REGIME_RANGE_BOUND)
        confidence: float = self.calculate_regime_confidence(latest_probs)

        # Crisis override: VIX > 35 ALWAYS triggers crisis regardless of HMM
        current_vix = float(features.iloc[-1]["vix"])
        if current_vix > VIX_CRISIS_THRESHOLD:
            regime = REGIME_CRISIS
            confidence = max(confidence, 0.95)
        elif regime == REGIME_CRISIS:
            # Safety net: HMM should never produce crisis (it's not in the
            # state mapping), but guard against stale models or mapping bugs.
            self._log.warning(
                "regime_crisis_override_corrected",
                reason="HMM predicted crisis but VIX is below threshold",
                vix=current_vix,
                threshold=VIX_CRISIS_THRESHOLD,
            )
            regime = REGIME_HIGH_VOL_TREND

        # Build state probability dict keyed by regime name
        prob_by_regime: dict[str, float] = {}
        for state_idx in range(self._n_states):
            regime_name = self._state_regime_map.get(state_idx, f"state_{state_idx}")
            prob_by_regime[regime_name] = round(float(latest_probs[state_idx]), 4)

        # Latest feature values
        latest_features: dict[str, float] = {
            "rolling_return": round(float(features.iloc[-1]["rolling_return"]), 4),
            "realized_vol": round(float(features.iloc[-1]["realized_vol"]), 4),
            "vix": round(float(features.iloc[-1]["vix"]), 4),
        }

        result = RegimeResult(
            regime=regime,
            confidence=round(confidence, 4),
            state_probabilities=prob_by_regime,
            features_used=latest_features,
        )

        self._log.info(
            "regime_predicted",
            regime=regime,
            confidence=round(confidence, 4),
            state_probabilities=prob_by_regime,
            features=latest_features,
        )

        return result

    # ------------------------------------------------------------------
    # State-to-regime mapping
    # ------------------------------------------------------------------

    def _map_states_to_regimes(self, model: GaussianHMM) -> dict[int, str]:
        """Map HMM state indices to human-readable regime names.

        The 3-state HMM maps to three non-crisis regimes only:
        ``low_vol_trend``, ``high_vol_trend``, and ``range_bound``.
        Crisis is NEVER an HMM state — it is exclusively triggered by
        the VIX > 35 hard override in :meth:`predict`.

        Mapping strategy:
        - Sort states by realized volatility mean (ascending).
        - Lowest vol → ``low_vol_trend`` (calm, trending market).
        - Highest vol → ``high_vol_trend`` (volatile but trending).
        - Middle vol → ``range_bound`` (choppy, premium-rich).

        Args:
            model: A fitted :class:`~hmmlearn.hmm.GaussianHMM`.

        Returns:
            Dictionary mapping state index (``int``) to regime name
            (``str``).
        """
        means = model.means_  # shape: (n_states, n_features)
        # Feature order: [rolling_return, realized_vol, vix]

        n = means.shape[0]
        mapping: dict[int, str] = {}

        # Sort states by realized vol (feature index 1) ascending
        vol_order: list[int] = sorted(range(n), key=lambda i: float(means[i, 1]))

        # 3-state mapping: crisis is NEVER an HMM state.
        # Crisis is exclusively the VIX > 35 hard override in predict().
        # HMM states map to: low_vol_trend, range_bound, high_vol_trend
        if n == 3:
            mapping[vol_order[0]] = REGIME_LOW_VOL_TREND
            mapping[vol_order[1]] = REGIME_RANGE_BOUND
            mapping[vol_order[2]] = REGIME_HIGH_VOL_TREND
        elif n == 2:
            mapping[vol_order[0]] = REGIME_LOW_VOL_TREND
            mapping[vol_order[1]] = REGIME_HIGH_VOL_TREND
        else:
            # n >= 4: lowest → low_vol, highest → high_vol, rest → range_bound
            mapping[vol_order[0]] = REGIME_LOW_VOL_TREND
            mapping[vol_order[-1]] = REGIME_HIGH_VOL_TREND
            for i in range(1, n - 1):
                mapping[vol_order[i]] = REGIME_RANGE_BOUND

        self._log.debug(
            "state_regime_mapping",
            mapping=mapping,
            means={
                str(i): {
                    "return": round(float(means[i, 0]), 4),
                    "vol": round(float(means[i, 1]), 4),
                    "vix": round(float(means[i, 2]), 4),
                }
                for i in range(n)
            },
        )

        return mapping

    # ------------------------------------------------------------------
    # Backup regime classification
    # ------------------------------------------------------------------

    def get_backup_regime(
        self,
        vix: float,
        adx: float,
        trend_direction: float,
    ) -> str:
        """Determine the market regime using rule-based VIX/ADX thresholds.

        This backup method is used when the HMM is not yet trained, has
        not converged, or is producing low-confidence predictions.

        Rules (evaluated in priority order):

        1. VIX > 35 --> ``"crisis"``
        2. VIX > 25 **and** ADX > 25 --> ``"high_vol_trend"``
        3. ADX < 20 --> ``"range_bound"``
        4. ADX >= 20 **and** VIX <= 20 --> ``"low_vol_trend"``
        5. Default --> ``"range_bound"``

        Args:
            vix: Current VIX index level.
            adx: Current ADX (Average Directional Index) value.
            trend_direction: Signed indicator of trend direction.
                Positive values indicate an uptrend, negative values a
                downtrend.  Not currently used for regime classification
                but is logged for diagnostics.

        Returns:
            One of the four canonical regime names.
        """
        regime: str

        if vix > VIX_CRISIS_THRESHOLD:
            regime = REGIME_CRISIS
        elif vix > VIX_HIGH_VOL_THRESHOLD and adx > ADX_STRONG_TREND_THRESHOLD:
            regime = REGIME_HIGH_VOL_TREND
        elif adx < ADX_TRENDING_THRESHOLD:
            regime = REGIME_RANGE_BOUND
        elif adx >= ADX_TRENDING_THRESHOLD and vix <= VIX_LOW_VOL_CEILING:
            regime = REGIME_LOW_VOL_TREND
        else:
            regime = REGIME_RANGE_BOUND

        self._log.info(
            "backup_regime_calculated",
            regime=regime,
            vix=vix,
            adx=adx,
            trend_direction=round(trend_direction, 4),
        )

        return regime

    # ------------------------------------------------------------------
    # Confidence calculation
    # ------------------------------------------------------------------

    @staticmethod
    def calculate_regime_confidence(state_probabilities: np.ndarray) -> float:
        """Calculate the confidence of a regime prediction.

        Confidence is defined as the maximum posterior probability
        across all states.  A value near ``1 / n_states`` indicates
        maximum uncertainty; a value near ``1.0`` indicates the HMM
        is very certain about the current state.

        Args:
            state_probabilities: 1-D array of posterior state
                probabilities from the HMM (must sum to ~1.0).

        Returns:
            The maximum probability value, clamped to [0.0, 1.0].
        """
        if state_probabilities.size == 0:
            return 0.0

        max_prob: float = float(np.max(state_probabilities))
        return max(0.0, min(1.0, max_prob))

    # ------------------------------------------------------------------
    # Signal packaging
    # ------------------------------------------------------------------

    def get_regime_signal(
        self,
        regime: str,
        confidence: float,
        vix: float,
    ) -> RegimeSignal:
        """Package the current regime prediction into a signal object.

        Tracks regime transitions by comparing the current regime
        against the previously observed regime.  The
        ``regime_duration_days`` counter increments when the regime
        is unchanged and resets to 1 on a transition.

        Args:
            regime: The current regime name.
            confidence: Confidence score (0.0--1.0) for the regime.
            vix: Current VIX level.

        Returns:
            A :class:`RegimeSignal` with transition information and
            duration tracking.
        """
        is_transitioning: bool = (
            self._previous_regime is not None and regime != self._previous_regime
        )

        now = datetime.now(UTC)

        if is_transitioning or self._regime_start_date is None:
            duration_days = 1
            self._regime_start_date = now
        else:
            elapsed = now - self._regime_start_date
            duration_days = max(1, elapsed.days)

        signal = RegimeSignal(
            regime=regime,
            confidence=confidence,
            vix=vix,
            is_transitioning=is_transitioning,
            previous_regime=self._previous_regime,
            regime_duration_days=duration_days,
        )

        if is_transitioning:
            self._log.info(
                "regime_transition",
                previous_regime=self._previous_regime,
                new_regime=regime,
                confidence=round(confidence, 4),
                vix=vix,
            )

        # Update tracking state
        self._previous_regime = regime

        return signal

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_fitted(self) -> bool:
        """Return ``True`` if the HMM has been fitted."""
        return self._model is not None

    @property
    def state_mapping(self) -> dict[int, str]:
        """Return the current state-to-regime mapping (empty if not fitted)."""
        return dict(self._state_regime_map)
