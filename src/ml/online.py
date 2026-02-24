"""River online learning with ADWIN drift detection for Project Titan.

Provides incremental learning capabilities that update models one observation
at a time, enabling real-time adaptation to evolving market conditions.
ADWIN (Adaptive Windowing) detects concept drift in strategy error rates and
triggers a graduated response — from logging a warning through full model
retraining.

The online learning system operates alongside the batch-trained ensemble
models, providing a fast-adapting signal that captures recent market regime
shifts before the next scheduled batch retraining.

Usage::

    from src.ml.online import OnlineModelManager, OnlineLearnerConfig

    config = OnlineLearnerConfig(model_type="hoeffding_adaptive_tree")
    manager = OnlineModelManager(config=config)
    prediction = await manager.predict(features)
    await manager.update(features, label)
    health = manager.get_model_health()
"""

from __future__ import annotations

import json
import pickle
import time
from collections import deque
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, Field

from src.utils.logging import get_logger

if TYPE_CHECKING:
    import structlog


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ROLLING_WINDOW_SIZE: int = 100
"""Number of recent predictions to track for rolling accuracy."""

MINOR_DRIFT_THRESHOLD: int = 1
"""Number of drift events within the detection window to trigger a warning."""

MODERATE_DRIFT_THRESHOLD: int = 3
"""Number of drift events within the detection window to increase learning rate."""

SEVERE_DRIFT_THRESHOLD: int = 5
"""Number of drift events within the detection window to reset model weights."""

PERSISTENT_DRIFT_THRESHOLD: int = 8
"""Number of drift events within the detection window to trigger full retraining."""

DRIFT_DETECTION_WINDOW: int = 50
"""Number of recent observations over which to count drift events."""

DEFAULT_LEARNING_RATE: float = 0.01
"""Default learning rate for applicable online models."""

INCREASED_LEARNING_RATE: float = 0.05
"""Elevated learning rate applied after moderate drift detection."""

CONFIDENCE_FLOOR: float = 0.01
"""Minimum probability output to avoid degenerate predictions."""

CONFIDENCE_CEILING: float = 0.99
"""Maximum probability output to avoid degenerate predictions."""

MAX_DRIFT_HISTORY: int = 500
"""Maximum number of drift events to retain in history."""

STATE_FILE_EXTENSION: str = ".pkl"
"""File extension for serialized learner state."""


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class DriftAction(StrEnum):
    """Graduated response actions triggered by concept drift detection.

    The severity of the response escalates with the persistence and
    magnitude of detected drift, from passive logging through full
    model retraining.
    """

    LOG_WARNING = "log_warning"
    INCREASE_LR = "increase_lr"
    RESET_WEIGHTS = "reset_weights"
    FULL_RETRAIN = "full_retrain"


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class OnlineLearnerConfig(BaseModel):
    """Configuration for the online learning system.

    Attributes:
        model_type: River model backend to use for incremental learning.
        n_ensemble_members: Number of base learners in the ensemble
            (only applicable when model_type is ``"adwin_bagging"``).
        adwin_delta: Sensitivity parameter for the ADWIN drift detector.
            Smaller values make drift detection more sensitive.
        rolling_window_size: Number of recent predictions for rolling
            metric calculation.
        drift_detection_window: Number of recent observations over
            which to count drift events for graduated response.
        learning_rate: Initial learning rate for models that support it.
        seed: Random seed for reproducibility.
    """

    model_type: Literal[
        "hoeffding_adaptive_tree",
        "adwin_bagging",
    ] = Field(
        default="hoeffding_adaptive_tree",
        description="River model backend for incremental learning",
    )
    n_ensemble_members: int = Field(
        default=10,
        ge=3,
        le=50,
        description=("Number of base learners in ensemble (adwin_bagging only)"),
    )
    adwin_delta: float = Field(
        default=0.002,
        gt=0.0,
        lt=1.0,
        description="ADWIN sensitivity parameter (smaller = more sensitive)",
    )
    rolling_window_size: int = Field(
        default=ROLLING_WINDOW_SIZE,
        ge=10,
        le=1000,
        description="Window size for rolling accuracy calculation",
    )
    drift_detection_window: int = Field(
        default=DRIFT_DETECTION_WINDOW,
        ge=10,
        le=500,
        description="Window over which to count drift events",
    )
    learning_rate: float = Field(
        default=DEFAULT_LEARNING_RATE,
        gt=0.0,
        lt=1.0,
        description="Initial learning rate for applicable models",
    )
    seed: int = Field(
        default=42,
        description="Random seed for reproducibility",
    )


class DriftEvent(BaseModel):
    """Record of a single concept drift detection event.

    Attributes:
        timestamp: UTC timestamp when drift was detected.
        action: The graduated response action taken.
        error_rate: Current rolling error rate at time of detection.
        adwin_width: Current ADWIN window width at detection time.
        observation_count: Total observations processed so far.
        details: Additional context about the drift event.
    """

    timestamp: datetime = Field(
        description="UTC timestamp of drift detection",
    )
    action: DriftAction = Field(
        description="Graduated response action taken",
    )
    error_rate: float = Field(
        description="Rolling error rate at detection time",
    )
    adwin_width: float = Field(
        description="ADWIN window width at detection time",
    )
    observation_count: int = Field(
        description="Total observations processed at detection time",
    )
    details: str = Field(
        default="",
        description="Additional context about the drift event",
    )


class OnlinePrediction(BaseModel):
    """Result of an online model prediction.

    Attributes:
        label: Predicted binary class label (0 or 1).
        confidence: Predicted probability for the positive class,
            clipped to [0.01, 0.99].
        model_type: Name of the online model that produced the
            prediction.
        observation_count: Total observations the model has been
            trained on.
        rolling_accuracy: Current rolling accuracy over the most
            recent window.
        drift_detected: Whether concept drift was detected on the
            most recent update.
    """

    label: int = Field(
        description="Predicted binary class label (0 or 1)",
    )
    confidence: float = Field(
        description="Predicted probability for positive class",
    )
    model_type: str = Field(
        description="Online model backend name",
    )
    observation_count: int = Field(
        description="Total observations model has been trained on",
    )
    rolling_accuracy: float = Field(
        description="Rolling accuracy over recent window",
    )
    drift_detected: bool = Field(
        default=False,
        description="Whether drift was detected on last update",
    )


class LearnerState(BaseModel):
    """Serializable snapshot of the online learner state.

    Captures everything needed to restore the online learning system
    to its exact state after a restart: model weights, scaler state,
    metric history, and drift detection state.

    Attributes:
        model_type: River model backend name.
        observation_count: Total observations processed.
        created_at: UTC timestamp when state was first initialized.
        last_updated_at: UTC timestamp of most recent update.
        rolling_accuracy: Current rolling accuracy value.
        total_correct: Cumulative correct predictions.
        total_predictions: Cumulative predictions made.
        drift_event_count: Total drift events detected.
        consecutive_correct: Current streak of correct predictions.
        consecutive_wrong: Current streak of incorrect predictions.
        current_learning_rate: Active learning rate.
        config: The learner configuration.
    """

    model_type: str = Field(
        description="River model backend name",
    )
    observation_count: int = Field(
        default=0,
        description="Total observations processed",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="UTC timestamp of state initialization",
    )
    last_updated_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="UTC timestamp of most recent update",
    )
    rolling_accuracy: float = Field(
        default=0.0,
        description="Current rolling accuracy",
    )
    total_correct: int = Field(
        default=0,
        description="Cumulative correct predictions",
    )
    total_predictions: int = Field(
        default=0,
        description="Cumulative predictions made",
    )
    drift_event_count: int = Field(
        default=0,
        description="Total drift events detected",
    )
    consecutive_correct: int = Field(
        default=0,
        description="Current consecutive correct prediction streak",
    )
    consecutive_wrong: int = Field(
        default=0,
        description="Current consecutive incorrect prediction streak",
    )
    current_learning_rate: float = Field(
        default=DEFAULT_LEARNING_RATE,
        description="Active learning rate",
    )
    config: OnlineLearnerConfig = Field(
        default_factory=OnlineLearnerConfig,
        description="Learner configuration",
    )


# ---------------------------------------------------------------------------
# OnlineLearner
# ---------------------------------------------------------------------------


class OnlineLearner:
    """Incremental learning model using River for single-observation updates.

    Wraps a River classifier with online feature normalization and
    rolling performance metrics.  Supports two model backends:

    - **HoeffdingAdaptiveTreeClassifier**: A single adaptive decision tree
      that restructures itself when drift is detected internally.
    - **ADWINBaggingClassifier**: An ensemble of base trees with ADWIN-based
      change detection on each member for automatic replacement of
      underperforming learners.

    Args:
        config: Configuration specifying model type, hyperparameters,
            and metric window sizes.
    """

    def __init__(self, config: OnlineLearnerConfig) -> None:
        self._config: OnlineLearnerConfig = config
        self._log: structlog.stdlib.BoundLogger = get_logger("ml.online.learner")

        # Build the River model pipeline
        self._model = self._build_model()
        self._scaler = self._build_scaler()

        # Rolling metrics
        self._rolling_correct: deque[bool] = deque(
            maxlen=config.rolling_window_size,
        )
        self._observation_count: int = 0
        self._total_correct: int = 0
        self._total_predictions: int = 0
        self._consecutive_correct: int = 0
        self._consecutive_wrong: int = 0
        self._current_learning_rate: float = config.learning_rate

        # River metrics
        self._accuracy_metric = self._build_accuracy_metric()
        self._f1_metric = self._build_f1_metric()
        self._rolling_accuracy_metric = self._build_rolling_accuracy_metric()

        self._log.info(
            "online_learner_initialized",
            model_type=config.model_type,
            rolling_window=config.rolling_window_size,
            learning_rate=config.learning_rate,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def learn_one(
        self,
        features: dict[str, float],
        label: int,
    ) -> bool:
        """Update the model with a single labeled observation.

        Normalizes the features using the online scaler, updates the
        model, and tracks rolling performance metrics.

        Args:
            features: Dictionary mapping feature names to values.
            label: Binary ground-truth label (0 or 1).

        Returns:
            Whether the model's prediction (made before learning)
            was correct for this observation.
        """
        # Scale features online
        scaled = self._scale_features(features)
        self._scaler.learn_one(features)

        # Predict before learning (for metric tracking)
        predicted_label = self._predict_label(scaled)
        is_correct = predicted_label == label

        # Update the model
        self._model.learn_one(scaled, label)

        # Update metrics
        self._observation_count += 1
        self._rolling_correct.append(is_correct)

        self._accuracy_metric.update(label, predicted_label)
        self._f1_metric.update(label, predicted_label)
        self._rolling_accuracy_metric.update(label, predicted_label)

        if is_correct:
            self._total_correct += 1
            self._consecutive_correct += 1
            self._consecutive_wrong = 0
        else:
            self._consecutive_wrong += 1
            self._consecutive_correct = 0

        self._total_predictions += 1

        if self._observation_count % 100 == 0:
            self._log.debug(
                "online_learner_progress",
                observations=self._observation_count,
                rolling_accuracy=self._get_rolling_accuracy(),
                total_accuracy=round(
                    self._total_correct / max(self._total_predictions, 1),
                    4,
                ),
            )

        return is_correct

    def predict_one(
        self,
        features: dict[str, float],
    ) -> OnlinePrediction:
        """Predict the class label for a single observation.

        Args:
            features: Dictionary mapping feature names to values.

        Returns:
            An :class:`OnlinePrediction` containing the predicted
            label, confidence, and current model state metrics.
        """
        scaled = self._scale_features(features)
        label = self._predict_label(scaled)
        proba = self._predict_proba(scaled)

        return OnlinePrediction(
            label=label,
            confidence=proba,
            model_type=self._config.model_type,
            observation_count=self._observation_count,
            rolling_accuracy=self._get_rolling_accuracy(),
        )

    def predict_proba_one(
        self,
        features: dict[str, float],
    ) -> float:
        """Predict the positive-class probability for a single observation.

        Args:
            features: Dictionary mapping feature names to values.

        Returns:
            Predicted probability for the positive class (label=1),
            clipped to [0.01, 0.99].
        """
        scaled = self._scale_features(features)
        return self._predict_proba(scaled)

    def get_metrics(self) -> dict[str, float]:
        """Return current rolling and cumulative performance metrics.

        Returns:
            Dictionary containing ``rolling_accuracy``,
            ``overall_accuracy``, ``f1_score``,
            ``observation_count``, ``consecutive_correct``,
            and ``consecutive_wrong``.
        """
        overall_accuracy = self._total_correct / max(self._total_predictions, 1)

        return {
            "rolling_accuracy": self._get_rolling_accuracy(),
            "overall_accuracy": round(overall_accuracy, 4),
            "f1_score": round(
                self._f1_metric.get() if self._total_predictions > 0 else 0.0,
                4,
            ),
            "observation_count": float(self._observation_count),
            "consecutive_correct": float(self._consecutive_correct),
            "consecutive_wrong": float(self._consecutive_wrong),
        }

    def save_state(self, path: str) -> None:
        """Serialize the complete learner state to disk.

        Saves the model, scaler, metric history, and configuration
        as a pickle file for later restoration.

        Args:
            path: File path to save the state.  The ``.pkl``
                extension is conventional.
        """
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "model": self._model,
            "scaler": self._scaler,
            "config": self._config.model_dump(),
            "observation_count": self._observation_count,
            "total_correct": self._total_correct,
            "total_predictions": self._total_predictions,
            "consecutive_correct": self._consecutive_correct,
            "consecutive_wrong": self._consecutive_wrong,
            "current_learning_rate": self._current_learning_rate,
            "rolling_correct": list(self._rolling_correct),
            "accuracy_metric": self._accuracy_metric,
            "f1_metric": self._f1_metric,
            "rolling_accuracy_metric": self._rolling_accuracy_metric,
        }

        with open(save_path, "wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)

        self._log.info(
            "online_learner_state_saved",
            path=str(save_path),
            observations=self._observation_count,
        )

    def load_state(self, path: str) -> None:
        """Restore the learner state from a serialized file.

        Overwrites the current model, scaler, metrics, and
        configuration with the values from the saved state.

        Args:
            path: File path to load the state from.

        Raises:
            FileNotFoundError: If the state file does not exist.
        """
        load_path = Path(path)
        if not load_path.exists():
            raise FileNotFoundError(f"Learner state file not found: {path}")

        with open(load_path, "rb") as f:
            state = pickle.load(f)  # noqa: S301

        self._model = state["model"]
        self._scaler = state["scaler"]
        self._config = OnlineLearnerConfig(**state["config"])
        self._observation_count = state["observation_count"]
        self._total_correct = state["total_correct"]
        self._total_predictions = state["total_predictions"]
        self._consecutive_correct = state["consecutive_correct"]
        self._consecutive_wrong = state["consecutive_wrong"]
        self._current_learning_rate = state["current_learning_rate"]
        self._rolling_correct = deque(
            state["rolling_correct"],
            maxlen=self._config.rolling_window_size,
        )
        self._accuracy_metric = state["accuracy_metric"]
        self._f1_metric = state["f1_metric"]
        self._rolling_accuracy_metric = state["rolling_accuracy_metric"]

        self._log.info(
            "online_learner_state_loaded",
            path=str(load_path),
            observations=self._observation_count,
            rolling_accuracy=self._get_rolling_accuracy(),
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def observation_count(self) -> int:
        """Return the total number of observations processed."""
        return self._observation_count

    @property
    def current_learning_rate(self) -> float:
        """Return the current learning rate."""
        return self._current_learning_rate

    @current_learning_rate.setter
    def current_learning_rate(self, value: float) -> None:
        """Set the current learning rate.

        Args:
            value: New learning rate, must be positive.
        """
        if value <= 0.0:
            raise ValueError(f"Learning rate must be positive, got {value}")
        self._current_learning_rate = value

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    def _build_model(self) -> Any:
        """Create the River model instance based on configuration.

        Returns:
            A River classifier ready for incremental learning.

        Raises:
            ValueError: If the configured model_type is not recognized.
        """
        from river.ensemble import ADWINBaggingClassifier
        from river.tree import HoeffdingAdaptiveTreeClassifier

        if self._config.model_type == "hoeffding_adaptive_tree":
            return HoeffdingAdaptiveTreeClassifier(
                seed=self._config.seed,
            )

        if self._config.model_type == "adwin_bagging":
            base_model = HoeffdingAdaptiveTreeClassifier(
                seed=self._config.seed,
            )
            return ADWINBaggingClassifier(
                model=base_model,
                n_models=self._config.n_ensemble_members,
                seed=self._config.seed,
            )

        raise ValueError(
            f"Unknown model_type '{self._config.model_type}'. "
            f"Supported: 'hoeffding_adaptive_tree', 'adwin_bagging'"
        )

    @staticmethod
    def _build_scaler() -> Any:
        """Create an online feature normalizer.

        Returns:
            A River StandardScaler for incremental normalization.
        """
        from river.preprocessing import StandardScaler

        return StandardScaler()

    @staticmethod
    def _build_accuracy_metric() -> Any:
        """Create a River Accuracy metric tracker.

        Returns:
            A River Accuracy metric instance.
        """
        from river.metrics import Accuracy

        return Accuracy()

    @staticmethod
    def _build_f1_metric() -> Any:
        """Create a River F1 metric tracker.

        Returns:
            A River F1 metric instance.
        """
        from river.metrics import F1

        return F1()

    def _build_rolling_accuracy_metric(self) -> Any:
        """Create a River RollingAccuracy metric tracker.

        Returns:
            A River Rolling metric wrapping Accuracy with the
            configured window size.
        """
        from river.metrics import Accuracy, Rolling

        return Rolling(
            metric=Accuracy(),
            window_size=self._config.rolling_window_size,
        )

    def _scale_features(
        self,
        features: dict[str, float],
    ) -> dict[str, float]:
        """Apply online StandardScaler to feature values.

        The scaler transforms features based on running mean and
        variance estimates.  On the very first observation the scaler
        returns the raw features unchanged (no statistics yet).

        Args:
            features: Raw feature dictionary.

        Returns:
            Scaled feature dictionary.
        """
        if self._observation_count == 0:
            return dict(features)
        return self._scaler.transform_one(features)

    def _predict_label(
        self,
        scaled_features: dict[str, float],
    ) -> int:
        """Predict the class label from scaled features.

        When the model has not yet seen any data, defaults to
        predicting class 0.

        Args:
            scaled_features: Normalized feature dictionary.

        Returns:
            Predicted class label (0 or 1).
        """
        if self._observation_count == 0:
            return 0
        prediction = self._model.predict_one(scaled_features)
        if prediction is None:
            return 0
        return int(prediction)

    def _predict_proba(
        self,
        scaled_features: dict[str, float],
    ) -> float:
        """Predict the positive-class probability from scaled features.

        Clips the result to [CONFIDENCE_FLOOR, CONFIDENCE_CEILING]
        to prevent degenerate probability outputs.  When the model
        has not yet seen any data, returns 0.5.

        Args:
            scaled_features: Normalized feature dictionary.

        Returns:
            Probability for the positive class (label=1).
        """
        if self._observation_count == 0:
            return 0.5

        proba_dict = self._model.predict_proba_one(scaled_features)
        if proba_dict is None or len(proba_dict) == 0:
            return 0.5

        positive_proba = proba_dict.get(1, proba_dict.get(True, 0.5))
        return max(
            CONFIDENCE_FLOOR,
            min(CONFIDENCE_CEILING, float(positive_proba)),
        )

    def _get_rolling_accuracy(self) -> float:
        """Compute the rolling accuracy over the recent window.

        Returns:
            Fraction of correct predictions in the rolling window.
            Returns 0.0 if no predictions have been made yet.
        """
        if not self._rolling_correct:
            return 0.0
        return round(
            sum(self._rolling_correct) / len(self._rolling_correct),
            4,
        )

    def reset_model(self) -> None:
        """Reset the model to a fresh state, preserving configuration.

        Called when severe drift is detected and the model weights
        need to be cleared.  Metrics and observation count are
        preserved so that drift history remains intact.
        """
        self._model = self._build_model()
        self._scaler = self._build_scaler()
        self._rolling_correct.clear()
        self._consecutive_correct = 0
        self._consecutive_wrong = 0

        self._log.info(
            "online_learner_model_reset",
            observations_at_reset=self._observation_count,
        )


# ---------------------------------------------------------------------------
# DriftDetector
# ---------------------------------------------------------------------------


class DriftDetector:
    """ADWIN-based concept drift detector with graduated response system.

    Monitors a stream of prediction errors and uses the ADWIN
    (Adaptive Windowing) algorithm to detect distributional changes.
    When drift is detected, the severity is assessed based on the
    frequency of recent drift events and a corresponding action is
    recommended.

    The graduated response ladder:
        1. Minor drift (1 event in window)   -> LOG_WARNING
        2. Moderate drift (3 events)         -> INCREASE_LR
        3. Severe drift (5 events)           -> RESET_WEIGHTS
        4. Persistent drift (8+ events)      -> FULL_RETRAIN

    Args:
        adwin_delta: Sensitivity of the ADWIN detector.  Smaller
            values produce fewer false positives but may delay
            detection of genuine drift.
        detection_window: Number of recent observations over which
            drift event frequency is counted for the graduated
            response.
    """

    def __init__(
        self,
        adwin_delta: float = 0.002,
        detection_window: int = DRIFT_DETECTION_WINDOW,
    ) -> None:
        self._log: structlog.stdlib.BoundLogger = get_logger("ml.online.drift")

        from river.drift import ADWIN

        self._adwin: Any = ADWIN(delta=adwin_delta)
        self._adwin_delta: float = adwin_delta
        self._detection_window: int = detection_window

        # Drift tracking state
        self._drift_history: deque[DriftEvent] = deque(
            maxlen=MAX_DRIFT_HISTORY,
        )
        self._recent_drift_timestamps: deque[int] = deque(
            maxlen=detection_window,
        )
        self._observation_count: int = 0
        self._last_drift_detected: bool = False

        self._log.info(
            "drift_detector_initialized",
            adwin_delta=adwin_delta,
            detection_window=detection_window,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, error: float) -> DriftEvent | None:
        """Feed a new error value and check for concept drift.

        The error value should be 0.0 for a correct prediction and
        1.0 for an incorrect prediction (binary error).  ADWIN
        maintains an adaptive window and detects changes in the
        error distribution.

        Args:
            error: Prediction error for the current observation.
                Typically 0.0 (correct) or 1.0 (incorrect).

        Returns:
            A :class:`DriftEvent` if drift was detected on this
            observation, otherwise ``None``.
        """
        self._observation_count += 1

        # Feed error to ADWIN
        self._adwin.update(error)
        self._last_drift_detected = self._adwin.drift_detected

        if not self._last_drift_detected:
            return None

        # Drift detected — record it and determine severity
        self._recent_drift_timestamps.append(
            self._observation_count,
        )

        # Count drift events within the detection window
        cutoff = self._observation_count - self._detection_window
        recent_count = sum(1 for ts in self._recent_drift_timestamps if ts > cutoff)

        action = self._determine_action(recent_count)
        current_error_rate = self._estimate_error_rate()

        event = DriftEvent(
            timestamp=datetime.now(UTC),
            action=action,
            error_rate=round(current_error_rate, 4),
            adwin_width=float(self._adwin.width),
            observation_count=self._observation_count,
            details=(
                f"Drift detected at observation "
                f"{self._observation_count}; "
                f"{recent_count} events in last "
                f"{self._detection_window} observations; "
                f"action={action.value}"
            ),
        )

        self._drift_history.append(event)

        self._log.warning(
            "concept_drift_detected",
            action=action.value,
            error_rate=event.error_rate,
            adwin_width=event.adwin_width,
            recent_drift_count=recent_count,
            observation=self._observation_count,
        )

        return event

    def check_drift(self) -> DriftEvent | None:
        """Return the most recent drift event if drift was detected.

        This is a non-consuming check that returns the last drift
        event recorded by :meth:`update`.  Useful for polling the
        detector state without feeding new data.

        Returns:
            The most recent :class:`DriftEvent` if drift was
            detected on the last ``update`` call, otherwise ``None``.
        """
        if not self._last_drift_detected:
            return None
        if not self._drift_history:
            return None
        return self._drift_history[-1]

    def get_drift_history(self) -> list[DriftEvent]:
        """Return the complete history of detected drift events.

        Returns:
            List of :class:`DriftEvent` instances ordered
            chronologically, up to :data:`MAX_DRIFT_HISTORY`
            most recent events.
        """
        return list(self._drift_history)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def total_drift_events(self) -> int:
        """Return the total number of drift events detected."""
        return len(self._drift_history)

    @property
    def observation_count(self) -> int:
        """Return the total observations processed by the detector."""
        return self._observation_count

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    def _determine_action(self, recent_count: int) -> DriftAction:
        """Map the recent drift event count to a graduated action.

        The response escalates with the frequency of drift events
        within the detection window.

        Args:
            recent_count: Number of drift events within the
                detection window.

        Returns:
            The appropriate :class:`DriftAction` for the severity.
        """
        if recent_count >= PERSISTENT_DRIFT_THRESHOLD:
            return DriftAction.FULL_RETRAIN
        if recent_count >= SEVERE_DRIFT_THRESHOLD:
            return DriftAction.RESET_WEIGHTS
        if recent_count >= MODERATE_DRIFT_THRESHOLD:
            return DriftAction.INCREASE_LR
        return DriftAction.LOG_WARNING

    def _estimate_error_rate(self) -> float:
        """Estimate the current error rate from the ADWIN window.

        Uses the ADWIN estimator's running mean as a proxy for the
        error rate within the current window.

        Returns:
            Estimated error rate in [0.0, 1.0].
        """
        try:
            return float(self._adwin.estimation)
        except (AttributeError, TypeError):
            return 0.0


# ---------------------------------------------------------------------------
# OnlineModelManager
# ---------------------------------------------------------------------------


class OnlineModelManager:
    """Orchestrates online learning, drift detection, and response actions.

    Combines an :class:`OnlineLearner` with a :class:`DriftDetector`
    to provide a single interface for incremental model updates,
    predictions, and automated drift response.  When drift is
    detected, the manager automatically executes the graduated
    response: logging, learning rate adjustment, model reset, or
    signaling for full retraining.

    Args:
        config: Configuration for the online learning system.
    """

    def __init__(
        self,
        config: OnlineLearnerConfig | None = None,
    ) -> None:
        self._config: OnlineLearnerConfig = config or OnlineLearnerConfig()
        self._log: structlog.stdlib.BoundLogger = get_logger("ml.online.manager")

        # Core components
        self._learner: OnlineLearner = OnlineLearner(self._config)
        self._drift_detector: DriftDetector = DriftDetector(
            adwin_delta=self._config.adwin_delta,
            detection_window=self._config.drift_detection_window,
        )

        # Manager-level tracking
        self._last_drift_event: DriftEvent | None = None
        self._full_retrain_requested: bool = False
        self._learning_rate_history: deque[tuple[datetime, float]] = deque(
            maxlen=MAX_DRIFT_HISTORY
        )
        self._created_at: datetime = datetime.now(UTC)

        # Record initial learning rate
        self._learning_rate_history.append(
            (self._created_at, self._config.learning_rate),
        )

        self._log.info(
            "online_model_manager_initialized",
            model_type=self._config.model_type,
            adwin_delta=self._config.adwin_delta,
            learning_rate=self._config.learning_rate,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def update(
        self,
        features: dict[str, float],
        label: int,
    ) -> DriftEvent | None:
        """Learn from a single observation and check for drift.

        Updates the online model with the new labeled observation,
        feeds the prediction error to the drift detector, and
        automatically executes any required drift response action.

        Args:
            features: Dictionary mapping feature names to values.
            label: Ground-truth binary label (0 or 1).

        Returns:
            A :class:`DriftEvent` if drift was detected, otherwise
            ``None``.
        """
        start_time = time.monotonic()

        # Learn and get correctness
        is_correct = self._learner.learn_one(features, label)

        # Feed error to drift detector
        error = 0.0 if is_correct else 1.0
        drift_event = self._drift_detector.update(error)

        # Execute drift response if needed
        if drift_event is not None:
            self._last_drift_event = drift_event
            self._execute_drift_action(drift_event)

        elapsed_ms = (time.monotonic() - start_time) * 1000

        if self._learner.observation_count % 100 == 0:
            self._log.debug(
                "online_update_batch_progress",
                observations=self._learner.observation_count,
                rolling_accuracy=(self._learner.get_metrics()["rolling_accuracy"]),
                drift_events=self._drift_detector.total_drift_events,
                latency_ms=round(elapsed_ms, 2),
            )

        return drift_event

    async def predict(
        self,
        features: dict[str, float],
    ) -> OnlinePrediction:
        """Generate a prediction from the current online model.

        Args:
            features: Dictionary mapping feature names to values.

        Returns:
            An :class:`OnlinePrediction` with the label, confidence,
            and current model health indicators.
        """
        prediction = self._learner.predict_one(features)

        # Annotate with drift status
        prediction.drift_detected = (
            self._last_drift_event is not None
            and self._drift_detector.observation_count
            - self._last_drift_event.observation_count
            < 10
        )

        return prediction

    def get_model_health(self) -> dict[str, Any]:
        """Return a comprehensive health report for the online model.

        Returns:
            Dictionary containing accuracy metrics, drift detection
            statistics, learning rate history, and timing information.
        """
        metrics = self._learner.get_metrics()
        drift_history = self._drift_detector.get_drift_history()

        last_drift_at: str | None = None
        last_drift_action: str | None = None
        if drift_history:
            last_event = drift_history[-1]
            last_drift_at = last_event.timestamp.isoformat()
            last_drift_action = last_event.action.value

        # Count actions by type
        action_counts: dict[str, int] = {action.value: 0 for action in DriftAction}
        for event in drift_history:
            action_counts[event.action.value] += 1

        return {
            "model_type": self._config.model_type,
            "observation_count": int(metrics["observation_count"]),
            "rolling_accuracy": metrics["rolling_accuracy"],
            "overall_accuracy": metrics["overall_accuracy"],
            "f1_score": metrics["f1_score"],
            "consecutive_correct": int(metrics["consecutive_correct"]),
            "consecutive_wrong": int(metrics["consecutive_wrong"]),
            "current_learning_rate": (self._learner.current_learning_rate),
            "drift_event_count": (self._drift_detector.total_drift_events),
            "last_drift_at": last_drift_at,
            "last_drift_action": last_drift_action,
            "drift_action_counts": action_counts,
            "full_retrain_requested": self._full_retrain_requested,
            "created_at": self._created_at.isoformat(),
            "uptime_seconds": round(
                (datetime.now(UTC) - self._created_at).total_seconds(),
                1,
            ),
        }

    def save_state(self, directory: str) -> str:
        """Save the complete manager state to a directory.

        Creates two files: one for the learner state and one for
        the manager metadata (drift history, learning rate history,
        configuration).

        Args:
            directory: Directory in which to save state files.

        Returns:
            Path to the directory containing saved state files.
        """
        save_dir = Path(directory)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save learner state
        learner_path = save_dir / f"learner{STATE_FILE_EXTENSION}"
        self._learner.save_state(str(learner_path))

        # Save manager metadata
        manager_state = {
            "config": self._config.model_dump(),
            "drift_history": [
                event.model_dump(mode="json")
                for event in self._drift_detector.get_drift_history()
            ],
            "learning_rate_history": [
                (ts.isoformat(), lr) for ts, lr in self._learning_rate_history
            ],
            "full_retrain_requested": self._full_retrain_requested,
            "created_at": self._created_at.isoformat(),
            "drift_detector_observations": (self._drift_detector.observation_count),
        }

        meta_path = save_dir / "manager_metadata.json"
        with open(meta_path, "w") as f:
            json.dump(manager_state, f, indent=2, default=str)

        # Save drift detector state
        detector_path = save_dir / f"drift_detector{STATE_FILE_EXTENSION}"
        with open(detector_path, "wb") as f:
            pickle.dump(
                self._drift_detector,
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

        self._log.info(
            "online_manager_state_saved",
            directory=str(save_dir),
            observations=self._learner.observation_count,
            drift_events=self._drift_detector.total_drift_events,
        )

        return str(save_dir)

    def load_state(self, directory: str) -> None:
        """Restore the complete manager state from a directory.

        Loads the learner model, drift detector, and manager
        metadata from previously saved files.

        Args:
            directory: Directory containing saved state files.

        Raises:
            FileNotFoundError: If the directory or required files
                do not exist.
        """
        load_dir = Path(directory)
        if not load_dir.exists():
            raise FileNotFoundError(f"State directory not found: {directory}")

        # Load learner state
        learner_path = load_dir / f"learner{STATE_FILE_EXTENSION}"
        if learner_path.exists():
            self._learner.load_state(str(learner_path))

        # Load drift detector
        detector_path = load_dir / f"drift_detector{STATE_FILE_EXTENSION}"
        if detector_path.exists():
            with open(detector_path, "rb") as f:
                self._drift_detector = pickle.load(f)  # noqa: S301

        # Load manager metadata
        meta_path = load_dir / "manager_metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                manager_state = json.load(f)

            self._config = OnlineLearnerConfig(**manager_state["config"])
            self._full_retrain_requested = manager_state.get(
                "full_retrain_requested", False
            )

            created_at_str = manager_state.get("created_at")
            if created_at_str:
                self._created_at = datetime.fromisoformat(created_at_str)

            # Restore learning rate history
            lr_history = manager_state.get("learning_rate_history", [])
            self._learning_rate_history = deque(maxlen=MAX_DRIFT_HISTORY)
            for ts_str, lr in lr_history:
                self._learning_rate_history.append(
                    (datetime.fromisoformat(ts_str), lr),
                )

        self._log.info(
            "online_manager_state_loaded",
            directory=str(load_dir),
            observations=self._learner.observation_count,
            drift_events=self._drift_detector.total_drift_events,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def full_retrain_requested(self) -> bool:
        """Whether persistent drift has triggered a full retrain request."""
        return self._full_retrain_requested

    @property
    def learner(self) -> OnlineLearner:
        """Return the underlying OnlineLearner instance."""
        return self._learner

    @property
    def drift_detector(self) -> DriftDetector:
        """Return the underlying DriftDetector instance."""
        return self._drift_detector

    def acknowledge_retrain(self) -> None:
        """Clear the full retrain request flag.

        Called after the batch training pipeline has completed a
        full retraining cycle in response to persistent drift.
        """
        self._full_retrain_requested = False
        self._log.info(
            "full_retrain_acknowledged",
            observations=self._learner.observation_count,
        )

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    def _execute_drift_action(
        self,
        drift_event: DriftEvent,
    ) -> None:
        """Execute the graduated drift response action.

        Applies the recommended action from the drift event:
        logging a warning, increasing the learning rate, resetting
        model weights, or signaling for full retraining.

        Args:
            drift_event: The drift event containing the action
                to execute.
        """
        action = drift_event.action

        if action == DriftAction.LOG_WARNING:
            self._log.warning(
                "drift_action_log_warning",
                error_rate=drift_event.error_rate,
                observation=drift_event.observation_count,
                message=("Minor concept drift detected; monitoring continues"),
            )

        elif action == DriftAction.INCREASE_LR:
            old_lr = self._learner.current_learning_rate
            new_lr = INCREASED_LEARNING_RATE
            self._learner.current_learning_rate = new_lr

            self._learning_rate_history.append(
                (datetime.now(UTC), new_lr),
            )

            self._log.warning(
                "drift_action_increase_lr",
                old_lr=old_lr,
                new_lr=new_lr,
                error_rate=drift_event.error_rate,
                observation=drift_event.observation_count,
            )

        elif action == DriftAction.RESET_WEIGHTS:
            self._learner.reset_model()

            # Reset learning rate to default
            self._learner.current_learning_rate = self._config.learning_rate
            self._learning_rate_history.append(
                (datetime.now(UTC), self._config.learning_rate),
            )

            self._log.warning(
                "drift_action_reset_weights",
                error_rate=drift_event.error_rate,
                observation=drift_event.observation_count,
                message=("Severe drift detected; model weights reset to initial state"),
            )

        elif action == DriftAction.FULL_RETRAIN:
            self._full_retrain_requested = True

            # Also reset the online model while waiting for retrain
            self._learner.reset_model()
            self._learner.current_learning_rate = self._config.learning_rate
            self._learning_rate_history.append(
                (datetime.now(UTC), self._config.learning_rate),
            )

            self._log.error(
                "drift_action_full_retrain_requested",
                error_rate=drift_event.error_rate,
                observation=drift_event.observation_count,
                total_drift_events=(self._drift_detector.total_drift_events),
                message=("Persistent drift detected; full batch retraining requested"),
            )
