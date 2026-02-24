"""Isotonic regression probability calibration for Project Titan.

Transforms raw model-predicted probabilities into well-calibrated
probabilities using isotonic regression.  Well-calibrated probabilities
ensure that when the model says "78% confidence", approximately 78% of
those predictions are correct — a critical property for the
confidence-threshold trading gate.

Supports calibration evaluation via Expected Calibration Error (ECE),
Maximum Calibration Error (MCE), and Brier score, along with reliability
diagram data for visualization.

Usage::

    from src.ml.calibration import ProbabilityCalibrator

    calibrator = ProbabilityCalibrator(method="isotonic")
    calibrator.fit(y_true, y_pred_proba)
    calibrated = calibrator.calibrate(raw_proba)
    metrics = calibrator.evaluate_calibration(y_true, y_pred_proba)
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import joblib
import numpy as np
from pydantic import BaseModel, Field
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss

from src.utils.logging import get_logger

if TYPE_CHECKING:
    import structlog


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROBABILITY_FLOOR: float = 0.01
"""Minimum calibrated probability to avoid degenerate 0.0 outputs."""

PROBABILITY_CEILING: float = 0.99
"""Maximum calibrated probability to avoid degenerate 1.0 outputs."""


# ---------------------------------------------------------------------------
# Pydantic result model
# ---------------------------------------------------------------------------


class CalibrationMetrics(BaseModel):
    """Metrics quantifying the quality of probability calibration.

    Attributes:
        ece: Expected Calibration Error — the weighted average of
            per-bin calibration gaps.
        mce: Maximum Calibration Error — the largest per-bin
            calibration gap.
        brier_score: Brier score (mean squared error of probabilities).
        n_bins: Number of bins used for the reliability diagram.
        bin_edges: Left edges of each calibration bin.
        bin_accuracies: Observed accuracy (fraction of positives) in
            each bin.
        bin_counts: Number of samples in each bin.
    """

    ece: float = Field(description="Expected Calibration Error")
    mce: float = Field(description="Maximum Calibration Error")
    brier_score: float = Field(description="Brier score (MSE of probabilities)")
    n_bins: int = Field(description="Number of calibration bins")
    bin_edges: list[float] = Field(description="Left edges of calibration bins")
    bin_accuracies: list[float] = Field(
        description="Observed positive rate per bin",
    )
    bin_counts: list[int] = Field(description="Sample count per bin")


# ---------------------------------------------------------------------------
# ProbabilityCalibrator
# ---------------------------------------------------------------------------


class ProbabilityCalibrator:
    """Calibrate raw model probabilities using isotonic or sigmoid methods.

    After fitting on a held-out calibration set, the ``calibrate`` method
    maps raw probabilities through a learned monotonic function that
    minimizes calibration error.

    Args:
        method: Calibration method.  ``"isotonic"`` for isotonic
            regression (default, non-parametric).  ``"sigmoid"`` for
            Platt scaling (parametric logistic).
    """

    def __init__(self, method: str = "isotonic") -> None:
        self._method: str = method
        self._log: structlog.stdlib.BoundLogger = get_logger("ml.calibration")
        self._fitted: bool = False

        if method == "isotonic":
            self._calibrator: IsotonicRegression = IsotonicRegression(
                y_min=PROBABILITY_FLOOR,
                y_max=PROBABILITY_CEILING,
                out_of_bounds="clip",
            )
        elif method == "sigmoid":
            # Platt scaling: use isotonic regression as a fallback container;
            # actual sigmoid fitting is done in _fit_sigmoid.
            self._calibrator = IsotonicRegression(
                y_min=PROBABILITY_FLOOR,
                y_max=PROBABILITY_CEILING,
                out_of_bounds="clip",
            )
            self._sigmoid_a: float = 0.0
            self._sigmoid_b: float = 0.0
        else:
            raise ValueError(
                f"Unknown calibration method '{method}'. "
                f"Supported: 'isotonic', 'sigmoid'"
            )

        self._log.info(
            "probability_calibrator_initialized",
            method=method,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
    ) -> None:
        """Fit the calibration mapping on predicted probabilities.

        Args:
            y_true: Ground-truth binary labels (0 or 1).
            y_pred_proba: Raw predicted probabilities for the positive
                class, in the range [0, 1].

        Raises:
            ValueError: If inputs have different lengths or contain
                invalid values.
        """
        y_true = np.asarray(y_true, dtype=np.float64)
        y_pred_proba = np.asarray(y_pred_proba, dtype=np.float64)

        if len(y_true) != len(y_pred_proba):
            raise ValueError(
                f"y_true and y_pred_proba must have the same length: "
                f"{len(y_true)} != {len(y_pred_proba)}"
            )

        if len(y_true) < 10:
            raise ValueError(
                f"At least 10 samples are required for calibration; got {len(y_true)}"
            )

        self._log.info(
            "fitting_calibrator",
            method=self._method,
            n_samples=len(y_true),
            positive_rate=round(float(y_true.mean()), 4),
        )

        if self._method == "isotonic":
            self._calibrator.fit(y_pred_proba, y_true)
        elif self._method == "sigmoid":
            self._fit_sigmoid(y_true, y_pred_proba)

        self._fitted = True

        self._log.info(
            "calibrator_fitted",
            method=self._method,
            n_samples=len(y_true),
        )

    def calibrate(
        self,
        raw_proba: float | np.ndarray,
    ) -> float | np.ndarray:
        """Transform raw probabilities through the calibration function.

        Args:
            raw_proba: A single probability value or an array of
                probabilities in [0, 1].

        Returns:
            Calibrated probability (or array of probabilities) clipped
            to [0.01, 0.99].

        Raises:
            RuntimeError: If ``fit`` has not been called yet.
        """
        if not self._fitted:
            raise RuntimeError("Calibrator has not been fitted. Call fit() first.")

        scalar_input = isinstance(raw_proba, (int, float))

        if scalar_input:
            raw_array = np.array([float(raw_proba)])
        else:
            raw_array = np.asarray(raw_proba, dtype=np.float64)

        if self._method == "isotonic":
            calibrated = self._calibrator.predict(raw_array)
        elif self._method == "sigmoid":
            calibrated = self._apply_sigmoid(raw_array)
        else:
            calibrated = raw_array

        # Clip to safe probability range
        calibrated = np.clip(calibrated, PROBABILITY_FLOOR, PROBABILITY_CEILING)

        if scalar_input:
            return float(calibrated[0])

        return calibrated

    def save(self, path: str) -> None:
        """Serialize the calibrator to disk.

        Args:
            path: File path to save the calibrator.  The ``.joblib``
                extension is conventional.

        Raises:
            RuntimeError: If the calibrator has not been fitted.
        """
        if not self._fitted:
            raise RuntimeError("Cannot save unfitted calibrator. Call fit() first.")

        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "method": self._method,
            "calibrator": self._calibrator,
            "fitted": self._fitted,
        }
        if self._method == "sigmoid":
            state["sigmoid_a"] = self._sigmoid_a
            state["sigmoid_b"] = self._sigmoid_b

        joblib.dump(state, save_path)

        self._log.info(
            "calibrator_saved",
            path=str(save_path),
            method=self._method,
        )

    def load(self, path: str) -> None:
        """Load a calibrator from disk.

        Args:
            path: File path from which to load the calibrator.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        load_path = Path(path)
        if not load_path.exists():
            raise FileNotFoundError(f"Calibrator file not found: {path}")

        state = joblib.load(load_path)

        self._method = state["method"]
        self._calibrator = state["calibrator"]
        self._fitted = state["fitted"]

        if self._method == "sigmoid":
            self._sigmoid_a = state.get("sigmoid_a", 0.0)
            self._sigmoid_b = state.get("sigmoid_b", 0.0)

        self._log.info(
            "calibrator_loaded",
            path=str(load_path),
            method=self._method,
            fitted=self._fitted,
        )

    def evaluate_calibration(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        n_bins: int = 10,
    ) -> CalibrationMetrics:
        """Evaluate calibration quality with ECE, MCE, and Brier score.

        Computes the reliability diagram data (bin edges, observed
        accuracies, and bin counts) along with summary calibration
        metrics.

        Args:
            y_true: Ground-truth binary labels.
            y_pred_proba: Predicted probabilities for the positive class.
                If the calibrator has been fitted, these should be the
                *calibrated* probabilities for a meaningful evaluation.
            n_bins: Number of equally-spaced bins for the reliability
                diagram.

        Returns:
            A :class:`CalibrationMetrics` instance with all computed
            metrics and reliability diagram data.
        """
        y_true = np.asarray(y_true, dtype=np.float64)
        y_pred_proba = np.asarray(y_pred_proba, dtype=np.float64)

        self._log.info(
            "evaluating_calibration",
            n_samples=len(y_true),
            n_bins=n_bins,
        )

        # Brier score
        brier = float(brier_score_loss(y_true, y_pred_proba))

        # Reliability diagram via sklearn
        try:
            fraction_of_positives, mean_predicted_value = calibration_curve(
                y_true,
                y_pred_proba,
                n_bins=n_bins,
                strategy="uniform",
            )
        except ValueError:
            self._log.warning(
                "calibration_curve_failed",
                reason="insufficient_data_or_degenerate_predictions",
            )
            return CalibrationMetrics(
                ece=1.0,
                mce=1.0,
                brier_score=brier,
                n_bins=n_bins,
                bin_edges=[],
                bin_accuracies=[],
                bin_counts=[],
            )

        # Compute bin-level statistics manually for ECE/MCE
        bin_edges_arr = np.linspace(0.0, 1.0, n_bins + 1)
        bin_accuracies: list[float] = []
        bin_counts: list[int] = []
        bin_edges_list: list[float] = []
        gaps: list[float] = []

        for i in range(n_bins):
            bin_lower = bin_edges_arr[i]
            bin_upper = bin_edges_arr[i + 1]

            # Select samples in this bin
            if i < n_bins - 1:
                in_bin = (y_pred_proba >= bin_lower) & (y_pred_proba < bin_upper)
            else:
                # Include the right edge for the last bin
                in_bin = (y_pred_proba >= bin_lower) & (y_pred_proba <= bin_upper)

            count = int(in_bin.sum())
            bin_counts.append(count)
            bin_edges_list.append(round(float(bin_lower), 4))

            if count > 0:
                avg_confidence = float(y_pred_proba[in_bin].mean())
                avg_accuracy = float(y_true[in_bin].mean())
                bin_accuracies.append(round(avg_accuracy, 4))
                gap = abs(avg_accuracy - avg_confidence)
                gaps.append(gap * count)
            else:
                bin_accuracies.append(0.0)
                gaps.append(0.0)

        total_samples = sum(bin_counts)

        # Expected Calibration Error (weighted average gap)
        ece = sum(gaps) / max(total_samples, 1)

        # Maximum Calibration Error (largest per-bin gap)
        per_bin_gaps: list[float] = []
        for i in range(n_bins):
            if bin_counts[i] > 0:
                bin_lower = bin_edges_arr[i]
                bin_upper = bin_edges_arr[i + 1]
                if i < n_bins - 1:
                    in_bin = (y_pred_proba >= bin_lower) & (y_pred_proba < bin_upper)
                else:
                    in_bin = (y_pred_proba >= bin_lower) & (y_pred_proba <= bin_upper)
                avg_confidence = float(y_pred_proba[in_bin].mean())
                gap = abs(bin_accuracies[i] - avg_confidence)
                per_bin_gaps.append(gap)

        mce = max(per_bin_gaps) if per_bin_gaps else 0.0

        metrics = CalibrationMetrics(
            ece=round(ece, 4),
            mce=round(mce, 4),
            brier_score=round(brier, 4),
            n_bins=n_bins,
            bin_edges=bin_edges_list,
            bin_accuracies=bin_accuracies,
            bin_counts=bin_counts,
        )

        self._log.info(
            "calibration_evaluated",
            ece=metrics.ece,
            mce=metrics.mce,
            brier_score=metrics.brier_score,
            n_bins=metrics.n_bins,
        )

        return metrics

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_fitted(self) -> bool:
        """Return whether the calibrator has been fitted."""
        return self._fitted

    @property
    def method(self) -> str:
        """Return the calibration method name."""
        return self._method

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    def _fit_sigmoid(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
    ) -> None:
        """Fit Platt scaling (sigmoid) calibration.

        Fits a logistic regression of the form::

            calibrated = 1 / (1 + exp(a * raw + b))

        Parameters *a* and *b* are optimized via maximum likelihood.

        Args:
            y_true: Ground-truth binary labels.
            y_pred_proba: Raw predicted probabilities.
        """
        from scipy.optimize import minimize

        def _neg_log_likelihood(params: np.ndarray) -> float:
            """Negative log-likelihood for sigmoid calibration."""
            a, b = params
            p = 1.0 / (1.0 + np.exp(a * y_pred_proba + b))
            p = np.clip(p, 1e-10, 1.0 - 1e-10)
            nll = -np.mean(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p))
            return float(nll)

        result = minimize(
            _neg_log_likelihood,
            x0=np.array([0.0, 0.0]),
            method="Nelder-Mead",
            options={"maxiter": 1000},
        )

        self._sigmoid_a = float(result.x[0])
        self._sigmoid_b = float(result.x[1])

        # Also fit the isotonic calibrator as a fallback representation
        calibrated_train = self._apply_sigmoid(y_pred_proba)
        self._calibrator.fit(y_pred_proba, calibrated_train)

        self._log.debug(
            "sigmoid_parameters_fitted",
            a=round(self._sigmoid_a, 6),
            b=round(self._sigmoid_b, 6),
            final_nll=round(result.fun, 6),
        )

    def _apply_sigmoid(self, raw_proba: np.ndarray) -> np.ndarray:
        """Apply the fitted sigmoid calibration function.

        Args:
            raw_proba: Array of raw probabilities.

        Returns:
            Array of sigmoid-calibrated probabilities.
        """
        calibrated = 1.0 / (1.0 + np.exp(self._sigmoid_a * raw_proba + self._sigmoid_b))
        return np.clip(calibrated, PROBABILITY_FLOOR, PROBABILITY_CEILING)
