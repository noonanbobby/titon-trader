"""Walk-forward training pipeline for Project Titan.

Implements purged k-fold cross-validation with embargo periods to prevent
information leakage, supports multiple gradient boosting backends
(XGBoost, LightGBM, CatBoost), and handles model serialization and
evaluation.

Walk-forward validation trains on earlier data and tests on later data
within each fold, with an embargo gap between train and test to prevent
lookahead bias from autocorrelated features.

Usage::

    from src.ml.trainer import WalkForwardTrainer

    trainer = WalkForwardTrainer(n_splits=5, embargo_days=5)
    result = trainer.train(X, y, model_type="xgboost")
    metrics = trainer.evaluate(model, X_test, y_test)
"""

from __future__ import annotations

import json
import time
from datetime import UTC, date, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.utils.logging import get_logger

if TYPE_CHECKING:
    import structlog


# ---------------------------------------------------------------------------
# Pydantic result models
# ---------------------------------------------------------------------------


class FoldMetrics(BaseModel):
    """Metrics from a single cross-validation fold.

    Attributes:
        fold_num: Zero-based fold index.
        train_size: Number of training samples in this fold.
        val_size: Number of validation samples in this fold.
        accuracy: Classification accuracy on the validation set.
        auc_roc: Area under the ROC curve on the validation set.
        log_loss: Logarithmic loss on the validation set.
        feature_importances: Mapping of feature name to importance score.
    """

    fold_num: int = Field(description="Zero-based fold index")
    train_size: int = Field(description="Number of training samples")
    val_size: int = Field(description="Number of validation samples")
    accuracy: float = Field(description="Validation accuracy")
    auc_roc: float = Field(description="Validation AUC-ROC")
    log_loss: float = Field(description="Validation log loss")
    feature_importances: dict[str, float] = Field(
        default_factory=dict,
        description="Feature name to importance mapping",
    )


class TrainResult(BaseModel):
    """Aggregated results from walk-forward training.

    Attributes:
        model_type: Name of the model backend used.
        n_folds: Number of cross-validation folds.
        fold_metrics: Per-fold metric details.
        avg_accuracy: Mean accuracy across all folds.
        avg_auc: Mean AUC-ROC across all folds.
        best_fold: Index of the fold with the highest AUC-ROC.
        feature_importances: Average feature importances across folds.
        train_time_seconds: Total wall-clock training time in seconds.
    """

    model_type: str = Field(description="Model backend name")
    n_folds: int = Field(description="Number of CV folds")
    fold_metrics: list[FoldMetrics] = Field(description="Per-fold metrics")
    avg_accuracy: float = Field(description="Mean accuracy across folds")
    avg_auc: float = Field(description="Mean AUC-ROC across folds")
    best_fold: int = Field(description="Index of best fold by AUC-ROC")
    feature_importances: dict[str, float] = Field(
        default_factory=dict,
        description="Average feature importances across folds",
    )
    train_time_seconds: float = Field(
        description="Total training wall-clock time in seconds",
    )


class ModelMetadata(BaseModel):
    """Metadata for a serialized model artifact.

    Persisted as a JSON sidecar file alongside the model binary.

    Attributes:
        model_name: Descriptive model name.
        version: Monotonically increasing version number.
        trained_at: UTC timestamp when training completed.
        train_start: Earliest date in the training data.
        train_end: Latest date in the training data.
        n_features: Number of input features.
        feature_names: Ordered list of feature names.
        hyperparams: Hyperparameter dictionary used for training.
        val_accuracy: Validation accuracy from cross-validation.
        val_auc: Validation AUC-ROC from cross-validation.
    """

    model_name: str = Field(description="Descriptive model name")
    version: int = Field(description="Model version number")
    trained_at: datetime = Field(description="UTC timestamp of training completion")
    train_start: date = Field(description="Earliest date in training data")
    train_end: date = Field(description="Latest date in training data")
    n_features: int = Field(description="Number of input features")
    feature_names: list[str] = Field(description="Ordered feature names")
    hyperparams: dict[str, Any] = Field(
        default_factory=dict,
        description="Hyperparameters used for training",
    )
    val_accuracy: float = Field(description="Cross-validation accuracy")
    val_auc: float = Field(description="Cross-validation AUC-ROC")


class EvalMetrics(BaseModel):
    """Evaluation metrics on a hold-out test set.

    Attributes:
        accuracy: Classification accuracy.
        precision: Precision for the positive class.
        recall: Recall for the positive class.
        f1: F1 score for the positive class.
        auc_roc: Area under the ROC curve.
        log_loss: Logarithmic loss.
        sharpe: Approximate Sharpe ratio of predicted trades.
        n_samples: Number of test samples evaluated.
    """

    accuracy: float = Field(description="Classification accuracy")
    precision: float = Field(description="Precision (positive class)")
    recall: float = Field(description="Recall (positive class)")
    f1: float = Field(description="F1 score (positive class)")
    auc_roc: float = Field(description="Area under ROC curve")
    log_loss: float = Field(description="Log loss")
    sharpe: float = Field(description="Sharpe ratio of predicted trades")
    n_samples: int = Field(description="Number of evaluation samples")


# ---------------------------------------------------------------------------
# WalkForwardTrainer
# ---------------------------------------------------------------------------


class WalkForwardTrainer:
    """Walk-forward training pipeline with purged k-fold cross-validation.

    Trains gradient boosting models using time-series aware splits that
    always train on earlier data and validate on later data.  An embargo
    gap between train and validation sets prevents information leakage
    from autocorrelated features.

    Args:
        n_splits: Number of cross-validation folds.
        embargo_days: Number of trading days to embargo between the
            end of the training set and the start of the validation set.
        model_dir: Directory path for saving trained model artifacts.
    """

    def __init__(
        self,
        n_splits: int = 5,
        embargo_days: int = 5,
        model_dir: str = "models/",
    ) -> None:
        self._n_splits: int = n_splits
        self._embargo_days: int = embargo_days
        self._model_dir: Path = Path(model_dir)
        self._model_dir.mkdir(parents=True, exist_ok=True)
        self._log: structlog.stdlib.BoundLogger = get_logger("ml.trainer")
        self._log.info(
            "walk_forward_trainer_initialized",
            n_splits=n_splits,
            embargo_days=embargo_days,
            model_dir=str(self._model_dir),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(
        self,
        X: pd.DataFrame,  # noqa: N803
        y: pd.Series,
        model_type: str = "xgboost",
    ) -> TrainResult:
        """Run walk-forward training with purged k-fold cross-validation.

        Trains a model on each fold, collects per-fold metrics, selects
        the best fold by AUC-ROC, saves the best model to disk, and
        returns aggregated training results.

        Args:
            X: Feature matrix.
            y: Binary target variable.
            model_type: Model backend to use.  One of ``"xgboost"``,
                ``"lightgbm"``, or ``"catboost"``.

        Returns:
            A :class:`TrainResult` containing per-fold and aggregate
            metrics, feature importances, and timing information.

        Raises:
            ValueError: If *X* and *y* have mismatched lengths or there
                is insufficient data for the requested number of splits.
        """
        if len(X) != len(y):
            raise ValueError(f"X and y must have the same length: {len(X)} != {len(y)}")

        self._log.info(
            "training_started",
            model_type=model_type,
            n_samples=len(X),
            n_features=len(X.columns),
            n_splits=self._n_splits,
        )

        start_time = time.monotonic()

        # Drop rows where target is NaN
        valid_mask = y.notna()
        X_valid = X.loc[valid_mask].copy()  # noqa: N806
        y_valid = y.loc[valid_mask].copy()

        if len(X_valid) < self._n_splits * 20:
            raise ValueError(
                f"Insufficient data for {self._n_splits} splits: "
                f"only {len(X_valid)} valid samples"
            )

        splits = self._purged_kfold_split(X_valid, y_valid)

        fold_metrics_list: list[FoldMetrics] = []
        best_model: Any = None
        best_auc: float = -1.0
        best_fold_idx: int = 0
        all_importances: list[dict[str, float]] = []

        for fold_num, (train_idx, val_idx) in enumerate(splits):
            X_train = X_valid.iloc[train_idx]  # noqa: N806
            y_train = y_valid.iloc[train_idx]
            X_val = X_valid.iloc[val_idx]  # noqa: N806
            y_val = y_valid.iloc[val_idx]

            self._log.debug(
                "training_fold",
                fold=fold_num,
                train_size=len(train_idx),
                val_size=len(val_idx),
            )

            model, fold_metrics = self._train_fold(
                X_train, y_train, X_val, y_val, model_type
            )
            fold_metrics.fold_num = fold_num
            fold_metrics_list.append(fold_metrics)
            all_importances.append(fold_metrics.feature_importances)

            if fold_metrics.auc_roc > best_auc:
                best_auc = fold_metrics.auc_roc
                best_model = model
                best_fold_idx = fold_num

            self._log.info(
                "fold_complete",
                fold=fold_num,
                accuracy=round(fold_metrics.accuracy, 4),
                auc_roc=round(fold_metrics.auc_roc, 4),
                log_loss=round(fold_metrics.log_loss, 4),
            )

        # Aggregate feature importances across folds
        avg_importances = self._average_importances(all_importances)

        # Compute aggregate metrics
        avg_accuracy = float(np.mean([fm.accuracy for fm in fold_metrics_list]))
        avg_auc = float(np.mean([fm.auc_roc for fm in fold_metrics_list]))

        elapsed = time.monotonic() - start_time

        result = TrainResult(
            model_type=model_type,
            n_folds=self._n_splits,
            fold_metrics=fold_metrics_list,
            avg_accuracy=round(avg_accuracy, 4),
            avg_auc=round(avg_auc, 4),
            best_fold=best_fold_idx,
            feature_importances=avg_importances,
            train_time_seconds=round(elapsed, 2),
        )

        # Save best model
        if best_model is not None:
            # Determine date range from the data index
            train_start, train_end = self._extract_date_range(X_valid)

            metadata = ModelMetadata(
                model_name=f"titan_{model_type}_ensemble",
                version=self._next_model_version(model_type),
                trained_at=datetime.now(UTC),
                train_start=train_start,
                train_end=train_end,
                n_features=len(X_valid.columns),
                feature_names=list(X_valid.columns),
                hyperparams=self._get_model_params(best_model),
                val_accuracy=round(avg_accuracy, 4),
                val_auc=round(avg_auc, 4),
            )

            model_path = self.save_model(best_model, metadata)
            self._log.info(
                "best_model_saved",
                path=model_path,
                best_fold=best_fold_idx,
                auc_roc=round(best_auc, 4),
            )

        self._log.info(
            "training_complete",
            model_type=model_type,
            avg_accuracy=round(avg_accuracy, 4),
            avg_auc=round(avg_auc, 4),
            best_fold=best_fold_idx,
            train_time_seconds=round(elapsed, 2),
        )

        return result

    def evaluate(
        self,
        model: Any,
        X_test: pd.DataFrame,  # noqa: N803
        y_test: pd.Series,
    ) -> EvalMetrics:
        """Evaluate a trained model on a hold-out test set.

        Computes classification metrics (accuracy, precision, recall, F1,
        AUC-ROC, log loss) and an approximate Sharpe ratio based on the
        predicted trades.

        Args:
            model: A trained model with ``predict`` and
                ``predict_proba`` methods.
            X_test: Test feature matrix.
            y_test: Test target variable.

        Returns:
            An :class:`EvalMetrics` instance with all computed metrics.
        """
        self._log.info(
            "evaluating_model",
            n_samples=len(X_test),
        )

        valid_mask = y_test.notna()
        X_eval = X_test.loc[valid_mask]  # noqa: N806
        y_eval = y_test.loc[valid_mask]

        if len(X_eval) == 0:
            self._log.error("no_valid_samples_for_evaluation")
            return EvalMetrics(
                accuracy=0.0,
                precision=0.0,
                recall=0.0,
                f1=0.0,
                auc_roc=0.5,
                log_loss=999.0,
                sharpe=0.0,
                n_samples=0,
            )

        y_pred = model.predict(X_eval)
        y_pred_proba = model.predict_proba(X_eval)[:, 1]

        accuracy = float(accuracy_score(y_eval, y_pred))
        precision = float(precision_score(y_eval, y_pred, zero_division=0))
        recall = float(recall_score(y_eval, y_pred, zero_division=0))
        f1 = float(f1_score(y_eval, y_pred, zero_division=0))

        try:
            auc = float(roc_auc_score(y_eval, y_pred_proba))
        except ValueError:
            # Happens when only one class is present
            auc = 0.5
            self._log.warning(
                "auc_roc_computation_failed",
                reason="single_class_in_test",
            )

        logloss = float(log_loss(y_eval, y_pred_proba))

        # Approximate Sharpe: treat predicted positive trades as +1 return,
        # and predicted negative trades as 0 return (neutral).
        # A simple proxy using the actual outcomes of predicted trades.
        sharpe = self._compute_trade_sharpe(y_eval.values, y_pred)

        metrics = EvalMetrics(
            accuracy=round(accuracy, 4),
            precision=round(precision, 4),
            recall=round(recall, 4),
            f1=round(f1, 4),
            auc_roc=round(auc, 4),
            log_loss=round(logloss, 4),
            sharpe=round(sharpe, 4),
            n_samples=len(X_eval),
        )

        self._log.info(
            "evaluation_complete",
            accuracy=metrics.accuracy,
            auc_roc=metrics.auc_roc,
            sharpe=metrics.sharpe,
            n_samples=metrics.n_samples,
        )

        return metrics

    def save_model(self, model: Any, metadata: ModelMetadata) -> str:
        """Save a trained model and its metadata to disk.

        The model is saved in its native format (JSON for XGBoost,
        txt for LightGBM, cbm for CatBoost) and a JSON sidecar file
        contains the metadata.

        Args:
            model: Trained model object.
            metadata: Model metadata to persist alongside the artifact.

        Returns:
            Absolute path to the saved model file.
        """
        model_name = metadata.model_name
        version = metadata.version

        # Determine file extension by model type
        model_class_name = type(model).__name__.lower()

        if "xgb" in model_class_name:
            ext = "json"
        elif "lgbm" in model_class_name or "lightgbm" in model_class_name:
            ext = "txt"
        elif "catboost" in model_class_name:
            ext = "cbm"
        else:
            ext = "pkl"

        model_filename = f"{model_name}_v{version}.{ext}"
        meta_filename = f"{model_name}_v{version}_metadata.json"

        model_path = self._model_dir / model_filename
        meta_path = self._model_dir / meta_filename

        # Save model in native format
        if ext == "json":
            model.save_model(str(model_path))
        elif ext == "txt":
            # LGBMClassifier: use booster_ if available, else pickle
            if hasattr(model, "booster_"):
                model.booster_.save_model(str(model_path))
            else:
                import pickle

                pkl_path = model_path.with_suffix(".pkl")
                with open(pkl_path, "wb") as pkl_f:
                    pickle.dump(model, pkl_f)
                model_path = pkl_path
        elif ext == "cbm":
            model.save_model(str(model_path))
        else:
            import pickle

            with open(model_path, "wb") as f:
                pickle.dump(model, f)

        # Save metadata sidecar
        meta_dict = metadata.model_dump(mode="json")
        with open(meta_path, "w") as f:
            json.dump(meta_dict, f, indent=2, default=str)

        self._log.info(
            "model_saved",
            model_path=str(model_path),
            metadata_path=str(meta_path),
            version=version,
        )

        return str(model_path.resolve())

    def load_model(self, model_path: str) -> tuple[Any, ModelMetadata]:
        """Load a trained model and its metadata from disk.

        Args:
            model_path: Path to the model file.  The metadata sidecar
                is expected at the same location with ``_metadata.json``
                suffix.

        Returns:
            A tuple of ``(model, metadata)``.

        Raises:
            FileNotFoundError: If the model file does not exist.
        """
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        ext = path.suffix.lstrip(".")

        if ext == "json":
            from xgboost import XGBClassifier

            model = XGBClassifier()
            model.load_model(str(path))
        elif ext == "txt":
            import lightgbm as lgb

            model = lgb.Booster(model_file=str(path))
        elif ext == "cbm":
            from catboost import CatBoostClassifier

            model = CatBoostClassifier()
            model.load_model(str(path))
        else:
            import pickle

            with open(path, "rb") as f:
                model = pickle.load(f)  # noqa: S301

        # Load metadata sidecar
        # Derive metadata path from model path
        stem = path.stem
        # Remove version suffix pattern to build metadata filename
        meta_path = path.parent / f"{stem}_metadata.json"
        if not meta_path.exists():
            # Try alternative naming: replace extension with _metadata.json
            meta_path = path.with_suffix("").with_suffix("_metadata.json")

        if meta_path.exists():
            with open(meta_path) as f:
                meta_dict = json.load(f)
            metadata = ModelMetadata(**meta_dict)
        else:
            self._log.warning(
                "metadata_sidecar_not_found",
                expected_path=str(meta_path),
            )
            metadata = ModelMetadata(
                model_name="unknown",
                version=0,
                trained_at=datetime.now(UTC),
                train_start=date.today(),
                train_end=date.today(),
                n_features=0,
                feature_names=[],
                hyperparams={},
                val_accuracy=0.0,
                val_auc=0.0,
            )

        self._log.info(
            "model_loaded",
            path=model_path,
            model_name=metadata.model_name,
            version=metadata.version,
        )

        return model, metadata

    # ------------------------------------------------------------------
    # Walk-Forward Ensemble Training
    # ------------------------------------------------------------------

    def train_walk_forward_ensemble(
        self,
        X: pd.DataFrame,  # noqa: N803
        y: pd.Series,
        dates: pd.Series,
        train_months: int = 12,
        test_months: int = 1,
        model_types: list[str] | None = None,
        verbose: bool = False,
    ) -> dict[str, Any]:
        """Monthly rolling walk-forward training with 3-model ensemble.

        For each window:
        1. Split data chronologically (train_months train, test_months test)
        2. Run purged 5-fold CV within training data for estimation
        3. Train final models on full training window with early stopping
        4. Evaluate ensemble on held-out test window (OOS)
        5. Record all metrics

        Args:
            X: Feature matrix (n_samples x n_features).
            y: Binary target variable.
            dates: Date for each row (used for temporal splitting).
            train_months: Training window size in months.
            test_months: Test window size in months.
            model_types: List of model backends. Defaults to all three.
            verbose: Print progress to stdout.

        Returns:
            Dictionary with all results, per-window metrics, and
            aggregate statistics.
        """
        if model_types is None:
            model_types = ["xgboost", "lightgbm", "catboost"]

        start_time = time.monotonic()

        # Convert dates to month periods for windowing
        months = dates.dt.to_period("M")
        unique_months = sorted(months.unique())

        if len(unique_months) < train_months + test_months:
            raise ValueError(
                f"Need at least {train_months + test_months} months of data, "
                f"got {len(unique_months)}"
            )

        window_results: list[dict[str, Any]] = []
        all_importances: dict[str, list[dict[str, float]]] = {
            mt: [] for mt in model_types
        }

        n_windows = len(unique_months) - train_months
        if verbose:
            print(
                f"Walk-forward: {n_windows} windows, "
                f"{train_months}mo train / {test_months}mo test"
            )

        for w_idx in range(train_months, len(unique_months)):
            test_month = unique_months[w_idx]
            train_start_month = unique_months[w_idx - train_months]
            train_end_month = unique_months[w_idx - 1]

            # Build masks
            train_mask = (months >= train_start_month) & (months <= train_end_month)
            test_mask = months == test_month

            X_train = X.loc[train_mask]  # noqa: N806
            y_train = y.loc[train_mask]
            X_test = X.loc[test_mask]  # noqa: N806
            y_test = y.loc[test_mask]

            if len(X_test) < 10 or len(X_train) < 100:
                continue

            # --- Inner purged CV on training data ---
            cv_aucs: list[float] = []
            inner_splits = self._purged_kfold_split(X_train, y_train)
            for train_idx, val_idx in inner_splits:
                X_cv_train = X_train.iloc[train_idx]  # noqa: N806
                y_cv_train = y_train.iloc[train_idx]
                X_cv_val = X_train.iloc[val_idx]  # noqa: N806
                y_cv_val = y_train.iloc[val_idx]

                # Quick XGBoost for CV estimation
                cv_model = self._get_model("xgboost")
                cv_model.fit(
                    X_cv_train,
                    y_cv_train,
                    eval_set=[(X_cv_val, y_cv_val)],
                    verbose=False,
                )
                try:
                    cv_proba = cv_model.predict_proba(X_cv_val)[:, 1]
                    cv_aucs.append(float(roc_auc_score(y_cv_val, cv_proba)))
                except ValueError:
                    cv_aucs.append(0.5)

            cv_auc_mean = float(np.mean(cv_aucs)) if cv_aucs else 0.5
            cv_auc_std = float(np.std(cv_aucs)) if cv_aucs else 0.0

            # --- Train all 3 models on full training window ---
            # Use last 20% of training window as eval set for early stopping
            es_split = int(len(X_train) * 0.8)
            X_train_fit = X_train.iloc[:es_split]  # noqa: N806
            y_train_fit = y_train.iloc[:es_split]
            X_train_eval = X_train.iloc[es_split:]  # noqa: N806
            y_train_eval = y_train.iloc[es_split:]

            window_models: dict[str, Any] = {}
            window_probas: dict[str, np.ndarray] = {}
            window_aucs: dict[str, float] = {}
            window_accs: dict[str, float] = {}

            for mt in model_types:
                model = self._get_model(mt)
                self._fit_model(
                    model,
                    mt,
                    X_train_fit,
                    y_train_fit,
                    X_train_eval,
                    y_train_eval,
                )

                # OOS predictions
                y_proba = model.predict_proba(X_test)[:, 1]
                y_pred = model.predict(X_test)

                try:
                    auc = float(roc_auc_score(y_test, y_proba))
                except ValueError:
                    auc = 0.5

                acc = float(accuracy_score(y_test, y_pred))

                window_models[mt] = model
                window_probas[mt] = y_proba
                window_aucs[mt] = auc
                window_accs[mt] = acc

                # Collect feature importances
                imps = self._extract_importances(model, X_train.columns.tolist())
                all_importances[mt].append(imps)

            # Ensemble prediction (simple average)
            ensemble_proba = np.mean([window_probas[mt] for mt in model_types], axis=0)
            ensemble_pred = (ensemble_proba >= 0.5).astype(int)

            try:
                ens_auc = float(roc_auc_score(y_test, ensemble_proba))
            except ValueError:
                ens_auc = 0.5

            ens_acc = float(accuracy_score(y_test, ensemble_pred))

            window_result = {
                "window": len(window_results) + 1,
                "train_period": f"{train_start_month} to {train_end_month}",
                "test_period": str(test_month),
                "train_size": len(X_train),
                "test_size": len(X_test),
                "cv_auc_mean": round(cv_auc_mean, 4),
                "cv_auc_std": round(cv_auc_std, 4),
                "oos_auc": round(ens_auc, 4),
                "oos_accuracy": round(ens_acc, 4),
                "per_model_auc": {mt: round(window_aucs[mt], 4) for mt in model_types},
                "per_model_acc": {mt: round(window_accs[mt], 4) for mt in model_types},
                "models": window_models,
            }
            window_results.append(window_result)

            if verbose:
                print(
                    f"  Window {window_result['window']:2d} | "
                    f"Train: {train_start_month}–{train_end_month} | "
                    f"Test: {test_month} | "
                    f"CV AUC: {cv_auc_mean:.3f}±{cv_auc_std:.3f} | "
                    f"OOS AUC: {ens_auc:.3f} | "
                    f"OOS Acc: {ens_acc:.3f}"
                )

        if not window_results:
            raise RuntimeError("No valid walk-forward windows produced")

        # --- Aggregate metrics ---
        oos_aucs = [w["oos_auc"] for w in window_results]
        oos_accs = [w["oos_accuracy"] for w in window_results]
        cv_aucs_all = [w["cv_auc_mean"] for w in window_results]

        # Average feature importances across all windows for each model
        avg_importances_per_model: dict[str, dict[str, float]] = {}
        for mt in model_types:
            avg_importances_per_model[mt] = self._average_importances(
                all_importances[mt]
            )

        # Combined importances (average across models)
        combined_imp: dict[str, list[float]] = {}
        for mt in model_types:
            for feat, imp in avg_importances_per_model[mt].items():
                combined_imp.setdefault(feat, []).append(imp)
        avg_importances_combined = {
            feat: round(float(np.mean(vals)), 6) for feat, vals in combined_imp.items()
        }

        elapsed = time.monotonic() - start_time

        result = {
            "n_windows": len(window_results),
            "n_models": len(model_types),
            "model_types": model_types,
            "train_months": train_months,
            "test_months": test_months,
            "window_results": window_results,
            "avg_oos_auc": round(float(np.mean(oos_aucs)), 4),
            "min_oos_auc": round(float(np.min(oos_aucs)), 4),
            "max_oos_auc": round(float(np.max(oos_aucs)), 4),
            "std_oos_auc": round(float(np.std(oos_aucs)), 4),
            "avg_oos_accuracy": round(float(np.mean(oos_accs)), 4),
            "avg_cv_auc": round(float(np.mean(cv_aucs_all)), 4),
            "feature_importances": avg_importances_combined,
            "per_model_importances": avg_importances_per_model,
            "total_train_samples": sum(w["train_size"] for w in window_results),
            "train_time_seconds": round(elapsed, 2),
        }

        self._log.info(
            "walk_forward_ensemble_complete",
            n_windows=result["n_windows"],
            avg_oos_auc=result["avg_oos_auc"],
            train_time_seconds=result["train_time_seconds"],
        )

        return result

    def prune_features(
        self,
        importances: dict[str, float],
        threshold: float = 0.005,
    ) -> list[str]:
        """Identify features to keep based on importance threshold.

        Args:
            importances: Feature name to average importance mapping.
            threshold: Minimum fraction of total importance to keep.

        Returns:
            List of feature names that pass the threshold.
        """
        total = sum(importances.values())
        if total == 0:
            return list(importances.keys())

        kept = [feat for feat, imp in importances.items() if imp / total >= threshold]

        self._log.info(
            "feature_pruning",
            original=len(importances),
            kept=len(kept),
            pruned=len(importances) - len(kept),
            threshold=threshold,
        )

        return kept

    def save_ensemble_models(
        self,
        window_results: list[dict[str, Any]],
        feature_names: list[str],
        metadata: dict[str, Any],
        model_types: list[str] | None = None,
    ) -> dict[str, str]:
        """Save the models from the most recent window.

        Args:
            window_results: List of per-window results from walk-forward.
            feature_names: Ordered list of feature names.
            metadata: Aggregate metadata dict.
            model_types: Model types to save.

        Returns:
            Dict mapping model type to saved file path.
        """
        if model_types is None:
            model_types = ["xgboost", "lightgbm", "catboost"]

        last_window = window_results[-1]
        models = last_window["models"]
        saved_paths: dict[str, str] = {}

        for mt in model_types:
            if mt not in models:
                continue

            model = models[mt]
            model_meta = ModelMetadata(
                model_name=f"titan_{mt}_ensemble",
                version=self._next_model_version(mt),
                trained_at=datetime.now(UTC),
                train_start=date.today(),
                train_end=date.today(),
                n_features=len(feature_names),
                feature_names=feature_names,
                hyperparams=self._get_model_params(model),
                val_accuracy=metadata.get("avg_oos_accuracy", 0.0),
                val_auc=metadata.get("avg_oos_auc", 0.0),
            )
            path = self.save_model(model, model_meta)
            saved_paths[mt] = path

        return saved_paths

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    def _fit_model(
        self,
        model: Any,
        model_type: str,
        X_train: pd.DataFrame,  # noqa: N803
        y_train: pd.Series,
        X_val: pd.DataFrame,  # noqa: N803
        y_val: pd.Series,
    ) -> None:
        """Fit a model with early stopping on a validation set.

        Handles the different fit() signatures for XGBoost, LightGBM,
        and CatBoost.
        """
        if model_type == "xgboost":
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )
        elif model_type == "lightgbm":
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
            )
        elif model_type == "catboost":
            from catboost import Pool

            eval_pool = Pool(X_val, y_val)
            model.fit(
                X_train,
                y_train,
                eval_set=eval_pool,
                verbose=False,
            )
        else:
            model.fit(X_train, y_train)

    def _purged_kfold_split(
        self,
        X: pd.DataFrame,  # noqa: N803
        y: pd.Series,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Generate purged k-fold splits for time-series data.

        Produces non-overlapping, chronologically ordered folds where
        each validation set follows the corresponding training set.
        An embargo period of ``embargo_days`` rows is excluded between
        the end of training and the start of validation to prevent
        information leakage from autocorrelated features.

        Args:
            X: Feature matrix (used only for its length and index).
            y: Target variable (used only for its length).

        Returns:
            A list of ``(train_indices, val_indices)`` tuples.
        """
        n_samples = len(X)
        fold_size = n_samples // (self._n_splits + 1)

        splits: list[tuple[np.ndarray, np.ndarray]] = []

        for i in range(self._n_splits):
            # Training set: all data up to the end of this fold
            train_end = fold_size * (i + 1)

            # Embargo: skip embargo_days after training end
            val_start = train_end + self._embargo_days

            # Validation set: next fold_size rows after embargo
            val_end = min(val_start + fold_size, n_samples)

            if val_start >= n_samples:
                self._log.warning(
                    "skipping_fold_insufficient_data",
                    fold=i,
                    val_start=val_start,
                    n_samples=n_samples,
                )
                continue

            train_idx = np.arange(0, train_end)
            val_idx = np.arange(val_start, val_end)

            if len(val_idx) < 10:
                self._log.warning(
                    "fold_validation_set_too_small",
                    fold=i,
                    val_size=len(val_idx),
                )
                continue

            splits.append((train_idx, val_idx))

            self._log.debug(
                "fold_split_created",
                fold=i,
                train_range=(0, train_end),
                embargo=self._embargo_days,
                val_range=(val_start, val_end),
                train_size=len(train_idx),
                val_size=len(val_idx),
            )

        if not splits:
            raise ValueError(
                f"Could not create any valid folds from {n_samples} samples "
                f"with {self._n_splits} splits and {self._embargo_days}-day embargo"
            )

        return splits

    def _train_fold(
        self,
        X_train: pd.DataFrame,  # noqa: N803
        y_train: pd.Series,
        X_val: pd.DataFrame,  # noqa: N803
        y_val: pd.Series,
        model_type: str,
    ) -> tuple[Any, FoldMetrics]:
        """Train a model on a single fold and compute validation metrics.

        Args:
            X_train: Training features.
            y_train: Training target.
            X_val: Validation features.
            y_val: Validation target.
            model_type: Model backend identifier.

        Returns:
            A tuple of ``(trained_model, fold_metrics)``.
        """
        model = self._get_model(model_type)
        self._fit_model(model, model_type, X_train, y_train, X_val, y_val)

        # Predict on validation set
        y_pred = model.predict(X_val)
        y_pred_proba = model.predict_proba(X_val)[:, 1]

        accuracy = float(accuracy_score(y_val, y_pred))

        try:
            auc = float(roc_auc_score(y_val, y_pred_proba))
        except ValueError:
            auc = 0.5

        logloss = float(log_loss(y_val, y_pred_proba))

        # Extract feature importances
        importances = self._extract_importances(model, X_train.columns.tolist())

        fold_metrics = FoldMetrics(
            fold_num=-1,  # Updated by caller in train() loop
            train_size=len(X_train),
            val_size=len(X_val),
            accuracy=round(accuracy, 4),
            auc_roc=round(auc, 4),
            log_loss=round(logloss, 4),
            feature_importances=importances,
        )

        return model, fold_metrics

    def _get_model(self, model_type: str) -> Any:
        """Create a model instance with sensible default hyperparameters.

        Args:
            model_type: One of ``"xgboost"``, ``"lightgbm"``, or
                ``"catboost"``.

        Returns:
            An unfitted classifier instance.

        Raises:
            ValueError: If *model_type* is not recognized.
        """
        if model_type == "xgboost":
            from xgboost import XGBClassifier

            return XGBClassifier(
                n_estimators=500,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=5,
                reg_alpha=0.1,
                reg_lambda=1.0,
                eval_metric="auc",
                early_stopping_rounds=50,
                random_state=42,
                n_jobs=-1,
                verbosity=0,
                tree_method="hist",
            )

        if model_type == "lightgbm":
            from lightgbm import LGBMClassifier

            return LGBMClassifier(
                n_estimators=500,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=5,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42,
                n_jobs=-1,
                verbose=-1,
            )

        if model_type == "catboost":
            from catboost import CatBoostClassifier

            return CatBoostClassifier(
                iterations=500,
                depth=6,
                learning_rate=0.05,
                l2_leaf_reg=3.0,
                random_seed=42,
                verbose=0,
            )

        raise ValueError(
            f"Unknown model_type '{model_type}'. "
            f"Supported: 'xgboost', 'lightgbm', 'catboost'"
        )

    def _extract_importances(
        self,
        model: Any,
        feature_names: list[str],
    ) -> dict[str, float]:
        """Extract feature importances from a trained model.

        Args:
            model: Trained classifier.
            feature_names: Ordered list of feature names.

        Returns:
            Dictionary mapping feature names to importance values.
        """
        try:
            raw_importances = model.feature_importances_
            total = raw_importances.sum()
            normalized = raw_importances / total if total > 0 else raw_importances

            return {
                name: round(float(imp), 6)
                for name, imp in zip(feature_names, normalized, strict=False)
            }
        except AttributeError:
            self._log.warning(
                "cannot_extract_feature_importances",
                model_type=type(model).__name__,
            )
            return {}

    def _average_importances(
        self,
        all_importances: list[dict[str, float]],
    ) -> dict[str, float]:
        """Average feature importances across multiple folds.

        Args:
            all_importances: List of per-fold importance dictionaries.

        Returns:
            Dictionary of averaged importances.
        """
        if not all_importances:
            return {}

        aggregated: dict[str, list[float]] = {}
        for fold_imp in all_importances:
            for name, value in fold_imp.items():
                aggregated.setdefault(name, []).append(value)

        return {
            name: round(float(np.mean(values)), 6)
            for name, values in sorted(aggregated.items())
        }

    def _get_model_params(self, model: Any) -> dict[str, Any]:
        """Extract hyperparameters from a trained model.

        Args:
            model: Trained model with ``get_params`` method.

        Returns:
            Dictionary of hyperparameters.  Returns an empty dict if
            parameter extraction fails.
        """
        try:
            params = model.get_params()
            # Filter out non-serializable values
            serializable_params: dict[str, Any] = {}
            for key, value in params.items():
                if isinstance(value, (str, int, float, bool, type(None), list)):
                    serializable_params[key] = value
            return serializable_params
        except AttributeError:
            return {}

    def _next_model_version(self, model_type: str) -> int:
        """Determine the next version number for a model type.

        Scans the model directory for existing artifacts and returns
        the next sequential version.

        Args:
            model_type: Model backend name (used in filename pattern).

        Returns:
            The next version number to use.
        """
        pattern = f"titan_{model_type}_ensemble_v*.json"
        existing = list(self._model_dir.glob(pattern))

        # Also check for other extensions
        for ext in ("txt", "cbm", "pkl"):
            existing.extend(
                self._model_dir.glob(f"titan_{model_type}_ensemble_v*.{ext}")
            )

        if not existing:
            return 1

        versions: list[int] = []
        for path in existing:
            stem = path.stem
            # Extract version from pattern: titan_xgboost_ensemble_v3
            parts = stem.split("_v")
            if len(parts) >= 2:
                try:
                    version_str = parts[-1].split("_")[0]
                    versions.append(int(version_str))
                except ValueError:
                    continue

        return max(versions, default=0) + 1

    def _extract_date_range(self, X: pd.DataFrame) -> tuple[date, date]:  # noqa: N803
        """Extract the date range from a DataFrame's index.

        Handles MultiIndex (ticker, timestamp), DatetimeIndex, and
        plain integer index cases.

        Args:
            X: Feature DataFrame.

        Returns:
            Tuple of ``(start_date, end_date)``.
        """
        try:
            if isinstance(X.index, pd.MultiIndex):
                # Try the second level (timestamp) first
                timestamps = X.index.get_level_values(-1)
            else:
                timestamps = X.index

            if isinstance(timestamps, pd.DatetimeIndex):
                return timestamps.min().date(), timestamps.max().date()

            # Try converting to datetime
            dt_index = pd.to_datetime(timestamps)
            return dt_index.min().date(), dt_index.max().date()
        except (ValueError, TypeError):
            self._log.warning("cannot_extract_date_range_from_index")
            return date.today(), date.today()

    @staticmethod
    def _compute_trade_sharpe(
        y_actual: np.ndarray,
        y_predicted: np.ndarray,
    ) -> float:
        """Compute an approximate Sharpe ratio for predicted trades.

        Simulates a simple strategy: enter a trade when the model
        predicts positive.  The "return" for each predicted-positive
        trade is +1 if correct and -1 if incorrect.  The Sharpe ratio
        is computed as mean(returns) / std(returns).

        Args:
            y_actual: Actual binary labels.
            y_predicted: Predicted binary labels.

        Returns:
            Annualized Sharpe ratio.  Returns 0.0 if no trades were
            predicted or if standard deviation is zero.
        """
        trade_mask = y_predicted == 1
        if not np.any(trade_mask):
            return 0.0

        # +1 for correct predictions, -1 for incorrect
        trade_returns = np.where(y_actual[trade_mask] == 1, 1.0, -1.0)

        if len(trade_returns) < 2:
            return 0.0

        mean_return = float(np.mean(trade_returns))
        std_return = float(np.std(trade_returns, ddof=1))

        if std_return == 0:
            return 0.0

        # Annualize assuming ~252 trading days and one opportunity per day
        daily_sharpe = mean_return / std_return
        annualized_sharpe = daily_sharpe * np.sqrt(252)

        return float(annualized_sharpe)
