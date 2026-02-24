"""Feature engineering pipeline for Project Titan.

Combines technical features, signal scores, and derived features into a
single feature matrix suitable for model training and inference.  Provides
feature selection via importance thresholding, normalization, target variable
creation, lagged feature generation, and correlated feature removal.

Usage::

    from src.ml.features import FeatureEngineer

    engineer = FeatureEngineer(importance_threshold=0.005)
    X = engineer.build_feature_matrix(price_data, signal_data)
    selected = engineer.select_features(X, y)
    X_norm, params = engineer.normalize_features(X[selected])
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from xgboost import XGBClassifier

from src.utils.logging import get_logger

if TYPE_CHECKING:
    import structlog


# ---------------------------------------------------------------------------
# Pydantic configuration model
# ---------------------------------------------------------------------------


class FeaturePipelineConfig(BaseModel):
    """Configuration for the feature engineering pipeline.

    Attributes:
        importance_threshold: Minimum feature importance to retain a feature
            during selection (fraction, e.g. 0.005 = 0.5%).
        normalize_method: Normalization strategy.  One of ``"standard"``,
            ``"minmax"``, or ``"robust"``.
        target_horizon: Number of trading days forward for the target
            variable.
        target_threshold: Minimum percentage move (as a decimal) to classify
            as a positive target.
        lag_periods: List of lag periods to generate for lagged features.
    """

    importance_threshold: float = Field(
        default=0.005,
        ge=0.0,
        le=1.0,
        description="Minimum feature importance to retain (0.5% = 0.005)",
    )
    normalize_method: Literal["standard", "minmax", "robust"] = Field(
        default="standard",
        description="Normalization method to apply to features",
    )
    target_horizon: int = Field(
        default=5,
        ge=1,
        description="Forward-looking window in trading days for the target variable",
    )
    target_threshold: float = Field(
        default=0.02,
        ge=0.0,
        description="Minimum percentage move to classify as positive target",
    )
    lag_periods: list[int] = Field(
        default=[1, 2, 3, 5],
        description="Lag periods to generate for lagged features",
    )


# ---------------------------------------------------------------------------
# FeatureEngineer
# ---------------------------------------------------------------------------


class FeatureEngineer:
    """Full feature engineering pipeline for ML model consumption.

    Combines raw price data and signal scores into a unified feature matrix,
    performs feature selection, normalization, and target variable creation.

    Args:
        importance_threshold: Minimum feature importance to keep during
            selection.  Defaults to 0.005 (0.5%).
    """

    def __init__(self, importance_threshold: float = 0.005) -> None:
        self._importance_threshold: float = importance_threshold
        self._log: structlog.stdlib.BoundLogger = get_logger("ml.features")
        self._feature_importances: dict[str, float] = {}
        self._log.info(
            "feature_engineer_initialized",
            importance_threshold=importance_threshold,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_feature_matrix(
        self,
        price_data: dict[str, pd.DataFrame],
        signal_data: dict[str, dict[str, Any]],
    ) -> pd.DataFrame:
        """Combine technical features, signal scores, and derived features.

        Each row in the output represents one ticker at one timestamp.
        Missing data is forward-filled first, then remaining NaNs are
        filled with the column median.

        Args:
            price_data: Mapping of ticker symbol to OHLCV+features DataFrame.
                Each DataFrame is expected to contain technical indicator
                columns produced by ``TechnicalSignalGenerator``.
            signal_data: Mapping of ticker symbol to a dictionary of signal
                scores.  Expected keys include ``"technical_score"``,
                ``"sentiment_score"``, ``"flow_score"``, ``"regime_score"``,
                ``"ensemble_score"``, and ``"regime"``.

        Returns:
            A clean feature matrix DataFrame with a MultiIndex of
            ``(ticker, timestamp)`` and all feature columns.  NaN values
            have been imputed.
        """
        self._log.info(
            "building_feature_matrix",
            n_tickers=len(price_data),
            n_signal_tickers=len(signal_data),
        )

        frames: list[pd.DataFrame] = []

        for ticker, df in price_data.items():
            if df.empty:
                self._log.warning(
                    "skipping_empty_price_data",
                    ticker=ticker,
                )
                continue

            ticker_df = df.copy()

            # Inject signal scores as columns
            signals = signal_data.get(ticker, {})
            for signal_name, signal_value in signals.items():
                if isinstance(signal_value, (int, float)):
                    ticker_df[f"signal_{signal_name}"] = signal_value
                elif isinstance(signal_value, str):
                    # Encode categorical signals (e.g. regime) as
                    # deterministic numeric codes using hashlib (Python's
                    # built-in hash() is randomized across sessions).
                    import hashlib

                    digest = int.from_bytes(
                        hashlib.sha256(signal_value.encode()).digest()[:4],
                        "big",
                    )
                    ticker_df[f"signal_{signal_name}"] = (digest % 1000) / 1000.0

            # Add derived cross-features
            ticker_df = self._add_derived_features(ticker_df)

            # Tag with ticker for multi-ticker matrix
            ticker_df["ticker"] = ticker

            frames.append(ticker_df)

        if not frames:
            self._log.error("no_valid_frames_for_feature_matrix")
            return pd.DataFrame()

        combined = pd.concat(frames, axis=0)

        # Set a MultiIndex of (ticker, timestamp)
        if "ticker" in combined.columns:
            if isinstance(combined.index, pd.DatetimeIndex):
                combined = combined.reset_index()
                timestamp_col = combined.columns[0]  # the former index
                combined = combined.rename(columns={timestamp_col: "timestamp"})
                combined = combined.set_index(["ticker", "timestamp"])
            else:
                combined = combined.set_index("ticker", append=True)
                combined.index.names = ["timestamp", "ticker"]
                combined = combined.swaplevel()

        # Impute missing values: forward fill within each ticker, then median
        combined = combined.groupby(level="ticker", group_keys=False).apply(
            lambda grp: grp.ffill()
        )
        numeric_cols = combined.select_dtypes(include=[np.number]).columns
        medians = combined[numeric_cols].median()
        combined[numeric_cols] = combined[numeric_cols].fillna(medians)

        # Drop any remaining non-numeric columns that slipped through
        non_numeric = combined.select_dtypes(exclude=[np.number]).columns
        if len(non_numeric) > 0:
            self._log.debug(
                "dropping_non_numeric_columns",
                columns=list(non_numeric),
            )
            combined = combined.drop(columns=non_numeric)

        self._log.info(
            "feature_matrix_built",
            shape=combined.shape,
            n_features=len(combined.columns),
            n_rows=len(combined),
        )

        return combined

    def build_trade_features(self, trade_df: pd.DataFrame) -> pd.DataFrame:
        """Build a feature matrix from closed trade records for model retraining.

        Extracts predictive features from trade metadata (no external API calls
        needed).  Designed for the weekly retrain pipeline where the input is a
        DataFrame of rows from the PostgreSQL ``trades`` table.

        Expected input columns: ``ticker``, ``strategy``, ``direction``,
        ``entry_price``, ``exit_price``, ``realized_pnl``, ``ml_confidence``,
        ``regime``, ``entry_time``, ``exit_time``.

        Args:
            trade_df: DataFrame of closed trade records.

        Returns:
            A clean numeric feature matrix with one row per trade.  NaN values
            are forward-filled then filled with column medians.
        """
        self._log.info(
            "building_trade_features",
            n_trades=len(trade_df),
            columns=list(trade_df.columns),
        )

        features = pd.DataFrame(index=trade_df.index)

        # --- ML confidence at entry ---
        if "ml_confidence" in trade_df.columns:
            features["ml_confidence"] = pd.to_numeric(
                trade_df["ml_confidence"], errors="coerce"
            )

        # --- One-hot encode regime (4 states) ---
        regime_dummies = pd.get_dummies(
            trade_df.get("regime", pd.Series(dtype=str)),
            prefix="regime",
        )
        features = pd.concat([features, regime_dummies], axis=1)

        # --- One-hot encode strategy (10 strategies) ---
        strategy_dummies = pd.get_dummies(
            trade_df.get("strategy", pd.Series(dtype=str)),
            prefix="strategy",
        )
        features = pd.concat([features, strategy_dummies], axis=1)

        # --- Direction encoded: 1=bullish, -1=bearish, 0=neutral ---
        direction_map = {"bullish": 1, "bearish": -1, "neutral": 0}
        if "direction" in trade_df.columns:
            features["direction_encoded"] = (
                trade_df["direction"]
                .str.lower()
                .map(direction_map)
                .fillna(0)
                .astype(int)
            )

        # --- Hold days (exit_time - entry_time) ---
        if "entry_time" in trade_df.columns and "exit_time" in trade_df.columns:
            entry = pd.to_datetime(trade_df["entry_time"], errors="coerce")
            exit_ = pd.to_datetime(trade_df["exit_time"], errors="coerce")
            hold_td = exit_ - entry
            features["hold_days"] = hold_td.dt.total_seconds() / 86400.0

        # --- Entry hour (hour of day) ---
        if "entry_time" in trade_df.columns:
            entry = pd.to_datetime(trade_df["entry_time"], errors="coerce")
            features["entry_hour"] = entry.dt.hour

        # --- Day of week (0=Mon, 4=Fri) ---
        if "entry_time" in trade_df.columns:
            entry = pd.to_datetime(trade_df["entry_time"], errors="coerce")
            features["day_of_week"] = entry.dt.dayofweek

        # --- Log entry price ---
        if "entry_price" in trade_df.columns:
            price = pd.to_numeric(trade_df["entry_price"], errors="coerce")
            features["log_price"] = np.log(price.clip(lower=0.01))

        # Ensure all columns are numeric
        features = features.apply(pd.to_numeric, errors="coerce")

        # Impute: forward-fill then median (same pattern as build_feature_matrix)
        features = features.ffill()
        medians = features.median()
        features = features.fillna(medians)

        # Final fallback: fill any remaining NaN with 0
        features = features.fillna(0)

        self._log.info(
            "trade_features_built",
            shape=features.shape,
            n_features=len(features.columns),
            columns=list(features.columns),
        )

        return features

    def select_features(
        self,
        X: pd.DataFrame,  # noqa: N803
        y: pd.Series,
        method: str = "importance",
    ) -> list[str]:
        """Select features by importance from a quick XGBoost model.

        Trains a lightweight XGBoost classifier to extract feature
        importances and retains only those features whose importance
        exceeds the configured threshold.

        Args:
            X: Feature matrix (rows = samples, columns = features).
            y: Binary target variable aligned with *X*.
            method: Selection strategy.  ``"importance"`` uses XGBoost
                feature importances (default).  ``"mutual_info"`` uses
                sklearn mutual information.  ``"variance"`` drops
                near-zero variance features.

        Returns:
            Sorted list of feature names that pass the selection filter.
        """
        self._log.info(
            "selecting_features",
            method=method,
            n_features=len(X.columns),
            n_samples=len(X),
            importance_threshold=self._importance_threshold,
        )

        if method == "importance":
            selected = self._select_by_importance(X, y)
        elif method == "mutual_info":
            selected = self._select_by_mutual_info(X, y)
        elif method == "variance":
            selected = self._select_by_variance(X)
        else:
            self._log.warning(
                "unknown_selection_method_falling_back",
                method=method,
                fallback="importance",
            )
            selected = self._select_by_importance(X, y)

        self._log.info(
            "features_selected",
            method=method,
            n_selected=len(selected),
            n_original=len(X.columns),
            pct_retained=round(len(selected) / max(len(X.columns), 1) * 100, 1),
        )

        return selected

    def normalize_features(
        self,
        X: pd.DataFrame,  # noqa: N803
        method: str = "standard",
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Normalize feature values using the specified scaler.

        Args:
            X: Feature matrix to normalize.
            method: One of ``"standard"`` (StandardScaler),
                ``"minmax"`` (MinMaxScaler), or ``"robust"``
                (RobustScaler).

        Returns:
            A tuple of ``(normalized_df, scaler_params)`` where
            ``scaler_params`` is a dictionary containing the scaler
            type, means/scales/etc. needed for inverse transformation.
        """
        self._log.info(
            "normalizing_features",
            method=method,
            shape=X.shape,
        )

        scaler: StandardScaler | MinMaxScaler | RobustScaler

        if method == "minmax":
            scaler = MinMaxScaler()
        elif method == "robust":
            scaler = RobustScaler()
        else:
            scaler = StandardScaler()

        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            self._log.warning("no_numeric_columns_to_normalize")
            return X.copy(), {"method": method, "columns": []}

        scaled_values = scaler.fit_transform(X[numeric_cols])
        normalized_df = X.copy()
        normalized_df[numeric_cols] = scaled_values

        # Build scaler params for later inverse transform
        scaler_params: dict[str, Any] = {
            "method": method,
            "columns": numeric_cols,
        }

        if isinstance(scaler, StandardScaler):
            scaler_params["mean"] = scaler.mean_.tolist()
            scaler_params["scale"] = scaler.scale_.tolist()
        elif isinstance(scaler, MinMaxScaler):
            scaler_params["min"] = scaler.data_min_.tolist()
            scaler_params["max"] = scaler.data_max_.tolist()
            scaler_params["scale"] = scaler.scale_.tolist()
        elif isinstance(scaler, RobustScaler):
            scaler_params["center"] = scaler.center_.tolist()
            scaler_params["scale"] = scaler.scale_.tolist()

        self._log.info(
            "features_normalized",
            method=method,
            n_columns=len(numeric_cols),
        )

        return normalized_df, scaler_params

    def create_target_variable(
        self,
        price_data: pd.DataFrame,
        horizon_days: int = 5,
        threshold: float = 0.02,
    ) -> pd.Series:
        """Create a binary classification target from forward returns.

        The target is 1 if the price moves up by more than *threshold*
        over the next *horizon_days*, and 0 otherwise.

        Args:
            price_data: DataFrame containing at least a ``"close"`` column,
                indexed by date.
            horizon_days: Number of trading days to look forward.
            threshold: Minimum forward return (as a decimal, e.g. 0.02
                for 2%) to label as positive.

        Returns:
            Binary Series aligned with *price_data* index.  The last
            *horizon_days* rows will be NaN (no future data available).
        """
        self._log.info(
            "creating_target_variable",
            horizon_days=horizon_days,
            threshold=threshold,
            n_rows=len(price_data),
        )

        if "close" not in price_data.columns:
            raise ValueError("price_data must contain a 'close' column")

        close = price_data["close"]

        # Forward return over the horizon
        forward_return = close.shift(-horizon_days) / close - 1.0

        # Binary classification: 1 if up > threshold, 0 otherwise
        target = (forward_return > threshold).astype(int)

        # The last horizon_days entries have no forward data
        target.iloc[-horizon_days:] = np.nan

        n_positive = int(target.sum())
        n_valid = int(target.notna().sum())
        positive_rate = n_positive / max(n_valid, 1)

        self._log.info(
            "target_variable_created",
            n_valid=n_valid,
            n_positive=n_positive,
            positive_rate=round(positive_rate, 4),
            n_nan=int(target.isna().sum()),
        )

        target.name = "target"
        return target

    def add_lagged_features(
        self,
        df: pd.DataFrame,
        columns: list[str],
        lags: list[int] | None = None,
    ) -> pd.DataFrame:
        """Add lagged versions of specified columns to the DataFrame.

        For each column and lag period, creates a new column named
        ``{column}_lag{lag}`` (e.g. ``RSI_14_lag1``, ``RSI_14_lag2``).

        Args:
            df: Input feature DataFrame.
            columns: Column names to create lagged versions of.
            lags: Lag periods to generate.  Defaults to ``[1, 2, 3, 5]``.

        Returns:
            DataFrame with additional lagged feature columns appended.
        """
        if lags is None:
            lags = [1, 2, 3, 5]

        self._log.info(
            "adding_lagged_features",
            n_columns=len(columns),
            lags=lags,
        )

        result = df.copy()
        new_col_count = 0

        for col in columns:
            if col not in result.columns:
                self._log.warning(
                    "skipping_missing_column_for_lag",
                    column=col,
                )
                continue

            for lag in lags:
                lag_col_name = f"{col}_lag{lag}"
                result[lag_col_name] = result[col].shift(lag)
                new_col_count += 1

        self._log.info(
            "lagged_features_added",
            new_columns=new_col_count,
            total_columns=len(result.columns),
        )

        return result

    def remove_correlated_features(
        self,
        df: pd.DataFrame,
        threshold: float = 0.95,
    ) -> pd.DataFrame:
        """Remove features with pairwise correlation above the threshold.

        When two features are highly correlated, the one with lower
        feature importance (if available from a prior ``select_features``
        call) is dropped.  If importances are not available, the second
        column (in order) is dropped.

        Args:
            df: Feature DataFrame.
            threshold: Absolute correlation coefficient above which one
                of the two features is removed.

        Returns:
            DataFrame with correlated features removed.
        """
        self._log.info(
            "removing_correlated_features",
            threshold=threshold,
            n_features=len(df.columns),
        )

        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.empty:
            return df

        corr_matrix = numeric_df.corr().abs()

        # Zero out the diagonal and lower triangle
        upper_triangle = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        upper_corr = corr_matrix.where(upper_triangle)

        to_drop: set[str] = set()

        for col in upper_corr.columns:
            highly_correlated = upper_corr.index[upper_corr[col] > threshold].tolist()
            for correlated_col in highly_correlated:
                if correlated_col in to_drop:
                    continue

                # Choose which to drop: keep the one with higher importance
                imp_col = self._feature_importances.get(col, 0.0)
                imp_corr = self._feature_importances.get(correlated_col, 0.0)

                drop_candidate = correlated_col if imp_col >= imp_corr else col
                to_drop.add(drop_candidate)

                self._log.debug(
                    "dropping_correlated_feature",
                    kept=col if drop_candidate == correlated_col else correlated_col,
                    dropped=drop_candidate,
                    correlation=round(
                        float(
                            upper_corr.loc[correlated_col, col]
                            if correlated_col in upper_corr.index
                            else 0.0
                        ),
                        4,
                    ),
                )

        result = df.drop(columns=[c for c in to_drop if c in df.columns])

        self._log.info(
            "correlated_features_removed",
            n_dropped=len(to_drop),
            n_remaining=len(result.columns),
        )

        return result

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def feature_importances(self) -> dict[str, float]:
        """Return the cached feature importances from the last selection."""
        return dict(self._feature_importances)

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    def _add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add cross-feature interactions and ratios.

        Creates interaction terms between key features to capture
        non-linear relationships that tree models might otherwise
        need deeper splits to discover.

        Args:
            df: Single-ticker feature DataFrame.

        Returns:
            DataFrame with additional derived columns.
        """
        result = df.copy()

        # RSI x Volume ratio interaction (momentum confirmed by volume)
        if "rsi_14" in result.columns and "vol_ratio_20" in result.columns:
            result["rsi_x_vol_ratio"] = result["rsi_14"] * result["vol_ratio_20"]

        # Trend strength x ADX interaction
        if "trend_strength" in result.columns and "adx_14" in result.columns:
            result["trend_x_adx"] = result["trend_strength"] * result["adx_14"]

        # Bollinger bandwidth x ATR percentile
        if "bb_bandwidth" in result.columns and "atr_percentile" in result.columns:
            result["bb_width_x_atr_pctl"] = (
                result["bb_bandwidth"] * result["atr_percentile"]
            )

        # HV ratio x RSI (volatility-adjusted momentum)
        if "hv_ratio_10_60" in result.columns and "rsi_14" in result.columns:
            result["hv_ratio_x_rsi"] = result["hv_ratio_10_60"] * result["rsi_14"]

        # MACD histogram x OBV slope (trend confirmation)
        if "macd_histogram" in result.columns and "obv" in result.columns:
            obv_slope = result["obv"].pct_change(periods=5)
            result["macd_hist_x_obv_slope"] = result["macd_histogram"] * obv_slope

        # Distance from 52w high x CMF (buying near highs with inflows)
        if "dist_from_52w_high" in result.columns and "cmf_20" in result.columns:
            result["dist_52w_high_x_cmf"] = (
                result["dist_from_52w_high"] * result["cmf_20"]
            )

        return result

    def _select_by_importance(
        self,
        X: pd.DataFrame,  # noqa: N803
        y: pd.Series,
    ) -> list[str]:
        """Select features using XGBoost feature importances.

        Trains a quick XGBoost model and retains features whose
        normalized importance exceeds the configured threshold.

        Args:
            X: Feature matrix.
            y: Binary target.

        Returns:
            Sorted list of selected feature names.
        """
        # Drop NaN rows for training
        valid_mask = y.notna()
        X_clean = X.loc[valid_mask].copy()  # noqa: N806
        y_clean = y.loc[valid_mask].copy()

        if len(X_clean) < 50:
            self._log.warning(
                "insufficient_samples_for_importance_selection",
                n_samples=len(X_clean),
                returning_all_features=True,
            )
            return sorted(X.columns.tolist())

        model = XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42,
            verbosity=0,
            n_jobs=-1,
        )

        model.fit(X_clean, y_clean)

        # Extract and normalize importances
        raw_importances = model.feature_importances_
        total_importance = raw_importances.sum()
        if total_importance > 0:
            normalized = raw_importances / total_importance
        else:
            normalized = raw_importances

        importance_dict: dict[str, float] = {}
        for col_name, imp_value in zip(X.columns, normalized, strict=False):
            importance_dict[col_name] = float(imp_value)

        # Cache for use in correlated feature removal
        self._feature_importances = importance_dict

        # Filter by threshold
        selected = sorted(
            [
                name
                for name, imp in importance_dict.items()
                if imp >= self._importance_threshold
            ]
        )

        self._log.debug(
            "importance_selection_complete",
            top_5=sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:5],
        )

        return selected

    def _select_by_mutual_info(
        self,
        X: pd.DataFrame,  # noqa: N803
        y: pd.Series,
    ) -> list[str]:
        """Select features using mutual information scores.

        Computes mutual information between each feature and the target,
        normalizes the scores, and retains features above the threshold.

        Args:
            X: Feature matrix.
            y: Binary target.

        Returns:
            Sorted list of selected feature names.
        """
        valid_mask = y.notna()
        X_clean = X.loc[valid_mask].copy()  # noqa: N806
        y_clean = y.loc[valid_mask].copy()

        if len(X_clean) < 50:
            self._log.warning(
                "insufficient_samples_for_mutual_info_selection",
                n_samples=len(X_clean),
                returning_all_features=True,
            )
            return sorted(X.columns.tolist())

        # Replace any remaining NaN/inf with 0 for MI calculation
        X_clean = X_clean.replace([np.inf, -np.inf], np.nan).fillna(0)  # noqa: N806

        mi_scores = mutual_info_classif(
            X_clean,
            y_clean,
            random_state=42,
            n_neighbors=5,
        )

        total_mi = mi_scores.sum()
        normalized_mi = mi_scores / total_mi if total_mi > 0 else mi_scores

        mi_dict: dict[str, float] = {}
        for col_name, mi_value in zip(X.columns, normalized_mi, strict=False):
            mi_dict[col_name] = float(mi_value)

        self._feature_importances = mi_dict

        selected = sorted(
            [name for name, mi in mi_dict.items() if mi >= self._importance_threshold]
        )

        return selected

    def _select_by_variance(self, X: pd.DataFrame) -> list[str]:  # noqa: N803
        """Select features by removing near-zero variance columns.

        Drops columns whose variance falls below a fraction of the mean
        variance across all features.

        Args:
            X: Feature matrix.

        Returns:
            Sorted list of selected feature names.
        """
        variances = X.var()
        mean_variance = variances.mean()

        if mean_variance == 0:
            self._log.warning("all_features_zero_variance")
            return sorted(X.columns.tolist())

        # Use importance_threshold as the fraction of mean variance
        variance_threshold = self._importance_threshold * mean_variance

        selected = sorted(variances[variances >= variance_threshold].index.tolist())

        return selected
