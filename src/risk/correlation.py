"""Rolling correlation monitoring for portfolio concentration risk.

Calculates pairwise rolling correlations between tickers in the portfolio
and blocks new entries when adding a position would create excessive
correlation with existing holdings.

Usage::

    from src.risk.correlation import CorrelationMonitor

    monitor = CorrelationMonitor(risk_config=config["correlation"])
    corr_matrix = await monitor.calculate_pairwise_correlations(
        tickers=["AAPL", "MSFT"],
        price_history={"AAPL": aapl_series, "MSFT": msft_series},
    )
    too_correlated, reason = monitor.is_too_correlated(
        "GOOG", ["AAPL", "MSFT"], corr_matrix,
    )
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from src.utils.logging import get_logger

if TYPE_CHECKING:
    import structlog

# ---------------------------------------------------------------------------
# CorrelationMonitor
# ---------------------------------------------------------------------------


class CorrelationMonitor:
    """Monitors pairwise correlations to prevent portfolio concentration.

    Calculates rolling correlations between tickers using daily returns
    and rejects new entries when the proposed ticker is too highly
    correlated with existing positions.

    Args:
        risk_config: The ``correlation`` section from ``risk_limits.yaml``.
    """

    def __init__(self, risk_config: dict[str, Any]) -> None:
        self._log: structlog.stdlib.BoundLogger = get_logger("risk.correlation")
        self._config = risk_config

        self._max_pairwise_correlation: float = float(
            self._config.get("max_pairwise_correlation", 0.80)
        )
        self._rolling_window_days: int = int(
            self._config.get("rolling_window_days", 60)
        )
        self._enabled: bool = bool(self._config.get("correlation_check_enabled", True))

        self._log.info(
            "correlation_monitor_initialized",
            max_pairwise_correlation=self._max_pairwise_correlation,
            rolling_window_days=self._rolling_window_days,
            enabled=self._enabled,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def calculate_pairwise_correlations(
        self,
        tickers: list[str],
        price_history: dict[str, pd.Series],
    ) -> pd.DataFrame:
        """Calculate the rolling pairwise correlation matrix.

        Converts price series to daily log returns and computes a rolling
        correlation matrix over the configured window.  The returned
        matrix uses the most recent rolling window observation.

        Args:
            tickers: List of ticker symbols to include in the matrix.
            price_history: Mapping of ticker symbol to a pandas Series of
                daily closing prices indexed by date.

        Returns:
            A pandas DataFrame containing the pairwise correlation matrix.
            Index and columns are ticker symbols.
        """
        if len(tickers) < 2:
            self._log.debug(
                "correlation_skipped_insufficient_tickers",
                ticker_count=len(tickers),
            )
            return pd.DataFrame(
                1.0,
                index=tickers,
                columns=tickers,
            )

        # Build a returns DataFrame from price histories
        returns_data: dict[str, pd.Series] = {}
        for ticker in tickers:
            prices = price_history.get(ticker)
            if prices is None or len(prices) < 2:
                self._log.debug(
                    "correlation_missing_price_data",
                    ticker=ticker,
                    data_points=0 if prices is None else len(prices),
                )
                continue
            # Daily log returns
            returns_data[ticker] = np.log(prices / prices.shift(1)).dropna()

        if len(returns_data) < 2:
            self._log.warning(
                "correlation_insufficient_return_data",
                tickers_with_data=list(returns_data.keys()),
            )
            available_tickers = list(returns_data.keys()) or tickers
            return pd.DataFrame(
                1.0,
                index=available_tickers,
                columns=available_tickers,
            )

        returns_df = pd.DataFrame(returns_data)

        # Calculate rolling correlation using the configured window
        # Use the last complete window for the current correlation snapshot
        window = min(self._rolling_window_days, len(returns_df))
        if window < 10:
            self._log.warning(
                "correlation_small_window",
                window=window,
                configured_window=self._rolling_window_days,
                available_data_points=len(returns_df),
            )

        recent_returns = returns_df.tail(window)
        correlation_matrix = recent_returns.corr()

        # Fill NaN values with 0 (for tickers with insufficient overlap)
        correlation_matrix = correlation_matrix.fillna(0.0)

        self._log.debug(
            "correlation_matrix_calculated",
            tickers=list(correlation_matrix.columns),
            window=window,
            shape=list(correlation_matrix.shape),
        )

        return correlation_matrix

    def is_too_correlated(
        self,
        ticker: str,
        existing_tickers: list[str],
        correlation_matrix: pd.DataFrame,
    ) -> tuple[bool, str]:
        """Check if a new ticker is too correlated with existing positions.

        Args:
            ticker: The proposed new ticker to add to the portfolio.
            existing_tickers: List of tickers currently in the portfolio.
            correlation_matrix: Pre-calculated pairwise correlation matrix
                that includes the proposed ticker.

        Returns:
            A tuple of ``(too_correlated, reason)``.  ``too_correlated``
            is ``True`` if the ticker has a correlation exceeding
            ``max_pairwise_correlation`` with any existing position.
        """
        if not self._enabled:
            return False, ""

        if not existing_tickers:
            return False, ""

        if ticker not in correlation_matrix.columns:
            self._log.debug(
                "correlation_ticker_not_in_matrix",
                ticker=ticker,
            )
            return False, ""

        high_correlations: list[tuple[str, float]] = []

        for existing_ticker in existing_tickers:
            if existing_ticker not in correlation_matrix.index:
                continue

            corr_value = correlation_matrix.loc[existing_ticker, ticker]

            if abs(corr_value) > self._max_pairwise_correlation:
                high_correlations.append((existing_ticker, float(corr_value)))

        if high_correlations:
            details = ", ".join(f"{t}: {c:.3f}" for t, c in high_correlations)
            reason = (
                f"{ticker} has high correlation with existing positions: "
                f"{details} (limit: {self._max_pairwise_correlation:.2f})"
            )
            self._log.info(
                "correlation_too_high",
                ticker=ticker,
                high_correlations=high_correlations,
                threshold=self._max_pairwise_correlation,
            )
            return True, reason

        return False, ""

    def get_correlation_risk_score(
        self,
        tickers: list[str],
        correlation_matrix: pd.DataFrame,
    ) -> float:
        """Calculate an overall portfolio correlation risk score.

        The score ranges from 0.0 (fully diversified, no correlation)
        to 1.0 (perfectly correlated, fully concentrated).  It is
        computed as the average of the absolute pairwise correlations
        among the provided tickers, excluding self-correlations.

        Args:
            tickers: List of tickers currently in the portfolio.
            correlation_matrix: Pre-calculated pairwise correlation matrix.

        Returns:
            A float between 0.0 and 1.0 representing overall portfolio
            correlation risk.  Higher values indicate greater
            concentration.
        """
        if len(tickers) < 2:
            return 0.0

        # Filter to tickers present in the matrix
        available = [t for t in tickers if t in correlation_matrix.columns]
        if len(available) < 2:
            return 0.0

        # Extract the sub-matrix for portfolio tickers
        sub_matrix = correlation_matrix.loc[available, available]

        # Calculate mean absolute pairwise correlation (excluding diagonal)
        n = len(available)
        total_corr = 0.0
        pair_count = 0

        for i in range(n):
            for j in range(i + 1, n):
                corr_val = sub_matrix.iloc[i, j]
                if not pd.isna(corr_val):
                    total_corr += abs(corr_val)
                    pair_count += 1

        if pair_count == 0:
            return 0.0

        avg_correlation = total_corr / pair_count

        # Clamp to [0, 1]
        risk_score = max(0.0, min(1.0, avg_correlation))

        self._log.debug(
            "correlation_risk_score",
            tickers=available,
            score=round(risk_score, 4),
            avg_pairwise_correlation=round(avg_correlation, 4),
            pair_count=pair_count,
        )

        return round(risk_score, 4)
