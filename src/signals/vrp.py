"""Volatility Risk Premium (VRP) calculation for Project Titan.

Computes the spread between implied volatility and realised volatility,
classifies the VRP regime (rich / fair / cheap), and produces a composite
signal that guides the strategy selector toward selling or buying volatility.

Key metrics calculated:

- **VRP** (IV minus RV): positive when implied vol exceeds realised.
- **IV Rank**: current IV position within the 52-week range.
- **IV Percentile**: percentage of days over the past year with IV below
  the current level.
- **HV/IV Ratio**: historical-vol / implied-vol; below 1.0 means IV is
  expensive relative to realised movement.

Usage::

    from src.signals.vrp import VRPCalculator, VRPResult, VRPSignal

    calculator = VRPCalculator(lookback_days=252)
    vrp_result = calculator.calculate_vrp(iv_current=25.0, rv_current=18.0)
    iv_rank = calculator.calculate_iv_rank(25.0, iv_history_series)
    signal = calculator.get_vrp_signal(
        iv_rank=72.0, iv_percentile=68.0,
        vrp=7.0, hv_iv_ratio=0.72,
    )
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

import numpy as np
from pydantic import BaseModel, Field

from src.utils.logging import get_logger

if TYPE_CHECKING:
    import pandas as pd
    import structlog


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TRADING_DAYS_PER_YEAR: int = 252
"""Number of trading days per year, used to annualise realised volatility."""

VRP_RICH_THRESHOLD: float = 5.0
"""VRP above this level classifies the regime as 'rich' (IV >> RV)."""

VRP_CHEAP_THRESHOLD: float = -2.0
"""VRP below this level classifies the regime as 'cheap' (IV << RV)."""


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class VRPResult(BaseModel):
    """Result of a single Volatility Risk Premium calculation.

    Attributes:
        iv: Current implied volatility (annualised, percentage points).
        rv: Current realised volatility (annualised, percentage points).
        vrp: Volatility risk premium (IV minus RV).
        regime: VRP regime classification -- ``"rich"``, ``"fair"``, or
            ``"cheap"``.
        timestamp: When the calculation was performed.
    """

    iv: float
    rv: float
    vrp: float
    regime: str = Field(
        ...,
        pattern=r"^(rich|fair|cheap)$",
        description="VRP regime: rich, fair, or cheap",
    )
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class VRPSignal(BaseModel):
    """Composite VRP signal combining all volatility metrics.

    The ``score`` ranges from 0.0 (strongly favour selling volatility)
    to 1.0 (strongly favour buying volatility).  The ``bias`` field
    provides a human-readable directional label.

    Attributes:
        iv_rank: IV Rank on a 0--100 scale.
        iv_percentile: IV Percentile on a 0--100 scale.
        vrp: Volatility risk premium (IV minus RV).
        hv_iv_ratio: Historical volatility divided by implied volatility.
        score: Composite signal score (0.0 = sell vol, 1.0 = buy vol).
        bias: Directional bias label -- ``"sell_vol"``, ``"neutral"``,
            or ``"buy_vol"``.
        timestamp: When the signal was generated.
    """

    iv_rank: float = Field(..., ge=0.0, le=100.0)
    iv_percentile: float = Field(..., ge=0.0, le=100.0)
    vrp: float
    hv_iv_ratio: float
    score: float = Field(..., ge=0.0, le=1.0)
    bias: str = Field(
        ...,
        pattern=r"^(sell_vol|neutral|buy_vol)$",
        description="Directional bias: sell_vol, neutral, or buy_vol",
    )
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


# ---------------------------------------------------------------------------
# VRPCalculator
# ---------------------------------------------------------------------------


class VRPCalculator:
    """Calculates Volatility Risk Premium metrics and composite signals.

    Combines implied volatility, realised volatility, IV Rank, IV
    Percentile, and the HV/IV ratio into a single directional signal
    that guides the strategy selector toward selling or buying volatility.

    Args:
        lookback_days: Number of trading days to use for IV Rank, IV
            Percentile, and realised volatility calculations.  Defaults
            to 252 (one trading year).
    """

    def __init__(self, lookback_days: int = 252) -> None:
        self._lookback_days: int = lookback_days
        self._log: structlog.stdlib.BoundLogger = get_logger("signals.vrp")
        self._log.info(
            "vrp_calculator_initialized",
            lookback_days=lookback_days,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def calculate_vrp(
        self,
        iv_current: float,
        rv_current: float,
    ) -> VRPResult:
        """Calculate the Volatility Risk Premium.

        VRP is defined as implied volatility minus realised volatility.
        A positive VRP means options are expensive relative to realised
        movement, favouring premium-selling strategies.

        Regime classification:
        - ``"rich"``: VRP > 5.0 -- IV significantly exceeds RV.
        - ``"fair"``: -2.0 <= VRP <= 5.0 -- normal conditions.
        - ``"cheap"``: VRP < -2.0 -- IV is below RV (unusual).

        Args:
            iv_current: Current annualised implied volatility
                (percentage points, e.g. 25.0 for 25%).
            rv_current: Current annualised realised volatility
                (percentage points, e.g. 18.0 for 18%).

        Returns:
            A :class:`VRPResult` with the VRP value and regime.
        """
        vrp = iv_current - rv_current

        if vrp > VRP_RICH_THRESHOLD:
            regime = "rich"
        elif vrp < VRP_CHEAP_THRESHOLD:
            regime = "cheap"
        else:
            regime = "fair"

        result = VRPResult(
            iv=round(iv_current, 4),
            rv=round(rv_current, 4),
            vrp=round(vrp, 4),
            regime=regime,
        )

        self._log.info(
            "vrp_calculated",
            iv=result.iv,
            rv=result.rv,
            vrp=result.vrp,
            regime=result.regime,
        )

        return result

    def calculate_iv_rank(
        self,
        current_iv: float,
        iv_history: pd.Series,
    ) -> float:
        """Calculate IV Rank on a 0--100 scale.

        IV Rank measures where the current IV sits within the 52-week
        (or configured lookback) range of IV values::

            IV Rank = (Current IV - 52w Low IV) / (52w High IV - 52w Low IV) * 100

        Args:
            current_iv: Current annualised implied volatility.
            iv_history: Historical IV values as a pandas Series.  At
                least two data points are required.

        Returns:
            IV Rank as a float between 0.0 and 100.0.  Returns 50.0 if
            the history is insufficient or the range is zero.
        """
        if iv_history is None or len(iv_history) < 2:
            self._log.warning(
                "iv_rank_insufficient_history",
                history_length=0 if iv_history is None else len(iv_history),
            )
            return 50.0

        # Use the configured lookback window
        history_window = iv_history.tail(self._lookback_days)
        iv_low = float(history_window.min())
        iv_high = float(history_window.max())

        iv_range = iv_high - iv_low
        if iv_range <= 0.0:
            self._log.debug(
                "iv_rank_zero_range",
                iv_low=iv_low,
                iv_high=iv_high,
            )
            return 50.0

        rank = ((current_iv - iv_low) / iv_range) * 100.0

        # Clamp to [0, 100]
        rank = max(0.0, min(100.0, rank))

        self._log.debug(
            "iv_rank_calculated",
            current_iv=round(current_iv, 4),
            iv_low=round(iv_low, 4),
            iv_high=round(iv_high, 4),
            iv_rank=round(rank, 2),
        )

        return round(rank, 2)

    def calculate_iv_percentile(
        self,
        current_iv: float,
        iv_history: pd.Series,
    ) -> float:
        """Calculate IV Percentile on a 0--100 scale.

        IV Percentile is the percentage of days in the lookback period
        where IV was *below* the current level::

            IV Percentile = (days with IV < current) / total days * 100

        This metric is often more informative than IV Rank because it
        accounts for the distribution of IV values, not just extremes.

        Args:
            current_iv: Current annualised implied volatility.
            iv_history: Historical IV values as a pandas Series.  At
                least two data points are required.

        Returns:
            IV Percentile as a float between 0.0 and 100.0.  Returns
            50.0 if the history is insufficient.
        """
        if iv_history is None or len(iv_history) < 2:
            self._log.warning(
                "iv_percentile_insufficient_history",
                history_length=0 if iv_history is None else len(iv_history),
            )
            return 50.0

        # Use the configured lookback window
        history_window = iv_history.tail(self._lookback_days)

        # Drop NaN values from history
        clean_history = history_window.dropna()
        if len(clean_history) < 2:
            return 50.0

        days_below = int(np.sum(clean_history < current_iv))
        total_days = len(clean_history)

        percentile = (days_below / total_days) * 100.0

        # Clamp to [0, 100]
        percentile = max(0.0, min(100.0, percentile))

        self._log.debug(
            "iv_percentile_calculated",
            current_iv=round(current_iv, 4),
            days_below=days_below,
            total_days=total_days,
            iv_percentile=round(percentile, 2),
        )

        return round(percentile, 2)

    def calculate_realized_vol(
        self,
        prices: pd.Series,
        window: int = 20,
    ) -> float:
        """Calculate annualised realised volatility from a price series.

        Uses the standard deviation of log returns over the specified
        window, annualised by multiplying by sqrt(252).

        Args:
            prices: Daily closing prices as a pandas Series.  Must
                contain at least ``window + 1`` data points.
            window: Number of trading days for the volatility window.
                Defaults to 20 (approximately one month).

        Returns:
            Annualised realised volatility as a percentage (e.g. 18.5
            for 18.5%).  Returns 0.0 if insufficient data.
        """
        if prices is None or len(prices) < window + 1:
            self._log.warning(
                "realized_vol_insufficient_data",
                data_length=0 if prices is None else len(prices),
                required_minimum=window + 1,
            )
            return 0.0

        log_returns = np.log(prices / prices.shift(1)).dropna()

        if len(log_returns) < window:
            self._log.warning(
                "realized_vol_insufficient_returns",
                returns_length=len(log_returns),
                required_window=window,
            )
            return 0.0

        # Use the most recent window of returns
        recent_returns = log_returns.tail(window)
        rv = float(recent_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR))

        # Convert to percentage points
        rv_pct = rv * 100.0

        self._log.debug(
            "realized_vol_calculated",
            window=window,
            realized_vol=round(rv_pct, 4),
        )

        return round(rv_pct, 4)

    def calculate_hv_iv_ratio(
        self,
        historical_vol: float,
        implied_vol: float,
    ) -> float:
        """Calculate the HV/IV ratio.

        A ratio below 1.0 means implied volatility exceeds historical
        (realised) volatility, indicating that options are relatively
        expensive.  A ratio above 1.0 means options are cheap relative
        to realised movement.

        Args:
            historical_vol: Annualised historical (realised) volatility
                (percentage points).
            implied_vol: Annualised implied volatility (percentage
                points).

        Returns:
            HV/IV ratio as a float.  Returns 1.0 if implied vol is zero
            or negative to avoid division errors.
        """
        if implied_vol <= 0.0:
            self._log.warning(
                "hv_iv_ratio_invalid_iv",
                implied_vol=implied_vol,
            )
            return 1.0

        ratio = historical_vol / implied_vol

        self._log.debug(
            "hv_iv_ratio_calculated",
            historical_vol=round(historical_vol, 4),
            implied_vol=round(implied_vol, 4),
            hv_iv_ratio=round(ratio, 4),
        )

        return round(ratio, 4)

    def get_vrp_signal(
        self,
        iv_rank: float,
        iv_percentile: float,
        vrp: float,
        hv_iv_ratio: float,
    ) -> VRPSignal:
        """Generate a composite VRP signal from all volatility metrics.

        Combines IV Rank, IV Percentile, VRP, and HV/IV ratio into a
        single score from 0.0 (sell volatility) to 1.0 (buy volatility).

        Scoring logic:
        - High IV Rank + positive VRP + low HV/IV = sell vol (score near 0)
        - Low IV Rank + negative VRP + high HV/IV = buy vol (score near 1)

        The score is computed as a weighted average of four normalised
        components:
        - IV Rank score (inverted): weight 0.30
        - IV Percentile score (inverted): weight 0.25
        - VRP score (inverted, normalised): weight 0.30
        - HV/IV score (direct): weight 0.15

        Bias classification:
        - score < 0.35: ``"sell_vol"``
        - 0.35 <= score <= 0.65: ``"neutral"``
        - score > 0.65: ``"buy_vol"``

        Args:
            iv_rank: IV Rank on a 0--100 scale.
            iv_percentile: IV Percentile on a 0--100 scale.
            vrp: Volatility risk premium (IV minus RV, in percentage points).
            hv_iv_ratio: Historical vol / implied vol ratio.

        Returns:
            A :class:`VRPSignal` with the composite score and directional
            bias.
        """
        # Component weights
        weight_iv_rank: float = 0.30
        weight_iv_pct: float = 0.25
        weight_vrp: float = 0.30
        weight_hv_iv: float = 0.15

        # Normalise IV Rank to [0, 1] and invert (high rank = low score = sell vol)
        iv_rank_score = 1.0 - max(0.0, min(1.0, iv_rank / 100.0))

        # Normalise IV Percentile and invert
        iv_pct_score = 1.0 - max(0.0, min(1.0, iv_percentile / 100.0))

        # Normalise VRP to [0, 1] using a sigmoid-like mapping and invert
        # VRP of +10 maps to ~0 (sell vol), VRP of -5 maps to ~1 (buy vol)
        vrp_normalised = _sigmoid_normalize(vrp, center=2.0, scale=5.0)
        vrp_score = 1.0 - vrp_normalised

        # HV/IV ratio: direct mapping -- high ratio = buy vol
        # Typical range: 0.5 to 1.5; normalise to [0, 1]
        hv_iv_score = max(0.0, min(1.0, (hv_iv_ratio - 0.5) / 1.0))

        # Weighted composite
        composite = (
            weight_iv_rank * iv_rank_score
            + weight_iv_pct * iv_pct_score
            + weight_vrp * vrp_score
            + weight_hv_iv * hv_iv_score
        )

        # Clamp to [0, 1]
        composite = max(0.0, min(1.0, composite))

        # Classify bias
        if composite < 0.35:
            bias = "sell_vol"
        elif composite > 0.65:
            bias = "buy_vol"
        else:
            bias = "neutral"

        signal = VRPSignal(
            iv_rank=round(iv_rank, 2),
            iv_percentile=round(iv_percentile, 2),
            vrp=round(vrp, 4),
            hv_iv_ratio=round(hv_iv_ratio, 4),
            score=round(composite, 4),
            bias=bias,
        )

        self._log.info(
            "vrp_signal_generated",
            iv_rank=signal.iv_rank,
            iv_percentile=signal.iv_percentile,
            vrp=signal.vrp,
            hv_iv_ratio=signal.hv_iv_ratio,
            score=signal.score,
            bias=signal.bias,
            components={
                "iv_rank_score": round(iv_rank_score, 4),
                "iv_pct_score": round(iv_pct_score, 4),
                "vrp_score": round(vrp_score, 4),
                "hv_iv_score": round(hv_iv_score, 4),
            },
        )

        return signal


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _sigmoid_normalize(
    value: float,
    center: float = 0.0,
    scale: float = 5.0,
) -> float:
    """Normalise a value to [0, 1] using a logistic sigmoid function.

    Maps the input to the range (0, 1) with the inflection point at
    ``center``.  The ``scale`` parameter controls the steepness: larger
    values produce a gentler transition.

    Args:
        value: The raw input value.
        center: The value that maps to 0.5.
        scale: Controls the width of the transition zone.

    Returns:
        A float in [0, 1].
    """
    exponent = -(value - center) / max(scale, 0.01)

    # Clamp exponent to avoid overflow
    exponent = max(-20.0, min(20.0, exponent))

    return 1.0 / (1.0 + np.exp(exponent))
