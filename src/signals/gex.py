"""Gamma Exposure (GEX) calculation and dealer positioning analysis.

Computes per-strike and aggregate Gamma Exposure from an options chain,
identifies key price levels (Call Wall, Put Wall, Volatility Trigger),
determines the gamma regime, and produces a trading signal indicating
whether the market microstructure favours range-bound or directional
strategies.

GEX Formula::

    call_gex = call_OI * call_gamma * 100 * spot_price
    put_gex  = put_OI  * put_gamma  * 100 * (-spot_price)
    net_gex  = sum(call_gex + put_gex) across all strikes

Usage::

    from src.signals.gex import GammaExposureCalculator

    calc = GammaExposureCalculator()
    profile = calc.calculate_gex(options_chain, spot_price=450.0)
    levels = calc.identify_levels(
        {s.strike: s.net_gex for s in profile.gex_by_strike},
        spot_price=450.0,
    )
    regime = calc.determine_regime(profile.net_gex, 450.0, levels.vol_trigger)
    signal = calc.get_gex_signal(profile, levels, 450.0)
    print(signal.score, signal.regime)
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

import structlog
from pydantic import BaseModel, Field

logger: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CONTRACT_MULTIPLIER: int = 100  # standard US equity option multiplier

# Score thresholds for regime-based signal generation
POSITIVE_GEX_BIAS: str = "range_bound"
NEGATIVE_GEX_BIAS: str = "directional"

# Minimum absolute GEX to consider a strike significant (filters noise
# from deep OTM options with near-zero gamma).
MIN_SIGNIFICANT_GEX: float = 1_000_000.0  # $1M notional gamma


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------
class GEXRegime(StrEnum):
    """Dealer gamma regime classification."""

    POSITIVE_GAMMA = "positive_gamma"
    NEGATIVE_GAMMA = "negative_gamma"


class HedgePressure(StrEnum):
    """Direction of expected dealer hedging flow."""

    BUYING = "buying"
    SELLING = "selling"
    NEUTRAL = "neutral"


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------
class StrikeGEX(BaseModel):
    """Gamma exposure breakdown for a single strike price."""

    strike: float = Field(description="Strike price")
    call_gex: float = Field(description="Call-side gamma exposure (USD notional)")
    put_gex: float = Field(description="Put-side gamma exposure (USD notional)")
    net_gex: float = Field(description="Net gamma exposure at this strike (USD)")
    call_oi: int = Field(ge=0, description="Call open interest")
    put_oi: int = Field(ge=0, description="Put open interest")
    call_gamma: float = Field(description="Call gamma value")
    put_gamma: float = Field(description="Put gamma value")


class GEXProfile(BaseModel):
    """Aggregate gamma exposure profile for an underlying."""

    ticker: str = Field(description="Underlying ticker symbol")
    spot_price: float = Field(gt=0, description="Current spot price of the underlying")
    net_gex: float = Field(description="Total net gamma exposure (USD notional)")
    gex_by_strike: list[StrikeGEX] = Field(
        default_factory=list,
        description="Per-strike gamma exposure breakdown",
    )
    total_call_gex: float = Field(
        default=0.0,
        description="Sum of call-side gamma exposure across all strikes",
    )
    total_put_gex: float = Field(
        default=0.0,
        description="Sum of put-side gamma exposure across all strikes (negative)",
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(tz=UTC),
        description="Profile computation time (UTC)",
    )


class GEXLevels(BaseModel):
    """Key gamma-derived price levels for an underlying."""

    call_wall: float | None = Field(
        default=None,
        description=(
            "Strike with highest positive call GEX above spot; "
            "acts as magnetic resistance"
        ),
    )
    put_wall: float | None = Field(
        default=None,
        description=(
            "Strike with highest positive put GEX below spot; acts as magnetic support"
        ),
    )
    vol_trigger: float | None = Field(
        default=None,
        description=(
            "Strike where net GEX flips sign; below this level "
            "dealers become short gamma and volatility increases"
        ),
    )
    zero_gamma: float | None = Field(
        default=None,
        description=("Strike where cumulative dealer gamma crosses zero"),
    )
    max_pain: float | None = Field(
        default=None,
        description=(
            "Strike at which total open interest premium loss "
            "is minimised (maximum pain for option holders)"
        ),
    )


class DealerPosition(BaseModel):
    """Estimated aggregate dealer positioning derived from GEX."""

    net_dealer_delta: float = Field(
        description="Estimated net dealer delta exposure",
    )
    net_dealer_gamma: float = Field(
        description="Estimated net dealer gamma exposure",
    )
    hedge_pressure: str = Field(
        description="Expected dealer hedging direction: buying, selling, or neutral",
    )


class GEXSignal(BaseModel):
    """Aggregated gamma exposure signal for the ensemble meta-learner."""

    ticker: str = Field(description="Underlying ticker symbol")
    score: float = Field(
        ge=-1.0,
        le=1.0,
        description=(
            "GEX-derived signal: -1.0 (negative gamma, expect vol) "
            "to +1.0 (positive gamma, expect stability)"
        ),
    )
    regime: str = Field(
        description="Gamma regime: positive_gamma or negative_gamma",
    )
    net_gex: float = Field(description="Total net GEX (USD notional)")
    levels: GEXLevels = Field(description="Key gamma-derived price levels")
    bias: str = Field(
        description=(
            "Strategy bias: 'range_bound' for positive gamma, "
            "'directional' for negative gamma"
        ),
    )
    calculated_at: datetime = Field(
        default_factory=lambda: datetime.now(tz=UTC),
        description="Signal computation time (UTC)",
    )


# ---------------------------------------------------------------------------
# GammaExposureCalculator
# ---------------------------------------------------------------------------
class GammaExposureCalculator:
    """Calculates Gamma Exposure from an options chain and derives trading signals.

    Gamma Exposure (GEX) measures the aggregate gamma held by market makers
    (dealers) across all strikes.  When dealers are net long gamma (positive
    GEX), their hedging activity dampens price swings, creating a
    range-bound environment.  When dealers are net short gamma (negative
    GEX), their hedging amplifies moves, creating a volatile / trending
    environment.

    This class computes per-strike GEX, identifies key price levels (Call
    Wall, Put Wall, Volatility Trigger), determines the gamma regime, and
    produces a normalised signal for downstream consumption.
    """

    def __init__(self) -> None:
        self._log: structlog.stdlib.BoundLogger = logger.bind(
            component="GammaExposureCalculator",
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def calculate_gex(
        self,
        options_chain: list[dict[str, Any]],
        spot_price: float,
    ) -> GEXProfile:
        """Calculate per-strike and aggregate gamma exposure.

        For each strike in the chain the formula is::

            call_gex = call_OI * call_gamma * 100 * spot_price
            put_gex  = put_OI  * put_gamma  * 100 * (-spot_price)
            net_gex  = call_gex + put_gex

        The sign convention follows dealer positioning: dealers are
        assumed to be net short customer positions, so calls contribute
        positive GEX (dealers hedge by buying the underlying when spot
        rises) and puts contribute negative GEX.

        Args:
            options_chain: List of dicts, each containing at minimum:
                ``strike``, ``call_oi``, ``put_oi``, ``call_gamma``,
                ``put_gamma``.  Additional fields are ignored.
            spot_price: Current spot price of the underlying.

        Returns:
            A :class:`GEXProfile` with per-strike and aggregate data.
        """
        if not options_chain:
            self._log.warning("empty_options_chain")
            return GEXProfile(
                ticker="",
                spot_price=spot_price,
                net_gex=0.0,
            )

        ticker = str(options_chain[0].get("ticker", options_chain[0].get("symbol", "")))

        strike_gex_list: list[StrikeGEX] = []
        total_call_gex: float = 0.0
        total_put_gex: float = 0.0

        # Aggregate by strike in case the chain has separate call/put rows
        strike_data: dict[float, dict[str, float]] = {}
        for entry in options_chain:
            strike = float(entry.get("strike", 0.0))
            if strike <= 0:
                continue

            if strike not in strike_data:
                strike_data[strike] = {
                    "call_oi": 0.0,
                    "put_oi": 0.0,
                    "call_gamma": 0.0,
                    "put_gamma": 0.0,
                }

            # Accept either pre-separated call/put fields or a right indicator
            right = str(entry.get("right", entry.get("option_type", ""))).upper()
            oi = float(entry.get("open_interest", entry.get("oi", 0)))
            gamma = float(entry.get("gamma", 0.0))

            if right in ("C", "CALL"):
                strike_data[strike]["call_oi"] += oi
                # Use the last gamma seen (or max for safety)
                if gamma > strike_data[strike]["call_gamma"]:
                    strike_data[strike]["call_gamma"] = gamma
            elif right in ("P", "PUT"):
                strike_data[strike]["put_oi"] += oi
                if gamma > strike_data[strike]["put_gamma"]:
                    strike_data[strike]["put_gamma"] = gamma
            else:
                # Entries may already have call_oi / put_oi split
                strike_data[strike]["call_oi"] += float(
                    entry.get("call_oi", entry.get("call_open_interest", 0))
                )
                strike_data[strike]["put_oi"] += float(
                    entry.get("put_oi", entry.get("put_open_interest", 0))
                )
                strike_data[strike]["call_gamma"] = max(
                    strike_data[strike]["call_gamma"],
                    float(entry.get("call_gamma", 0.0)),
                )
                strike_data[strike]["put_gamma"] = max(
                    strike_data[strike]["put_gamma"],
                    float(entry.get("put_gamma", 0.0)),
                )

        for strike in sorted(strike_data):
            sd = strike_data[strike]
            call_oi = int(sd["call_oi"])
            put_oi = int(sd["put_oi"])
            call_gamma = sd["call_gamma"]
            put_gamma = sd["put_gamma"]

            call_gex = call_oi * call_gamma * CONTRACT_MULTIPLIER * spot_price
            put_gex = put_oi * put_gamma * CONTRACT_MULTIPLIER * (-spot_price)
            net_gex = call_gex + put_gex

            total_call_gex += call_gex
            total_put_gex += put_gex

            strike_gex_list.append(
                StrikeGEX(
                    strike=strike,
                    call_gex=round(call_gex, 2),
                    put_gex=round(put_gex, 2),
                    net_gex=round(net_gex, 2),
                    call_oi=call_oi,
                    put_oi=put_oi,
                    call_gamma=call_gamma,
                    put_gamma=put_gamma,
                )
            )

        net_gex = total_call_gex + total_put_gex

        profile = GEXProfile(
            ticker=ticker,
            spot_price=spot_price,
            net_gex=round(net_gex, 2),
            gex_by_strike=strike_gex_list,
            total_call_gex=round(total_call_gex, 2),
            total_put_gex=round(total_put_gex, 2),
        )

        self._log.info(
            "gex_calculated",
            ticker=ticker,
            spot_price=spot_price,
            net_gex=profile.net_gex,
            total_call_gex=profile.total_call_gex,
            total_put_gex=profile.total_put_gex,
            num_strikes=len(strike_gex_list),
        )

        return profile

    def identify_levels(
        self,
        gex_by_strike: dict[float, float],
        spot_price: float,
    ) -> GEXLevels:
        """Identify key gamma-derived price levels.

        Scans per-strike net GEX to locate:

        - **Call Wall**: strike with the highest positive net GEX *above*
          spot.  This level acts as magnetic resistance because dealer
          hedging activity absorbs buying pressure.
        - **Put Wall**: strike with the most negative net GEX (i.e. the
          most negative put_gex) *below* spot.  Acts as magnetic support.
        - **Volatility Trigger**: the strike nearest to spot where net GEX
          transitions from positive to negative.  Below this level, dealer
          hedging amplifies volatility.
        - **Zero Gamma**: the strike at which cumulative net GEX (summed
          from the lowest strike upward) crosses zero.
        - **Max Pain**: the strike at which total open-interest-weighted
          premium loss is minimised.  (Approximated here as the strike
          nearest to zero cumulative net GEX.)

        Args:
            gex_by_strike: Mapping of strike price to net GEX at that
                strike (USD notional).
            spot_price: Current underlying spot price.

        Returns:
            A :class:`GEXLevels` with identified price levels, or ``None``
            for any level that cannot be determined.
        """
        if not gex_by_strike:
            self._log.warning("empty_gex_by_strike")
            return GEXLevels()

        sorted_strikes = sorted(gex_by_strike.keys())

        # -- Call Wall: highest positive net GEX above spot ----------------
        call_wall: float | None = None
        call_wall_gex: float = 0.0
        for strike in sorted_strikes:
            gex_val = gex_by_strike[strike]
            if strike > spot_price and gex_val > call_wall_gex:
                call_wall_gex = gex_val
                call_wall = strike

        # -- Put Wall: most negative net GEX below spot --------------------
        #    (highest absolute put-side exposure)
        put_wall: float | None = None
        put_wall_gex: float = 0.0
        for strike in sorted_strikes:
            gex_val = gex_by_strike[strike]
            if strike < spot_price and gex_val < put_wall_gex:
                put_wall_gex = gex_val
                put_wall = strike

        # -- Volatility Trigger: closest sign-flip strike to spot ----------
        vol_trigger: float | None = None
        min_vol_trigger_dist: float = float("inf")
        for i in range(len(sorted_strikes) - 1):
            gex_curr = gex_by_strike[sorted_strikes[i]]
            gex_next = gex_by_strike[sorted_strikes[i + 1]]

            # Look for a positive-to-negative transition (above to below)
            if gex_curr > 0 and gex_next <= 0:
                midpoint = (sorted_strikes[i] + sorted_strikes[i + 1]) / 2.0
                dist = abs(midpoint - spot_price)
                if dist < min_vol_trigger_dist:
                    min_vol_trigger_dist = dist
                    vol_trigger = midpoint

        # -- Zero Gamma: cumulative sum crossing zero ----------------------
        zero_gamma: float | None = None
        cumulative: float = 0.0
        prev_cumulative: float = 0.0
        for strike in sorted_strikes:
            prev_cumulative = cumulative
            cumulative += gex_by_strike[strike]
            if prev_cumulative != 0.0 and (
                (prev_cumulative > 0 and cumulative <= 0)
                or (prev_cumulative < 0 and cumulative >= 0)
            ):
                zero_gamma = strike
                break

        # -- Max Pain approximation ----------------------------------------
        # Approximate max pain as the strike nearest to zero cumulative GEX.
        max_pain: float | None = None
        cumulative = 0.0
        min_abs_cumulative: float = float("inf")
        for strike in sorted_strikes:
            cumulative += gex_by_strike[strike]
            if abs(cumulative) < min_abs_cumulative:
                min_abs_cumulative = abs(cumulative)
                max_pain = strike

        levels = GEXLevels(
            call_wall=call_wall,
            put_wall=put_wall,
            vol_trigger=vol_trigger,
            zero_gamma=zero_gamma,
            max_pain=max_pain,
        )

        self._log.info(
            "gex_levels_identified",
            spot_price=spot_price,
            call_wall=levels.call_wall,
            put_wall=levels.put_wall,
            vol_trigger=levels.vol_trigger,
            zero_gamma=levels.zero_gamma,
            max_pain=levels.max_pain,
        )

        return levels

    def determine_regime(
        self,
        net_gex: float,
        spot_price: float,
        vol_trigger: float | None,
    ) -> str:
        """Determine the gamma regime based on net GEX and the volatility trigger.

        - **Positive gamma** (spot above vol trigger or net GEX > 0):
          dealers are long gamma.  Their hedging dampens price movement,
          creating a mean-reverting, range-bound environment.
        - **Negative gamma** (spot below vol trigger or net GEX < 0):
          dealers are short gamma.  Their hedging amplifies price movement,
          creating a trending or volatile environment.

        Args:
            net_gex: Total net gamma exposure (USD notional).
            spot_price: Current underlying spot price.
            vol_trigger: Volatility trigger level, or ``None`` if it
                could not be determined.

        Returns:
            Regime string: ``"positive_gamma"`` or ``"negative_gamma"``.
        """
        if vol_trigger is not None:
            if spot_price >= vol_trigger:
                regime = GEXRegime.POSITIVE_GAMMA
            else:
                regime = GEXRegime.NEGATIVE_GAMMA
        else:
            # Fallback: use sign of net GEX
            regime = (
                GEXRegime.POSITIVE_GAMMA if net_gex >= 0 else GEXRegime.NEGATIVE_GAMMA
            )

        self._log.debug(
            "gex_regime_determined",
            regime=regime.value,
            net_gex=net_gex,
            spot_price=spot_price,
            vol_trigger=vol_trigger,
        )

        return regime.value

    def calculate_dealer_positioning(
        self,
        options_chain: list[dict[str, Any]],
        spot_price: float,
    ) -> DealerPosition:
        """Estimate aggregate dealer delta and gamma exposure.

        Dealers are typically on the other side of customer flow: they are
        short what customers buy and long what customers sell.  For the
        purpose of this estimate, we assume customers are net long calls
        and net long puts (the dominant retail positioning), so dealers
        are net short calls and net short puts.

        Dealer delta per option::

            dealer_delta_call = -call_OI * call_delta * 100
            dealer_delta_put  = -put_OI  * put_delta  * 100

        Dealer gamma per option::

            dealer_gamma = -(call_OI * call_gamma + put_OI * put_gamma) * 100

        Hedge pressure is derived from the net dealer delta:

        - Positive dealer delta -> dealers must sell to hedge -> selling pressure.
        - Negative dealer delta -> dealers must buy to hedge -> buying pressure.

        Args:
            options_chain: List of dicts with per-strike option data
                including ``strike``, ``call_oi``, ``put_oi``,
                ``call_delta``, ``put_delta``, ``call_gamma``, ``put_gamma``.
            spot_price: Current underlying spot price.

        Returns:
            A :class:`DealerPosition` with estimated delta, gamma, and
            inferred hedge pressure direction.
        """
        if not options_chain:
            return DealerPosition(
                net_dealer_delta=0.0,
                net_dealer_gamma=0.0,
                hedge_pressure=HedgePressure.NEUTRAL.value,
            )

        net_delta: float = 0.0
        net_gamma: float = 0.0

        for entry in options_chain:
            right = str(entry.get("right", entry.get("option_type", ""))).upper()
            oi = float(entry.get("open_interest", entry.get("oi", 0)))
            delta = float(entry.get("delta", 0.0))
            gamma = float(entry.get("gamma", 0.0))

            if right in ("C", "CALL"):
                # Dealers short calls: delta contribution = -OI * delta * multiplier
                net_delta += -oi * delta * CONTRACT_MULTIPLIER
                net_gamma += -oi * gamma * CONTRACT_MULTIPLIER
            elif right in ("P", "PUT"):
                # Dealers short puts: delta contribution = -OI * delta * multiplier
                # (put delta is negative, so -OI * negative_delta = positive)
                net_delta += -oi * delta * CONTRACT_MULTIPLIER
                net_gamma += -oi * gamma * CONTRACT_MULTIPLIER
            else:
                # Pre-split format with call_oi / put_oi
                call_oi = float(
                    entry.get("call_oi", entry.get("call_open_interest", 0))
                )
                put_oi = float(entry.get("put_oi", entry.get("put_open_interest", 0)))
                call_delta = float(entry.get("call_delta", 0.0))
                put_delta = float(entry.get("put_delta", 0.0))
                call_gamma = float(entry.get("call_gamma", 0.0))
                put_gamma = float(entry.get("put_gamma", 0.0))

                net_delta += -call_oi * call_delta * CONTRACT_MULTIPLIER
                net_delta += -put_oi * put_delta * CONTRACT_MULTIPLIER
                net_gamma += (
                    -(call_oi * call_gamma + put_oi * put_gamma) * CONTRACT_MULTIPLIER
                )

        # Determine hedge pressure
        # Positive dealer delta means dealers are already long -> they sell to hedge
        # Negative dealer delta means dealers are already short -> they buy to hedge
        HEDGE_THRESHOLD: float = 1000.0  # noqa: N806
        if net_delta > HEDGE_THRESHOLD:
            hedge_pressure = HedgePressure.SELLING
        elif net_delta < -HEDGE_THRESHOLD:
            hedge_pressure = HedgePressure.BUYING
        else:
            hedge_pressure = HedgePressure.NEUTRAL

        position = DealerPosition(
            net_dealer_delta=round(net_delta, 2),
            net_dealer_gamma=round(net_gamma, 2),
            hedge_pressure=hedge_pressure.value,
        )

        self._log.info(
            "dealer_positioning_calculated",
            spot_price=spot_price,
            net_dealer_delta=position.net_dealer_delta,
            net_dealer_gamma=position.net_dealer_gamma,
            hedge_pressure=position.hedge_pressure,
        )

        return position

    def get_gex_signal(
        self,
        gex_profile: GEXProfile,
        gex_levels: GEXLevels,
        spot_price: float,
    ) -> GEXSignal:
        """Produce a normalised GEX-based trading signal.

        Combines the GEX profile, identified levels, and spot price into
        a single signal suitable for the ensemble meta-learner.

        Scoring logic:

        - **Positive gamma regime**: score trends toward +1.0.  Favour
          range-bound, premium-selling strategies (iron condors, calendars,
          short strangles).
        - **Negative gamma regime**: score trends toward -1.0.  Favour
          directional or long-volatility strategies (vertical spreads,
          long straddles).

        The magnitude of the score is scaled by the distance of spot from
        the volatility trigger and the absolute size of the net GEX.

        Args:
            gex_profile: Computed :class:`GEXProfile`.
            gex_levels: Identified :class:`GEXLevels`.
            spot_price: Current underlying spot price.

        Returns:
            A :class:`GEXSignal` with a normalised score in ``[-1.0, 1.0]``.
        """
        regime = self.determine_regime(
            gex_profile.net_gex,
            spot_price,
            gex_levels.vol_trigger,
        )

        # Base score from regime
        if regime == GEXRegime.POSITIVE_GAMMA.value:
            base_score: float = 0.5
        else:
            base_score = -0.5

        # Distance modifier: how far is spot from the vol trigger?
        # Farther into positive gamma territory -> stronger positive score
        # Farther into negative gamma territory -> stronger negative score
        distance_modifier: float = 0.0
        if gex_levels.vol_trigger is not None and spot_price > 0:
            distance_pct = (spot_price - gex_levels.vol_trigger) / spot_price
            # Clamp to [-0.5, 0.5] to keep total score in [-1, 1]
            distance_modifier = max(-0.5, min(0.5, distance_pct * 10.0))

        # Magnitude modifier: larger absolute GEX -> stronger conviction
        # Normalise by spot^2 * 1e6 to get a dimensionless ratio
        magnitude_modifier: float = 0.0
        normalizer = spot_price * spot_price * 1e6
        if normalizer > 0:
            gex_ratio = abs(gex_profile.net_gex) / normalizer
            # Sigmoid-like scaling: caps at ~0.3 for very large GEX
            magnitude_modifier = min(0.3, gex_ratio / (1.0 + gex_ratio))
            # Apply sign: positive GEX adds to score, negative subtracts
            if gex_profile.net_gex < 0:
                magnitude_modifier = -magnitude_modifier

        # Combine components
        raw_score = base_score + distance_modifier + magnitude_modifier
        score = max(-1.0, min(1.0, round(raw_score, 4)))

        # Strategy bias
        if regime == GEXRegime.POSITIVE_GAMMA.value:
            bias = POSITIVE_GEX_BIAS
        else:
            bias = NEGATIVE_GEX_BIAS

        signal = GEXSignal(
            ticker=gex_profile.ticker,
            score=score,
            regime=regime,
            net_gex=gex_profile.net_gex,
            levels=gex_levels,
            bias=bias,
        )

        self._log.info(
            "gex_signal_generated",
            ticker=signal.ticker,
            score=signal.score,
            regime=signal.regime,
            net_gex=signal.net_gex,
            bias=signal.bias,
            call_wall=gex_levels.call_wall,
            put_wall=gex_levels.put_wall,
            vol_trigger=gex_levels.vol_trigger,
        )

        return signal
