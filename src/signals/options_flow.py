"""Unusual options activity detection via Unusual Whales API.

Monitors options flow data for signals of institutional or informed trading
activity.  Detects unusual volume-to-open-interest ratios, aggressive sweeps,
large block trades, and multi-day directional consistency.

Usage::

    from src.signals.options_flow import OptionsFlowAnalyzer

    analyzer = OptionsFlowAnalyzer(api_key="uw_abc123")
    transactions = await analyzer.fetch_flow_data("AAPL")
    activities = analyzer.detect_unusual_activity(transactions)
    signal = analyzer.calculate_flow_score(activities, "AAPL")
    print(signal.score, signal.dominant_direction)
"""

from __future__ import annotations

from collections import defaultdict
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Any

import httpx
import structlog
from pydantic import BaseModel, Field
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
UW_BASE_URL: str = "https://api.unusualwhales.com"
UW_FLOW_ENDPOINT: str = "/api/stock/{ticker}/options-flow"
UW_REQUEST_TIMEOUT: float = 15.0
UW_MAX_RETRIES: int = 3

# Detection thresholds
VOL_OI_THRESHOLD: float = 1.25
BLOCK_PREMIUM_THRESHOLD: float = 500_000.0
SWEEP_MIN_PREMIUM: float = 50_000.0
REPEATED_ACTIVITY_DAYS: int = 3
CONSISTENCY_LOOKBACK_DAYS: int = 5

# Scoring weights
WEIGHT_SWEEP: float = 1.5
WEIGHT_BLOCK: float = 1.25
WEIGHT_REGULAR: float = 1.0
WEIGHT_HIGH_VOL_OI: float = 1.1

# Confidence thresholds per activity type
CONFIDENCE_SWEEP: float = 0.85
CONFIDENCE_BLOCK: float = 0.80
CONFIDENCE_HIGH_VOL_OI: float = 0.65
CONFIDENCE_REPEATED: float = 0.90


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------
class FlowDirection(StrEnum):
    """Direction classification for an options flow transaction."""

    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


class ActivityType(StrEnum):
    """Type of unusual activity detected in options flow."""

    HIGH_VOL_OI = "high_vol_oi"
    SWEEP = "sweep"
    BLOCK = "block"
    REPEATED = "repeated"


class TradeSide(StrEnum):
    """Execution side of the flow transaction relative to the NBBO."""

    ASK = "ask"
    BID = "bid"
    MID = "mid"


class OptionType(StrEnum):
    """Option contract type."""

    CALL = "call"
    PUT = "put"


class TradeType(StrEnum):
    """Execution type of the flow transaction."""

    SWEEP = "sweep"
    BLOCK = "block"
    REGULAR = "regular"


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------
class FlowTransaction(BaseModel):
    """A single options flow transaction from the Unusual Whales feed."""

    ticker: str = Field(description="Underlying ticker symbol")
    timestamp: datetime = Field(description="Transaction timestamp (UTC)")
    option_type: OptionType = Field(description="CALL or PUT")
    strike: float = Field(description="Strike price")
    expiry: str = Field(description="Expiration date (YYYY-MM-DD)")
    premium: float = Field(description="Total premium paid/received in USD")
    volume: int = Field(ge=0, description="Contracts traded")
    open_interest: int = Field(ge=0, description="Open interest at time of trade")
    vol_oi_ratio: float = Field(
        ge=0.0,
        description="Volume-to-open-interest ratio",
    )
    trade_type: TradeType = Field(
        description="Execution type: sweep, block, or regular",
    )
    side: TradeSide = Field(description="Execution side: ask, bid, or mid")


class UnusualActivity(BaseModel):
    """A detected instance of unusual options activity."""

    ticker: str = Field(description="Underlying ticker symbol")
    activity_type: ActivityType = Field(
        description="Type of unusual activity detected",
    )
    direction: FlowDirection = Field(
        description="Inferred directional bias: bullish, bearish, or neutral",
    )
    premium: float = Field(description="Associated premium in USD")
    details: str = Field(description="Human-readable description of the activity")
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score for this activity detection",
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(tz=UTC),
        description="When this activity was detected (UTC)",
    )


class FlowSignal(BaseModel):
    """Aggregated options flow signal for a single ticker."""

    ticker: str = Field(description="Underlying ticker symbol")
    score: float = Field(
        ge=-1.0,
        le=1.0,
        description="Directional score: -1.0 (bearish) to +1.0 (bullish)",
    )
    net_premium: float = Field(
        description="Net directional premium: bullish - bearish (USD)",
    )
    num_unusual: int = Field(
        ge=0,
        description="Number of unusual activity instances detected",
    )
    consistency: float = Field(
        ge=0.0,
        le=1.0,
        description="Multi-day directional consistency score",
    )
    dominant_direction: str = Field(
        description="Overall dominant direction: bullish, bearish, or neutral",
    )
    activities: list[UnusualActivity] = Field(
        default_factory=list,
        description="Individual unusual activity detections",
    )
    calculated_at: datetime = Field(
        default_factory=lambda: datetime.now(tz=UTC),
        description="Timestamp when this signal was computed (UTC)",
    )


# ---------------------------------------------------------------------------
# OptionsFlowAnalyzer
# ---------------------------------------------------------------------------
class OptionsFlowAnalyzer:
    """Detects unusual options activity and produces directional flow signals.

    Integrates with the Unusual Whales API to fetch recent options flow
    data, then applies heuristic filters to identify sweeps, block trades,
    high volume/OI ratios, and multi-day directional persistence.  The
    resulting :class:`FlowSignal` encodes a normalised directional score
    suitable for consumption by the ensemble meta-learner.

    Args:
        api_key: Unusual Whales API key.
        base_url: API base URL override (useful for testing).
        request_timeout: HTTP request timeout in seconds.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = UW_BASE_URL,
        request_timeout: float = UW_REQUEST_TIMEOUT,
    ) -> None:
        self._api_key: str = api_key
        self._base_url: str = base_url.rstrip("/")
        self._request_timeout: float = request_timeout
        self._log: structlog.stdlib.BoundLogger = logger.bind(
            component="OptionsFlowAnalyzer",
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @retry(
        retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.TransportError)),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        stop=stop_after_attempt(UW_MAX_RETRIES),
        reraise=True,
    )
    async def fetch_flow_data(self, ticker: str) -> list[FlowTransaction]:
        """Fetch recent options flow data for a ticker from Unusual Whales.

        Sends a GET request to the Unusual Whales options flow endpoint,
        parses the response into :class:`FlowTransaction` models, and
        returns them sorted by timestamp descending.

        Args:
            ticker: The stock symbol to query (e.g. ``"AAPL"``).

        Returns:
            List of :class:`FlowTransaction` instances ordered newest-first.

        Raises:
            httpx.HTTPStatusError: If the API returns a non-2xx status code
                after all retry attempts.
            httpx.TransportError: If a network-level error persists after
                all retry attempts.
            ValueError: If the API response cannot be parsed.
        """
        url = f"{self._base_url}{UW_FLOW_ENDPOINT.format(ticker=ticker)}"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Accept": "application/json",
        }

        self._log.info("fetching_flow_data", ticker=ticker, url=url)

        async with httpx.AsyncClient(timeout=self._request_timeout) as client:
            response = await client.get(url, headers=headers)
            response.raise_for_status()

        payload: dict[str, Any] = response.json()
        raw_transactions: list[dict[str, Any]] = payload.get("data", [])

        if not raw_transactions:
            self._log.info("no_flow_data", ticker=ticker)
            return []

        transactions: list[FlowTransaction] = []
        for raw in raw_transactions:
            try:
                txn = self._parse_transaction(raw, ticker)
                transactions.append(txn)
            except (KeyError, ValueError, TypeError) as exc:
                self._log.debug(
                    "parse_transaction_failed",
                    ticker=ticker,
                    error=str(exc),
                    raw=str(raw)[:200],
                )

        # Sort newest first
        transactions.sort(key=lambda t: t.timestamp, reverse=True)

        self._log.info(
            "flow_data_fetched",
            ticker=ticker,
            total_transactions=len(transactions),
            sweeps=sum(1 for t in transactions if t.trade_type == TradeType.SWEEP),
            blocks=sum(1 for t in transactions if t.trade_type == TradeType.BLOCK),
        )

        return transactions

    def detect_unusual_activity(
        self,
        transactions: list[FlowTransaction],
    ) -> list[UnusualActivity]:
        """Detect unusual activity patterns in a list of flow transactions.

        Applies four detection filters:

        1. **High volume/OI**: volume-to-open-interest ratio exceeds
           :data:`VOL_OI_THRESHOLD` (1.25).
        2. **Sweeps**: aggressive multi-exchange fills above
           :data:`SWEEP_MIN_PREMIUM`.
        3. **Block trades**: single-fill trades with premium exceeding
           :data:`BLOCK_PREMIUM_THRESHOLD` ($500K).
        4. **Repeated activity**: same directional bias on 3+ distinct days
           within the lookback window.

        Args:
            transactions: Flow transactions to analyse.

        Returns:
            List of :class:`UnusualActivity` instances, one per detected
            anomaly.
        """
        if not transactions:
            return []

        activities: list[UnusualActivity] = []
        ticker = transactions[0].ticker

        # 1. High volume/OI ratio
        for txn in transactions:
            if txn.vol_oi_ratio >= VOL_OI_THRESHOLD and txn.open_interest > 0:
                direction = self._classify_direction(txn)
                activities.append(
                    UnusualActivity(
                        ticker=ticker,
                        activity_type=ActivityType.HIGH_VOL_OI,
                        direction=direction,
                        premium=txn.premium,
                        details=(
                            f"Vol/OI={txn.vol_oi_ratio:.2f} "
                            f"({txn.option_type.value} {txn.strike} "
                            f"{txn.expiry}) vol={txn.volume} OI={txn.open_interest}"
                        ),
                        confidence=CONFIDENCE_HIGH_VOL_OI,
                        timestamp=txn.timestamp,
                    )
                )

        # 2. Sweeps (aggressive multi-exchange fills)
        for txn in transactions:
            if txn.trade_type == TradeType.SWEEP and txn.premium >= SWEEP_MIN_PREMIUM:
                direction = self._classify_direction(txn)
                activities.append(
                    UnusualActivity(
                        ticker=ticker,
                        activity_type=ActivityType.SWEEP,
                        direction=direction,
                        premium=txn.premium,
                        details=(
                            f"Sweep ${txn.premium:,.0f} "
                            f"({txn.option_type.value} {txn.strike} "
                            f"{txn.expiry}) at {txn.side.value}"
                        ),
                        confidence=CONFIDENCE_SWEEP,
                        timestamp=txn.timestamp,
                    )
                )

        # 3. Block trades (large single-fill orders)
        for txn in transactions:
            if (
                txn.trade_type == TradeType.BLOCK
                and txn.premium >= BLOCK_PREMIUM_THRESHOLD
            ):
                direction = self._classify_direction(txn)
                activities.append(
                    UnusualActivity(
                        ticker=ticker,
                        activity_type=ActivityType.BLOCK,
                        direction=direction,
                        premium=txn.premium,
                        details=(
                            f"Block ${txn.premium:,.0f} "
                            f"({txn.option_type.value} {txn.strike} "
                            f"{txn.expiry}) {txn.volume} contracts"
                        ),
                        confidence=CONFIDENCE_BLOCK,
                        timestamp=txn.timestamp,
                    )
                )

        # 4. Repeated activity (same direction on 3+ distinct days)
        repeated = self._detect_repeated_activity(transactions)
        activities.extend(repeated)

        self._log.info(
            "unusual_activity_detected",
            ticker=ticker,
            total_activities=len(activities),
            high_vol_oi=sum(
                1 for a in activities if a.activity_type == ActivityType.HIGH_VOL_OI
            ),
            sweeps=sum(1 for a in activities if a.activity_type == ActivityType.SWEEP),
            blocks=sum(1 for a in activities if a.activity_type == ActivityType.BLOCK),
            repeated=sum(
                1 for a in activities if a.activity_type == ActivityType.REPEATED
            ),
        )

        return activities

    def calculate_flow_score(
        self,
        activities: list[UnusualActivity],
        ticker: str,
    ) -> FlowSignal:
        """Calculate an aggregated directional flow signal from unusual activities.

        Computes a normalised score in ``[-1.0, +1.0]`` by weighting each
        activity's premium by its aggressiveness (sweeps > blocks > regular)
        and direction, then factoring in multi-day consistency.

        Args:
            activities: Unusual activity detections for the ticker.
            ticker: The underlying ticker symbol.

        Returns:
            A :class:`FlowSignal` encapsulating the directional score,
            net premium, consistency, and underlying activity data.
        """
        if not activities:
            return FlowSignal(
                ticker=ticker,
                score=0.0,
                net_premium=0.0,
                num_unusual=0,
                consistency=0.0,
                dominant_direction=FlowDirection.NEUTRAL.value,
                activities=[],
            )

        bullish_premium: float = 0.0
        bearish_premium: float = 0.0

        for activity in activities:
            weight = self._activity_weight(activity)
            weighted_premium = activity.premium * weight

            if activity.direction == FlowDirection.BULLISH:
                bullish_premium += weighted_premium
            elif activity.direction == FlowDirection.BEARISH:
                bearish_premium += weighted_premium
            # Neutral activities do not contribute to directional premium.

        net_premium = bullish_premium - bearish_premium
        total_premium = bullish_premium + bearish_premium

        # Raw directional score: net / total, bounded to [-1, 1]
        raw_score = net_premium / total_premium if total_premium > 0 else 0.0

        # Multi-day consistency multiplier
        consistency = self._calculate_consistency(activities)
        consistency_multiplier = 0.5 + 0.5 * consistency  # range [0.5, 1.0]

        # Final score: raw direction scaled by consistency
        score = max(-1.0, min(1.0, raw_score * consistency_multiplier))

        # Determine dominant direction
        if score > 0.1:
            dominant = FlowDirection.BULLISH.value
        elif score < -0.1:
            dominant = FlowDirection.BEARISH.value
        else:
            dominant = FlowDirection.NEUTRAL.value

        signal = FlowSignal(
            ticker=ticker,
            score=round(score, 4),
            net_premium=round(net_premium, 2),
            num_unusual=len(activities),
            consistency=round(consistency, 4),
            dominant_direction=dominant,
            activities=activities,
        )

        self._log.info(
            "flow_score_calculated",
            ticker=ticker,
            score=signal.score,
            net_premium=signal.net_premium,
            num_unusual=signal.num_unusual,
            consistency=signal.consistency,
            dominant_direction=signal.dominant_direction,
        )

        return signal

    # ------------------------------------------------------------------
    # Direction classification
    # ------------------------------------------------------------------

    def _classify_direction(self, transaction: FlowTransaction) -> FlowDirection:
        """Classify the directional intent of a flow transaction.

        Uses the combination of option type (call/put) and execution side
        (ask/bid/mid) to infer whether the trade is bullish or bearish:

        - Call bought at ask = bullish (aggressive call buying).
        - Call sold at bid  = bearish (aggressive call selling).
        - Put bought at ask = bearish (aggressive put buying).
        - Put sold at bid   = bullish (aggressive put selling).
        - Any trade near the mid is ambiguous and classified as neutral.

        Args:
            transaction: The flow transaction to classify.

        Returns:
            :class:`FlowDirection` indicating inferred direction.
        """
        if transaction.side == TradeSide.MID:
            return FlowDirection.NEUTRAL

        if transaction.option_type == OptionType.CALL:
            if transaction.side == TradeSide.ASK:
                return FlowDirection.BULLISH
            if transaction.side == TradeSide.BID:
                return FlowDirection.BEARISH

        if transaction.option_type == OptionType.PUT:
            if transaction.side == TradeSide.ASK:
                return FlowDirection.BEARISH
            if transaction.side == TradeSide.BID:
                return FlowDirection.BULLISH

        return FlowDirection.NEUTRAL

    # ------------------------------------------------------------------
    # Multi-day consistency
    # ------------------------------------------------------------------

    def _calculate_consistency(
        self,
        activities: list[UnusualActivity],
        days: int = CONSISTENCY_LOOKBACK_DAYS,
    ) -> float:
        """Calculate multi-day directional consistency of unusual activities.

        Groups activities by calendar day and checks whether the dominant
        direction is consistent across days.  A higher return value means
        more days agree on the same directional bias.

        Args:
            activities: List of unusual activity detections.
            days: Number of lookback days to consider.

        Returns:
            Consistency score in ``[0.0, 1.0]``.  A score of 1.0 means
            every day in the window has the same dominant direction.
        """
        if not activities:
            return 0.0

        now = datetime.now(tz=UTC)
        cutoff = now - timedelta(days=days)

        # Group activities by calendar date
        daily_directions: dict[str, list[FlowDirection]] = defaultdict(list)
        for activity in activities:
            if activity.timestamp >= cutoff:
                day_key = activity.timestamp.strftime("%Y-%m-%d")
                if activity.direction != FlowDirection.NEUTRAL:
                    daily_directions[day_key].append(activity.direction)

        if not daily_directions:
            return 0.0

        # Determine dominant direction per day
        daily_dominants: list[FlowDirection] = []
        for _day, directions in daily_directions.items():
            bullish_count = sum(1 for d in directions if d == FlowDirection.BULLISH)
            bearish_count = sum(1 for d in directions if d == FlowDirection.BEARISH)

            if bullish_count > bearish_count:
                daily_dominants.append(FlowDirection.BULLISH)
            elif bearish_count > bullish_count:
                daily_dominants.append(FlowDirection.BEARISH)
            # Ties are excluded as they are ambiguous.

        if not daily_dominants:
            return 0.0

        # Consistency = fraction of days that agree with the overall dominant
        overall_bullish = sum(1 for d in daily_dominants if d == FlowDirection.BULLISH)
        overall_bearish = sum(1 for d in daily_dominants if d == FlowDirection.BEARISH)

        if overall_bullish >= overall_bearish:
            agreeing_days = overall_bullish
        else:
            agreeing_days = overall_bearish

        consistency = agreeing_days / len(daily_dominants)
        return round(consistency, 4)

    # ------------------------------------------------------------------
    # Repeated activity detection
    # ------------------------------------------------------------------

    def _detect_repeated_activity(
        self,
        transactions: list[FlowTransaction],
    ) -> list[UnusualActivity]:
        """Detect repeated directional activity across multiple days.

        If the same directional bias (bullish or bearish) appears on
        :data:`REPEATED_ACTIVITY_DAYS` or more distinct calendar days
        within the lookback window, a :class:`UnusualActivity` with type
        ``REPEATED`` is emitted.

        Args:
            transactions: Flow transactions to analyse.

        Returns:
            List of :class:`UnusualActivity` for any repeated patterns found.
        """
        if not transactions:
            return []

        ticker = transactions[0].ticker
        now = datetime.now(tz=UTC)
        cutoff = now - timedelta(days=CONSISTENCY_LOOKBACK_DAYS)

        # Track bullish and bearish days
        bullish_days: set[str] = set()
        bearish_days: set[str] = set()

        for txn in transactions:
            if txn.timestamp < cutoff:
                continue
            day_key = txn.timestamp.strftime("%Y-%m-%d")
            direction = self._classify_direction(txn)
            if direction == FlowDirection.BULLISH:
                bullish_days.add(day_key)
            elif direction == FlowDirection.BEARISH:
                bearish_days.add(day_key)

        results: list[UnusualActivity] = []

        if len(bullish_days) >= REPEATED_ACTIVITY_DAYS:
            total_bullish_premium = sum(
                txn.premium
                for txn in transactions
                if (
                    txn.timestamp >= cutoff
                    and self._classify_direction(txn) == FlowDirection.BULLISH
                )
            )
            results.append(
                UnusualActivity(
                    ticker=ticker,
                    activity_type=ActivityType.REPEATED,
                    direction=FlowDirection.BULLISH,
                    premium=total_bullish_premium,
                    details=(
                        f"Bullish flow on {len(bullish_days)} days "
                        f"in past {CONSISTENCY_LOOKBACK_DAYS} days: "
                        f"{', '.join(sorted(bullish_days))}"
                    ),
                    confidence=CONFIDENCE_REPEATED,
                )
            )

        if len(bearish_days) >= REPEATED_ACTIVITY_DAYS:
            total_bearish_premium = sum(
                txn.premium
                for txn in transactions
                if (
                    txn.timestamp >= cutoff
                    and self._classify_direction(txn) == FlowDirection.BEARISH
                )
            )
            results.append(
                UnusualActivity(
                    ticker=ticker,
                    activity_type=ActivityType.REPEATED,
                    direction=FlowDirection.BEARISH,
                    premium=total_bearish_premium,
                    details=(
                        f"Bearish flow on {len(bearish_days)} days "
                        f"in past {CONSISTENCY_LOOKBACK_DAYS} days: "
                        f"{', '.join(sorted(bearish_days))}"
                    ),
                    confidence=CONFIDENCE_REPEATED,
                )
            )

        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _activity_weight(activity: UnusualActivity) -> float:
        """Return the scoring weight for an unusual activity by type.

        Sweeps are weighted most heavily because they indicate urgency
        (the trader paid up across multiple exchanges), followed by block
        trades (large single fills) and then volume/OI anomalies.

        Args:
            activity: The unusual activity to weight.

        Returns:
            Multiplicative weight in ``[1.0, 1.5]``.
        """
        weight_map: dict[ActivityType, float] = {
            ActivityType.SWEEP: WEIGHT_SWEEP,
            ActivityType.BLOCK: WEIGHT_BLOCK,
            ActivityType.HIGH_VOL_OI: WEIGHT_HIGH_VOL_OI,
            ActivityType.REPEATED: WEIGHT_SWEEP,  # repeated is high conviction
        }
        return weight_map.get(activity.activity_type, WEIGHT_REGULAR)

    def _parse_transaction(
        self,
        raw: dict[str, Any],
        ticker: str,
    ) -> FlowTransaction:
        """Parse a raw API response dict into a :class:`FlowTransaction`.

        Maps field names from the Unusual Whales JSON schema to the
        internal Pydantic model.  Handles common field-name variations
        and provides sensible defaults for missing optional data.

        Args:
            raw: A single transaction dict from the API response.
            ticker: The underlying ticker symbol (used as fallback).

        Returns:
            A validated :class:`FlowTransaction`.

        Raises:
            KeyError: If required fields are missing.
            ValueError: If values cannot be coerced to expected types.
        """
        # Timestamp: accept ISO format or epoch seconds
        raw_ts = raw.get("date") or raw.get("timestamp") or raw.get("executed_at")
        if isinstance(raw_ts, (int, float)):
            timestamp = datetime.fromtimestamp(raw_ts, tz=UTC)
        elif isinstance(raw_ts, str):
            # Handle common ISO formats; strip trailing Z for fromisoformat
            ts_str = raw_ts.replace("Z", "+00:00")
            timestamp = datetime.fromisoformat(ts_str)
            if timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=UTC)
        else:
            timestamp = datetime.now(tz=UTC)

        # Option type
        raw_type = str(raw.get("option_type", raw.get("put_call", ""))).upper()
        if raw_type in ("C", "CALL", "CALLS"):
            option_type = OptionType.CALL
        elif raw_type in ("P", "PUT", "PUTS"):
            option_type = OptionType.PUT
        else:
            option_type = OptionType.CALL  # defensive default

        # Strike
        strike = float(raw.get("strike", raw.get("strike_price", 0.0)))

        # Expiry
        expiry = str(
            raw.get("expiry", raw.get("expiration_date", raw.get("expires", "")))
        )

        # Premium
        premium = float(raw.get("premium", raw.get("total_premium", 0.0)))

        # Volume and OI
        volume = int(raw.get("volume", 0))
        open_interest = int(raw.get("open_interest", raw.get("oi", 0)))

        # Vol/OI ratio: compute if not provided
        raw_vol_oi = raw.get("vol_oi_ratio", raw.get("volume_oi_ratio"))
        if raw_vol_oi is not None:
            vol_oi_ratio = float(raw_vol_oi)
        elif open_interest > 0:
            vol_oi_ratio = volume / open_interest
        else:
            vol_oi_ratio = 0.0

        # Trade type
        raw_trade_type = str(
            raw.get("trade_type", raw.get("option_activity_type", ""))
        ).lower()
        if "sweep" in raw_trade_type:
            trade_type = TradeType.SWEEP
        elif "block" in raw_trade_type:
            trade_type = TradeType.BLOCK
        else:
            trade_type = TradeType.REGULAR

        # Side (ask/bid/mid)
        raw_side = str(raw.get("side", raw.get("sentiment", ""))).lower()
        if "ask" in raw_side or "above" in raw_side:
            side = TradeSide.ASK
        elif "bid" in raw_side or "below" in raw_side:
            side = TradeSide.BID
        else:
            side = TradeSide.MID

        return FlowTransaction(
            ticker=raw.get("ticker", raw.get("symbol", ticker)),
            timestamp=timestamp,
            option_type=option_type,
            strike=strike,
            expiry=expiry,
            premium=premium,
            volume=volume,
            open_interest=open_interest,
            vol_oi_ratio=vol_oi_ratio,
            trade_type=trade_type,
            side=side,
        )
