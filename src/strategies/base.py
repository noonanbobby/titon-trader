"""Abstract base strategy class for Project Titan.

Defines the contract that every options strategy must fulfil.  All ten
strategy implementations (bull call spread, iron condor, etc.) inherit from
:class:`BaseStrategy` and implement its five abstract methods.  Shared
mechanical-exit logic, eligibility checks, option chain filtering, and
delta-based strike selection are provided as concrete methods so that
strategy subclasses focus exclusively on strategy-specific behaviour.

Pydantic models used across the entire strategy subsystem are co-located
here: :class:`StrategyConfig`, :class:`LegSpec`, :class:`TradeSignal`,
:class:`ExitSignal`, :class:`OptionData`, :class:`GreeksSnapshot`, and
:class:`TradeRecord`.

Usage::

    from src.strategies.base import (
        BaseStrategy, StrategyConfig, TradeSignal, ExitSignal,
        LegSpec, OptionData, GreeksSnapshot, TradeRecord,
    )

    class BullCallSpread(BaseStrategy):
        @property
        def name(self) -> str:
            return "bull_call_spread"

        async def check_entry(self, ticker, spot_price, iv_rank, regime,
                              greeks, options_chain) -> TradeSignal | None:
            ...
"""

from __future__ import annotations

import abc
from datetime import date, datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field, field_validator

from src.utils.logging import get_logger

if TYPE_CHECKING:
    from uuid import UUID

    import structlog

# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class Direction(StrEnum):
    """Trade direction: LONG (debit) or SHORT (credit)."""

    LONG = "LONG"
    SHORT = "SHORT"


class OptionRight(StrEnum):
    """Option type: call or put."""

    CALL = "C"
    PUT = "P"


class LegAction(StrEnum):
    """Whether a leg is bought or sold."""

    BUY = "BUY"
    SELL = "SELL"


class ExitType(StrEnum):
    """Reason for closing a position."""

    PROFIT_TARGET = "PROFIT_TARGET"
    STOP_LOSS = "STOP_LOSS"
    TIME_DECAY = "TIME_DECAY"
    MECHANICAL = "MECHANICAL"
    MANUAL = "MANUAL"


class ExitReason(StrEnum):
    """Extended exit reason covering all possible close triggers.

    :data:`PROFIT_TARGET`, :data:`STOP_LOSS`, and :data:`DTE_LIMIT` map
    to the three mechanical rules.  The remaining values cover
    strategy-specific, manual, circuit-breaker, and event-risk exits.
    """

    PROFIT_TARGET = "PROFIT_TARGET"
    STOP_LOSS = "STOP_LOSS"
    DTE_LIMIT = "DTE_LIMIT"
    STRATEGY_SPECIFIC = "STRATEGY_SPECIFIC"
    MANUAL = "MANUAL"
    CIRCUIT_BREAKER = "CIRCUIT_BREAKER"
    EVENT_RISK = "EVENT_RISK"


# ---------------------------------------------------------------------------
# Global defaults (from config/strategies.yaml -> defaults section)
# ---------------------------------------------------------------------------

DEFAULT_MIN_OPEN_INTEREST: int = 500
DEFAULT_MAX_BID_ASK_SPREAD_PCT: float = 0.05
DEFAULT_CLOSE_BEFORE_EXPIRY_DTE: int = 5
DEFAULT_MIN_UNDERLYING_PRICE: float = 20.0
DEFAULT_MAX_UNDERLYING_PRICE: float = 5000.0


# ---------------------------------------------------------------------------
# Pydantic data models
# ---------------------------------------------------------------------------


class StrategyConfig(BaseModel):
    """Configuration parameters for a single strategy, loaded from
    ``config/strategies.yaml``.

    Fields mirror the keys under each strategy entry in the YAML file.
    Optional fields cover strategy-specific parameters such as wing widths,
    ratios, margin requirements, and catalyst flags.

    Attributes:
        enabled: Master switch for the strategy.
        description: Human-readable summary of the strategy.
        regimes: Market regimes where this strategy is eligible.
        min_iv_rank: Minimum IV Rank (0--100) required for entry.
        max_iv_rank: Maximum IV Rank (0--100) allowed for entry.
        target_dte: Ideal days to expiration.  An ``int`` for simple
            strategies or a ``dict`` mapping leg roles to target DTEs
            for multi-expiration strategies (calendars, diagonals, PMCCs).
        dte_range: Acceptable DTE window.  A ``[min, max]`` list for
            simple strategies or a ``dict`` mapping leg roles to
            ``[min, max]`` lists.
        delta_range: Acceptable delta ranges per leg role.  Always a
            ``dict`` mapping role names (e.g. ``"long_leg"``,
            ``"short_leg"``) to ``[min, max]`` lists.
        wing_width: Strike distance in dollars for symmetric spreads.
        wing_widths: Strike distances for asymmetric spreads (e.g.
            broken wing butterfly).
        profit_target_pct: Close when this fraction of max profit is
            reached.
        stop_loss_pct: Close when unrealised loss reaches this multiple
            of max loss (or credit received).
        max_positions: Maximum simultaneous positions of this strategy.
        ratio: Leg ratio as a string like ``"1:2"`` (optional).
        margin_required: Whether the strategy requires margin.
        catalyst_required: Whether a catalyst event must be pending.
        roll_short_at_dte: DTE at which to roll the short leg (PMCCs).
    """

    enabled: bool = True
    description: str = ""
    regimes: list[str] = Field(default_factory=list)
    min_iv_rank: float = 0.0
    max_iv_rank: float = 100.0
    target_dte: int | dict[str, int] = 45
    dte_range: list[int] | dict[str, list[int]] = Field(
        default_factory=lambda: [30, 60],
    )
    delta_range: dict[str, list[float] | float] = Field(default_factory=dict)
    wing_width: float | None = None
    wing_widths: dict[str, float] | None = None
    profit_target_pct: float = 0.50
    stop_loss_pct: float = 1.00
    max_positions: int = 3
    ratio: str | None = None
    margin_required: bool = False
    catalyst_required: bool = False
    roll_short_at_dte: int | None = None

    @field_validator("regimes", mode="before")
    @classmethod
    def _coerce_regimes(cls, v: Any) -> list[str]:
        """Accept a single string and wrap it in a list."""
        if isinstance(v, str):
            return [v]
        return list(v)

    @field_validator("profit_target_pct", "stop_loss_pct")
    @classmethod
    def _validate_positive(cls, v: float) -> float:
        if v <= 0.0:
            raise ValueError(f"Must be positive, got {v}")
        return v

    @field_validator("max_positions")
    @classmethod
    def _validate_max_positions(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"max_positions must be >= 1, got {v}")
        return v

    # -- Convenience helpers -----------------------------------------------

    def get_primary_dte_range(self) -> tuple[int, int]:
        """Return the primary DTE range as ``(min, max)``.

        For single-DTE strategies the range is returned directly.  For
        multi-expiration strategies the ``front_month`` range is returned,
        falling back to the first available key.
        """
        if isinstance(self.dte_range, list):
            return (self.dte_range[0], self.dte_range[1])
        if "front_month" in self.dte_range:
            r = self.dte_range["front_month"]
            return (r[0], r[1])
        first_key = next(iter(self.dte_range))
        r = self.dte_range[first_key]
        return (r[0], r[1])

    def get_primary_target_dte(self) -> int:
        """Return the primary target DTE as an integer.

        For single-DTE strategies returns the value directly.  For
        multi-expiration strategies returns ``front_month``, falling back
        to the first available key.
        """
        if isinstance(self.target_dte, int):
            return self.target_dte
        if "front_month" in self.target_dte:
            return self.target_dte["front_month"]
        return next(iter(self.target_dte.values()))


class OptionData(BaseModel):
    """Snapshot of a single option contract from the options chain.

    Attributes:
        strike: Strike price.
        right: ``C`` for call, ``P`` for put.
        expiry: Expiration date.
        delta: Option delta.
        gamma: Option gamma.
        theta: Option theta (daily decay).
        vega: Option vega.
        implied_vol: Implied volatility.
        bid: Best bid price.
        ask: Best ask price.
        mid_price: Midpoint of bid and ask.
        open_interest: Open interest.
        volume: Daily volume (default 0).
    """

    strike: float
    right: str
    expiry: date
    delta: float
    gamma: float
    theta: float
    vega: float
    implied_vol: float
    bid: float
    ask: float
    mid_price: float
    open_interest: int
    volume: int = 0

    model_config = {"frozen": True}


class GreeksSnapshot(BaseModel):
    """Aggregate Greeks for the underlying or a position.

    Attributes:
        delta: Net delta.
        gamma: Net gamma.
        theta: Net theta.
        vega: Net vega.
        iv: Current implied volatility.
    """

    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0
    vega: float = 0.0
    iv: float = 0.0


class LegSpec(BaseModel):
    """Specification for a single leg in a multi-leg options order.

    This is the *strategy output* -- it describes what to trade without
    being tied to IBKR contract objects.

    Attributes:
        action: ``BUY`` or ``SELL``.
        right: ``C`` or ``P``.
        strike: Strike price.
        expiry: Expiration date in ``YYYYMMDD`` string format.
        quantity: Number of contracts (always positive).
        delta: Actual delta of the option at leg-construction time.
    """

    action: str  # BUY or SELL
    right: str  # C or P
    strike: float
    expiry: str = Field(
        ...,
        min_length=8,
        max_length=8,
        description="Expiry in YYYYMMDD format",
    )
    quantity: int = 1
    delta: float | None = None

    model_config = {"frozen": True}

    @field_validator("expiry")
    @classmethod
    def _validate_expiry(cls, v: str) -> str:
        """Ensure expiry is a plausible ``YYYYMMDD`` date string."""
        if not v.isdigit():
            raise ValueError(f"expiry must be numeric YYYYMMDD, got '{v}'")
        year, month, day = int(v[:4]), int(v[4:6]), int(v[6:8])
        if not (2000 <= year <= 2100):
            raise ValueError(f"expiry year out of range: {year}")
        if not (1 <= month <= 12):
            raise ValueError(f"expiry month out of range: {month}")
        if not (1 <= day <= 31):
            raise ValueError(f"expiry day out of range: {day}")
        return v


class TradeSignal(BaseModel):
    """Signal emitted by a strategy when entry criteria are met.

    Contains everything needed for the risk manager to evaluate the
    proposal and for the execution engine to place the order.

    Attributes:
        ticker: Underlying symbol.
        strategy_name: Canonical strategy name that generated the signal.
        direction: ``LONG`` (debit) or ``SHORT`` (credit).
        confidence: ML / ensemble confidence score (0.0 -- 1.0).
        legs: Ordered list of leg specifications.
        max_profit: Maximum profit in dollars per spread unit.
        max_loss: Maximum loss in dollars per spread unit (positive number).
        reward_risk_ratio: ``max_profit / max_loss``.
        entry_reasoning: Human-readable rationale for the trade.
    """

    ticker: str = Field(..., min_length=1, max_length=10)
    strategy_name: str
    direction: str  # LONG or SHORT
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    legs: list[LegSpec] = Field(..., min_length=1)
    max_profit: float
    max_loss: float = Field(..., gt=0, description="Positive number")
    reward_risk_ratio: float = 0.0
    entry_reasoning: str = ""


class ExitSignal(BaseModel):
    """Signal emitted when a position should be closed.

    Attributes:
        trade_id: UUID of the trade to close (from the ``trades`` table).
        exit_type: Category of the exit trigger.
        current_pnl: Current unrealised P&L in dollars.
        current_pnl_pct: P&L as a fraction of max loss.
        reasoning: Human-readable explanation for the exit.
    """

    trade_id: str
    exit_type: str  # one of ExitType values
    current_pnl: float
    current_pnl_pct: float
    reasoning: str = ""


class TradeRecord(BaseModel):
    """Lightweight representation of an open trade for exit checking.

    Attributes:
        id: Unique trade identifier.
        ticker: Underlying symbol.
        strategy: Strategy that opened the trade.
        direction: ``LONG`` or ``SHORT``.
        entry_price: Net premium at entry.
        max_profit: Maximum profit in dollars.
        max_loss: Maximum loss in dollars.
        legs: Legs of the trade.
        entry_time: When the trade was opened.
    """

    id: UUID
    ticker: str
    strategy: str
    direction: str
    entry_price: float
    max_profit: float
    max_loss: float
    legs: list[LegSpec] = Field(default_factory=list)
    entry_time: datetime | None = None


# ---------------------------------------------------------------------------
# Abstract base strategy
# ---------------------------------------------------------------------------


class BaseStrategy(abc.ABC):
    """Abstract base class that all options trading strategies must inherit.

    Subclasses implement five abstract methods that define strategy-specific
    entry/exit logic, leg construction, and profit/loss calculation.  The
    base class supplies:

    - :meth:`is_eligible` -- regime, IV rank, and underlying-price checks.
    - :meth:`check_mechanical_exit` -- profit target, stop loss, and DTE
      limit rules that apply universally.
    - :meth:`calculate_reward_risk_ratio` -- simple max-profit / max-loss.
    - :meth:`filter_options_by_dte` -- DTE-based chain filtering.
    - :meth:`find_strike_by_delta` -- delta-proximity strike selection.
    - :meth:`find_option_by_delta` -- range-based delta search on
      :class:`OptionData` objects.
    - :meth:`validate_bid_ask_spread` -- liquidity quality check.

    Args:
        name: Unique strategy identifier matching the key in
            ``strategies.yaml`` (e.g. ``"bull_call_spread"``).
        config: Strategy-specific configuration loaded from YAML.
    """

    def __init__(self, name: str, config: StrategyConfig) -> None:
        self._name: str = name
        self._config: StrategyConfig = config
        self._log: structlog.stdlib.BoundLogger = get_logger(f"strategy.{name}")

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        """Return the canonical strategy identifier."""
        return self._name

    @property
    def config(self) -> StrategyConfig:
        """Return the strategy configuration."""
        return self._config

    # ------------------------------------------------------------------
    # Abstract methods -- every strategy MUST implement these
    # ------------------------------------------------------------------

    @abc.abstractmethod
    async def check_entry(
        self,
        ticker: str,
        spot_price: float,
        iv_rank: float,
        regime: str,
        greeks: dict[str, float],
        options_chain: list[dict[str, Any]],
    ) -> TradeSignal | None:
        """Evaluate whether to open a new position for *ticker*.

        Implementations should:

        1. Call :meth:`is_eligible` to verify regime and IV rank.
        2. Filter the options chain for suitable DTE via
           :meth:`filter_options_by_dte`.
        3. Select strikes via :meth:`find_strike_by_delta` or
           :meth:`find_option_by_delta`.
        4. Call :meth:`construct_legs` to build the leg list.
        5. Compute max profit/loss and reward-risk ratio.
        6. Return a :class:`TradeSignal` if all entry criteria are met,
           or ``None`` if the setup is not attractive.

        Args:
            ticker: Underlying symbol (e.g. ``"AAPL"``).
            spot_price: Current price of the underlying.
            iv_rank: Current IV Rank (0--100) for the ticker.
            regime: Current market regime identifier (e.g.
                ``"low_vol_trend"``).
            greeks: Aggregate Greeks dict with keys ``"delta"``,
                ``"gamma"``, ``"theta"``, ``"vega"``.
            options_chain: List of option dicts from the market data
                provider.  Each dict contains keys such as ``"strike"``,
                ``"expiry"``, ``"right"``, ``"bid"``, ``"ask"``,
                ``"delta"``, ``"gamma"``, ``"theta"``, ``"vega"``,
                ``"open_interest"``, ``"volume"``.

        Returns:
            A :class:`TradeSignal` if entry criteria are satisfied, or
            ``None``.
        """
        ...

    @abc.abstractmethod
    async def check_exit(
        self,
        trade: dict[str, Any],
        spot_price: float,
        current_pnl: float,
        current_pnl_pct: float,
        dte_remaining: int,
        greeks: dict[str, float],
    ) -> ExitSignal | None:
        """Evaluate whether to close an existing position.

        Implementations should first delegate to
        :meth:`check_mechanical_exit` for the universal profit-target,
        stop-loss, and DTE-limit checks.  Strategy-specific exit logic
        (e.g. rolling the short leg of a PMCC) may then be layered on
        top.

        Args:
            trade: Dict representing the open trade from the database.
                Expected keys: ``"id"``, ``"ticker"``, ``"strategy"``,
                ``"entry_price"``, ``"max_profit"``, ``"max_loss"``.
            spot_price: Current price of the underlying.
            current_pnl: Unrealised P&L in dollars.
            current_pnl_pct: P&L as a fraction of max loss.  Positive
                means profitable.
            dte_remaining: Days until the nearest leg expires.
            greeks: Current aggregate Greeks for the position.

        Returns:
            An :class:`ExitSignal` if the position should be closed, or
            ``None``.
        """
        ...

    @abc.abstractmethod
    def construct_legs(
        self,
        spot_price: float,
        options_chain: list[dict[str, Any]],
        **kwargs: Any,
    ) -> list[LegSpec]:
        """Build the leg specifications for this strategy.

        Searches the *options_chain* for strikes matching the strategy's
        delta and width targets defined in configuration.

        Args:
            spot_price: Current underlying price.
            options_chain: Available options to choose from.  Each dict
                contains at minimum ``"strike"``, ``"expiry"``,
                ``"right"``, ``"bid"``, ``"ask"``, ``"delta"``,
                ``"open_interest"``.
            **kwargs: Strategy-specific parameters.

        Returns:
            Ordered list of :class:`LegSpec` describing each leg.

        Raises:
            ValueError: If suitable strikes cannot be found.
        """
        ...

    @abc.abstractmethod
    def calculate_max_profit(
        self,
        legs: list[LegSpec],
        net_premium: float,
    ) -> float:
        """Calculate maximum theoretical profit for the trade in dollars.

        Args:
            legs: The constructed legs.
            net_premium: Net premium paid (positive = debit) or received
                (negative = credit) per spread unit.

        Returns:
            Maximum profit in dollars (always positive).
        """
        ...

    @abc.abstractmethod
    def calculate_max_loss(
        self,
        legs: list[LegSpec],
        net_premium: float,
    ) -> float:
        """Calculate maximum theoretical loss for the trade in dollars.

        Args:
            legs: The constructed legs.
            net_premium: Net premium paid (positive = debit) or received
                (negative = credit) per spread unit.

        Returns:
            Maximum loss in dollars (always positive).
        """
        ...

    # ------------------------------------------------------------------
    # Concrete methods -- shared across all strategies
    # ------------------------------------------------------------------

    def is_eligible(self, regime: str, iv_rank: float) -> bool:
        """Check whether the strategy is eligible for the current conditions.

        A strategy is eligible when **all** of the following hold:

        1. It is enabled in the configuration (``enabled: true``).
        2. The current market regime is among the strategy's allowed
           regimes.
        3. IV Rank falls within ``[min_iv_rank, max_iv_rank]``.

        Args:
            regime: Current market regime identifier.
            iv_rank: Current IV Rank (0--100).

        Returns:
            ``True`` if the strategy may be considered for entry.
        """
        if not self._config.enabled:
            self._log.debug("not_eligible", reason="strategy_disabled")
            return False

        if regime not in self._config.regimes:
            self._log.debug(
                "not_eligible",
                reason="regime_mismatch",
                current_regime=regime,
                allowed_regimes=self._config.regimes,
            )
            return False

        if iv_rank < self._config.min_iv_rank:
            self._log.debug(
                "not_eligible",
                reason="iv_rank_too_low",
                iv_rank=iv_rank,
                min_iv_rank=self._config.min_iv_rank,
            )
            return False

        if iv_rank > self._config.max_iv_rank:
            self._log.debug(
                "not_eligible",
                reason="iv_rank_too_high",
                iv_rank=iv_rank,
                max_iv_rank=self._config.max_iv_rank,
            )
            return False

        self._log.debug(
            "eligible",
            regime=regime,
            iv_rank=iv_rank,
        )
        return True

    def check_mechanical_exit(
        self,
        current_pnl_pct: float,
        dte_remaining: int,
    ) -> ExitSignal | None:
        """Evaluate universal mechanical exit rules.

        Three rules are checked in priority order.  These are
        non-discretionary -- when triggered, the position **must** be
        closed regardless of any other signal.

        1. **Stop loss**: ``current_pnl_pct <= -config.stop_loss_pct``
        2. **Profit target**: ``current_pnl_pct >= config.profit_target_pct``
        3. **Time decay**: ``dte_remaining <= close_before_expiry_dte``

        The returned :class:`ExitSignal` has ``trade_id`` set to the empty
        string and ``current_pnl`` set to ``0.0``.  The caller is
        responsible for populating these with actual values.

        Args:
            current_pnl_pct: P&L expressed as a fraction of max profit
                (positive) or max loss (negative).  For example, ``0.50``
                means 50 % of max profit has been captured; ``-1.0`` means
                the full max loss has been realised.
            dte_remaining: Calendar days until the nearest leg expires.

        Returns:
            An :class:`ExitSignal` if a mechanical exit is warranted, or
            ``None``.
        """
        close_dte = DEFAULT_CLOSE_BEFORE_EXPIRY_DTE

        # 1. Stop loss -- highest priority
        if current_pnl_pct <= -self._config.stop_loss_pct:
            reasoning = (
                f"Stop loss triggered: P&L {current_pnl_pct:.2%} exceeds "
                f"threshold of -{self._config.stop_loss_pct:.2%}"
            )
            self._log.info(
                "mechanical_exit",
                exit_type="STOP_LOSS",
                current_pnl_pct=round(current_pnl_pct, 4),
                threshold=-self._config.stop_loss_pct,
            )
            return ExitSignal(
                trade_id="",
                exit_type=ExitType.STOP_LOSS,
                current_pnl=0.0,
                current_pnl_pct=current_pnl_pct,
                reasoning=reasoning,
            )

        # 2. Profit target
        if current_pnl_pct >= self._config.profit_target_pct:
            reasoning = (
                f"Profit target reached: P&L {current_pnl_pct:.2%} meets "
                f"target of {self._config.profit_target_pct:.2%}"
            )
            self._log.info(
                "mechanical_exit",
                exit_type="PROFIT_TARGET",
                current_pnl_pct=round(current_pnl_pct, 4),
                threshold=self._config.profit_target_pct,
            )
            return ExitSignal(
                trade_id="",
                exit_type=ExitType.PROFIT_TARGET,
                current_pnl=0.0,
                current_pnl_pct=current_pnl_pct,
                reasoning=reasoning,
            )

        # 3. Time decay -- close before expiry
        if dte_remaining <= close_dte:
            reasoning = (
                f"Time decay exit: {dte_remaining} DTE remaining, "
                f"threshold is {close_dte} DTE"
            )
            self._log.info(
                "mechanical_exit",
                exit_type="TIME_DECAY",
                dte_remaining=dte_remaining,
                close_before_expiry_dte=close_dte,
            )
            return ExitSignal(
                trade_id="",
                exit_type=ExitType.TIME_DECAY,
                current_pnl=0.0,
                current_pnl_pct=current_pnl_pct,
                reasoning=reasoning,
            )

        return None

    def calculate_reward_risk_ratio(
        self,
        max_profit: float,
        max_loss: float,
    ) -> float:
        """Calculate the reward-to-risk ratio.

        Args:
            max_profit: Maximum potential profit in dollars.
            max_loss: Maximum potential loss in dollars (positive number).

        Returns:
            ``max_profit / max_loss``, rounded to four decimal places.
            Returns ``0.0`` if *max_loss* is zero or negative.
        """
        if max_loss <= 0.0:
            self._log.warning(
                "invalid_max_loss_for_rr",
                max_profit=max_profit,
                max_loss=max_loss,
            )
            return 0.0
        return round(max_profit / max_loss, 4)

    def filter_options_by_dte(
        self,
        options: list[dict[str, Any]],
        min_dte: int,
        max_dte: int,
    ) -> list[dict[str, Any]]:
        """Filter an options chain to contracts within a DTE range.

        Each option dict must contain an ``"expiry"`` key with a value in
        ``YYYYMMDD`` string format.  A computed ``"dte"`` key is added to
        every returned dict for downstream convenience.

        Args:
            options: List of option dicts from the chain.
            min_dte: Minimum acceptable days to expiration (inclusive).
            max_dte: Maximum acceptable days to expiration (inclusive).

        Returns:
            Filtered list containing only options whose DTE falls in
            ``[min_dte, max_dte]``.
        """
        today = date.today()
        filtered: list[dict[str, Any]] = []

        for opt in options:
            expiry_raw = opt.get("expiry", "")

            # Handle both date objects and YYYYMMDD strings
            if isinstance(expiry_raw, date):
                expiry_date = expiry_raw
            elif isinstance(expiry_raw, str) and len(expiry_raw) == 8:
                try:
                    expiry_date = datetime.strptime(expiry_raw, "%Y%m%d").date()
                except ValueError:
                    continue
            else:
                continue

            dte = (expiry_date - today).days
            if min_dte <= dte <= max_dte:
                filtered.append({**opt, "dte": dte})

        self._log.debug(
            "options_filtered_by_dte",
            total=len(options),
            matched=len(filtered),
            min_dte=min_dte,
            max_dte=max_dte,
        )
        return filtered

    def find_strike_by_delta(
        self,
        options: list[dict[str, Any]],
        target_delta: float,
        right: str,
        tolerance: float = 0.05,
    ) -> dict[str, Any] | None:
        """Find the option closest to a target delta value.

        Searches *options* for the contract whose absolute delta is nearest
        to ``abs(target_delta)`` within *tolerance*.  Only options matching
        the specified *right* (``"C"`` or ``"P"``) are considered.

        Args:
            options: List of option dicts, each containing at minimum
                ``"strike"``, ``"right"``, and ``"delta"``.
            target_delta: Desired delta value.  The comparison is always
                performed on absolute values, so the sign does not matter
                for filtering purposes.
            right: ``"C"`` for calls or ``"P"`` for puts.
            tolerance: Maximum acceptable deviation from *target_delta*
                (default ``0.05``).

        Returns:
            The option dict closest to *target_delta*, or ``None`` if no
            match is found within *tolerance*.
        """
        import math

        abs_target = abs(target_delta)
        best_match: dict[str, Any] | None = None
        best_distance: float = float("inf")

        for opt in options:
            if opt.get("right") != right:
                continue

            opt_delta = opt.get("delta")
            if opt_delta is None:
                continue
            if isinstance(opt_delta, float) and math.isnan(opt_delta):
                continue

            abs_delta = abs(opt_delta)
            distance = abs(abs_delta - abs_target)

            if distance <= tolerance and distance < best_distance:
                best_distance = distance
                best_match = opt

        if best_match is not None:
            self._log.debug(
                "strike_found_by_delta",
                target_delta=target_delta,
                right=right,
                matched_strike=best_match.get("strike"),
                matched_delta=best_match.get("delta"),
                distance=round(best_distance, 4),
            )
        else:
            self._log.debug(
                "no_strike_found_by_delta",
                target_delta=target_delta,
                right=right,
                tolerance=tolerance,
                options_searched=len(options),
            )

        return best_match

    # ------------------------------------------------------------------
    # OptionData-based helpers (for typed chain usage)
    # ------------------------------------------------------------------

    @staticmethod
    def find_option_by_delta(
        options: list[OptionData],
        right: str,
        target_delta_min: float,
        target_delta_max: float,
    ) -> OptionData | None:
        """Find the option whose delta is closest to the midpoint of a
        target range.

        Searches *options* for contracts matching *right* whose delta
        falls within ``[target_delta_min, target_delta_max]`` and that
        meet the minimum open-interest requirement.  Returns the best
        match (closest to midpoint) or ``None``.

        For puts, deltas are negative -- the caller should pass negative
        bounds (e.g. ``-0.35, -0.20``).  For calls, pass positive bounds.

        Args:
            options: Typed option snapshots to search.
            right: ``"C"`` or ``"P"``.
            target_delta_min: Lower bound of acceptable delta.
            target_delta_max: Upper bound of acceptable delta.

        Returns:
            Best matching :class:`OptionData`, or ``None``.
        """
        target_mid = (target_delta_min + target_delta_max) / 2.0

        candidates: list[tuple[float, OptionData]] = []
        for opt in options:
            if opt.right != right:
                continue
            if opt.open_interest < DEFAULT_MIN_OPEN_INTEREST:
                continue
            if target_delta_min <= opt.delta <= target_delta_max:
                distance = abs(opt.delta - target_mid)
                candidates.append((distance, opt))

        if not candidates:
            return None

        candidates.sort(key=lambda x: x[0])
        return candidates[0][1]

    @staticmethod
    def validate_bid_ask_spread(option: OptionData) -> bool:
        """Check that the bid-ask spread is within acceptable limits.

        Args:
            option: The option to validate.

        Returns:
            ``True`` if the spread is acceptable, ``False`` otherwise.
        """
        if option.mid_price <= 0:
            return False
        spread = option.ask - option.bid
        spread_pct = spread / option.mid_price
        return spread_pct <= DEFAULT_MAX_BID_ASK_SPREAD_PCT

    # ------------------------------------------------------------------
    # String representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"<{type(self).__name__}("
            f"name={self._name!r}, "
            f"enabled={self._config.enabled}, "
            f"regimes={self._config.regimes}"
            f")>"
        )
