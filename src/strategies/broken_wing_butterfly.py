"""Broken-wing butterfly strategy implementation.

A broken-wing butterfly (BWB) is an asymmetric butterfly spread where the
wings are unequal in width.  The strategy uses puts and is constructed by:

- Buying 1 lower put (closer to ATM — narrow side)
- Selling 2 middle puts (the body)
- Buying 1 upper put (further OTM — wide side)

The narrow wing is closer to the body than the wide wing, creating a
position that can be entered for a net credit or at even money.  The trade
profits when the underlying stays at or above the body strike.

Typical use: neutral to slightly bullish outlook where the trader wants
limited risk with a credit entry and maximum profit at the body strike.

Usage::

    from src.strategies.broken_wing_butterfly import BrokenWingButterfly
    from src.strategies.base import StrategyConfig

    config = StrategyConfig(...)
    strategy = BrokenWingButterfly("broken_wing_butterfly", config)
    signal = await strategy.check_entry("AAPL", 175.0, 45.0, "range_bound",
                                         greeks, options_chain)
"""

from __future__ import annotations

from datetime import date
from typing import Any

from src.strategies.base import (
    DEFAULT_MIN_OPEN_INTEREST,
    BaseStrategy,
    ExitReason,
    ExitSignal,
    GreeksSnapshot,
    LegSpec,
    OptionData,
    StrategyConfig,
    TradeRecord,
    TradeSignal,
)
from src.utils.logging import get_logger

logger = get_logger("strategy.broken_wing_butterfly")

# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------

# Target DTE — Carl Allen methodology: 21 DTE for optimal theta decay
TARGET_DTE: int = 21
MIN_DTE: int = 14
MAX_DTE: int = 28

# Body (2x short puts) delta range — Carl Allen: target 28Δ
BODY_DELTA_MIN: float = -0.32
BODY_DELTA_MAX: float = -0.24

# Upper wing (1x long put, further OTM — wide side) delta range — Carl Allen: 20Δ
UPPER_WING_DELTA_MIN: float = -0.24
UPPER_WING_DELTA_MAX: float = -0.16

# Lower wing (1x long put, closer to ATM — narrow side) delta range — Carl Allen: 32Δ
LOWER_WING_DELTA_MIN: float = -0.36
LOWER_WING_DELTA_MAX: float = -0.28

# Default wing widths in dollars
NARROW_WING_WIDTH: float = 5.0
WIDE_WING_WIDTH: float = 10.0

# Credit target: 10–15% of narrow wing width (Carl Allen methodology)
MIN_CREDIT_PCT: float = 0.10
MAX_CREDIT_PCT: float = 0.15

# Exit thresholds — Carl Allen: exit at 2% of narrow wing width profit
# or at 7 DTE, whichever comes first
PROFIT_TARGET_PCT: float = 0.02  # 2% of narrow wing width
STOP_LOSS_MULTIPLIER: float = 1.50
EXIT_DTE_THRESHOLD: int = 7

# Default to puts for the BWB
DEFAULT_RIGHT: str = "P"


# ---------------------------------------------------------------------------
# BrokenWingButterfly strategy
# ---------------------------------------------------------------------------


class BrokenWingButterfly(BaseStrategy):
    """Broken-wing butterfly: asymmetric put butterfly with unequal wing widths.

    The position is:
    - Buy 1 lower put (narrow side, closer to ATM, higher strike in absolute terms)
    - Sell 2 middle puts (body)
    - Buy 1 upper put (wide side, further OTM, lower strike in absolute terms)

    Note on naming convention: "lower" and "upper" refer to the position
    relative to the body in delta terms, not in strike terms.  The "lower"
    wing (closer to ATM, higher delta magnitude) has a *higher* strike than
    the body, while the "upper" wing (further OTM, lower delta magnitude)
    has a *lower* strike than the body.

    The trade is entered for a net credit or even money and profits when:
    - The underlying stays at or above the body strike.
    - Maximum profit at the body strike equals the net credit plus the
      narrow wing width.
    - Risk is limited to the wide side: (wide_width - narrow_width - credit).

    Entry criteria:
        - Body (2x short): delta -0.35 to -0.25.
        - Lower wing (1x long, narrow side): delta -0.50 to -0.40.
        - Upper wing (1x long, wide side): delta -0.10 to -0.05.
        - Wing widths: narrow = 5, wide = 10 (configurable).
        - Entered for a credit or even money.

    Exit rules:
        - Profit target: 50% of max profit.
        - Stop loss: 1.5x max risk.
        - Close at DTE <= 5.
        - Exit if underlying breaks through the wide wing.
    """

    def __init__(self, name: str, config: StrategyConfig) -> None:
        super().__init__(name, config)

    # ------------------------------------------------------------------
    # Entry
    # ------------------------------------------------------------------

    async def check_entry(
        self,
        ticker: str,
        spot_price: float,
        iv_rank: float,
        regime: str,
        greeks: GreeksSnapshot,
        options_chain: list[OptionData],
    ) -> TradeSignal | None:
        """Evaluate whether a broken-wing butterfly entry is warranted.

        Steps:
        1. Verify eligibility (regime, IV rank, underlying price).
        2. Find a suitable expiration within the DTE range.
        3. Find the body strike (2x short puts) in the target delta range.
        4. Find the lower wing (narrow side, closer to ATM, higher strike).
        5. Find the upper wing (wide side, further OTM, lower strike).
        6. Validate wing widths and structure.
        7. Verify the position can be entered for a credit or even money.
        8. Construct legs and build the trade signal.

        Args:
            ticker: Underlying symbol.
            spot_price: Current price of the underlying.
            iv_rank: IV Rank (0--100).
            regime: Current market regime label.
            greeks: Aggregate Greeks snapshot.
            options_chain: Available options with Greeks and pricing.

        Returns:
            A :class:`TradeSignal` if all entry criteria are met, else ``None``.
        """
        if not self.is_eligible(regime, iv_rank, spot_price):
            return None

        if not options_chain:
            self._log.debug("no_options_chain", ticker=ticker)
            return None

        right = DEFAULT_RIGHT

        # Find suitable expiration
        target_dte, min_dte, max_dte = self._get_dte_params()
        expiry = self._find_expiry_near_dte(options_chain, target_dte, min_dte, max_dte)
        if expiry is None:
            self._log.debug(
                "no_suitable_expiry",
                ticker=ticker,
                target_dte=target_dte,
                min_dte=min_dte,
                max_dte=max_dte,
            )
            return None

        # Filter chain to this expiry and right
        expiry_options = [
            o for o in options_chain if o.expiry == expiry and o.right == right
        ]
        if not expiry_options:
            self._log.debug(
                "no_options_at_expiry",
                ticker=ticker,
                expiry=expiry.isoformat(),
            )
            return None

        # Find body strike (2x short puts)
        body_delta_min, body_delta_max = self._get_body_delta_range()
        body_option = self.find_option_by_delta(
            expiry_options, right, body_delta_min, body_delta_max
        )
        if body_option is None:
            self._log.debug(
                "no_body_option",
                ticker=ticker,
                delta_range=[body_delta_min, body_delta_max],
            )
            return None

        if not self.validate_bid_ask_spread(body_option):
            self._log.debug(
                "body_option_illiquid",
                ticker=ticker,
                strike=body_option.strike,
            )
            return None

        body_strike = body_option.strike
        narrow_width, wide_width = self._get_wing_widths()

        # Lower wing: closer to ATM, higher strike than body (narrow side)
        # For puts, "closer to ATM" means higher strike (higher absolute delta)
        lower_wing_strike = body_strike + narrow_width
        lower_wing_option = self._find_option_by_strike(
            expiry_options, lower_wing_strike
        )
        if lower_wing_option is None:
            self._log.debug(
                "no_lower_wing_option",
                ticker=ticker,
                target_strike=lower_wing_strike,
                body_strike=body_strike,
                narrow_width=narrow_width,
            )
            return None

        if not self.validate_bid_ask_spread(lower_wing_option):
            self._log.debug(
                "lower_wing_illiquid",
                ticker=ticker,
                strike=lower_wing_option.strike,
            )
            return None

        # Upper wing: further OTM, lower strike than body (wide side)
        # For puts, "further OTM" means lower strike (lower absolute delta)
        upper_wing_strike = body_strike - wide_width
        upper_wing_option = self._find_option_by_strike(
            expiry_options, upper_wing_strike
        )
        if upper_wing_option is None:
            self._log.debug(
                "no_upper_wing_option",
                ticker=ticker,
                target_strike=upper_wing_strike,
                body_strike=body_strike,
                wide_width=wide_width,
            )
            return None

        if not self.validate_bid_ask_spread(upper_wing_option):
            self._log.debug(
                "upper_wing_illiquid",
                ticker=ticker,
                strike=upper_wing_option.strike,
            )
            return None

        # Validate strike ordering: upper_wing < body < lower_wing
        if not (
            upper_wing_option.strike < body_option.strike < lower_wing_option.strike
        ):
            self._log.debug(
                "invalid_strike_ordering",
                ticker=ticker,
                upper_wing=upper_wing_option.strike,
                body=body_option.strike,
                lower_wing=lower_wing_option.strike,
            )
            return None

        # Construct legs
        try:
            legs = self.construct_legs(
                spot_price=spot_price,
                options_chain=options_chain,
                right=right,
                expiry=expiry,
                body_option=body_option,
                lower_wing_option=lower_wing_option,
                upper_wing_option=upper_wing_option,
            )
        except ValueError as exc:
            self._log.debug("construct_legs_failed", ticker=ticker, error=str(exc))
            return None

        # Calculate net premium (negative = credit for BWB)
        net_premium = self._calculate_net_premium(legs)

        # BWB should be entered for a credit or even money
        if net_premium > 0:
            self._log.debug(
                "non_credit_bwb",
                ticker=ticker,
                net_premium=net_premium,
            )
            return None

        max_profit = self.calculate_max_profit(legs, net_premium)
        max_loss = self.calculate_max_loss(legs, net_premium)

        dte = (expiry - date.today()).days

        reasoning = (
            f"Broken-wing butterfly on {ticker}: "
            f"buy {right} {lower_wing_option.strike} / "
            f"sell 2x {right} {body_option.strike} / "
            f"buy {right} {upper_wing_option.strike} "
            f"({dte}d). "
            f"Net credit ${abs(net_premium):.2f}. "
            f"Narrow wing ${narrow_width:.0f}, wide wing ${wide_width:.0f}. "
            f"Max profit ${max_profit:.2f}, max loss ${max_loss:.2f}. "
            f"IV rank {iv_rank:.1f}, regime {regime}."
        )

        self._log.info(
            "entry_signal_generated",
            ticker=ticker,
            right=right,
            body_strike=body_strike,
            lower_wing_strike=lower_wing_option.strike,
            upper_wing_strike=upper_wing_option.strike,
            narrow_width=narrow_width,
            wide_width=wide_width,
            net_premium=net_premium,
            max_profit=max_profit,
            max_loss=max_loss,
            expiry=expiry.isoformat(),
            dte=dte,
            iv_rank=iv_rank,
            regime=regime,
        )

        return TradeSignal(
            strategy=self.name,
            ticker=ticker,
            direction="SHORT",
            legs=legs,
            net_premium=net_premium,
            max_profit=max_profit,
            max_loss=max_loss,
            confidence=0.0,
            regime=regime,
            iv_rank=iv_rank,
            reasoning=reasoning,
        )

    # ------------------------------------------------------------------
    # Exit
    # ------------------------------------------------------------------

    async def check_exit(
        self,
        trade: TradeRecord,
        spot_price: float,
        current_pnl: float,
        current_pnl_pct: float,
        dte_remaining: int,
        greeks: GreeksSnapshot,
    ) -> ExitSignal | None:
        """Evaluate whether an open broken-wing butterfly should be closed.

        Checks in order:
        1. Mechanical exits (profit target, stop loss, DTE limit).
        2. Underlying has broken through the wide wing (lower strike).

        The stop loss for a BWB is 1.5x the max risk, which is handled by
        the mechanical exit using the configured ``stop_loss_pct`` of 1.50.

        Args:
            trade: Record of the open BWB.
            spot_price: Current underlying price.
            current_pnl: Unrealised P&L in dollars.
            current_pnl_pct: Unrealised P&L as fraction of max profit/loss.
            dte_remaining: Days to expiration.
            greeks: Current Greeks of the position.

        Returns:
            An :class:`ExitSignal` if any exit rule triggers, else ``None``.
        """
        # 1. Mechanical exits (profit target, stop loss, DTE limit)
        mechanical = self.check_mechanical_exit(
            trade, current_pnl, current_pnl_pct, dte_remaining
        )
        if mechanical is not None:
            return mechanical

        # 2. Underlying has broken through the wide wing
        wide_wing_strike = self._get_wide_wing_strike(trade)
        if (
            wide_wing_strike is not None
            and wide_wing_strike > 0
            and spot_price < wide_wing_strike
        ):
            self._log.info(
                "exit_wide_wing_breach",
                trade_id=str(trade.id),
                ticker=trade.ticker,
                spot_price=spot_price,
                wide_wing_strike=wide_wing_strike,
            )
            return ExitSignal(
                trade_id=trade.id,
                reason=ExitReason.STRATEGY_SPECIFIC,
                details=(
                    f"Underlying ${spot_price:.2f} has broken through the "
                    f"wide wing at ${wide_wing_strike:.2f}. Position is in "
                    f"maximum loss territory."
                ),
                urgency=5,
            )

        return None

    # ------------------------------------------------------------------
    # Leg construction
    # ------------------------------------------------------------------

    def construct_legs(
        self,
        spot_price: float,
        options_chain: list[OptionData],
        **kwargs: Any,
    ) -> list[LegSpec]:
        """Build the four legs (three unique strikes) of a broken-wing butterfly.

        Leg structure:
        1. Buy 1 lower put (narrow side, higher strike, closer to ATM).
        2. Sell 2 middle puts (body).
        3. Buy 1 upper put (wide side, lower strike, further OTM).

        Requires the following keyword arguments:

        - ``right``: Option type (``"P"``).
        - ``expiry``: Expiration date.
        - ``body_option``: :class:`OptionData` for the body strike.
        - ``lower_wing_option``: :class:`OptionData` for the narrow wing.
        - ``upper_wing_option``: :class:`OptionData` for the wide wing.

        Args:
            spot_price: Current underlying price.
            options_chain: Full options chain.
            **kwargs: Strategy-specific parameters as described above.

        Returns:
            Four-element list: [lower_wing (buy 1), body (sell 2), upper_wing (buy 1)].
            The body is split into two individual sell legs with quantity=1 each
            for clarity, resulting in a three-element list with the body having
            quantity=2.

        Raises:
            ValueError: If required kwargs are missing.
        """
        right: str = kwargs.get("right", DEFAULT_RIGHT)
        expiry: date | None = kwargs.get("expiry")
        body_option: OptionData | None = kwargs.get("body_option")
        lower_wing_option: OptionData | None = kwargs.get("lower_wing_option")
        upper_wing_option: OptionData | None = kwargs.get("upper_wing_option")

        if expiry is None:
            raise ValueError("Broken-wing butterfly requires an expiry.")
        if body_option is None:
            raise ValueError("Broken-wing butterfly requires a body_option.")
        if lower_wing_option is None:
            raise ValueError("Broken-wing butterfly requires a lower_wing_option.")
        if upper_wing_option is None:
            raise ValueError("Broken-wing butterfly requires an upper_wing_option.")

        # Buy 1 lower put (narrow side, closer to ATM)
        lower_wing_leg = LegSpec(
            action="BUY",
            right=right,
            strike=lower_wing_option.strike,
            expiry=expiry,
            quantity=1,
            mid_price=lower_wing_option.mid_price,
        )

        # Sell 2 middle puts (body)
        body_leg = LegSpec(
            action="SELL",
            right=right,
            strike=body_option.strike,
            expiry=expiry,
            quantity=2,
            mid_price=body_option.mid_price,
        )

        # Buy 1 upper put (wide side, further OTM)
        upper_wing_leg = LegSpec(
            action="BUY",
            right=right,
            strike=upper_wing_option.strike,
            expiry=expiry,
            quantity=1,
            mid_price=upper_wing_option.mid_price,
        )

        self._log.debug(
            "legs_constructed",
            right=right,
            expiry=expiry.isoformat(),
            lower_wing_strike=lower_wing_option.strike,
            body_strike=body_option.strike,
            upper_wing_strike=upper_wing_option.strike,
            lower_wing_mid=lower_wing_option.mid_price,
            body_mid=body_option.mid_price,
            upper_wing_mid=upper_wing_option.mid_price,
        )

        return [lower_wing_leg, body_leg, upper_wing_leg]

    # ------------------------------------------------------------------
    # P&L calculations
    # ------------------------------------------------------------------

    def calculate_max_profit(
        self,
        legs: list[LegSpec],
        net_premium: float,
    ) -> float:
        """Calculate maximum profit for a broken-wing butterfly.

        Max profit occurs at the body strike and equals:
            net_credit + (narrow_width * 100)

        where ``narrow_width`` is the distance between the lower wing (narrow
        side) and the body strike, and ``net_credit`` is the absolute value
        of the credit received (net_premium is negative for credits).

        Args:
            legs: The BWB legs.
            net_premium: Net premium (negative = credit).

        Returns:
            Maximum profit in dollars (positive value, per spread unit).
        """
        narrow_width = self._extract_narrow_width(legs)
        # net_premium is negative for credit entries, so abs gives the credit
        net_credit_dollars = abs(net_premium) * 100.0
        max_profit = net_credit_dollars + (narrow_width * 100.0)
        return round(max_profit, 2)

    def calculate_max_loss(
        self,
        legs: list[LegSpec],
        net_premium: float,
    ) -> float:
        """Calculate maximum loss for a broken-wing butterfly.

        Max loss occurs on the wide side (below the upper wing strike) and
        equals:
            (wide_width - narrow_width - net_credit) * 100

        where ``wide_width`` is the distance between the body and the upper
        wing, ``narrow_width`` is the distance between the lower wing and
        the body, and ``net_credit`` is the absolute credit received.

        Args:
            legs: The BWB legs.
            net_premium: Net premium (negative = credit).

        Returns:
            Maximum loss in dollars (positive value, per spread unit).
        """
        narrow_width = self._extract_narrow_width(legs)
        wide_width = self._extract_wide_width(legs)
        net_credit_per_share = abs(net_premium)

        max_loss = (wide_width - narrow_width - net_credit_per_share) * 100.0
        # Max loss cannot be negative (would mean risk-free trade)
        return round(max(max_loss, 0.0), 2)

    # ------------------------------------------------------------------
    # Order construction & Greeks
    # ------------------------------------------------------------------

    async def construct_order(
        self,
        signal: TradeSignal,
        contract_factory: Any,
    ) -> Any:
        """Build an IBKR combo order from the trade signal legs.

        Args:
            signal: The approved trade signal with leg specifications.
            contract_factory: A :class:`ContractFactory` instance for building
                IBKR combo contracts.

        Returns:
            An IBKR ``Contract`` object representing the multi-leg spread.
        """
        legs_for_broker: list[dict[str, Any]] = []
        for leg in signal.legs:
            expiry = leg.expiry
            if hasattr(expiry, "strftime"):
                expiry = expiry.strftime("%Y%m%d")
            legs_for_broker.append(
                {
                    "action": leg.action,
                    "expiry": expiry,
                    "strike": leg.strike,
                    "right": leg.right,
                    "ratio": leg.quantity,
                }
            )
        return await contract_factory.build_spread(
            ticker=signal.ticker,
            legs=legs_for_broker,
        )

    def calculate_greeks(
        self,
        legs: list[LegSpec],
        greeks: dict[str, float],
    ) -> dict[str, float]:
        """Aggregate Greeks across all legs of the spread.

        Args:
            legs: The constructed leg specifications.
            greeks: Per-leg Greeks keyed by ``strike_right`` identifier.

        Returns:
            Dictionary with aggregated delta, gamma, theta, vega.
        """
        total_delta = 0.0
        total_gamma = 0.0
        total_theta = 0.0
        total_vega = 0.0
        for leg in legs:
            multiplier = leg.quantity if leg.action == "BUY" else -leg.quantity
            key = f"{leg.strike}_{leg.right}"
            leg_greeks = greeks.get(key, {})
            total_delta += multiplier * float(leg_greeks.get("delta", 0.0))
            total_gamma += multiplier * float(leg_greeks.get("gamma", 0.0))
            total_theta += multiplier * float(leg_greeks.get("theta", 0.0))
            total_vega += multiplier * float(leg_greeks.get("vega", 0.0))
        return {
            "delta": round(total_delta, 6),
            "gamma": round(total_gamma, 6),
            "theta": round(total_theta, 6),
            "vega": round(total_vega, 6),
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_dte_params(self) -> tuple[int, int, int]:
        """Return (target_dte, min_dte, max_dte) from config or defaults.

        Returns:
            Tuple of (target, min, max) DTE values.
        """
        target_dte = self._config.target_dte
        dte_range = self._config.dte_range

        if isinstance(target_dte, dict):
            target = target_dte.get("target", TARGET_DTE)
        else:
            target = target_dte if isinstance(target_dte, int) else TARGET_DTE

        if isinstance(dte_range, dict):
            range_list = dte_range.get("range", [MIN_DTE, MAX_DTE])
        elif isinstance(dte_range, list) and len(dte_range) == 2:
            range_list = dte_range
        else:
            range_list = [MIN_DTE, MAX_DTE]

        return target, range_list[0], range_list[1]

    def _get_body_delta_range(self) -> tuple[float, float]:
        """Return the delta range for the body (short puts).

        Returns:
            Tuple of (min_delta, max_delta) for the body puts.
        """
        delta_range = self._config.delta_range
        if "body_strike" in delta_range:
            raw = delta_range["body_strike"]
            if isinstance(raw, list) and len(raw) == 2:
                return float(raw[0]), float(raw[1])
        return BODY_DELTA_MIN, BODY_DELTA_MAX

    def _get_wing_widths(self) -> tuple[float, float]:
        """Return (narrow_width, wide_width) from config or defaults.

        Returns:
            Tuple of (narrow_width, wide_width) in dollars.
        """
        wing_widths = self._config.wing_widths
        if wing_widths is not None:
            narrow = wing_widths.get("narrow", NARROW_WING_WIDTH)
            wide = wing_widths.get("wide", WIDE_WING_WIDTH)
            return float(narrow), float(wide)
        return NARROW_WING_WIDTH, WIDE_WING_WIDTH

    def _find_expiry_near_dte(
        self,
        options_chain: list[OptionData],
        target_dte: int,
        min_dte: int,
        max_dte: int,
    ) -> date | None:
        """Find the expiration date closest to *target_dte* within bounds.

        Args:
            options_chain: Full options chain.
            target_dte: Ideal days to expiration.
            min_dte: Minimum acceptable DTE.
            max_dte: Maximum acceptable DTE.

        Returns:
            Best matching expiry date, or ``None`` if none qualifies.
        """
        today = date.today()
        expiries: set[date] = {o.expiry for o in options_chain}
        candidates: list[tuple[int, date]] = []

        for exp in expiries:
            dte = (exp - today).days
            if min_dte <= dte <= max_dte:
                candidates.append((abs(dte - target_dte), exp))

        if not candidates:
            return None

        candidates.sort(key=lambda x: x[0])
        return candidates[0][1]

    @staticmethod
    def _find_option_by_strike(
        options: list[OptionData],
        target_strike: float,
    ) -> OptionData | None:
        """Find an option at or closest to the target strike price.

        Looks for an exact match first.  If no exact match is found, returns
        the option whose strike is closest (within $1 tolerance).

        Args:
            options: Options filtered by expiry and right.
            target_strike: Desired strike price.

        Returns:
            Matching :class:`OptionData`, or ``None`` if no suitable option.
        """
        # Exact match first
        for opt in options:
            if (
                opt.strike == target_strike
                and opt.open_interest >= DEFAULT_MIN_OPEN_INTEREST
            ):
                return opt

        # Closest within $1 tolerance
        tolerance = 1.0
        best: OptionData | None = None
        best_distance = float("inf")

        for opt in options:
            if opt.open_interest < DEFAULT_MIN_OPEN_INTEREST:
                continue
            distance = abs(opt.strike - target_strike)
            if distance <= tolerance and distance < best_distance:
                best_distance = distance
                best = opt

        return best

    @staticmethod
    def _calculate_net_premium(legs: list[LegSpec]) -> float:
        """Calculate net premium from legs (positive = debit, negative = credit).

        For a BWB entered for credit, the result should be negative.

        Args:
            legs: The spread legs.

        Returns:
            Net premium per share.
        """
        net = 0.0
        for leg in legs:
            if leg.action == "BUY":
                net += leg.mid_price * leg.quantity
            else:
                net -= leg.mid_price * leg.quantity
        return round(net, 4)

    @staticmethod
    def _extract_narrow_width(legs: list[LegSpec]) -> float:
        """Extract the narrow wing width from the BWB legs.

        The narrow width is the distance between the lower wing (highest
        strike, bought) and the body (middle strike, sold).

        For a put BWB:
        - Lower wing (buy) has the highest strike (closest to ATM).
        - Body (sell) is the middle strike.
        - Narrow width = lower_wing_strike - body_strike.

        Args:
            legs: The BWB legs.

        Returns:
            Narrow wing width in dollars.
        """
        buy_strikes: list[float] = []
        sell_strike: float = 0.0

        for leg in legs:
            if leg.action == "BUY":
                buy_strikes.append(leg.strike)
            elif leg.action == "SELL":
                sell_strike = leg.strike

        if len(buy_strikes) < 2 or sell_strike == 0.0:
            return NARROW_WING_WIDTH

        # Lower wing is the higher of the two buy strikes (closer to ATM for puts)
        lower_wing_strike = max(buy_strikes)
        return abs(lower_wing_strike - sell_strike)

    @staticmethod
    def _extract_wide_width(legs: list[LegSpec]) -> float:
        """Extract the wide wing width from the BWB legs.

        The wide width is the distance between the body (middle strike,
        sold) and the upper wing (lowest strike, bought).

        For a put BWB:
        - Body (sell) is the middle strike.
        - Upper wing (buy) has the lowest strike (furthest OTM).
        - Wide width = body_strike - upper_wing_strike.

        Args:
            legs: The BWB legs.

        Returns:
            Wide wing width in dollars.
        """
        buy_strikes: list[float] = []
        sell_strike: float = 0.0

        for leg in legs:
            if leg.action == "BUY":
                buy_strikes.append(leg.strike)
            elif leg.action == "SELL":
                sell_strike = leg.strike

        if len(buy_strikes) < 2 or sell_strike == 0.0:
            return WIDE_WING_WIDTH

        # Upper wing is the lower of the two buy strikes (further OTM for puts)
        upper_wing_strike = min(buy_strikes)
        return abs(sell_strike - upper_wing_strike)

    @staticmethod
    def _get_wide_wing_strike(trade: TradeRecord) -> float | None:
        """Extract the wide wing (furthest OTM) strike from the trade.

        For a put BWB, the wide wing is the bought option with the lowest
        strike price.

        Args:
            trade: The open trade record.

        Returns:
            The wide wing strike price, or ``None`` if not determinable.
        """
        buy_strikes: list[float] = []
        for leg in trade.legs:
            if leg.action == "BUY":
                buy_strikes.append(leg.strike)

        if not buy_strikes:
            return None

        # The wide wing is the lowest strike among the bought legs
        return min(buy_strikes)
