"""Calendar spread strategy implementation.

A calendar spread (also called a time spread or horizontal spread) sells a
near-term option and buys a same-strike longer-term option.  The trade profits
from the accelerated time decay of the short front-month leg relative to the
long back-month leg, and from an increase in implied volatility.

Typical use: neutral outlook on the underlying with an expectation that price
stays near the chosen strike through front-month expiration.

Usage::

    from src.strategies.calendar_spread import CalendarSpread
    from src.strategies.base import StrategyConfig

    config = StrategyConfig(...)
    strategy = CalendarSpread(config)
    signal = await strategy.check_entry("AAPL", 175.0, 35.0, "range_bound",
                                         greeks, options_chain)
"""

from __future__ import annotations

from datetime import date
from typing import Any

from src.strategies.base import (
    DEFAULT_CLOSE_BEFORE_EXPIRY_DTE,
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

logger = get_logger("strategy.calendar_spread")

# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------

# Front month DTE bounds
FRONT_MONTH_TARGET_DTE: int = 30
FRONT_MONTH_MIN_DTE: int = 21
FRONT_MONTH_MAX_DTE: int = 45

# Back month DTE bounds
BACK_MONTH_TARGET_DTE: int = 60
BACK_MONTH_MIN_DTE: int = 45
BACK_MONTH_MAX_DTE: int = 90

# ATM delta range (absolute value)
ATM_DELTA_MIN: float = 0.45
ATM_DELTA_MAX: float = 0.55

# Exit thresholds
PROFIT_TARGET_PCT: float = 0.25
STOP_LOSS_PCT: float = 0.50
FRONT_MONTH_DTE_EXIT: int = 7
UNDERLYING_MOVE_EXIT_PCT: float = 0.05

# Calendar spreads are typically entered with puts for a neutral stance
DEFAULT_RIGHT: str = "P"


# ---------------------------------------------------------------------------
# CalendarSpread strategy
# ---------------------------------------------------------------------------


class CalendarSpread(BaseStrategy):
    """Calendar spread: sell front-month ATM, buy back-month ATM at the same strike.

    The trade is entered for a net debit (long calendar) and profits when:

    - The underlying stays near the chosen strike.
    - The front-month option decays faster than the back-month.
    - Implied volatility increases (long vega position).

    Entry criteria:
        - Front month DTE: 21--45 days (target 30).
        - Back month DTE: 45--90 days (target 60).
        - Strike: ATM (absolute delta 0.45--0.55).
        - Entered as puts by default for a neutral stance.

    Exit rules:
        - Profit target: 25% of debit paid.
        - Stop loss: 50% of debit paid.
        - Front month DTE <= 7.
        - Underlying moves more than 5% from the entry strike.
    """

    def __init__(self, config: StrategyConfig) -> None:
        super().__init__(config)
        self._log = get_logger(f"strategy.{self.name}")

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        """Return the canonical strategy name."""
        return "calendar_spread"

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
        """Evaluate whether a calendar spread entry is warranted.

        Steps:
        1. Verify eligibility (regime, IV rank, underlying price).
        2. Determine the option right to use (puts for neutral).
        3. Find suitable front-month and back-month expirations.
        4. Find ATM strikes within the delta range.
        5. Construct legs and validate liquidity.
        6. Calculate P&L bounds and build the trade signal.

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

        # Determine option right from config or default
        right = self._get_option_right()

        # Find front-month and back-month expirations
        front_dte_target, front_min, front_max = self._get_front_month_dte()
        back_dte_target, back_min, back_max = self._get_back_month_dte()

        front_expiry = self._find_expiry_near_dte(
            options_chain, front_dte_target, front_min, front_max
        )
        if front_expiry is None:
            self._log.debug(
                "no_front_expiry",
                ticker=ticker,
                target_dte=front_dte_target,
                min_dte=front_min,
                max_dte=front_max,
            )
            return None

        back_expiry = self._find_expiry_near_dte(
            options_chain, back_dte_target, back_min, back_max
        )
        if back_expiry is None:
            self._log.debug(
                "no_back_expiry",
                ticker=ticker,
                target_dte=back_dte_target,
                min_dte=back_min,
                max_dte=back_max,
            )
            return None

        # Calendar spreads require different expirations
        if front_expiry == back_expiry:
            self._log.debug(
                "same_expiry_front_back",
                ticker=ticker,
                expiry=front_expiry.isoformat(),
            )
            return None

        # Find ATM option in the front month for the strike
        delta_min, delta_max = self._get_atm_delta_range(right)
        front_options = [
            o for o in options_chain if o.expiry == front_expiry and o.right == right
        ]
        front_atm = self.find_option_by_delta(
            front_options, right, delta_min, delta_max
        )
        if front_atm is None:
            self._log.debug(
                "no_front_atm_option",
                ticker=ticker,
                expiry=front_expiry.isoformat(),
                delta_range=[delta_min, delta_max],
            )
            return None

        # Validate front-month liquidity
        if not self.validate_bid_ask_spread(front_atm):
            self._log.debug(
                "front_atm_illiquid",
                ticker=ticker,
                strike=front_atm.strike,
                bid=front_atm.bid,
                ask=front_atm.ask,
            )
            return None

        # Find the same strike in the back month
        strike = front_atm.strike
        back_options = [
            o for o in options_chain if o.expiry == back_expiry and o.right == right
        ]
        back_option = self._find_option_by_strike(back_options, strike)
        if back_option is None:
            self._log.debug(
                "no_back_month_at_strike",
                ticker=ticker,
                strike=strike,
                expiry=back_expiry.isoformat(),
            )
            return None

        # Validate back-month liquidity
        if not self.validate_bid_ask_spread(back_option):
            self._log.debug(
                "back_option_illiquid",
                ticker=ticker,
                strike=back_option.strike,
                bid=back_option.bid,
                ask=back_option.ask,
            )
            return None

        # Calendar spread must be entered for a net debit (back > front)
        if back_option.mid_price <= front_atm.mid_price:
            self._log.debug(
                "no_debit_calendar",
                ticker=ticker,
                front_mid=front_atm.mid_price,
                back_mid=back_option.mid_price,
            )
            return None

        # Construct legs
        try:
            legs = self.construct_legs(
                spot_price=spot_price,
                options_chain=options_chain,
                right=right,
                front_expiry=front_expiry,
                back_expiry=back_expiry,
                strike=strike,
                front_option=front_atm,
                back_option=back_option,
            )
        except ValueError as exc:
            self._log.debug("construct_legs_failed", ticker=ticker, error=str(exc))
            return None

        # Calculate net premium (positive = debit)
        net_premium = self._calculate_net_premium(legs)
        if net_premium <= 0:
            self._log.debug(
                "non_debit_calendar",
                ticker=ticker,
                net_premium=net_premium,
            )
            return None

        # Calculate P&L bounds
        max_profit = self.calculate_max_profit(legs, net_premium)
        max_loss = self.calculate_max_loss(legs, net_premium)

        front_dte = (front_expiry - date.today()).days
        back_dte = (back_expiry - date.today()).days

        reasoning = (
            f"Calendar spread on {ticker}: sell {right} {strike} "
            f"({front_dte}d) / buy {right} {strike} ({back_dte}d). "
            f"Net debit ${net_premium:.2f}. "
            f"IV rank {iv_rank:.1f}, regime {regime}. "
            f"Front delta {front_atm.delta:.3f}, back delta {back_option.delta:.3f}."
        )

        self._log.info(
            "entry_signal_generated",
            ticker=ticker,
            strike=strike,
            right=right,
            front_expiry=front_expiry.isoformat(),
            back_expiry=back_expiry.isoformat(),
            net_premium=net_premium,
            max_profit=max_profit,
            max_loss=max_loss,
            iv_rank=iv_rank,
            regime=regime,
        )

        return TradeSignal(
            strategy=self.name,
            ticker=ticker,
            direction="LONG",
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
        """Evaluate whether an open calendar spread should be closed.

        Checks in order:
        1. Mechanical exits (profit target, stop loss, DTE limit).
        2. Front month DTE < 7: close to avoid pin risk and gamma expansion.
        3. Underlying has moved > 5% from entry strike: position is losing
           its time-spread edge.

        Args:
            trade: Record of the open calendar spread.
            spot_price: Current underlying price.
            current_pnl: Unrealised P&L in dollars.
            current_pnl_pct: Unrealised P&L as fraction of max profit/loss.
            dte_remaining: Days to expiration of the front (nearest) leg.
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

        # 2. Front month DTE approaching expiration
        front_dte_exit = FRONT_MONTH_DTE_EXIT
        if (
            dte_remaining <= front_dte_exit
            and dte_remaining > DEFAULT_CLOSE_BEFORE_EXPIRY_DTE
        ):
            self._log.info(
                "exit_front_month_dte",
                trade_id=str(trade.id),
                ticker=trade.ticker,
                dte_remaining=dte_remaining,
                threshold=front_dte_exit,
            )
            return ExitSignal(
                trade_id=trade.id,
                reason=ExitReason.STRATEGY_SPECIFIC,
                details=(
                    f"Front month DTE {dte_remaining} is below threshold "
                    f"{front_dte_exit} — closing to avoid gamma risk."
                ),
                urgency=3,
            )

        # 3. Underlying moved too far from entry strike
        entry_strike = self._get_entry_strike(trade)
        if entry_strike is not None and entry_strike > 0:
            move_pct = abs(spot_price - entry_strike) / entry_strike
            if move_pct > UNDERLYING_MOVE_EXIT_PCT:
                self._log.info(
                    "exit_underlying_move",
                    trade_id=str(trade.id),
                    ticker=trade.ticker,
                    spot_price=spot_price,
                    entry_strike=entry_strike,
                    move_pct=round(move_pct, 4),
                    threshold=UNDERLYING_MOVE_EXIT_PCT,
                )
                return ExitSignal(
                    trade_id=trade.id,
                    reason=ExitReason.STRATEGY_SPECIFIC,
                    details=(
                        f"Underlying moved {move_pct:.1%} from entry strike "
                        f"${entry_strike:.2f} "
                        f"(threshold {UNDERLYING_MOVE_EXIT_PCT:.0%}). "
                        f"Current price ${spot_price:.2f}."
                    ),
                    urgency=3,
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
        """Build the two legs of a calendar spread.

        Requires the following keyword arguments (passed from ``check_entry``):

        - ``right``: Option type (``"C"`` or ``"P"``).
        - ``front_expiry``: Front month expiration date.
        - ``back_expiry``: Back month expiration date.
        - ``strike``: The shared strike price.
        - ``front_option``: The :class:`OptionData` for the front leg.
        - ``back_option``: The :class:`OptionData` for the back leg.

        Args:
            spot_price: Current underlying price.
            options_chain: Full options chain (used as fallback lookup).
            **kwargs: Strategy-specific parameters as described above.

        Returns:
            Two-element list: [short front leg, long back leg].

        Raises:
            ValueError: If required kwargs are missing or invalid.
        """
        right: str = kwargs.get("right", DEFAULT_RIGHT)
        front_expiry: date | None = kwargs.get("front_expiry")
        back_expiry: date | None = kwargs.get("back_expiry")
        strike: float | None = kwargs.get("strike")
        front_option: OptionData | None = kwargs.get("front_option")
        back_option: OptionData | None = kwargs.get("back_option")

        if front_expiry is None or back_expiry is None:
            raise ValueError(
                "Calendar spread requires both front_expiry and back_expiry."
            )
        if strike is None:
            raise ValueError("Calendar spread requires a strike price.")

        # If specific option objects were provided, use their mid prices
        front_mid = front_option.mid_price if front_option is not None else 0.0
        back_mid = back_option.mid_price if back_option is not None else 0.0

        # Sell front month (short), buy back month (long) — same strike
        short_front = LegSpec(
            action="SELL",
            right=right,
            strike=strike,
            expiry=front_expiry,
            quantity=1,
            mid_price=front_mid,
        )
        long_back = LegSpec(
            action="BUY",
            right=right,
            strike=strike,
            expiry=back_expiry,
            quantity=1,
            mid_price=back_mid,
        )

        self._log.debug(
            "legs_constructed",
            strike=strike,
            right=right,
            front_expiry=front_expiry.isoformat(),
            back_expiry=back_expiry.isoformat(),
            front_mid=front_mid,
            back_mid=back_mid,
        )

        return [short_front, long_back]

    # ------------------------------------------------------------------
    # P&L calculations
    # ------------------------------------------------------------------

    def calculate_max_profit(
        self,
        legs: list[LegSpec],
        net_premium: float,
    ) -> float:
        """Estimate maximum profit for a calendar spread.

        The true maximum profit of a calendar spread depends on the implied
        volatility of the back-month option at front-month expiration, which
        cannot be known in advance.  As a conservative estimate, we use 25%
        of the net debit paid.

        Args:
            legs: The calendar spread legs.
            net_premium: Net debit paid (positive value).

        Returns:
            Estimated maximum profit in dollars (per spread unit).
        """
        # Conservative estimate: 25% of debit paid (per share), scaled by 100
        estimated_max_profit = abs(net_premium) * 0.25 * 100.0
        return round(estimated_max_profit, 2)

    def calculate_max_loss(
        self,
        legs: list[LegSpec],
        net_premium: float,
    ) -> float:
        """Calculate maximum loss for a calendar spread.

        Maximum loss is the net debit paid (the entire premium invested),
        which occurs if the underlying moves far away from the strike.

        Args:
            legs: The calendar spread legs.
            net_premium: Net debit paid (positive value).

        Returns:
            Maximum loss in dollars (positive value, per spread unit).
        """
        return round(abs(net_premium) * 100.0, 2)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_option_right(self) -> str:
        """Determine the option right (C or P) from config or default.

        Returns:
            ``"C"`` or ``"P"``.
        """
        delta_range = self._config.delta_range
        # Config has delta_range.strike — implies neutral, use puts
        if "strike" in delta_range:
            return DEFAULT_RIGHT
        return DEFAULT_RIGHT

    def _get_front_month_dte(self) -> tuple[int, int, int]:
        """Return (target_dte, min_dte, max_dte) for the front month.

        Reads from config if available, otherwise uses module constants.

        Returns:
            Tuple of (target, min, max) DTE values.
        """
        target_dte = self._config.target_dte
        dte_range = self._config.dte_range

        if isinstance(target_dte, dict):
            front_target = target_dte.get("front_month", FRONT_MONTH_TARGET_DTE)
        else:
            front_target = FRONT_MONTH_TARGET_DTE

        if isinstance(dte_range, dict):
            front_range = dte_range.get(
                "front_month", [FRONT_MONTH_MIN_DTE, FRONT_MONTH_MAX_DTE]
            )
        else:
            front_range = [FRONT_MONTH_MIN_DTE, FRONT_MONTH_MAX_DTE]

        return front_target, front_range[0], front_range[1]

    def _get_back_month_dte(self) -> tuple[int, int, int]:
        """Return (target_dte, min_dte, max_dte) for the back month.

        Reads from config if available, otherwise uses module constants.

        Returns:
            Tuple of (target, min, max) DTE values.
        """
        target_dte = self._config.target_dte
        dte_range = self._config.dte_range

        if isinstance(target_dte, dict):
            back_target = target_dte.get("back_month", BACK_MONTH_TARGET_DTE)
        else:
            back_target = BACK_MONTH_TARGET_DTE

        if isinstance(dte_range, dict):
            back_range = dte_range.get(
                "back_month", [BACK_MONTH_MIN_DTE, BACK_MONTH_MAX_DTE]
            )
        else:
            back_range = [BACK_MONTH_MIN_DTE, BACK_MONTH_MAX_DTE]

        return back_target, back_range[0], back_range[1]

    def _get_atm_delta_range(self, right: str) -> tuple[float, float]:
        """Return the delta range for ATM options based on option right.

        For puts, deltas are negative so the range is returned as negative.
        For calls, deltas are positive.

        Args:
            right: ``"C"`` or ``"P"``.

        Returns:
            Tuple of (min_delta, max_delta) in the sign convention of the
            option type.
        """
        delta_range = self._config.delta_range
        if "strike" in delta_range:
            raw = delta_range["strike"]
            if isinstance(raw, list) and len(raw) == 2:
                low, high = float(raw[0]), float(raw[1])
            else:
                low, high = ATM_DELTA_MIN, ATM_DELTA_MAX
        else:
            low, high = ATM_DELTA_MIN, ATM_DELTA_MAX

        if right == "P":
            return -high, -low  # e.g. -0.55 to -0.45
        return low, high  # e.g. 0.45 to 0.55

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
        strike: float,
    ) -> OptionData | None:
        """Find an option at exactly the given strike price.

        Args:
            options: Options filtered by expiry and right.
            strike: Target strike price.

        Returns:
            Matching :class:`OptionData`, or ``None`` if not found.
        """
        for opt in options:
            if opt.strike == strike and opt.open_interest >= DEFAULT_MIN_OPEN_INTEREST:
                return opt
        return None

    @staticmethod
    def _calculate_net_premium(legs: list[LegSpec]) -> float:
        """Calculate net premium from legs (positive = debit).

        Args:
            legs: The spread legs.

        Returns:
            Net premium per share (multiply by 100 for per-contract).
        """
        net = 0.0
        for leg in legs:
            if leg.action == "BUY":
                net += leg.mid_price * leg.quantity
            else:
                net -= leg.mid_price * leg.quantity
        return round(net, 4)

    @staticmethod
    def _get_entry_strike(trade: TradeRecord) -> float | None:
        """Extract the common strike from a calendar spread's legs.

        In a calendar spread both legs share the same strike. Returns the
        strike of the first leg, or ``None`` if the trade has no legs.

        Args:
            trade: The open trade record.

        Returns:
            The entry strike price, or ``None``.
        """
        if trade.legs:
            return trade.legs[0].strike
        return None
