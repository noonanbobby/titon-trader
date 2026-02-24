"""Diagonal spread strategy implementation.

A diagonal spread combines elements of a vertical spread and a calendar
spread: buy a longer-term, deeper in-the-money (or ATM) option and sell a
shorter-term out-of-the-money option at a *different* strike and expiration.

The strategy profits from time decay of the short leg, directional movement
toward the short strike, and potential implied volatility increases on the
long leg.

Typical use: moderately bullish outlook (call diagonal) where the trader
wants to finance a longer-dated directional position by selling near-term
premium.

Usage::

    from src.strategies.diagonal_spread import DiagonalSpread
    from src.strategies.base import StrategyConfig

    config = StrategyConfig(...)
    strategy = DiagonalSpread("diagonal_spread", config)
    signal = await strategy.check_entry("AAPL", 175.0, 25.0, "low_vol_trend",
                                         greeks, options_chain)
"""

from __future__ import annotations

from datetime import date
from typing import Any

from src.strategies.base import (
    DEFAULT_CLOSE_BEFORE_EXPIRY_DTE,
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

logger = get_logger("strategy.diagonal_spread")

# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------

# Long leg (back month, higher delta / ITM-ish)
LONG_LEG_DELTA_MIN: float = 0.60
LONG_LEG_DELTA_MAX: float = 0.75
LONG_LEG_TARGET_DTE: int = 60
LONG_LEG_MIN_DTE: int = 45
LONG_LEG_MAX_DTE: int = 90

# Short leg (front month, lower delta / OTM)
SHORT_LEG_DELTA_MIN: float = 0.25
SHORT_LEG_DELTA_MAX: float = 0.40
SHORT_LEG_TARGET_DTE: int = 30
SHORT_LEG_MIN_DTE: int = 21
SHORT_LEG_MAX_DTE: int = 45

# Exit thresholds
PROFIT_TARGET_PCT: float = 0.25
STOP_LOSS_PCT: float = 0.50
SHORT_LEG_DTE_ROLL: int = 7
UNDERLYING_BELOW_LONG_STRIKE_EXIT: bool = True

# Default to call diagonal for bullish outlook
DEFAULT_RIGHT: str = "C"


# ---------------------------------------------------------------------------
# DiagonalSpread strategy
# ---------------------------------------------------------------------------


class DiagonalSpread(BaseStrategy):
    """Diagonal spread: buy longer-term ITM/ATM option, sell shorter-term OTM option.

    The trade is entered for a net debit and profits when:

    - The underlying moves toward (but not through) the short strike.
    - The short leg decays faster than the long leg.
    - The long leg retains or gains value.

    Entry criteria:
        - Long leg: delta 0.60--0.75, DTE 45--90 (target 60).
        - Short leg: delta 0.25--0.40, DTE 21--45 (target 30).
        - Typically call diagonal for bullish outlook.
        - Short leg strike must be above long leg strike (for calls).

    Exit rules:
        - Profit target: 25% of debit paid.
        - Stop loss: 50% of debit paid.
        - Short leg DTE <= 7 (roll or close).
        - Underlying drops below long leg strike (for calls).
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
        """Evaluate whether a diagonal spread entry is warranted.

        Steps:
        1. Verify eligibility (regime, IV rank, underlying price).
        2. Determine option right (calls for bullish).
        3. Find suitable long-leg and short-leg expirations.
        4. Find ITM/ATM option for the long leg (higher delta).
        5. Find OTM option for the short leg (lower delta).
        6. Validate strike ordering: for calls, short strike > long strike.
        7. Construct legs and build the trade signal.

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

        right = self._get_option_right()

        # Find expirations for long and short legs
        long_target, long_min, long_max = self._get_long_leg_dte()
        short_target, short_min, short_max = self._get_short_leg_dte()

        long_expiry = self._find_expiry_near_dte(
            options_chain, long_target, long_min, long_max
        )
        if long_expiry is None:
            self._log.debug(
                "no_long_expiry",
                ticker=ticker,
                target_dte=long_target,
                min_dte=long_min,
                max_dte=long_max,
            )
            return None

        short_expiry = self._find_expiry_near_dte(
            options_chain, short_target, short_min, short_max
        )
        if short_expiry is None:
            self._log.debug(
                "no_short_expiry",
                ticker=ticker,
                target_dte=short_target,
                min_dte=short_min,
                max_dte=short_max,
            )
            return None

        # Diagonal requires different expirations
        if long_expiry == short_expiry:
            self._log.debug(
                "same_expiry_diagonal",
                ticker=ticker,
                expiry=long_expiry.isoformat(),
            )
            return None

        # Short leg must expire before long leg
        if short_expiry >= long_expiry:
            self._log.debug(
                "short_expiry_not_before_long",
                ticker=ticker,
                short_expiry=short_expiry.isoformat(),
                long_expiry=long_expiry.isoformat(),
            )
            return None

        # Find long leg option (higher delta, ITM/ATM)
        long_delta_min, long_delta_max = self._get_long_delta_range(right)
        long_options = [
            o for o in options_chain if o.expiry == long_expiry and o.right == right
        ]
        long_option = self.find_option_by_delta(
            long_options, right, long_delta_min, long_delta_max
        )
        if long_option is None:
            self._log.debug(
                "no_long_leg_option",
                ticker=ticker,
                expiry=long_expiry.isoformat(),
                delta_range=[long_delta_min, long_delta_max],
            )
            return None

        if not self.validate_bid_ask_spread(long_option):
            self._log.debug(
                "long_leg_illiquid",
                ticker=ticker,
                strike=long_option.strike,
                bid=long_option.bid,
                ask=long_option.ask,
            )
            return None

        # Find short leg option (lower delta, OTM)
        short_delta_min, short_delta_max = self._get_short_delta_range(right)
        short_options = [
            o for o in options_chain if o.expiry == short_expiry and o.right == right
        ]
        short_option = self.find_option_by_delta(
            short_options, right, short_delta_min, short_delta_max
        )
        if short_option is None:
            self._log.debug(
                "no_short_leg_option",
                ticker=ticker,
                expiry=short_expiry.isoformat(),
                delta_range=[short_delta_min, short_delta_max],
            )
            return None

        if not self.validate_bid_ask_spread(short_option):
            self._log.debug(
                "short_leg_illiquid",
                ticker=ticker,
                strike=short_option.strike,
                bid=short_option.bid,
                ask=short_option.ask,
            )
            return None

        # Validate strike ordering
        if not self._validate_strike_order(
            right, long_option.strike, short_option.strike
        ):
            self._log.debug(
                "invalid_strike_order",
                ticker=ticker,
                right=right,
                long_strike=long_option.strike,
                short_strike=short_option.strike,
            )
            return None

        # Construct legs
        try:
            legs = self.construct_legs(
                spot_price=spot_price,
                options_chain=options_chain,
                right=right,
                long_expiry=long_expiry,
                short_expiry=short_expiry,
                long_option=long_option,
                short_option=short_option,
            )
        except ValueError as exc:
            self._log.debug("construct_legs_failed", ticker=ticker, error=str(exc))
            return None

        # Calculate net premium (positive = debit)
        net_premium = self._calculate_net_premium(legs)
        if net_premium <= 0:
            self._log.debug(
                "non_debit_diagonal",
                ticker=ticker,
                net_premium=net_premium,
            )
            return None

        max_profit = self.calculate_max_profit(legs, net_premium)
        max_loss = self.calculate_max_loss(legs, net_premium)

        long_dte = (long_expiry - date.today()).days
        short_dte = (short_expiry - date.today()).days

        reasoning = (
            f"Diagonal spread on {ticker}: buy {right} {long_option.strike} "
            f"({long_dte}d, delta {long_option.delta:.3f}) / "
            f"sell {right} {short_option.strike} "
            f"({short_dte}d, delta {short_option.delta:.3f}). "
            f"Net debit ${net_premium:.2f}. "
            f"IV rank {iv_rank:.1f}, regime {regime}."
        )

        self._log.info(
            "entry_signal_generated",
            ticker=ticker,
            right=right,
            long_strike=long_option.strike,
            short_strike=short_option.strike,
            long_expiry=long_expiry.isoformat(),
            short_expiry=short_expiry.isoformat(),
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
        """Evaluate whether an open diagonal spread should be closed.

        Checks in order:
        1. Mechanical exits (profit target, stop loss, DTE limit).
        2. Short leg DTE < 7: should roll or close the position.
        3. For call diagonals: underlying drops below long strike.
           For put diagonals: underlying rises above long strike.

        Args:
            trade: Record of the open diagonal spread.
            spot_price: Current underlying price.
            current_pnl: Unrealised P&L in dollars.
            current_pnl_pct: Unrealised P&L as fraction of max profit/loss.
            dte_remaining: Days to expiration of the short (nearest) leg.
            greeks: Current Greeks of the position.

        Returns:
            An :class:`ExitSignal` if any exit rule triggers, else ``None``.
        """
        # 1. Mechanical exits
        mechanical = self.check_mechanical_exit(
            trade, current_pnl, current_pnl_pct, dte_remaining
        )
        if mechanical is not None:
            return mechanical

        # 2. Short leg approaching expiration — needs roll or close
        if (
            dte_remaining <= SHORT_LEG_DTE_ROLL
            and dte_remaining > DEFAULT_CLOSE_BEFORE_EXPIRY_DTE
        ):
            self._log.info(
                "exit_short_leg_dte_roll",
                trade_id=str(trade.id),
                ticker=trade.ticker,
                dte_remaining=dte_remaining,
                threshold=SHORT_LEG_DTE_ROLL,
            )
            return ExitSignal(
                trade_id=trade.id,
                reason=ExitReason.STRATEGY_SPECIFIC,
                details=(
                    f"Short leg DTE {dte_remaining} is at or below roll "
                    f"threshold {SHORT_LEG_DTE_ROLL}. Close or roll the "
                    f"short leg to manage gamma risk."
                ),
                urgency=3,
            )

        # 3. Underlying has moved adversely relative to long strike
        long_strike = self._get_long_strike(trade)
        if long_strike is not None and long_strike > 0:
            right = self._get_trade_right(trade)

            if right == "C" and spot_price < long_strike:
                # For call diagonal, underlying below the long call strike
                # means the position is losing its directional edge
                self._log.info(
                    "exit_underlying_below_long_strike",
                    trade_id=str(trade.id),
                    ticker=trade.ticker,
                    spot_price=spot_price,
                    long_strike=long_strike,
                )
                return ExitSignal(
                    trade_id=trade.id,
                    reason=ExitReason.STRATEGY_SPECIFIC,
                    details=(
                        f"Underlying ${spot_price:.2f} has dropped below "
                        f"long call strike ${long_strike:.2f}. "
                        f"Directional thesis invalidated."
                    ),
                    urgency=4,
                )

            if right == "P" and spot_price > long_strike:
                # For put diagonal, underlying above the long put strike
                self._log.info(
                    "exit_underlying_above_long_strike",
                    trade_id=str(trade.id),
                    ticker=trade.ticker,
                    spot_price=spot_price,
                    long_strike=long_strike,
                )
                return ExitSignal(
                    trade_id=trade.id,
                    reason=ExitReason.STRATEGY_SPECIFIC,
                    details=(
                        f"Underlying ${spot_price:.2f} has risen above "
                        f"long put strike ${long_strike:.2f}. "
                        f"Directional thesis invalidated."
                    ),
                    urgency=4,
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
        """Build the two legs of a diagonal spread.

        Requires the following keyword arguments (passed from ``check_entry``):

        - ``right``: Option type (``"C"`` or ``"P"``).
        - ``long_expiry``: Back month expiration date for the long leg.
        - ``short_expiry``: Front month expiration date for the short leg.
        - ``long_option``: The :class:`OptionData` for the long leg.
        - ``short_option``: The :class:`OptionData` for the short leg.

        Args:
            spot_price: Current underlying price.
            options_chain: Full options chain.
            **kwargs: Strategy-specific parameters as described above.

        Returns:
            Two-element list: [long back-month leg, short front-month leg].

        Raises:
            ValueError: If required kwargs are missing.
        """
        right: str = kwargs.get("right", DEFAULT_RIGHT)
        long_expiry: date | None = kwargs.get("long_expiry")
        short_expiry: date | None = kwargs.get("short_expiry")
        long_option: OptionData | None = kwargs.get("long_option")
        short_option: OptionData | None = kwargs.get("short_option")

        if long_expiry is None or short_expiry is None:
            raise ValueError(
                "Diagonal spread requires both long_expiry and short_expiry."
            )
        if long_option is None or short_option is None:
            raise ValueError(
                "Diagonal spread requires both long_option and short_option."
            )

        # Buy long-dated ITM/ATM option
        long_leg = LegSpec(
            action="BUY",
            right=right,
            strike=long_option.strike,
            expiry=long_expiry,
            quantity=1,
            mid_price=long_option.mid_price,
        )

        # Sell short-dated OTM option
        short_leg = LegSpec(
            action="SELL",
            right=right,
            strike=short_option.strike,
            expiry=short_expiry,
            quantity=1,
            mid_price=short_option.mid_price,
        )

        self._log.debug(
            "legs_constructed",
            right=right,
            long_strike=long_option.strike,
            short_strike=short_option.strike,
            long_expiry=long_expiry.isoformat(),
            short_expiry=short_expiry.isoformat(),
            long_mid=long_option.mid_price,
            short_mid=short_option.mid_price,
        )

        return [long_leg, short_leg]

    # ------------------------------------------------------------------
    # P&L calculations
    # ------------------------------------------------------------------

    def calculate_max_profit(
        self,
        legs: list[LegSpec],
        net_premium: float,
    ) -> float:
        """Estimate maximum profit for a diagonal spread.

        The exact maximum profit depends on the remaining time value of the
        long leg at short-leg expiration, which cannot be determined in
        advance.  As a conservative estimate, we use 25% of the net debit.

        The actual maximum occurs when the underlying is at the short strike
        at short-leg expiration and the long leg retains significant value.

        Args:
            legs: The diagonal spread legs.
            net_premium: Net debit paid (positive value).

        Returns:
            Estimated maximum profit in dollars (per spread unit).
        """
        estimated_max_profit = abs(net_premium) * 0.25 * 100.0
        return round(estimated_max_profit, 2)

    def calculate_max_loss(
        self,
        legs: list[LegSpec],
        net_premium: float,
    ) -> float:
        """Calculate maximum loss for a diagonal spread.

        Maximum loss is the full net debit paid, which occurs if the
        underlying moves far against the position and both options lose
        value.

        Args:
            legs: The diagonal spread legs.
            net_premium: Net debit paid (positive value).

        Returns:
            Maximum loss in dollars (positive value, per spread unit).
        """
        return round(abs(net_premium) * 100.0, 2)

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

    def _get_option_right(self) -> str:
        """Determine option right from config or default to calls.

        Returns:
            ``"C"`` or ``"P"``.
        """
        return DEFAULT_RIGHT

    def _get_long_leg_dte(self) -> tuple[int, int, int]:
        """Return (target_dte, min_dte, max_dte) for the long (back) leg.

        Reads from config if available, otherwise uses module constants.

        Returns:
            Tuple of (target, min, max) DTE values.
        """
        target_dte = self._config.target_dte
        dte_range = self._config.dte_range

        if isinstance(target_dte, dict):
            long_target = target_dte.get("back_month", LONG_LEG_TARGET_DTE)
        else:
            long_target = LONG_LEG_TARGET_DTE

        if isinstance(dte_range, dict):
            long_range = dte_range.get(
                "back_month", [LONG_LEG_MIN_DTE, LONG_LEG_MAX_DTE]
            )
        else:
            long_range = [LONG_LEG_MIN_DTE, LONG_LEG_MAX_DTE]

        return long_target, long_range[0], long_range[1]

    def _get_short_leg_dte(self) -> tuple[int, int, int]:
        """Return (target_dte, min_dte, max_dte) for the short (front) leg.

        Reads from config if available, otherwise uses module constants.

        Returns:
            Tuple of (target, min, max) DTE values.
        """
        target_dte = self._config.target_dte
        dte_range = self._config.dte_range

        if isinstance(target_dte, dict):
            short_target = target_dte.get("front_month", SHORT_LEG_TARGET_DTE)
        else:
            short_target = SHORT_LEG_TARGET_DTE

        if isinstance(dte_range, dict):
            short_range = dte_range.get(
                "front_month", [SHORT_LEG_MIN_DTE, SHORT_LEG_MAX_DTE]
            )
        else:
            short_range = [SHORT_LEG_MIN_DTE, SHORT_LEG_MAX_DTE]

        return short_target, short_range[0], short_range[1]

    def _get_long_delta_range(self, right: str) -> tuple[float, float]:
        """Return the delta range for the long leg.

        For calls, higher delta means more ITM.
        For puts, the range is negated.

        Args:
            right: ``"C"`` or ``"P"``.

        Returns:
            Tuple of (min_delta, max_delta) in the sign convention of the
            option type.
        """
        delta_range = self._config.delta_range
        if "long_leg" in delta_range:
            raw = delta_range["long_leg"]
            if isinstance(raw, list) and len(raw) == 2:
                low, high = float(raw[0]), float(raw[1])
            else:
                low, high = LONG_LEG_DELTA_MIN, LONG_LEG_DELTA_MAX
        else:
            low, high = LONG_LEG_DELTA_MIN, LONG_LEG_DELTA_MAX

        if right == "P":
            return -high, -low
        return low, high

    def _get_short_delta_range(self, right: str) -> tuple[float, float]:
        """Return the delta range for the short leg.

        For calls, lower delta means more OTM.
        For puts, the range is negated.

        Args:
            right: ``"C"`` or ``"P"``.

        Returns:
            Tuple of (min_delta, max_delta) in the sign convention of the
            option type.
        """
        delta_range = self._config.delta_range
        if "short_leg" in delta_range:
            raw = delta_range["short_leg"]
            if isinstance(raw, list) and len(raw) == 2:
                low, high = float(raw[0]), float(raw[1])
            else:
                low, high = SHORT_LEG_DELTA_MIN, SHORT_LEG_DELTA_MAX
        else:
            low, high = SHORT_LEG_DELTA_MIN, SHORT_LEG_DELTA_MAX

        if right == "P":
            return -high, -low
        return low, high

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
    def _validate_strike_order(
        right: str,
        long_strike: float,
        short_strike: float,
    ) -> bool:
        """Validate that strikes are ordered correctly for the diagonal.

        For call diagonals: long strike (ITM, lower) < short strike (OTM, higher).
        For put diagonals: long strike (ITM, higher) > short strike (OTM, lower).

        Args:
            right: ``"C"`` or ``"P"``.
            long_strike: Strike of the long (bought) option.
            short_strike: Strike of the short (sold) option.

        Returns:
            ``True`` if the strike ordering is valid.
        """
        if right == "C":
            return long_strike < short_strike
        # For puts, the ITM long leg has a higher strike
        return long_strike > short_strike

    @staticmethod
    def _calculate_net_premium(legs: list[LegSpec]) -> float:
        """Calculate net premium from legs (positive = debit).

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
    def _get_long_strike(trade: TradeRecord) -> float | None:
        """Extract the long leg's strike from the trade record.

        The long leg is the one with action ``BUY``.

        Args:
            trade: The open trade record.

        Returns:
            Strike price of the long leg, or ``None`` if not found.
        """
        for leg in trade.legs:
            if leg.action == "BUY":
                return leg.strike
        return None

    @staticmethod
    def _get_trade_right(trade: TradeRecord) -> str:
        """Extract the option right from the trade's legs.

        All legs in a diagonal share the same right, so we return the
        right of the first leg.

        Args:
            trade: The open trade record.

        Returns:
            ``"C"`` or ``"P"``, or ``"C"`` as default if no legs exist.
        """
        if trade.legs:
            return trade.legs[0].right
        return DEFAULT_RIGHT
