"""Ratio Spread strategy implementation.

A ratio spread (1:2 call ratio) is a partially financed directional bet
that profits from a moderate move in the underlying.  The trade consists of:

- **Buy** 1 call at a lower strike (delta 0.55--0.65)
- **Sell** 2 calls at a higher strike (delta 0.25--0.35)

All legs share the same expiration (target DTE 45).  The extra short call
creates undefined upside risk above the upper breakeven, so this strategy
requires margin.

Maximum profit occurs when the underlying expires exactly at the short
strike.  Below the long strike, the trade loses the net debit (or gains
a small credit).  Above the upper breakeven, losses are unlimited.

This strategy is selected in high-volatility trending or range-bound regimes
when IV Rank is elevated (>= 40), making the short premium rich.

Usage::

    from src.strategies.base import StrategyConfig
    from src.strategies.ratio_spread import RatioSpread

    config = StrategyConfig(
        enabled=True,
        regimes=["high_vol_trend", "range_bound"],
        min_iv_rank=40,
        max_iv_rank=100,
        delta_range={"long_leg": [0.55, 0.65], "short_leg": [0.25, 0.35]},
        profit_target_pct=0.50,
        stop_loss_pct=1.50,
        max_positions=1,
    )
    strategy = RatioSpread("ratio_spread", config)
"""

from __future__ import annotations

from typing import Any

from src.strategies.base import (
    DEFAULT_MIN_OPEN_INTEREST,
    BaseStrategy,
    Direction,
    ExitReason,
    ExitSignal,
    GreeksSnapshot,
    LegSpec,
    OptionData,
    StrategyConfig,
    TradeRecord,
    TradeSignal,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Default delta ranges for the long and short legs.
_DEFAULT_LONG_DELTA_MIN: float = 0.55
_DEFAULT_LONG_DELTA_MAX: float = 0.65
_DEFAULT_SHORT_DELTA_MIN: float = 0.25
_DEFAULT_SHORT_DELTA_MAX: float = 0.35

# Ratio: buy 1, sell 2.
_LONG_QUANTITY: int = 1
_SHORT_QUANTITY: int = 2

# Practical max-loss multiplier for undefined upside risk.
# For position sizing, max loss = 3x the max profit.
_UNDEFINED_RISK_MULTIPLIER: float = 3.0

# Options multiplier (standard US equity options).
_MULTIPLIER: int = 100


class RatioSpread(BaseStrategy):
    """Ratio spread: buy 1 call, sell 2 calls at higher strike (1:2 ratio).

    A neutral-to-slightly-bullish strategy that profits when the underlying
    moves moderately higher to the short strike.  The extra naked short call
    creates unlimited upside risk, requiring margin.

    Args:
        config: Strategy configuration from ``config/strategies.yaml``.
    """

    def __init__(self, name: str, config: StrategyConfig) -> None:
        super().__init__(name, config)

        # Extract delta ranges from config, falling back to defaults.
        delta_cfg = self._config.delta_range
        long_range = delta_cfg.get(
            "long_leg",
            [_DEFAULT_LONG_DELTA_MIN, _DEFAULT_LONG_DELTA_MAX],
        )
        short_range = delta_cfg.get(
            "short_leg",
            [_DEFAULT_SHORT_DELTA_MIN, _DEFAULT_SHORT_DELTA_MAX],
        )

        if isinstance(long_range, list) and len(long_range) == 2:
            self._long_delta_min: float = float(long_range[0])
            self._long_delta_max: float = float(long_range[1])
        else:
            self._long_delta_min = _DEFAULT_LONG_DELTA_MIN
            self._long_delta_max = _DEFAULT_LONG_DELTA_MAX

        if isinstance(short_range, list) and len(short_range) == 2:
            self._short_delta_min: float = float(short_range[0])
            self._short_delta_max: float = float(short_range[1])
        else:
            self._short_delta_min = _DEFAULT_SHORT_DELTA_MIN
            self._short_delta_max = _DEFAULT_SHORT_DELTA_MAX

    # ------------------------------------------------------------------
    # Entry evaluation
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
        """Evaluate whether to open a ratio spread.

        Steps:
        1. Verify eligibility (regime, IV rank, underlying price).
        2. Find the long call (delta 0.55--0.65).
        3. Find the short calls (delta 0.25--0.35, quantity 2).
        4. Validate bid-ask spreads and open interest.
        5. Ensure both legs share the same expiration.
        6. Calculate net premium, max profit, and max loss.
        7. Verify the risk/reward profile is acceptable.

        Args:
            ticker: Underlying symbol.
            spot_price: Current price of the underlying.
            iv_rank: IV Rank (0--100).
            regime: Current market regime.
            greeks: Aggregate Greeks snapshot.
            options_chain: Available options with Greeks and prices.

        Returns:
            A :class:`TradeSignal` if entry criteria are met, or ``None``.
        """
        if not self.is_eligible(regime, iv_rank, spot_price):
            return None

        self._log.info(
            "evaluating_entry",
            ticker=ticker,
            spot_price=spot_price,
            iv_rank=iv_rank,
            regime=regime,
        )

        # Construct legs (handles all validation internally).
        try:
            legs = self.construct_legs(spot_price, options_chain)
        except ValueError as exc:
            self._log.debug(
                "leg_construction_failed",
                ticker=ticker,
                error=str(exc),
            )
            return None

        # Identify the long and short legs.
        long_leg = next((leg for leg in legs if leg.action == "BUY"), None)
        short_legs = [leg for leg in legs if leg.action == "SELL"]

        if long_leg is None or not short_legs:
            self._log.warning(
                "invalid_legs",
                ticker=ticker,
                num_legs=len(legs),
            )
            return None

        # Net premium: cost of long leg minus credit from short legs.
        # Positive = net debit, negative = net credit.
        long_cost = long_leg.mid_price * long_leg.quantity
        short_credit = sum(leg.mid_price * leg.quantity for leg in short_legs)
        net_premium = long_cost - short_credit

        max_profit = self.calculate_max_profit(legs, net_premium)
        max_loss = self.calculate_max_loss(legs, net_premium)

        # Verify max profit is reasonable.
        if max_profit <= 0:
            self._log.debug(
                "no_profit_potential",
                ticker=ticker,
                max_profit=max_profit,
            )
            return None

        self._log.info(
            "entry_signal_generated",
            ticker=ticker,
            long_strike=long_leg.strike,
            short_strike=short_legs[0].strike,
            net_premium=round(net_premium, 4),
            max_profit=round(max_profit, 2),
            max_loss=round(max_loss, 2),
            ratio=f"{_LONG_QUANTITY}:{_SHORT_QUANTITY}",
        )

        return TradeSignal(
            strategy=self.name,
            ticker=ticker,
            direction=Direction.LONG,
            legs=legs,
            net_premium=round(net_premium, 4),
            max_profit=round(max_profit, 2),
            max_loss=round(max_loss, 2),
            regime=regime,
            iv_rank=iv_rank,
            reasoning=(
                f"Ratio spread on {ticker}: buy {_LONG_QUANTITY}x "
                f"{long_leg.strike}C / sell {_SHORT_QUANTITY}x "
                f"{short_legs[0].strike}C for "
                f"${abs(net_premium):.2f} {'debit' if net_premium > 0 else 'credit'}. "
                f"Max profit ${max_profit:.2f} at short strike, "
                f"sizing max loss ${max_loss:.2f}. "
                f"Regime: {regime}, IV Rank: {iv_rank:.1f}. "
                f"Undefined upside risk — margin required."
            ),
        )

    # ------------------------------------------------------------------
    # Exit evaluation
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
        """Evaluate whether to close an open ratio spread.

        Checks mechanical exits first (profit target, stop loss, DTE), then
        applies strategy-specific exit rules:

        - Profit target: 50% of max profit.
        - Stop loss: 1.5x max profit.
        - Close at DTE <= 5.
        - Exit if underlying exceeds upper breakeven.

        The upper breakeven = short_strike + max_profit_per_share, where
        max_profit_per_share = (short_strike - long_strike - net_debit).

        Args:
            trade: The open trade to evaluate.
            spot_price: Current price of the underlying.
            current_pnl: Unrealised P&L in dollars.
            current_pnl_pct: Unrealised P&L as fraction of max values.
            dte_remaining: Days to expiration of the nearest leg.
            greeks: Current Greeks of the position.

        Returns:
            An :class:`ExitSignal` if exit criteria are met, or ``None``.
        """
        # Check shared mechanical exit rules first.
        mechanical = self.check_mechanical_exit(
            trade=trade,
            current_pnl=current_pnl,
            current_pnl_pct=current_pnl_pct,
            dte_remaining=dte_remaining,
        )
        if mechanical is not None:
            return mechanical

        # Strategy-specific: exit if underlying exceeds upper breakeven.
        long_leg = next((leg for leg in trade.legs if leg.action == "BUY"), None)
        short_leg = next((leg for leg in trade.legs if leg.action == "SELL"), None)

        if long_leg is not None and short_leg is not None:
            # Calculate the max profit per share for breakeven computation.
            spread_width = short_leg.strike - long_leg.strike
            net_debit_per_share = trade.entry_price  # net premium at entry

            # Max profit per share at the short strike.
            max_profit_per_share = spread_width - net_debit_per_share

            if max_profit_per_share > 0:
                upper_breakeven = short_leg.strike + max_profit_per_share

                if spot_price > upper_breakeven:
                    self._log.info(
                        "exit_upper_breakeven_breached",
                        trade_id=str(trade.id),
                        ticker=trade.ticker,
                        spot_price=spot_price,
                        upper_breakeven=round(upper_breakeven, 2),
                        short_strike=short_leg.strike,
                    )
                    return ExitSignal(
                        trade_id=trade.id,
                        reason=ExitReason.STRATEGY_SPECIFIC,
                        details=(
                            f"Underlying ${spot_price:.2f} exceeds upper breakeven "
                            f"${upper_breakeven:.2f} (short strike "
                            f"${short_leg.strike:.2f} + max profit/share "
                            f"${max_profit_per_share:.2f}). Naked short call "
                            f"generating losses."
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
        """Construct the legs of a 1:2 ratio spread.

        Searches the options chain for:
        1. A long call with delta in [0.55, 0.65] (1 contract).
        2. A short call with delta in [0.25, 0.35] at a higher strike
           (2 contracts).
        Both must share the same expiration.

        Args:
            spot_price: Current underlying price.
            options_chain: Available options to search.
            **kwargs: Unused.

        Returns:
            List of two :class:`LegSpec` objects: [long_call, short_call].
            The short call has quantity=2.

        Raises:
            ValueError: If suitable options cannot be found.
        """
        # Find the long call (higher delta, closer to ITM).
        long_call = self.find_option_by_delta(
            options=options_chain,
            right="C",
            target_delta_min=self._long_delta_min,
            target_delta_max=self._long_delta_max,
        )
        if long_call is None:
            raise ValueError(
                f"No call found with delta in "
                f"[{self._long_delta_min}, {self._long_delta_max}]"
            )

        if not self.validate_bid_ask_spread(long_call):
            raise ValueError(
                f"Long call at strike {long_call.strike} has excessive "
                f"bid-ask spread: bid={long_call.bid}, ask={long_call.ask}"
            )

        # Find the short call (lower delta, further OTM) with matching expiry.
        same_expiry_chain = [
            opt for opt in options_chain if opt.expiry == long_call.expiry
        ]

        short_call = self.find_option_by_delta(
            options=same_expiry_chain,
            right="C",
            target_delta_min=self._short_delta_min,
            target_delta_max=self._short_delta_max,
        )
        if short_call is None:
            raise ValueError(
                f"No call found with delta in "
                f"[{self._short_delta_min}, {self._short_delta_max}] "
                f"at expiry {long_call.expiry}"
            )

        if not self.validate_bid_ask_spread(short_call):
            raise ValueError(
                f"Short call at strike {short_call.strike} has excessive "
                f"bid-ask spread: bid={short_call.bid}, ask={short_call.ask}"
            )

        # Validate: short strike must be above long strike.
        if short_call.strike <= long_call.strike:
            raise ValueError(
                f"Short call strike ({short_call.strike}) must be above "
                f"long call strike ({long_call.strike})"
            )

        # Validate: short strike must have sufficient OI for 2 contracts.
        if short_call.open_interest < DEFAULT_MIN_OPEN_INTEREST * _SHORT_QUANTITY:
            raise ValueError(
                f"Short call at strike {short_call.strike} has insufficient "
                f"open interest ({short_call.open_interest}) for "
                f"{_SHORT_QUANTITY} contracts (minimum "
                f"{DEFAULT_MIN_OPEN_INTEREST * _SHORT_QUANTITY})"
            )

        self._log.debug(
            "legs_constructed",
            long_strike=long_call.strike,
            long_delta=long_call.delta,
            long_mid=long_call.mid_price,
            short_strike=short_call.strike,
            short_delta=short_call.delta,
            short_mid=short_call.mid_price,
            ratio=f"{_LONG_QUANTITY}:{_SHORT_QUANTITY}",
            expiry=str(long_call.expiry),
        )

        return [
            LegSpec(
                action="BUY",
                right="C",
                strike=long_call.strike,
                expiry=long_call.expiry,
                quantity=_LONG_QUANTITY,
                mid_price=long_call.mid_price,
            ),
            LegSpec(
                action="SELL",
                right="C",
                strike=short_call.strike,
                expiry=short_call.expiry,
                quantity=_SHORT_QUANTITY,
                mid_price=short_call.mid_price,
            ),
        ]

    # ------------------------------------------------------------------
    # Profit / Loss calculation
    # ------------------------------------------------------------------

    def calculate_max_profit(self, legs: list[LegSpec], net_premium: float) -> float:
        """Calculate maximum profit for a ratio spread.

        Max profit = (short_strike - long_strike - net_debit) x multiplier.

        This occurs when the underlying expires exactly at the short strike,
        where the long call has maximum intrinsic value and the short calls
        expire worthless.

        Args:
            legs: The legs [long_call (qty 1), short_call (qty 2)].
            net_premium: Net premium (positive = debit, negative = credit).

        Returns:
            Maximum profit in dollars (positive).
        """
        long_leg = next((leg for leg in legs if leg.action == "BUY"), None)
        short_leg = next((leg for leg in legs if leg.action == "SELL"), None)

        if long_leg is None or short_leg is None:
            return 0.0

        spread_width = short_leg.strike - long_leg.strike
        max_profit_per_share = spread_width - net_premium
        max_profit = max_profit_per_share * _MULTIPLIER

        if max_profit < 0:
            self._log.warning(
                "negative_max_profit",
                spread_width=spread_width,
                net_premium=net_premium,
                calculated=max_profit,
            )
            return 0.0

        return round(max_profit, 2)

    def calculate_max_loss(self, legs: list[LegSpec], net_premium: float) -> float:
        """Calculate practical maximum loss for a ratio spread.

        Theoretical max loss is unlimited above the upper breakeven (naked
        short call exposure).  For position sizing purposes, we use 3x the
        max profit as a practical worst-case estimate.

        Below the long strike, max loss is the net debit (if any).

        Args:
            legs: The legs [long_call (qty 1), short_call (qty 2)].
            net_premium: Net premium (positive = debit, negative = credit).

        Returns:
            Practical maximum loss in dollars (positive) for sizing.
        """
        max_profit = self.calculate_max_profit(legs, net_premium)

        # Practical max loss = 3x the max profit (for sizing).
        upside_loss = max_profit * _UNDEFINED_RISK_MULTIPLIER

        # Downside loss = net debit (if any) x multiplier.
        downside_loss = max(0.0, net_premium) * _MULTIPLIER

        # Use the larger of the two as the sizing max loss.
        max_loss = max(upside_loss, downside_loss)

        return round(max_loss, 2)

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
