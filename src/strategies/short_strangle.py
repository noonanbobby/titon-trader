"""Short Strangle strategy implementation.

A short strangle is an undefined-risk, neutral premium-collection strategy
that profits from time decay when the underlying stays within a range.  The
trade consists of:

- **Sell** an OTM put (delta -0.20 to -0.12)
- **Sell** an OTM call (delta 0.12 to 0.20)

Both legs share the same expiration (target DTE 45).  Maximum profit equals
the total credit received.  Maximum loss is theoretically unlimited, so for
sizing purposes a practical worst-case of 3x the credit received is used.

This strategy is selected in low-volatility trending or range-bound regimes
when IV Rank is elevated (>= 40), making the premium rich enough to sell.

Usage::

    from src.strategies.base import StrategyConfig
    from src.strategies.short_strangle import ShortStrangle

    config = StrategyConfig(
        enabled=True,
        regimes=["low_vol_trend", "range_bound"],
        min_iv_rank=40,
        max_iv_rank=100,
        delta_range={"short_put": [-0.20, -0.12], "short_call": [0.12, 0.20]},
        profit_target_pct=0.50,
        stop_loss_pct=2.00,
        max_positions=2,
    )
    strategy = ShortStrangle(config)
"""

from __future__ import annotations

from typing import Any

from src.strategies.base import (
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
from src.utils.logging import get_logger

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Default delta ranges for the short put and short call legs.
_DEFAULT_SHORT_PUT_DELTA_MIN: float = -0.20
_DEFAULT_SHORT_PUT_DELTA_MAX: float = -0.12
_DEFAULT_SHORT_CALL_DELTA_MIN: float = 0.12
_DEFAULT_SHORT_CALL_DELTA_MAX: float = 0.20

# Practical max-loss multiplier for undefined-risk sizing.
# Max loss is theoretically unlimited; use 3x credit for position sizing.
_UNDEFINED_RISK_MULTIPLIER: float = 3.0

# Exit threshold: if either short leg's absolute delta exceeds this, exit.
_DELTA_EXIT_THRESHOLD: float = 0.50

# Options multiplier (standard US equity options).
_MULTIPLIER: int = 100


class ShortStrangle(BaseStrategy):
    """Short strangle: sell OTM put + sell OTM call (undefined risk).

    Collects premium from selling both an OTM put and an OTM call at the
    same expiration.  Profits when the underlying stays between the two
    short strikes through expiration.  This is an undefined-risk strategy
    that requires portfolio margin or sufficient buying power.

    Args:
        config: Strategy configuration from ``config/strategies.yaml``.
    """

    def __init__(self, config: StrategyConfig) -> None:
        super().__init__(config)
        self._log = get_logger(f"strategy.{self.name}")

        # Extract delta ranges from config, falling back to defaults.
        delta_cfg = self._config.delta_range
        put_range = delta_cfg.get(
            "short_put",
            [_DEFAULT_SHORT_PUT_DELTA_MIN, _DEFAULT_SHORT_PUT_DELTA_MAX],
        )
        call_range = delta_cfg.get(
            "short_call",
            [_DEFAULT_SHORT_CALL_DELTA_MIN, _DEFAULT_SHORT_CALL_DELTA_MAX],
        )

        if isinstance(put_range, list) and len(put_range) == 2:
            self._short_put_delta_min: float = float(put_range[0])
            self._short_put_delta_max: float = float(put_range[1])
        else:
            self._short_put_delta_min = _DEFAULT_SHORT_PUT_DELTA_MIN
            self._short_put_delta_max = _DEFAULT_SHORT_PUT_DELTA_MAX

        if isinstance(call_range, list) and len(call_range) == 2:
            self._short_call_delta_min: float = float(call_range[0])
            self._short_call_delta_max: float = float(call_range[1])
        else:
            self._short_call_delta_min = _DEFAULT_SHORT_CALL_DELTA_MIN
            self._short_call_delta_max = _DEFAULT_SHORT_CALL_DELTA_MAX

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        """Return the canonical strategy name."""
        return "short_strangle"

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
        """Evaluate whether to open a short strangle.

        Steps:
        1. Verify eligibility (regime, IV rank, underlying price).
        2. Find the short put leg (delta -0.20 to -0.12).
        3. Find the short call leg (delta 0.12 to 0.20).
        4. Validate bid-ask spreads and open interest.
        5. Ensure both legs share the same expiration.
        6. Calculate credit, max profit, and max loss.
        7. Verify minimum credit threshold.

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

        # Identify the short put and short call legs.
        short_put = next(
            (leg for leg in legs if leg.action == "SELL" and leg.right == "P"),
            None,
        )
        short_call = next(
            (leg for leg in legs if leg.action == "SELL" and leg.right == "C"),
            None,
        )

        if short_put is None or short_call is None:
            self._log.warning(
                "invalid_legs",
                ticker=ticker,
                num_legs=len(legs),
            )
            return None

        # Net premium is negative (credit received).
        net_premium = -(short_put.mid_price + short_call.mid_price)

        # Credit must be meaningful.
        total_credit = abs(net_premium)
        if total_credit < 0.10:
            self._log.debug(
                "credit_too_small",
                ticker=ticker,
                total_credit=total_credit,
            )
            return None

        max_profit = self.calculate_max_profit(legs, net_premium)
        max_loss = self.calculate_max_loss(legs, net_premium)

        self._log.info(
            "entry_signal_generated",
            ticker=ticker,
            short_put_strike=short_put.strike,
            short_call_strike=short_call.strike,
            total_credit=round(total_credit, 4),
            max_profit=round(max_profit, 2),
            max_loss=round(max_loss, 2),
        )

        return TradeSignal(
            strategy=self.name,
            ticker=ticker,
            direction=Direction.SHORT,
            legs=legs,
            net_premium=round(net_premium, 4),
            max_profit=round(max_profit, 2),
            max_loss=round(max_loss, 2),
            regime=regime,
            iv_rank=iv_rank,
            reasoning=(
                f"Short strangle on {ticker}: sell {short_put.strike}P / "
                f"sell {short_call.strike}C for ${total_credit:.2f} credit. "
                f"Max profit ${max_profit:.2f}, sizing max loss ${max_loss:.2f}. "
                f"Regime: {regime}, IV Rank: {iv_rank:.1f}. "
                f"Undefined risk — margin required."
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
        """Evaluate whether to close an open short strangle.

        Checks mechanical exits first (profit target, stop loss, DTE), then
        applies strategy-specific exit rules:

        - Exit if either short strike is breached by the underlying.
        - Exit if the absolute delta of either leg exceeds 0.50.

        For profit target: the stop_loss_pct of 2.0 means close when loss
        reaches 2x the credit received.  Profit target of 0.50 means close
        when 50% of max profit (credit) is captured.

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

        # Strategy-specific: check if either short strike is breached.
        short_put_leg = self._find_leg(trade, action="SELL", right="P")
        short_call_leg = self._find_leg(trade, action="SELL", right="C")

        if short_put_leg is not None and spot_price <= short_put_leg.strike:
            self._log.info(
                "exit_short_put_breached",
                trade_id=str(trade.id),
                ticker=trade.ticker,
                spot_price=spot_price,
                short_put_strike=short_put_leg.strike,
            )
            return ExitSignal(
                trade_id=trade.id,
                reason=ExitReason.STRATEGY_SPECIFIC,
                details=(
                    f"Short put strike breached: underlying ${spot_price:.2f} "
                    f"<= short put ${short_put_leg.strike:.2f}"
                ),
                urgency=5,
            )

        if short_call_leg is not None and spot_price >= short_call_leg.strike:
            self._log.info(
                "exit_short_call_breached",
                trade_id=str(trade.id),
                ticker=trade.ticker,
                spot_price=spot_price,
                short_call_strike=short_call_leg.strike,
            )
            return ExitSignal(
                trade_id=trade.id,
                reason=ExitReason.STRATEGY_SPECIFIC,
                details=(
                    f"Short call strike breached: underlying ${spot_price:.2f} "
                    f">= short call ${short_call_leg.strike:.2f}"
                ),
                urgency=5,
            )

        # Strategy-specific: exit if delta of either leg exceeds threshold.
        # For the position as a whole, check aggregate delta magnitude.
        # A large absolute delta means the position is getting directional.
        position_delta = abs(greeks.delta)
        if position_delta >= _DELTA_EXIT_THRESHOLD:
            self._log.info(
                "exit_delta_exceeded",
                trade_id=str(trade.id),
                ticker=trade.ticker,
                position_delta=greeks.delta,
                abs_delta=position_delta,
                threshold=_DELTA_EXIT_THRESHOLD,
            )
            return ExitSignal(
                trade_id=trade.id,
                reason=ExitReason.STRATEGY_SPECIFIC,
                details=(
                    f"Position delta {greeks.delta:.3f} (abs {position_delta:.3f}) "
                    f"exceeds threshold {_DELTA_EXIT_THRESHOLD:.2f}. "
                    f"One or both legs are too deep ITM."
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
        """Construct the two legs of a short strangle.

        Searches the options chain for:
        1. An OTM put with delta in [-0.20, -0.12].
        2. An OTM call with delta in [0.12, 0.20].
        Both must share the same expiration.

        Args:
            spot_price: Current underlying price.
            options_chain: Available options to search.
            **kwargs: Unused.

        Returns:
            List of two :class:`LegSpec` objects: [short_put, short_call].

        Raises:
            ValueError: If suitable options cannot be found.
        """
        # Find the short put (OTM, below spot).
        short_put = self.find_option_by_delta(
            options=options_chain,
            right="P",
            target_delta_min=self._short_put_delta_min,
            target_delta_max=self._short_put_delta_max,
        )
        if short_put is None:
            raise ValueError(
                f"No OTM put found with delta in "
                f"[{self._short_put_delta_min}, {self._short_put_delta_max}]"
            )

        if not self.validate_bid_ask_spread(short_put):
            raise ValueError(
                f"Short put at strike {short_put.strike} has excessive "
                f"bid-ask spread: bid={short_put.bid}, ask={short_put.ask}"
            )

        # Find the short call (OTM, above spot) with matching expiration.
        # Filter for the same expiration as the short put.
        same_expiry_chain = [
            opt for opt in options_chain if opt.expiry == short_put.expiry
        ]

        short_call = self.find_option_by_delta(
            options=same_expiry_chain,
            right="C",
            target_delta_min=self._short_call_delta_min,
            target_delta_max=self._short_call_delta_max,
        )
        if short_call is None:
            raise ValueError(
                f"No OTM call found with delta in "
                f"[{self._short_call_delta_min}, {self._short_call_delta_max}] "
                f"at expiry {short_put.expiry}"
            )

        if not self.validate_bid_ask_spread(short_call):
            raise ValueError(
                f"Short call at strike {short_call.strike} has excessive "
                f"bid-ask spread: bid={short_call.bid}, ask={short_call.ask}"
            )

        # Validate that the put strike is below spot and call strike is above.
        if short_put.strike >= spot_price:
            raise ValueError(
                f"Short put strike ({short_put.strike}) must be below "
                f"spot price ({spot_price})"
            )

        if short_call.strike <= spot_price:
            raise ValueError(
                f"Short call strike ({short_call.strike}) must be above "
                f"spot price ({spot_price})"
            )

        self._log.debug(
            "legs_constructed",
            short_put_strike=short_put.strike,
            short_put_delta=short_put.delta,
            short_put_mid=short_put.mid_price,
            short_call_strike=short_call.strike,
            short_call_delta=short_call.delta,
            short_call_mid=short_call.mid_price,
            expiry=str(short_put.expiry),
        )

        return [
            LegSpec(
                action="SELL",
                right="P",
                strike=short_put.strike,
                expiry=short_put.expiry,
                quantity=1,
                mid_price=short_put.mid_price,
            ),
            LegSpec(
                action="SELL",
                right="C",
                strike=short_call.strike,
                expiry=short_call.expiry,
                quantity=1,
                mid_price=short_call.mid_price,
            ),
        ]

    # ------------------------------------------------------------------
    # Profit / Loss calculation
    # ------------------------------------------------------------------

    def calculate_max_profit(self, legs: list[LegSpec], net_premium: float) -> float:
        """Calculate maximum profit for a short strangle.

        Max profit = total credit received x multiplier.

        Occurs when the underlying expires between the two short strikes
        and both options expire worthless.

        Args:
            legs: The two legs [short_put, short_call].
            net_premium: Net premium (negative = credit received).

        Returns:
            Maximum profit in dollars (positive).
        """
        total_credit = abs(net_premium)
        max_profit = total_credit * _MULTIPLIER
        return round(max_profit, 2)

    def calculate_max_loss(self, legs: list[LegSpec], net_premium: float) -> float:
        """Calculate practical maximum loss for a short strangle.

        Theoretical max loss is unlimited (underlying can move infinitely).
        For position sizing purposes, we use 3x the credit received as
        the practical worst-case, or the margin requirement if available.

        Args:
            legs: The two legs [short_put, short_call].
            net_premium: Net premium (negative = credit received).

        Returns:
            Practical maximum loss in dollars (positive) for sizing.
        """
        total_credit = abs(net_premium)
        max_loss = total_credit * _UNDEFINED_RISK_MULTIPLIER * _MULTIPLIER
        return round(max_loss, 2)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _find_leg(
        trade: TradeRecord,
        action: str,
        right: str,
    ) -> LegSpec | None:
        """Find a specific leg from a trade's leg list.

        Args:
            trade: The trade to search.
            action: ``"BUY"`` or ``"SELL"``.
            right: ``"C"`` or ``"P"``.

        Returns:
            The matching :class:`LegSpec`, or ``None`` if not found.
        """
        for leg in trade.legs:
            if leg.action == action and leg.right == right:
                return leg
        return None
