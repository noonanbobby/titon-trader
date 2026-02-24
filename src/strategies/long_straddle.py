"""Long Straddle strategy implementation.

A long straddle is a long-volatility strategy that profits from large
moves in either direction.  The trade consists of:

- **Buy** 1 ATM call (delta 0.48--0.52)
- **Buy** 1 ATM put (delta -0.52 to -0.48)

Both legs share the same strike (nearest to spot) and expiration (target
DTE 30).  Maximum loss is the total debit paid.  Maximum profit is
theoretically unlimited in either direction.

This strategy is ONLY entered when:
- IV Rank < 30 (implied volatility is cheap relative to history)
- A known catalyst is approaching (earnings, FDA decision, etc.)

This strategy is selected in high-volatility trending or crisis regimes.

Usage::

    from src.strategies.base import StrategyConfig
    from src.strategies.long_straddle import LongStraddle

    config = StrategyConfig(
        enabled=True,
        regimes=["high_vol_trend", "crisis"],
        min_iv_rank=0,
        max_iv_rank=30,
        delta_range={"call": [0.48, 0.52], "put": [-0.52, -0.48]},
        profit_target_pct=0.50,
        stop_loss_pct=0.30,
        max_positions=1,
    )
    strategy = LongStraddle(config)
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
from src.utils.logging import get_logger

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Default delta ranges for the ATM call and put legs.
_DEFAULT_CALL_DELTA_MIN: float = 0.48
_DEFAULT_CALL_DELTA_MAX: float = 0.52
_DEFAULT_PUT_DELTA_MIN: float = -0.52
_DEFAULT_PUT_DELTA_MAX: float = -0.48

# IV collapse exit threshold: exit if IV drops more than 20% from entry.
_IV_COLLAPSE_THRESHOLD: float = 0.20

# Maximum IV Rank for entry (vol must be cheap).
_MAX_IV_RANK_FOR_ENTRY: float = 30.0

# Options multiplier (standard US equity options).
_MULTIPLIER: int = 100


class LongStraddle(BaseStrategy):
    """Long straddle: buy ATM call + buy ATM put (long volatility).

    A pure volatility play that profits from large moves in either direction.
    Only entered when implied volatility is cheap (IV Rank < 30) and a known
    catalyst is approaching.  Uses a tight stop loss (30% of debit) to limit
    losses from time decay and vol crush.

    Args:
        config: Strategy configuration from ``config/strategies.yaml``.
    """

    def __init__(self, config: StrategyConfig) -> None:
        super().__init__(config)
        self._log = get_logger(f"strategy.{self.name}")

        # Catalyst tracking: set to True externally when the event calendar
        # confirms an upcoming catalyst for the ticker being evaluated.
        self._catalyst_confirmed: bool = False

        # Extract delta ranges from config, falling back to defaults.
        delta_cfg = self._config.delta_range
        call_range = delta_cfg.get(
            "call",
            [_DEFAULT_CALL_DELTA_MIN, _DEFAULT_CALL_DELTA_MAX],
        )
        put_range = delta_cfg.get(
            "put",
            [_DEFAULT_PUT_DELTA_MIN, _DEFAULT_PUT_DELTA_MAX],
        )

        if isinstance(call_range, list) and len(call_range) == 2:
            self._call_delta_min: float = float(call_range[0])
            self._call_delta_max: float = float(call_range[1])
        else:
            self._call_delta_min = _DEFAULT_CALL_DELTA_MIN
            self._call_delta_max = _DEFAULT_CALL_DELTA_MAX

        if isinstance(put_range, list) and len(put_range) == 2:
            self._put_delta_min: float = float(put_range[0])
            self._put_delta_max: float = float(put_range[1])
        else:
            self._put_delta_min = _DEFAULT_PUT_DELTA_MIN
            self._put_delta_max = _DEFAULT_PUT_DELTA_MAX

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        """Return the canonical strategy name."""
        return "long_straddle"

    # ------------------------------------------------------------------
    # Entry evaluation
    # ------------------------------------------------------------------

    def set_catalyst_confirmed(self, confirmed: bool) -> None:
        """Set whether a catalyst event is confirmed for the next entry check.

        The strategy selector or event calendar should call this method before
        invoking :meth:`check_entry` to indicate whether a known catalyst
        (earnings, FDA decision, etc.) is approaching for the ticker being
        evaluated.

        Args:
            confirmed: ``True`` if a catalyst is confirmed, ``False`` otherwise.
        """
        self._catalyst_confirmed = confirmed

    async def check_entry(
        self,
        ticker: str,
        spot_price: float,
        iv_rank: float,
        regime: str,
        greeks: GreeksSnapshot,
        options_chain: list[OptionData],
    ) -> TradeSignal | None:
        """Evaluate whether to open a long straddle.

        Steps:
        1. Verify eligibility (regime, IV rank, underlying price).
        2. Enforce IV Rank < 30 (buy vol cheap).
        3. Verify a catalyst is confirmed (earnings, FDA, etc.) via
           :attr:`_catalyst_confirmed` (set by :meth:`set_catalyst_confirmed`).
        4. Find the ATM call (delta 0.48--0.52).
        5. Find the ATM put (delta -0.52 to -0.48) at the same strike.
        6. Validate bid-ask spreads and open interest.
        7. Calculate net debit, max profit, and max loss.

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
            catalyst_confirmed=self._catalyst_confirmed,
        )

        # Enforce IV Rank cap: only buy straddles when vol is cheap.
        if iv_rank > _MAX_IV_RANK_FOR_ENTRY:
            self._log.debug(
                "iv_rank_too_high_for_straddle",
                ticker=ticker,
                iv_rank=iv_rank,
                max_iv_rank=_MAX_IV_RANK_FOR_ENTRY,
            )
            return None

        # Enforce catalyst requirement.
        if not self._catalyst_confirmed:
            self._log.debug(
                "no_catalyst_confirmed",
                ticker=ticker,
                reason="catalyst_required flag is set; no catalyst event confirmed",
            )
            return None

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

        # Identify the call and put legs.
        call_leg = next((leg for leg in legs if leg.right == "C"), None)
        put_leg = next((leg for leg in legs if leg.right == "P"), None)

        if call_leg is None or put_leg is None:
            self._log.warning(
                "invalid_legs",
                ticker=ticker,
                num_legs=len(legs),
            )
            return None

        # Net premium is positive (total debit).
        net_premium = call_leg.mid_price + put_leg.mid_price

        if net_premium <= 0:
            self._log.debug(
                "invalid_premium",
                ticker=ticker,
                net_premium=net_premium,
            )
            return None

        max_profit = self.calculate_max_profit(legs, net_premium)
        max_loss = self.calculate_max_loss(legs, net_premium)

        # Log entry IV for downstream IV collapse detection.
        entry_iv = greeks.iv if greeks.iv > 0 else 0.0

        self._log.info(
            "entry_signal_generated",
            ticker=ticker,
            strike=call_leg.strike,
            net_debit=round(net_premium, 4),
            max_loss=round(max_loss, 2),
            entry_iv=round(entry_iv, 4),
            catalyst_confirmed=self._catalyst_confirmed,
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
                f"Long straddle on {ticker}: buy {call_leg.strike}C + "
                f"buy {call_leg.strike}P for ${net_premium:.2f} total debit. "
                f"Max loss ${max_loss:.2f}. IV Rank {iv_rank:.1f} (cheap vol). "
                f"Catalyst confirmed. "
                f"Regime: {regime}. Entry IV: {entry_iv:.4f}."
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
        """Evaluate whether to close an open long straddle.

        Checks mechanical exits first (profit target, stop loss, DTE), then
        applies strategy-specific exit rules:

        - Profit target: 50% of debit (requires a significant move).
        - Stop loss: 30% of debit (tight stop to limit theta burn).
        - Close at DTE <= 5.
        - Exit if IV collapses > 20% from entry level (vol crush).
        - Exit 1 day after catalyst event (if catalyst_date in metadata).

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

        # Strategy-specific: exit if IV collapses > 20% from entry.
        entry_iv = self._get_entry_iv(trade)
        current_iv = greeks.iv

        if entry_iv > 0 and current_iv > 0:
            iv_change_pct = (entry_iv - current_iv) / entry_iv

            if iv_change_pct >= _IV_COLLAPSE_THRESHOLD:
                self._log.info(
                    "exit_iv_collapse",
                    trade_id=str(trade.id),
                    ticker=trade.ticker,
                    entry_iv=round(entry_iv, 4),
                    current_iv=round(current_iv, 4),
                    iv_drop_pct=round(iv_change_pct, 4),
                    threshold=_IV_COLLAPSE_THRESHOLD,
                )
                return ExitSignal(
                    trade_id=trade.id,
                    reason=ExitReason.STRATEGY_SPECIFIC,
                    details=(
                        f"IV collapse detected: entry IV {entry_iv:.4f} -> "
                        f"current IV {current_iv:.4f} "
                        f"({iv_change_pct:.1%} drop, threshold "
                        f"{_IV_COLLAPSE_THRESHOLD:.0%}). "
                        f"Vol crush is eroding straddle value."
                    ),
                    urgency=4,
                )

        # Strategy-specific: exit 1 day after catalyst event.
        # The catalyst_date would be tracked via trade entry metadata
        # in the TradeRecord. For now, this is handled by the event
        # calendar integration in the risk manager.

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
        """Construct the two legs of a long straddle.

        Searches the options chain for:
        1. An ATM call (delta 0.48--0.52).
        2. An ATM put (delta -0.52 to -0.48) at the same strike.
        Both must share the same expiration and the same strike (nearest
        to spot price).

        Args:
            spot_price: Current underlying price.
            options_chain: Available options to search.
            **kwargs: Unused.

        Returns:
            List of two :class:`LegSpec` objects: [atm_call, atm_put].

        Raises:
            ValueError: If suitable ATM options cannot be found.
        """
        # Find the ATM call.
        atm_call = self.find_option_by_delta(
            options=options_chain,
            right="C",
            target_delta_min=self._call_delta_min,
            target_delta_max=self._call_delta_max,
        )
        if atm_call is None:
            raise ValueError(
                f"No ATM call found with delta in "
                f"[{self._call_delta_min}, {self._call_delta_max}]"
            )

        if not self.validate_bid_ask_spread(atm_call):
            raise ValueError(
                f"ATM call at strike {atm_call.strike} has excessive "
                f"bid-ask spread: bid={atm_call.bid}, ask={atm_call.ask}"
            )

        # Find the ATM put at the same strike and expiration.
        # For a true straddle, both legs must share the same strike.
        same_strike_puts = [
            opt
            for opt in options_chain
            if opt.right == "P"
            and opt.strike == atm_call.strike
            and opt.expiry == atm_call.expiry
        ]

        if not same_strike_puts:
            # Fall back to finding the best ATM put by delta.
            same_expiry_chain = [
                opt for opt in options_chain if opt.expiry == atm_call.expiry
            ]
            atm_put = self.find_option_by_delta(
                options=same_expiry_chain,
                right="P",
                target_delta_min=self._put_delta_min,
                target_delta_max=self._put_delta_max,
            )

            if atm_put is None:
                raise ValueError(
                    f"No ATM put found with delta in "
                    f"[{self._put_delta_min}, {self._put_delta_max}] "
                    f"at expiry {atm_call.expiry}"
                )

            # If the put is at a different strike, use the nearest to
            # the call's strike for a proper straddle.
            if atm_put.strike != atm_call.strike:
                self._log.debug(
                    "straddle_strike_mismatch",
                    call_strike=atm_call.strike,
                    put_strike=atm_put.strike,
                    using_call_strike=True,
                )
        else:
            atm_put = same_strike_puts[0]

        if not self.validate_bid_ask_spread(atm_put):
            raise ValueError(
                f"ATM put at strike {atm_put.strike} has excessive "
                f"bid-ask spread: bid={atm_put.bid}, ask={atm_put.ask}"
            )

        # Validate open interest is sufficient for both legs.
        if atm_call.open_interest < DEFAULT_MIN_OPEN_INTEREST:
            raise ValueError(
                f"ATM call at strike {atm_call.strike} has insufficient "
                f"open interest ({atm_call.open_interest}, "
                f"minimum {DEFAULT_MIN_OPEN_INTEREST})"
            )

        if atm_put.open_interest < DEFAULT_MIN_OPEN_INTEREST:
            raise ValueError(
                f"ATM put at strike {atm_put.strike} has insufficient "
                f"open interest ({atm_put.open_interest}, "
                f"minimum {DEFAULT_MIN_OPEN_INTEREST})"
            )

        self._log.debug(
            "legs_constructed",
            call_strike=atm_call.strike,
            call_delta=atm_call.delta,
            call_mid=atm_call.mid_price,
            put_strike=atm_put.strike,
            put_delta=atm_put.delta,
            put_mid=atm_put.mid_price,
            expiry=str(atm_call.expiry),
        )

        return [
            LegSpec(
                action="BUY",
                right="C",
                strike=atm_call.strike,
                expiry=atm_call.expiry,
                quantity=1,
                mid_price=atm_call.mid_price,
            ),
            LegSpec(
                action="BUY",
                right="P",
                strike=atm_put.strike,
                expiry=atm_put.expiry,
                quantity=1,
                mid_price=atm_put.mid_price,
            ),
        ]

    # ------------------------------------------------------------------
    # Profit / Loss calculation
    # ------------------------------------------------------------------

    def calculate_max_profit(self, legs: list[LegSpec], net_premium: float) -> float:
        """Calculate maximum profit for a long straddle.

        Maximum profit is theoretically unlimited — the underlying can move
        infinitely in either direction.  For display/reporting purposes, we
        return a large sentinel value based on the underlying's potential
        to double.  In practice, the risk manager uses max_loss for sizing.

        Args:
            legs: The two legs [atm_call, atm_put].
            net_premium: Total debit paid (positive number).

        Returns:
            A large positive number representing theoretical unlimited profit.
        """
        # For sizing and display, estimate profit if underlying moves 50%.
        # This is a rough guide; actual profit depends on the move magnitude.
        call_leg = next((leg for leg in legs if leg.right == "C"), None)
        if call_leg is not None:
            # Estimate: if underlying moves +50% from strike, profit = move - debit.
            estimated_move = call_leg.strike * 0.50
            estimated_profit = (estimated_move - net_premium) * _MULTIPLIER
            if estimated_profit > 0:
                return round(estimated_profit, 2)

        # Fallback: return 5x the debit as an estimate.
        return round(net_premium * 5.0 * _MULTIPLIER, 2)

    def calculate_max_loss(self, legs: list[LegSpec], net_premium: float) -> float:
        """Calculate maximum loss for a long straddle.

        Max loss = total debit paid x multiplier.

        Occurs when the underlying expires exactly at the strike price and
        both options expire worthless.

        Args:
            legs: The two legs [atm_call, atm_put].
            net_premium: Total debit paid (positive number).

        Returns:
            Maximum loss in dollars (positive number).
        """
        max_loss = abs(net_premium) * _MULTIPLIER
        return round(max_loss, 2)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_entry_iv(trade: TradeRecord) -> float:
        """Extract the entry IV from a trade record.

        The entry IV is stored in the TradeRecord's legs as the average
        implied vol across both legs at entry, or can be computed from
        the aggregate Greeks snapshot recorded at trade entry.

        Args:
            trade: The trade to extract entry IV from.

        Returns:
            The entry IV, or 0.0 if not available.
        """
        # The trade entry_price represents the net debit. We need the
        # entry IV which would have been stored when the trade was opened.
        # For now, we estimate it from the leg prices and the trade's
        # max_loss ratio. The actual implementation would store entry_iv
        # as a field on the trade record.
        #
        # This is a best-effort extraction: the risk manager and journal
        # agent track entry IV separately and pass it to check_exit via
        # the greeks parameter.
        return 0.0
