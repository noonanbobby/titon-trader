"""Iron Condor strategy implementation.

An iron condor simultaneously sells an OTM put spread and an OTM call
spread, collecting premium on both sides.  The trade consists of four
legs:

- **Sell** a short put  (delta -0.20 to -0.10)
- **Buy** a long put   (delta -0.08 to -0.02, further OTM)
- **Sell** a short call (delta  0.10 to  0.20)
- **Buy** a long call  (delta  0.02 to  0.08, further OTM)

All four legs share the same expiration.  The total net credit received
is the maximum profit.  The maximum loss is the wider wing width minus
the net credit, multiplied by 100.

This strategy is selected in low-volatility trending or range-bound
regimes when IV Rank is between 25 and 75 (adequate premium without
excessive risk of large moves).

Usage::

    from src.strategies.base import StrategyConfig
    from src.strategies.iron_condor import IronCondor

    config = StrategyConfig(
        enabled=True,
        regimes=["low_vol_trend", "range_bound"],
        min_iv_rank=25,
        max_iv_rank=75,
        delta_range={
            "short_put": [-0.20, -0.10],
            "long_put": [-0.08, -0.02],
            "short_call": [0.10, 0.20],
            "long_call": [0.02, 0.08],
        },
        wing_width=5,
        profit_target_pct=0.50,
        stop_loss_pct=2.00,
        max_positions=2,
    )
    strategy = IronCondor("iron_condor", config)
"""

from __future__ import annotations

import math
from datetime import date, datetime
from typing import Any

from src.strategies.base import (
    DEFAULT_MIN_OPEN_INTEREST,
    BaseStrategy,
    Direction,
    ExitSignal,
    ExitType,
    LegSpec,
    StrategyConfig,
    TradeSignal,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Default delta ranges for the four legs.
_DEFAULT_SHORT_PUT_DELTA_MIN: float = -0.20
_DEFAULT_SHORT_PUT_DELTA_MAX: float = -0.10
_DEFAULT_LONG_PUT_DELTA_MIN: float = -0.08
_DEFAULT_LONG_PUT_DELTA_MAX: float = -0.02

_DEFAULT_SHORT_CALL_DELTA_MIN: float = 0.10
_DEFAULT_SHORT_CALL_DELTA_MAX: float = 0.20
_DEFAULT_LONG_CALL_DELTA_MIN: float = 0.02
_DEFAULT_LONG_CALL_DELTA_MAX: float = 0.08

# Default wing width in dollars.
_DEFAULT_WING_WIDTH: float = 5.0

# Options multiplier (standard US equity options).
_MULTIPLIER: int = 100


class IronCondor(BaseStrategy):
    """Iron condor (combined put credit spread + call credit spread).

    Sells an OTM put spread and an OTM call spread simultaneously,
    collecting premium from both sides.  Profits when the underlying
    remains within the range defined by the two short strikes through
    expiration.

    Args:
        name: Strategy identifier (e.g. ``"iron_condor"``).
        config: Strategy configuration from ``config/strategies.yaml``.
    """

    def __init__(self, name: str, config: StrategyConfig) -> None:
        super().__init__(name, config)

        # Extract delta ranges from config, falling back to defaults.
        delta_cfg = self._config.delta_range

        sp_range = delta_cfg.get(
            "short_put", [_DEFAULT_SHORT_PUT_DELTA_MIN, _DEFAULT_SHORT_PUT_DELTA_MAX]
        )
        lp_range = delta_cfg.get(
            "long_put", [_DEFAULT_LONG_PUT_DELTA_MIN, _DEFAULT_LONG_PUT_DELTA_MAX]
        )
        sc_range = delta_cfg.get(
            "short_call", [_DEFAULT_SHORT_CALL_DELTA_MIN, _DEFAULT_SHORT_CALL_DELTA_MAX]
        )
        lc_range = delta_cfg.get(
            "long_call", [_DEFAULT_LONG_CALL_DELTA_MIN, _DEFAULT_LONG_CALL_DELTA_MAX]
        )

        self._short_put_delta_min: float = (
            float(sp_range[0])
            if isinstance(sp_range, list) and len(sp_range) == 2
            else _DEFAULT_SHORT_PUT_DELTA_MIN
        )
        self._short_put_delta_max: float = (
            float(sp_range[1])
            if isinstance(sp_range, list) and len(sp_range) == 2
            else _DEFAULT_SHORT_PUT_DELTA_MAX
        )
        self._long_put_delta_min: float = (
            float(lp_range[0])
            if isinstance(lp_range, list) and len(lp_range) == 2
            else _DEFAULT_LONG_PUT_DELTA_MIN
        )
        self._long_put_delta_max: float = (
            float(lp_range[1])
            if isinstance(lp_range, list) and len(lp_range) == 2
            else _DEFAULT_LONG_PUT_DELTA_MAX
        )
        self._short_call_delta_min: float = (
            float(sc_range[0])
            if isinstance(sc_range, list) and len(sc_range) == 2
            else _DEFAULT_SHORT_CALL_DELTA_MIN
        )
        self._short_call_delta_max: float = (
            float(sc_range[1])
            if isinstance(sc_range, list) and len(sc_range) == 2
            else _DEFAULT_SHORT_CALL_DELTA_MAX
        )
        self._long_call_delta_min: float = (
            float(lc_range[0])
            if isinstance(lc_range, list) and len(lc_range) == 2
            else _DEFAULT_LONG_CALL_DELTA_MIN
        )
        self._long_call_delta_max: float = (
            float(lc_range[1])
            if isinstance(lc_range, list) and len(lc_range) == 2
            else _DEFAULT_LONG_CALL_DELTA_MAX
        )

        self._wing_width: float = float(self._config.wing_width or _DEFAULT_WING_WIDTH)

    # ------------------------------------------------------------------
    # Entry evaluation
    # ------------------------------------------------------------------

    async def check_entry(
        self,
        ticker: str,
        spot_price: float,
        iv_rank: float,
        regime: str,
        greeks: dict[str, float],
        options_chain: list[dict[str, Any]],
    ) -> TradeSignal | None:
        """Evaluate whether to open an iron condor.

        Steps:
        1. Verify eligibility (regime, IV rank).
        2. Filter for correct DTE range.
        3. Construct all four legs via :meth:`construct_legs`.
        4. Calculate the total net credit.
        5. Calculate max profit and max loss.
        6. Verify the credit-to-risk ratio is at least 20 %.

        Args:
            ticker: Underlying symbol.
            spot_price: Current price of the underlying.
            iv_rank: IV Rank (0--100).
            regime: Current market regime.
            greeks: Aggregate Greeks dict.
            options_chain: List of option dicts.

        Returns:
            A :class:`TradeSignal` if entry criteria are met, or ``None``.
        """
        # Step 1: Eligibility check
        if not self.is_eligible(regime, iv_rank):
            return None

        self._log.info(
            "evaluating_entry",
            ticker=ticker,
            spot_price=spot_price,
            iv_rank=iv_rank,
            regime=regime,
        )

        # Step 2: Filter for correct DTE range
        dte_min, dte_max = self._config.get_primary_dte_range()
        filtered = self.filter_options_by_dte(options_chain, dte_min, dte_max)
        if not filtered:
            self._log.debug(
                "no_options_in_dte_range",
                ticker=ticker,
                dte_min=dte_min,
                dte_max=dte_max,
            )
            return None

        # Step 3: Construct all four legs
        try:
            legs = self.construct_legs(spot_price, filtered)
        except ValueError as exc:
            self._log.debug(
                "leg_construction_failed",
                ticker=ticker,
                error=str(exc),
            )
            return None

        # Step 4: Calculate net credit.
        total_credit: float = 0.0
        total_debit: float = 0.0
        for leg in legs:
            price = self._get_mid_price(filtered, leg.strike, leg.right)
            if price is None:
                self._log.debug(
                    "missing_price_for_leg",
                    ticker=ticker,
                    strike=leg.strike,
                    right=leg.right,
                )
                return None
            if leg.action == "SELL":
                total_credit += price
            else:
                total_debit += price

        net_credit = total_credit - total_debit
        if net_credit <= 0:
            self._log.debug(
                "invalid_credit",
                ticker=ticker,
                net_credit=net_credit,
                total_credit=total_credit,
                total_debit=total_debit,
            )
            return None

        # Net premium is negative for credit trades.
        net_premium = -net_credit

        max_profit = self.calculate_max_profit(legs, net_premium)
        max_loss = self.calculate_max_loss(legs, net_premium)

        if max_loss <= 0:
            self._log.debug("zero_max_loss", ticker=ticker)
            return None

        # Step 6: Verify credit-to-risk ratio.
        rr_ratio = self.calculate_reward_risk_ratio(max_profit, max_loss)
        if rr_ratio < 0.20:
            self._log.debug(
                "insufficient_credit_to_risk",
                ticker=ticker,
                max_profit=max_profit,
                max_loss=max_loss,
                ratio=rr_ratio,
            )
            return None

        # Identify strikes for the reasoning string.
        sp_leg = next(
            (leg for leg in legs if leg.action == "SELL" and leg.right == "P"),
            None,
        )
        lp_leg = next(
            (leg for leg in legs if leg.action == "BUY" and leg.right == "P"),
            None,
        )
        sc_leg = next(
            (leg for leg in legs if leg.action == "SELL" and leg.right == "C"),
            None,
        )
        lc_leg = next(
            (leg for leg in legs if leg.action == "BUY" and leg.right == "C"),
            None,
        )

        sp_strike = sp_leg.strike if sp_leg else "?"
        lp_strike = lp_leg.strike if lp_leg else "?"
        sc_strike = sc_leg.strike if sc_leg else "?"
        lc_strike = lc_leg.strike if lc_leg else "?"

        self._log.info(
            "entry_signal_generated",
            ticker=ticker,
            short_put_strike=sp_strike,
            long_put_strike=lp_strike,
            short_call_strike=sc_strike,
            long_call_strike=lc_strike,
            net_credit=round(net_credit, 4),
            max_profit=round(max_profit, 2),
            max_loss=round(max_loss, 2),
            rr_ratio=rr_ratio,
        )

        return TradeSignal(
            ticker=ticker,
            strategy_name=self.name,
            direction=Direction.SHORT,
            legs=legs,
            max_profit=round(max_profit, 2),
            max_loss=round(max_loss, 2),
            reward_risk_ratio=rr_ratio,
            entry_reasoning=(
                f"Iron condor on {ticker}: "
                f"put side {lp_strike}/{sp_strike}P, "
                f"call side {sc_strike}/{lc_strike}C "
                f"for ${net_credit:.2f} total credit. "
                f"Max profit ${max_profit:.2f}, max loss ${max_loss:.2f}. "
                f"Regime: {regime}, IV Rank: {iv_rank:.1f}."
            ),
        )

    # ------------------------------------------------------------------
    # Exit evaluation
    # ------------------------------------------------------------------

    async def check_exit(
        self,
        trade: dict[str, Any],
        spot_price: float,
        current_pnl: float,
        current_pnl_pct: float,
        dte_remaining: int,
        greeks: dict[str, float],
    ) -> ExitSignal | None:
        """Evaluate whether to close an open iron condor.

        Checks mechanical exits first (profit target at 50 % of credit,
        stop loss at 2x credit, DTE limit), then applies strategy-specific
        logic:

        - Exit if either short strike is breached (underlying trades
          below the short put strike or above the short call strike).

        Args:
            trade: Dict representing the open trade.
            spot_price: Current price of the underlying.
            current_pnl: Unrealised P&L in dollars.
            current_pnl_pct: P&L as fraction of max profit / max loss.
            dte_remaining: Days to expiration of the nearest leg.
            greeks: Current aggregate Greeks for the position.

        Returns:
            An :class:`ExitSignal` if exit criteria are met, or ``None``.
        """
        # Check shared mechanical exit rules first.
        mechanical = self.check_mechanical_exit(
            current_pnl_pct=current_pnl_pct,
            dte_remaining=dte_remaining,
        )
        if mechanical is not None:
            mechanical.trade_id = str(trade.get("id", ""))
            mechanical.current_pnl = current_pnl
            return mechanical

        # Strategy-specific exit: either short strike breached.
        short_put_strike = self._find_leg_strike(trade, action="SELL", right="P")
        short_call_strike = self._find_leg_strike(trade, action="SELL", right="C")

        # Check if underlying broke below the short put strike.
        if short_put_strike is not None and spot_price < short_put_strike:
            self._log.info(
                "exit_short_put_breached",
                trade_id=str(trade.get("id", "")),
                ticker=trade.get("ticker", ""),
                spot_price=spot_price,
                short_put_strike=short_put_strike,
            )
            return ExitSignal(
                trade_id=str(trade.get("id", "")),
                exit_type=ExitType.MECHANICAL,
                current_pnl=current_pnl,
                current_pnl_pct=current_pnl_pct,
                reasoning=(
                    f"Short put strike breached: underlying ${spot_price:.2f} "
                    f"traded below short put strike ${short_put_strike:.2f}"
                ),
            )

        # Check if underlying broke above the short call strike.
        if short_call_strike is not None and spot_price > short_call_strike:
            self._log.info(
                "exit_short_call_breached",
                trade_id=str(trade.get("id", "")),
                ticker=trade.get("ticker", ""),
                spot_price=spot_price,
                short_call_strike=short_call_strike,
            )
            return ExitSignal(
                trade_id=str(trade.get("id", "")),
                exit_type=ExitType.MECHANICAL,
                current_pnl=current_pnl,
                current_pnl_pct=current_pnl_pct,
                reasoning=(
                    f"Short call strike breached: underlying ${spot_price:.2f} "
                    f"traded above short call strike ${short_call_strike:.2f}"
                ),
            )

        return None

    # ------------------------------------------------------------------
    # Leg construction
    # ------------------------------------------------------------------

    def construct_legs(
        self,
        spot_price: float,
        options_chain: list[dict[str, Any]],
        **kwargs: Any,
    ) -> list[LegSpec]:
        """Construct the four legs of an iron condor.

        Searches the options chain for:
        1. A short put  (delta -0.20 to -0.10, OTM)
        2. A long put   (delta -0.08 to -0.02, further OTM)
        3. A short call (delta  0.10 to  0.20, OTM)
        4. A long call  (delta  0.02 to  0.08, further OTM)

        All four legs share the same expiration.  The put side has the
        short put at a higher strike than the long put.  The call side
        has the short call at a lower strike than the long call.

        Args:
            spot_price: Current underlying price.
            options_chain: Options dicts.
            **kwargs: Optional ``wing_width`` override.

        Returns:
            List of four :class:`LegSpec`:
            ``[short_put, long_put, short_call, long_call]``.

        Raises:
            ValueError: If suitable strikes cannot be found for any leg.
        """
        wing_width = kwargs.get("wing_width", self._wing_width)

        # --- Put side ---
        short_put_delta_target = (
            self._short_put_delta_min + self._short_put_delta_max
        ) / 2.0
        short_put = self.find_strike_by_delta(
            options=options_chain,
            target_delta=short_put_delta_target,
            right="P",
            tolerance=abs(self._short_put_delta_max - self._short_put_delta_min) / 2.0,
        )
        if short_put is None:
            raise ValueError(
                f"No short put found with delta near {short_put_delta_target:.2f}"
            )
        if not self._check_liquidity(short_put):
            raise ValueError(
                f"Short put at strike {short_put['strike']} has insufficient liquidity"
            )

        short_put_strike: float = short_put["strike"]
        short_put_expiry: str = self._normalise_expiry(short_put["expiry"])

        # Long put: prefer a strike approximately wing_width below the short.
        target_long_put_strike = short_put_strike - wing_width
        long_put = self._find_option_near_strike_with_delta(
            options=options_chain,
            right="P",
            target_strike=target_long_put_strike,
            delta_target=(self._long_put_delta_min + self._long_put_delta_max) / 2.0,
            delta_tolerance=abs(self._long_put_delta_max - self._long_put_delta_min)
            / 2.0,
        )
        if long_put is None:
            raise ValueError(
                f"No long put found with delta in "
                f"[{self._long_put_delta_min}, {self._long_put_delta_max}] "
                f"near strike {target_long_put_strike}"
            )
        if not self._check_liquidity(long_put):
            raise ValueError(
                f"Long put at strike {long_put['strike']} has insufficient liquidity"
            )

        long_put_strike: float = long_put["strike"]
        long_put_expiry: str = self._normalise_expiry(long_put["expiry"])

        if short_put_strike <= long_put_strike:
            raise ValueError(
                f"Short put strike ({short_put_strike}) must be above "
                f"long put strike ({long_put_strike})"
            )

        # --- Call side ---
        short_call_delta_target = (
            self._short_call_delta_min + self._short_call_delta_max
        ) / 2.0
        short_call = self.find_strike_by_delta(
            options=options_chain,
            target_delta=short_call_delta_target,
            right="C",
            tolerance=(self._short_call_delta_max - self._short_call_delta_min) / 2.0,
        )
        if short_call is None:
            raise ValueError(
                f"No short call found with delta near {short_call_delta_target:.2f}"
            )
        if not self._check_liquidity(short_call):
            raise ValueError(
                f"Short call at strike {short_call['strike']} "
                f"has insufficient liquidity"
            )

        short_call_strike: float = short_call["strike"]
        short_call_expiry: str = self._normalise_expiry(short_call["expiry"])

        # Long call: prefer a strike approximately wing_width above the short.
        target_long_call_strike = short_call_strike + wing_width
        long_call = self._find_option_near_strike_with_delta(
            options=options_chain,
            right="C",
            target_strike=target_long_call_strike,
            delta_target=(self._long_call_delta_min + self._long_call_delta_max) / 2.0,
            delta_tolerance=(self._long_call_delta_max - self._long_call_delta_min)
            / 2.0,
        )
        if long_call is None:
            raise ValueError(
                f"No long call found with delta in "
                f"[{self._long_call_delta_min}, {self._long_call_delta_max}] "
                f"near strike {target_long_call_strike}"
            )
        if not self._check_liquidity(long_call):
            raise ValueError(
                f"Long call at strike {long_call['strike']} has insufficient liquidity"
            )

        long_call_strike: float = long_call["strike"]
        long_call_expiry: str = self._normalise_expiry(long_call["expiry"])

        if short_call_strike >= long_call_strike:
            raise ValueError(
                f"Short call strike ({short_call_strike}) must be below "
                f"long call strike ({long_call_strike})"
            )

        # Warn if short strikes are on the wrong side of spot.
        if short_put_strike >= spot_price:
            self._log.warning(
                "short_put_not_otm",
                short_put_strike=short_put_strike,
                spot_price=spot_price,
            )
        if short_call_strike <= spot_price:
            self._log.warning(
                "short_call_not_otm",
                short_call_strike=short_call_strike,
                spot_price=spot_price,
            )

        self._log.debug(
            "legs_constructed",
            short_put_strike=short_put_strike,
            short_put_delta=short_put.get("delta"),
            long_put_strike=long_put_strike,
            long_put_delta=long_put.get("delta"),
            short_call_strike=short_call_strike,
            short_call_delta=short_call.get("delta"),
            long_call_strike=long_call_strike,
            long_call_delta=long_call.get("delta"),
            put_wing_width=short_put_strike - long_put_strike,
            call_wing_width=long_call_strike - short_call_strike,
        )

        return [
            LegSpec(
                action="SELL",
                right="P",
                strike=short_put_strike,
                expiry=short_put_expiry,
                quantity=1,
                delta=short_put.get("delta"),
            ),
            LegSpec(
                action="BUY",
                right="P",
                strike=long_put_strike,
                expiry=long_put_expiry,
                quantity=1,
                delta=long_put.get("delta"),
            ),
            LegSpec(
                action="SELL",
                right="C",
                strike=short_call_strike,
                expiry=short_call_expiry,
                quantity=1,
                delta=short_call.get("delta"),
            ),
            LegSpec(
                action="BUY",
                right="C",
                strike=long_call_strike,
                expiry=long_call_expiry,
                quantity=1,
                delta=long_call.get("delta"),
            ),
        ]

    # ------------------------------------------------------------------
    # Profit / Loss calculation
    # ------------------------------------------------------------------

    def calculate_max_profit(self, legs: list[LegSpec], net_premium: float) -> float:
        """Calculate maximum profit for an iron condor.

        ``Max profit = total_net_credit x multiplier``.

        Occurs when the underlying remains between the two short strikes
        at expiration (all four options expire worthless or the short legs
        expire with only intrinsic that nets to zero against the credit).

        Args:
            legs: The four legs.
            net_premium: Net premium for the trade.  Negative means credit
                received (normal for this strategy).

        Returns:
            Maximum profit in dollars (positive).
        """
        credit = abs(net_premium)
        max_profit = credit * _MULTIPLIER
        return round(max_profit, 2)

    def calculate_max_loss(self, legs: list[LegSpec], net_premium: float) -> float:
        """Calculate maximum loss for an iron condor.

        ``Max loss = wider_wing_width - net_credit``, times the multiplier.

        The loss is limited to whichever side (put or call) has the wider
        wing.  In practice both sides usually have the same wing width,
        but this method handles asymmetric condors correctly.

        Args:
            legs: The four legs.
            net_premium: Net premium for the trade.  Negative means credit.

        Returns:
            Maximum loss in dollars (positive number).
        """
        credit = abs(net_premium)

        # Calculate wing width for each side.
        put_legs = [leg for leg in legs if leg.right == "P"]
        call_legs = [leg for leg in legs if leg.right == "C"]

        put_wing_width: float = 0.0
        call_wing_width: float = 0.0

        if len(put_legs) == 2:
            put_strikes = sorted([leg.strike for leg in put_legs])
            put_wing_width = put_strikes[1] - put_strikes[0]

        if len(call_legs) == 2:
            call_strikes = sorted([leg.strike for leg in call_legs])
            call_wing_width = call_strikes[1] - call_strikes[0]

        wider_wing = max(put_wing_width, call_wing_width)

        if wider_wing <= 0:
            self._log.warning(
                "invalid_wing_width",
                put_wing_width=put_wing_width,
                call_wing_width=call_wing_width,
            )
            return 0.01  # TradeSignal requires max_loss > 0

        max_loss = (wider_wing - credit) * _MULTIPLIER

        if max_loss < 0:
            self._log.warning(
                "negative_max_loss",
                wider_wing=wider_wing,
                credit=credit,
                calculated=max_loss,
            )
            return 0.01

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
            legs_for_broker.append(
                {
                    "action": leg.action,
                    "expiry": leg.expiry,
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
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _find_leg_strike(
        trade: dict[str, Any],
        action: str,
        right: str,
    ) -> float | None:
        """Find a specific leg strike from a trade dict by action and right.

        Args:
            trade: Trade dict with ``"legs"`` key.
            action: ``"BUY"`` or ``"SELL"``.
            right: ``"C"`` or ``"P"``.

        Returns:
            The matching strike price, or ``None`` if not found.
        """
        legs = trade.get("legs", [])
        for leg in legs:
            if isinstance(leg, dict):
                if leg.get("action") == action and leg.get("right") == right:
                    return leg.get("strike")
            elif (
                hasattr(leg, "action")
                and hasattr(leg, "right")
                and leg.action == action
                and leg.right == right
            ):
                return leg.strike
        return None

    @staticmethod
    def _find_option_near_strike_with_delta(
        options: list[dict[str, Any]],
        right: str,
        target_strike: float,
        delta_target: float,
        delta_tolerance: float,
    ) -> dict[str, Any] | None:
        """Find an option near a target strike within a delta range.

        Searches for options matching *right* whose absolute delta is
        within ``delta_tolerance`` of ``abs(delta_target)``, then picks
        the one whose strike is closest to *target_strike*.

        Args:
            options: Options chain dicts.
            right: ``"C"`` or ``"P"``.
            target_strike: Ideal strike price.
            delta_target: Target delta value.
            delta_tolerance: Acceptable deviation from target delta.

        Returns:
            The best matching option dict, or ``None``.
        """
        abs_delta_target = abs(delta_target)

        candidates: list[tuple[float, dict[str, Any]]] = []
        for opt in options:
            if opt.get("right") != right:
                continue
            oi = opt.get("open_interest", 0)
            if oi < DEFAULT_MIN_OPEN_INTEREST:
                continue
            opt_delta = opt.get("delta")
            if opt_delta is None or (
                isinstance(opt_delta, float) and math.isnan(opt_delta)
            ):
                continue
            if abs(abs(opt_delta) - abs_delta_target) <= delta_tolerance:
                strike_distance = abs(opt["strike"] - target_strike)
                candidates.append((strike_distance, opt))

        if not candidates:
            return None

        candidates.sort(key=lambda x: x[0])
        return candidates[0][1]

    @staticmethod
    def _get_mid_price(
        chain: list[dict[str, Any]],
        strike: float,
        right: str,
    ) -> float | None:
        """Look up the mid-price for a specific strike and right.

        Args:
            chain: Options chain dicts.
            strike: Target strike price.
            right: ``"C"`` or ``"P"``.

        Returns:
            The mid-price, or ``None`` if not found.
        """
        for opt in chain:
            if opt.get("right") == right and opt.get("strike") == strike:
                bid = opt.get("bid", 0.0)
                ask = opt.get("ask", 0.0)
                mid = opt.get("mid_price")
                if mid is not None:
                    return float(mid)
                if bid is not None and ask is not None:
                    return (float(bid) + float(ask)) / 2.0
        return None

    @staticmethod
    def _check_liquidity(opt: dict[str, Any]) -> bool:
        """Verify an option dict has sufficient liquidity.

        Args:
            opt: Option dict.

        Returns:
            ``True`` if the option passes liquidity checks.
        """
        oi = opt.get("open_interest", 0)
        if oi < DEFAULT_MIN_OPEN_INTEREST:
            return False

        bid = opt.get("bid", 0.0) or 0.0
        ask = opt.get("ask", 0.0) or 0.0
        mid = (bid + ask) / 2.0
        if mid <= 0:
            return False

        spread_pct = (ask - bid) / mid
        return spread_pct <= 0.05

    @staticmethod
    def _normalise_expiry(expiry_raw: Any) -> str:
        """Convert an expiry value to ``YYYYMMDD`` string format.

        Args:
            expiry_raw: Expiry as date, datetime, or YYYYMMDD string.

        Returns:
            Expiry in ``YYYYMMDD`` string format.

        Raises:
            ValueError: If the expiry cannot be parsed.
        """
        if isinstance(expiry_raw, datetime):
            return expiry_raw.strftime("%Y%m%d")
        if isinstance(expiry_raw, date):
            return expiry_raw.strftime("%Y%m%d")
        if (
            isinstance(expiry_raw, str)
            and len(expiry_raw) == 8
            and expiry_raw.isdigit()
        ):
            return expiry_raw
        raise ValueError(f"Cannot normalise expiry: {expiry_raw!r}")
