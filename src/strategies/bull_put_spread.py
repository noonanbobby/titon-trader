"""Bull Put Spread strategy implementation.

A bull put spread is a vertical credit spread that profits when the
underlying stays above the short put strike or moves higher.  The trade
consists of:

- **Sell** a higher-strike put (delta -0.35 to -0.20, closer to ATM)
- **Buy** a lower-strike put (delta -0.15 to -0.05, further OTM)

Both legs share the same expiration.  The net credit received is the
maximum profit, and the maximum loss is
``(strike_width - net_credit) x 100``.

This strategy is selected in low-volatility trending or range-bound
regimes when IV Rank is above 30 (richer premiums to sell).

Usage::

    from src.strategies.base import StrategyConfig
    from src.strategies.bull_put_spread import BullPutSpread

    config = StrategyConfig(
        enabled=True,
        regimes=["low_vol_trend", "range_bound"],
        min_iv_rank=30,
        max_iv_rank=100,
        delta_range={"short_leg": [-0.35, -0.20], "long_leg": [-0.15, -0.05]},
        wing_width=5,
        profit_target_pct=0.50,
        stop_loss_pct=2.00,
        max_positions=3,
    )
    strategy = BullPutSpread("bull_put_spread", config)
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

# Default delta ranges for put credit spread legs.
_DEFAULT_SHORT_DELTA_MIN: float = -0.35
_DEFAULT_SHORT_DELTA_MAX: float = -0.20
_DEFAULT_LONG_DELTA_MIN: float = -0.15
_DEFAULT_LONG_DELTA_MAX: float = -0.05

# Default wing width in dollars.
_DEFAULT_WING_WIDTH: float = 5.0

# Options multiplier (standard US equity options).
_MULTIPLIER: int = 100


class BullPutSpread(BaseStrategy):
    """Bull put spread (vertical credit spread).

    Sells a higher-strike put and buys a lower-strike put at the same
    expiration.  Profits from the underlying remaining above the short put
    strike through expiration, collecting the net credit as income.

    Args:
        name: Strategy identifier (e.g. ``"bull_put_spread"``).
        config: Strategy configuration from ``config/strategies.yaml``.
    """

    def __init__(self, name: str, config: StrategyConfig) -> None:
        super().__init__(name, config)

        # Extract delta ranges from config, falling back to defaults.
        delta_cfg = self._config.delta_range
        short_range = delta_cfg.get(
            "short_leg", [_DEFAULT_SHORT_DELTA_MIN, _DEFAULT_SHORT_DELTA_MAX]
        )
        long_range = delta_cfg.get(
            "long_leg", [_DEFAULT_LONG_DELTA_MIN, _DEFAULT_LONG_DELTA_MAX]
        )

        if isinstance(short_range, list) and len(short_range) == 2:
            self._short_delta_min: float = float(short_range[0])
            self._short_delta_max: float = float(short_range[1])
        else:
            self._short_delta_min = _DEFAULT_SHORT_DELTA_MIN
            self._short_delta_max = _DEFAULT_SHORT_DELTA_MAX

        if isinstance(long_range, list) and len(long_range) == 2:
            self._long_delta_min: float = float(long_range[0])
            self._long_delta_max: float = float(long_range[1])
        else:
            self._long_delta_min = _DEFAULT_LONG_DELTA_MIN
            self._long_delta_max = _DEFAULT_LONG_DELTA_MAX

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
        """Evaluate whether to open a bull put spread.

        Steps:
        1. Verify eligibility (regime, IV rank).
        2. Filter the options chain for puts within the target DTE window.
        3. Find the short put (delta -0.35 to -0.20) and long put
           (delta -0.15 to -0.05).
        4. Validate bid-ask spreads and open interest.
        5. Construct legs and calculate max profit / max loss.
        6. Verify the credit is meaningful relative to risk (>= 20 %).

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

        # Step 3: Find legs via construct_legs
        try:
            legs = self.construct_legs(spot_price, filtered)
        except ValueError as exc:
            self._log.debug(
                "leg_construction_failed",
                ticker=ticker,
                error=str(exc),
            )
            return None

        # Identify the short and long legs for premium calculation.
        short_leg = next((leg for leg in legs if leg.action == "SELL"), None)
        long_leg = next((leg for leg in legs if leg.action == "BUY"), None)

        if short_leg is None or long_leg is None:
            self._log.warning("invalid_legs", ticker=ticker, num_legs=len(legs))
            return None

        # Step 5: Calculate net credit.
        short_price = self._get_mid_price(filtered, short_leg.strike, "P")
        long_price = self._get_mid_price(filtered, long_leg.strike, "P")

        if short_price is None or long_price is None:
            self._log.debug("missing_prices", ticker=ticker)
            return None

        net_credit = short_price - long_price
        if net_credit <= 0:
            self._log.debug(
                "invalid_credit",
                ticker=ticker,
                net_credit=net_credit,
                short_price=short_price,
                long_price=long_price,
            )
            return None

        # For a credit spread, net_premium is negative (credit received).
        net_premium = -net_credit

        max_profit = self.calculate_max_profit(legs, net_premium)
        max_loss = self.calculate_max_loss(legs, net_premium)

        if max_loss <= 0:
            self._log.debug("zero_max_loss", ticker=ticker)
            return None

        # Step 6: Verify the credit is meaningful relative to risk.
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

        self._log.info(
            "entry_signal_generated",
            ticker=ticker,
            short_strike=short_leg.strike,
            long_strike=long_leg.strike,
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
                f"Bull put spread on {ticker}: sell {short_leg.strike}P / "
                f"buy {long_leg.strike}P for ${net_credit:.2f} credit. "
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
        """Evaluate whether to close an open bull put spread.

        Checks mechanical exits first (profit target at 50 % of credit,
        stop loss at 2x credit, DTE limit), then applies strategy-specific
        logic:

        - Exit if the underlying breaks below the long put strike (max
          loss zone -- no further benefit from holding).

        Args:
            trade: Dict representing the open trade.
            spot_price: Current price of the underlying.
            current_pnl: Unrealised P&L in dollars.
            current_pnl_pct: P&L as fraction of max profit (positive) or
                max loss (negative).
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

        # Strategy-specific exit: underlying below long put strike.
        long_strike = self._find_long_strike(trade)
        if long_strike is not None and spot_price < long_strike:
            self._log.info(
                "exit_underlying_below_long_strike",
                trade_id=str(trade.get("id", "")),
                ticker=trade.get("ticker", ""),
                spot_price=spot_price,
                long_strike=long_strike,
            )
            return ExitSignal(
                trade_id=str(trade.get("id", "")),
                exit_type=ExitType.MECHANICAL,
                current_pnl=current_pnl,
                current_pnl_pct=current_pnl_pct,
                reasoning=(
                    f"Underlying ${spot_price:.2f} broke below long put "
                    f"strike ${long_strike:.2f} -- at max loss zone, "
                    f"no further benefit from holding"
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
        """Construct the two legs of a bull put spread.

        Searches the options chain for:
        1. A short put with delta in the configured range
           (-0.35 to -0.20).
        2. A long put with delta in the configured range
           (-0.15 to -0.05), preferably ``wing_width`` below the short.

        Both must be puts with the same expiration.  The short put has the
        higher strike (closer to ATM); the long put has the lower strike
        (further OTM).

        Args:
            spot_price: Current underlying price.
            options_chain: Options dicts.
            **kwargs: Optional ``wing_width`` override.

        Returns:
            List of two :class:`LegSpec`: ``[short_put, long_put]``.

        Raises:
            ValueError: If suitable strikes cannot be found.
        """
        wing_width = kwargs.get("wing_width", self._wing_width)

        # Find the short put (higher absolute delta, higher strike).
        # For puts, delta is negative.  find_strike_by_delta uses abs values.
        short_delta_target = (self._short_delta_min + self._short_delta_max) / 2.0
        short_put = self.find_strike_by_delta(
            options=options_chain,
            target_delta=short_delta_target,
            right="P",
            tolerance=abs(self._short_delta_max - self._short_delta_min) / 2.0,
        )
        if short_put is None:
            raise ValueError(
                f"No put found with delta near {short_delta_target:.2f} "
                f"(range [{self._short_delta_min}, {self._short_delta_max}])"
            )

        if not self._check_liquidity(short_put):
            raise ValueError(
                f"Short put at strike {short_put['strike']} has insufficient liquidity"
            )

        short_strike: float = short_put["strike"]
        short_expiry: str = self._normalise_expiry(short_put["expiry"])

        # Find the long put (lower absolute delta, lower strike, further OTM).
        target_long_strike = short_strike - wing_width
        long_delta_target = (self._long_delta_min + self._long_delta_max) / 2.0

        long_candidates: list[tuple[float, dict[str, Any]]] = []
        for opt in options_chain:
            if opt.get("right") != "P":
                continue
            opt_delta = opt.get("delta")
            if opt_delta is None or (
                isinstance(opt_delta, float) and math.isnan(opt_delta)
            ):
                continue
            abs_delta = abs(opt_delta)
            abs_target = abs(long_delta_target)
            delta_tol = abs(self._long_delta_max - self._long_delta_min) / 2.0
            if abs(abs_delta - abs_target) <= delta_tol:
                strike_distance = abs(opt["strike"] - target_long_strike)
                long_candidates.append((strike_distance, opt))

        if not long_candidates:
            raise ValueError(
                f"No put found with delta near {long_delta_target:.2f} "
                f"(range [{self._long_delta_min}, {self._long_delta_max}])"
            )

        long_candidates.sort(key=lambda x: x[0])
        long_put = long_candidates[0][1]

        if not self._check_liquidity(long_put):
            raise ValueError(
                f"Long put at strike {long_put['strike']} has insufficient liquidity"
            )

        long_strike: float = long_put["strike"]
        long_expiry: str = self._normalise_expiry(long_put["expiry"])

        # Ensure short strike is above long strike.
        if short_strike <= long_strike:
            raise ValueError(
                f"Short put strike ({short_strike}) must be above "
                f"long put strike ({long_strike})"
            )

        self._log.debug(
            "legs_constructed",
            short_strike=short_strike,
            short_delta=short_put.get("delta"),
            long_strike=long_strike,
            long_delta=long_put.get("delta"),
            actual_width=short_strike - long_strike,
        )

        return [
            LegSpec(
                action="SELL",
                right="P",
                strike=short_strike,
                expiry=short_expiry,
                quantity=1,
                delta=short_put.get("delta"),
            ),
            LegSpec(
                action="BUY",
                right="P",
                strike=long_strike,
                expiry=long_expiry,
                quantity=1,
                delta=long_put.get("delta"),
            ),
        ]

    # ------------------------------------------------------------------
    # Profit / Loss calculation
    # ------------------------------------------------------------------

    def calculate_max_profit(self, legs: list[LegSpec], net_premium: float) -> float:
        """Calculate maximum profit for a bull put spread.

        ``Max profit = net_credit x multiplier``.

        Occurs when the underlying is at or above the short put strike
        at expiration (both puts expire worthless).

        Args:
            legs: The two legs ``[short_put, long_put]``.
            net_premium: Net premium for the spread.  Negative means credit
                received (normal for this strategy).

        Returns:
            Maximum profit in dollars (positive).
        """
        credit = abs(net_premium)
        max_profit = credit * _MULTIPLIER
        return round(max_profit, 2)

    def calculate_max_loss(self, legs: list[LegSpec], net_premium: float) -> float:
        """Calculate maximum loss for a bull put spread.

        ``Max loss = (strike_width - net_credit) x multiplier``.

        Occurs when the underlying is at or below the long put strike
        at expiration.

        Args:
            legs: The two legs ``[short_put, long_put]``.
            net_premium: Net premium for the spread.  Negative means credit
                received.

        Returns:
            Maximum loss in dollars (positive number).
        """
        strikes = sorted([leg.strike for leg in legs])
        strike_width = strikes[-1] - strikes[0]
        credit = abs(net_premium)
        max_loss = (strike_width - credit) * _MULTIPLIER

        if max_loss < 0:
            self._log.warning(
                "negative_max_loss",
                strike_width=strike_width,
                credit=credit,
                calculated=max_loss,
            )
            return 0.01  # TradeSignal requires max_loss > 0

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
    def _find_long_strike(trade: dict[str, Any]) -> float | None:
        """Find the long (bought) put strike from a trade dict.

        Args:
            trade: Trade dict with ``"legs"`` key.

        Returns:
            The long strike price, or ``None`` if not found.
        """
        legs = trade.get("legs", [])
        for leg in legs:
            if isinstance(leg, dict):
                if leg.get("action") == "BUY":
                    return leg.get("strike")
            elif hasattr(leg, "action") and leg.action == "BUY":
                return leg.strike
        return None

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
