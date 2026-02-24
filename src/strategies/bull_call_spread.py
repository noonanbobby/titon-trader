"""Bull Call Spread strategy implementation.

A bull call spread is a vertical debit spread that profits when the
underlying moves moderately higher.  The trade consists of:

- **Buy** a lower-strike call (delta 0.55--0.70, deeper ITM)
- **Sell** a higher-strike call (delta 0.30--0.45, further OTM)

Both legs share the same expiration.  The net debit paid is the maximum
risk, and the maximum profit is ``(strike_width - net_debit) x 100``.

This strategy is selected in low-volatility trending or high-volatility
trending regimes when IV Rank is below 50 (cheaper to buy options).

Usage::

    from src.strategies.base import StrategyConfig
    from src.strategies.bull_call_spread import BullCallSpread

    config = StrategyConfig(
        enabled=True,
        regimes=["low_vol_trend", "high_vol_trend"],
        min_iv_rank=0,
        max_iv_rank=50,
        delta_range={"long_leg": [0.55, 0.70], "short_leg": [0.30, 0.45]},
        wing_width=5,
        profit_target_pct=0.50,
        stop_loss_pct=1.00,
        max_positions=3,
    )
    strategy = BullCallSpread("bull_call_spread", config)
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

# Default delta ranges if not provided in configuration.
_DEFAULT_LONG_DELTA_MIN: float = 0.55
_DEFAULT_LONG_DELTA_MAX: float = 0.70
_DEFAULT_SHORT_DELTA_MIN: float = 0.30
_DEFAULT_SHORT_DELTA_MAX: float = 0.45

# Default wing width in dollars.
_DEFAULT_WING_WIDTH: float = 5.0

# Strategy-specific exit: underlying below short strike by this fraction.
_UNDERLYING_BELOW_SHORT_STRIKE_THRESHOLD: float = 0.02

# Options multiplier (standard US equity options).
_MULTIPLIER: int = 100


class BullCallSpread(BaseStrategy):
    """Bull call spread (vertical debit spread).

    Buys a lower-strike call and sells a higher-strike call at the same
    expiration.  Profits from a moderate upward move in the underlying.

    Args:
        name: Strategy identifier (e.g. ``"bull_call_spread"``).
        config: Strategy configuration from ``config/strategies.yaml``.
    """

    def __init__(self, name: str, config: StrategyConfig) -> None:
        super().__init__(name, config)

        # Extract delta ranges from config, falling back to defaults.
        delta_cfg = self._config.delta_range
        long_range = delta_cfg.get(
            "long_leg", [_DEFAULT_LONG_DELTA_MIN, _DEFAULT_LONG_DELTA_MAX]
        )
        short_range = delta_cfg.get(
            "short_leg", [_DEFAULT_SHORT_DELTA_MIN, _DEFAULT_SHORT_DELTA_MAX]
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
        """Evaluate whether to open a bull call spread.

        Steps:
        1. Verify eligibility (regime, IV rank).
        2. Filter the options chain for calls within the target DTE window.
        3. Find the long call (delta 0.55--0.70) and short call
           (delta 0.30--0.45).
        4. Validate bid-ask spreads and open interest.
        5. Construct legs and calculate max profit / max loss.
        6. Verify the reward-to-risk ratio is acceptable (>= 1.0).

        Args:
            ticker: Underlying symbol.
            spot_price: Current price of the underlying.
            iv_rank: IV Rank (0--100).
            regime: Current market regime.
            greeks: Aggregate Greeks dict.
            options_chain: List of option dicts with ``"strike"``,
                ``"expiry"``, ``"right"``, ``"bid"``, ``"ask"``,
                ``"delta"``, ``"open_interest"`` etc.

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

        # Identify the long and short leg for premium calculation.
        long_leg = next((leg for leg in legs if leg.action == "BUY"), None)
        short_leg = next((leg for leg in legs if leg.action == "SELL"), None)

        if long_leg is None or short_leg is None:
            self._log.warning("invalid_legs", ticker=ticker, num_legs=len(legs))
            return None

        # Step 5: Calculate net premium (debit = positive).
        # Look up prices from the original filtered chain for each leg.
        long_price = self._get_mid_price(filtered, long_leg.strike, "C")
        short_price = self._get_mid_price(filtered, short_leg.strike, "C")

        if long_price is None or short_price is None:
            self._log.debug("missing_prices", ticker=ticker)
            return None

        net_premium = long_price - short_price
        if net_premium <= 0:
            self._log.debug(
                "invalid_premium",
                ticker=ticker,
                net_premium=net_premium,
                long_price=long_price,
                short_price=short_price,
            )
            return None

        max_profit = self.calculate_max_profit(legs, net_premium)
        max_loss = self.calculate_max_loss(legs, net_premium)

        if max_loss <= 0:
            self._log.debug("zero_max_loss", ticker=ticker)
            return None

        # Step 6: Verify reward-to-risk ratio (minimum 1:1).
        rr_ratio = self.calculate_reward_risk_ratio(max_profit, max_loss)
        if rr_ratio < 1.0:
            self._log.debug(
                "insufficient_reward_risk",
                ticker=ticker,
                max_profit=max_profit,
                max_loss=max_loss,
                ratio=rr_ratio,
            )
            return None

        self._log.info(
            "entry_signal_generated",
            ticker=ticker,
            long_strike=long_leg.strike,
            short_strike=short_leg.strike,
            net_premium=round(net_premium, 4),
            max_profit=round(max_profit, 2),
            max_loss=round(max_loss, 2),
            rr_ratio=rr_ratio,
        )

        return TradeSignal(
            ticker=ticker,
            strategy_name=self.name,
            direction=Direction.LONG,
            legs=legs,
            max_profit=round(max_profit, 2),
            max_loss=round(max_loss, 2),
            reward_risk_ratio=rr_ratio,
            entry_reasoning=(
                f"Bull call spread on {ticker}: buy {long_leg.strike}C / "
                f"sell {short_leg.strike}C for ${net_premium:.2f} debit. "
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
        """Evaluate whether to close an open bull call spread.

        Checks mechanical exits first (profit target, stop loss, DTE), then
        applies strategy-specific logic:

        - Exit if the underlying drops below the short strike by more
          than 2 %.

        Args:
            trade: Dict representing the open trade.  Expected keys:
                ``"id"``, ``"ticker"``, ``"max_profit"``, ``"max_loss"``,
                ``"legs"`` (list of leg dicts with ``"action"`` and
                ``"strike"``).
            spot_price: Current price of the underlying.
            current_pnl: Unrealised P&L in dollars.
            current_pnl_pct: P&L as a fraction of max profit (positive) or
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
            # Populate trade-specific fields.
            mechanical.trade_id = str(trade.get("id", ""))
            mechanical.current_pnl = current_pnl
            return mechanical

        # Strategy-specific exit: underlying dropped below short strike by > 2%.
        short_strike = self._find_short_strike(trade)
        if short_strike is not None:
            threshold_price = short_strike * (
                1.0 - _UNDERLYING_BELOW_SHORT_STRIKE_THRESHOLD
            )
            if spot_price < threshold_price:
                self._log.info(
                    "exit_underlying_below_short_strike",
                    trade_id=str(trade.get("id", "")),
                    ticker=trade.get("ticker", ""),
                    spot_price=spot_price,
                    short_strike=short_strike,
                    threshold=round(threshold_price, 2),
                )
                return ExitSignal(
                    trade_id=str(trade.get("id", "")),
                    exit_type=ExitType.MECHANICAL,
                    current_pnl=current_pnl,
                    current_pnl_pct=current_pnl_pct,
                    reasoning=(
                        f"Underlying ${spot_price:.2f} dropped below short "
                        f"strike ${short_strike:.2f} by more than "
                        f"{_UNDERLYING_BELOW_SHORT_STRIKE_THRESHOLD:.0%} "
                        f"(threshold ${threshold_price:.2f})"
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
        """Construct the two legs of a bull call spread.

        Searches the options chain for:
        1. A long call with delta in the configured range (0.55--0.70).
        2. A short call with delta in the configured range (0.30--0.45),
           preferably ``wing_width`` dollars above the long strike.

        Both must be calls with the same expiration.

        Args:
            spot_price: Current underlying price.
            options_chain: Options dicts with ``"strike"``, ``"expiry"``,
                ``"right"``, ``"delta"``, ``"bid"``, ``"ask"``,
                ``"open_interest"`` etc.
            **kwargs: Optional ``wing_width`` override.

        Returns:
            List of two :class:`LegSpec`: ``[long_call, short_call]``.

        Raises:
            ValueError: If suitable strikes cannot be found in the chain.
        """
        wing_width = kwargs.get("wing_width", self._wing_width)

        # Find the long call (higher delta, lower strike, closer to ITM).
        long_delta_target = (self._long_delta_min + self._long_delta_max) / 2.0
        long_call = self.find_strike_by_delta(
            options=options_chain,
            target_delta=long_delta_target,
            right="C",
            tolerance=(self._long_delta_max - self._long_delta_min) / 2.0,
        )
        if long_call is None:
            raise ValueError(
                f"No call found with delta near {long_delta_target:.2f} "
                f"(range [{self._long_delta_min}, {self._long_delta_max}])"
            )

        if not self._check_liquidity(long_call):
            raise ValueError(
                f"Long call at strike {long_call['strike']} has insufficient "
                f"liquidity (OI or bid-ask spread)"
            )

        long_strike: float = long_call["strike"]
        long_expiry: str = self._normalise_expiry(long_call["expiry"])

        # Find the short call (lower delta, higher strike, further OTM).
        # Prefer strikes that are approximately wing_width above the long.
        target_short_strike = long_strike + wing_width
        short_delta_target = (self._short_delta_min + self._short_delta_max) / 2.0

        # First, collect all candidates in the delta range.
        short_candidates: list[tuple[float, dict[str, Any]]] = []
        for opt in options_chain:
            if opt.get("right") != "C":
                continue
            opt_delta = opt.get("delta")
            if opt_delta is None or (
                isinstance(opt_delta, float) and math.isnan(opt_delta)
            ):
                continue
            abs_delta = abs(opt_delta)
            abs_target = abs(short_delta_target)
            delta_tol = (self._short_delta_max - self._short_delta_min) / 2.0
            if abs(abs_delta - abs_target) <= delta_tol:
                strike_distance = abs(opt["strike"] - target_short_strike)
                short_candidates.append((strike_distance, opt))

        if not short_candidates:
            raise ValueError(
                f"No call found with delta near {short_delta_target:.2f} "
                f"(range [{self._short_delta_min}, {self._short_delta_max}])"
            )

        short_candidates.sort(key=lambda x: x[0])
        short_call = short_candidates[0][1]

        if not self._check_liquidity(short_call):
            raise ValueError(
                f"Short call at strike {short_call['strike']} has "
                f"insufficient liquidity"
            )

        short_strike: float = short_call["strike"]
        short_expiry: str = self._normalise_expiry(short_call["expiry"])

        # Ensure the short strike is above the long strike.
        if short_strike <= long_strike:
            raise ValueError(
                f"Short call strike ({short_strike}) must be above "
                f"long call strike ({long_strike})"
            )

        self._log.debug(
            "legs_constructed",
            long_strike=long_strike,
            long_delta=long_call.get("delta"),
            short_strike=short_strike,
            short_delta=short_call.get("delta"),
            actual_width=short_strike - long_strike,
        )

        return [
            LegSpec(
                action="BUY",
                right="C",
                strike=long_strike,
                expiry=long_expiry,
                quantity=1,
                delta=long_call.get("delta"),
            ),
            LegSpec(
                action="SELL",
                right="C",
                strike=short_strike,
                expiry=short_expiry,
                quantity=1,
                delta=short_call.get("delta"),
            ),
        ]

    # ------------------------------------------------------------------
    # Profit / Loss calculation
    # ------------------------------------------------------------------

    def calculate_max_profit(self, legs: list[LegSpec], net_premium: float) -> float:
        """Calculate maximum profit for a bull call spread.

        ``Max profit = (strike_width - net_debit) x multiplier``.

        Occurs when the underlying is at or above the short call strike
        at expiration.

        Args:
            legs: The two legs ``[long_call, short_call]``.
            net_premium: Net debit paid per spread unit (positive number).

        Returns:
            Maximum profit in dollars (positive).
        """
        strikes = sorted([leg.strike for leg in legs])
        strike_width = strikes[-1] - strikes[0]
        max_profit = (strike_width - net_premium) * _MULTIPLIER

        if max_profit < 0:
            self._log.warning(
                "negative_max_profit",
                strike_width=strike_width,
                net_premium=net_premium,
                calculated=max_profit,
            )
            return 0.01  # TradeSignal requires max_loss > 0; avoid zero

        return round(max_profit, 2)

    def calculate_max_loss(self, legs: list[LegSpec], net_premium: float) -> float:
        """Calculate maximum loss for a bull call spread.

        ``Max loss = net_debit x multiplier``.

        Occurs when the underlying is at or below the long call strike
        at expiration.

        Args:
            legs: The two legs ``[long_call, short_call]``.
            net_premium: Net debit paid per spread unit (positive number).

        Returns:
            Maximum loss in dollars (positive number).
        """
        max_loss = abs(net_premium) * _MULTIPLIER
        return round(max_loss, 2)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _find_short_strike(trade: dict[str, Any]) -> float | None:
        """Find the short (sold) call strike from a trade dict.

        Searches the ``"legs"`` list for a leg with ``action == "SELL"``.

        Args:
            trade: Trade dict with ``"legs"`` key.

        Returns:
            The short strike price, or ``None`` if not found.
        """
        legs = trade.get("legs", [])
        for leg in legs:
            if isinstance(leg, dict):
                if leg.get("action") == "SELL":
                    return leg.get("strike")
            elif hasattr(leg, "action") and leg.action == "SELL":
                return leg.strike
        return None

    @staticmethod
    def _get_mid_price(
        chain: list[dict[str, Any]],
        strike: float,
        right: str,
    ) -> float | None:
        """Look up the mid-price for a specific strike and right in the chain.

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
        """Verify that an option dict has sufficient open interest and
        a reasonable bid-ask spread.

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

        Accepts ``date``, ``datetime``, and ``str`` inputs.

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
