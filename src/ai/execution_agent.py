"""Execution Agent for Project Titan.

Translates approved trade proposals from the Risk Agent into concrete
IBKR combo/spread orders and monitors their execution lifecycle.  The
agent calls Claude to interpret complex proposals into precise order
specifications, then builds the final order with slippage adjustments
and monitors fill status.

Usage::

    from src.ai.execution_agent import ExecutionAgent

    agent = ExecutionAgent(api_key="sk-ant-...", model="claude-sonnet-4-6")
    plan = await agent.plan_execution(proposal)
    order = agent.build_order_from_plan(plan)
    result = await agent.monitor_execution(order_id=12345, timeout_seconds=60)
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import TYPE_CHECKING, Any

from anthropic import AsyncAnthropic
from pydantic import BaseModel, Field
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.ai.prompts import EXECUTION_AGENT_SYSTEM_PROMPT
from src.utils.logging import get_logger

if TYPE_CHECKING:
    import structlog

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_SLIPPAGE_PCT: float = 0.15
MAX_SLIPPAGE_PCT: float = 0.30
DEFAULT_TIMEOUT_SECONDS: int = 60
POLL_INTERVAL_SECONDS: float = 1.0
MAX_API_TOKENS: int = 4096

# Strategy types that require margin checks
MARGIN_STRATEGIES: frozenset[str] = frozenset(
    {
        "short_strangle",
        "ratio_spread",
    }
)

# Mapping of credit vs debit strategies for price adjustment
CREDIT_STRATEGIES: frozenset[str] = frozenset(
    {
        "bull_put_spread",
        "iron_condor",
        "short_strangle",
        "calendar_spread",
        "diagonal_spread",
        "broken_wing_butterfly",
        "pmcc",
    }
)

DEBIT_STRATEGIES: frozenset[str] = frozenset(
    {
        "bull_call_spread",
        "long_straddle",
        "ratio_spread",
    }
)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class LegOrderSpec(BaseModel):
    """Specification for a single leg in a multi-leg options order.

    Attributes:
        symbol: Underlying ticker symbol.
        expiry: Option expiration date in ISO format (YYYY-MM-DD).
        strike: Strike price of the option.
        right: Option right -- ``C`` for call, ``P`` for put.
        action: Whether this leg is bought or sold (``BUY`` or ``SELL``).
        ratio: Number of contracts per unit of the spread.
    """

    symbol: str = Field(..., description="Underlying ticker symbol")
    expiry: str = Field(..., description="Option expiration date (ISO format)")
    strike: float = Field(..., gt=0, description="Strike price")
    right: str = Field(..., description="Option right: C (call) or P (put)")
    action: str = Field(..., description="BUY or SELL")
    ratio: int = Field(default=1, ge=1, description="Contract ratio per spread unit")


class OrderSpec(BaseModel):
    """Complete order specification for a multi-leg options spread.

    Attributes:
        order_type: IBKR order type (always ``LMT`` for limit).
        action: Top-level action for the combo (``BUY`` or ``SELL``).
        total_quantity: Number of spread units to trade.
        limit_price: Net limit price for the entire combo.
        time_in_force: Time-in-force qualifier (``GTC`` or ``DAY``).
        legs: Individual leg specifications.
        combo_smart_routing: Use SMART routing for combo orders.
        non_guaranteed: Whether legs may fill independently.
    """

    order_type: str = Field(
        default="LMT",
        description="IBKR order type (always LMT for spreads)",
    )
    action: str = Field(..., description="Top-level combo action: BUY or SELL")
    total_quantity: int = Field(..., ge=1, description="Number of spread units")
    limit_price: float = Field(..., description="Net limit price for the combo")
    time_in_force: str = Field(
        default="GTC",
        description="Time-in-force: GTC or DAY",
    )
    legs: list[LegOrderSpec] = Field(
        ..., min_length=1, description="Leg specifications"
    )
    combo_smart_routing: bool = Field(
        default=True,
        description="Use SMART combo routing for best execution",
    )
    non_guaranteed: bool = Field(
        default=False,
        description=(
            "When False, all legs fill atomically (guaranteed). "
            "Always False for spreads."
        ),
    )


class ExecutionPlan(BaseModel):
    """Plan produced by the Execution Agent before order submission.

    Attributes:
        order_specs: List of order specifications to submit.
        strategy_type: Name of the strategy being executed.
        expected_fill_price: Estimated fill price before slippage.
        slippage_estimate: Estimated slippage in dollars.
        execution_notes: Agent commentary on execution considerations.
        requires_margin_check: Whether a margin check is needed.
    """

    order_specs: list[OrderSpec] = Field(
        ..., min_length=1, description="Orders to submit"
    )
    strategy_type: str = Field(..., description="Strategy name")
    expected_fill_price: float = Field(
        ..., description="Expected fill price before slippage"
    )
    slippage_estimate: float = Field(
        default=0.0,
        description="Estimated slippage in dollars per spread unit",
    )
    execution_notes: str = Field(
        default="",
        description="Agent commentary on execution considerations",
    )
    requires_margin_check: bool = Field(
        default=False,
        description="Whether margin requirements should be verified",
    )


class ExecutionResult(BaseModel):
    """Result of order execution after monitoring.

    Attributes:
        success: Whether the order filled successfully.
        order_id: IBKR order identifier (None if submission failed).
        fill_price: Actual fill price (None if not filled).
        fill_time: ISO timestamp of fill (None if not filled).
        slippage: Actual slippage vs limit price (None if not filled).
        status: Terminal order status.
        error_message: Error description if execution failed.
    """

    success: bool = Field(..., description="Whether the order filled successfully")
    order_id: int | None = Field(default=None, description="IBKR order identifier")
    fill_price: float | None = Field(default=None, description="Actual fill price")
    fill_time: str | None = Field(default=None, description="ISO timestamp of fill")
    slippage: float | None = Field(
        default=None,
        description="Actual slippage vs expected price",
    )
    status: str = Field(
        default="PENDING",
        description=("Terminal status: FILLED, PARTIAL, CANCELLED, FAILED, or PENDING"),
    )
    error_message: str | None = Field(
        default=None, description="Error description if failed"
    )


# ---------------------------------------------------------------------------
# ExecutionAgent
# ---------------------------------------------------------------------------


class ExecutionAgent:
    """Translates approved trade proposals into IBKR orders.

    The agent uses Claude to interpret complex trade proposals, builds
    precise multi-leg order specifications with slippage adjustments,
    and monitors execution status.

    Args:
        api_key: Anthropic API key.
        model: Claude model identifier for API calls.
        order_manager: Optional :class:`~src.broker.orders.OrderManager`
            for live order status polling.  When ``None``, order status
            polling returns a static ``PENDING`` response.
    """

    def __init__(
        self,
        api_key: str,
        model: str,
        order_manager: Any | None = None,
    ) -> None:
        self._client: AsyncAnthropic = AsyncAnthropic(api_key=api_key)
        self._model: str = model
        self._order_manager: Any | None = order_manager
        self._log: structlog.stdlib.BoundLogger = get_logger("ai.execution_agent")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        reraise=True,
    )
    async def plan_execution(self, proposal: dict[str, Any]) -> ExecutionPlan:
        """Create an execution plan from an approved trade proposal.

        Calls Claude to interpret the proposal and produce structured
        order specifications.  The system prompt is cached for
        efficiency across repeated calls.

        Args:
            proposal: Approved trade proposal dict containing at
                minimum ``ticker``, ``strategy``, ``direction``,
                ``parameters`` (with strikes, expiries, etc.),
                and optionally ``bid``, ``ask``, ``natural_price``.

        Returns:
            A fully populated :class:`ExecutionPlan`.

        Raises:
            ValueError: If the proposal is missing required fields.
            anthropic.APIError: On unrecoverable API failures.
        """
        self._log.info(
            "planning_execution",
            ticker=proposal.get("ticker"),
            strategy=proposal.get("strategy"),
        )

        self._validate_proposal(proposal)

        strategy = proposal["strategy"]
        legs = self._build_legs_for_strategy(strategy, proposal)
        natural_price = self._extract_natural_price(proposal)
        bid = proposal.get("bid", 0.0)
        ask = proposal.get("ask", 0.0)
        slippage = self._calculate_slippage(bid, ask)

        user_message = self._build_execution_prompt(
            proposal, legs, natural_price, slippage
        )

        response = await self._client.messages.create(
            model=self._model,
            max_tokens=MAX_API_TOKENS,
            system=[
                {
                    "type": "text",
                    "text": EXECUTION_AGENT_SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                },
            ],
            messages=[{"role": "user", "content": user_message}],
        )

        execution_notes = self._extract_text(response)

        direction = proposal.get("direction", "LONG")
        action = "BUY" if direction == "LONG" else "SELL"
        quantity = proposal.get("quantity", 1)
        time_in_force = proposal.get("time_in_force", "GTC")
        requires_margin = strategy in MARGIN_STRATEGIES

        order_spec = OrderSpec(
            order_type="LMT",
            action=action,
            total_quantity=quantity,
            limit_price=round(natural_price, 2),
            time_in_force=time_in_force,
            legs=legs,
            combo_smart_routing=True,
            non_guaranteed=False,
        )

        plan = ExecutionPlan(
            order_specs=[order_spec],
            strategy_type=strategy,
            expected_fill_price=round(natural_price, 2),
            slippage_estimate=round(slippage, 4),
            execution_notes=execution_notes,
            requires_margin_check=requires_margin,
        )

        self._log.info(
            "execution_plan_created",
            strategy=strategy,
            expected_fill=plan.expected_fill_price,
            slippage=plan.slippage_estimate,
            num_legs=len(legs),
            requires_margin=requires_margin,
        )

        return plan

    def build_order_from_plan(self, plan: ExecutionPlan) -> OrderSpec:
        """Translate an execution plan into a final order specification.

        Applies slippage adjustment to the limit price based on
        whether the strategy is a debit or credit spread.  Always
        uses limit orders -- never market orders.

        For credit spreads the limit price is reduced by the slippage
        estimate (willing to accept less credit).  For debit spreads
        the limit price is increased by the slippage estimate (willing
        to pay more).

        Args:
            plan: The execution plan to finalise.

        Returns:
            A slippage-adjusted :class:`OrderSpec` ready for
            submission to the broker.
        """
        if not plan.order_specs:
            raise ValueError("Execution plan has no order specs")

        base_spec = plan.order_specs[0]
        adjusted_price = self._apply_slippage_adjustment(
            natural_price=base_spec.limit_price,
            slippage=plan.slippage_estimate,
            strategy=plan.strategy_type,
        )

        adjusted_order = OrderSpec(
            order_type=base_spec.order_type,
            action=base_spec.action,
            total_quantity=base_spec.total_quantity,
            limit_price=round(adjusted_price, 2),
            time_in_force=base_spec.time_in_force,
            legs=base_spec.legs,
            combo_smart_routing=base_spec.combo_smart_routing,
            non_guaranteed=base_spec.non_guaranteed,
        )

        self._log.info(
            "order_built_from_plan",
            strategy=plan.strategy_type,
            original_price=base_spec.limit_price,
            adjusted_price=adjusted_order.limit_price,
            slippage_applied=plan.slippage_estimate,
        )

        return adjusted_order

    async def monitor_execution(
        self,
        order_id: int,
        timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    ) -> ExecutionResult:
        """Monitor an order until it reaches a terminal state or times out.

        Polls order status at regular intervals and returns the final
        execution result.  In production this would interface with
        :class:`~src.broker.orders.OrderManager` callbacks; this
        implementation provides the polling framework and logging.

        Args:
            order_id: IBKR order identifier to monitor.
            timeout_seconds: Maximum seconds to wait for a fill.

        Returns:
            An :class:`ExecutionResult` describing the final state.
        """
        self._log.info(
            "monitoring_execution",
            order_id=order_id,
            timeout_seconds=timeout_seconds,
        )

        start_time = time.monotonic()
        last_status = "PENDING"

        while (time.monotonic() - start_time) < timeout_seconds:
            status_info = await self._poll_order_status(order_id)
            current_status = status_info.get("status", "PENDING")

            if current_status != last_status:
                self._log.info(
                    "order_status_changed",
                    order_id=order_id,
                    old_status=last_status,
                    new_status=current_status,
                )
                last_status = current_status

            if current_status == "FILLED":
                fill_price = status_info.get("fill_price", 0.0)
                expected = status_info.get("expected_price", fill_price)
                actual_slippage = abs(fill_price - expected)

                result = ExecutionResult(
                    success=True,
                    order_id=order_id,
                    fill_price=fill_price,
                    fill_time=status_info.get("fill_time"),
                    slippage=round(actual_slippage, 4),
                    status="FILLED",
                )
                self._log.info(
                    "order_filled",
                    order_id=order_id,
                    fill_price=fill_price,
                    slippage=actual_slippage,
                )
                return result

            if current_status == "CANCELLED":
                result = ExecutionResult(
                    success=False,
                    order_id=order_id,
                    status="CANCELLED",
                    error_message=status_info.get("reason", "Order cancelled"),
                )
                self._log.warning(
                    "order_cancelled",
                    order_id=order_id,
                    reason=result.error_message,
                )
                return result

            if current_status == "PARTIAL":
                filled_qty = status_info.get("filled_qty", 0)
                self._log.info(
                    "order_partial_fill",
                    order_id=order_id,
                    filled_qty=filled_qty,
                )

            if current_status in ("REJECTED", "ERROR"):
                result = ExecutionResult(
                    success=False,
                    order_id=order_id,
                    status="FAILED",
                    error_message=status_info.get("reason", "Order rejected"),
                )
                self._log.error(
                    "order_failed",
                    order_id=order_id,
                    reason=result.error_message,
                )
                return result

            await asyncio.sleep(POLL_INTERVAL_SECONDS)

        # Timeout reached
        self._log.warning(
            "order_timeout",
            order_id=order_id,
            timeout_seconds=timeout_seconds,
            last_status=last_status,
        )

        final_status = "PARTIAL" if last_status == "PARTIAL" else "PENDING"
        return ExecutionResult(
            success=False,
            order_id=order_id,
            status=final_status,
            error_message=(
                f"Order did not fill within {timeout_seconds}s. "
                f"Last status: {last_status}"
            ),
        )

    # ------------------------------------------------------------------
    # Strategy-specific leg construction
    # ------------------------------------------------------------------

    def _build_legs_for_strategy(
        self,
        strategy: str,
        proposal: dict[str, Any],
    ) -> list[LegOrderSpec]:
        """Build leg specifications for a given strategy.

        Each strategy has a fixed leg structure. Parameters such as
        strikes, expiries, and the underlying symbol are extracted
        from the proposal dict.

        Args:
            strategy: Canonical strategy name.
            proposal: Trade proposal containing ``ticker`` and
                ``parameters`` with strikes and expiries.

        Returns:
            Ordered list of :class:`LegOrderSpec` instances.

        Raises:
            ValueError: If the strategy is unknown or parameters
                are missing.
        """
        builders = {
            "bull_call_spread": self._legs_bull_call_spread,
            "bull_put_spread": self._legs_bull_put_spread,
            "iron_condor": self._legs_iron_condor,
            "calendar_spread": self._legs_calendar_spread,
            "diagonal_spread": self._legs_diagonal_spread,
            "broken_wing_butterfly": self._legs_broken_wing_butterfly,
            "short_strangle": self._legs_short_strangle,
            "pmcc": self._legs_pmcc,
            "ratio_spread": self._legs_ratio_spread,
            "long_straddle": self._legs_long_straddle,
        }

        builder = builders.get(strategy)
        if builder is None:
            raise ValueError(f"Unknown strategy: {strategy}")

        return builder(proposal)

    def _legs_bull_call_spread(self, proposal: dict[str, Any]) -> list[LegOrderSpec]:
        """BUY lower strike call, SELL higher strike call."""
        params = self._get_params(proposal)
        symbol = proposal["ticker"]
        expiry = params["expiry"]

        return [
            LegOrderSpec(
                symbol=symbol,
                expiry=expiry,
                strike=params["long_strike"],
                right="C",
                action="BUY",
            ),
            LegOrderSpec(
                symbol=symbol,
                expiry=expiry,
                strike=params["short_strike"],
                right="C",
                action="SELL",
            ),
        ]

    def _legs_bull_put_spread(self, proposal: dict[str, Any]) -> list[LegOrderSpec]:
        """SELL higher strike put, BUY lower strike put."""
        params = self._get_params(proposal)
        symbol = proposal["ticker"]
        expiry = params["expiry"]

        return [
            LegOrderSpec(
                symbol=symbol,
                expiry=expiry,
                strike=params["short_strike"],
                right="P",
                action="SELL",
            ),
            LegOrderSpec(
                symbol=symbol,
                expiry=expiry,
                strike=params["long_strike"],
                right="P",
                action="BUY",
            ),
        ]

    def _legs_iron_condor(self, proposal: dict[str, Any]) -> list[LegOrderSpec]:
        """SELL OTM put, BUY lower put, SELL OTM call, BUY higher call."""
        params = self._get_params(proposal)
        symbol = proposal["ticker"]
        expiry = params["expiry"]

        return [
            LegOrderSpec(
                symbol=symbol,
                expiry=expiry,
                strike=params["short_put_strike"],
                right="P",
                action="SELL",
            ),
            LegOrderSpec(
                symbol=symbol,
                expiry=expiry,
                strike=params["long_put_strike"],
                right="P",
                action="BUY",
            ),
            LegOrderSpec(
                symbol=symbol,
                expiry=expiry,
                strike=params["short_call_strike"],
                right="C",
                action="SELL",
            ),
            LegOrderSpec(
                symbol=symbol,
                expiry=expiry,
                strike=params["long_call_strike"],
                right="C",
                action="BUY",
            ),
        ]

    def _legs_calendar_spread(self, proposal: dict[str, Any]) -> list[LegOrderSpec]:
        """SELL front month, BUY back month (same strike)."""
        params = self._get_params(proposal)
        symbol = proposal["ticker"]
        strike = params["strike"]
        right = params.get("right", "C")

        return [
            LegOrderSpec(
                symbol=symbol,
                expiry=params["front_expiry"],
                strike=strike,
                right=right,
                action="SELL",
            ),
            LegOrderSpec(
                symbol=symbol,
                expiry=params["back_expiry"],
                strike=strike,
                right=right,
                action="BUY",
            ),
        ]

    def _legs_diagonal_spread(self, proposal: dict[str, Any]) -> list[LegOrderSpec]:
        """SELL front month OTM, BUY back month ITM."""
        params = self._get_params(proposal)
        symbol = proposal["ticker"]
        right = params.get("right", "C")

        return [
            LegOrderSpec(
                symbol=symbol,
                expiry=params["front_expiry"],
                strike=params["short_strike"],
                right=right,
                action="SELL",
            ),
            LegOrderSpec(
                symbol=symbol,
                expiry=params["back_expiry"],
                strike=params["long_strike"],
                right=right,
                action="BUY",
            ),
        ]

    def _legs_broken_wing_butterfly(
        self, proposal: dict[str, Any]
    ) -> list[LegOrderSpec]:
        """BUY lower, SELL 2x middle, BUY upper (skip-strike)."""
        params = self._get_params(proposal)
        symbol = proposal["ticker"]
        expiry = params["expiry"]
        right = params.get("right", "P")

        return [
            LegOrderSpec(
                symbol=symbol,
                expiry=expiry,
                strike=params["lower_strike"],
                right=right,
                action="BUY",
            ),
            LegOrderSpec(
                symbol=symbol,
                expiry=expiry,
                strike=params["middle_strike"],
                right=right,
                action="SELL",
                ratio=2,
            ),
            LegOrderSpec(
                symbol=symbol,
                expiry=expiry,
                strike=params["upper_strike"],
                right=right,
                action="BUY",
            ),
        ]

    def _legs_short_strangle(self, proposal: dict[str, Any]) -> list[LegOrderSpec]:
        """SELL OTM put, SELL OTM call."""
        params = self._get_params(proposal)
        symbol = proposal["ticker"]
        expiry = params["expiry"]

        return [
            LegOrderSpec(
                symbol=symbol,
                expiry=expiry,
                strike=params["put_strike"],
                right="P",
                action="SELL",
            ),
            LegOrderSpec(
                symbol=symbol,
                expiry=expiry,
                strike=params["call_strike"],
                right="C",
                action="SELL",
            ),
        ]

    def _legs_pmcc(self, proposal: dict[str, Any]) -> list[LegOrderSpec]:
        """BUY LEAPS call, SELL near-term OTM call."""
        params = self._get_params(proposal)
        symbol = proposal["ticker"]

        return [
            LegOrderSpec(
                symbol=symbol,
                expiry=params["leaps_expiry"],
                strike=params["leaps_strike"],
                right="C",
                action="BUY",
            ),
            LegOrderSpec(
                symbol=symbol,
                expiry=params["short_expiry"],
                strike=params["short_strike"],
                right="C",
                action="SELL",
            ),
        ]

    def _legs_ratio_spread(self, proposal: dict[str, Any]) -> list[LegOrderSpec]:
        """BUY 1x ITM, SELL 2x OTM."""
        params = self._get_params(proposal)
        symbol = proposal["ticker"]
        expiry = params["expiry"]
        right = params.get("right", "C")
        short_ratio = params.get("short_ratio", 2)

        return [
            LegOrderSpec(
                symbol=symbol,
                expiry=expiry,
                strike=params["long_strike"],
                right=right,
                action="BUY",
                ratio=1,
            ),
            LegOrderSpec(
                symbol=symbol,
                expiry=expiry,
                strike=params["short_strike"],
                right=right,
                action="SELL",
                ratio=short_ratio,
            ),
        ]

    def _legs_long_straddle(self, proposal: dict[str, Any]) -> list[LegOrderSpec]:
        """BUY ATM call, BUY ATM put."""
        params = self._get_params(proposal)
        symbol = proposal["ticker"]
        expiry = params["expiry"]
        strike = params["strike"]

        return [
            LegOrderSpec(
                symbol=symbol,
                expiry=expiry,
                strike=strike,
                right="C",
                action="BUY",
            ),
            LegOrderSpec(
                symbol=symbol,
                expiry=expiry,
                strike=strike,
                right="P",
                action="BUY",
            ),
        ]

    # ------------------------------------------------------------------
    # Slippage and price adjustment
    # ------------------------------------------------------------------

    def _calculate_slippage(
        self,
        bid: float,
        ask: float,
        assumed_pct: float = DEFAULT_SLIPPAGE_PCT,
    ) -> float:
        """Calculate estimated slippage from bid-ask spread.

        Slippage is estimated as a fraction of the bid-ask spread
        width.  The default assumption is 15% of the spread.

        Args:
            bid: Best bid price.
            ask: Best ask price.
            assumed_pct: Fraction of the spread to assume as
                slippage (0.0 to 1.0).

        Returns:
            Estimated slippage in dollars per contract.
        """
        if ask <= 0.0 or bid <= 0.0 or ask <= bid:
            return 0.0

        spread_width = ask - bid
        slippage = spread_width * min(assumed_pct, MAX_SLIPPAGE_PCT)
        return round(slippage, 4)

    def _apply_slippage_adjustment(
        self,
        natural_price: float,
        slippage: float,
        strategy: str,
    ) -> float:
        """Adjust the limit price for expected slippage.

        For credit spreads (selling premium), the limit price is
        reduced -- we accept slightly less credit to improve fill
        probability.  For debit spreads (buying premium), the limit
        price is increased -- we are willing to pay slightly more.

        Args:
            natural_price: Unadjusted mid-market price.
            slippage: Slippage amount to apply.
            strategy: Canonical strategy name.

        Returns:
            Slippage-adjusted limit price.
        """
        if strategy in CREDIT_STRATEGIES:
            adjusted = natural_price - slippage
        elif strategy in DEBIT_STRATEGIES:
            adjusted = natural_price + slippage
        else:
            # Conservative default: increase price (debit assumption)
            adjusted = natural_price + slippage

        return round(adjusted, 2)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_proposal(self, proposal: dict[str, Any]) -> None:
        """Validate that a proposal contains required fields.

        Args:
            proposal: Trade proposal dict.

        Raises:
            ValueError: If required fields are missing.
        """
        required_fields = ("ticker", "strategy", "direction")
        missing = [f for f in required_fields if f not in proposal]
        if missing:
            raise ValueError(f"Proposal missing required fields: {missing}")

        if "parameters" not in proposal:
            raise ValueError(
                "Proposal must include 'parameters' dict with "
                "strike prices and expiry dates"
            )

    def _get_params(self, proposal: dict[str, Any]) -> dict[str, Any]:
        """Extract the parameters dict from a proposal.

        Args:
            proposal: Trade proposal dict.

        Returns:
            The ``parameters`` sub-dict.

        Raises:
            ValueError: If parameters key is missing.
        """
        params = proposal.get("parameters")
        if params is None:
            raise ValueError("Proposal missing 'parameters' key")
        return params

    def _extract_natural_price(self, proposal: dict[str, Any]) -> float:
        """Extract or calculate the natural mid-market price.

        Tries the following sources in order:
        1. ``natural_price`` key in proposal
        2. ``mid_price`` key in proposal
        3. Midpoint of ``bid`` and ``ask`` keys
        4. ``parameters.estimated_price``

        Args:
            proposal: Trade proposal dict.

        Returns:
            Natural price as a float.

        Raises:
            ValueError: If no price information can be found.
        """
        if "natural_price" in proposal:
            return float(proposal["natural_price"])

        if "mid_price" in proposal:
            return float(proposal["mid_price"])

        bid = proposal.get("bid", 0.0)
        ask = proposal.get("ask", 0.0)
        if bid > 0.0 and ask > 0.0:
            return round((bid + ask) / 2.0, 4)

        params = proposal.get("parameters", {})
        if "estimated_price" in params:
            return float(params["estimated_price"])

        raise ValueError(
            "Cannot determine natural price from proposal. "
            "Provide 'natural_price', 'mid_price', 'bid'/'ask', "
            "or 'parameters.estimated_price'."
        )

    def _build_execution_prompt(
        self,
        proposal: dict[str, Any],
        legs: list[LegOrderSpec],
        natural_price: float,
        slippage: float,
    ) -> str:
        """Build the user message for the Claude API call.

        Args:
            proposal: Trade proposal dict.
            legs: Constructed leg specifications.
            natural_price: Calculated natural price.
            slippage: Estimated slippage.

        Returns:
            Formatted prompt string.
        """
        legs_text = "\n".join(
            f"  Leg {i + 1}: {leg.action} {leg.symbol} "
            f"{leg.expiry} {leg.strike} {leg.right} "
            f"x{leg.ratio}"
            for i, leg in enumerate(legs)
        )

        return (
            f"Translate this approved trade proposal into an IBKR "
            f"combo order.\n\n"
            f"Ticker: {proposal['ticker']}\n"
            f"Strategy: {proposal['strategy']}\n"
            f"Direction: {proposal['direction']}\n"
            f"Quantity: {proposal.get('quantity', 1)}\n"
            f"Natural Price: {natural_price:.4f}\n"
            f"Slippage Estimate: {slippage:.4f}\n"
            f"Bid: {proposal.get('bid', 'N/A')}\n"
            f"Ask: {proposal.get('ask', 'N/A')}\n\n"
            f"Constructed Legs:\n{legs_text}\n\n"
            f"Parameters: {json.dumps(proposal.get('parameters', {}), indent=2)}\n\n"
            f"Provide execution notes: any concerns about liquidity, "
            f"timing, fill probability, or suggested order "
            f"modifications. Keep the response concise."
        )

    def _extract_text(self, response: Any) -> str:
        """Extract text content from a Claude API response.

        Args:
            response: Anthropic API response object.

        Returns:
            Concatenated text content from all text blocks.
        """
        parts: list[str] = []
        for block in response.content:
            if hasattr(block, "text"):
                parts.append(block.text)
        return "\n".join(parts) if parts else ""

    async def _poll_order_status(self, order_id: int) -> dict[str, Any]:
        """Poll the current status of an order from IBKR.

        Queries the :class:`~src.broker.orders.OrderManager` for the
        live Trade object matching *order_id*, then maps the ib_async
        order-status string to the polling contract expected by
        :meth:`monitor_execution`.

        Args:
            order_id: IBKR order identifier.

        Returns:
            Dict with ``status`` key and optional fill details
            (``fill_price``, ``filled_qty``, ``fill_time``,
            ``expected_price``, ``reason``).
        """
        self._log.debug("polling_order_status", order_id=order_id)

        if self._order_manager is None:
            return {"status": "PENDING", "order_id": order_id}

        # ib_async status strings → our canonical statuses
        status_map: dict[str, str] = {
            "Submitted": "PENDING",
            "PreSubmitted": "PENDING",
            "PendingSubmit": "PENDING",
            "PendingCancel": "PENDING",
            "ApiPending": "PENDING",
            "Filled": "FILLED",
            "Cancelled": "CANCELLED",
            "ApiCancelled": "CANCELLED",
            "Inactive": "REJECTED",
        }

        try:
            open_trades = self._order_manager.get_open_orders()
            # Also check filled trades via ib_async
            all_trades = open_trades
            if hasattr(self._order_manager, "_ib"):
                all_trades = list(self._order_manager._ib.trades())

            for trade in all_trades:
                if trade.order.orderId == order_id:
                    ib_status = trade.orderStatus.status
                    canonical = status_map.get(ib_status, "PENDING")

                    result: dict[str, Any] = {
                        "status": canonical,
                        "order_id": order_id,
                    }

                    if canonical == "FILLED":
                        avg_price = trade.orderStatus.avgFillPrice
                        result["fill_price"] = avg_price
                        result["filled_qty"] = int(trade.orderStatus.filled)
                        # Use expected price from order manager if tracked
                        if hasattr(self._order_manager, "_expected_prices"):
                            result["expected_price"] = (
                                self._order_manager._expected_prices.get(
                                    order_id,
                                    avg_price,
                                )
                            )
                        # Capture fill time from the last fill event
                        if trade.fills:
                            last_fill = trade.fills[-1]
                            result["fill_time"] = str(last_fill.time)

                    elif canonical == "CANCELLED":
                        result["reason"] = f"Order cancelled (IB status: {ib_status})"

                    elif canonical == "REJECTED":
                        result["reason"] = (
                            f"Order rejected/inactive (IB status: {ib_status})"
                        )

                    elif ib_status == "Filled" and trade.orderStatus.remaining > 0:
                        result["status"] = "PARTIAL"
                        result["filled_qty"] = int(trade.orderStatus.filled)

                    return result

            # Order not found in current trades — may have already
            # been cleaned up or is from a previous session.
            self._log.warning(
                "order_not_found_in_trades",
                order_id=order_id,
            )
            return {"status": "PENDING", "order_id": order_id}

        except Exception:
            self._log.exception("poll_order_status_error", order_id=order_id)
            return {"status": "PENDING", "order_id": order_id}
