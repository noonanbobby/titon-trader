"""Order execution and management for Interactive Brokers.

Provides order construction for single-leg and multi-leg spread orders,
placement, modification, cancellation, fill monitoring, and slippage
tracking.  All operations are asynchronous and fully instrumented with
structured logging.

Usage::

    from src.broker.orders import OrderManager

    mgr = OrderManager(ib)
    trade = await mgr.place_spread_order(contract, "BUY", 1, 2.35)
    trade = await mgr.wait_for_fill(trade, timeout=30.0)
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    import structlog
    from ib_async import IB

from ib_async import (
    Contract,
    Fill,
    LimitOrder,
    Order,
    OrderStatus,
    TagValue,
    Trade,
)

from src.utils.logging import get_logger

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class OrderResult(BaseModel):
    """Structured result returned after an order reaches a terminal state."""

    order_id: int = Field(description="IBKR order identifier")
    status: str = Field(description="Terminal order status string")
    filled_qty: int = Field(default=0, description="Total filled quantity")
    avg_fill_price: float = Field(
        default=0.0, description="Volume-weighted average fill price"
    )
    commission: float = Field(
        default=0.0, description="Total commission charged for all fills"
    )
    slippage: float = Field(
        default=0.0,
        description="Signed slippage vs. expected limit price (positive = worse)",
    )


class SpreadOrderRequest(BaseModel):
    """Encapsulates parameters for a spread order submission."""

    contract: object = Field(description="ib_async Contract (Bag or Option)")
    action: str = Field(description="BUY or SELL")
    quantity: int = Field(ge=1, description="Number of spread units")
    limit_price: float = Field(description="Net limit price for the combo")
    order_type: str = Field(default="LMT", description="IBKR order type (default LMT)")


# ---------------------------------------------------------------------------
# OrderManager
# ---------------------------------------------------------------------------


class OrderManager:
    """Builds, places, modifies, cancels, and monitors IBKR orders.

    Args:
        ib: A connected ``ib_async.IB`` instance.
    """

    # Slippage tracking: maps orderId -> expected limit price at submission
    _expected_prices: dict[int, float]

    def __init__(self, ib: IB) -> None:
        self._ib: IB = ib
        self._log: structlog.stdlib.BoundLogger = get_logger("broker.orders")
        self._expected_prices = {}

        # Wire up the global order-status callback so we log every transition.
        self._ib.orderStatusEvent += self._on_order_status

    # ------------------------------------------------------------------
    # Order building
    # ------------------------------------------------------------------

    def build_limit_order(
        self,
        action: str,
        quantity: int,
        limit_price: float,
        account: str = "",
    ) -> LimitOrder:
        """Build a simple limit order for a single contract.

        Args:
            action: ``"BUY"`` or ``"SELL"``.
            quantity: Number of contracts.
            limit_price: Maximum (buy) or minimum (sell) acceptable price.
            account: Optional IBKR account ID for multi-account setups.

        Returns:
            A configured :class:`ib_async.LimitOrder`.
        """
        order = LimitOrder(
            action=action,
            totalQuantity=float(quantity),
            lmtPrice=limit_price,
        )
        order.tif = "GTC"
        order.outsideRth = False
        if account:
            order.account = account

        self._log.debug(
            "built_limit_order",
            action=action,
            quantity=quantity,
            limit_price=limit_price,
        )
        return order

    def build_combo_order(
        self,
        action: str,
        quantity: int,
        limit_price: float,
        non_guaranteed: bool = False,
    ) -> Order:
        """Build a combo/spread limit order for multi-leg contracts.

        The order uses SMART combo routing and sets the ``NonGuaranteed``
        flag as requested.  Guaranteed fills (``NonGuaranteed = "0"``) cost
        slightly more but ensure all legs fill atomically.

        Args:
            action: ``"BUY"`` or ``"SELL"``.
            quantity: Number of spread units.
            limit_price: Net debit (positive) or credit (negative) price.
            non_guaranteed: When ``True`` legs may fill independently.

        Returns:
            A configured :class:`ib_async.Order` suitable for BAG contracts.
        """
        order = Order()
        order.action = action
        order.totalQuantity = float(quantity)
        order.orderType = "LMT"
        order.lmtPrice = limit_price
        order.tif = "GTC"
        order.outsideRth = False

        # SMART combo routing for best execution across exchanges.
        order.smartComboRoutingParams = [
            TagValue(tag="NonGuaranteed", value="1" if non_guaranteed else "0"),
        ]

        self._log.debug(
            "built_combo_order",
            action=action,
            quantity=quantity,
            limit_price=limit_price,
            non_guaranteed=non_guaranteed,
        )
        return order

    # ------------------------------------------------------------------
    # Order placement
    # ------------------------------------------------------------------

    async def place_spread_order(
        self,
        contract: Contract,
        action: str,
        quantity: int,
        limit_price: float,
    ) -> Trade:
        """Build and submit a combo spread order.

        Constructs the appropriate combo limit order, places it through the
        IB Gateway, registers status callbacks, and records the expected
        price for slippage tracking.

        Args:
            contract: A qualified BAG :class:`ib_async.Contract`.
            action: ``"BUY"`` or ``"SELL"``.
            quantity: Number of spread units.
            limit_price: Net limit price for the combo.

        Returns:
            The live-updated :class:`ib_async.Trade` object.
        """
        order = self.build_combo_order(
            action=action,
            quantity=quantity,
            limit_price=limit_price,
        )

        trade: Trade = self._ib.placeOrder(contract, order)
        self._expected_prices[trade.order.orderId] = limit_price

        # Register per-trade event listeners for granular tracking.
        trade.filledEvent += self._on_trade_filled
        trade.cancelledEvent += self._on_trade_cancelled

        self._log.info(
            "spread_order_placed",
            order_id=trade.order.orderId,
            action=action,
            quantity=quantity,
            limit_price=limit_price,
            contract_symbol=contract.symbol,
        )

        # Yield control so IB Gateway can start processing the order.
        await asyncio.sleep(0)
        return trade

    async def place_single_order(
        self,
        contract: Contract,
        action: str,
        quantity: int,
        limit_price: float,
    ) -> Trade:
        """Build and submit a single-leg limit order.

        Args:
            contract: A qualified :class:`ib_async.Contract` (e.g. Option).
            action: ``"BUY"`` or ``"SELL"``.
            quantity: Number of contracts.
            limit_price: Limit price.

        Returns:
            The live-updated :class:`ib_async.Trade` object.
        """
        order = self.build_limit_order(
            action=action,
            quantity=quantity,
            limit_price=limit_price,
        )

        trade: Trade = self._ib.placeOrder(contract, order)
        self._expected_prices[trade.order.orderId] = limit_price

        trade.filledEvent += self._on_trade_filled
        trade.cancelledEvent += self._on_trade_cancelled

        self._log.info(
            "single_order_placed",
            order_id=trade.order.orderId,
            action=action,
            quantity=quantity,
            limit_price=limit_price,
            contract_symbol=contract.symbol,
            sec_type=contract.secType,
        )

        await asyncio.sleep(0)
        return trade

    # ------------------------------------------------------------------
    # Order management
    # ------------------------------------------------------------------

    async def cancel_order(self, trade: Trade) -> None:
        """Cancel an open order and wait for confirmation.

        Args:
            trade: The trade to cancel.

        Raises:
            asyncio.TimeoutError: If cancellation is not confirmed within
                10 seconds.
        """
        if trade.isDone():
            self._log.warning(
                "cancel_skipped_already_done",
                order_id=trade.order.orderId,
                status=trade.orderStatus.status,
            )
            return

        self._log.info(
            "cancelling_order",
            order_id=trade.order.orderId,
        )
        self._ib.cancelOrder(trade.order)

        # Wait until the order reaches a done state.
        cancel_timeout_seconds = 10.0
        try:
            await asyncio.wait_for(
                self._wait_until_done(trade),
                timeout=cancel_timeout_seconds,
            )
        except TimeoutError:
            self._log.error(
                "cancel_timeout",
                order_id=trade.order.orderId,
                timeout_seconds=cancel_timeout_seconds,
            )
            raise

        self._log.info(
            "order_cancelled",
            order_id=trade.order.orderId,
            status=trade.orderStatus.status,
        )

    async def cancel_all_orders(self) -> None:
        """Cancel every open order via IB's global cancel request."""
        self._log.warning("cancelling_all_orders")
        self._ib.reqGlobalCancel()
        # Give IB Gateway a moment to process the mass cancellation.
        await asyncio.sleep(1.0)
        self._log.info("global_cancel_sent")

    async def modify_order(
        self,
        trade: Trade,
        new_limit_price: float,
    ) -> Trade:
        """Modify the limit price of an existing open order.

        The modification is submitted by re-placing the order with the
        same ``orderId`` but an updated limit price.

        Args:
            trade: The trade whose limit price should change.
            new_limit_price: The new limit price.

        Returns:
            The updated :class:`ib_async.Trade` object.

        Raises:
            ValueError: If the order is already in a done state.
        """
        if trade.isDone():
            raise ValueError(
                f"Cannot modify order {trade.order.orderId}: "
                f"status is {trade.orderStatus.status}"
            )

        old_price = trade.order.lmtPrice
        trade.order.lmtPrice = new_limit_price
        self._expected_prices[trade.order.orderId] = new_limit_price

        updated_trade: Trade = self._ib.placeOrder(trade.contract, trade.order)

        self._log.info(
            "order_modified",
            order_id=trade.order.orderId,
            old_limit_price=old_price,
            new_limit_price=new_limit_price,
        )

        await asyncio.sleep(0)
        return updated_trade

    async def wait_for_fill(
        self,
        trade: Trade,
        timeout: float = 30.0,
    ) -> Trade:
        """Wait for an order to reach a terminal state.

        Args:
            trade: The trade to monitor.
            timeout: Maximum seconds to wait before raising
                :class:`asyncio.TimeoutError`.

        Returns:
            The :class:`ib_async.Trade` with final fill information.

        Raises:
            asyncio.TimeoutError: If the order does not complete within
                *timeout* seconds.
        """
        if trade.isDone():
            return trade

        self._log.debug(
            "waiting_for_fill",
            order_id=trade.order.orderId,
            timeout_seconds=timeout,
        )

        try:
            await asyncio.wait_for(
                self._wait_until_done(trade),
                timeout=timeout,
            )
        except TimeoutError:
            self._log.warning(
                "fill_timeout",
                order_id=trade.order.orderId,
                timeout_seconds=timeout,
                filled=trade.orderStatus.filled,
                remaining=trade.orderStatus.remaining,
            )
            raise

        result = self._build_order_result(trade)
        self._log.info(
            "order_completed",
            order_id=result.order_id,
            status=result.status,
            filled_qty=result.filled_qty,
            avg_fill_price=result.avg_fill_price,
            commission=result.commission,
            slippage=result.slippage,
        )
        return trade

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def get_open_orders(self) -> list[Trade]:
        """Return all orders that have not yet reached a terminal state.

        Returns:
            List of open :class:`ib_async.Trade` objects.
        """
        return self._ib.openTrades()

    def get_fill_history(self) -> list[Fill]:
        """Return all fills from the current session.

        Returns:
            List of :class:`ib_async.Fill` objects.
        """
        return self._ib.fills()

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _on_order_status(self, trade: Trade) -> None:
        """Global callback invoked on every order-status transition.

        Logs the transition and, when an order reaches a terminal state,
        computes slippage against the originally submitted limit price.
        """
        status = trade.orderStatus
        self._log.info(
            "order_status_change",
            order_id=status.orderId,
            status=status.status,
            filled=status.filled,
            remaining=status.remaining,
            avg_fill_price=status.avgFillPrice,
            last_fill_price=status.lastFillPrice,
            perm_id=status.permId,
        )

        # Compute slippage once the order is fully filled.
        if status.status == OrderStatus.Filled:
            self._record_slippage(trade)

    def _on_trade_filled(self, trade: Trade) -> None:
        """Per-trade callback fired when the order is completely filled."""
        self._log.info(
            "trade_filled",
            order_id=trade.order.orderId,
            symbol=trade.contract.symbol,
            avg_fill_price=trade.orderStatus.avgFillPrice,
            filled=trade.orderStatus.filled,
        )

    def _on_trade_cancelled(self, trade: Trade) -> None:
        """Per-trade callback fired when the order is cancelled."""
        self._log.warning(
            "trade_cancelled",
            order_id=trade.order.orderId,
            symbol=trade.contract.symbol,
            filled=trade.orderStatus.filled,
            remaining=trade.orderStatus.remaining,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _record_slippage(self, trade: Trade) -> None:
        """Compute and log slippage for a filled order.

        Slippage is defined as the signed difference between the average
        fill price and the expected limit price at submission time.  For
        BUY orders positive slippage means the fill was worse (higher)
        than expected; for SELL orders positive slippage means worse
        (lower).
        """
        expected = self._expected_prices.pop(trade.order.orderId, None)
        if expected is None:
            return

        avg_fill = trade.orderStatus.avgFillPrice
        if trade.order.action.upper() == "BUY":
            slippage = avg_fill - expected
        else:
            slippage = expected - avg_fill

        self._log.info(
            "slippage_recorded",
            order_id=trade.order.orderId,
            expected_price=expected,
            avg_fill_price=avg_fill,
            slippage=round(slippage, 6),
            action=trade.order.action,
        )

    def _build_order_result(self, trade: Trade) -> OrderResult:
        """Construct an :class:`OrderResult` from a completed trade."""
        expected = self._expected_prices.get(trade.order.orderId, 0.0)
        avg_fill = trade.orderStatus.avgFillPrice

        if trade.order.action.upper() == "BUY":
            slippage = avg_fill - expected
        else:
            slippage = expected - avg_fill

        total_commission = sum(
            fill.commissionReport.commission
            for fill in trade.fills
            if fill.commissionReport.commission > 0
        )

        return OrderResult(
            order_id=trade.order.orderId,
            status=trade.orderStatus.status,
            filled_qty=int(trade.orderStatus.filled),
            avg_fill_price=avg_fill,
            commission=round(total_commission, 4),
            slippage=round(slippage, 6),
        )

    @staticmethod
    async def _wait_until_done(trade: Trade) -> None:
        """Block until *trade* reaches a terminal order status.

        Uses the trade's ``statusEvent`` to avoid busy-waiting.
        """
        if trade.isDone():
            return

        done_event = asyncio.Event()

        def _check(t: Trade) -> None:
            if t.isDone():
                done_event.set()

        trade.statusEvent += _check
        try:
            # Double-check in case the status changed between our guard
            # and the event registration.
            if trade.isDone():
                return
            await done_event.wait()
        finally:
            trade.statusEvent -= _check
