"""Account state, positions, P&L tracking, and margin analysis.

Wraps the ``ib_async`` account-related methods behind Pydantic models and
structured logging.  Provides a clean async interface for the rest of the
system to query account health, position details, and real-time profit and
loss without coupling to the raw IB API.

Usage::

    from src.broker.account import AccountManager

    mgr = AccountManager(ib)
    summary = await mgr.get_account_summary()
    print(summary.net_liquidation)
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    import structlog
    from ib_async import (
        IB,
        AccountValue,
        Contract,
        Order,
        OrderState,
    )
    from ib_async import PnL as IBPnL
    from ib_async import PnLSingle as IBPnLSingle
    from ib_async import PortfolioItem as IBPortfolioItem
    from ib_async import Position as IBPosition


from src.utils.logging import get_logger

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Account value tags requested from IBKR.
_SUMMARY_TAGS: dict[str, str] = {
    "NetLiquidation": "net_liquidation",
    "TotalCashValue": "total_cash",
    "BuyingPower": "buying_power",
    "ExcessLiquidity": "excess_liquidity",
    "FullMaintMarginReq": "maint_margin",
    "GrossPositionValue": "gross_position_value",
}

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class AccountSummary(BaseModel):
    """Snapshot of key account metrics."""

    net_liquidation: float = Field(
        default=0.0,
        description="Total account value including all assets and liabilities",
    )
    total_cash: float = Field(
        default=0.0, description="Total settled and unsettled cash balance"
    )
    buying_power: float = Field(
        default=0.0,
        description="Available buying power under Reg-T or portfolio margin",
    )
    excess_liquidity: float = Field(
        default=0.0,
        description="Funds available above maintenance margin requirements",
    )
    maint_margin: float = Field(
        default=0.0, description="Full maintenance margin requirement"
    )
    gross_position_value: float = Field(
        default=0.0, description="Absolute market value of all positions"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="UTC timestamp when the snapshot was taken",
    )


class PositionInfo(BaseModel):
    """Normalized representation of a single position."""

    con_id: int = Field(description="IBKR contract identifier")
    ticker: str = Field(description="Underlying symbol")
    sec_type: str = Field(description="Security type (STK, OPT, BAG, ...)")
    quantity: float = Field(description="Signed position size")
    avg_cost: float = Field(description="Average cost basis per unit")
    market_value: float = Field(
        default=0.0,
        description="Current market value of the position",
    )
    unrealized_pnl: float = Field(
        default=0.0,
        description="Unrealized profit or loss",
    )


class PortfolioItemInfo(BaseModel):
    """Enriched portfolio item with market price and realized P&L."""

    con_id: int = Field(description="IBKR contract identifier")
    ticker: str = Field(description="Underlying symbol")
    sec_type: str = Field(description="Security type")
    quantity: float = Field(description="Signed position size")
    avg_cost: float = Field(description="Average cost basis per unit")
    market_price: float = Field(description="Last known market price per unit")
    market_value: float = Field(description="Current market value")
    unrealized_pnl: float = Field(description="Unrealized profit or loss")
    realized_pnl: float = Field(description="Realized profit or loss")
    account: str = Field(description="IBKR account identifier")


class MarginImpact(BaseModel):
    """Projected margin and commission impact of a hypothetical order."""

    init_margin: float = Field(
        default=0.0,
        description="Initial margin requirement after the order",
    )
    maint_margin: float = Field(
        default=0.0,
        description="Maintenance margin requirement after the order",
    )
    equity_with_loan: float = Field(
        default=0.0,
        description="Equity with loan value after the order",
    )
    commission: float = Field(
        default=0.0,
        description="Estimated commission for the order",
    )


class PnLInfo(BaseModel):
    """Account-level profit and loss snapshot."""

    daily_pnl: float = Field(default=0.0, description="Realized + unrealized P&L today")
    unrealized_pnl: float = Field(
        default=0.0, description="Total unrealized P&L across all positions"
    )
    realized_pnl: float = Field(
        default=0.0, description="Total realized P&L for the session"
    )


class PnLSingleInfo(BaseModel):
    """Single-position profit and loss snapshot."""

    con_id: int = Field(description="IBKR contract identifier")
    daily_pnl: float = Field(default=0.0, description="Daily P&L for this position")
    unrealized_pnl: float = Field(
        default=0.0, description="Unrealized P&L for this position"
    )
    realized_pnl: float = Field(
        default=0.0, description="Realized P&L for this position"
    )
    position: int = Field(default=0, description="Current position size")
    value: float = Field(default=0.0, description="Current market value")


class AccountSnapshot(BaseModel):
    """Full account state suitable for persistence to PostgreSQL."""

    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="UTC timestamp of the snapshot",
    )
    net_liquidation: float = Field(default=0.0)
    buying_power: float = Field(default=0.0)
    excess_liquidity: float = Field(default=0.0)
    realized_pnl_day: float = Field(default=0.0)
    unrealized_pnl: float = Field(default=0.0)
    total_positions: int = Field(default=0)
    regime: str = Field(
        default="unknown",
        description="Current market regime label from the regime detector",
    )


# ---------------------------------------------------------------------------
# AccountManager
# ---------------------------------------------------------------------------


class AccountManager:
    """Queries and monitors IBKR account state, positions, and P&L.

    Args:
        ib: A connected ``ib_async.IB`` instance.
    """

    def __init__(self, ib: IB) -> None:
        self._ib: IB = ib
        self._log: structlog.stdlib.BoundLogger = get_logger("broker.account")
        self._pnl_subscription: IBPnL | None = None
        self._pnl_single_subscriptions: dict[int, IBPnLSingle] = {}
        self._account_id: str = ""

    # ------------------------------------------------------------------
    # Account data
    # ------------------------------------------------------------------

    async def get_account_summary(self) -> AccountSummary:
        """Retrieve a snapshot of key account metrics.

        Fetches account summary values from the IB Gateway and maps
        the requested tags into an :class:`AccountSummary` model.

        Returns:
            Populated :class:`AccountSummary`.
        """
        raw_values: list[AccountValue] = await self._ib.accountSummaryAsync()

        # Build a lookup keyed on (tag, currency=USD).
        value_map: dict[str, float] = {}
        for av in raw_values:
            if av.tag in _SUMMARY_TAGS and av.currency in ("USD", "BASE", ""):
                try:
                    value_map[_SUMMARY_TAGS[av.tag]] = float(av.value)
                except (ValueError, TypeError):
                    self._log.warning(
                        "unparseable_account_value",
                        tag=av.tag,
                        raw_value=av.value,
                    )

            # Capture the account ID for later PnL subscriptions.
            if av.account and not self._account_id:
                self._account_id = av.account

        summary = AccountSummary(
            net_liquidation=value_map.get("net_liquidation", 0.0),
            total_cash=value_map.get("total_cash", 0.0),
            buying_power=value_map.get("buying_power", 0.0),
            excess_liquidity=value_map.get("excess_liquidity", 0.0),
            maint_margin=value_map.get("maint_margin", 0.0),
            gross_position_value=value_map.get("gross_position_value", 0.0),
        )

        self._log.info(
            "account_summary_retrieved",
            net_liquidation=summary.net_liquidation,
            buying_power=summary.buying_power,
            excess_liquidity=summary.excess_liquidity,
        )
        return summary

    async def get_positions(self) -> list[PositionInfo]:
        """Return all current positions.

        Translates raw :class:`ib_async.Position` tuples into
        :class:`PositionInfo` Pydantic models.

        Returns:
            List of :class:`PositionInfo` for each held position.
        """
        raw_positions: list[IBPosition] = self._ib.positions()

        positions: list[PositionInfo] = []
        for pos in raw_positions:
            contract = pos.contract
            positions.append(
                PositionInfo(
                    con_id=contract.conId,
                    ticker=contract.symbol or contract.localSymbol or "",
                    sec_type=contract.secType or "",
                    quantity=pos.position,
                    avg_cost=pos.avgCost,
                    # Market value is not in Position; callers should use
                    # get_portfolio() for market-aware data.
                    market_value=0.0,
                    unrealized_pnl=0.0,
                )
            )

        self._log.debug(
            "positions_retrieved",
            count=len(positions),
        )
        return positions

    async def get_portfolio(self) -> list[PortfolioItemInfo]:
        """Return enriched portfolio items with live market data.

        Portfolio items include market price, market value, and both
        unrealized and realized P&L unlike the simpler
        :meth:`get_positions`.

        Returns:
            List of :class:`PortfolioItemInfo`.
        """
        raw_items: list[IBPortfolioItem] = self._ib.portfolio()

        items: list[PortfolioItemInfo] = []
        for pi in raw_items:
            contract = pi.contract
            items.append(
                PortfolioItemInfo(
                    con_id=contract.conId,
                    ticker=contract.symbol or contract.localSymbol or "",
                    sec_type=contract.secType or "",
                    quantity=pi.position,
                    avg_cost=pi.averageCost,
                    market_price=pi.marketPrice,
                    market_value=pi.marketValue,
                    unrealized_pnl=pi.unrealizedPNL,
                    realized_pnl=pi.realizedPNL,
                    account=pi.account,
                )
            )

        self._log.debug(
            "portfolio_retrieved",
            count=len(items),
        )
        return items

    async def get_pnl(self) -> PnLInfo:
        """Return account-level P&L.

        On the first call a PnL subscription is started with the IB
        Gateway.  Subsequent calls return the latest values from the
        live-updated subscription object.

        Returns:
            :class:`PnLInfo` with daily, unrealized, and realized P&L.
        """
        account = await self._resolve_account()

        if self._pnl_subscription is None:
            self._pnl_subscription = self._ib.reqPnL(account)
            self._log.info("pnl_subscription_started", account=account)

        pnl = self._pnl_subscription
        result = PnLInfo(
            daily_pnl=(pnl.dailyPnL if _is_valid_float(pnl.dailyPnL) else 0.0),
            unrealized_pnl=(
                pnl.unrealizedPnL if _is_valid_float(pnl.unrealizedPnL) else 0.0
            ),
            realized_pnl=(pnl.realizedPnL if _is_valid_float(pnl.realizedPnL) else 0.0),
        )

        self._log.debug(
            "pnl_retrieved",
            daily_pnl=result.daily_pnl,
            unrealized_pnl=result.unrealized_pnl,
            realized_pnl=result.realized_pnl,
        )
        return result

    async def get_position_pnl(self, con_id: int) -> PnLSingleInfo:
        """Return P&L for a single position identified by contract ID.

        A subscription is created on the first call for each ``con_id``
        and the live-updated object is reused on subsequent calls.

        Args:
            con_id: IBKR contract identifier.

        Returns:
            :class:`PnLSingleInfo` for the specified position.
        """
        account = await self._resolve_account()

        if con_id not in self._pnl_single_subscriptions:
            pnl_single = self._ib.reqPnLSingle(
                account=account,
                modelCode="",
                conId=con_id,
            )
            self._pnl_single_subscriptions[con_id] = pnl_single
            self._log.info(
                "pnl_single_subscription_started",
                account=account,
                con_id=con_id,
            )

        pnl = self._pnl_single_subscriptions[con_id]
        result = PnLSingleInfo(
            con_id=con_id,
            daily_pnl=(pnl.dailyPnL if _is_valid_float(pnl.dailyPnL) else 0.0),
            unrealized_pnl=(
                pnl.unrealizedPnL if _is_valid_float(pnl.unrealizedPnL) else 0.0
            ),
            realized_pnl=(pnl.realizedPnL if _is_valid_float(pnl.realizedPnL) else 0.0),
            position=pnl.position,
            value=(pnl.value if _is_valid_float(pnl.value) else 0.0),
        )

        self._log.debug(
            "position_pnl_retrieved",
            con_id=con_id,
            daily_pnl=result.daily_pnl,
            unrealized_pnl=result.unrealized_pnl,
            position=result.position,
        )
        return result

    # ------------------------------------------------------------------
    # Margin
    # ------------------------------------------------------------------

    async def calculate_margin_impact(
        self,
        contract: Contract,
        order: Order,
    ) -> MarginImpact:
        """Estimate the margin and commission impact of a hypothetical order.

        Uses IBKR's ``whatIfOrder`` facility which simulates the order
        without actually submitting it.

        Args:
            contract: The contract to evaluate.
            order: The order to simulate.

        Returns:
            :class:`MarginImpact` with projected margin requirements and
            estimated commission.
        """
        order_state: OrderState = await self._ib.whatIfOrderAsync(contract, order)
        numeric = order_state.numeric(digits=2)

        impact = MarginImpact(
            init_margin=(
                numeric.initMarginAfter if numeric.initMarginAfter is not None else 0.0
            ),
            maint_margin=(
                numeric.maintMarginAfter
                if numeric.maintMarginAfter is not None
                else 0.0
            ),
            equity_with_loan=(
                numeric.equityWithLoanAfter
                if numeric.equityWithLoanAfter is not None
                else 0.0
            ),
            commission=(numeric.commission if numeric.commission is not None else 0.0),
        )

        self._log.info(
            "margin_impact_calculated",
            symbol=contract.symbol,
            init_margin=impact.init_margin,
            maint_margin=impact.maint_margin,
            equity_with_loan=impact.equity_with_loan,
            commission=impact.commission,
        )
        return impact

    # ------------------------------------------------------------------
    # Monitoring
    # ------------------------------------------------------------------

    async def start_account_updates(self) -> None:
        """Subscribe to real-time account and portfolio updates.

        This triggers a streaming subscription so that :meth:`get_positions`,
        :meth:`get_portfolio`, and :meth:`get_account_summary` always reflect
        the latest data without extra round-trips.
        """
        account = await self._resolve_account()

        self._log.info("starting_account_updates", account=account)
        await self._ib.reqAccountUpdatesAsync(account)

        # Also subscribe to PnL if not already done.
        if self._pnl_subscription is None:
            self._pnl_subscription = self._ib.reqPnL(account)

        self._log.info("account_updates_started", account=account)

    async def take_snapshot(self, regime: str = "unknown") -> AccountSnapshot:
        """Capture the full account state for database persistence.

        Combines account summary values with live P&L and position count
        into an :class:`AccountSnapshot` model that maps directly to the
        ``account_snapshots`` PostgreSQL table.

        Args:
            regime: Current market regime label (e.g. ``"low_vol_trend"``).

        Returns:
            :class:`AccountSnapshot` ready for insertion.
        """
        summary = await self.get_account_summary()
        pnl = await self.get_pnl()
        positions = await self.get_positions()

        snapshot = AccountSnapshot(
            net_liquidation=summary.net_liquidation,
            buying_power=summary.buying_power,
            excess_liquidity=summary.excess_liquidity,
            realized_pnl_day=pnl.realized_pnl,
            unrealized_pnl=pnl.unrealized_pnl,
            total_positions=len(positions),
            regime=regime,
        )

        self._log.info(
            "account_snapshot_taken",
            net_liquidation=snapshot.net_liquidation,
            buying_power=snapshot.buying_power,
            realized_pnl_day=snapshot.realized_pnl_day,
            unrealized_pnl=snapshot.unrealized_pnl,
            total_positions=snapshot.total_positions,
            regime=snapshot.regime,
        )
        return snapshot

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _resolve_account(self) -> str:
        """Return the IBKR account identifier.

        If not yet known, triggers an account summary request to discover
        it.

        Returns:
            The IBKR account identifier string.

        Raises:
            RuntimeError: If no account could be discovered.
        """
        if self._account_id:
            return self._account_id

        # Trigger a summary to discover the account id.
        raw_values: list[AccountValue] = await self._ib.accountSummaryAsync()
        for av in raw_values:
            if av.account:
                self._account_id = av.account
                break

        if not self._account_id:
            # Fallback: check managed accounts from the wrapper.
            managed = getattr(self._ib.wrapper, "accounts", [])
            if managed:
                self._account_id = managed[0]

        if not self._account_id:
            raise RuntimeError(
                "Unable to determine IBKR account ID. "
                "Ensure the gateway is connected and account data is available."
            )

        self._log.info("account_id_resolved", account=self._account_id)
        return self._account_id


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _is_valid_float(value: float) -> bool:
    """Return True if *value* is a finite number (not NaN or inf).

    The IB API initializes many float fields to ``float('nan')`` before
    data arrives.  This helper guards against passing those through.
    """
    import math

    return not (math.isnan(value) or math.isinf(value))
