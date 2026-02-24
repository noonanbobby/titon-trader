"""Contract builders for Interactive Brokers using ib_async.

Provides :class:`ContractFactory` — a high-level helper that wraps the
low-level ``ib_async`` contract objects and qualification workflow.  Every
public method qualifies contracts via the IB Gateway so that the returned
objects are guaranteed to carry a valid ``conId`` and can be used directly
for market-data subscriptions or order submission.

Usage::

    from ib_async import IB
    from src.broker.contracts import ContractFactory

    ib = IB()
    await ib.connectAsync("127.0.0.1", 4002, clientId=1)

    factory = ContractFactory(ib)
    stock = await factory.create_stock("AAPL")
    option = await factory.create_option("AAPL", "20250321", 150.0, "C")
    spread = await factory.build_spread("AAPL", [
        {"action": "BUY",  "expiry": "20250321", "strike": 150.0, "right": "C"},
        {"action": "SELL", "expiry": "20250321", "strike": 160.0, "right": "C"},
    ])
"""

from __future__ import annotations

import asyncio
from enum import StrEnum
from typing import Literal

import structlog
from ib_async import (
    IB,
    Bag,
    ComboLeg,
    Contract,
    Index,
    Option,
    OptionChain,
    Stock,
    Ticker,
)
from pydantic import BaseModel, Field, field_validator

logger: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class OptionRight(StrEnum):
    """Valid option right values accepted by IB."""

    CALL = "C"
    PUT = "P"


class SpreadLeg(BaseModel):
    """Description of a single leg in a multi-leg spread order.

    Attributes:
        action: Whether this leg is bought or sold.
        expiry: Option expiration date in ``YYYYMMDD`` format.
        strike: Strike price of the option.
        right: ``C`` for call, ``P`` for put.
        ratio: Number of contracts per unit of the spread (default 1).
    """

    action: Literal["BUY", "SELL"]
    expiry: str = Field(
        ...,
        min_length=8,
        max_length=8,
        description="Expiry in YYYYMMDD format",
    )
    strike: float = Field(..., gt=0, description="Strike price")
    right: Literal["C", "P"]
    ratio: int = Field(default=1, ge=1, description="Contract ratio per spread unit")

    @field_validator("expiry")
    @classmethod
    def _validate_expiry_format(cls, v: str) -> str:
        """Ensure expiry is a plausible YYYYMMDD date string."""
        if not v.isdigit():
            raise ValueError(f"expiry must be numeric YYYYMMDD, got '{v}'")
        year = int(v[:4])
        month = int(v[4:6])
        day = int(v[6:8])
        if not (2000 <= year <= 2100):
            raise ValueError(f"expiry year out of range: {year}")
        if not (1 <= month <= 12):
            raise ValueError(f"expiry month out of range: {month}")
        if not (1 <= day <= 31):
            raise ValueError(f"expiry day out of range: {day}")
        return v


class OptionChainParams(BaseModel):
    """Structured representation of option chain parameters returned by IB.

    Maps to the ``ib_async.OptionChain`` dataclass but uses Pydantic for
    validation, serialisation, and immutability.

    Attributes:
        exchange: The exchange where the options are traded.
        underlying_con_id: The ``conId`` of the underlying instrument.
        trading_class: The IB trading-class identifier.
        multiplier: Contract multiplier (usually ``"100"``).
        expirations: Set of available expiration dates (``YYYYMMDD`` strings).
        strikes: Set of available strike prices.
    """

    exchange: str
    underlying_con_id: int
    trading_class: str
    multiplier: str
    expirations: set[str]
    strikes: set[float]

    model_config = {"frozen": True}


# ---------------------------------------------------------------------------
# Rate-limit helper
# ---------------------------------------------------------------------------

# IB Gateway allows a maximum of ~50 messages per second.  We insert a short
# sleep between rapid-fire requests so we stay comfortably below that ceiling.
_IB_REQUEST_PAUSE_SECS: float = 0.05


# ---------------------------------------------------------------------------
# Contract factory
# ---------------------------------------------------------------------------


class ContractFactory:
    """High-level builder for IB contracts.

    All methods are async and perform contract qualification through the live
    IB Gateway connection so that returned objects always carry a valid
    ``conId``.

    Args:
        ib: A connected :class:`ib_async.IB` instance.
    """

    def __init__(self, ib: IB) -> None:
        self._ib = ib
        self._log = logger.bind(component="ContractFactory")

    # ------------------------------------------------------------------
    # Stock
    # ------------------------------------------------------------------

    async def create_stock(
        self,
        ticker: str,
        exchange: str = "SMART",
        currency: str = "USD",
    ) -> Stock:
        """Create and qualify a stock contract.

        Args:
            ticker: The stock symbol (e.g. ``"AAPL"``).
            exchange: Destination exchange (default ``"SMART"`` for best routing).
            currency: Underlying currency (default ``"USD"``).

        Returns:
            A fully qualified :class:`ib_async.Stock` contract.

        Raises:
            ValueError: If the contract cannot be qualified by IB Gateway.
        """
        contract = Stock(symbol=ticker, exchange=exchange, currency=currency)
        self._log.debug("qualifying_stock", ticker=ticker, exchange=exchange)

        qualified = await self._ib.qualifyContractsAsync(contract)
        if qualified[0] is None:
            raise ValueError(
                f"Failed to qualify stock contract: "
                f"ticker={ticker}, exchange={exchange}, currency={currency}"
            )

        self._log.info(
            "stock_qualified",
            ticker=ticker,
            con_id=contract.conId,
            exchange=contract.exchange,
        )
        return contract

    # ------------------------------------------------------------------
    # Index
    # ------------------------------------------------------------------

    async def create_index(
        self,
        ticker: str,
        exchange: str = "CBOE",
        currency: str = "USD",
    ) -> Index:
        """Create and qualify an index contract (e.g. VIX, SPX).

        Args:
            ticker: The index symbol (e.g. ``"VIX"``, ``"SPX"``).
            exchange: Destination exchange (default ``"CBOE"``).
            currency: Underlying currency (default ``"USD"``).

        Returns:
            A fully qualified :class:`ib_async.Index` contract.

        Raises:
            ValueError: If the contract cannot be qualified by IB Gateway.
        """
        contract = Index(symbol=ticker, exchange=exchange, currency=currency)
        self._log.debug("qualifying_index", ticker=ticker, exchange=exchange)

        qualified = await self._ib.qualifyContractsAsync(contract)
        if qualified[0] is None:
            raise ValueError(
                f"Failed to qualify index contract: "
                f"ticker={ticker}, exchange={exchange}, currency={currency}"
            )

        self._log.info(
            "index_qualified",
            ticker=ticker,
            con_id=contract.conId,
            exchange=contract.exchange,
        )
        return contract

    # ------------------------------------------------------------------
    # Option
    # ------------------------------------------------------------------

    async def create_option(
        self,
        ticker: str,
        expiry: str,
        strike: float,
        right: str,
        exchange: str = "SMART",
        currency: str = "USD",
        multiplier: str = "100",
    ) -> Option:
        """Create and qualify a single option contract.

        Args:
            ticker: The underlying symbol (e.g. ``"AAPL"``).
            expiry: Expiration date in ``YYYYMMDD`` format.
            strike: Strike price.
            right: ``"C"`` for call or ``"P"`` for put.
            exchange: Destination exchange (default ``"SMART"``).
            currency: Underlying currency (default ``"USD"``).
            multiplier: Contract multiplier (default ``"100"``).

        Returns:
            A fully qualified :class:`ib_async.Option` contract.

        Raises:
            ValueError: If the contract cannot be qualified by IB Gateway.
        """
        contract = Option(
            symbol=ticker,
            lastTradeDateOrContractMonth=expiry,
            strike=strike,
            right=right,
            exchange=exchange,
            multiplier=multiplier,
            currency=currency,
        )
        self._log.debug(
            "qualifying_option",
            ticker=ticker,
            expiry=expiry,
            strike=strike,
            right=right,
        )

        qualified = await self._ib.qualifyContractsAsync(contract)
        if qualified[0] is None:
            raise ValueError(
                f"Failed to qualify option contract: "
                f"ticker={ticker}, expiry={expiry}, strike={strike}, right={right}"
            )

        self._log.info(
            "option_qualified",
            ticker=ticker,
            expiry=expiry,
            strike=strike,
            right=right,
            con_id=contract.conId,
        )
        return contract

    # ------------------------------------------------------------------
    # Combo / BAG
    # ------------------------------------------------------------------

    async def create_combo(
        self,
        ticker: str,
        legs: list[ComboLeg],
        exchange: str = "SMART",
        currency: str = "USD",
    ) -> Contract:
        """Create a BAG (combo) contract from pre-built combo legs.

        This is the low-level combo builder.  Each :class:`ib_async.ComboLeg`
        must already have a valid ``conId`` (obtained by qualifying the
        individual option contracts first).

        Args:
            ticker: The underlying symbol.
            legs: List of :class:`ib_async.ComboLeg` objects.
            exchange: Destination exchange (default ``"SMART"``).
            currency: Currency (default ``"USD"``).

        Returns:
            A :class:`ib_async.Contract` with ``secType="BAG"`` and the
            given combo legs attached.

        Raises:
            ValueError: If *legs* is empty.
        """
        if not legs:
            raise ValueError("Cannot create combo contract with zero legs")

        combo = Bag(
            symbol=ticker,
            exchange=exchange,
            currency=currency,
            comboLegs=legs,
        )

        self._log.info(
            "combo_created",
            ticker=ticker,
            num_legs=len(legs),
            leg_con_ids=[leg.conId for leg in legs],
        )
        return combo

    # ------------------------------------------------------------------
    # build_spread — high-level helper
    # ------------------------------------------------------------------

    async def build_spread(
        self,
        ticker: str,
        legs: list[dict[str, str | float | int]],
        exchange: str = "SMART",
        currency: str = "USD",
    ) -> Contract:
        """Build a multi-leg spread from simple leg descriptors.

        This is the preferred high-level method for constructing spreads such
        as bull call spreads, iron condors, butterflies, etc.  It qualifies
        each leg, extracts the ``conId``, assembles :class:`ib_async.ComboLeg`
        objects, and returns a ready-to-trade BAG contract.

        Args:
            ticker: The underlying symbol.
            legs: List of dicts, each containing the keys expected by
                :class:`SpreadLeg` (``action``, ``expiry``, ``strike``,
                ``right``, and optionally ``ratio``).
            exchange: Destination exchange (default ``"SMART"``).
            currency: Currency (default ``"USD"``).

        Returns:
            A BAG :class:`ib_async.Contract` representing the spread.

        Raises:
            ValueError: If any individual leg fails to qualify.

        Example::

            spread = await factory.build_spread("AAPL", [
                {"action": "BUY",  "expiry": "20250321", "strike": 150.0, "right": "C"},
                {"action": "SELL", "expiry": "20250321", "strike": 160.0, "right": "C"},
            ])
        """
        # Validate all leg descriptors up-front via Pydantic
        parsed_legs: list[SpreadLeg] = [SpreadLeg(**leg) for leg in legs]  # type: ignore[arg-type]

        self._log.info(
            "building_spread",
            ticker=ticker,
            num_legs=len(parsed_legs),
            legs=[
                {
                    "action": pl.action,
                    "strike": pl.strike,
                    "right": pl.right,
                    "expiry": pl.expiry,
                    "ratio": pl.ratio,
                }
                for pl in parsed_legs
            ],
        )

        # Qualify all option legs concurrently.  We build the Option objects
        # first, then qualify them in a single batch call to minimise round-
        # trips to the gateway.
        option_contracts: list[Option] = []
        for pl in parsed_legs:
            opt = Option(
                symbol=ticker,
                lastTradeDateOrContractMonth=pl.expiry,
                strike=pl.strike,
                right=pl.right,
                exchange=exchange,
                multiplier="100",
                currency=currency,
            )
            option_contracts.append(opt)

        qualified = await self._ib.qualifyContractsAsync(*option_contracts)

        # Verify every leg qualified successfully
        combo_legs: list[ComboLeg] = []
        for idx, (pl, qual, opt) in enumerate(
            zip(parsed_legs, qualified, option_contracts, strict=False)
        ):
            if qual is None:
                raise ValueError(
                    f"Failed to qualify spread leg {idx}: "
                    f"ticker={ticker}, expiry={pl.expiry}, "
                    f"strike={pl.strike}, right={pl.right}"
                )
            combo_legs.append(
                ComboLeg(
                    conId=opt.conId,
                    ratio=pl.ratio,
                    action=pl.action,
                    exchange=exchange,
                )
            )

        combo = await self.create_combo(
            ticker=ticker,
            legs=combo_legs,
            exchange=exchange,
            currency=currency,
        )

        self._log.info(
            "spread_built",
            ticker=ticker,
            num_legs=len(combo_legs),
            leg_details=[
                {
                    "con_id": cl.conId,
                    "action": cl.action,
                    "ratio": cl.ratio,
                }
                for cl in combo_legs
            ],
        )
        return combo

    # ------------------------------------------------------------------
    # Option chain parameters
    # ------------------------------------------------------------------

    async def get_option_chain_params(
        self,
        ticker: str,
    ) -> list[OptionChainParams]:
        """Retrieve available option chain parameters for an underlying.

        Calls ``reqSecDefOptParams`` which has **no rate limit** — it is
        safe to call this frequently without throttling.

        The underlying stock is qualified first so that the correct
        ``conId`` is passed to the request.

        Args:
            ticker: The underlying symbol (e.g. ``"AAPL"``).

        Returns:
            A list of :class:`OptionChainParams` — one per exchange /
            trading-class combination.

        Raises:
            ValueError: If the underlying stock cannot be qualified.
        """
        stock = await self.create_stock(ticker)

        self._log.debug(
            "requesting_option_chain_params",
            ticker=ticker,
            con_id=stock.conId,
        )

        chains: list[OptionChain] = await self._ib.reqSecDefOptParamsAsync(
            underlyingSymbol=ticker,
            futFopExchange="",
            underlyingSecType="STK",
            underlyingConId=stock.conId,
        )

        result: list[OptionChainParams] = []
        for chain in chains:
            result.append(
                OptionChainParams(
                    exchange=chain.exchange,
                    underlying_con_id=chain.underlyingConId,
                    trading_class=chain.tradingClass,
                    multiplier=chain.multiplier,
                    expirations=set(chain.expirations),
                    strikes=set(chain.strikes),
                )
            )

        self._log.info(
            "option_chain_params_received",
            ticker=ticker,
            num_chains=len(result),
            exchanges=[p.exchange for p in result],
        )
        return result

    # ------------------------------------------------------------------
    # Available strikes
    # ------------------------------------------------------------------

    async def get_available_strikes(
        self,
        ticker: str,
        expiry: str,
        right: str = "",
    ) -> list[float]:
        """Get all available strikes for a given ticker and expiration.

        Fetches the option chain parameters from IB and filters to the
        requested expiration.  Strikes are returned sorted in ascending
        order.

        Args:
            ticker: The underlying symbol.
            expiry: Expiration date in ``YYYYMMDD`` format.
            right: Optional filter — ``"C"`` for calls only, ``"P"`` for puts
                only, or ``""`` for both (default).

        Returns:
            Sorted list of available strike prices.

        Raises:
            ValueError: If the underlying cannot be qualified or the expiry
                is not found in any chain.
        """
        chains = await self.get_option_chain_params(ticker)

        all_strikes: set[float] = set()
        found_expiry = False

        for chain in chains:
            if expiry in chain.expirations:
                found_expiry = True
                all_strikes.update(chain.strikes)

        if not found_expiry:
            raise ValueError(
                f"Expiry {expiry} not found in option chains for {ticker}. "
                f"Available expirations (first chain): "
                f"{sorted(chains[0].expirations)[:10] if chains else '[]'}..."
            )

        sorted_strikes = sorted(all_strikes)

        self._log.info(
            "available_strikes",
            ticker=ticker,
            expiry=expiry,
            right=right or "ALL",
            num_strikes=len(sorted_strikes),
            min_strike=sorted_strikes[0] if sorted_strikes else None,
            max_strike=sorted_strikes[-1] if sorted_strikes else None,
        )
        return sorted_strikes

    # ------------------------------------------------------------------
    # Filter strikes by delta
    # ------------------------------------------------------------------

    async def filter_strikes_by_delta(
        self,
        ticker: str,
        expiry: str,
        right: str,
        target_delta: float,
        delta_tolerance: float = 0.05,
        max_candidates: int = 20,
    ) -> list[Option]:
        """Find option contracts near a target delta.

        The method:
        1. Retrieves all available strikes for the given expiry.
        2. Selects up to *max_candidates* strikes centred around the target
           delta (estimated by strike proximity to spot; refined with live
           Greeks).
        3. Qualifies the candidate option contracts.
        4. Requests a snapshot of market data to obtain live Greeks.
        5. Filters to strikes whose absolute delta falls within
           *target_delta* +/- *delta_tolerance*.
        6. Returns the matching options sorted by proximity to *target_delta*.

        Args:
            ticker: The underlying symbol.
            expiry: Expiration date in ``YYYYMMDD`` format.
            right: ``"C"`` for call or ``"P"`` for put.
            target_delta: The desired absolute delta value (e.g. ``0.30``).
            delta_tolerance: Acceptable deviation from *target_delta*
                (default ``0.05``).
            max_candidates: Maximum number of strikes to qualify and request
                data for (default 20).  Keeps IB message traffic manageable.

        Returns:
            List of qualified :class:`ib_async.Option` contracts sorted by
            ascending distance of their delta from *target_delta*.

        Raises:
            ValueError: If the underlying or option contracts cannot be
                qualified, or if no strikes match the delta filter.
        """
        if right not in ("C", "P"):
            raise ValueError(f"right must be 'C' or 'P', got '{right}'")

        self._log.info(
            "filtering_strikes_by_delta",
            ticker=ticker,
            expiry=expiry,
            right=right,
            target_delta=target_delta,
            delta_tolerance=delta_tolerance,
        )

        # 1. Get all available strikes for this expiry
        all_strikes = await self.get_available_strikes(ticker, expiry, right)

        if not all_strikes:
            raise ValueError(
                f"No strikes available for {ticker} expiry={expiry} right={right}"
            )

        # 2. Get the current underlying price to estimate which strikes are
        #    near our target delta.  Request a snapshot of the stock.
        stock = await self.create_stock(ticker)
        stock_ticker: Ticker = self._ib.reqMktData(stock, snapshot=True)

        # Allow some time for the snapshot to populate
        await asyncio.sleep(2.0)

        spot_price: float | None = (
            stock_ticker.marketPrice()
            if hasattr(stock_ticker, "marketPrice")
            else stock_ticker.last
        )

        # Clean up the stock data subscription
        self._ib.cancelMktData(stock)

        if spot_price is None or spot_price != spot_price:  # NaN check
            raise ValueError(
                f"Unable to determine spot price for {ticker}. "
                f"Ensure market data subscriptions are active."
            )

        self._log.debug(
            "spot_price_obtained",
            ticker=ticker,
            spot_price=spot_price,
        )

        # 3. Select candidate strikes around the estimated delta region.
        #    For calls, lower strikes have higher delta; for puts, higher
        #    strikes have higher absolute delta.  We pick strikes centred
        #    around the spot and take the nearest *max_candidates*.
        candidate_strikes = _select_candidate_strikes(
            all_strikes=all_strikes,
            spot_price=spot_price,
            max_candidates=max_candidates,
        )

        # 4. Build and qualify candidate option contracts
        candidates: list[Option] = []
        for strike in candidate_strikes:
            opt = Option(
                symbol=ticker,
                lastTradeDateOrContractMonth=expiry,
                strike=strike,
                right=right,
                exchange="SMART",
                multiplier="100",
                currency="USD",
            )
            candidates.append(opt)

        qualified = await self._ib.qualifyContractsAsync(*candidates)

        # Keep only successfully qualified contracts
        valid_options: list[Option] = []
        for opt, qual in zip(candidates, qualified, strict=False):
            if qual is not None:
                valid_options.append(opt)

        if not valid_options:
            raise ValueError(
                f"No option contracts could be qualified for "
                f"{ticker} expiry={expiry} right={right}"
            )

        self._log.debug(
            "candidates_qualified",
            ticker=ticker,
            num_qualified=len(valid_options),
            num_attempted=len(candidates),
        )

        # 5. Request market data snapshots for all valid options to get Greeks.
        #    We use generic tick 106 (impliedVolatility) to ensure model
        #    Greeks are populated.
        tickers: list[Ticker] = []
        for opt in valid_options:
            t = self._ib.reqMktData(opt, genericTickList="106", snapshot=True)
            tickers.append(t)
            await asyncio.sleep(_IB_REQUEST_PAUSE_SECS)

        # Wait for snapshots to populate
        await asyncio.sleep(3.0)

        # 6. Filter by delta proximity and collect results
        matching: list[tuple[float, Option]] = []
        for opt, tkr in zip(valid_options, tickers, strict=False):
            greeks = tkr.modelGreeks
            if greeks is None or greeks.delta is None:
                self._log.debug(
                    "no_greeks_for_strike",
                    ticker=ticker,
                    strike=opt.strike,
                    right=right,
                )
                continue

            option_delta = abs(greeks.delta)
            distance = abs(option_delta - target_delta)

            if distance <= delta_tolerance:
                matching.append((distance, opt))
                self._log.debug(
                    "delta_match",
                    ticker=ticker,
                    strike=opt.strike,
                    right=right,
                    delta=greeks.delta,
                    abs_delta=option_delta,
                    distance=distance,
                )

        # Cancel market data for all options
        for opt in valid_options:
            self._ib.cancelMktData(opt)

        if not matching:
            raise ValueError(
                f"No strikes found within delta_tolerance={delta_tolerance} "
                f"of target_delta={target_delta} for {ticker} "
                f"expiry={expiry} right={right}. "
                f"Try increasing delta_tolerance or max_candidates."
            )

        # Sort by proximity to target delta (closest first)
        matching.sort(key=lambda item: item[0])
        result = [opt for _, opt in matching]

        self._log.info(
            "delta_filter_complete",
            ticker=ticker,
            expiry=expiry,
            right=right,
            target_delta=target_delta,
            num_matches=len(result),
            best_strike=result[0].strike if result else None,
        )
        return result


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _select_candidate_strikes(
    all_strikes: list[float],
    spot_price: float,
    max_candidates: int,
) -> list[float]:
    """Select a window of strikes centred around the spot price.

    Picks the *max_candidates* strikes whose distance from *spot_price* is
    smallest.  This gives a good spread of ITM / ATM / OTM options to
    evaluate for delta filtering.

    Args:
        all_strikes: Sorted list of all available strikes.
        spot_price: Current price of the underlying.
        max_candidates: Maximum number of strikes to return.

    Returns:
        List of selected strike prices, sorted ascending.
    """
    if len(all_strikes) <= max_candidates:
        return list(all_strikes)

    # Sort by absolute distance from spot, then take the closest N
    by_distance = sorted(all_strikes, key=lambda s: abs(s - spot_price))
    selected = sorted(by_distance[:max_candidates])
    return selected
