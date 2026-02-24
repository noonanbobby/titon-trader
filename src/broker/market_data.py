"""Real-time market data streaming and options chain management.

Provides the :class:`MarketDataManager` which wraps the ``ib_async`` market
data APIs with rate limiting, subscription tracking, and structured Pydantic
models for downstream consumers.

Usage::

    from ib_async import IB
    from src.broker.contracts import ContractFactory
    from src.broker.market_data import MarketDataManager

    ib = IB()
    await ib.connectAsync("127.0.0.1", 4002, clientId=1)
    factory = ContractFactory(ib)
    mdm = MarketDataManager(ib, factory)
    ticker = await mdm.subscribe_ticker("AAPL")
"""

from __future__ import annotations

import asyncio
import json
import math
from datetime import UTC, date, datetime
from typing import TYPE_CHECKING, Any

import pandas as pd
import structlog
from ib_async import IB, Option, Ticker
from ib_async.contract import Contract
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from ib_async.objects import OptionComputation

    from src.broker.contracts import ContractFactory, OptionChainParams

logger: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# IB API rate-limit constants
# ---------------------------------------------------------------------------
IB_RATE_LIMIT_DELAY: float = 0.05  # 50 msg/sec ceiling → 20 ms per message
MAX_CONCURRENT_STREAMS: int = 100  # IB default concurrent data lines
GREEKS_SETTLE_SECONDS: float = 2.5  # time to wait for model greeks to populate
SNAPSHOT_SETTLE_SECONDS: float = 1.5  # time to wait for snapshot to populate
HISTORICAL_IV_WHAT_TO_SHOW: str = "OPTION_IMPLIED_VOLATILITY"

# Generic tick IDs requested alongside regular market data.
# 100 = option volume, 101 = option open interest, 104 = historical vol,
# 106 = implied volatility, 165 = misc stats, 221 = mark price
STOCK_GENERIC_TICKS: str = "100,101,104,106,165,221"
OPTION_GENERIC_TICKS: str = "100,101,104,106,221"

# CBOE indices that must be fetched as Index contracts, not Stock.
_INDEX_SYMBOLS: frozenset[str] = frozenset(
    {"VIX", "VXN", "OVX", "GVZ", "VVIX", "SPX", "NDX", "RUT", "DJX"}
)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------
class OptionGreeks(BaseModel):
    """Greeks snapshot for a single option contract."""

    con_id: int = Field(description="IB contract identifier")
    symbol: str = Field(default="", description="Underlying symbol")
    expiry: str = Field(default="", description="Expiration date YYYYMMDD")
    strike: float = Field(default=0.0, description="Strike price")
    right: str = Field(default="", description="C or P")
    delta: float = Field(default=0.0, description="Option delta")
    gamma: float = Field(default=0.0, description="Option gamma")
    theta: float = Field(default=0.0, description="Option theta (daily)")
    vega: float = Field(default=0.0, description="Option vega")
    implied_vol: float = Field(default=0.0, description="Implied volatility")
    und_price: float = Field(
        default=0.0,
        description="Underlying price at time of calc",
    )
    mid_price: float = Field(default=0.0, description="Mid price of the option")


class MarketSnapshot(BaseModel):
    """Point-in-time market data snapshot."""

    ticker: str = Field(description="Ticker symbol")
    bid: float = Field(default=float("nan"), description="Best bid price")
    ask: float = Field(default=float("nan"), description="Best ask price")
    last: float = Field(default=float("nan"), description="Last traded price")
    volume: float = Field(default=float("nan"), description="Volume traded today")
    mid: float = Field(default=float("nan"), description="Midpoint of bid/ask")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(tz=UTC),
        description="Snapshot capture time (UTC)",
    )


class IVSurfacePoint(BaseModel):
    """Single point on the IV surface."""

    iv: float = Field(default=0.0, description="Implied volatility")
    delta: float = Field(default=0.0, description="Delta")
    gamma: float = Field(default=0.0, description="Gamma")
    theta: float = Field(default=0.0, description="Theta")
    vega: float = Field(default=0.0, description="Vega")


class IVSurface(BaseModel):
    """Implied volatility surface for an underlying across strikes and expirations."""

    ticker: str = Field(description="Underlying ticker symbol")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(tz=UTC),
        description="Surface capture time (UTC)",
    )
    und_price: float = Field(default=0.0, description="Underlying spot price")
    data: dict[str, dict[str, IVSurfacePoint]] = Field(
        default_factory=dict,
        description="Nested dict: expiry -> strike -> IVSurfacePoint",
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _safe_float(val: Any, default: float = 0.0) -> float:
    """Convert a value to float, returning *default* for None / NaN / inf."""
    if val is None:
        return default
    try:
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            return default
        return f
    except (TypeError, ValueError):
        return default


def _expiry_to_date(expiry_str: str) -> date:
    """Parse a YYYYMMDD expiration string into a :class:`date`."""
    return datetime.strptime(expiry_str, "%Y%m%d").date()


def _dte(expiry_str: str, ref: date | None = None) -> int:
    """Return days-to-expiration from *ref* (defaults to today)."""
    ref = ref or date.today()
    return (_expiry_to_date(expiry_str) - ref).days


def _pick_smart_chain(chains: list[OptionChainParams]) -> OptionChainParams:
    """Select the SMART-exchange chain, falling back to the first available.

    Args:
        chains: List of chain parameter sets returned by
            :meth:`ContractFactory.get_option_chain_params`.

    Returns:
        The :class:`OptionChainParams` for the SMART exchange, or the
        first entry if SMART is not present.
    """
    for chain in chains:
        if chain.exchange == "SMART":
            return chain
    return chains[0]


# ---------------------------------------------------------------------------
# MarketDataManager
# ---------------------------------------------------------------------------
class MarketDataManager:
    """High-level market data manager wrapping ``ib_async.IB``.

    Handles real-time streaming subscriptions, options chain retrieval,
    Greeks extraction, IV surface construction, and historical IV data.
    All IB API calls are rate-limited to stay within the 50-msg/sec ceiling.

    Args:
        ib: A connected ``ib_async.IB`` instance.
        contract_factory: A :class:`ContractFactory` used to build and
            qualify contracts.
        redis_client: Optional ``redis.asyncio.Redis`` instance.  When
            provided, incoming ticks are published to the
            ``market_data:{ticker}`` Redis Pub/Sub channel.
    """

    def __init__(
        self,
        ib: IB,
        contract_factory: ContractFactory,
        redis_client: Any | None = None,
    ) -> None:
        self._ib: IB = ib
        self._factory: ContractFactory = contract_factory
        self._redis: Any | None = redis_client

        # symbol -> (Contract, Ticker) for active streaming subscriptions
        self._subscriptions: dict[str, tuple[Contract, Ticker]] = {}

        # conId -> (Contract, Ticker) for active option data subscriptions
        self._option_subscriptions: dict[int, tuple[Contract, Ticker]] = {}

        self._log = logger.bind(component="MarketDataManager")

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def subscription_count(self) -> int:
        """Return the number of active streaming subscriptions."""
        return len(self._subscriptions) + len(self._option_subscriptions)

    @property
    def subscribed_tickers(self) -> list[str]:
        """Return the list of currently subscribed ticker symbols."""
        return list(self._subscriptions.keys())

    # ------------------------------------------------------------------
    # Real-time quote subscriptions
    # ------------------------------------------------------------------
    async def subscribe_ticker(self, ticker: str) -> Ticker:
        """Subscribe to real-time streaming market data for a stock.

        Creates a stock contract via the contract factory, requests
        streaming market data from IB Gateway, registers a tick callback,
        and stores the subscription internally.

        Args:
            ticker: The stock symbol (e.g. ``"AAPL"``).

        Returns:
            The ``ib_async.Ticker`` object that will be continuously
            updated with live quotes.

        Raises:
            ValueError: If the contract cannot be qualified.
            RuntimeError: If the maximum concurrent stream limit is reached.
        """
        if ticker in self._subscriptions:
            self._log.debug("already_subscribed", ticker=ticker)
            return self._subscriptions[ticker][1]

        if self.subscription_count >= MAX_CONCURRENT_STREAMS:
            raise RuntimeError(
                f"Maximum concurrent stream limit ({MAX_CONCURRENT_STREAMS}) "
                f"reached.  Unsubscribe from existing streams before adding new ones."
            )

        # create_stock raises ValueError if qualification fails
        contract = await self._factory.create_stock(ticker)

        ib_ticker: Ticker = self._ib.reqMktData(
            contract,
            genericTickList=STOCK_GENERIC_TICKS,
            snapshot=False,
            regulatorySnapshot=False,
        )

        ib_ticker.updateEvent += self._tick_callback

        self._subscriptions[ticker] = (contract, ib_ticker)

        self._log.info(
            "subscribed_ticker",
            ticker=ticker,
            con_id=contract.conId,
            active_streams=self.subscription_count,
        )

        return ib_ticker

    async def subscribe_tickers(self, tickers: list[str]) -> dict[str, Ticker]:
        """Subscribe to real-time streaming data for multiple stocks.

        Requests are spaced by :data:`IB_RATE_LIMIT_DELAY` to respect
        the IB Gateway 50-msg/sec rate limit.

        Args:
            tickers: List of stock symbols to subscribe.

        Returns:
            Dict mapping each ticker symbol to its ``ib_async.Ticker``.
        """
        result: dict[str, Ticker] = {}
        for symbol in tickers:
            try:
                ib_ticker = await self.subscribe_ticker(symbol)
                result[symbol] = ib_ticker
            except (ValueError, RuntimeError) as exc:
                self._log.warning(
                    "subscribe_failed",
                    ticker=symbol,
                    error=str(exc),
                )
            await asyncio.sleep(IB_RATE_LIMIT_DELAY)
        return result

    async def unsubscribe_ticker(self, ticker: str) -> None:
        """Cancel streaming market data for a single ticker.

        Args:
            ticker: The stock symbol to unsubscribe.
        """
        entry = self._subscriptions.pop(ticker, None)
        if entry is None:
            self._log.debug("not_subscribed", ticker=ticker)
            return

        contract, ib_ticker = entry
        ib_ticker.updateEvent -= self._tick_callback
        self._ib.cancelMktData(contract)

        self._log.info(
            "unsubscribed_ticker",
            ticker=ticker,
            active_streams=self.subscription_count,
        )

    async def unsubscribe_all(self) -> None:
        """Cancel all active market data subscriptions (stocks and options)."""
        symbols = list(self._subscriptions.keys())
        for symbol in symbols:
            await self.unsubscribe_ticker(symbol)

        con_ids = list(self._option_subscriptions.keys())
        for con_id in con_ids:
            entry = self._option_subscriptions.pop(con_id, None)
            if entry is not None:
                contract, ib_ticker = entry
                ib_ticker.updateEvent -= self._tick_callback
                self._ib.cancelMktData(contract)

        self._log.info(
            "unsubscribed_all",
            cleared_stocks=len(symbols),
            cleared_options=len(con_ids),
        )

    # ------------------------------------------------------------------
    # Options chain
    # ------------------------------------------------------------------
    async def get_options_chain(
        self,
        ticker: str,
        min_dte: int = 20,
        max_dte: int = 60,
        num_strikes: int = 10,
    ) -> list[Option]:
        """Retrieve a filtered set of qualified option contracts.

        Steps:
        1. Fetch option chain parameters via ``reqSecDefOptParams``.
        2. Filter expirations to those within *min_dte* .. *max_dte*.
        3. Obtain the current underlying price.
        4. Select *num_strikes* strikes nearest the money on each side.
        5. Build ``Option`` contracts for every (expiry, strike, right).
        6. Qualify all contracts via ``qualifyContractsAsync`` in rate-
           limited batches.

        Args:
            ticker: Underlying symbol.
            min_dte: Minimum days to expiration (inclusive).
            max_dte: Maximum days to expiration (inclusive).
            num_strikes: Number of strikes on each side of ATM.

        Returns:
            List of fully qualified ``ib_async.Option`` contracts.
        """
        self._log.info(
            "get_options_chain",
            ticker=ticker,
            min_dte=min_dte,
            max_dte=max_dte,
            num_strikes=num_strikes,
        )

        # Step 1 — Option chain parameters
        chain_params = await self._factory.get_option_chain_params(ticker)
        if not chain_params:
            self._log.warning("no_chain_params", ticker=ticker)
            return []

        # Pick the SMART exchange chain (preferred for routing)
        smart_chain = _pick_smart_chain(chain_params)

        self._log.debug(
            "chain_params_selected",
            ticker=ticker,
            exchange=smart_chain.exchange,
            total_expirations=len(smart_chain.expirations),
            total_strikes=len(smart_chain.strikes),
        )

        # Step 2 — Filter expirations by DTE
        today = date.today()
        valid_expirations: list[str] = []
        for exp in sorted(smart_chain.expirations):
            dte = _dte(exp, today)
            if min_dte <= dte <= max_dte:
                valid_expirations.append(exp)

        if not valid_expirations:
            self._log.warning(
                "no_valid_expirations",
                ticker=ticker,
                min_dte=min_dte,
                max_dte=max_dte,
            )
            return []

        self._log.debug(
            "filtered_expirations",
            ticker=ticker,
            count=len(valid_expirations),
            expirations=valid_expirations,
        )

        # Step 3 — Current underlying price
        spot_price = await self._get_spot_price(ticker)
        if spot_price <= 0:
            self._log.warning("invalid_spot_price", ticker=ticker, price=spot_price)
            return []

        # Step 4 — Filter strikes to nearest the money
        all_strikes = sorted(smart_chain.strikes)
        selected_strikes = self._select_nearest_strikes(
            all_strikes, spot_price, num_strikes
        )

        if not selected_strikes:
            self._log.warning("no_valid_strikes", ticker=ticker)
            return []

        self._log.debug(
            "filtered_strikes",
            ticker=ticker,
            count=len(selected_strikes),
            spot=spot_price,
            strikes=selected_strikes,
        )

        # Step 5 — Build option contracts
        raw_contracts: list[Option] = []
        for exp in valid_expirations:
            for strike in selected_strikes:
                for right in ("C", "P"):
                    opt = Option(
                        symbol=ticker,
                        lastTradeDateOrContractMonth=exp,
                        strike=strike,
                        right=right,
                        exchange="SMART",
                        multiplier=smart_chain.multiplier,
                        currency="USD",
                    )
                    raw_contracts.append(opt)

        self._log.info(
            "qualifying_options",
            ticker=ticker,
            total_contracts=len(raw_contracts),
        )

        # Step 6 — Qualify in rate-limited batches
        qualified = await self._qualify_contracts_batched(raw_contracts)

        self._log.info(
            "options_chain_ready",
            ticker=ticker,
            qualified_count=len(qualified),
            expirations=valid_expirations,
            strike_range=(
                f"{selected_strikes[0]}-{selected_strikes[-1]}"
                if selected_strikes
                else "none"
            ),
        )

        return qualified

    async def get_option_greeks(self, options: list[Option]) -> dict[int, OptionGreeks]:
        """Request market data for options and extract model Greeks.

        For each option contract, streaming market data is requested and
        then the ``modelGreeks`` attribute is read from the resulting
        ``Ticker``.  Data is collected after a brief settle delay to
        allow IB Gateway to compute the model values.

        Args:
            options: List of qualified ``Option`` contracts.

        Returns:
            Dict mapping ``conId`` to :class:`OptionGreeks`.
        """
        if not options:
            return {}

        self._log.info("requesting_option_greeks", count=len(options))

        # Request market data for each option (rate limited)
        tickers_map: dict[int, tuple[Option, Ticker]] = {}
        for opt in options:
            if opt.conId in self._option_subscriptions:
                _, existing_ticker = self._option_subscriptions[opt.conId]
                tickers_map[opt.conId] = (opt, existing_ticker)
            else:
                ib_ticker = self._ib.reqMktData(
                    opt,
                    genericTickList=OPTION_GENERIC_TICKS,
                    snapshot=False,
                    regulatorySnapshot=False,
                )
                tickers_map[opt.conId] = (opt, ib_ticker)
                self._option_subscriptions[opt.conId] = (opt, ib_ticker)
            await asyncio.sleep(IB_RATE_LIMIT_DELAY)

        # Wait for model greeks to populate
        await asyncio.sleep(GREEKS_SETTLE_SECONDS)

        # Extract greeks
        result: dict[int, OptionGreeks] = {}
        for con_id, (opt, ib_ticker) in tickers_map.items():
            greeks = self._extract_greeks(opt, ib_ticker)
            result[con_id] = greeks

        self._log.info(
            "option_greeks_ready",
            total=len(result),
            with_delta=sum(1 for g in result.values() if g.delta != 0.0),
        )

        return result

    async def get_iv_surface(
        self,
        ticker: str,
        expirations: list[str] | None = None,
        num_strikes: int = 15,
    ) -> IVSurface:
        """Build an implied-volatility surface across expirations and strikes.

        Retrieves the full options chain, requests Greeks for all
        contracts, and organises the data into a nested structure
        indexed by expiration and strike.

        Args:
            ticker: Underlying symbol.
            expirations: Specific expirations to include (``YYYYMMDD``
                format).  If ``None`` the method selects all
                expirations between 7 and 90 DTE.
            num_strikes: Number of strikes on each side of ATM.

        Returns:
            An :class:`IVSurface` with ``data[expiry][strike]`` entries.
        """
        self._log.info(
            "building_iv_surface",
            ticker=ticker,
            expirations=expirations,
            num_strikes=num_strikes,
        )

        # Get chain parameters
        chain_params = await self._factory.get_option_chain_params(ticker)
        if not chain_params:
            self._log.warning("no_chain_params_iv_surface", ticker=ticker)
            return IVSurface(ticker=ticker)

        smart_chain = _pick_smart_chain(chain_params)

        # Filter expirations
        today = date.today()
        if expirations is not None:
            target_expirations = [
                e for e in expirations if e in smart_chain.expirations
            ]
        else:
            target_expirations = [
                e for e in sorted(smart_chain.expirations) if 7 <= _dte(e, today) <= 90
            ]

        if not target_expirations:
            self._log.warning("no_expirations_for_iv_surface", ticker=ticker)
            return IVSurface(ticker=ticker)

        # Spot price
        spot_price = await self._get_spot_price(ticker)
        if spot_price <= 0:
            self._log.warning("invalid_spot_iv_surface", ticker=ticker)
            return IVSurface(ticker=ticker)

        all_strikes = sorted(smart_chain.strikes)
        selected_strikes = self._select_nearest_strikes(
            all_strikes, spot_price, num_strikes
        )

        # Build call options only for IV surface (calls and puts should
        # yield the same IV in theory; we use calls for convention and
        # add puts for put-side skew).
        raw_contracts: list[Option] = []
        for exp in target_expirations:
            for strike in selected_strikes:
                for right in ("C", "P"):
                    raw_contracts.append(
                        Option(
                            symbol=ticker,
                            lastTradeDateOrContractMonth=exp,
                            strike=strike,
                            right=right,
                            exchange="SMART",
                            multiplier=smart_chain.multiplier,
                            currency="USD",
                        )
                    )

        qualified = await self._qualify_contracts_batched(raw_contracts)
        if not qualified:
            return IVSurface(ticker=ticker, und_price=spot_price)

        greeks_map = await self.get_option_greeks(qualified)

        # Assemble the surface
        surface_data: dict[str, dict[str, IVSurfacePoint]] = {}
        for _con_id, greeks in greeks_map.items():
            exp_key = greeks.expiry
            strike_key = str(greeks.strike)
            right_suffix = f"_{greeks.right}" if greeks.right else ""
            full_key = f"{strike_key}{right_suffix}"

            if exp_key not in surface_data:
                surface_data[exp_key] = {}

            surface_data[exp_key][full_key] = IVSurfacePoint(
                iv=greeks.implied_vol,
                delta=greeks.delta,
                gamma=greeks.gamma,
                theta=greeks.theta,
                vega=greeks.vega,
            )

        surface = IVSurface(
            ticker=ticker,
            und_price=spot_price,
            data=surface_data,
        )

        self._log.info(
            "iv_surface_ready",
            ticker=ticker,
            expirations=len(surface_data),
            total_points=sum(len(v) for v in surface_data.values()),
        )

        return surface

    # ------------------------------------------------------------------
    # Snapshot methods
    # ------------------------------------------------------------------
    async def get_snapshot(self, ticker: str) -> MarketSnapshot:
        """Get a single point-in-time snapshot of bid/ask/last/volume.

        Uses ``reqMktData`` with ``snapshot=True`` so no streaming
        subscription is created.

        Args:
            ticker: Stock symbol.

        Returns:
            A :class:`MarketSnapshot` with current market prices.
        """
        try:
            if ticker in _INDEX_SYMBOLS:
                contract = await self._factory.create_index(ticker)
            else:
                contract = await self._factory.create_stock(ticker)
        except ValueError:
            self._log.warning("snapshot_qualify_failed", ticker=ticker)
            return MarketSnapshot(ticker=ticker)

        ib_ticker: Ticker = self._ib.reqMktData(
            contract,
            genericTickList=STOCK_GENERIC_TICKS,
            snapshot=True,
            regulatorySnapshot=False,
        )

        # Wait for the snapshot data to arrive
        await asyncio.sleep(SNAPSHOT_SETTLE_SECONDS)

        mid = float("nan")
        bid = _safe_float(ib_ticker.bid, float("nan"))
        ask = _safe_float(ib_ticker.ask, float("nan"))
        if not (math.isnan(bid) or math.isnan(ask)) and bid > 0 and ask > 0:
            mid = (bid + ask) / 2.0

        snapshot = MarketSnapshot(
            ticker=ticker,
            bid=bid,
            ask=ask,
            last=_safe_float(ib_ticker.last, float("nan")),
            volume=_safe_float(ib_ticker.volume, float("nan")),
            mid=mid,
            timestamp=datetime.now(tz=UTC),
        )

        self._log.debug(
            "snapshot_captured",
            ticker=ticker,
            bid=snapshot.bid,
            ask=snapshot.ask,
            last=snapshot.last,
        )

        return snapshot

    async def get_historical_bars(
        self,
        ticker: str,
        duration: str = "100 D",
        bar_size: str = "1 day",
    ) -> pd.DataFrame | None:
        """Fetch historical OHLCV bars from IB Gateway.

        Args:
            ticker: Underlying symbol (e.g. ``"AAPL"``).
            duration: IB duration string (e.g. ``"100 D"``, ``"2 Y"``).
            bar_size: IB bar size string (e.g. ``"1 day"``).

        Returns:
            A :class:`pd.DataFrame` with columns ``['open', 'high',
            'low', 'close', 'volume']`` indexed by date, or ``None`` on
            failure.
        """
        try:
            if ticker in _INDEX_SYMBOLS:
                contract = await self._factory.create_index(ticker)
            else:
                contract = await self._factory.create_stock(ticker)
        except ValueError:
            self._log.warning("hist_bars_qualify_failed", ticker=ticker)
            return None

        try:
            bars = await self._ib.reqHistoricalDataAsync(
                contract=contract,
                endDateTime="",
                durationStr=duration,
                barSizeSetting=bar_size,
                whatToShow="TRADES",
                useRTH=True,
                formatDate=1,
                keepUpToDate=False,
                timeout=30,
            )
        except Exception:
            self._log.exception("historical_bars_request_failed", ticker=ticker)
            return None

        if not bars:
            self._log.warning("historical_bars_empty", ticker=ticker)
            return None

        records: list[dict[str, Any]] = []
        for bar in bars:
            bar_date = bar.date
            if isinstance(bar_date, datetime):
                bar_date = bar_date.date()
            records.append(
                {
                    "date": bar_date,
                    "open": _safe_float(bar.open, 0.0),
                    "high": _safe_float(bar.high, 0.0),
                    "low": _safe_float(bar.low, 0.0),
                    "close": _safe_float(bar.close, 0.0),
                    "volume": int(getattr(bar, "volume", 0) or 0),
                }
            )

        df = pd.DataFrame(records)
        if not df.empty:
            df = df.sort_values("date").set_index("date")

        return df

    async def get_historical_iv(
        self,
        ticker: str,
        days: int = 252,
    ) -> pd.DataFrame:
        """Request historical implied volatility data for IV Rank / Percentile.

        Fetches daily bars of ``OPTION_IMPLIED_VOLATILITY`` from IB Gateway
        for the specified look-back period.

        Args:
            ticker: Underlying symbol.
            days: Number of calendar days of history to request.  Defaults
                to 252 (approximately one trading year).

        Returns:
            A :class:`pd.DataFrame` with columns ``['date', 'iv']`` sorted
            by date ascending.  Returns an empty DataFrame on failure.
        """
        self._log.info(
            "requesting_historical_iv",
            ticker=ticker,
            days=days,
        )

        try:
            if ticker in _INDEX_SYMBOLS:
                contract = await self._factory.create_index(ticker)
            else:
                contract = await self._factory.create_stock(ticker)
        except ValueError:
            self._log.warning("hist_iv_qualify_failed", ticker=ticker)
            return pd.DataFrame(columns=["date", "iv"])

        # Convert days to an IB duration string (weeks or days)
        if days > 365:
            duration_str = f"{max(1, days // 365)} Y"
        elif days > 30:
            duration_str = f"{max(1, days // 7)} W"
        else:
            duration_str = f"{days} D"

        try:
            bars = await self._ib.reqHistoricalDataAsync(
                contract=contract,
                endDateTime="",
                durationStr=duration_str,
                barSizeSetting="1 day",
                whatToShow=HISTORICAL_IV_WHAT_TO_SHOW,
                useRTH=True,
                formatDate=1,
                keepUpToDate=False,
                timeout=30,
            )
        except Exception:
            self._log.exception("historical_iv_request_failed", ticker=ticker)
            return pd.DataFrame(columns=["date", "iv"])

        if not bars:
            self._log.warning("historical_iv_empty", ticker=ticker)
            return pd.DataFrame(columns=["date", "iv"])

        records: list[dict[str, Any]] = []
        for bar in bars:
            bar_date = bar.date
            if isinstance(bar_date, datetime):
                bar_date = bar_date.date()
            iv_value = _safe_float(bar.close, float("nan"))
            if not math.isnan(iv_value):
                records.append({"date": bar_date, "iv": iv_value})

        df = pd.DataFrame(records)
        if not df.empty:
            df = df.sort_values("date").reset_index(drop=True)

        self._log.info(
            "historical_iv_ready",
            ticker=ticker,
            rows=len(df),
            start=str(df["date"].iloc[0]) if not df.empty else "N/A",
            end=str(df["date"].iloc[-1]) if not df.empty else "N/A",
        )

        return df

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _tick_callback(self, ticker: Ticker) -> None:
        """Process incoming ticks and optionally publish to Redis Pub/Sub.

        Called by ``ib_async`` on every tick update for subscribed
        contracts.  If a Redis client was provided at construction time,
        the tick payload is published asynchronously to a channel named
        ``market_data:{symbol}``.

        Args:
            ticker: The ``Ticker`` object that was just updated.
        """
        if ticker.contract is None:
            return

        symbol = ticker.contract.symbol

        if self._redis is not None:
            payload = {
                "symbol": symbol,
                "bid": _safe_float(ticker.bid),
                "ask": _safe_float(ticker.ask),
                "last": _safe_float(ticker.last),
                "volume": _safe_float(ticker.volume),
                "timestamp": datetime.now(tz=UTC).isoformat(),
            }
            # Fire-and-forget publish; do not block the event loop
            asyncio.ensure_future(self._publish_tick(symbol, payload))

    async def _publish_tick(self, symbol: str, payload: dict[str, Any]) -> None:
        """Publish a tick payload to Redis Pub/Sub.

        Args:
            symbol: Ticker symbol used as channel suffix.
            payload: Serialised tick data.
        """
        try:
            channel = f"market_data:{symbol}"
            await self._redis.publish(channel, json.dumps(payload))
        except Exception:
            # Redis publish failures should never crash the data pipeline
            self._log.debug("redis_publish_failed", symbol=symbol, exc_info=True)

    async def _get_spot_price(self, ticker: str) -> float:
        """Return the current midpoint or last price for *ticker*.

        If the ticker is already subscribed, the live Ticker is used.
        Otherwise a snapshot is taken.

        Args:
            ticker: Stock symbol.

        Returns:
            The spot price as a float.  Returns ``0.0`` on failure.
        """
        if ticker in self._subscriptions:
            _, ib_ticker = self._subscriptions[ticker]
            price = ib_ticker.marketPrice()
            if not math.isnan(price) and price > 0:
                return price

        # Fall back to snapshot
        snapshot = await self.get_snapshot(ticker)
        if not math.isnan(snapshot.mid) and snapshot.mid > 0:
            return snapshot.mid
        if not math.isnan(snapshot.last) and snapshot.last > 0:
            return snapshot.last

        return 0.0

    @staticmethod
    def _select_nearest_strikes(
        all_strikes: list[float],
        spot_price: float,
        num_strikes: int,
    ) -> list[float]:
        """Select *num_strikes* strikes on each side of *spot_price*.

        The ATM strike (closest to spot) is always included, yielding
        up to ``2 * num_strikes + 1`` strikes in total.

        Args:
            all_strikes: Sorted list of available strikes.
            spot_price: Current underlying price.
            num_strikes: Number of strikes on each side of ATM.

        Returns:
            Sorted list of selected strikes.
        """
        if not all_strikes:
            return []

        # Find the ATM strike index
        atm_idx = min(
            range(len(all_strikes)),
            key=lambda i: abs(all_strikes[i] - spot_price),
        )

        lower_bound = max(0, atm_idx - num_strikes)
        upper_bound = min(len(all_strikes), atm_idx + num_strikes + 1)

        return all_strikes[lower_bound:upper_bound]

    async def _qualify_contracts_batched(
        self, contracts: list[Option], batch_size: int = 40
    ) -> list[Option]:
        """Qualify option contracts in rate-limited batches.

        IB Gateway can handle batches of ``qualifyContracts`` but we
        should not flood the wire with hundreds of contract detail
        requests simultaneously.

        Args:
            contracts: Raw option contracts to qualify.
            batch_size: Maximum contracts per batch.

        Returns:
            List of successfully qualified ``Option`` contracts.
        """
        qualified: list[Option] = []

        for i in range(0, len(contracts), batch_size):
            batch = contracts[i : i + batch_size]
            try:
                results = await self._ib.qualifyContractsAsync(*batch)
                for result in results:
                    if (
                        result is not None
                        and isinstance(result, Contract)
                        and result.conId
                    ):
                        qualified.append(result)  # type: ignore[arg-type]
            except Exception:
                self._log.warning(
                    "qualify_batch_failed",
                    batch_start=i,
                    batch_size=len(batch),
                    exc_info=True,
                )
            # Rate-limit between batches
            await asyncio.sleep(IB_RATE_LIMIT_DELAY * 2)

        return qualified

    def _extract_greeks(self, opt: Option, ib_ticker: Ticker) -> OptionGreeks:
        """Extract model Greeks from an ``ib_async.Ticker``.

        The preferred source is ``modelGreeks`` (IB's Black-Scholes
        computation).  Falls back to ``lastGreeks`` or ``bidGreeks``
        if the model computation is not available.

        Args:
            opt: The option contract.
            ib_ticker: The Ticker with populated Greeks.

        Returns:
            An :class:`OptionGreeks` with the best available data.
        """
        # Try model greeks first, then last, then bid
        comp: OptionComputation | None = (
            ib_ticker.modelGreeks or ib_ticker.lastGreeks or ib_ticker.bidGreeks
        )

        mid_price = 0.0
        bid = _safe_float(ib_ticker.bid)
        ask = _safe_float(ib_ticker.ask)
        if bid > 0 and ask > 0:
            mid_price = (bid + ask) / 2.0

        if comp is None:
            return OptionGreeks(
                con_id=opt.conId,
                symbol=opt.symbol,
                expiry=opt.lastTradeDateOrContractMonth,
                strike=opt.strike,
                right=opt.right,
                mid_price=mid_price,
            )

        return OptionGreeks(
            con_id=opt.conId,
            symbol=opt.symbol,
            expiry=opt.lastTradeDateOrContractMonth,
            strike=opt.strike,
            right=opt.right,
            delta=_safe_float(comp.delta),
            gamma=_safe_float(comp.gamma),
            theta=_safe_float(comp.theta),
            vega=_safe_float(comp.vega),
            implied_vol=_safe_float(comp.impliedVol),
            und_price=_safe_float(comp.undPrice),
            mid_price=mid_price,
        )
