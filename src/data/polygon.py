"""Polygon.io API client for Project Titan.

Provides async methods for fetching historical OHLCV bars, options chain
snapshots, IV surface data, and cross-asset indices (VIX, DXY, copper,
gold).  The operator has a Polygon Options Advanced subscription which
enables full options chain and historical IV data.

Usage::

    from src.data.polygon import PolygonClient

    client = PolygonClient(api_key="your_polygon_key")
    bars = await client.get_bars("AAPL", timespan="day", limit=252)
    snapshot = await client.get_options_snapshot("AAPL")
    await client.close()
"""

from __future__ import annotations

import asyncio
from datetime import UTC, date, datetime, timedelta
from typing import TYPE_CHECKING, Any

import httpx
from pydantic import BaseModel
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.utils.logging import get_logger

if TYPE_CHECKING:
    import pandas as pd
    import structlog

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE_URL: str = "https://api.polygon.io"
HTTP_TIMEOUT_SECONDS: float = 30.0
MAX_RESULTS_PER_PAGE: int = 50_000
RATE_LIMIT_DELAY: float = 0.25  # 5 calls/sec for Options Advanced


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class Bar(BaseModel):
    """A single OHLCV bar."""

    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int = 0
    vwap: float = 0.0
    num_trades: int = 0


class OptionsContract(BaseModel):
    """Summary of a single options contract from a snapshot."""

    ticker: str
    underlying: str
    contract_type: str  # "call" or "put"
    strike: float
    expiration: date
    bid: float = 0.0
    ask: float = 0.0
    last: float = 0.0
    mid: float = 0.0
    volume: int = 0
    open_interest: int = 0
    implied_volatility: float = 0.0
    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0
    vega: float = 0.0


class IndexValue(BaseModel):
    """A data point for an index like VIX, DXY, etc."""

    ticker: str
    value: float
    timestamp: datetime


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class PolygonClient:
    """Async Polygon.io API client.

    Parameters
    ----------
    api_key:
        Polygon.io API key.
    cache:
        Optional RedisCache for response caching.
    """

    def __init__(
        self,
        api_key: str,
        cache: Any | None = None,
    ) -> None:
        self._api_key = api_key
        self._cache = cache
        self._client: httpx.AsyncClient | None = None
        self._log: structlog.BoundLogger = get_logger("data.polygon")

    async def _ensure_client(self) -> httpx.AsyncClient:
        """Lazily create the HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=BASE_URL,
                timeout=HTTP_TIMEOUT_SECONDS,
                headers={"Authorization": f"Bearer {self._api_key}"},
            )
        return self._client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    # ------------------------------------------------------------------
    # Internal request helper
    # ------------------------------------------------------------------

    @retry(
        retry=retry_if_exception_type(
            (httpx.HTTPStatusError, httpx.ConnectError, httpx.TimeoutException)
        ),
        wait=wait_exponential(multiplier=1, min=1, max=30),
        stop=stop_after_attempt(4),
    )
    async def _get(
        self, path: str, params: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Execute a GET request with retry logic."""
        client = await self._ensure_client()
        await asyncio.sleep(RATE_LIMIT_DELAY)
        resp = await client.get(path, params=params or {})
        resp.raise_for_status()
        return resp.json()

    # ------------------------------------------------------------------
    # Aggregates (bars)
    # ------------------------------------------------------------------

    async def get_bars(
        self,
        ticker: str,
        timespan: str = "day",
        multiplier: int = 1,
        from_date: date | None = None,
        to_date: date | None = None,
        limit: int = 252,
    ) -> list[Bar]:
        """Fetch OHLCV bars for a stock or index.

        Parameters
        ----------
        ticker:
            Stock ticker (e.g. ``"AAPL"``) or index (e.g. ``"I:VIX"``).
        timespan:
            Bar size: ``"minute"``, ``"hour"``, ``"day"``, ``"week"``, ``"month"``.
        multiplier:
            Multiplier for the timespan.
        from_date:
            Start date.  Defaults to ``limit`` trading days ago.
        to_date:
            End date.  Defaults to today.
        limit:
            Maximum bars to return.

        Returns
        -------
        list[Bar]
            Bars sorted chronologically.
        """
        if to_date is None:
            to_date = date.today()
        if from_date is None:
            from_date = to_date - timedelta(days=int(limit * 1.6))

        cache_key = (
            f"polygon:bars:{ticker}:{timespan}:{multiplier}:{from_date}:{to_date}"
        )
        if self._cache is not None:
            cached = await self._cache.get_json(cache_key)
            if cached is not None:
                self._log.debug("polygon_bars_cache_hit", ticker=ticker)
                return [Bar(**b) for b in cached]

        path = (
            f"/v2/aggs/ticker/{ticker}/range"
            f"/{multiplier}/{timespan}/{from_date}/{to_date}"
        )
        data = await self._get(
            path, params={"adjusted": "true", "sort": "asc", "limit": limit}
        )

        bars: list[Bar] = []
        for r in data.get("results", []):
            bars.append(
                Bar(
                    timestamp=datetime.fromtimestamp(r["t"] / 1000, tz=UTC),
                    open=r["o"],
                    high=r["h"],
                    low=r["l"],
                    close=r["c"],
                    volume=int(r.get("v", 0)),
                    vwap=r.get("vw", 0.0),
                    num_trades=int(r.get("n", 0)),
                )
            )

        if self._cache is not None and bars:
            ttl = 300 if timespan == "day" else 60
            await self._cache.set_json(
                cache_key, [b.model_dump(mode="json") for b in bars], ttl=ttl
            )

        self._log.debug("polygon_bars_fetched", ticker=ticker, count=len(bars))
        return bars

    async def get_bars_df(
        self,
        ticker: str,
        timespan: str = "day",
        multiplier: int = 1,
        from_date: date | None = None,
        to_date: date | None = None,
        limit: int = 252,
    ) -> pd.DataFrame:
        """Fetch bars and return as a pandas DataFrame."""
        import pandas as _pd

        bars = await self.get_bars(
            ticker, timespan, multiplier, from_date, to_date, limit
        )
        if not bars:
            return _pd.DataFrame()
        df = _pd.DataFrame([b.model_dump() for b in bars])
        df["timestamp"] = _pd.to_datetime(df["timestamp"], utc=True)
        return df.set_index("timestamp")

    # ------------------------------------------------------------------
    # Options snapshot
    # ------------------------------------------------------------------

    async def get_options_snapshot(
        self,
        underlying: str,
        expiration_date_gte: date | None = None,
        expiration_date_lte: date | None = None,
        strike_price_gte: float | None = None,
        strike_price_lte: float | None = None,
    ) -> list[OptionsContract]:
        """Fetch a snapshot of all options contracts for an underlying.

        Parameters
        ----------
        underlying:
            Underlying stock ticker.
        expiration_date_gte:
            Minimum expiration date filter.
        expiration_date_lte:
            Maximum expiration date filter.
        strike_price_gte:
            Minimum strike filter.
        strike_price_lte:
            Maximum strike filter.

        Returns
        -------
        list[OptionsContract]
            All matching options contracts with Greeks and pricing.
        """
        params: dict[str, Any] = {"limit": 250}
        if expiration_date_gte:
            params["expiration_date.gte"] = str(expiration_date_gte)
        if expiration_date_lte:
            params["expiration_date.lte"] = str(expiration_date_lte)
        if strike_price_gte is not None:
            params["strike_price.gte"] = strike_price_gte
        if strike_price_lte is not None:
            params["strike_price.lte"] = strike_price_lte

        contracts: list[OptionsContract] = []
        path = f"/v3/snapshot/options/{underlying}"

        while path:
            data = await self._get(path, params=params)
            for r in data.get("results", []):
                details = r.get("details", {})
                greeks = r.get("greeks", {})
                day = r.get("day", {})
                last_quote = r.get("last_quote", {})

                contracts.append(
                    OptionsContract(
                        ticker=details.get("ticker", ""),
                        underlying=underlying,
                        contract_type=details.get("contract_type", ""),
                        strike=details.get("strike_price", 0.0),
                        expiration=date.fromisoformat(details["expiration_date"])
                        if "expiration_date" in details
                        else date.today(),
                        bid=last_quote.get("bid", 0.0),
                        ask=last_quote.get("ask", 0.0),
                        last=day.get("close", 0.0),
                        mid=(last_quote.get("bid", 0.0) + last_quote.get("ask", 0.0))
                        / 2,
                        volume=int(day.get("volume", 0)),
                        open_interest=int(r.get("open_interest", 0)),
                        implied_volatility=r.get("implied_volatility", 0.0),
                        delta=greeks.get("delta", 0.0),
                        gamma=greeks.get("gamma", 0.0),
                        theta=greeks.get("theta", 0.0),
                        vega=greeks.get("vega", 0.0),
                    )
                )

            next_url = data.get("next_url")
            if next_url:
                path = next_url.replace(BASE_URL, "")
                params = {}
            else:
                path = ""

        self._log.info(
            "polygon_options_snapshot",
            underlying=underlying,
            contracts=len(contracts),
        )
        return contracts

    # ------------------------------------------------------------------
    # Index / cross-asset values
    # ------------------------------------------------------------------

    async def get_previous_close(self, ticker: str) -> float | None:
        """Get the previous trading day's close for a ticker or index.

        Parameters
        ----------
        ticker:
            Stock ticker or index (e.g. ``"I:VIX"``, ``"I:DXY"``).

        Returns
        -------
        float or None
            The closing price, or None if unavailable.
        """
        cache_key = f"polygon:prev:{ticker}"
        if self._cache is not None:
            cached = await self._cache.get_json(cache_key)
            if cached is not None:
                return cached

        try:
            data = await self._get(f"/v2/aggs/ticker/{ticker}/prev")
            results = data.get("results", [])
            if results:
                val = results[0].get("c")
                if self._cache is not None and val is not None:
                    await self._cache.set_json(cache_key, val, ttl=3600)
                return val
        except (httpx.HTTPStatusError, KeyError) as exc:
            self._log.warning(
                "polygon_prev_close_failed", ticker=ticker, error=str(exc)
            )
        return None

    async def get_index_values(
        self,
        tickers: list[str],
    ) -> dict[str, float]:
        """Fetch previous close values for multiple indices.

        Parameters
        ----------
        tickers:
            List of index tickers (e.g. ``["I:VIX", "I:VIX3M", "I:DXY"]``).

        Returns
        -------
        dict[str, float]
            Mapping from ticker to close value.
        """
        results: dict[str, float] = {}
        for ticker in tickers:
            val = await self.get_previous_close(ticker)
            if val is not None:
                results[ticker] = val
        return results

    # ------------------------------------------------------------------
    # Ticker details
    # ------------------------------------------------------------------

    async def get_ticker_details(self, ticker: str) -> dict[str, Any]:
        """Fetch details for a ticker (name, sector, market cap, etc.).

        Parameters
        ----------
        ticker:
            Stock ticker symbol.

        Returns
        -------
        dict[str, Any]
            Ticker detail fields from the Polygon API.
        """
        data = await self._get(f"/v3/reference/tickers/{ticker}")
        return data.get("results", {})

    # ------------------------------------------------------------------
    # Historical IV (Options Advanced)
    # ------------------------------------------------------------------

    async def get_historical_iv(
        self,
        ticker: str,
        from_date: date | None = None,
        to_date: date | None = None,
    ) -> list[dict[str, Any]]:
        """Fetch historical implied volatility data.

        Requires Options Advanced subscription.

        Parameters
        ----------
        ticker:
            Stock ticker.
        from_date:
            Start date.  Defaults to 1 year ago.
        to_date:
            End date.  Defaults to today.

        Returns
        -------
        list[dict[str, Any]]
            Historical IV data points.
        """
        if to_date is None:
            to_date = date.today()
        if from_date is None:
            from_date = to_date - timedelta(days=365)

        path = f"/v2/aggs/ticker/{ticker}/range/1/day/{from_date}/{to_date}"
        data = await self._get(
            path, params={"adjusted": "true", "sort": "asc", "limit": 500}
        )
        return data.get("results", [])

    # ------------------------------------------------------------------
    # Forex / commodities (for cross-asset signals)
    # ------------------------------------------------------------------

    async def get_forex_pair(
        self,
        pair: str,
        from_date: date | None = None,
        to_date: date | None = None,
    ) -> list[Bar]:
        """Fetch daily bars for a forex pair (e.g. ``"C:XAUUSD"`` for gold).

        Parameters
        ----------
        pair:
            Polygon forex/crypto ticker.
        from_date:
            Start date.
        to_date:
            End date.

        Returns
        -------
        list[Bar]
            Daily bars.
        """
        return await self.get_bars(
            pair, timespan="day", from_date=from_date, to_date=to_date, limit=30
        )
