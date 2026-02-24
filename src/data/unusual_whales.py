"""Unusual Whales API client for Project Titan.

Provides async methods for fetching options flow data including unusual
activity, sweeps, and block trades.  Used by the options flow signal
module (``src/signals/options_flow.py``).

.. important::

    The operator does NOT currently have an Unusual Whales API key.
    This client is designed to **degrade gracefully** — all methods
    return empty results when no API key is configured, and the
    ``available`` property reports readiness.

Usage::

    from src.data.unusual_whales import UnusualWhalesClient

    client = UnusualWhalesClient(api_key="your_key")  # or api_key=""
    if client.available:
        flow = await client.get_flow_alerts("AAPL")
        sweeps = await client.get_sweeps()
    else:
        # Gracefully degraded — no data
        flow = []

    await client.close()
"""

from __future__ import annotations

import asyncio
import contextlib
from datetime import UTC, date, datetime
from typing import TYPE_CHECKING, Any

import httpx
from pydantic import BaseModel, Field
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.utils.logging import get_logger

if TYPE_CHECKING:
    import structlog

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE_URL: str = "https://api.unusualwhales.com/api"
HTTP_TIMEOUT_SECONDS: float = 30.0
RATE_LIMIT_DELAY: float = 0.5


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class FlowAlert(BaseModel):
    """A single unusual options flow alert."""

    ticker: str
    strike: float = 0.0
    expiry: date | None = None
    contract_type: str = ""  # "call" or "put"
    sentiment: str = ""  # "bullish", "bearish", "neutral"
    volume: int = 0
    open_interest: int = 0
    volume_oi_ratio: float = 0.0
    premium: float = 0.0
    trade_type: str = ""  # "sweep", "block", "split"
    size: str = ""  # "normal", "large", "unusual"
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class SweepAlert(BaseModel):
    """An options sweep alert (aggressive, multi-exchange order)."""

    ticker: str
    strike: float = 0.0
    expiry: date | None = None
    contract_type: str = ""
    sentiment: str = ""
    total_premium: float = 0.0
    fills: int = 0
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class FlowSummary(BaseModel):
    """Aggregated flow summary for a ticker."""

    ticker: str
    total_premium_calls: float = 0.0
    total_premium_puts: float = 0.0
    net_premium: float = 0.0  # positive = bullish
    sweep_count: int = 0
    block_count: int = 0
    unusual_count: int = 0
    put_call_ratio: float = 0.0
    sentiment_bias: str = "neutral"  # "bullish", "bearish", "neutral"


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class UnusualWhalesClient:
    """Async Unusual Whales API client with graceful degradation.

    Parameters
    ----------
    api_key:
        Unusual Whales API key.  Pass ``""`` to run in degraded mode.
    cache:
        Optional RedisCache for response caching.
    """

    def __init__(
        self,
        api_key: str = "",
        cache: Any | None = None,
    ) -> None:
        self._api_key = api_key
        self._cache = cache
        self._client: httpx.AsyncClient | None = None
        self._log: structlog.BoundLogger = get_logger("data.unusual_whales")

        if not self._api_key:
            self._log.warning(
                "unusual_whales_no_api_key",
                msg="Unusual Whales API key not configured. "
                "Module will return empty results.",
            )

    @property
    def available(self) -> bool:
        """Return True if the API key is configured."""
        return bool(self._api_key and self._api_key != "your_unusual_whales_key")

    async def _ensure_client(self) -> httpx.AsyncClient | None:
        """Lazily create the HTTP client, or return None if no key."""
        if not self.available:
            return None
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=BASE_URL,
                timeout=HTTP_TIMEOUT_SECONDS,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Accept": "application/json",
                },
            )
        return self._client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    @retry(
        retry=retry_if_exception_type(
            (httpx.HTTPStatusError, httpx.ConnectError, httpx.TimeoutException)
        ),
        wait=wait_exponential(multiplier=1, min=1, max=30),
        stop=stop_after_attempt(3),
    )
    async def _get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        """Execute a rate-limited GET request."""
        client = await self._ensure_client()
        if client is None:
            return {}
        await asyncio.sleep(RATE_LIMIT_DELAY)
        resp = await client.get(path, params=params or {})
        resp.raise_for_status()
        return resp.json()

    # ------------------------------------------------------------------
    # Flow alerts
    # ------------------------------------------------------------------

    async def get_flow_alerts(
        self,
        ticker: str | None = None,
        min_premium: float = 50_000.0,
        limit: int = 100,
    ) -> list[FlowAlert]:
        """Fetch unusual options flow alerts.

        Parameters
        ----------
        ticker:
            Filter by ticker, or None for all tickers.
        min_premium:
            Minimum premium filter.
        limit:
            Maximum alerts to return.

        Returns
        -------
        list[FlowAlert]
            Unusual flow alerts. Empty if API key not configured.
        """
        if not self.available:
            return []

        params: dict[str, Any] = {"limit": limit}
        if ticker:
            params["ticker_symbol"] = ticker

        cache_key = f"uw:flow:{ticker or 'all'}:{min_premium}"
        if self._cache is not None:
            cached = await self._cache.get_json(cache_key)
            if cached is not None:
                return [FlowAlert(**a) for a in cached]

        try:
            data = await self._get("/stock/flow-alerts", params=params)
        except (httpx.HTTPStatusError, httpx.ConnectError) as exc:
            self._log.warning("uw_flow_fetch_failed", error=str(exc))
            return []

        alerts: list[FlowAlert] = []
        for item in data.get("data", []):
            premium = float(item.get("total_premium", 0))
            if premium < min_premium:
                continue

            expiry = None
            if "expiry" in item:
                with contextlib.suppress(ValueError, TypeError):
                    expiry = date.fromisoformat(item["expiry"][:10])

            alerts.append(
                FlowAlert(
                    ticker=item.get("ticker_symbol", ticker or ""),
                    strike=float(item.get("strike", 0)),
                    expiry=expiry,
                    contract_type=item.get("put_call", "").lower(),
                    sentiment=item.get("sentiment", ""),
                    volume=int(item.get("volume", 0)),
                    open_interest=int(item.get("open_interest", 0)),
                    volume_oi_ratio=float(item.get("vol_oi", 0)),
                    premium=premium,
                    trade_type=item.get("trade_type", ""),
                    size=item.get("size", "normal"),
                )
            )

        if self._cache is not None and alerts:
            await self._cache.set_json(
                cache_key,
                [a.model_dump(mode="json") for a in alerts],
                ttl=300,
            )

        self._log.debug("uw_flow_fetched", ticker=ticker, count=len(alerts))
        return alerts

    # ------------------------------------------------------------------
    # Sweeps
    # ------------------------------------------------------------------

    async def get_sweeps(
        self,
        ticker: str | None = None,
        min_premium: float = 100_000.0,
    ) -> list[SweepAlert]:
        """Fetch options sweep alerts.

        Parameters
        ----------
        ticker:
            Filter by ticker, or None for all.
        min_premium:
            Minimum premium threshold.

        Returns
        -------
        list[SweepAlert]
            Sweep alerts.  Empty if API key not configured.
        """
        if not self.available:
            return []

        params: dict[str, Any] = {}
        if ticker:
            params["ticker_symbol"] = ticker

        try:
            data = await self._get("/stock/sweeps", params=params)
        except (httpx.HTTPStatusError, httpx.ConnectError) as exc:
            self._log.warning("uw_sweeps_fetch_failed", error=str(exc))
            return []

        sweeps: list[SweepAlert] = []
        for item in data.get("data", []):
            premium = float(item.get("total_premium", 0))
            if premium < min_premium:
                continue

            expiry = None
            if "expiry" in item:
                with contextlib.suppress(ValueError, TypeError):
                    expiry = date.fromisoformat(item["expiry"][:10])

            sweeps.append(
                SweepAlert(
                    ticker=item.get("ticker_symbol", ticker or ""),
                    strike=float(item.get("strike", 0)),
                    expiry=expiry,
                    contract_type=item.get("put_call", "").lower(),
                    sentiment=item.get("sentiment", ""),
                    total_premium=premium,
                    fills=int(item.get("fill_count", 0)),
                )
            )

        self._log.debug("uw_sweeps_fetched", ticker=ticker, count=len(sweeps))
        return sweeps

    # ------------------------------------------------------------------
    # Flow summary
    # ------------------------------------------------------------------

    async def get_flow_summary(self, ticker: str) -> FlowSummary:
        """Build an aggregated flow summary for a ticker.

        Parameters
        ----------
        ticker:
            Stock ticker symbol.

        Returns
        -------
        FlowSummary
            Aggregated flow data.  All zeros if API key not configured.
        """
        if not self.available:
            return FlowSummary(ticker=ticker)

        alerts = await self.get_flow_alerts(ticker=ticker, min_premium=25_000.0)

        call_premium = sum(a.premium for a in alerts if a.contract_type == "call")
        put_premium = sum(a.premium for a in alerts if a.contract_type == "put")
        sweep_count = sum(1 for a in alerts if a.trade_type == "sweep")
        block_count = sum(1 for a in alerts if a.trade_type == "block")
        pcr = put_premium / call_premium if call_premium > 0 else 0.0

        if call_premium > put_premium * 1.5:
            bias = "bullish"
        elif put_premium > call_premium * 1.5:
            bias = "bearish"
        else:
            bias = "neutral"

        return FlowSummary(
            ticker=ticker,
            total_premium_calls=call_premium,
            total_premium_puts=put_premium,
            net_premium=call_premium - put_premium,
            sweep_count=sweep_count,
            block_count=block_count,
            unusual_count=len(alerts),
            put_call_ratio=round(pcr, 3),
            sentiment_bias=bias,
        )
