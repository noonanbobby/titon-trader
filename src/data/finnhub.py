"""Finnhub API client for Project Titan.

Provides async methods for fetching company news, earnings calendar,
and economic calendar data.  Used by the sentiment analysis module
(``src/signals/sentiment.py``) and the event calendar risk module
(``src/risk/event_calendar.py``).

Finnhub free tier: 30 API calls per minute, 60 per second.

Usage::

    from src.data.finnhub import FinnhubClient

    client = FinnhubClient(api_key="your_finnhub_key")
    news = await client.get_company_news("AAPL", days_back=3)
    earnings = await client.get_earnings_calendar()
    econ = await client.get_economic_calendar()
    await client.close()
"""

from __future__ import annotations

import asyncio
import contextlib
from datetime import UTC, date, datetime, timedelta
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

BASE_URL: str = "https://finnhub.io/api/v1"
HTTP_TIMEOUT_SECONDS: float = 30.0
RATE_LIMIT_CALLS: int = 30
RATE_LIMIT_WINDOW_SECONDS: float = 60.0
DELAY_BETWEEN_CALLS: float = RATE_LIMIT_WINDOW_SECONDS / RATE_LIMIT_CALLS  # 2.0s


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class NewsArticle(BaseModel):
    """A single news article from Finnhub."""

    headline: str
    source: str = ""
    url: str = ""
    summary: str = ""
    category: str = ""
    related: str = ""
    image: str = ""
    published_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class EarningsEvent(BaseModel):
    """A single earnings event."""

    ticker: str
    date: date
    hour: str = ""  # "bmo" (before market open), "amc" (after market close)
    eps_estimate: float | None = None
    eps_actual: float | None = None
    revenue_estimate: float | None = None
    revenue_actual: float | None = None
    quarter: int = 0
    year: int = 0


class EconomicEvent(BaseModel):
    """A single economic calendar event (FOMC, CPI, NFP, etc.)."""

    event: str
    country: str = "US"
    event_date: date | None = Field(default=None, alias="date")
    time: str = ""
    impact: str = ""  # "high", "medium", "low"
    forecast: float | None = None
    previous: float | None = None
    actual: float | None = None

    model_config = {"populate_by_name": True}
    unit: str = ""


class CompanyProfile(BaseModel):
    """Basic company profile information."""

    ticker: str
    name: str = ""
    exchange: str = ""
    industry: str = ""
    sector: str = ""
    market_cap: float = 0.0
    shares_outstanding: float = 0.0


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class FinnhubClient:
    """Async Finnhub API client with rate limiting.

    Parameters
    ----------
    api_key:
        Finnhub API key.
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
        self._log: structlog.BoundLogger = get_logger("data.finnhub")
        self._last_call: float = 0.0

    async def _ensure_client(self) -> httpx.AsyncClient:
        """Lazily create the HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=BASE_URL,
                timeout=HTTP_TIMEOUT_SECONDS,
                params={"token": self._api_key},
            )
        return self._client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def _rate_limit_wait(self) -> None:
        """Enforce rate limiting between calls."""
        import time

        now = time.monotonic()
        elapsed = now - self._last_call
        if elapsed < DELAY_BETWEEN_CALLS:
            await asyncio.sleep(DELAY_BETWEEN_CALLS - elapsed)
        self._last_call = time.monotonic()

    @retry(
        retry=retry_if_exception_type(
            (httpx.HTTPStatusError, httpx.ConnectError, httpx.TimeoutException)
        ),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        stop=stop_after_attempt(4),
    )
    async def _get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        """Execute a rate-limited GET request with retry logic."""
        client = await self._ensure_client()
        await self._rate_limit_wait()
        resp = await client.get(path, params=params or {})
        resp.raise_for_status()
        return resp.json()

    # ------------------------------------------------------------------
    # Company news
    # ------------------------------------------------------------------

    async def get_company_news(
        self,
        ticker: str,
        days_back: int = 3,
    ) -> list[NewsArticle]:
        """Fetch recent news for a company.

        Parameters
        ----------
        ticker:
            Stock ticker symbol.
        days_back:
            Number of days of news to retrieve.

        Returns
        -------
        list[NewsArticle]
            News articles sorted by publish time (newest first).
        """
        cache_key = f"finnhub:news:{ticker}:{days_back}"
        if self._cache is not None:
            cached = await self._cache.get_json(cache_key)
            if cached is not None:
                return [NewsArticle(**a) for a in cached]

        to_date = date.today()
        from_date = to_date - timedelta(days=days_back)

        data = await self._get(
            "/company-news",
            params={
                "symbol": ticker,
                "from": str(from_date),
                "to": str(to_date),
            },
        )

        articles: list[NewsArticle] = []
        for item in data if isinstance(data, list) else []:
            articles.append(
                NewsArticle(
                    headline=item.get("headline", ""),
                    source=item.get("source", ""),
                    url=item.get("url", ""),
                    summary=item.get("summary", ""),
                    category=item.get("category", ""),
                    related=item.get("related", ""),
                    image=item.get("image", ""),
                    published_at=datetime.fromtimestamp(
                        item.get("datetime", 0),
                        tz=UTC,
                    ),
                )
            )

        if self._cache is not None and articles:
            await self._cache.set_json(
                cache_key,
                [a.model_dump(mode="json") for a in articles],
                ttl=900,  # 15 minutes
            )

        self._log.debug("finnhub_news_fetched", ticker=ticker, count=len(articles))
        return articles

    # ------------------------------------------------------------------
    # Earnings calendar
    # ------------------------------------------------------------------

    async def get_earnings_calendar(
        self,
        from_date: date | None = None,
        to_date: date | None = None,
    ) -> list[EarningsEvent]:
        """Fetch upcoming earnings events.

        Parameters
        ----------
        from_date:
            Start date.  Defaults to today.
        to_date:
            End date.  Defaults to 30 days from now.

        Returns
        -------
        list[EarningsEvent]
            Earnings events in the date range.
        """
        if from_date is None:
            from_date = date.today()
        if to_date is None:
            to_date = from_date + timedelta(days=30)

        cache_key = f"finnhub:earnings:{from_date}:{to_date}"
        if self._cache is not None:
            cached = await self._cache.get_json(cache_key)
            if cached is not None:
                return [EarningsEvent(**e) for e in cached]

        data = await self._get(
            "/calendar/earnings",
            params={"from": str(from_date), "to": str(to_date)},
        )

        events: list[EarningsEvent] = []
        for item in data.get("earningsCalendar", []):
            events.append(
                EarningsEvent(
                    ticker=item.get("symbol", ""),
                    date=date.fromisoformat(item["date"])
                    if "date" in item
                    else from_date,
                    hour=item.get("hour", ""),
                    eps_estimate=item.get("epsEstimate"),
                    eps_actual=item.get("epsActual"),
                    revenue_estimate=item.get("revenueEstimate"),
                    revenue_actual=item.get("revenueActual"),
                    quarter=item.get("quarter", 0),
                    year=item.get("year", 0),
                )
            )

        if self._cache is not None and events:
            await self._cache.set_json(
                cache_key,
                [e.model_dump(mode="json") for e in events],
                ttl=3600,
            )

        self._log.debug("finnhub_earnings_fetched", count=len(events))
        return events

    async def get_earnings_for_ticker(
        self,
        ticker: str,
        days_ahead: int = 30,
    ) -> EarningsEvent | None:
        """Get the next earnings event for a specific ticker.

        Parameters
        ----------
        ticker:
            Stock ticker symbol.
        days_ahead:
            Number of days to look ahead.

        Returns
        -------
        EarningsEvent or None
            The next earnings event, or None if none found.
        """
        events = await self.get_earnings_calendar(
            to_date=date.today() + timedelta(days=days_ahead),
        )
        for event in events:
            if event.ticker.upper() == ticker.upper() and event.date >= date.today():
                return event
        return None

    # ------------------------------------------------------------------
    # Economic calendar
    # ------------------------------------------------------------------

    async def get_economic_calendar(
        self,
        from_date: date | None = None,
        to_date: date | None = None,
    ) -> list[EconomicEvent]:
        """Fetch economic calendar events (FOMC, CPI, NFP, etc.).

        Parameters
        ----------
        from_date:
            Start date.  Defaults to today.
        to_date:
            End date.  Defaults to 30 days from now.

        Returns
        -------
        list[EconomicEvent]
            Economic calendar events.
        """
        if from_date is None:
            from_date = date.today()
        if to_date is None:
            to_date = from_date + timedelta(days=30)

        cache_key = f"finnhub:econ:{from_date}:{to_date}"
        if self._cache is not None:
            cached = await self._cache.get_json(cache_key)
            if cached is not None:
                return [EconomicEvent(**e) for e in cached]

        data = await self._get(
            "/calendar/economic",
            params={"from": str(from_date), "to": str(to_date)},
        )

        events: list[EconomicEvent] = []
        for item in data.get("economicCalendar", []):
            ev_date = None
            if "date" in item:
                with contextlib.suppress(ValueError, TypeError):
                    ev_date = date.fromisoformat(item["date"][:10])

            events.append(
                EconomicEvent(
                    event=item.get("event", ""),
                    country=item.get("country", "US"),
                    date=ev_date,
                    time=item.get("time", ""),
                    impact=item.get("impact", ""),
                    forecast=item.get("estimate"),
                    previous=item.get("prev"),
                    actual=item.get("actual"),
                    unit=item.get("unit", ""),
                )
            )

        if self._cache is not None and events:
            await self._cache.set_json(
                cache_key,
                [e.model_dump(mode="json") for e in events],
                ttl=3600,
            )

        self._log.debug("finnhub_econ_calendar_fetched", count=len(events))
        return events

    # ------------------------------------------------------------------
    # Company profile
    # ------------------------------------------------------------------

    async def get_company_profile(self, ticker: str) -> CompanyProfile:
        """Fetch basic company profile (sector, industry, market cap).

        Parameters
        ----------
        ticker:
            Stock ticker symbol.

        Returns
        -------
        CompanyProfile
            Company profile data.
        """
        cache_key = f"finnhub:profile:{ticker}"
        if self._cache is not None:
            cached = await self._cache.get_json(cache_key)
            if cached is not None:
                return CompanyProfile(**cached)

        data = await self._get("/stock/profile2", params={"symbol": ticker})
        profile = CompanyProfile(
            ticker=data.get("ticker", ticker),
            name=data.get("name", ""),
            exchange=data.get("exchange", ""),
            industry=data.get("finnhubIndustry", ""),
            sector=data.get("finnhubIndustry", ""),
            market_cap=data.get("marketCapitalization", 0.0) * 1_000_000,
            shares_outstanding=data.get("shareOutstanding", 0.0) * 1_000_000,
        )

        if self._cache is not None:
            await self._cache.set_json(
                cache_key,
                profile.model_dump(mode="json"),
                ttl=86400,  # 24 hours
            )

        return profile

    # ------------------------------------------------------------------
    # Quote
    # ------------------------------------------------------------------

    async def get_quote(self, ticker: str) -> dict[str, Any]:
        """Fetch real-time quote for a ticker.

        Parameters
        ----------
        ticker:
            Stock ticker.

        Returns
        -------
        dict[str, Any]
            Quote data with keys: c (current), h (high), l (low), o (open),
            pc (previous close), t (timestamp).
        """
        return await self._get("/quote", params={"symbol": ticker})
