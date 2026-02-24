"""Quiver Quantitative API client for Project Titan.

Provides async methods for fetching alternative data including
Congressional trading, government contracts, lobbying disclosures,
and insider trading aggregates.  The operator has a Quiver Hobbyist
subscription.

Usage::

    from src.data.quiver import QuiverClient

    client = QuiverClient(api_key="your_quiver_key")
    trades = await client.get_congress_trading("AAPL")
    contracts = await client.get_government_contracts("AAPL")
    lobbying = await client.get_lobbying("AAPL")
    await client.close()
"""

from __future__ import annotations

import asyncio
import contextlib
from datetime import date, timedelta
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

BASE_URL: str = "https://api.quiverquant.com/beta"
HTTP_TIMEOUT_SECONDS: float = 30.0
RATE_LIMIT_DELAY: float = 1.0  # Conservative for Hobbyist tier


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class CongressTrade(BaseModel):
    """A single Congressional trading transaction."""

    model_config = {"populate_by_name": True}

    ticker: str
    representative: str = ""
    house: str = ""  # "House" or "Senate"
    transaction_type: str = ""  # "Purchase", "Sale", "Sale (Partial)", "Sale (Full)"
    amount_low: float = 0.0
    amount_high: float = 0.0
    trade_date: date | None = Field(default=None, alias="date")
    disclosure_date: date | None = None


class GovernmentContract(BaseModel):
    """A single government contract award."""

    model_config = {"populate_by_name": True}

    ticker: str
    agency: str = ""
    amount: float = 0.0
    description: str = ""
    contract_date: date | None = Field(default=None, alias="date")


class LobbyingRecord(BaseModel):
    """A single lobbying disclosure record."""

    model_config = {"populate_by_name": True}

    ticker: str
    client: str = ""
    amount: float = 0.0
    issue: str = ""
    record_date: date | None = Field(default=None, alias="date")


class InsiderSummary(BaseModel):
    """Aggregated insider trading summary from Quiver."""

    ticker: str
    net_shares: int = 0
    buy_count: int = 0
    sell_count: int = 0
    total_value: float = 0.0
    period_days: int = 90


class QuiverSignal(BaseModel):
    """Composite Quiver alternative-data signal."""

    ticker: str
    congress_score: float = 0.0  # -1.0 to 1.0
    contract_score: float = 0.0
    lobbying_score: float = 0.0
    composite_score: float = 0.0
    data_freshness_days: int = 0


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class QuiverClient:
    """Async Quiver Quantitative API client.

    Parameters
    ----------
    api_key:
        Quiver Quantitative API key.
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
        self._log: structlog.BoundLogger = get_logger("data.quiver")

    async def _ensure_client(self) -> httpx.AsyncClient:
        """Lazily create the HTTP client."""
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
        stop=stop_after_attempt(4),
    )
    async def _get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        """Execute a rate-limited GET request with retry logic."""
        client = await self._ensure_client()
        await asyncio.sleep(RATE_LIMIT_DELAY)
        resp = await client.get(path, params=params or {})
        resp.raise_for_status()
        return resp.json()

    # ------------------------------------------------------------------
    # Congressional trading
    # ------------------------------------------------------------------

    async def get_congress_trading(
        self,
        ticker: str,
        days_back: int = 90,
    ) -> list[CongressTrade]:
        """Fetch Congressional trading for a ticker.

        Parameters
        ----------
        ticker:
            Stock ticker symbol.
        days_back:
            How many days of history to retrieve.

        Returns
        -------
        list[CongressTrade]
            Congressional trades.
        """
        cache_key = f"quiver:congress:{ticker}:{days_back}"
        if self._cache is not None:
            cached = await self._cache.get_json(cache_key)
            if cached is not None:
                return [CongressTrade(**t) for t in cached]

        try:
            data = await self._get(f"/historical/congresstrading/{ticker}")
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                return []
            raise

        cutoff = date.today() - timedelta(days=days_back)
        trades: list[CongressTrade] = []
        for item in data if isinstance(data, list) else []:
            trade_date = None
            if "Date" in item:
                with contextlib.suppress(ValueError, TypeError):
                    trade_date = date.fromisoformat(item["Date"][:10])

            if trade_date is not None and trade_date < cutoff:
                continue

            amt_str = item.get("Amount", "$0")
            low, high = self._parse_amount_range(amt_str)

            trades.append(
                CongressTrade(
                    ticker=ticker,
                    representative=item.get("Representative", ""),
                    house=item.get("House", ""),
                    transaction_type=item.get("Transaction", ""),
                    amount_low=low,
                    amount_high=high,
                    date=trade_date,
                    disclosure_date=trade_date,
                )
            )

        if self._cache is not None and trades:
            await self._cache.set_json(
                cache_key,
                [t.model_dump(mode="json") for t in trades],
                ttl=3600,
            )

        self._log.debug("quiver_congress_fetched", ticker=ticker, count=len(trades))
        return trades

    @staticmethod
    def _parse_amount_range(amount_str: str) -> tuple[float, float]:
        """Parse Quiver's amount range strings like '$1,001 - $15,000'."""
        clean = amount_str.replace("$", "").replace(",", "").strip()
        if " - " in clean:
            parts = clean.split(" - ")
            try:
                return float(parts[0]), float(parts[1])
            except (ValueError, IndexError):
                pass
        try:
            val = float(clean)
            return val, val
        except ValueError:
            return 0.0, 0.0

    # ------------------------------------------------------------------
    # Government contracts
    # ------------------------------------------------------------------

    async def get_government_contracts(
        self,
        ticker: str,
        days_back: int = 180,
    ) -> list[GovernmentContract]:
        """Fetch government contracts for a ticker.

        Parameters
        ----------
        ticker:
            Stock ticker symbol.
        days_back:
            How many days of history to retrieve.

        Returns
        -------
        list[GovernmentContract]
            Government contract awards.
        """
        cache_key = f"quiver:contracts:{ticker}:{days_back}"
        if self._cache is not None:
            cached = await self._cache.get_json(cache_key)
            if cached is not None:
                return [GovernmentContract(**c) for c in cached]

        try:
            data = await self._get(f"/historical/govcontractsall/{ticker}")
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                return []
            raise

        cutoff = date.today() - timedelta(days=days_back)
        contracts: list[GovernmentContract] = []
        for item in data if isinstance(data, list) else []:
            contract_date = None
            if "Date" in item:
                with contextlib.suppress(ValueError, TypeError):
                    contract_date = date.fromisoformat(item["Date"][:10])

            if contract_date is not None and contract_date < cutoff:
                continue

            contracts.append(
                GovernmentContract(
                    ticker=ticker,
                    agency=item.get("Agency", ""),
                    amount=float(item.get("Amount", 0)),
                    description=item.get("Description", ""),
                    date=contract_date,
                )
            )

        if self._cache is not None and contracts:
            await self._cache.set_json(
                cache_key,
                [c.model_dump(mode="json") for c in contracts],
                ttl=7200,
            )

        self._log.debug("quiver_contracts_fetched", ticker=ticker, count=len(contracts))
        return contracts

    # ------------------------------------------------------------------
    # Lobbying
    # ------------------------------------------------------------------

    async def get_lobbying(
        self,
        ticker: str,
        days_back: int = 365,
    ) -> list[LobbyingRecord]:
        """Fetch lobbying disclosure records for a ticker.

        Parameters
        ----------
        ticker:
            Stock ticker symbol.
        days_back:
            How many days of history to retrieve.

        Returns
        -------
        list[LobbyingRecord]
            Lobbying records.
        """
        cache_key = f"quiver:lobbying:{ticker}:{days_back}"
        if self._cache is not None:
            cached = await self._cache.get_json(cache_key)
            if cached is not None:
                return [LobbyingRecord(**r) for r in cached]

        try:
            data = await self._get(f"/historical/lobbying/{ticker}")
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                return []
            raise

        cutoff = date.today() - timedelta(days=days_back)
        records: list[LobbyingRecord] = []
        for item in data if isinstance(data, list) else []:
            rec_date = None
            if "Date" in item:
                with contextlib.suppress(ValueError, TypeError):
                    rec_date = date.fromisoformat(item["Date"][:10])

            if rec_date is not None and rec_date < cutoff:
                continue

            records.append(
                LobbyingRecord(
                    ticker=ticker,
                    client=item.get("Client", ""),
                    amount=float(item.get("Amount", 0)),
                    issue=item.get("Issue", ""),
                    date=rec_date,
                )
            )

        if self._cache is not None and records:
            await self._cache.set_json(
                cache_key,
                [r.model_dump(mode="json") for r in records],
                ttl=7200,
            )

        self._log.debug("quiver_lobbying_fetched", ticker=ticker, count=len(records))
        return records

    # ------------------------------------------------------------------
    # Composite signal
    # ------------------------------------------------------------------

    async def get_signal(self, ticker: str) -> QuiverSignal:
        """Calculate a composite Quiver alternative-data signal.

        Combines Congressional trading, government contracts, and lobbying
        into a single signal score.

        Parameters
        ----------
        ticker:
            Stock ticker symbol.

        Returns
        -------
        QuiverSignal
            Composite signal with component scores.
        """
        congress = await self.get_congress_trading(ticker, days_back=90)
        contracts = await self.get_government_contracts(ticker, days_back=180)
        lobbying = await self.get_lobbying(ticker, days_back=365)

        # Congressional trading score: net buying = positive
        buy_count = sum(1 for t in congress if "purchase" in t.transaction_type.lower())
        sell_count = sum(1 for t in congress if "sale" in t.transaction_type.lower())
        total_trades = buy_count + sell_count
        congress_score = (buy_count - sell_count) / max(total_trades, 1)

        # Government contracts score: more/larger contracts = positive
        total_contract_value = sum(c.amount for c in contracts)
        contract_score = (
            min(total_contract_value / 1_000_000_000, 1.0) if contracts else 0.0
        )

        # Lobbying score: higher spending = slight positive (more political capital)
        total_lobbying = sum(r.amount for r in lobbying)
        lobbying_score = min(total_lobbying / 50_000_000, 1.0) if lobbying else 0.0

        # Composite: Congressional = highest signal, contracts next, lobbying least
        composite = 0.6 * congress_score + 0.25 * contract_score + 0.15 * lobbying_score

        return QuiverSignal(
            ticker=ticker,
            congress_score=round(congress_score, 4),
            contract_score=round(contract_score, 4),
            lobbying_score=round(lobbying_score, 4),
            composite_score=round(composite, 4),
            data_freshness_days=90,
        )
