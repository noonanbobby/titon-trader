"""SEC EDGAR Form 4 parser for Project Titan.

Fetches and parses SEC EDGAR Form 4 filings to provide raw insider
transaction data to the insider signal module (``src/signals/insider.py``).
SEC EDGAR is free, requires no API key, but enforces:

- **Rate limit**: 10 requests per second
- **User-Agent**: must be descriptive (name + email)

This module handles the EDGAR full-text search API and XML filing
retrieval, returning structured Pydantic models.

Usage::

    from src.data.sec_edgar import SECEdgarClient

    client = SECEdgarClient()
    filings = await client.get_form4_filings("AAPL", days_back=90)
    for f in filings:
        print(f.insider_name, f.transaction_type, f.shares, f.price)
    await client.close()
"""

from __future__ import annotations

import asyncio
import contextlib
import xml.etree.ElementTree as ET
from datetime import date, timedelta
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
    import structlog

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EDGAR_EFTS_URL: str = "https://efts.sec.gov/LATEST/search-index"
EDGAR_FULL_TEXT_URL: str = "https://efts.sec.gov/LATEST/search"
EDGAR_ARCHIVES_URL: str = "https://www.sec.gov/Archives/edgar/data"
EDGAR_COMPANY_TICKERS_URL: str = "https://www.sec.gov/files/company_tickers.json"

# SEC enforces 10 requests/second and requires a descriptive User-Agent
USER_AGENT: str = "ProjectTitan/1.0 (bobby@titan-trading.com)"
RATE_LIMIT_DELAY: float = 0.12  # ~8 req/s to stay comfortably under 10
HTTP_TIMEOUT_SECONDS: float = 30.0

# Transaction codes indicating meaningful insider activity
PURCHASE_CODES: frozenset[str] = frozenset({"P"})  # Open-market purchase
SALE_CODES: frozenset[str] = frozenset({"S"})  # Open-market sale
EXERCISE_CODES: frozenset[str] = frozenset({"M", "C", "A"})  # Option exercise / awards

# Insider title seniority weights
SENIORITY_WEIGHTS: dict[str, float] = {
    "ceo": 1.0,
    "chief executive": 1.0,
    "cfo": 0.9,
    "chief financial": 0.9,
    "president": 0.85,
    "coo": 0.85,
    "chief operating": 0.85,
    "cto": 0.8,
    "evp": 0.75,
    "svp": 0.7,
    "vp": 0.65,
    "vice president": 0.65,
    "director": 0.5,
    "10%": 0.6,
    "officer": 0.6,
}


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class InsiderTransaction(BaseModel):
    """A single parsed Form 4 insider transaction."""

    ticker: str
    insider_name: str = ""
    insider_title: str = ""
    is_director: bool = False
    is_officer: bool = False
    is_ten_percent_owner: bool = False
    transaction_type: str = ""  # "P" (purchase), "S" (sale), etc.
    transaction_date: date | None = None
    shares: float = 0.0
    price: float = 0.0
    total_value: float = 0.0
    shares_owned_after: float = 0.0
    is_10b5_1: bool = False
    filing_url: str = ""
    seniority_weight: float = 0.5


class Form4Filing(BaseModel):
    """Metadata for a Form 4 filing from EDGAR search."""

    accession_number: str
    filing_date: date
    issuer_cik: str = ""
    issuer_name: str = ""
    issuer_ticker: str = ""
    reporter_name: str = ""
    form_type: str = "4"
    filing_url: str = ""


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class SECEdgarClient:
    """Async SEC EDGAR Form 4 client.

    Parameters
    ----------
    cache:
        Optional RedisCache for response caching.
    user_agent:
        User-Agent string for SEC compliance.
    """

    def __init__(
        self,
        cache: Any | None = None,
        user_agent: str = USER_AGENT,
    ) -> None:
        self._cache = cache
        self._user_agent = user_agent
        self._client: httpx.AsyncClient | None = None
        self._log: structlog.BoundLogger = get_logger("data.sec_edgar")
        self._cik_cache: dict[str, str] = {}

    async def _ensure_client(self) -> httpx.AsyncClient:
        """Lazily create the HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=HTTP_TIMEOUT_SECONDS,
                headers={
                    "User-Agent": self._user_agent,
                    "Accept": "application/json, application/xml, text/xml",
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
        wait=wait_exponential(multiplier=1, min=1, max=20),
        stop=stop_after_attempt(4),
    )
    async def _get(
        self, url: str, params: dict[str, Any] | None = None
    ) -> httpx.Response:
        """Execute a rate-limited GET request."""
        client = await self._ensure_client()
        await asyncio.sleep(RATE_LIMIT_DELAY)
        resp = await client.get(url, params=params or {})
        resp.raise_for_status()
        return resp

    # ------------------------------------------------------------------
    # CIK lookup
    # ------------------------------------------------------------------

    async def _get_cik(self, ticker: str) -> str | None:
        """Look up the CIK number for a ticker symbol.

        Parameters
        ----------
        ticker:
            Stock ticker symbol.

        Returns
        -------
        str or None
            10-digit zero-padded CIK, or None if not found.
        """
        ticker_upper = ticker.upper()
        if ticker_upper in self._cik_cache:
            return self._cik_cache[ticker_upper]

        try:
            resp = await self._get(EDGAR_COMPANY_TICKERS_URL)
            data = resp.json()
            for entry in data.values():
                if entry.get("ticker", "").upper() == ticker_upper:
                    cik = str(entry["cik_str"]).zfill(10)
                    self._cik_cache[ticker_upper] = cik
                    return cik
        except (httpx.HTTPStatusError, KeyError) as exc:
            self._log.warning("edgar_cik_lookup_failed", ticker=ticker, error=str(exc))
        return None

    # ------------------------------------------------------------------
    # Form 4 search and parsing
    # ------------------------------------------------------------------

    async def search_form4_filings(
        self,
        ticker: str,
        days_back: int = 90,
    ) -> list[Form4Filing]:
        """Search EDGAR for Form 4 filings for a ticker.

        Parameters
        ----------
        ticker:
            Stock ticker symbol.
        days_back:
            How many days of filings to search.

        Returns
        -------
        list[Form4Filing]
            Form 4 filing metadata.
        """
        from_date = date.today() - timedelta(days=days_back)
        to_date = date.today()

        params = {
            "q": f'"{ticker}"',
            "dateRange": "custom",
            "startdt": str(from_date),
            "enddt": str(to_date),
            "forms": "4",
        }

        try:
            resp = await self._get(EDGAR_FULL_TEXT_URL, params=params)
            data = resp.json()
        except (httpx.HTTPStatusError, httpx.DecodingError) as exc:
            self._log.warning("edgar_search_failed", ticker=ticker, error=str(exc))
            return []

        filings: list[Form4Filing] = []
        for hit in data.get("hits", {}).get("hits", []):
            source = hit.get("_source", {})
            file_date_str = source.get("file_date", "")
            try:
                filing_date = date.fromisoformat(file_date_str[:10])
            except (ValueError, TypeError):
                continue

            accession = source.get("file_num", hit.get("_id", ""))
            entity_name = source.get("entity_name", "")

            filings.append(
                Form4Filing(
                    accession_number=accession,
                    filing_date=filing_date,
                    issuer_ticker=ticker.upper(),
                    reporter_name=entity_name,
                    filing_url=source.get("file_url", ""),
                )
            )

        self._log.debug("edgar_search_results", ticker=ticker, count=len(filings))
        return filings

    async def parse_form4_xml(
        self, filing_url: str, ticker: str
    ) -> list[InsiderTransaction]:
        """Fetch and parse a single Form 4 XML filing.

        Parameters
        ----------
        filing_url:
            URL to the Form 4 XML document.
        ticker:
            Stock ticker for tagging results.

        Returns
        -------
        list[InsiderTransaction]
            Parsed transactions from this filing.
        """
        try:
            resp = await self._get(filing_url)
            content = resp.text
        except (httpx.HTTPStatusError, httpx.ConnectError) as exc:
            self._log.warning("edgar_xml_fetch_failed", url=filing_url, error=str(exc))
            return []

        try:
            root = ET.fromstring(content)
        except ET.ParseError:
            self._log.warning("edgar_xml_parse_failed", url=filing_url)
            return []

        transactions: list[InsiderTransaction] = []

        # Parse reporting owner info
        owner_name = ""
        owner_title = ""
        is_director = False
        is_officer = False
        is_ten_pct = False

        for owner in root.iter("reportingOwner"):
            owner_id = owner.find("reportingOwnerId")
            if owner_id is not None:
                name_el = owner_id.find("rptOwnerName")
                owner_name = (
                    name_el.text.strip() if name_el is not None and name_el.text else ""
                )

            relationship = owner.find("reportingOwnerRelationship")
            if relationship is not None:
                dir_el = relationship.find("isDirector")
                is_director = dir_el is not None and dir_el.text in ("1", "true")
                off_el = relationship.find("isOfficer")
                is_officer = off_el is not None and off_el.text in ("1", "true")
                ten_el = relationship.find("isTenPercentOwner")
                is_ten_pct = ten_el is not None and ten_el.text in ("1", "true")
                title_el = relationship.find("officerTitle")
                owner_title = (
                    title_el.text.strip()
                    if title_el is not None and title_el.text
                    else ""
                )

        seniority = self._calculate_seniority(
            owner_title, is_director, is_officer, is_ten_pct
        )

        # Parse non-derivative transactions
        for txn in root.iter("nonDerivativeTransaction"):
            tx = self._parse_transaction_element(txn)
            if tx is None:
                continue

            # Check for 10b5-1 plan
            is_10b5_1 = False
            for _coding in txn.iter("transactionCoding"):
                # Check footnotes for 10b5-1 mentions
                for fn in txn.iter("footnoteId"):
                    fn_id = fn.get("id", "")
                    if fn_id:
                        for fn_text in root.iter("footnote"):
                            if (
                                fn_text.get("id") == fn_id
                                and fn_text.text
                                and (
                                    "10b5-1" in fn_text.text.lower()
                                    or "rule 10b5" in fn_text.text.lower()
                                )
                            ):
                                is_10b5_1 = True

            transactions.append(
                InsiderTransaction(
                    ticker=ticker.upper(),
                    insider_name=owner_name,
                    insider_title=owner_title,
                    is_director=is_director,
                    is_officer=is_officer,
                    is_ten_percent_owner=is_ten_pct,
                    transaction_type=tx["code"],
                    transaction_date=tx["date"],
                    shares=tx["shares"],
                    price=tx["price"],
                    total_value=tx["shares"] * tx["price"],
                    shares_owned_after=tx["owned_after"],
                    is_10b5_1=is_10b5_1,
                    filing_url=filing_url,
                    seniority_weight=seniority,
                )
            )

        return transactions

    def _parse_transaction_element(self, txn: ET.Element) -> dict[str, Any] | None:
        """Parse a single transaction XML element."""
        coding = txn.find("transactionCoding")
        if coding is None:
            return None
        code_el = coding.find("transactionCode")
        code = code_el.text.strip() if code_el is not None and code_el.text else ""
        if not code:
            return None

        amounts = txn.find("transactionAmounts")
        if amounts is None:
            return None

        shares_el = amounts.find("transactionShares/value")
        shares = (
            float(shares_el.text) if shares_el is not None and shares_el.text else 0.0
        )

        price_el = amounts.find("transactionPricePerShare/value")
        price = float(price_el.text) if price_el is not None and price_el.text else 0.0

        date_el = txn.find("transactionDate/value")
        tx_date = None
        if date_el is not None and date_el.text:
            with contextlib.suppress(ValueError):
                tx_date = date.fromisoformat(date_el.text[:10])

        owned_el = txn.find(
            "postTransactionAmounts/sharesOwnedFollowingTransaction/value"
        )
        owned_after = (
            float(owned_el.text) if owned_el is not None and owned_el.text else 0.0
        )

        return {
            "code": code,
            "date": tx_date,
            "shares": shares,
            "price": price,
            "owned_after": owned_after,
        }

    @staticmethod
    def _calculate_seniority(
        title: str,
        is_director: bool,
        is_officer: bool,
        is_ten_pct: bool,
    ) -> float:
        """Calculate seniority weight based on title and role flags."""
        title_lower = title.lower()
        weight = 0.5  # default

        for keyword, w in SENIORITY_WEIGHTS.items():
            if keyword in title_lower:
                weight = max(weight, w)
                break

        if is_ten_pct:
            weight = max(weight, 0.6)
        if is_officer and weight < 0.6:
            weight = 0.6
        if is_director and weight < 0.5:
            weight = 0.5

        return weight

    # ------------------------------------------------------------------
    # High-level API
    # ------------------------------------------------------------------

    async def get_form4_filings(
        self,
        ticker: str,
        days_back: int = 90,
        max_filings: int = 50,
    ) -> list[InsiderTransaction]:
        """Fetch and parse Form 4 filings for a ticker.

        This is the main entry point — it searches for filings and then
        parses each one to extract structured transactions.

        Parameters
        ----------
        ticker:
            Stock ticker symbol.
        days_back:
            How many days of filings to search.
        max_filings:
            Maximum number of XML filings to parse (rate-limit protection).

        Returns
        -------
        list[InsiderTransaction]
            All parsed insider transactions, excluding 10b5-1 plan trades.
        """
        cache_key = f"edgar:form4:{ticker}:{days_back}"
        if self._cache is not None:
            cached = await self._cache.get_json(cache_key)
            if cached is not None:
                return [InsiderTransaction(**t) for t in cached]

        filings = await self.search_form4_filings(ticker, days_back=days_back)
        all_transactions: list[InsiderTransaction] = []

        for filing in filings[:max_filings]:
            if not filing.filing_url:
                continue
            txns = await self.parse_form4_xml(filing.filing_url, ticker)
            all_transactions.extend(txns)

        # Filter out 10b5-1 plan trades
        meaningful = [t for t in all_transactions if not t.is_10b5_1]

        if self._cache is not None and meaningful:
            await self._cache.set_json(
                cache_key,
                [t.model_dump(mode="json") for t in meaningful],
                ttl=3600,
            )

        self._log.info(
            "edgar_form4_fetched",
            ticker=ticker,
            filings=len(filings),
            transactions=len(meaningful),
            filtered_10b5_1=len(all_transactions) - len(meaningful),
        )
        return meaningful
