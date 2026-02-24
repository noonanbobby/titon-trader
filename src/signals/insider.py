"""SEC EDGAR Form 4 insider transaction detection for Project Titan.

Fetches and parses SEC EDGAR Form 4 filings to detect insider buying and
selling clusters.  A cluster is defined as 3 or more distinct insiders
executing open-market purchases (transaction code ``P``) within a rolling
30-day window.  Trades filed under Rule 10b5-1 automatic plans are excluded
as they carry weaker informational content.

Each insider is weighted by corporate seniority (CEO > CFO > VP > Director)
and by the dollar amount committed, producing a composite insider signal
score from -1.0 (strong selling cluster) to +1.0 (strong buying cluster).

Usage::

    from src.signals.insider import InsiderSignalGenerator

    generator = InsiderSignalGenerator()
    filings = await generator.fetch_form4_filings("AAPL", days_back=90)
    signal = generator.calculate_insider_signal("AAPL", filings)
    print(signal.score, signal.signal_strength)

.. note::

   SEC EDGAR is free and requires no API key.  However, the SEC enforces a
   rate limit of 10 requests per second and requires a descriptive
   ``User-Agent`` header.
"""

from __future__ import annotations

import asyncio
import xml.etree.ElementTree as ET
from datetime import date, timedelta
from typing import TYPE_CHECKING, Literal

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

# SEC EDGAR full-text search API endpoint for filing queries.
EDGAR_EFTS_SEARCH_URL: str = "https://efts.sec.gov/LATEST/search-index"

# EDGAR filing archives base URL for retrieving individual filing documents.
EDGAR_ARCHIVES_BASE_URL: str = "https://www.sec.gov/Archives/edgar/data"

# SEC requires a descriptive User-Agent header.
SEC_USER_AGENT: str = "Project Titan Options Bot research@example.com"

# SEC rate limit: 10 requests per second.
SEC_RATE_LIMIT_DELAY: float = 0.11  # ~9 requests/second for safety margin

HTTP_TIMEOUT_SECONDS: float = 30.0

# Minimum cluster size: at least 3 distinct insiders buying.
MIN_CLUSTER_SIZE: int = 3

# Default rolling window for cluster detection (days).
DEFAULT_CLUSTER_WINDOW_DAYS: int = 30

# Transaction codes: P = open-market purchase, S = open-market sale.
_CODE_PURCHASE: str = "P"
_CODE_SALE: str = "S"

# Keywords that indicate a Rule 10b5-1 automatic trading plan.
_10B5_1_KEYWORDS: tuple[str, ...] = (
    "10b5-1",
    "rule 10b5-1",
    "10b5",
    "trading plan",
    "pre-arranged",
)

# Seniority weights by title pattern.  Patterns are checked with
# case-insensitive substring matching, from most senior to least.
_SENIORITY_TIERS: list[tuple[tuple[str, ...], float]] = [
    (("chief executive officer", "ceo", "chairman", "chair of the board"), 1.0),
    (("chief financial officer", "cfo", "chief operating officer", "coo"), 0.9),
    (("executive vice president", "evp", "senior vice president", "svp"), 0.8),
    (("vice president", "vp"), 0.7),
    (("director",), 0.6),
    (("officer", "secretary", "treasurer", "controller", "comptroller"), 0.5),
    (("10% owner", "10% holder", "beneficial owner"), 0.4),
]

# Dollar-amount weight tiers.
_DOLLAR_TIERS: list[tuple[float, float]] = [
    (1_000_000.0, 1.0),
    (500_000.0, 0.8),
    (100_000.0, 0.6),
    (50_000.0, 0.4),
    (0.0, 0.2),
]

# Signal strength classification boundaries.
_STRONG_BUY_THRESHOLD: float = 0.6
_BUY_THRESHOLD: float = 0.25
_SELL_THRESHOLD: float = -0.25
_STRONG_SELL_THRESHOLD: float = -0.6


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class Form4Filing(BaseModel):
    """A single SEC Form 4 insider transaction filing.

    Attributes:
        ticker: The stock ticker symbol.
        filer_name: Name of the insider who filed.
        filer_title: Corporate title of the insider.
        transaction_date: Date the transaction was executed.
        transaction_code: SEC transaction code (``P`` = purchase,
            ``S`` = sale, ``A`` = award, etc.).
        shares: Number of shares transacted.
        price_per_share: Price per share at time of transaction.
        total_value: Total dollar value (shares * price_per_share).
        is_acquisition: ``True`` if the transaction increased the
            insider's holdings.
        is_10b5_1: ``True`` if the transaction was made under a
            Rule 10b5-1 pre-arranged trading plan.
        filing_url: URL to the full SEC filing.
    """

    ticker: str
    filer_name: str
    filer_title: str
    transaction_date: date
    transaction_code: str
    shares: int = Field(ge=0)
    price_per_share: float = Field(ge=0.0)
    total_value: float = Field(ge=0.0)
    is_acquisition: bool
    is_10b5_1: bool
    filing_url: str


class InsiderCluster(BaseModel):
    """A detected cluster of insider buying or selling activity.

    Attributes:
        ticker: The stock ticker symbol.
        num_insiders: Number of distinct insiders in the cluster.
        total_shares: Aggregate shares transacted.
        total_value: Aggregate dollar value.
        avg_seniority_weight: Mean seniority weight across cluster members.
        window_start: Earliest transaction date in the cluster.
        window_end: Latest transaction date in the cluster.
        filings: The individual Form 4 filings that compose the cluster.
    """

    ticker: str
    num_insiders: int = Field(ge=0)
    total_shares: int = Field(ge=0)
    total_value: float = Field(ge=0.0)
    avg_seniority_weight: float = Field(ge=0.0, le=1.0)
    window_start: date
    window_end: date
    filings: list[Form4Filing]


class InsiderSignal(BaseModel):
    """Composite insider trading signal for a ticker.

    Attributes:
        ticker: The stock ticker symbol.
        score: Signal score from -1.0 (strong selling) to +1.0 (strong buying).
        cluster: Detected insider cluster, or ``None`` if no cluster found.
        num_buys: Total open-market purchases in the lookback window.
        num_sells: Total open-market sales in the lookback window.
        net_shares: Net shares (buys - sells).
        net_value: Net dollar value (buys - sells).
        signal_strength: Human-readable signal classification.
    """

    ticker: str
    score: float = Field(ge=-1.0, le=1.0)
    cluster: InsiderCluster | None
    num_buys: int = Field(ge=0)
    num_sells: int = Field(ge=0)
    net_shares: int
    net_value: float
    signal_strength: Literal["strong_buy", "buy", "neutral", "sell", "strong_sell"]


# ---------------------------------------------------------------------------
# InsiderSignalGenerator
# ---------------------------------------------------------------------------


class InsiderSignalGenerator:
    """Detect insider buying/selling clusters from SEC EDGAR Form 4 filings.

    SEC EDGAR is free and requires no API key.  The generator queries the
    full-text search API for Form 4 filings, parses the XML content of each
    filing to extract transaction details, and runs cluster detection to
    identify coordinated insider activity.

    The constructor takes no arguments.  All configuration is via constants
    at the module level.
    """

    def __init__(self) -> None:
        self._log: structlog.stdlib.BoundLogger = get_logger("signals.insider")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def fetch_form4_filings(
        self,
        ticker: str,
        days_back: int = 90,
    ) -> list[Form4Filing]:
        """Fetch and parse Form 4 filings for a ticker from SEC EDGAR.

        Queries the SEC EDGAR full-text search API for recent Form 4 filings,
        downloads the XML content of each filing, and extracts transaction
        details.

        Args:
            ticker: The stock symbol (e.g. ``"AAPL"``).
            days_back: Number of calendar days to look back (default 90).

        Returns:
            A list of :class:`Form4Filing` objects sorted by transaction date
            descending.
        """
        self._log.info(
            "fetching_form4_filings",
            ticker=ticker,
            days_back=days_back,
        )

        end_date = date.today()
        start_date = end_date - timedelta(days=days_back)

        filing_urls = await self._search_form4_filings(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
        )

        self._log.debug(
            "form4_urls_found",
            ticker=ticker,
            num_urls=len(filing_urls),
        )

        filings: list[Form4Filing] = []

        async with httpx.AsyncClient(
            timeout=HTTP_TIMEOUT_SECONDS,
            headers={"User-Agent": SEC_USER_AGENT},
        ) as client:
            for url in filing_urls:
                # Respect SEC rate limit.
                await asyncio.sleep(SEC_RATE_LIMIT_DELAY)

                xml_content = await self._download_filing(client, url)
                if xml_content is None:
                    continue

                filing = self._parse_form4_xml(xml_content, ticker, url)
                if filing is not None:
                    filings.append(filing)

        # Sort by transaction date descending (most recent first).
        filings.sort(key=lambda f: f.transaction_date, reverse=True)

        self._log.info(
            "form4_filings_fetched",
            ticker=ticker,
            total_filings=len(filings),
            purchases=sum(1 for f in filings if f.transaction_code == _CODE_PURCHASE),
            sales=sum(1 for f in filings if f.transaction_code == _CODE_SALE),
        )
        return filings

    def detect_clusters(
        self,
        filings: list[Form4Filing],
        window_days: int = DEFAULT_CLUSTER_WINDOW_DAYS,
    ) -> InsiderCluster | None:
        """Detect a cluster of insider buying within a rolling window.

        A cluster requires at least :data:`MIN_CLUSTER_SIZE` (3) distinct
        insiders executing open-market purchases (transaction code ``P``)
        within *window_days*.  Filings under Rule 10b5-1 plans are excluded.

        Args:
            filings: Form 4 filings to analyse.
            window_days: Rolling window size in days (default 30).

        Returns:
            An :class:`InsiderCluster` if a buying cluster is detected,
            ``None`` otherwise.
        """
        # Filter to open-market purchases, excluding 10b5-1 plans.
        purchases = [
            f
            for f in filings
            if f.transaction_code == _CODE_PURCHASE and not f.is_10b5_1
        ]

        if len(purchases) < MIN_CLUSTER_SIZE:
            self._log.debug(
                "insufficient_purchases_for_cluster",
                num_purchases=len(purchases),
                min_required=MIN_CLUSTER_SIZE,
            )
            return None

        # Sort by transaction date ascending for rolling-window analysis.
        purchases_sorted = sorted(purchases, key=lambda f: f.transaction_date)

        # Sliding window: for each filing, collect all filings within
        # window_days and check for distinct insiders.
        best_cluster: InsiderCluster | None = None
        best_cluster_size: int = 0

        for i, anchor in enumerate(purchases_sorted):
            window_start = anchor.transaction_date
            window_end = window_start + timedelta(days=window_days)

            # Collect filings within the window.
            window_filings: list[Form4Filing] = []
            distinct_insiders: set[str] = set()

            for filing in purchases_sorted[i:]:
                if filing.transaction_date > window_end:
                    break
                window_filings.append(filing)
                distinct_insiders.add(filing.filer_name.lower().strip())

            if len(distinct_insiders) < MIN_CLUSTER_SIZE:
                continue

            # This is a valid cluster.  Keep the one with the most insiders.
            if len(distinct_insiders) > best_cluster_size:
                best_cluster_size = len(distinct_insiders)

                total_shares = sum(f.shares for f in window_filings)
                total_value = sum(f.total_value for f in window_filings)

                seniority_weights = [
                    self._weight_by_seniority(f.filer_title) for f in window_filings
                ]
                avg_seniority = (
                    sum(seniority_weights) / len(seniority_weights)
                    if seniority_weights
                    else 0.0
                )

                actual_end = max(f.transaction_date for f in window_filings)

                best_cluster = InsiderCluster(
                    ticker=window_filings[0].ticker,
                    num_insiders=len(distinct_insiders),
                    total_shares=total_shares,
                    total_value=round(total_value, 2),
                    avg_seniority_weight=round(avg_seniority, 4),
                    window_start=window_start,
                    window_end=actual_end,
                    filings=window_filings,
                )

        if best_cluster is not None:
            self._log.info(
                "insider_cluster_detected",
                ticker=best_cluster.ticker,
                num_insiders=best_cluster.num_insiders,
                total_value=best_cluster.total_value,
                avg_seniority=best_cluster.avg_seniority_weight,
                window_start=best_cluster.window_start.isoformat(),
                window_end=best_cluster.window_end.isoformat(),
            )
        else:
            self._log.debug("no_insider_cluster_detected")

        return best_cluster

    def calculate_insider_signal(
        self,
        ticker: str,
        filings: list[Form4Filing],
    ) -> InsiderSignal:
        """Calculate a composite insider signal from Form 4 filings.

        Combines cluster detection, seniority weighting, and dollar-amount
        weighting into a single score in [-1.0, 1.0].

        Args:
            ticker: The stock symbol.
            filings: List of Form 4 filings for the ticker.

        Returns:
            An :class:`InsiderSignal` with the composite score.
        """
        # Separate buys and sells (only open-market transactions).
        buys = [
            f
            for f in filings
            if f.transaction_code == _CODE_PURCHASE and not f.is_10b5_1
        ]
        sells = [
            f for f in filings if f.transaction_code == _CODE_SALE and not f.is_10b5_1
        ]

        num_buys = len(buys)
        num_sells = len(sells)

        buy_shares = sum(f.shares for f in buys)
        sell_shares = sum(f.shares for f in sells)
        net_shares = buy_shares - sell_shares

        buy_value = sum(f.total_value for f in buys)
        sell_value = sum(f.total_value for f in sells)
        net_value = buy_value - sell_value

        # Detect buying cluster.
        cluster = self.detect_clusters(filings)

        # Calculate composite score.
        score = self._compute_composite_score(buys, sells, cluster)

        # Classify signal strength.
        signal_strength = self._classify_signal_strength(score)

        signal = InsiderSignal(
            ticker=ticker,
            score=round(score, 4),
            cluster=cluster,
            num_buys=num_buys,
            num_sells=num_sells,
            net_shares=net_shares,
            net_value=round(net_value, 2),
            signal_strength=signal_strength,
        )

        self._log.info(
            "insider_signal_calculated",
            ticker=ticker,
            score=signal.score,
            signal_strength=signal.signal_strength,
            num_buys=num_buys,
            num_sells=num_sells,
            net_value=round(net_value, 2),
            has_cluster=cluster is not None,
        )
        return signal

    # ------------------------------------------------------------------
    # Internal: SEC EDGAR search
    # ------------------------------------------------------------------

    @retry(
        retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    async def _search_form4_filings(
        self,
        ticker: str,
        start_date: date,
        end_date: date,
    ) -> list[str]:
        """Search SEC EDGAR for Form 4 filings matching a ticker.

        Uses the EDGAR full-text search API to find filings, then extracts
        URLs to the individual filing XML documents.

        Args:
            ticker: The stock symbol.
            start_date: Start of the search window.
            end_date: End of the search window.

        Returns:
            A list of URLs pointing to Form 4 XML filing documents.
        """
        urls: list[str] = []

        try:
            async with httpx.AsyncClient(
                timeout=HTTP_TIMEOUT_SECONDS,
                headers={"User-Agent": SEC_USER_AGENT},
            ) as client:
                # Use the EDGAR full-text search API.
                response = await client.get(
                    EDGAR_EFTS_SEARCH_URL,
                    params={
                        "q": f'"{ticker}"',
                        "dateRange": "custom",
                        "startdt": start_date.isoformat(),
                        "enddt": end_date.isoformat(),
                        "forms": "4",
                    },
                )
                response.raise_for_status()
                data = response.json()

                hits = data.get("hits", {}).get("hits", [])

                for hit in hits:
                    source = hit.get("_source", {})
                    source.get("file_num", "")
                    source.get("file_date", "")
                    # Construct the filing URL from accession number.
                    accession = source.get("accession_no", "")

                    if accession:
                        # SEC accession numbers are formatted as
                        # 0001234567-YY-NNNNNN; the URL path removes dashes.
                        clean_accession = accession.replace("-", "")
                        # Extract CIK from the filing.
                        entity_id = source.get("entity_id", "")
                        if entity_id:
                            filing_url = (
                                f"https://www.sec.gov/Archives/edgar/data/"
                                f"{entity_id}/{clean_accession}/{accession}.txt"
                            )
                            urls.append(filing_url)

        except httpx.HTTPStatusError as exc:
            self._log.warning(
                "edgar_search_http_error",
                ticker=ticker,
                status_code=exc.response.status_code,
            )
            raise
        except Exception:
            self._log.exception("edgar_search_failed", ticker=ticker)
            raise

        self._log.debug(
            "edgar_form4_search_complete",
            ticker=ticker,
            num_results=len(urls),
        )
        return urls

    @retry(
        retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException)),
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=1, max=5),
    )
    async def _download_filing(
        self,
        client: httpx.AsyncClient,
        url: str,
    ) -> str | None:
        """Download the raw content of a single SEC filing.

        Args:
            client: The HTTP client to use.
            url: Full URL to the filing document.

        Returns:
            The raw text/XML content of the filing, or ``None`` if the
            download fails.
        """
        try:
            response = await client.get(url)
            response.raise_for_status()
            return response.text
        except httpx.HTTPStatusError as exc:
            self._log.warning(
                "filing_download_http_error",
                url=url,
                status_code=exc.response.status_code,
            )
            return None
        except Exception:
            self._log.exception("filing_download_failed", url=url)
            return None

    # ------------------------------------------------------------------
    # Internal: XML parsing
    # ------------------------------------------------------------------

    def _parse_form4_xml(
        self,
        xml_content: str,
        ticker: str,
        filing_url: str,
    ) -> Form4Filing | None:
        """Parse a single SEC Form 4 XML filing to extract transaction data.

        Extracts the reporting owner's name and title, the transaction details
        (code, date, shares, price), and whether the trade was under a
        10b5-1 plan.

        Only open-market purchases (code ``P``) and sales (code ``S``) are
        retained.  Award grants (code ``A``), gifts (code ``G``), and other
        derivative transactions are filtered out.

        Args:
            xml_content: The raw XML string from the SEC filing.
            ticker: The stock ticker for context.
            filing_url: The URL of the filing (for logging and output).

        Returns:
            A :class:`Form4Filing` if a valid P or S transaction is found,
            ``None`` otherwise.
        """
        try:
            # The SEC filing may contain a full-text header followed by the
            # XML document.  Extract the XML portion.
            xml_start = xml_content.find("<ownershipDocument")
            if xml_start == -1:
                xml_start = xml_content.find("<XML>")
                if xml_start != -1:
                    xml_start = xml_content.find("<ownershipDocument", xml_start)

            if xml_start == -1:
                self._log.debug(
                    "no_xml_found_in_filing",
                    filing_url=filing_url,
                )
                return None

            xml_end = xml_content.find("</ownershipDocument>", xml_start)
            if xml_end == -1:
                self._log.debug(
                    "no_xml_end_tag",
                    filing_url=filing_url,
                )
                return None

            end_tag_len = len("</ownershipDocument>")
            xml_fragment = xml_content[xml_start : xml_end + end_tag_len]

            root = ET.fromstring(xml_fragment)

        except ET.ParseError:
            self._log.debug(
                "xml_parse_error",
                filing_url=filing_url,
            )
            return None

        # Extract reporting owner information.
        filer_name = self._extract_text(
            root, ".//reportingOwner/reportingOwnerId/rptOwnerName"
        )
        filer_title = self._extract_text(
            root,
            ".//reportingOwner/reportingOwnerRelationship/officerTitle",
        )
        is_director = self._extract_text(
            root,
            ".//reportingOwner/reportingOwnerRelationship/isDirector",
        )
        is_officer = self._extract_text(
            root,
            ".//reportingOwner/reportingOwnerRelationship/isOfficer",
        )
        is_ten_pct_owner = self._extract_text(
            root,
            ".//reportingOwner/reportingOwnerRelationship/isTenPercentOwner",
        )

        # Build a descriptive title if the officerTitle field is empty.
        if not filer_title:
            title_parts: list[str] = []
            if is_director and is_director.strip() in ("1", "true"):
                title_parts.append("Director")
            if is_officer and is_officer.strip() in ("1", "true"):
                title_parts.append("Officer")
            if is_ten_pct_owner and is_ten_pct_owner.strip() in ("1", "true"):
                title_parts.append("10% Owner")
            filer_title = ", ".join(title_parts) if title_parts else "Unknown"

        if not filer_name:
            filer_name = "Unknown Filer"

        # Check footnotes for 10b5-1 plan indication.
        is_10b5_1 = self._detect_10b5_1(root, xml_fragment)

        # Extract non-derivative transactions (common stock).
        transactions = root.findall(".//nonDerivativeTable/nonDerivativeTransaction")

        for txn in transactions:
            txn_code = self._extract_text(txn, ".//transactionCoding/transactionCode")

            # Only keep open-market purchases and sales.
            if txn_code not in (_CODE_PURCHASE, _CODE_SALE):
                continue

            txn_date_str = self._extract_text(txn, ".//transactionDate/value")
            shares_str = self._extract_text(
                txn, ".//transactionAmounts/transactionShares/value"
            )
            price_str = self._extract_text(
                txn, ".//transactionAmounts/transactionPricePerShare/value"
            )
            acq_disp_code = self._extract_text(
                txn,
                ".//transactionAmounts/transactionAcquiredDisposedCode/value",
            )

            # Parse transaction date.
            try:
                txn_date = (
                    date.fromisoformat(txn_date_str) if txn_date_str else date.today()
                )
            except ValueError:
                txn_date = date.today()

            # Parse shares.
            try:
                shares = int(float(shares_str)) if shares_str else 0
            except (ValueError, TypeError):
                shares = 0

            # Parse price per share.
            try:
                price = float(price_str) if price_str else 0.0
            except (ValueError, TypeError):
                price = 0.0

            is_acquisition = (acq_disp_code or "").upper() == "A"
            total_value = abs(shares * price)

            return Form4Filing(
                ticker=ticker.upper(),
                filer_name=filer_name,
                filer_title=filer_title,
                transaction_date=txn_date,
                transaction_code=txn_code,
                shares=abs(shares),
                price_per_share=round(price, 4),
                total_value=round(total_value, 2),
                is_acquisition=is_acquisition,
                is_10b5_1=is_10b5_1,
                filing_url=filing_url,
            )

        # No valid P/S transactions found in this filing.
        return None

    # ------------------------------------------------------------------
    # Internal: Weighting functions
    # ------------------------------------------------------------------

    def _weight_by_seniority(self, title: str) -> float:
        """Return a seniority weight based on the insider's corporate title.

        Weights range from 1.0 (CEO/Chairman) down to 0.4 (10% holder).
        Unknown titles receive 0.5.

        Args:
            title: The insider's corporate title string.

        Returns:
            A float between 0.4 and 1.0.
        """
        title_lower = title.lower().strip()

        for keywords, weight in _SENIORITY_TIERS:
            if any(kw in title_lower for kw in keywords):
                return weight

        return 0.5  # Default for unrecognised titles.

    def _weight_by_dollar_amount(self, shares: int, price: float) -> float:
        """Return a weight based on the total dollar amount of the transaction.

        Larger dollar commitments carry more informational weight.

        * $1M+: 1.0
        * $500K--$1M: 0.8
        * $100K--$500K: 0.6
        * $50K--$100K: 0.4
        * Below $50K: 0.2

        Args:
            shares: Number of shares transacted.
            price: Price per share.

        Returns:
            A float between 0.2 and 1.0.
        """
        total_value = abs(shares * price)

        for threshold, weight in _DOLLAR_TIERS:
            if total_value >= threshold:
                return weight

        return 0.2  # Fallback (should not reach here).

    # ------------------------------------------------------------------
    # Internal: Composite score
    # ------------------------------------------------------------------

    def _compute_composite_score(
        self,
        buys: list[Form4Filing],
        sells: list[Form4Filing],
        cluster: InsiderCluster | None,
    ) -> float:
        """Compute the composite insider signal score.

        The score is a weighted combination of:

        1. **Net direction** (buys vs. sells count ratio): 40 % weight.
        2. **Dollar-weighted direction**: 30 % weight.
        3. **Cluster bonus**: 30 % weight (if a buying cluster is detected).

        The result is clamped to [-1.0, 1.0].

        Args:
            buys: Open-market purchase filings (excluding 10b5-1).
            sells: Open-market sale filings (excluding 10b5-1).
            cluster: Detected insider buying cluster, if any.

        Returns:
            A score from -1.0 to 1.0.
        """
        total_filings = len(buys) + len(sells)

        if total_filings == 0:
            return 0.0

        # Component 1: Count-based net direction [-1, 1].
        count_ratio = (len(buys) - len(sells)) / total_filings

        # Component 2: Dollar-weighted net direction [-1, 1].
        buy_weighted_value = sum(
            f.total_value
            * self._weight_by_seniority(f.filer_title)
            * self._weight_by_dollar_amount(f.shares, f.price_per_share)
            for f in buys
        )
        sell_weighted_value = sum(
            f.total_value
            * self._weight_by_seniority(f.filer_title)
            * self._weight_by_dollar_amount(f.shares, f.price_per_share)
            for f in sells
        )
        total_weighted = buy_weighted_value + sell_weighted_value

        if total_weighted > 0:
            dollar_ratio = (buy_weighted_value - sell_weighted_value) / total_weighted
        else:
            dollar_ratio = 0.0

        # Component 3: Cluster bonus [0, 1] (only for buying clusters).
        cluster_score = 0.0
        if cluster is not None:
            # Scale by number of insiders and average seniority.
            insider_factor = min(cluster.num_insiders / 5.0, 1.0)
            cluster_score = insider_factor * cluster.avg_seniority_weight

        # Weighted combination.
        raw_score = 0.40 * count_ratio + 0.30 * dollar_ratio + 0.30 * cluster_score

        # Clamp to [-1.0, 1.0].
        return max(-1.0, min(1.0, raw_score))

    # ------------------------------------------------------------------
    # Internal: Signal strength classification
    # ------------------------------------------------------------------

    @staticmethod
    def _classify_signal_strength(
        score: float,
    ) -> Literal["strong_buy", "buy", "neutral", "sell", "strong_sell"]:
        """Classify a numeric score into a signal strength label.

        Args:
            score: The composite insider signal score in [-1.0, 1.0].

        Returns:
            One of ``"strong_buy"``, ``"buy"``, ``"neutral"``, ``"sell"``,
            or ``"strong_sell"``.
        """
        if score >= _STRONG_BUY_THRESHOLD:
            return "strong_buy"
        if score >= _BUY_THRESHOLD:
            return "buy"
        if score <= _STRONG_SELL_THRESHOLD:
            return "strong_sell"
        if score <= _SELL_THRESHOLD:
            return "sell"
        return "neutral"

    # ------------------------------------------------------------------
    # Internal: XML helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_text(element: ET.Element, xpath: str) -> str:
        """Extract text content from an XML element by XPath.

        Args:
            element: The parent XML element to search within.
            xpath: XPath expression to locate the child element.

        Returns:
            The text content of the found element, or an empty string
            if the element does not exist or has no text.
        """
        found = element.find(xpath)
        if found is not None and found.text:
            return found.text.strip()
        return ""

    @staticmethod
    def _detect_10b5_1(root: ET.Element, raw_xml: str) -> bool:
        """Detect whether a Form 4 filing involves a Rule 10b5-1 trading plan.

        Checks footnotes within the XML document and falls back to scanning
        the raw XML text for 10b5-1 keywords.

        Args:
            root: Parsed XML root element.
            raw_xml: The raw XML string for broader keyword search.

        Returns:
            ``True`` if evidence of a 10b5-1 plan is found.
        """
        # Check structured footnotes.
        footnotes = root.findall(".//footnotes/footnote")
        for footnote in footnotes:
            text = (footnote.text or "").lower()
            if any(kw in text for kw in _10B5_1_KEYWORDS):
                return True

        # Broader keyword scan on the raw XML (catches footnotes that may
        # not be structurally nested under <footnotes>).
        raw_lower = raw_xml.lower()
        return any(kw in raw_lower for kw in _10B5_1_KEYWORDS)
