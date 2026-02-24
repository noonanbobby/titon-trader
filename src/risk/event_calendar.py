"""Event calendar for blocking trades near high-impact events.

Fetches earnings dates and economic calendar events from Finnhub and
enforces exclusion windows around earnings, FOMC, CPI, NFP, and OPEX
dates.  New entries are blocked within configurable windows, and existing
positions can be flagged for closure before events.

When the Finnhub economic calendar endpoint is unavailable (e.g. free-tier
403), the system falls back to hardcoded FOMC, CPI, and NFP dates published
by the Federal Reserve and Bureau of Labor Statistics.

Usage::

    from src.risk.event_calendar import EventCalendar

    calendar = EventCalendar(
        risk_config=config["event_exclusions"],
        finnhub_api_key="your_key",
    )
    await calendar.refresh(tickers=["AAPL", "MSFT", "GOOG"])
    blocked, reason = calendar.is_blocked("AAPL")
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from typing import TYPE_CHECKING, Any

import httpx
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

FINNHUB_BASE_URL = "https://finnhub.io/api/v1"
HTTP_TIMEOUT_SECONDS = 30.0

# Known FOMC meeting dates are published well in advance.  These keywords
# help identify FOMC events in the Finnhub economic calendar.
_FOMC_KEYWORDS = ("fomc", "federal funds rate", "interest rate decision")
_CPI_KEYWORDS = ("cpi", "consumer price index")
_NFP_KEYWORDS = ("nonfarm payroll", "non-farm payroll", "nfp", "employment change")

# ---------------------------------------------------------------------------
# Hardcoded economic event dates — official sources
# ---------------------------------------------------------------------------
# These are used as a fallback when the Finnhub economic calendar endpoint
# is unavailable (e.g. requires a paid tier).  Dates are from:
#   FOMC: https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm
#   CPI:  https://www.bls.gov/schedule/news_release/cpi.htm
#   NFP:  https://www.bls.gov/schedule/news_release/empsit.htm
#
# FOMC dates include both days of each two-day meeting.
# UPDATE ANNUALLY: add the next year's dates when published.

_STATIC_FOMC_DATES: list[str] = [
    # 2025
    "2025-01-28",
    "2025-01-29",
    "2025-03-18",
    "2025-03-19",
    "2025-05-06",
    "2025-05-07",
    "2025-06-17",
    "2025-06-18",
    "2025-07-29",
    "2025-07-30",
    "2025-09-16",
    "2025-09-17",
    "2025-10-28",
    "2025-10-29",
    "2025-12-09",
    "2025-12-10",
    # 2026
    "2026-01-27",
    "2026-01-28",
    "2026-03-17",
    "2026-03-18",
    "2026-04-28",
    "2026-04-29",
    "2026-06-16",
    "2026-06-17",
    "2026-07-28",
    "2026-07-29",
    "2026-09-15",
    "2026-09-16",
    "2026-10-27",
    "2026-10-28",
    "2026-12-08",
    "2026-12-09",
]

_STATIC_CPI_DATES: list[str] = [
    # 2025
    "2025-01-15",
    "2025-02-12",
    "2025-03-12",
    "2025-04-10",
    "2025-05-13",
    "2025-06-11",
    "2025-07-15",
    "2025-08-12",
    "2025-09-11",
    "2025-10-24",
    "2025-12-18",
    # 2026
    "2026-01-13",
    "2026-02-13",
    "2026-03-11",
    "2026-04-10",
    "2026-05-12",
    "2026-06-10",
    "2026-07-14",
    "2026-08-12",
    "2026-09-11",
    "2026-10-14",
    "2026-11-10",
    "2026-12-10",
]

_STATIC_NFP_DATES: list[str] = [
    # 2025
    "2025-01-10",
    "2025-02-07",
    "2025-03-07",
    "2025-04-04",
    "2025-05-02",
    "2025-06-06",
    "2025-07-03",
    "2025-08-01",
    "2025-09-05",
    "2025-11-20",
    "2025-12-16",
    # 2026
    "2026-01-09",
    "2026-02-11",
    "2026-03-06",
    "2026-04-03",
    "2026-05-08",
    "2026-06-05",
    "2026-07-02",
    "2026-08-07",
    "2026-09-04",
    "2026-10-02",
    "2026-11-06",
    "2026-12-04",
]


def _build_static_economic_events() -> list[dict[str, Any]]:
    """Build a list of Finnhub-style event dicts from the hardcoded dates."""
    events: list[dict[str, Any]] = []
    for d in _STATIC_FOMC_DATES:
        events.append(
            {"date": d, "event": "FOMC Interest Rate Decision", "country": "US"}
        )
    cpi_event = "CPI Consumer Price Index"
    for d in _STATIC_CPI_DATES:
        events.append({"date": d, "event": cpi_event, "country": "US"})
    nfp_event = "Nonfarm Payroll Employment Change"
    for d in _STATIC_NFP_DATES:
        events.append({"date": d, "event": nfp_event, "country": "US"})
    return events


# ---------------------------------------------------------------------------
# EventCalendar
# ---------------------------------------------------------------------------


class EventCalendar:
    """Tracks earnings and economic events to block entries near volatility catalysts.

    Fetches data from Finnhub and enforces configurable exclusion windows
    around earnings, FOMC meetings, CPI releases, NFP releases, and
    monthly options expiration (OPEX).

    Args:
        risk_config: The ``event_exclusions`` section from ``risk_limits.yaml``.
        finnhub_api_key: API key for Finnhub data requests.
    """

    def __init__(
        self,
        risk_config: dict[str, Any],
        finnhub_api_key: str,
    ) -> None:
        self._log: structlog.stdlib.BoundLogger = get_logger("risk.event_calendar")
        self._config = risk_config
        self._api_key = finnhub_api_key

        # Internal state
        self._earnings_dates: dict[str, list[date]] = {}
        self._economic_events: list[dict[str, Any]] = []
        self._opex_dates: list[date] = []
        self._last_refresh: datetime | None = None

        # Parse exclusion window config
        self._earnings_cfg = self._config.get("earnings", {})
        self._fomc_cfg = self._config.get("fomc", {})
        self._cpi_cfg = self._config.get("cpi", {})
        self._nfp_cfg = self._config.get("nfp", {})
        self._opex_cfg = self._config.get("opex", {})

        # Pre-compute OPEX dates for the next 12 months
        self._compute_opex_dates()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def refresh(self, tickers: list[str]) -> None:
        """Fetch earnings and economic calendar data from Finnhub.

        Should be called daily at 8 AM ET to keep event data current.
        Earnings and economic calendar fetches are isolated — a failure
        in one does not prevent the other from completing.  If the
        economic calendar endpoint is unavailable (e.g. 403 on free
        tier), hardcoded FOMC/CPI/NFP dates are used as a fallback.

        Args:
            tickers: List of ticker symbols to fetch earnings dates for.
        """
        self._log.info("event_calendar_refresh_start", ticker_count=len(tickers))

        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_SECONDS) as client:
            # Fetch earnings for each ticker — isolated per ticker
            for ticker in tickers:
                try:
                    await self._fetch_earnings(client, ticker)
                except Exception:
                    self._log.warning("earnings_fetch_skipped", ticker=ticker)

            # Fetch economic calendar — isolated, with static fallback
            try:
                await self._fetch_economic_calendar(client)
                self._log.info(
                    "economic_calendar_source",
                    source="finnhub_api",
                    event_count=len(self._economic_events),
                )
            except Exception:
                self._economic_events = _build_static_economic_events()
                self._log.warning(
                    "economic_calendar_using_static_fallback",
                    source="hardcoded_2025_2026",
                    event_count=len(self._economic_events),
                )

        self._last_refresh = datetime.now(UTC)
        self._log.info(
            "event_calendar_refresh_complete",
            tickers_with_earnings=len(self._earnings_dates),
            economic_events=len(self._economic_events),
            opex_dates=len(self._opex_dates),
        )

    def is_blocked(
        self,
        ticker: str,
        event_type: str | None = None,
    ) -> tuple[bool, str]:
        """Check if a ticker is blocked from new entries due to upcoming events.

        Evaluates all configured exclusion windows.  If ``event_type`` is
        provided, only that specific event type is checked.

        Args:
            ticker: The ticker symbol to check.
            event_type: Optional event type to check specifically.
                One of ``"earnings"``, ``"fomc"``, ``"cpi"``, ``"nfp"``,
                ``"opex"``.  When ``None``, all event types are checked.

        Returns:
            A tuple of ``(blocked, reason)``.  ``blocked`` is ``True`` if
            the ticker should not receive new entries.  ``reason`` provides
            a human-readable explanation.
        """
        today = date.today()

        if event_type is None or event_type == "earnings":
            blocked, reason = self._check_earnings_window(ticker, today)
            if blocked:
                return True, reason

        if event_type is None or event_type == "fomc":
            blocked, reason = self._check_economic_window(
                today,
                _FOMC_KEYWORDS,
                "FOMC",
                self._fomc_cfg.get("days_before", 1),
                self._fomc_cfg.get("days_after", 1),
            )
            if blocked:
                return True, reason

        if event_type is None or event_type == "cpi":
            blocked, reason = self._check_economic_window(
                today,
                _CPI_KEYWORDS,
                "CPI",
                self._cpi_cfg.get("days_before", 1),
                self._cpi_cfg.get("days_after", 0),
            )
            if blocked:
                return True, reason

        if event_type is None or event_type == "nfp":
            blocked, reason = self._check_economic_window(
                today,
                _NFP_KEYWORDS,
                "NFP",
                self._nfp_cfg.get("days_before", 1),
                self._nfp_cfg.get("days_after", 0),
            )
            if blocked:
                return True, reason

        if event_type is None or event_type == "opex":
            blocked, reason = self._check_opex_window(today)
            if blocked:
                return True, reason

        return False, ""

    def get_size_adjustment(self, ticker: str) -> float:
        """Return a position size multiplier based on proximity to events.

        Unlike :meth:`is_blocked`, this method does not outright block the
        trade but may recommend reduced sizing near certain events.

        Args:
            ticker: The ticker symbol to evaluate.

        Returns:
            A float between 0.0 and 1.0.  ``1.0`` means no adjustment.
        """
        today = date.today()
        min_multiplier = 1.0

        # FOMC size reduction
        if self._fomc_cfg.get("reduce_size", False):
            fomc_multiplier = self._fomc_cfg.get("size_multiplier", 0.50)
            for event in self._economic_events:
                if self._matches_keywords(event, _FOMC_KEYWORDS):
                    event_date = self._parse_event_date(event)
                    if event_date is None:
                        continue
                    days_until = (event_date - today).days
                    # Apply reduction within a wider window than the block window
                    window = self._fomc_cfg.get("days_before", 1) + 1
                    if 0 <= days_until <= window:
                        min_multiplier = min(min_multiplier, fomc_multiplier)

        # OPEX size reduction
        if self._opex_cfg.get("reduce_size", False):
            opex_multiplier = self._opex_cfg.get("size_multiplier", 0.75)
            for opex_date in self._opex_dates:
                days_until = (opex_date - today).days
                window = self._opex_cfg.get("days_before", 1) + 1
                if 0 <= days_until <= window:
                    min_multiplier = min(min_multiplier, opex_multiplier)

        # Earnings proximity (not blocked but close)
        earnings_days_before = self._earnings_cfg.get("days_before", 3)
        ticker_earnings = self._earnings_dates.get(ticker.upper(), [])
        for earnings_date in ticker_earnings:
            days_until = (earnings_date - today).days
            # If within an extended earnings window but not blocked, reduce size
            extended_window = earnings_days_before + 2
            if earnings_days_before < days_until <= extended_window:
                min_multiplier = min(min_multiplier, 0.50)

        return min_multiplier

    def get_upcoming_events(
        self,
        ticker: str,
        days_ahead: int = 7,
    ) -> list[dict[str, Any]]:
        """Return upcoming events relevant to a ticker within a time window.

        Args:
            ticker: The ticker symbol to check.
            days_ahead: Number of calendar days to look ahead.

        Returns:
            A list of event dictionaries with keys ``type``, ``date``,
            ``description``, and ``days_until``.
        """
        today = date.today()
        cutoff = today + timedelta(days=days_ahead)
        events: list[dict[str, Any]] = []

        # Earnings events for this ticker
        ticker_earnings = self._earnings_dates.get(ticker.upper(), [])
        for earnings_date in ticker_earnings:
            if today <= earnings_date <= cutoff:
                events.append(
                    {
                        "type": "earnings",
                        "date": earnings_date.isoformat(),
                        "description": f"{ticker.upper()} earnings report",
                        "days_until": (earnings_date - today).days,
                    }
                )

        # Economic events
        for event in self._economic_events:
            event_date = self._parse_event_date(event)
            if event_date is None:
                continue
            if today <= event_date <= cutoff:
                event_name = event.get("event", "Unknown")
                event_type = self._classify_economic_event(event)
                events.append(
                    {
                        "type": event_type,
                        "date": event_date.isoformat(),
                        "description": event_name,
                        "days_until": (event_date - today).days,
                    }
                )

        # OPEX dates
        for opex_date in self._opex_dates:
            if today <= opex_date <= cutoff:
                events.append(
                    {
                        "type": "opex",
                        "date": opex_date.isoformat(),
                        "description": "Monthly options expiration (OPEX)",
                        "days_until": (opex_date - today).days,
                    }
                )

        # Sort by date
        events.sort(key=lambda e: e["date"])
        return events

    def should_close_position(self, ticker: str) -> tuple[bool, str]:
        """Check if an existing position should be closed before an event.

        Evaluates the ``close_before`` configuration for earnings.  If an
        earnings date is within the ``close_before_days`` window, the
        position should be exited.

        Args:
            ticker: The ticker symbol to check.

        Returns:
            A tuple of ``(should_close, reason)``.
        """
        today = date.today()

        # Earnings close-before check
        if self._earnings_cfg.get("close_before", False):
            close_days = self._earnings_cfg.get("close_before_days", 1)
            ticker_earnings = self._earnings_dates.get(ticker.upper(), [])
            for earnings_date in ticker_earnings:
                days_until = (earnings_date - today).days
                if 0 <= days_until <= close_days:
                    return (
                        True,
                        f"{ticker.upper()} earnings in {days_until} day(s) on "
                        f"{earnings_date.isoformat()} — close position before event",
                    )

        return False, ""

    # ------------------------------------------------------------------
    # Internal: Data fetching
    # ------------------------------------------------------------------

    @retry(
        retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    async def _fetch_earnings(
        self,
        client: httpx.AsyncClient,
        ticker: str,
    ) -> None:
        """Fetch earnings dates for a single ticker from Finnhub.

        Args:
            client: The HTTP client to use.
            ticker: The ticker symbol.
        """
        today = date.today()
        from_date = (today - timedelta(days=7)).isoformat()
        to_date = (today + timedelta(days=90)).isoformat()

        try:
            response = await client.get(
                f"{FINNHUB_BASE_URL}/calendar/earnings",
                params={
                    "symbol": ticker.upper(),
                    "from": from_date,
                    "to": to_date,
                    "token": self._api_key,
                },
            )
            response.raise_for_status()
            data = response.json()

            earnings_list = data.get("earningsCalendar", [])
            dates: list[date] = []
            for entry in earnings_list:
                date_str = entry.get("date")
                if date_str:
                    try:
                        earnings_date = date.fromisoformat(date_str)
                        dates.append(earnings_date)
                    except ValueError:
                        self._log.debug(
                            "invalid_earnings_date",
                            ticker=ticker,
                            date_str=date_str,
                        )

            if dates:
                self._earnings_dates[ticker.upper()] = sorted(dates)
                self._log.debug(
                    "earnings_dates_fetched",
                    ticker=ticker,
                    count=len(dates),
                    next_date=dates[0].isoformat() if dates else None,
                )

        except httpx.HTTPStatusError as exc:
            self._log.warning(
                "earnings_fetch_http_error",
                ticker=ticker,
                status_code=exc.response.status_code,
            )
            raise
        except Exception:
            self._log.exception("earnings_fetch_failed", ticker=ticker)
            raise

    @retry(
        retry=retry_if_exception_type(httpx.TimeoutException),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    async def _fetch_economic_calendar(self, client: httpx.AsyncClient) -> None:
        """Fetch the economic calendar from Finnhub.

        Filters for FOMC, CPI, and NFP events.  Does **not** retry on
        HTTP 4xx errors (e.g. 403 for free-tier keys) since those are
        permanent authorization failures, not transient issues.

        Args:
            client: The HTTP client to use.
        """
        today = date.today()
        from_date = today.isoformat()
        to_date = (today + timedelta(days=90)).isoformat()

        response = await client.get(
            f"{FINNHUB_BASE_URL}/calendar/economic",
            params={
                "from": from_date,
                "to": to_date,
                "token": self._api_key,
            },
        )

        if response.status_code == 403:
            self._log.warning(
                "economic_calendar_403_not_retrying",
                detail="Finnhub /calendar/economic requires paid economic-1 tier",
            )
            raise httpx.HTTPStatusError(
                "403 Forbidden — endpoint requires paid tier",
                request=response.request,
                response=response,
            )

        response.raise_for_status()
        data = response.json()

        all_events = data.get("economicCalendar", [])
        filtered: list[dict[str, Any]] = []
        for event in all_events:
            event_name = (event.get("event", "") or "").lower()
            country = (event.get("country", "") or "").upper()

            # Only US events
            if country != "US":
                continue

            # Filter for relevant event types
            is_relevant = False
            for keyword_set in (_FOMC_KEYWORDS, _CPI_KEYWORDS, _NFP_KEYWORDS):
                if any(kw in event_name for kw in keyword_set):
                    is_relevant = True
                    break

            if is_relevant:
                filtered.append(event)

        self._economic_events = filtered
        self._log.debug(
            "economic_calendar_fetched",
            total_events=len(all_events),
            relevant_events=len(filtered),
        )

    # ------------------------------------------------------------------
    # Internal: Exclusion window checks
    # ------------------------------------------------------------------

    def _check_earnings_window(
        self,
        ticker: str,
        today: date,
    ) -> tuple[bool, str]:
        """Check if the ticker is within an earnings exclusion window.

        Args:
            ticker: The ticker symbol.
            today: Today's date.

        Returns:
            ``(blocked, reason)`` tuple.
        """
        days_before = self._earnings_cfg.get("days_before", 3)
        days_after = self._earnings_cfg.get("days_after", 1)

        ticker_earnings = self._earnings_dates.get(ticker.upper(), [])
        for earnings_date in ticker_earnings:
            days_diff = (earnings_date - today).days

            # Before earnings
            if 0 <= days_diff <= days_before:
                return (
                    True,
                    f"{ticker.upper()} earnings in {days_diff} day(s) on "
                    f"{earnings_date.isoformat()} — blocked {days_before} days before",
                )

            # After earnings
            if -days_after <= days_diff < 0:
                return (
                    True,
                    f"{ticker.upper()} had earnings {abs(days_diff)} day(s) ago on "
                    f"{earnings_date.isoformat()} — blocked {days_after} day(s) after",
                )

        return False, ""

    def _check_economic_window(
        self,
        today: date,
        keywords: tuple[str, ...],
        event_label: str,
        days_before: int,
        days_after: int,
    ) -> tuple[bool, str]:
        """Check if today falls within an economic event exclusion window.

        Args:
            today: Today's date.
            keywords: Tuple of keyword strings to match against event names.
            event_label: Human-readable label for the event (e.g. ``"FOMC"``).
            days_before: Number of days before the event to block.
            days_after: Number of days after the event to block.

        Returns:
            ``(blocked, reason)`` tuple.
        """
        for event in self._economic_events:
            if not self._matches_keywords(event, keywords):
                continue

            event_date = self._parse_event_date(event)
            if event_date is None:
                continue

            days_diff = (event_date - today).days

            # Before event
            if 0 <= days_diff <= days_before:
                return (
                    True,
                    f"{event_label} in {days_diff} day(s) on "
                    f"{event_date.isoformat()} — blocked {days_before} day(s) before",
                )

            # After event
            if days_after > 0 and -days_after <= days_diff < 0:
                return (
                    True,
                    f"{event_label} was {abs(days_diff)} day(s) ago on "
                    f"{event_date.isoformat()} — blocked {days_after} day(s) after",
                )

        return False, ""

    def _check_opex_window(self, today: date) -> tuple[bool, str]:
        """Check if today falls within an OPEX exclusion window.

        Args:
            today: Today's date.

        Returns:
            ``(blocked, reason)`` tuple.
        """
        days_before = self._opex_cfg.get("days_before", 1)
        days_after = self._opex_cfg.get("days_after", 0)

        for opex_date in self._opex_dates:
            days_diff = (opex_date - today).days

            if 0 <= days_diff <= days_before:
                return (
                    True,
                    f"OPEX in {days_diff} day(s) on {opex_date.isoformat()} — "
                    f"blocked {days_before} day(s) before",
                )

            if days_after > 0 and -days_after <= days_diff < 0:
                return (
                    True,
                    f"OPEX was {abs(days_diff)} day(s) ago on "
                    f"{opex_date.isoformat()} — blocked {days_after} day(s) after",
                )

        return False, ""

    # ------------------------------------------------------------------
    # Internal: Helpers
    # ------------------------------------------------------------------

    def _compute_opex_dates(self) -> None:
        """Pre-compute monthly OPEX dates for the next 12 months.

        OPEX is the third Friday of each month.
        """
        today = date.today()
        self._opex_dates = []

        for month_offset in range(13):
            year = today.year + (today.month + month_offset - 1) // 12
            month = (today.month + month_offset - 1) % 12 + 1
            opex = _third_friday(year, month)
            if opex >= today - timedelta(days=7):
                self._opex_dates.append(opex)

        self._log.debug(
            "opex_dates_computed",
            count=len(self._opex_dates),
            next_opex=self._opex_dates[0].isoformat() if self._opex_dates else None,
        )

    @staticmethod
    def _matches_keywords(
        event: dict[str, Any],
        keywords: tuple[str, ...],
    ) -> bool:
        """Check if an economic event matches any of the given keywords.

        Args:
            event: The event dictionary from Finnhub.
            keywords: Tuple of lowercase keyword strings.

        Returns:
            ``True`` if the event name matches any keyword.
        """
        event_name = (event.get("event", "") or "").lower()
        return any(kw in event_name for kw in keywords)

    @staticmethod
    def _parse_event_date(event: dict[str, Any]) -> date | None:
        """Parse the date from a Finnhub economic calendar event.

        Args:
            event: The event dictionary.

        Returns:
            A ``date`` object, or ``None`` if parsing fails.
        """
        date_str = event.get("date")
        if not date_str:
            return None
        try:
            # Finnhub returns dates as "YYYY-MM-DD"
            return date.fromisoformat(date_str[:10])
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _classify_economic_event(event: dict[str, Any]) -> str:
        """Classify an economic event into a type category.

        Args:
            event: The event dictionary.

        Returns:
            One of ``"fomc"``, ``"cpi"``, ``"nfp"``, or ``"economic"``.
        """
        event_name = (event.get("event", "") or "").lower()
        if any(kw in event_name for kw in _FOMC_KEYWORDS):
            return "fomc"
        if any(kw in event_name for kw in _CPI_KEYWORDS):
            return "cpi"
        if any(kw in event_name for kw in _NFP_KEYWORDS):
            return "nfp"
        return "economic"


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _third_friday(year: int, month: int) -> date:
    """Calculate the third Friday of a given month and year.

    Monthly options expiration (OPEX) occurs on the third Friday.

    Args:
        year: Calendar year.
        month: Calendar month (1-12).

    Returns:
        The date of the third Friday.
    """
    # Find the first day of the month
    first_day = date(year, month, 1)
    # weekday(): Monday=0, Friday=4
    first_friday_offset = (4 - first_day.weekday()) % 7
    first_friday = first_day + timedelta(days=first_friday_offset)
    # Third Friday is two weeks after the first Friday
    return first_friday + timedelta(weeks=2)
