"""Common utility functions for Project Titan.

Provides date/time helpers for US equity market hours, price rounding,
formatting, and a generic async retry wrapper.

Usage::

    from src.utils.helpers import market_is_open, calculate_dte, safe_divide

    if market_is_open():
        dte = calculate_dte("2026-03-20")
        ratio = safe_divide(premium, width, default=0.0)
"""

from __future__ import annotations

import asyncio
import functools
from datetime import date, datetime, time, timedelta
from typing import TYPE_CHECKING, Any
from zoneinfo import ZoneInfo

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

US_EASTERN = ZoneInfo("America/New_York")

MARKET_OPEN = time(9, 30)
MARKET_CLOSE = time(16, 0)


# ---------------------------------------------------------------------------
# Dynamic NYSE holiday computation
# ---------------------------------------------------------------------------


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    """Return the *n*-th occurrence of *weekday* in *month/year*.

    Parameters
    ----------
    year:
        Calendar year.
    month:
        Calendar month (1–12).
    weekday:
        Day of week (0 = Monday, 6 = Sunday).
    n:
        Occurrence count (1-based).  Use negative values for last.
    """
    if n > 0:
        first = date(year, month, 1)
        # Days until the first target weekday
        delta = (weekday - first.weekday()) % 7
        candidate = first + timedelta(days=delta)
        candidate += timedelta(weeks=n - 1)
        return candidate
    else:
        # Last occurrence: start from last day of month
        if month == 12:
            last = date(year + 1, 1, 1) - timedelta(days=1)
        else:
            last = date(year, month + 1, 1) - timedelta(days=1)
        delta = (last.weekday() - weekday) % 7
        candidate = last - timedelta(days=delta)
        candidate -= timedelta(weeks=abs(n) - 1)
        return candidate


def _easter_date(year: int) -> date:
    """Compute Easter Sunday using the Anonymous Gregorian algorithm."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l_ = (32 + 2 * e + 2 * i - h - k) % 7  # noqa: E741
    m = (a + 11 * h + 22 * l_) // 451
    month, day = divmod(h + l_ - 7 * m + 114, 31)
    return date(year, month, day + 1)


def _observed(d: date) -> date:
    """Apply the NYSE observation rule for fixed holidays.

    If the holiday falls on Saturday, the prior Friday is observed.
    If the holiday falls on Sunday, the following Monday is observed.
    """
    if d.weekday() == 5:  # Saturday
        return d - timedelta(days=1)
    if d.weekday() == 6:  # Sunday
        return d + timedelta(days=1)
    return d


@functools.lru_cache(maxsize=8)
def compute_nyse_holidays(year: int) -> frozenset[date]:
    """Compute all NYSE-observed market holidays for *year*.

    This replaces the hardcoded holiday sets and works for any year.
    Covers: New Year's Day, MLK Day, Presidents' Day, Good Friday,
    Memorial Day, Juneteenth, Independence Day, Labor Day,
    Thanksgiving Day, and Christmas Day.

    Returns:
        A frozenset of dates on which the NYSE is closed.
    """
    holidays: set[date] = set()

    # New Year's Day (Jan 1, observed)
    holidays.add(_observed(date(year, 1, 1)))

    # MLK Day — 3rd Monday of January
    holidays.add(_nth_weekday(year, 1, 0, 3))

    # Presidents' Day — 3rd Monday of February
    holidays.add(_nth_weekday(year, 2, 0, 3))

    # Good Friday — Friday before Easter Sunday
    easter = _easter_date(year)
    holidays.add(easter - timedelta(days=2))

    # Memorial Day — last Monday of May
    holidays.add(_nth_weekday(year, 5, 0, -1))

    # Juneteenth (June 19, observed) — NYSE holiday since 2022
    holidays.add(_observed(date(year, 6, 19)))

    # Independence Day (July 4, observed)
    holidays.add(_observed(date(year, 7, 4)))

    # Labor Day — 1st Monday of September
    holidays.add(_nth_weekday(year, 9, 0, 1))

    # Thanksgiving Day — 4th Thursday of November
    holidays.add(_nth_weekday(year, 11, 3, 4))

    # Christmas Day (Dec 25, observed)
    holidays.add(_observed(date(year, 12, 25)))

    return frozenset(holidays)


def is_market_holiday(d: date) -> bool:
    """Return ``True`` if *d* is a known NYSE market holiday."""
    return d in compute_nyse_holidays(d.year)


# ---------------------------------------------------------------------------
# Market time helpers
# ---------------------------------------------------------------------------


def _is_trading_day(d: date) -> bool:
    """Return True if *d* is a weekday and not a known market holiday."""
    return d.weekday() < 5 and not is_market_holiday(d)


def market_is_open() -> bool:
    """Check whether the US stock market is currently open.

    Returns ``True`` when the current time in US/Eastern falls between
    09:30 and 16:00 on a weekday that is not a major market holiday.
    """
    now = datetime.now(tz=US_EASTERN)
    if not _is_trading_day(now.date()):
        return False
    return MARKET_OPEN <= now.time() < MARKET_CLOSE


def next_market_open() -> datetime:
    """Return the next market open time as a timezone-aware datetime.

    If the market is currently open, returns the *following* session's
    open.  The returned datetime is in the ``America/New_York`` timezone.
    """
    now = datetime.now(tz=US_EASTERN)
    candidate = now.date()

    # If we are already past today's open (or it's not a trading day),
    # start looking from tomorrow.
    if not _is_trading_day(candidate) or now.time() >= MARKET_OPEN:
        candidate += timedelta(days=1)

    while not _is_trading_day(candidate):
        candidate += timedelta(days=1)

    return datetime.combine(candidate, MARKET_OPEN, tzinfo=US_EASTERN)


def trading_days_between(start: date, end: date) -> int:
    """Count the number of trading days between two dates (inclusive).

    Parameters
    ----------
    start:
        Start date (inclusive).
    end:
        End date (inclusive).

    Returns
    -------
    int
        Number of trading days in the range ``[start, end]``.  Returns 0
        if ``end < start``.
    """
    if end < start:
        return 0

    count = 0
    current = start
    while current <= end:
        if _is_trading_day(current):
            count += 1
        current += timedelta(days=1)
    return count


def calculate_dte(expiry: str | date) -> int:
    """Calculate days to expiration from today.

    Parameters
    ----------
    expiry:
        Expiration date as an ISO-format string (``"YYYY-MM-DD"``) or a
        :class:`datetime.date` object.

    Returns
    -------
    int
        Calendar days remaining until expiry.  Returns 0 if expiry is
        today or in the past.
    """
    if isinstance(expiry, str):
        expiry = date.fromisoformat(expiry)
    delta = (expiry - date.today()).days
    return max(delta, 0)


# ---------------------------------------------------------------------------
# Numeric helpers
# ---------------------------------------------------------------------------


def round_to_tick(price: float, tick_size: float = 0.01) -> float:
    """Round a price to the nearest tick increment.

    Parameters
    ----------
    price:
        The raw price value.
    tick_size:
        Minimum price increment.  Defaults to ``0.01`` (one cent).

    Returns
    -------
    float
        The price rounded to the nearest multiple of *tick_size*.
    """
    if tick_size <= 0:
        return price
    return round(round(price / tick_size) * tick_size, 10)


def safe_divide(
    numerator: float,
    denominator: float,
    default: float = 0.0,
) -> float:
    """Perform division with protection against zero or near-zero denominators.

    Parameters
    ----------
    numerator:
        The dividend.
    denominator:
        The divisor.
    default:
        Value to return when *denominator* is zero or nearly zero.

    Returns
    -------
    float
        ``numerator / denominator`` if safe, otherwise *default*.
    """
    if abs(denominator) < 1e-12:
        return default
    return numerator / denominator


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def format_currency(amount: float) -> str:
    """Format a numeric amount as a US-dollar string.

    Parameters
    ----------
    amount:
        The dollar amount.

    Returns
    -------
    str
        Formatted string such as ``"$1,234.56"`` or ``"-$789.00"``.
    """
    if amount < 0:
        return f"-${abs(amount):,.2f}"
    return f"${amount:,.2f}"


# ---------------------------------------------------------------------------
# Async retry helper
# ---------------------------------------------------------------------------


async def retry_async[T](
    coro_func: Callable[..., Awaitable[T]],
    *args: Any,
    max_retries: int = 3,
    delay: float = 1.0,
    backoff_factor: float = 2.0,
    **kwargs: Any,
) -> T:
    """Execute an async callable with automatic retries on failure.

    On each failure the delay is multiplied by *backoff_factor* to
    implement exponential backoff.

    Parameters
    ----------
    coro_func:
        An async callable (coroutine function) to invoke.
    *args:
        Positional arguments forwarded to *coro_func*.
    max_retries:
        Maximum number of retry attempts after the initial call.
    delay:
        Initial delay in seconds between retries.
    backoff_factor:
        Multiplier applied to *delay* after each failed attempt.
    **kwargs:
        Keyword arguments forwarded to *coro_func*.

    Returns
    -------
    T
        The return value of *coro_func* on the first successful call.

    Raises
    ------
    Exception
        The last exception encountered if all retries are exhausted.
    """
    last_exception: BaseException | None = None
    current_delay = delay

    for attempt in range(1 + max_retries):
        try:
            return await coro_func(*args, **kwargs)
        except Exception as exc:
            last_exception = exc
            if attempt < max_retries:
                await asyncio.sleep(current_delay)
                current_delay *= backoff_factor

    # All retries exhausted -- re-raise the final exception.
    raise last_exception  # type: ignore[misc]
