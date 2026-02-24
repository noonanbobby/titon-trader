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
from datetime import date, datetime, time, timedelta
from typing import TYPE_CHECKING, Any, TypeVar
from zoneinfo import ZoneInfo

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

T = TypeVar("T")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

US_EASTERN = ZoneInfo("America/New_York")

MARKET_OPEN = time(9, 30)
MARKET_CLOSE = time(16, 0)

# Major US market holidays that are fixed or can be computed for the
# current/next year.  This list is refreshed manually at the start of
# each calendar year.  Federal holidays where NYSE is closed:
_MARKET_HOLIDAYS_2026: set[date] = {
    date(2026, 1, 1),  # New Year's Day
    date(2026, 1, 19),  # MLK Day
    date(2026, 2, 16),  # Presidents' Day
    date(2026, 4, 3),  # Good Friday
    date(2026, 5, 25),  # Memorial Day
    date(2026, 7, 3),  # Independence Day (observed)
    date(2026, 9, 7),  # Labor Day
    date(2026, 11, 26),  # Thanksgiving
    date(2026, 12, 25),  # Christmas
}

_MARKET_HOLIDAYS_2025: set[date] = {
    date(2025, 1, 1),  # New Year's Day
    date(2025, 1, 20),  # MLK Day
    date(2025, 2, 17),  # Presidents' Day
    date(2025, 4, 18),  # Good Friday
    date(2025, 5, 26),  # Memorial Day
    date(2025, 7, 4),  # Independence Day
    date(2025, 9, 1),  # Labor Day
    date(2025, 11, 27),  # Thanksgiving
    date(2025, 12, 25),  # Christmas
}

_ALL_HOLIDAYS: set[date] = _MARKET_HOLIDAYS_2025 | _MARKET_HOLIDAYS_2026


# ---------------------------------------------------------------------------
# Market time helpers
# ---------------------------------------------------------------------------


def _is_trading_day(d: date) -> bool:
    """Return True if *d* is a weekday and not a known market holiday."""
    return d.weekday() < 5 and d not in _ALL_HOLIDAYS


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
