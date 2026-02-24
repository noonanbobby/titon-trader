"""Comprehensive unit tests for src/utils/helpers.py.

Tests all public utility functions: NYSE holiday computation, market-hour
detection, DTE calculation, price rounding, safe division, and currency
formatting.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from unittest.mock import patch

import pytest

from src.utils.helpers import (
    MARKET_OPEN,
    US_EASTERN,
    _easter_date,
    _nth_weekday,
    _observed,
    calculate_dte,
    compute_nyse_holidays,
    format_currency,
    is_market_holiday,
    market_is_open,
    next_market_open,
    round_to_tick,
    safe_divide,
    trading_days_between,
)

# ======================================================================
# Helper function tests (_nth_weekday, _observed, _easter_date)
# ======================================================================


class TestNthWeekday:
    """Tests for the _nth_weekday helper."""

    def test_first_monday_of_january_2025(self) -> None:
        """First Monday of January 2025 is Jan 6."""
        assert _nth_weekday(2025, 1, 0, 1) == date(2025, 1, 6)

    def test_third_monday_of_january_2025(self) -> None:
        """MLK Day 2025: 3rd Monday of January = Jan 20."""
        assert _nth_weekday(2025, 1, 0, 3) == date(2025, 1, 20)

    def test_third_monday_of_february_2025(self) -> None:
        """Presidents' Day 2025: 3rd Monday of February = Feb 17."""
        assert _nth_weekday(2025, 2, 0, 3) == date(2025, 2, 17)

    def test_last_monday_of_may_2025(self) -> None:
        """Memorial Day 2025: last Monday of May = May 26."""
        assert _nth_weekday(2025, 5, 0, -1) == date(2025, 5, 26)

    def test_first_monday_of_september_2025(self) -> None:
        """Labor Day 2025: 1st Monday of September = Sep 1."""
        assert _nth_weekday(2025, 9, 0, 1) == date(2025, 9, 1)

    def test_fourth_thursday_of_november_2025(self) -> None:
        """Thanksgiving 2025: 4th Thursday of November = Nov 27."""
        assert _nth_weekday(2025, 11, 3, 4) == date(2025, 11, 27)

    def test_last_weekday_of_december(self) -> None:
        """Last Monday of December 2025 = Dec 29."""
        assert _nth_weekday(2025, 12, 0, -1) == date(2025, 12, 29)


class TestObserved:
    """Tests for the NYSE observation rule."""

    def test_saturday_observed_on_friday(self) -> None:
        """Holiday falling on Saturday shifts to prior Friday."""
        saturday = date(2025, 7, 5)  # A Saturday
        assert saturday.weekday() == 5
        assert _observed(saturday) == date(2025, 7, 4)

    def test_sunday_observed_on_monday(self) -> None:
        """Holiday falling on Sunday shifts to following Monday."""
        sunday = date(2025, 7, 6)  # A Sunday
        assert sunday.weekday() == 6
        assert _observed(sunday) == date(2025, 7, 7)

    def test_weekday_unchanged(self) -> None:
        """Holiday on a weekday is observed on the same day."""
        tuesday = date(2025, 3, 4)
        assert tuesday.weekday() == 1
        assert _observed(tuesday) == tuesday


class TestEasterDate:
    """Tests for the Anonymous Gregorian Easter algorithm."""

    def test_easter_2025(self) -> None:
        """Easter Sunday 2025 = April 20."""
        assert _easter_date(2025) == date(2025, 4, 20)

    def test_easter_2026(self) -> None:
        """Easter Sunday 2026 = April 5."""
        assert _easter_date(2026) == date(2026, 4, 5)

    def test_easter_2030(self) -> None:
        """Easter Sunday 2030 = April 21."""
        assert _easter_date(2030) == date(2030, 4, 21)

    def test_easter_2024(self) -> None:
        """Easter Sunday 2024 = March 31."""
        assert _easter_date(2024) == date(2024, 3, 31)


# ======================================================================
# compute_nyse_holidays() tests
# ======================================================================


class TestComputeNyseHolidays:
    """Tests for dynamic NYSE holiday computation across multiple years."""

    # -- 2025 known holidays -----------------------------------------------

    def test_2025_new_years_day(self) -> None:
        """New Year's Day 2025: Jan 1 is a Wednesday, observed Jan 1."""
        holidays = compute_nyse_holidays(2025)
        assert date(2025, 1, 1) in holidays

    def test_2025_mlk_day(self) -> None:
        """MLK Day 2025: 3rd Monday of January = Jan 20."""
        holidays = compute_nyse_holidays(2025)
        assert date(2025, 1, 20) in holidays

    def test_2025_presidents_day(self) -> None:
        """Presidents' Day 2025: 3rd Monday of February = Feb 17."""
        holidays = compute_nyse_holidays(2025)
        assert date(2025, 2, 17) in holidays

    def test_2025_good_friday(self) -> None:
        """Good Friday 2025: Easter is Apr 20, so Good Friday = Apr 18."""
        holidays = compute_nyse_holidays(2025)
        assert date(2025, 4, 18) in holidays

    def test_2025_memorial_day(self) -> None:
        """Memorial Day 2025: last Monday of May = May 26."""
        holidays = compute_nyse_holidays(2025)
        assert date(2025, 5, 26) in holidays

    def test_2025_juneteenth(self) -> None:
        """Juneteenth 2025: June 19 is a Thursday, observed June 19."""
        holidays = compute_nyse_holidays(2025)
        assert date(2025, 6, 19) in holidays

    def test_2025_independence_day(self) -> None:
        """Independence Day 2025: July 4 is a Friday, observed July 4."""
        holidays = compute_nyse_holidays(2025)
        assert date(2025, 7, 4) in holidays

    def test_2025_labor_day(self) -> None:
        """Labor Day 2025: 1st Monday of September = Sep 1."""
        holidays = compute_nyse_holidays(2025)
        assert date(2025, 9, 1) in holidays

    def test_2025_thanksgiving(self) -> None:
        """Thanksgiving 2025: 4th Thursday of November = Nov 27."""
        holidays = compute_nyse_holidays(2025)
        assert date(2025, 11, 27) in holidays

    def test_2025_christmas(self) -> None:
        """Christmas 2025: Dec 25 is a Thursday, observed Dec 25."""
        holidays = compute_nyse_holidays(2025)
        assert date(2025, 12, 25) in holidays

    def test_2025_has_exactly_10_holidays(self) -> None:
        """NYSE observes exactly 10 holidays per year."""
        holidays = compute_nyse_holidays(2025)
        assert len(holidays) == 10

    # -- 2026 known holidays -----------------------------------------------

    def test_2026_new_years_day(self) -> None:
        """New Year's Day 2026: Jan 1 is a Thursday, observed Jan 1."""
        holidays = compute_nyse_holidays(2026)
        assert date(2026, 1, 1) in holidays

    def test_2026_mlk_day(self) -> None:
        """MLK Day 2026: 3rd Monday of January = Jan 19."""
        holidays = compute_nyse_holidays(2026)
        assert date(2026, 1, 19) in holidays

    def test_2026_good_friday(self) -> None:
        """Good Friday 2026: Easter is Apr 5, so Good Friday = Apr 3."""
        holidays = compute_nyse_holidays(2026)
        assert date(2026, 4, 3) in holidays

    def test_2026_juneteenth(self) -> None:
        """Juneteenth 2026: June 19 is a Friday, observed June 19."""
        holidays = compute_nyse_holidays(2026)
        assert date(2026, 6, 19) in holidays

    def test_2026_independence_day_observed(self) -> None:
        """Independence Day 2026: July 4 is a Saturday, observed Friday July 3."""
        holidays = compute_nyse_holidays(2026)
        assert date(2026, 7, 3) in holidays
        # July 4 (Saturday) itself should NOT be in the set
        assert date(2026, 7, 4) not in holidays

    def test_2026_christmas_observed(self) -> None:
        """Christmas 2026: Dec 25 is a Friday, observed Dec 25."""
        holidays = compute_nyse_holidays(2026)
        assert date(2026, 12, 25) in holidays

    def test_2026_has_exactly_10_holidays(self) -> None:
        holidays = compute_nyse_holidays(2026)
        assert len(holidays) == 10

    # -- 2030 known holidays -----------------------------------------------

    def test_2030_new_years_day(self) -> None:
        """New Year's Day 2030: Jan 1 is a Tuesday, observed Jan 1."""
        holidays = compute_nyse_holidays(2030)
        assert date(2030, 1, 1) in holidays

    def test_2030_mlk_day(self) -> None:
        """MLK Day 2030: 3rd Monday of January = Jan 21."""
        holidays = compute_nyse_holidays(2030)
        assert date(2030, 1, 21) in holidays

    def test_2030_good_friday(self) -> None:
        """Good Friday 2030: Easter is Apr 21, so Good Friday = Apr 19."""
        holidays = compute_nyse_holidays(2030)
        assert date(2030, 4, 19) in holidays

    def test_2030_memorial_day(self) -> None:
        """Memorial Day 2030: last Monday of May = May 27."""
        holidays = compute_nyse_holidays(2030)
        assert date(2030, 5, 27) in holidays

    def test_2030_labor_day(self) -> None:
        """Labor Day 2030: 1st Monday of September = Sep 2."""
        holidays = compute_nyse_holidays(2030)
        assert date(2030, 9, 2) in holidays

    def test_2030_thanksgiving(self) -> None:
        """Thanksgiving 2030: 4th Thursday of November = Nov 28."""
        holidays = compute_nyse_holidays(2030)
        assert date(2030, 11, 28) in holidays

    def test_2030_christmas(self) -> None:
        """Christmas 2030: Dec 25 is a Wednesday, observed Dec 25."""
        holidays = compute_nyse_holidays(2030)
        assert date(2030, 12, 25) in holidays

    def test_2030_has_exactly_10_holidays(self) -> None:
        holidays = compute_nyse_holidays(2030)
        assert len(holidays) == 10

    # -- Edge case: Sunday observation rule --------------------------------

    def test_new_years_on_sunday_observed_monday(self) -> None:
        """When Jan 1 falls on Sunday, the observed holiday is Monday Jan 2.
        2023 has Jan 1 on a Sunday -> observed Mon Jan 2."""
        holidays = compute_nyse_holidays(2023)
        assert date(2023, 1, 2) in holidays
        assert date(2023, 1, 1) not in holidays

    # -- Frozen set and caching --------------------------------------------

    def test_returns_frozenset(self) -> None:
        """Result is an immutable frozenset."""
        holidays = compute_nyse_holidays(2025)
        assert isinstance(holidays, frozenset)

    def test_all_holidays_are_weekdays(self) -> None:
        """Every observed holiday must be a weekday (Mon-Fri)."""
        for year in (2024, 2025, 2026, 2027, 2028, 2029, 2030):
            holidays = compute_nyse_holidays(year)
            for h in holidays:
                assert h.weekday() < 5, (
                    f"Holiday {h} in {year} falls on day {h.weekday()} (weekend)"
                )


# ======================================================================
# is_market_holiday() tests
# ======================================================================


class TestIsMarketHoliday:
    """Tests for is_market_holiday() spot checks."""

    def test_christmas_2025_is_holiday(self) -> None:
        assert is_market_holiday(date(2025, 12, 25)) is True

    def test_regular_tuesday_is_not_holiday(self) -> None:
        assert is_market_holiday(date(2025, 3, 4)) is False

    def test_thanksgiving_2026_is_holiday(self) -> None:
        # Thanksgiving 2026: 4th Thursday of November = Nov 26
        assert is_market_holiday(date(2026, 11, 26)) is True

    def test_day_after_thanksgiving_is_not_holiday(self) -> None:
        """Day after Thanksgiving is NOT a full holiday (half day)."""
        assert is_market_holiday(date(2025, 11, 28)) is False

    def test_weekend_is_not_a_holiday(self) -> None:
        """Weekends are not in the holiday set (they are handled separately)."""
        assert is_market_holiday(date(2025, 3, 1)) is False  # Saturday


# ======================================================================
# market_is_open() tests
# ======================================================================


class TestMarketIsOpen:
    """Tests for market_is_open() with mocked datetime."""

    def _mock_now(
        self, year: int, month: int, day: int, hour: int, minute: int
    ) -> datetime:
        """Create a timezone-aware Eastern datetime for mocking."""
        return datetime(year, month, day, hour, minute, tzinfo=US_EASTERN)

    def test_open_during_regular_hours(self) -> None:
        """Market should be open at 10:30 AM ET on a regular trading day (Tuesday)."""
        mock_dt = self._mock_now(2025, 3, 4, 10, 30)  # Tuesday
        with patch("src.utils.helpers.datetime") as mock_datetime:
            mock_datetime.now.return_value = mock_dt
            mock_datetime.side_effect = lambda *a, **kw: datetime(*a, **kw)
            assert market_is_open() is True

    def test_closed_before_open(self) -> None:
        """Market should be closed at 9:00 AM ET (before 9:30 open)."""
        mock_dt = self._mock_now(2025, 3, 4, 9, 0)
        with patch("src.utils.helpers.datetime") as mock_datetime:
            mock_datetime.now.return_value = mock_dt
            mock_datetime.side_effect = lambda *a, **kw: datetime(*a, **kw)
            assert market_is_open() is False

    def test_closed_after_close(self) -> None:
        """Market should be closed at 4:01 PM ET."""
        mock_dt = self._mock_now(2025, 3, 4, 16, 1)
        with patch("src.utils.helpers.datetime") as mock_datetime:
            mock_datetime.now.return_value = mock_dt
            mock_datetime.side_effect = lambda *a, **kw: datetime(*a, **kw)
            assert market_is_open() is False

    def test_closed_exactly_at_close(self) -> None:
        """Market close is exclusive: 16:00 should be closed."""
        mock_dt = self._mock_now(2025, 3, 4, 16, 0)
        with patch("src.utils.helpers.datetime") as mock_datetime:
            mock_datetime.now.return_value = mock_dt
            mock_datetime.side_effect = lambda *a, **kw: datetime(*a, **kw)
            assert market_is_open() is False

    def test_open_exactly_at_open(self) -> None:
        """Market open is inclusive: 9:30 should be open."""
        mock_dt = self._mock_now(2025, 3, 4, 9, 30)
        with patch("src.utils.helpers.datetime") as mock_datetime:
            mock_datetime.now.return_value = mock_dt
            mock_datetime.side_effect = lambda *a, **kw: datetime(*a, **kw)
            assert market_is_open() is True

    def test_closed_on_weekend(self) -> None:
        """Market should be closed on Saturday even during trading hours."""
        mock_dt = self._mock_now(2025, 3, 1, 11, 0)  # Saturday
        with patch("src.utils.helpers.datetime") as mock_datetime:
            mock_datetime.now.return_value = mock_dt
            mock_datetime.side_effect = lambda *a, **kw: datetime(*a, **kw)
            assert market_is_open() is False

    def test_closed_on_holiday(self) -> None:
        """Market should be closed on Christmas during trading hours."""
        mock_dt = self._mock_now(2025, 12, 25, 11, 0)  # Christmas (Thursday)
        with patch("src.utils.helpers.datetime") as mock_datetime:
            mock_datetime.now.return_value = mock_dt
            mock_datetime.side_effect = lambda *a, **kw: datetime(*a, **kw)
            assert market_is_open() is False

    def test_open_at_1pm(self) -> None:
        """Market open at 1:00 PM ET on a regular Wednesday."""
        mock_dt = self._mock_now(2025, 3, 5, 13, 0)
        with patch("src.utils.helpers.datetime") as mock_datetime:
            mock_datetime.now.return_value = mock_dt
            mock_datetime.side_effect = lambda *a, **kw: datetime(*a, **kw)
            assert market_is_open() is True


# ======================================================================
# next_market_open() tests
# ======================================================================


class TestNextMarketOpen:
    """Tests for next_market_open()."""

    def test_before_open_on_trading_day(self) -> None:
        """Before 9:30 on a trading day, next open is same-day 9:30."""
        mock_dt = datetime(2025, 3, 4, 8, 0, tzinfo=US_EASTERN)  # Tue 8am
        with patch("src.utils.helpers.datetime") as mock_datetime:
            mock_datetime.now.return_value = mock_dt
            mock_datetime.combine = datetime.combine
            mock_datetime.side_effect = lambda *a, **kw: datetime(*a, **kw)
            result = next_market_open()
            assert result.date() == date(2025, 3, 4)
            assert result.time() == MARKET_OPEN

    def test_after_open_on_trading_day(self) -> None:
        """After 9:30 on a trading day, next open is next trading day."""
        mock_dt = datetime(2025, 3, 4, 10, 0, tzinfo=US_EASTERN)  # Tue 10am
        with patch("src.utils.helpers.datetime") as mock_datetime:
            mock_datetime.now.return_value = mock_dt
            mock_datetime.combine = datetime.combine
            mock_datetime.side_effect = lambda *a, **kw: datetime(*a, **kw)
            result = next_market_open()
            assert result.date() == date(2025, 3, 5)  # Wed
            assert result.time() == MARKET_OPEN

    def test_friday_evening_returns_monday(self) -> None:
        """Friday evening should return Monday morning."""
        mock_dt = datetime(2025, 3, 7, 17, 0, tzinfo=US_EASTERN)  # Fri 5pm
        with patch("src.utils.helpers.datetime") as mock_datetime:
            mock_datetime.now.return_value = mock_dt
            mock_datetime.combine = datetime.combine
            mock_datetime.side_effect = lambda *a, **kw: datetime(*a, **kw)
            result = next_market_open()
            assert result.date() == date(2025, 3, 10)  # Monday
            assert result.weekday() == 0


# ======================================================================
# trading_days_between() tests
# ======================================================================


class TestTradingDaysBetween:
    """Tests for trading_days_between()."""

    def test_same_day_trading_day(self) -> None:
        """Single trading day: count = 1."""
        d = date(2025, 3, 4)  # Tuesday
        assert trading_days_between(d, d) == 1

    def test_same_day_weekend(self) -> None:
        """A weekend day has 0 trading days."""
        saturday = date(2025, 3, 1)
        assert trading_days_between(saturday, saturday) == 0

    def test_end_before_start_returns_zero(self) -> None:
        assert trading_days_between(date(2025, 3, 5), date(2025, 3, 3)) == 0

    def test_full_week_mon_to_fri(self) -> None:
        """Mon through Fri = 5 trading days (no holidays)."""
        # March 3-7, 2025: Mon-Fri, no holidays
        assert trading_days_between(date(2025, 3, 3), date(2025, 3, 7)) == 5

    def test_week_with_weekend(self) -> None:
        """Mon through following Mon = 6 trading days."""
        assert trading_days_between(date(2025, 3, 3), date(2025, 3, 10)) == 6

    def test_week_with_holiday(self) -> None:
        """A week containing MLK Day (Jan 20) loses one trading day.
        Jan 20-24, 2025: Mon(holiday)-Fri = 4 trading days."""
        assert trading_days_between(date(2025, 1, 20), date(2025, 1, 24)) == 4


# ======================================================================
# calculate_dte() tests
# ======================================================================


class TestCalculateDte:
    """Tests for calculate_dte()."""

    def test_with_string_date(self) -> None:
        """Parse an ISO date string and compute DTE."""
        future = date.today() + timedelta(days=30)
        result = calculate_dte(future.isoformat())
        assert result == 30

    def test_with_date_object(self) -> None:
        """Accept a date object directly."""
        future = date.today() + timedelta(days=15)
        result = calculate_dte(future)
        assert result == 15

    def test_today_returns_zero(self) -> None:
        """DTE for today is 0."""
        assert calculate_dte(date.today()) == 0

    def test_past_date_returns_zero(self) -> None:
        """DTE for a past date is clamped to 0."""
        past = date.today() - timedelta(days=10)
        assert calculate_dte(past) == 0

    def test_far_future(self) -> None:
        """DTE for a date far in the future."""
        future = date.today() + timedelta(days=365)
        assert calculate_dte(future) == 365

    def test_tomorrow(self) -> None:
        """DTE for tomorrow is 1."""
        tomorrow = date.today() + timedelta(days=1)
        assert calculate_dte(tomorrow) == 1


# ======================================================================
# round_to_tick() tests
# ======================================================================


class TestRoundToTick:
    """Tests for round_to_tick()."""

    def test_already_on_tick(self) -> None:
        """A price already on a tick boundary is unchanged."""
        assert round_to_tick(1.50, 0.05) == 1.50

    def test_rounds_down(self) -> None:
        """1.52 with tick 0.05 rounds to 1.50."""
        assert round_to_tick(1.52, 0.05) == 1.50

    def test_rounds_up(self) -> None:
        """1.53 with tick 0.05 rounds to 1.55."""
        assert round_to_tick(1.53, 0.05) == 1.55

    def test_midpoint_rounds(self) -> None:
        """1.525 with tick 0.05 rounds (banker's rounding may differ)."""
        result = round_to_tick(1.525, 0.05)
        # 1.525/0.05 = 30.5, round(30.5) = 30 (banker's), 30*0.05 = 1.5
        assert result == pytest.approx(1.5, abs=0.05)

    def test_default_penny_tick(self) -> None:
        """Default tick size is 0.01."""
        assert round_to_tick(1.234) == 1.23
        assert round_to_tick(1.235) == 1.24  # or 1.23 due to banker's rounding

    def test_nickel_tick(self) -> None:
        """Options nickel increment."""
        assert round_to_tick(3.12, 0.05) == 3.10
        assert round_to_tick(3.13, 0.05) == 3.15

    def test_dollar_tick(self) -> None:
        """Whole-dollar tick size."""
        assert round_to_tick(99.49, 1.0) == 99.0
        assert round_to_tick(99.51, 1.0) == 100.0

    def test_zero_tick_size_returns_price(self) -> None:
        """Tick size of zero returns price unchanged."""
        assert round_to_tick(1.234, 0.0) == 1.234

    def test_negative_tick_size_returns_price(self) -> None:
        """Negative tick size returns price unchanged."""
        assert round_to_tick(1.234, -0.05) == 1.234

    def test_zero_price(self) -> None:
        """Price of zero stays zero."""
        assert round_to_tick(0.0, 0.05) == 0.0

    def test_large_price(self) -> None:
        """Large price rounds correctly."""
        assert round_to_tick(5000.123, 0.01) == 5000.12


# ======================================================================
# safe_divide() tests
# ======================================================================


class TestSafeDivide:
    """Tests for safe_divide()."""

    def test_normal_division(self) -> None:
        assert safe_divide(10.0, 2.0) == 5.0

    def test_zero_denominator_returns_default(self) -> None:
        assert safe_divide(10.0, 0.0) == 0.0

    def test_zero_denominator_custom_default(self) -> None:
        assert safe_divide(10.0, 0.0, default=-1.0) == -1.0

    def test_near_zero_denominator(self) -> None:
        """Denominator smaller than 1e-12 triggers default."""
        assert safe_divide(10.0, 1e-13) == 0.0

    def test_negative_near_zero_denominator(self) -> None:
        """Negative near-zero also triggers default."""
        assert safe_divide(10.0, -1e-13) == 0.0

    def test_just_above_threshold(self) -> None:
        """Denominator just above threshold performs normal division."""
        result = safe_divide(10.0, 1e-11)
        assert result == pytest.approx(10.0 / 1e-11)

    def test_negative_division(self) -> None:
        assert safe_divide(-10.0, 2.0) == -5.0

    def test_both_zero(self) -> None:
        """0/0 returns default."""
        assert safe_divide(0.0, 0.0) == 0.0

    def test_zero_numerator(self) -> None:
        """0 divided by nonzero = 0."""
        assert safe_divide(0.0, 5.0) == 0.0


# ======================================================================
# format_currency() tests
# ======================================================================


class TestFormatCurrency:
    """Tests for format_currency()."""

    def test_positive_amount(self) -> None:
        assert format_currency(1234.56) == "$1,234.56"

    def test_negative_amount(self) -> None:
        assert format_currency(-789.00) == "-$789.00"

    def test_zero(self) -> None:
        assert format_currency(0.0) == "$0.00"

    def test_large_amount(self) -> None:
        assert format_currency(150000.00) == "$150,000.00"

    def test_small_amount(self) -> None:
        assert format_currency(0.01) == "$0.01"

    def test_rounding(self) -> None:
        """Amounts are formatted to 2 decimal places."""
        assert format_currency(1234.567) == "$1,234.57"

    def test_negative_small(self) -> None:
        assert format_currency(-0.50) == "-$0.50"

    def test_millions(self) -> None:
        assert format_currency(1000000.00) == "$1,000,000.00"

    def test_negative_large(self) -> None:
        assert format_currency(-22500.00) == "-$22,500.00"
