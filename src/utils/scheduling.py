"""APScheduler-based task scheduling for Project Titan.

Provides a centralized scheduler that manages all recurring jobs required
by the trading system: market-open scans, intraday opportunity checks,
position exit evaluations, risk monitoring, end-of-day journaling, weekly
model retraining, and monthly reporting.

All scheduled times use the ``America/New_York`` timezone to align with
US equity market hours.

Usage::

    from src.utils.scheduling import TitanScheduler

    scheduler = TitanScheduler()
    scheduler.register_callbacks({
        "market_open_scan": run_full_scan,
        "position_check": check_exits,
    })
    await scheduler.start()
"""

from __future__ import annotations

import asyncio
import time as _time
from typing import TYPE_CHECKING, Any

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from src.utils.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine
    from datetime import datetime

# -----------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------

TIMEZONE: str = "US/Eastern"

# Market hours boundaries (Eastern Time)
MARKET_OPEN_HOUR: int = 9
MARKET_OPEN_MINUTE: int = 30
MARKET_CLOSE_HOUR: int = 16
MARKET_CLOSE_MINUTE: int = 0

# Weekday constants for APScheduler cron (mon-fri)
WEEKDAYS: str = "mon-fri"

# Job names — single source of truth for all job identifiers
JOB_MARKET_OPEN_SCAN: str = "market_open_scan"
JOB_INTRADAY_SCAN_1: str = "intraday_scan_1"
JOB_INTRADAY_SCAN_2: str = "intraday_scan_2"
JOB_INTRADAY_SCAN_3: str = "intraday_scan_3"
JOB_POSITION_CHECK: str = "position_check"
JOB_RISK_MONITOR: str = "risk_monitor"
JOB_EOD_JOURNAL: str = "eod_journal"
JOB_DAILY_SUMMARY: str = "daily_summary"
JOB_DAILY_CLEANUP: str = "daily_cleanup"
JOB_WEEKLY_RETRAIN: str = "weekly_retrain"
JOB_WEEKLY_REPORT: str = "weekly_report"
JOB_EVENT_CALENDAR_REFRESH: str = "event_calendar_refresh"
JOB_MONTHLY_REPORT: str = "monthly_report"

# Position check interval in minutes
POSITION_CHECK_INTERVAL_MINUTES: int = 15

# Risk monitor interval in minutes
RISK_MONITOR_INTERVAL_MINUTES: int = 5

logger = get_logger("utils.scheduling")


# -----------------------------------------------------------------------
# Job definitions — each entry describes one scheduled task
# -----------------------------------------------------------------------

_JOB_DEFINITIONS: list[dict[str, Any]] = [
    # -- Market hours jobs (cron-based, specific times) --
    {
        "name": JOB_MARKET_OPEN_SCAN,
        "trigger_type": "cron",
        "trigger_kwargs": {
            "day_of_week": WEEKDAYS,
            "hour": 9,
            "minute": 35,
            "timezone": TIMEZONE,
        },
        "description": ("Full universe scan: regime + signals + strategy selection"),
    },
    {
        "name": JOB_INTRADAY_SCAN_1,
        "trigger_type": "cron",
        "trigger_kwargs": {
            "day_of_week": WEEKDAYS,
            "hour": 11,
            "minute": 30,
            "timezone": TIMEZONE,
        },
        "description": "Intraday opportunity scan (11:30 AM ET)",
    },
    {
        "name": JOB_INTRADAY_SCAN_2,
        "trigger_type": "cron",
        "trigger_kwargs": {
            "day_of_week": WEEKDAYS,
            "hour": 13,
            "minute": 30,
            "timezone": TIMEZONE,
        },
        "description": "Intraday opportunity scan (1:30 PM ET)",
    },
    {
        "name": JOB_INTRADAY_SCAN_3,
        "trigger_type": "cron",
        "trigger_kwargs": {
            "day_of_week": WEEKDAYS,
            "hour": 15,
            "minute": 30,
            "timezone": TIMEZONE,
        },
        "description": "Intraday opportunity scan (3:30 PM ET)",
    },
    # -- Market hours jobs (interval-based, during trading hours) --
    {
        "name": JOB_POSITION_CHECK,
        "trigger_type": "cron_interval",
        "trigger_kwargs": {
            "day_of_week": WEEKDAYS,
            "hour": f"{MARKET_OPEN_HOUR}-{MARKET_CLOSE_HOUR - 1}",
            "minute": "*/15",
            "timezone": TIMEZONE,
        },
        "description": ("Check open positions for exit criteria every 15 min"),
    },
    {
        "name": JOB_RISK_MONITOR,
        "trigger_type": "cron_interval",
        "trigger_kwargs": {
            "day_of_week": WEEKDAYS,
            "hour": f"{MARKET_OPEN_HOUR}-{MARKET_CLOSE_HOUR - 1}",
            "minute": "*/5",
            "timezone": TIMEZONE,
        },
        "description": "Update risk metrics and P&L every 5 min",
    },
    # -- After hours jobs --
    {
        "name": JOB_EOD_JOURNAL,
        "trigger_type": "cron",
        "trigger_kwargs": {
            "day_of_week": WEEKDAYS,
            "hour": 16,
            "minute": 15,
            "timezone": TIMEZONE,
        },
        "description": "Journal Agent reviews all trades",
    },
    {
        "name": JOB_DAILY_SUMMARY,
        "trigger_type": "cron",
        "trigger_kwargs": {
            "day_of_week": WEEKDAYS,
            "hour": 16,
            "minute": 30,
            "timezone": TIMEZONE,
        },
        "description": "Daily P&L summary to Telegram",
    },
    {
        "name": JOB_DAILY_CLEANUP,
        "trigger_type": "cron",
        "trigger_kwargs": {
            "day_of_week": WEEKDAYS,
            "hour": 17,
            "minute": 0,
            "timezone": TIMEZONE,
        },
        "description": ("Update signal databases and cache cleanup"),
    },
    # -- Weekly jobs --
    {
        "name": JOB_WEEKLY_RETRAIN,
        "trigger_type": "cron",
        "trigger_kwargs": {
            "day_of_week": "sat",
            "hour": 6,
            "minute": 0,
            "timezone": TIMEZONE,
        },
        "description": ("Weekly model retraining + Optuna optimization"),
    },
    {
        "name": JOB_WEEKLY_REPORT,
        "trigger_type": "cron",
        "trigger_kwargs": {
            "day_of_week": "sat",
            "hour": 7,
            "minute": 0,
            "timezone": TIMEZONE,
        },
        "description": "Generate weekly QuantStats report",
    },
    # -- Daily jobs --
    {
        "name": JOB_EVENT_CALENDAR_REFRESH,
        "trigger_type": "cron",
        "trigger_kwargs": {
            "day_of_week": "mon-sat",
            "hour": 8,
            "minute": 0,
            "timezone": TIMEZONE,
        },
        "description": ("Refresh earnings, FOMC, CPI dates"),
    },
    # -- Monthly jobs --
    {
        "name": JOB_MONTHLY_REPORT,
        "trigger_type": "cron",
        "trigger_kwargs": {
            "day": 1,
            "hour": 7,
            "minute": 0,
            "timezone": TIMEZONE,
        },
        "description": "Generate monthly performance report",
    },
]


class TitanScheduler:
    """Central scheduler for all recurring Titan tasks.

    Wraps :class:`apscheduler.schedulers.asyncio.AsyncIOScheduler` and
    provides a callback-registration interface so the rest of the
    codebase does not need to know APScheduler internals.

    All time references use the ``US/Eastern`` timezone to align with
    US equity market hours.

    Usage::

        scheduler = TitanScheduler()
        scheduler.register_callbacks({
            "market_open_scan": my_scan_func,
            "position_check": my_exit_check_func,
        })
        await scheduler.start()
    """

    def __init__(self) -> None:
        """Initialise the underlying APScheduler instance."""
        self._scheduler = AsyncIOScheduler(timezone=TIMEZONE)
        self._callbacks: dict[str, Callable[..., Coroutine[Any, Any, Any]]] = {}
        self._started: bool = False

    # ---------------------------------------------------------------
    # Callback registration
    # ---------------------------------------------------------------

    def register_callbacks(
        self,
        callbacks: dict[str, Callable[..., Coroutine[Any, Any, Any]]],
    ) -> None:
        """Register async callback functions for scheduled jobs.

        The *callbacks* dict maps job names (e.g. ``"market_open_scan"``)
        to async callables.  Only jobs whose names appear in *callbacks*
        will be scheduled when :meth:`start` is called.

        Parameters
        ----------
        callbacks:
            Mapping of job name to async callable.  Unrecognised keys
            are logged as warnings but do not prevent startup.
        """
        known_names = {d["name"] for d in _JOB_DEFINITIONS}
        for name, func in callbacks.items():
            if name not in known_names:
                logger.warning(
                    "unknown job name registered — will be treated as custom job",
                    job_name=name,
                )
            self._callbacks[name] = func
            logger.debug(
                "callback registered",
                job_name=name,
                callable=getattr(func, "__qualname__", str(func)),
            )

    # ---------------------------------------------------------------
    # Job wrapper
    # ---------------------------------------------------------------

    async def _execute_job(self, name: str) -> None:
        """Execute a registered callback with logging and error handling.

        Parameters
        ----------
        name:
            The job name whose callback should be invoked.
        """
        callback = self._callbacks.get(name)
        if callback is None:
            logger.warning(
                "no callback registered for job",
                job_name=name,
            )
            return

        start = _time.monotonic()
        logger.info("job started", job_name=name)
        try:
            await callback()
        except asyncio.CancelledError:
            logger.warning("job cancelled", job_name=name)
            raise
        except Exception:
            elapsed = _time.monotonic() - start
            logger.exception(
                "job failed",
                job_name=name,
                elapsed_seconds=round(elapsed, 3),
            )
            return

        elapsed = _time.monotonic() - start
        logger.info(
            "job completed",
            job_name=name,
            elapsed_seconds=round(elapsed, 3),
        )

    # ---------------------------------------------------------------
    # Internal: build trigger from job definition
    # ---------------------------------------------------------------

    def _build_trigger(
        self,
        trigger_type: str,
        trigger_kwargs: dict[str, Any],
    ) -> CronTrigger | IntervalTrigger:
        """Build an APScheduler trigger from a job definition dict.

        Parameters
        ----------
        trigger_type:
            One of ``"cron"``, ``"cron_interval"``, or ``"interval"``.
        trigger_kwargs:
            Keyword arguments forwarded to the trigger constructor.

        Returns
        -------
        CronTrigger | IntervalTrigger
            The constructed APScheduler trigger.
        """
        if trigger_type in ("cron", "cron_interval"):
            return CronTrigger(**trigger_kwargs)
        return IntervalTrigger(**trigger_kwargs)

    # ---------------------------------------------------------------
    # Lifecycle
    # ---------------------------------------------------------------

    async def start(self) -> None:
        """Start the scheduler and register all jobs with callbacks.

        Jobs whose names do not have a registered callback are skipped
        with a debug-level log message.  The scheduler itself is started
        after all jobs have been added.
        """
        if self._started:
            logger.warning("scheduler already started")
            return

        jobs_added = 0
        for defn in _JOB_DEFINITIONS:
            name = defn["name"]
            if name not in self._callbacks:
                logger.debug(
                    "skipping job — no callback registered",
                    job_name=name,
                )
                continue

            trigger = self._build_trigger(
                defn["trigger_type"],
                defn["trigger_kwargs"],
            )
            self._scheduler.add_job(
                self._execute_job,
                trigger=trigger,
                args=[name],
                id=name,
                name=defn.get("description", name),
                replace_existing=True,
                misfire_grace_time=300,
            )
            jobs_added += 1
            logger.debug(
                "job scheduled",
                job_name=name,
                description=defn.get("description", ""),
            )

        self._scheduler.start()
        self._started = True
        logger.info(
            "scheduler started",
            total_jobs=jobs_added,
            skipped=len(_JOB_DEFINITIONS) - jobs_added,
        )

    async def stop(self) -> None:
        """Shut down the scheduler gracefully.

        Waits for currently executing jobs to finish before returning.
        """
        if not self._started:
            logger.warning("scheduler not running — nothing to stop")
            return

        self._scheduler.shutdown(wait=True)
        self._started = False
        logger.info("scheduler stopped")

    # ---------------------------------------------------------------
    # Custom job management
    # ---------------------------------------------------------------

    def add_job(
        self,
        name: str,
        func: Callable[..., Coroutine[Any, Any, Any]],
        trigger: CronTrigger | IntervalTrigger,
        **kwargs: Any,
    ) -> None:
        """Add a custom job to the scheduler.

        This bypasses the predefined job definitions and registers
        a one-off job directly with the APScheduler instance.

        Parameters
        ----------
        name:
            Unique identifier for the job.
        func:
            Async callable to execute.
        trigger:
            An APScheduler trigger (``CronTrigger`` or
            ``IntervalTrigger``).
        **kwargs:
            Additional keyword arguments forwarded to
            ``scheduler.add_job()``.
        """
        self._callbacks[name] = func
        self._scheduler.add_job(
            self._execute_job,
            trigger=trigger,
            args=[name],
            id=name,
            name=name,
            replace_existing=True,
            misfire_grace_time=kwargs.pop("misfire_grace_time", 300),
            **kwargs,
        )
        logger.info("custom job added", job_name=name)

    def remove_job(self, name: str) -> None:
        """Remove a job by name.

        Parameters
        ----------
        name:
            The job identifier to remove.  If the job does not exist,
            a warning is logged but no exception is raised.
        """
        try:
            self._scheduler.remove_job(name)
            self._callbacks.pop(name, None)
            logger.info("job removed", job_name=name)
        except Exception:
            logger.warning(
                "failed to remove job — may not exist",
                job_name=name,
            )

    # ---------------------------------------------------------------
    # Inspection
    # ---------------------------------------------------------------

    def get_next_run_times(self) -> dict[str, datetime | None]:
        """Return a dict mapping job names to their next scheduled run.

        Returns
        -------
        dict[str, datetime | None]
            Job name to next fire time.  The value is ``None`` if the
            job is paused or has no upcoming execution.
        """
        result: dict[str, datetime | None] = {}
        for job in self._scheduler.get_jobs():
            result[job.id] = job.next_run_time
        return result

    # ---------------------------------------------------------------
    # Pause / resume
    # ---------------------------------------------------------------

    def pause_all(self) -> None:
        """Pause all scheduled jobs.

        Typically invoked when a circuit breaker reaches the HALT
        level.  Jobs remain registered but will not fire until
        :meth:`resume_all` is called.
        """
        for job in self._scheduler.get_jobs():
            job.pause()
        logger.warning("all jobs paused")

    def resume_all(self) -> None:
        """Resume all previously paused jobs.

        Called when circuit breaker conditions are cleared and trading
        may resume.
        """
        for job in self._scheduler.get_jobs():
            job.resume()
        logger.info("all jobs resumed")

    # ---------------------------------------------------------------
    # Status
    # ---------------------------------------------------------------

    def is_running(self) -> bool:
        """Check if the scheduler is currently running.

        Returns
        -------
        bool
            ``True`` if the scheduler has been started and has not yet
            been shut down.
        """
        return self._started and self._scheduler.running
