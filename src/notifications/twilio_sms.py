"""Twilio SMS alerting for critical system events.

Sends SMS messages for high-severity conditions such as prolonged
connectivity loss, circuit-breaker triggers, and emergency stops.
Enforces a strict per-condition rate limit of one SMS per hour to
avoid alert fatigue and Twilio costs.

Usage::

    from src.notifications.twilio_sms import TwilioSMSNotifier

    notifier = TwilioSMSNotifier(
        account_sid="AC...",
        auth_token="secret",
        from_number="+12025551234",
        to_number="+12025559876",
    )
    await notifier.send_connectivity_alert(minutes_down=7)
"""

from __future__ import annotations

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from enum import StrEnum
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field
from twilio.rest import Client as TwilioClient

from src.utils.logging import get_logger

if TYPE_CHECKING:
    import structlog

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Rate limit: at most one SMS per condition within this window (seconds).
_RATE_LIMIT_SECONDS: int = 3600  # 1 hour

# SMS single-segment length (GSM-7 encoding).
_SMS_SINGLE_SEGMENT_LENGTH: int = 160

# Thread pool for blocking Twilio API calls.
_TWILIO_THREAD_POOL_SIZE: int = 2


# ---------------------------------------------------------------------------
# Predefined alert conditions
# ---------------------------------------------------------------------------


class AlertCondition(StrEnum):
    """Predefined conditions that may trigger an SMS alert."""

    CONNECTIVITY_LOSS = "CONNECTIVITY_LOSS"
    CIRCUIT_BREAKER = "CIRCUIT_BREAKER"
    EMERGENCY_STOP = "EMERGENCY_STOP"
    SYSTEM_ERROR = "SYSTEM_ERROR"


class AlertSeverity(StrEnum):
    """Severity levels for SMS alerts."""

    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class SMSAlert(BaseModel):
    """Represents a single SMS alert event.

    Used internally for logging and tracking purposes.
    """

    severity: AlertSeverity = Field(description="Alert severity: CRITICAL or WARNING")
    condition: str = Field(description="The alert condition identifier")
    message: str = Field(description="Human-readable alert message body")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="UTC timestamp when the alert was created",
    )


# ---------------------------------------------------------------------------
# Notifier
# ---------------------------------------------------------------------------


class TwilioSMSNotifier:
    """SMS notification channel for critical trading system events.

    Wraps the Twilio REST API with per-condition rate limiting and
    async-safe execution (blocking Twilio calls are offloaded to a
    thread pool).

    Args:
        account_sid: Twilio account SID.
        auth_token: Twilio auth token.
        from_number: Originating phone number in E.164 format.
        to_number: Destination phone number in E.164 format.
    """

    def __init__(
        self,
        account_sid: str,
        auth_token: str,
        from_number: str,
        to_number: str,
    ) -> None:
        self._account_sid: str = account_sid
        self._auth_token: str = auth_token
        self._from_number: str = from_number
        self._to_number: str = to_number

        self._log: structlog.stdlib.BoundLogger = get_logger("notifications.twilio_sms")

        # Twilio REST client (thread-safe, created once).
        self._client: TwilioClient = TwilioClient(account_sid, auth_token)

        # Per-condition rate-limit tracker: condition -> last send epoch.
        self._last_sent: dict[str, float] = {}

        # Dedicated thread pool for blocking Twilio calls.
        self._executor: ThreadPoolExecutor = ThreadPoolExecutor(
            max_workers=_TWILIO_THREAD_POOL_SIZE,
            thread_name_prefix="titan-twilio",
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def send_alert(
        self,
        condition: str,
        message: str,
        severity: str = "CRITICAL",
    ) -> None:
        """Send an SMS alert if the rate limit allows.

        The alert is always logged regardless of whether the SMS is
        actually sent. If the same *condition* was alerted within the
        last hour, the SMS is suppressed.

        Args:
            condition: A short identifier for the alert condition
                (e.g. ``CONNECTIVITY_LOSS``).
            message: The alert body text.
            severity: ``CRITICAL`` or ``WARNING``.
        """
        alert = SMSAlert(
            severity=AlertSeverity(severity.upper()),
            condition=condition,
            message=message,
        )

        # Always log the alert event.
        self._log.info(
            "sms_alert_triggered",
            condition=condition,
            severity=severity,
            message=message,
        )

        if not self._check_rate_limit(condition):
            self._log.info(
                "sms_rate_limited",
                condition=condition,
                next_allowed_in_seconds=self._seconds_until_allowed(condition),
            )
            return

        # Truncate to single SMS segment.
        body = self._truncate_body(f"[TITAN {alert.severity}] {alert.message}")

        await self._send_sms(body, condition)

    async def send_connectivity_alert(self, minutes_down: int) -> None:
        """Send an SMS when IB Gateway connectivity is lost.

        Only triggers if the outage exceeds 5 minutes.

        Args:
            minutes_down: Number of minutes the connection has been
                down.
        """
        if minutes_down < 5:
            return

        await self.send_alert(
            condition=AlertCondition.CONNECTIVITY_LOSS,
            message=(
                f"IB Gateway disconnected for {minutes_down} min. "
                f"Check connection immediately."
            ),
            severity=AlertSeverity.CRITICAL,
        )

    async def send_circuit_breaker_alert(self, level: str, drawdown_pct: float) -> None:
        """Send an SMS when a circuit breaker is triggered.

        Only sends for WARNING, HALT, and EMERGENCY levels.

        Args:
            level: Circuit breaker level (e.g. ``WARNING``, ``HALT``,
                ``EMERGENCY``).
            drawdown_pct: Current drawdown as a decimal fraction
                (e.g. 0.05 for 5%).
        """
        notifiable_levels = {"WARNING", "HALT", "EMERGENCY"}
        if level.upper() not in notifiable_levels:
            return

        await self.send_alert(
            condition=AlertCondition.CIRCUIT_BREAKER,
            message=(
                f"Circuit breaker {level}. "
                f"Drawdown: {drawdown_pct:.1%}. "
                f"Trading restricted."
            ),
            severity=AlertSeverity.CRITICAL,
        )

    async def send_emergency_stop_alert(self) -> None:
        """Send an SMS when the emergency kill switch is activated."""
        await self.send_alert(
            condition=AlertCondition.EMERGENCY_STOP,
            message=(
                "EMERGENCY STOP activated. "
                "All trading halted. "
                "Manual intervention required."
            ),
            severity=AlertSeverity.CRITICAL,
        )

    # ------------------------------------------------------------------
    # Rate limiting
    # ------------------------------------------------------------------

    def _check_rate_limit(self, condition: str) -> bool:
        """Check whether an SMS for the given condition is allowed.

        Returns ``True`` if more than ``_RATE_LIMIT_SECONDS`` (1 hour)
        have elapsed since the last SMS for this condition, or if no
        SMS has ever been sent for it.

        Args:
            condition: The alert condition identifier.

        Returns:
            ``True`` if sending is allowed, ``False`` if rate-limited.
        """
        last = self._last_sent.get(condition)
        if last is None:
            return True
        return (time.monotonic() - last) > _RATE_LIMIT_SECONDS

    def _seconds_until_allowed(self, condition: str) -> int:
        """Return seconds remaining until the next SMS is allowed.

        Args:
            condition: The alert condition identifier.

        Returns:
            Seconds remaining, or 0 if sending is allowed now.
        """
        last = self._last_sent.get(condition)
        if last is None:
            return 0
        elapsed = time.monotonic() - last
        remaining = _RATE_LIMIT_SECONDS - elapsed
        return max(0, int(remaining))

    # ------------------------------------------------------------------
    # SMS delivery
    # ------------------------------------------------------------------

    async def _send_sms(self, body: str, condition: str) -> None:
        """Deliver an SMS via the Twilio REST API.

        The blocking Twilio client call is offloaded to a thread
        executor so the async event loop is never blocked.  Errors are
        caught and logged — a failed SMS never crashes the system.

        Args:
            body: The SMS message body.
            condition: The condition identifier (for rate-limit
                tracking).
        """
        loop = asyncio.get_running_loop()
        try:
            message = await loop.run_in_executor(
                self._executor,
                self._blocking_send,
                body,
            )
            # Record send time for rate limiting.
            self._last_sent[condition] = time.monotonic()
            self._log.info(
                "sms_sent",
                condition=condition,
                sid=message.sid,
                to=self._to_number,
                body_length=len(body),
            )
        except Exception:
            self._log.exception(
                "sms_send_failed",
                condition=condition,
                to=self._to_number,
            )

    def _blocking_send(self, body: str) -> object:
        """Execute the synchronous Twilio API call.

        This method runs inside a thread pool worker and must not
        touch any async constructs.

        Args:
            body: SMS body text.

        Returns:
            The Twilio ``MessageInstance`` object.
        """
        return self._client.messages.create(
            body=body,
            from_=self._from_number,
            to=self._to_number,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _truncate_body(
        body: str,
        max_len: int = _SMS_SINGLE_SEGMENT_LENGTH,
    ) -> str:
        """Truncate SMS body to fit within a single message segment.

        Args:
            body: Raw message text.
            max_len: Maximum character count (default 160).

        Returns:
            The (possibly truncated) body string.
        """
        if len(body) <= max_len:
            return body
        suffix = "..."
        return body[: max_len - len(suffix)] + suffix
