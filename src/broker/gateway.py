"""IB Gateway connection manager for Project Titan.

Provides a production-grade, async connection manager around :class:`ib_async.IB`
with automatic reconnection, structured logging, rate limiting, and health
monitoring.

Usage::

    from config.settings import get_settings
    from src.broker.gateway import GatewayManager

    settings = get_settings()
    gw = GatewayManager(
        host="127.0.0.1",
        port=settings.ibkr.gateway_port,
        client_id=settings.ibkr.client_id,
    )
    await gw.connect()
    print(gw.is_connected)
    await gw.disconnect()
"""

from __future__ import annotations

import asyncio
import contextlib
import random
import time

import structlog
from ib_async import IB, Contract, Trade
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.utils.logging import get_logger

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# IB error codes grouped by severity / required action.
_ERR_NOT_CONNECTED: frozenset[int] = frozenset({502, 504})
_ERR_CONNECTIVITY_LOST: frozenset[int] = frozenset({1100})
_ERR_CONNECTIVITY_RESTORED: frozenset[int] = frozenset({1102})
_ERR_DATA_FARM_INFO: frozenset[int] = frozenset({2104, 2106, 2158})
_ERR_COMPETING_SESSION: frozenset[int] = frozenset({10197})

# Rate-limiting: IB allows 50 messages per second.
_IB_RATE_LIMIT_PER_SECOND: int = 50
_RATE_LIMIT_INTERVAL: float = 1.0 / _IB_RATE_LIMIT_PER_SECOND  # 0.02 s

# Reconnection defaults.
_INITIAL_BACKOFF_SECONDS: float = 1.0
_MAX_BACKOFF_SECONDS: float = 60.0
_BACKOFF_FACTOR: float = 2.0
_DEFAULT_MAX_RETRIES: int = 50
_JITTER_RANGE: float = 0.5  # +/- 50 % of calculated delay

# Market data types as defined by IB.
_MARKET_DATA_LIVE: int = 1
_MARKET_DATA_DELAYED: int = 3


class GatewayManager:
    """Manages the lifecycle of a single IB Gateway API connection.

    Responsibilities:
    - Establishing and maintaining the connection to IB Gateway.
    - Automatic reconnection with exponential back-off and jitter.
    - Classifying and routing IB error/status codes.
    - Token-bucket rate limiting to stay under the 50 msg/s IB limit.
    - Exposing health-check and connection metrics.

    Args:
        host: IB Gateway hostname or IP address.
        port: IB Gateway API port (4001 for live, 4002 for paper).
        client_id: Unique client identifier for this API session.
        readonly: If True, the connection will not fetch or place orders.
        max_retries: Maximum number of consecutive reconnection attempts
            before giving up.  Set to 0 for unlimited retries.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 4002,
        client_id: int = 1,
        readonly: bool = False,
        max_retries: int = _DEFAULT_MAX_RETRIES,
    ) -> None:
        self._host: str = host
        self._port: int = port
        self._client_id: int = client_id
        self._readonly: bool = readonly
        self._max_retries: int = max_retries

        # Underlying ib_async connection.
        self._ib: IB = IB()

        # Structured logger bound to this component.
        self._log: structlog.stdlib.BoundLogger = get_logger(
            "broker.gateway",
        ).bind(host=host, port=port, client_id=client_id)

        # Connection bookkeeping.
        self._connected_at: float | None = None
        self._reconnect_count: int = 0
        self._total_disconnections: int = 0
        self._shutting_down: bool = False

        # Background reconnection task handle.
        self._reconnect_task: asyncio.Task[None] | None = None

        # Token-bucket rate limiter: simple semaphore + sleep.
        self._rate_semaphore: asyncio.Semaphore = asyncio.Semaphore(
            _IB_RATE_LIMIT_PER_SECOND
        )

        # Connectivity-lost flag: suppresses reconnect attempts while IB
        # itself reports the link is down (error 1100).
        self._connectivity_lost: bool = False

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def ib(self) -> IB:
        """Return the underlying :class:`ib_async.IB` instance.

        Callers should prefer the higher-level helpers in this class but
        may need direct access for specialised requests (e.g. streaming,
        contract qualification).
        """
        return self._ib

    @property
    def is_connected(self) -> bool:
        """Return True when the API socket is ready for messages."""
        return self._ib.isConnected()

    @property
    def connection_time(self) -> float:
        """Return seconds elapsed since the current connection was established.

        Returns 0.0 if not currently connected.
        """
        if self._connected_at is None or not self.is_connected:
            return 0.0
        return time.monotonic() - self._connected_at

    @property
    def reconnect_count(self) -> int:
        """Return the total number of successful reconnections during this
        manager's lifetime."""
        return self._reconnect_count

    @property
    def total_disconnections(self) -> int:
        """Return the total number of disconnection events observed."""
        return self._total_disconnections

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Connect to IB Gateway with retry logic.

        Uses exponential back-off (initial 1 s, max 60 s, factor 2x) via
        :mod:`tenacity`.  On success the error and disconnect callbacks are
        registered, and the market data type is set according to the
        configured trading mode (live vs. paper / delayed).

        Raises:
            ConnectionError: If all retry attempts are exhausted.
        """
        self._shutting_down = False
        await self._connect_with_retry()

    @retry(
        retry=retry_if_exception_type((ConnectionError, OSError, asyncio.TimeoutError)),
        wait=wait_exponential(
            multiplier=_INITIAL_BACKOFF_SECONDS,
            max=_MAX_BACKOFF_SECONDS,
            exp_base=_BACKOFF_FACTOR,
        ),
        stop=stop_after_attempt(10),
        before_sleep=before_sleep_log(
            structlog.stdlib.get_logger("broker.gateway"),
            "WARNING",  # type: ignore[arg-type]
        ),
        reraise=True,
    )
    async def _connect_with_retry(self) -> None:
        """Internal: perform the actual connection attempt with tenacity retry."""
        self._log.info(
            "connecting",
            host=self._host,
            port=self._port,
            client_id=self._client_id,
            readonly=self._readonly,
        )

        try:
            await self._ib.connectAsync(
                host=self._host,
                port=self._port,
                clientId=self._client_id,
                readonly=self._readonly,
                timeout=30,
            )
        except Exception as exc:
            self._log.warning(
                "connection_attempt_failed",
                error=str(exc),
                error_type=type(exc).__name__,
            )
            raise ConnectionError(
                f"Failed to connect to IB Gateway at {self._host}:{self._port}: {exc}"
            ) from exc

        # Record connection timestamp.
        self._connected_at = time.monotonic()
        self._connectivity_lost = False

        # Register event callbacks.
        self._register_callbacks()

        # Set market data type:
        # Live (1) for production, Delayed (3) for paper trading or when
        # the user does not have a live data subscription.
        if self._port == 4002:
            self._ib.reqMarketDataType(_MARKET_DATA_DELAYED)
            self._log.info("market_data_type_set", type="DELAYED")
        else:
            self._ib.reqMarketDataType(_MARKET_DATA_LIVE)
            self._log.info("market_data_type_set", type="LIVE")

        self._log.info(
            "connected",
            server_version=self._ib.client.serverVersion(),
        )

    async def disconnect(self) -> None:
        """Gracefully disconnect from IB Gateway.

        Cancels all pending (unfilled) orders, cancels the background
        reconnection task if running, and disconnects the socket.
        """
        self._shutting_down = True

        # Cancel background reconnection if in progress.
        if self._reconnect_task is not None and not self._reconnect_task.done():
            self._reconnect_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._reconnect_task
            self._reconnect_task = None

        if not self._ib.isConnected():
            self._log.info("disconnect_skipped", reason="not_connected")
            return

        # Cancel all open (unfilled) orders.
        open_trades: list[Trade] = self._ib.openTrades()
        if open_trades:
            self._log.info("cancelling_pending_orders", count=len(open_trades))
            for trade in open_trades:
                try:
                    self._ib.cancelOrder(trade.order)
                except Exception as exc:
                    self._log.warning(
                        "cancel_order_error",
                        order_id=trade.order.orderId,
                        error=str(exc),
                    )
            # Give IB a moment to process the cancellations.
            await asyncio.sleep(0.5)

        self._log.info("disconnecting")
        self._ib.disconnect()
        self._connected_at = None
        self._log.info("disconnected")

    # ------------------------------------------------------------------
    # Error / event callbacks
    # ------------------------------------------------------------------

    def _register_callbacks(self) -> None:
        """Attach error and disconnected event handlers to the IB instance.

        Clears any previously registered Titan callbacks first to avoid
        duplicate registrations on reconnect.
        """
        # Disconnect old handlers (safe even if not previously registered).
        with contextlib.suppress(ValueError):
            self._ib.errorEvent -= self._on_error
        with contextlib.suppress(ValueError):
            self._ib.disconnectedEvent -= self._on_disconnected

        self._ib.errorEvent += self._on_error
        self._ib.disconnectedEvent += self._on_disconnected

    def _on_error(
        self,
        reqId: int,  # noqa: N803
        errorCode: int,  # noqa: N803
        errorString: str,  # noqa: N803
        contract: Contract | str,
    ) -> None:
        """Handle IB error/status messages.

        Routes each code to the appropriate action:

        - **502, 504** (not connected): schedule a reconnection.
        - **1100** (connectivity lost): log a warning and wait for restore.
        - **1102** (connectivity restored): log info, clear lost flag.
        - **2104, 2106, 2158** (data farm connections): informational only.
        - **10197** (competing session): log a warning.
        - All other codes are logged at warning level.

        Args:
            reqId: The request ID that generated the error, or -1 for
                connection-level events.
            errorCode: Numeric IB error/status code.
            errorString: Human-readable description from IB.
            contract: The contract associated with the error, or an empty
                string when not applicable.
        """
        contract_repr = str(contract) if contract else None

        if errorCode in _ERR_NOT_CONNECTED:
            self._log.error(
                "ib_not_connected",
                error_code=errorCode,
                error_string=errorString,
                req_id=reqId,
            )
            self._schedule_reconnect()

        elif errorCode in _ERR_CONNECTIVITY_LOST:
            self._connectivity_lost = True
            self._log.warning(
                "ib_connectivity_lost",
                error_code=errorCode,
                error_string=errorString,
            )
            # Do NOT trigger reconnect here -- IB will emit 1102 when the
            # link is restored, or the disconnectedEvent will fire if the
            # socket actually drops.

        elif errorCode in _ERR_CONNECTIVITY_RESTORED:
            self._connectivity_lost = False
            self._log.info(
                "ib_connectivity_restored",
                error_code=errorCode,
                error_string=errorString,
            )

        elif errorCode in _ERR_DATA_FARM_INFO:
            self._log.info(
                "ib_data_farm",
                error_code=errorCode,
                error_string=errorString,
            )

        elif errorCode in _ERR_COMPETING_SESSION:
            self._log.warning(
                "ib_competing_session",
                error_code=errorCode,
                error_string=errorString,
                contract=contract_repr,
                req_id=reqId,
            )

        else:
            self._log.warning(
                "ib_error",
                error_code=errorCode,
                error_string=errorString,
                contract=contract_repr,
                req_id=reqId,
            )

    def _on_disconnected(self) -> None:
        """Handle an unexpected disconnection from IB Gateway.

        Increments the disconnection counter and schedules a background
        reconnection loop unless the manager is intentionally shutting
        down.
        """
        self._total_disconnections += 1
        self._connected_at = None
        self._log.warning(
            "ib_disconnected_event",
            total_disconnections=self._total_disconnections,
            shutting_down=self._shutting_down,
        )

        if not self._shutting_down:
            self._schedule_reconnect()

    # ------------------------------------------------------------------
    # Reconnection
    # ------------------------------------------------------------------

    def _schedule_reconnect(self) -> None:
        """Kick off the background reconnection loop if not already running."""
        if self._reconnect_task is not None and not self._reconnect_task.done():
            self._log.debug("reconnect_already_scheduled")
            return
        self._reconnect_task = asyncio.create_task(
            self._reconnect_loop(), name="titan-gateway-reconnect"
        )

    async def _reconnect_loop(self) -> None:
        """Background task that attempts reconnection with exponential back-off.

        Applies jitter (+/- 50 %) to the delay to avoid thundering-herd
        effects if multiple clients reconnect simultaneously.  Gives up
        after ``max_retries`` consecutive failures (0 means unlimited).
        """
        attempt: int = 0
        delay: float = _INITIAL_BACKOFF_SECONDS

        while not self._shutting_down:
            attempt += 1

            if self._max_retries > 0 and attempt > self._max_retries:
                self._log.error(
                    "reconnect_exhausted",
                    max_retries=self._max_retries,
                )
                return

            # Apply jitter: uniform random in [delay * (1 - j), delay * (1 + j)].
            jitter = random.uniform(-_JITTER_RANGE, _JITTER_RANGE)
            jittered_delay = delay * (1.0 + jitter)
            self._log.info(
                "reconnect_attempt",
                attempt=attempt,
                delay_seconds=round(jittered_delay, 2),
                max_retries=self._max_retries,
            )

            await asyncio.sleep(jittered_delay)

            if self._shutting_down:
                return

            try:
                # Disconnect cleanly first if the socket is in a half-open state.
                if self._ib.isConnected():
                    self._ib.disconnect()

                await self._ib.connectAsync(
                    host=self._host,
                    port=self._port,
                    clientId=self._client_id,
                    readonly=self._readonly,
                    timeout=30,
                )

                # Success.
                self._connected_at = time.monotonic()
                self._connectivity_lost = False
                self._reconnect_count += 1
                self._register_callbacks()

                # Restore market data type.
                if self._port == 4002:
                    self._ib.reqMarketDataType(_MARKET_DATA_DELAYED)
                else:
                    self._ib.reqMarketDataType(_MARKET_DATA_LIVE)

                self._log.info(
                    "reconnected",
                    attempt=attempt,
                    total_reconnections=self._reconnect_count,
                    server_version=self._ib.client.serverVersion(),
                )
                return

            except Exception as exc:
                self._log.warning(
                    "reconnect_failed",
                    attempt=attempt,
                    error=str(exc),
                    error_type=type(exc).__name__,
                )
                # Exponential back-off: double the delay, cap at max.
                delay = min(delay * _BACKOFF_FACTOR, _MAX_BACKOFF_SECONDS)

    # ------------------------------------------------------------------
    # Health monitoring
    # ------------------------------------------------------------------

    async def health_check(self) -> bool:
        """Verify the connection is alive by requesting the server time.

        Returns:
            True if the gateway responds within 5 seconds, False otherwise.
        """
        if not self._ib.isConnected():
            return False

        try:
            server_time = await asyncio.wait_for(
                self._ib.reqCurrentTimeAsync(),
                timeout=5.0,
            )
            self._log.debug(
                "health_check_ok",
                server_time=str(server_time),
                uptime_seconds=round(self.connection_time, 1),
            )
            return True
        except (TimeoutError, OSError, ConnectionError) as exc:
            self._log.warning(
                "health_check_failed",
                error=str(exc),
                error_type=type(exc).__name__,
            )
            return False

    # ------------------------------------------------------------------
    # Rate limiting
    # ------------------------------------------------------------------

    async def rate_limit(self) -> None:
        """Enforce the IB 50-messages-per-second rate limit.

        Callers should ``await`` this method before sending any request
        through the IB API.  Internally it uses a semaphore combined with
        an ``asyncio.sleep`` to meter outgoing messages.
        """
        async with self._rate_semaphore:
            await asyncio.sleep(_RATE_LIMIT_INTERVAL)

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------

    async def __aenter__(self) -> GatewayManager:
        await self.connect()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        await self.disconnect()

    # ------------------------------------------------------------------
    # String representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        status = "connected" if self.is_connected else "disconnected"
        return (
            f"<GatewayManager("
            f"host={self._host!r}, "
            f"port={self._port}, "
            f"client_id={self._client_id}, "
            f"status={status}, "
            f"reconnections={self._reconnect_count}"
            f")>"
        )
