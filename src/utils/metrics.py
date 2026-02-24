"""Prometheus metrics definitions and FastAPI metrics server for Project Titan.

All metrics are defined as module-level constants so any subsystem can
import and update them directly::

    from src.utils.metrics import TRADE_COUNT, API_LATENCY

    TRADE_COUNT.labels(strategy="iron_condor", status="win").inc()
    with API_LATENCY.labels(api="ibkr").time():
        await place_order(...)

The module also provides a FastAPI application with ``/health``,
``/metrics``, ``/status``, and ``/positions`` endpoints.  Call
:func:`start_metrics_server` at startup to run the server in the
background on a configurable port (default 8080).
"""

from __future__ import annotations

import asyncio
import threading
import time as _time
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from prometheus_client import (
    REGISTRY,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

from src.utils.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Callable

# -----------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------

DEFAULT_METRICS_PORT: int = 8080
PROMETHEUS_CONTENT_TYPE: str = "text/plain; version=0.0.4; charset=utf-8"

logger = get_logger("utils.metrics")

# -----------------------------------------------------------------------
# Trade metrics
# -----------------------------------------------------------------------

TRADE_COUNT: Counter = Counter(
    "titan_trade_count_total",
    "Total number of trades executed",
    labelnames=["strategy", "status"],
)

TRADE_PNL: Histogram = Histogram(
    "titan_trade_pnl_dollars",
    "Realized P&L distribution per trade in USD",
    labelnames=["strategy"],
    buckets=(
        -5000,
        -2000,
        -1000,
        -500,
        -200,
        -100,
        0,
        100,
        200,
        500,
        1000,
        2000,
        5000,
    ),
)

WIN_RATE: Gauge = Gauge(
    "titan_win_rate",
    "Current win rate for each strategy (0.0 - 1.0)",
    labelnames=["strategy"],
)

# -----------------------------------------------------------------------
# Portfolio / risk metrics
# -----------------------------------------------------------------------

DRAWDOWN_PCT: Gauge = Gauge(
    "titan_drawdown_pct",
    "Current drawdown as a fraction of high-water mark",
)

POSITIONS_OPEN: Gauge = Gauge(
    "titan_positions_open",
    "Number of currently open positions",
)

CIRCUIT_BREAKER_LEVEL: Gauge = Gauge(
    "titan_circuit_breaker_level",
    "Current circuit-breaker level"
    " (0=normal, 1=caution, 2=warning, 3=halt, 4=emergency)",
)

PORTFOLIO_GREEKS: Gauge = Gauge(
    "titan_portfolio_greeks",
    "Aggregate portfolio-level Greeks",
    labelnames=["greek"],
)

# -----------------------------------------------------------------------
# Signal metrics
# -----------------------------------------------------------------------

REGIME_STATE: Gauge = Gauge(
    "titan_regime_state",
    "Active market regime indicator (1=active, 0=inactive)",
    labelnames=["regime"],
)

CONFIDENCE_SCORE: Histogram = Histogram(
    "titan_confidence_score",
    "Distribution of ML ensemble confidence scores",
    buckets=(
        0.0,
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.78,
        0.85,
        0.9,
        0.95,
        1.0,
    ),
)

SIGNAL_SCORE: Gauge = Gauge(
    "titan_signal_score",
    "Latest signal score by type and ticker",
    labelnames=["signal_type", "ticker"],
)

# -----------------------------------------------------------------------
# Infrastructure / latency metrics
# -----------------------------------------------------------------------

API_LATENCY: Histogram = Histogram(
    "titan_api_latency_seconds",
    "External API call latency in seconds",
    labelnames=["api"],
    buckets=(
        0.01,
        0.025,
        0.05,
        0.1,
        0.25,
        0.5,
        1.0,
        2.5,
        5.0,
        10.0,
    ),
)

CONNECTION_STATUS: Gauge = Gauge(
    "titan_connection_status",
    "Service connectivity (1=connected, 0=disconnected)",
    labelnames=["service"],
)

ORDER_FILL_TIME: Histogram = Histogram(
    "titan_order_fill_time_seconds",
    "Time from order submission to complete fill in seconds",
    buckets=(
        0.1,
        0.5,
        1.0,
        2.0,
        5.0,
        10.0,
        30.0,
        60.0,
        120.0,
        300.0,
    ),
)

# -----------------------------------------------------------------------
# Additional metrics (Phase 5 enhancements)
# -----------------------------------------------------------------------

ML_CONFIDENCE: Histogram = Histogram(
    "titan_ml_confidence",
    "Histogram of ML confidence scores across all models",
    buckets=(
        0.0,
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.78,
        0.85,
        0.9,
        0.95,
        1.0,
    ),
)

UPTIME_SECONDS: Gauge = Gauge(
    "titan_uptime_seconds",
    "System uptime in seconds since application start",
)

DAILY_PNL: Gauge = Gauge(
    "titan_daily_pnl_dollars",
    "Realized daily P&L in USD",
)

SCHEDULER_JOB_DURATION: Histogram = Histogram(
    "titan_scheduler_job_duration_seconds",
    "Duration of scheduled job executions",
    labelnames=["job_name"],
    buckets=(
        0.1,
        0.5,
        1.0,
        5.0,
        10.0,
        30.0,
        60.0,
        300.0,
        600.0,
    ),
)

SCHEDULER_JOB_ERRORS: Counter = Counter(
    "titan_scheduler_job_errors_total",
    "Total number of scheduler job execution errors",
    labelnames=["job_name"],
)

# -----------------------------------------------------------------------
# Application start time (set once at module import)
# -----------------------------------------------------------------------

_APP_START_TIME: float = _time.monotonic()
_APP_START_DATETIME: datetime = datetime.now(tz=UTC)

# -----------------------------------------------------------------------
# Health provider registry
# -----------------------------------------------------------------------

_health_providers: dict[str, Callable[..., Any]] = {}
_health_lock: threading.Lock = threading.Lock()


def register_health_providers(
    providers: dict[str, Callable[..., Any]],
) -> None:
    """Register callables that provide health check data.

    Each provider is a callable (sync or async) that returns the
    current value for its named health metric.  Providers are invoked
    when the ``/health`` or ``/status`` endpoint is requested.

    Parameters
    ----------
    providers:
        Mapping of health metric name to a callable returning the
        current value.  Common names: ``"ib_connected"``,
        ``"redis_connected"``, ``"postgres_connected"``,
        ``"regime"``, ``"circuit_breaker"``, ``"positions_open"``,
        ``"daily_pnl"``.
    """
    with _health_lock:
        _health_providers.update(providers)
    logger.debug(
        "health providers registered",
        provider_names=list(providers.keys()),
    )


async def _collect_health_data() -> dict[str, Any]:
    """Invoke all registered health providers and collect results.

    Returns
    -------
    dict[str, Any]
        Aggregated health data from all providers.
    """
    data: dict[str, Any] = {}
    with _health_lock:
        providers = dict(_health_providers)

    for name, provider in providers.items():
        try:
            result = provider()
            if asyncio.iscoroutine(result):
                result = await result
            data[name] = result
        except Exception:
            logger.warning(
                "health provider failed",
                provider_name=name,
            )
            data[name] = None
    return data


def _get_uptime_seconds() -> float:
    """Return elapsed seconds since the application started."""
    return round(_time.monotonic() - _APP_START_TIME, 1)


# -----------------------------------------------------------------------
# FastAPI application factory
# -----------------------------------------------------------------------


def create_app() -> Any:
    """Create and configure the FastAPI application with all routes.

    Returns
    -------
    FastAPI
        A configured FastAPI application with ``/health``,
        ``/metrics``, ``/status``, and ``/positions`` endpoints.
    """
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse, PlainTextResponse

    app = FastAPI(
        title="Titan Metrics",
        description="Project Titan monitoring and health endpoints",
        docs_url=None,
        redoc_url=None,
    )

    @app.get("/health")
    async def health() -> JSONResponse:
        """Liveness and readiness probe.

        Returns a JSON object with system health indicators including
        connection statuses, regime state, circuit breaker level,
        open positions count, and daily P&L.
        """
        uptime = _get_uptime_seconds()
        UPTIME_SECONDS.set(uptime)

        health_data = await _collect_health_data()

        payload: dict[str, Any] = {
            "status": "healthy",
            "uptime_seconds": uptime,
            "ib_connected": health_data.get("ib_connected", False),
            "redis_connected": health_data.get("redis_connected", False),
            "postgres_connected": health_data.get("postgres_connected", False),
            "regime": health_data.get("regime", "unknown"),
            "circuit_breaker": health_data.get("circuit_breaker", "NORMAL"),
            "positions_open": health_data.get("positions_open", 0),
            "daily_pnl": health_data.get("daily_pnl", 0.0),
        }
        return JSONResponse(content=payload)

    @app.get("/metrics")
    async def metrics() -> PlainTextResponse:
        """Prometheus-compatible metrics endpoint."""
        UPTIME_SECONDS.set(_get_uptime_seconds())
        return PlainTextResponse(
            content=generate_latest(REGISTRY).decode("utf-8"),
            media_type=PROMETHEUS_CONTENT_TYPE,
        )

    @app.get("/status")
    async def status() -> JSONResponse:
        """Detailed system status with all component statuses.

        Provides a comprehensive view of every health provider,
        scheduler state, and infrastructure connection.
        """
        uptime = _get_uptime_seconds()
        UPTIME_SECONDS.set(uptime)

        health_data = await _collect_health_data()

        payload: dict[str, Any] = {
            "application": "Project Titan",
            "status": "running",
            "uptime_seconds": uptime,
            "started_at": _APP_START_DATETIME.isoformat(),
            "connections": {
                "ib_gateway": health_data.get("ib_connected", False),
                "redis": health_data.get("redis_connected", False),
                "postgres": health_data.get("postgres_connected", False),
                "questdb": health_data.get("questdb_connected", False),
            },
            "trading": {
                "regime": health_data.get("regime", "unknown"),
                "circuit_breaker": health_data.get("circuit_breaker", "NORMAL"),
                "positions_open": health_data.get("positions_open", 0),
                "daily_pnl": health_data.get("daily_pnl", 0.0),
                "weekly_pnl": health_data.get("weekly_pnl", 0.0),
                "drawdown_pct": health_data.get("drawdown_pct", 0.0),
            },
            "scheduler": {
                "running": health_data.get("scheduler_running", False),
                "next_jobs": health_data.get("scheduler_next_jobs", {}),
            },
            "models": {
                "active_version": health_data.get("model_version", None),
                "last_trained": health_data.get("model_last_trained", None),
            },
        }
        return JSONResponse(content=payload)

    @app.get("/positions")
    async def positions() -> JSONResponse:
        """Current open positions list.

        Returns a JSON array of open positions with their details
        including ticker, strategy, Greeks, and P&L.
        """
        health_data = await _collect_health_data()
        position_list = health_data.get("positions_list", [])
        return JSONResponse(
            content={
                "count": len(position_list),
                "positions": position_list,
            }
        )

    return app


# -----------------------------------------------------------------------
# Metrics server lifecycle
# -----------------------------------------------------------------------

_server_thread: threading.Thread | None = None
_server_should_stop: threading.Event = threading.Event()


def setup_metrics_server(port: int = DEFAULT_METRICS_PORT) -> threading.Thread:
    """Start a FastAPI server exposing health and metrics endpoints.

    The server runs in a daemon thread so it does not block the main
    asyncio event loop.  Prometheus should be configured to scrape
    ``http://<host>:<port>/metrics``.

    Parameters
    ----------
    port:
        TCP port to bind the HTTP server on.  Defaults to ``8080``.

    Returns
    -------
    threading.Thread
        The daemon thread running the uvicorn server.  Callers rarely
        need to interact with it directly.
    """
    app = create_app()

    def _run_server() -> None:
        import uvicorn

        uvicorn.run(
            app,
            host="0.0.0.0",  # noqa: S104 — bound to 127.0.0.1 by Docker port mapping
            port=port,
            log_level="warning",
            access_log=False,
        )

    thread = threading.Thread(target=_run_server, daemon=True, name="metrics-server")
    thread.start()
    logger.info("metrics server started", port=port)
    return thread


async def start_metrics_server(
    port: int = DEFAULT_METRICS_PORT,
) -> None:
    """Start the uvicorn metrics server in a background thread.

    This is the preferred async entry point for starting the metrics
    server during application startup.  It delegates to a daemon
    thread so the asyncio event loop is not blocked.

    Parameters
    ----------
    port:
        TCP port to bind.  Defaults to ``8080``.
    """
    global _server_thread  # noqa: PLW0603
    if _server_thread is not None and _server_thread.is_alive():
        logger.warning("metrics server already running — skipping")
        return

    _server_should_stop.clear()
    app = create_app()

    def _run_server() -> None:
        import uvicorn

        config = uvicorn.Config(
            app=app,
            host="0.0.0.0",  # noqa: S104 — bound to 127.0.0.1 by Docker port mapping
            port=port,
            log_level="warning",
            access_log=False,
        )
        server = uvicorn.Server(config)
        server.run()

    _server_thread = threading.Thread(
        target=_run_server,
        daemon=True,
        name="metrics-server",
    )
    _server_thread.start()
    logger.info(
        "metrics server started (async)",
        port=port,
    )


async def stop_metrics_server() -> None:
    """Stop the background metrics server gracefully.

    Signals the server thread to terminate.  Since the thread is a
    daemon, it will be cleaned up automatically when the process
    exits even if this method is not called.
    """
    global _server_thread  # noqa: PLW0603
    if _server_thread is None or not _server_thread.is_alive():
        logger.debug("metrics server not running — nothing to stop")
        return

    _server_should_stop.set()
    _server_thread = None
    logger.info("metrics server stop requested")
