"""Structured logging configuration for Project Titan.

Sets up structlog with JSON output for production and colored console
output for local development.  Call :func:`configure_logging` once at
application startup, then obtain per-component loggers with
:func:`get_logger`.

Usage::

    from src.utils.logging import configure_logging, get_logger

    configure_logging(log_level="INFO", json_output=True)
    logger = get_logger("broker.gateway")
    logger.info("connected to IB Gateway", port=4002)
"""

from __future__ import annotations

import logging
import sys

import structlog


def configure_logging(log_level: str = "INFO", json_output: bool = True) -> None:
    """Configure structlog and stdlib logging for the entire application.

    This must be called exactly once during application startup, before any
    loggers are used.  It wires up structlog processors (ISO timestamp, log
    level, logger name, stack info) and configures the stdlib root logger
    to flow through the same pipeline.

    Parameters
    ----------
    log_level:
        Root log level as a string (e.g. ``"DEBUG"``, ``"INFO"``,
        ``"WARNING"``).
    json_output:
        When ``True`` (the default), log events are rendered as single-line
        JSON objects suitable for log aggregation pipelines.  Set to
        ``False`` for human-readable colored console output during local
        development.
    """
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    # -- Shared processors applied to every log event ---------------------
    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    if json_output:
        renderer: structlog.types.Processor = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    # -- Configure structlog itself ---------------------------------------
    # The wrap_for_formatter processor bridges structlog events into
    # stdlib's ProcessorFormatter so that a single rendering pipeline
    # handles both structlog and stdlib log records.
    structlog.configure(
        processors=[
            *shared_processors,
            structlog.processors.format_exc_info,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # -- Configure stdlib logging to route through structlog ---------------
    formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
        foreign_pre_chain=shared_processors,
    )

    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    # Remove any pre-existing handlers to avoid duplicate output.
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(numeric_level)

    # Quiet down noisy third-party libraries.
    for noisy_logger in (
        "asyncio",
        "urllib3",
        "httpx",
        "httpcore",
        "ib_async",
        "apscheduler",
        "uvicorn.access",
    ):
        logging.getLogger(noisy_logger).setLevel(max(numeric_level, logging.WARNING))


def get_logger(component: str) -> structlog.stdlib.BoundLogger:
    """Return a structlog logger bound with the given component name.

    Every log event emitted through the returned logger will carry a
    ``component`` key automatically, making it easy to filter logs by
    subsystem in production.

    Parameters
    ----------
    component:
        A dot-separated identifier for the subsystem (e.g.
        ``"broker.gateway"``, ``"risk.circuit_breakers"``).

    Returns
    -------
    structlog.stdlib.BoundLogger
        A bound logger instance with ``component`` pre-attached to every
        log event emitted through it.
    """
    return structlog.get_logger(component).bind(component=component)
