#!/usr/bin/env python3
"""Service health check script for Project Titan's Docker Compose stack.

Checks connectivity and basic health of all seven services in the stack:
PostgreSQL, Redis, QuestDB, IB Gateway, Titan app, Prometheus, and Grafana.

Runnable standalone::

    python scripts/health_check.py

Exit codes:
    0 — All services healthy
    1 — One or more services unreachable or unhealthy
"""

from __future__ import annotations

import os
import socket
import sys
import time
from typing import NamedTuple

# ---------------------------------------------------------------------------
# ANSI colour helpers (gracefully degrade when piped to a file)
# ---------------------------------------------------------------------------
_USE_COLOR = sys.stdout.isatty()

GREEN = "\033[92m" if _USE_COLOR else ""
RED = "\033[91m" if _USE_COLOR else ""
BOLD = "\033[1m" if _USE_COLOR else ""
RESET = "\033[0m" if _USE_COLOR else ""

CHECK_MARK = "\u2713"  # Unicode check mark
CROSS_MARK = "\u2717"  # Unicode ballot X

TIMEOUT_SECONDS = 2


class CheckResult(NamedTuple):
    """Outcome of a single service health check."""

    service: str
    healthy: bool
    message: str
    latency_ms: float


# ---------------------------------------------------------------------------
# Individual service checks
# ---------------------------------------------------------------------------


def check_postgresql() -> CheckResult:
    """Connect to PostgreSQL and execute a trivial query."""
    host = os.environ.get("POSTGRES_HOST", "localhost")
    port = int(os.environ.get("POSTGRES_PORT", "5432"))
    db = os.environ.get("POSTGRES_DB", "titan")
    user = os.environ.get("POSTGRES_USER", "titan")
    password = os.environ.get("POSTGRES_PASSWORD", "")

    start = time.monotonic()
    try:
        import psycopg2  # noqa: PLC0415

        conn = psycopg2.connect(
            host=host,
            port=port,
            dbname=db,
            user=user,
            password=password,
            connect_timeout=TIMEOUT_SECONDS,
        )
        cur = conn.cursor()
        cur.execute("SELECT 1;")
        cur.fetchone()
        cur.close()
        conn.close()
        latency = (time.monotonic() - start) * 1000
        return CheckResult("PostgreSQL", True, f"{host}:{port}/{db}", latency)
    except Exception as exc:
        latency = (time.monotonic() - start) * 1000
        return CheckResult(
            "PostgreSQL", False, str(exc).strip().split("\n")[0], latency
        )


def check_redis() -> CheckResult:
    """Connect to Redis and send a PING command."""
    host = os.environ.get("REDIS_HOST", "localhost")
    port = int(os.environ.get("REDIS_PORT", "6379"))

    start = time.monotonic()
    try:
        import redis as redis_lib  # noqa: PLC0415

        client = redis_lib.Redis(
            host=host,
            port=port,
            socket_connect_timeout=TIMEOUT_SECONDS,
            socket_timeout=TIMEOUT_SECONDS,
        )
        response = client.ping()
        client.close()
        latency = (time.monotonic() - start) * 1000
        if response:
            return CheckResult("Redis", True, f"{host}:{port} PONG", latency)
        return CheckResult("Redis", False, "PING returned False", latency)
    except Exception as exc:
        latency = (time.monotonic() - start) * 1000
        return CheckResult("Redis", False, str(exc).strip().split("\n")[0], latency)


def check_questdb() -> CheckResult:
    """HTTP health check against QuestDB's REST API on port 9000."""
    host = os.environ.get("QUESTDB_HOST", "localhost")
    port = int(os.environ.get("QUESTDB_HTTP_PORT", "9000"))
    url = f"http://{host}:{port}/"

    start = time.monotonic()
    try:
        import httpx  # noqa: PLC0415

        resp = httpx.get(url, timeout=TIMEOUT_SECONDS)
        latency = (time.monotonic() - start) * 1000
        if resp.status_code == 200:
            return CheckResult("QuestDB", True, f"{host}:{port} HTTP 200", latency)
        return CheckResult("QuestDB", False, f"HTTP {resp.status_code}", latency)
    except Exception as exc:
        latency = (time.monotonic() - start) * 1000
        return CheckResult("QuestDB", False, str(exc).strip().split("\n")[0], latency)


def check_ib_gateway() -> CheckResult:
    """TCP socket test against IB Gateway API ports.

    Checks the live port (4001) first, then the paper port (4002).
    Either being reachable counts as healthy.
    """
    host = os.environ.get("IBKR_GATEWAY_HOST", "localhost")
    live_port = int(os.environ.get("IBKR_LIVE_PORT", "4001"))
    paper_port = int(os.environ.get("IBKR_PAPER_PORT", "4002"))

    for label, port in [("live", live_port), ("paper", paper_port)]:
        start = time.monotonic()
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(TIMEOUT_SECONDS)
            sock.connect((host, port))
            sock.close()
            latency = (time.monotonic() - start) * 1000
            return CheckResult(
                "IB Gateway", True, f"{host}:{port} ({label}) reachable", latency
            )
        except Exception:
            pass  # Try the next port

    return CheckResult(
        "IB Gateway",
        False,
        f"Neither {host}:{live_port} (live) nor {host}:{paper_port} (paper) reachable",
        0.0,
    )


def check_titan_app() -> CheckResult:
    """HTTP health check against the Titan FastAPI endpoint."""
    host = os.environ.get("TITAN_HOST", "localhost")
    port = int(os.environ.get("TITAN_PORT", "8080"))
    url = f"http://{host}:{port}/health"

    start = time.monotonic()
    try:
        import httpx  # noqa: PLC0415

        resp = httpx.get(url, timeout=TIMEOUT_SECONDS)
        latency = (time.monotonic() - start) * 1000
        if resp.status_code == 200:
            return CheckResult(
                "Titan App", True, f"{host}:{port}/health HTTP 200", latency
            )
        return CheckResult("Titan App", False, f"HTTP {resp.status_code}", latency)
    except Exception as exc:
        latency = (time.monotonic() - start) * 1000
        return CheckResult("Titan App", False, str(exc).strip().split("\n")[0], latency)


def check_prometheus() -> CheckResult:
    """HTTP health check against Prometheus /-/healthy endpoint."""
    host = os.environ.get("PROMETHEUS_HOST", "localhost")
    port = int(os.environ.get("PROMETHEUS_PORT", "9090"))
    url = f"http://{host}:{port}/-/healthy"

    start = time.monotonic()
    try:
        import httpx  # noqa: PLC0415

        resp = httpx.get(url, timeout=TIMEOUT_SECONDS)
        latency = (time.monotonic() - start) * 1000
        if resp.status_code == 200:
            return CheckResult(
                "Prometheus", True, f"{host}:{port}/-/healthy HTTP 200", latency
            )
        return CheckResult("Prometheus", False, f"HTTP {resp.status_code}", latency)
    except Exception as exc:
        latency = (time.monotonic() - start) * 1000
        return CheckResult(
            "Prometheus", False, str(exc).strip().split("\n")[0], latency
        )


def check_grafana() -> CheckResult:
    """HTTP health check against Grafana /api/health endpoint."""
    host = os.environ.get("GRAFANA_HOST", "localhost")
    port = int(os.environ.get("GRAFANA_PORT", "3000"))
    url = f"http://{host}:{port}/api/health"

    start = time.monotonic()
    try:
        import httpx  # noqa: PLC0415

        resp = httpx.get(url, timeout=TIMEOUT_SECONDS)
        latency = (time.monotonic() - start) * 1000
        if resp.status_code == 200:
            return CheckResult(
                "Grafana", True, f"{host}:{port}/api/health HTTP 200", latency
            )
        return CheckResult("Grafana", False, f"HTTP {resp.status_code}", latency)
    except Exception as exc:
        latency = (time.monotonic() - start) * 1000
        return CheckResult("Grafana", False, str(exc).strip().split("\n")[0], latency)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

ALL_CHECKS = [
    ("PostgreSQL", check_postgresql),
    ("Redis", check_redis),
    ("QuestDB", check_questdb),
    ("IB Gateway", check_ib_gateway),
    ("Titan App", check_titan_app),
    ("Prometheus", check_prometheus),
    ("Grafana", check_grafana),
]


def main() -> int:
    """Run all health checks and print a status table.

    Returns:
        0 if every service is healthy, 1 otherwise.
    """
    print(f"\n{BOLD}Project Titan — Service Health Check{RESET}")
    print("=" * 60)

    results: list[CheckResult] = []
    for _name, check_fn in ALL_CHECKS:
        result = check_fn()
        results.append(result)

        if result.healthy:
            icon = f"{GREEN}{CHECK_MARK}{RESET}"
            status = f"{GREEN}HEALTHY{RESET}"
        else:
            icon = f"{RED}{CROSS_MARK}{RESET}"
            status = f"{RED}FAILED{RESET}"

        latency_str = (
            f"{result.latency_ms:6.1f}ms" if result.latency_ms > 0 else "   N/A"
        )
        print(
            f"  {icon} {result.service:<14s}  {status:<20s}"
            f"  {latency_str}  {result.message}"
        )

    print("=" * 60)

    healthy_count = sum(1 for r in results if r.healthy)
    total_count = len(results)
    all_healthy = healthy_count == total_count

    if all_healthy:
        print(f"  {GREEN}{BOLD}All {total_count} services healthy.{RESET}\n")
    else:
        failed_count = total_count - healthy_count
        print(
            f"  {RED}{BOLD}{failed_count}/{total_count}"
            f" service(s) unreachable or unhealthy."
            f"{RESET}\n"
        )

    return 0 if all_healthy else 1


if __name__ == "__main__":
    sys.exit(main())
