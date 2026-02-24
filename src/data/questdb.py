"""QuestDB time-series writer and reader for Project Titan.

Provides async methods to insert and query high-frequency market data,
GEX levels, and signal scores using both the InfluxDB Line Protocol
(ILP) for writes and the PostgreSQL wire protocol for reads.

Usage::

    from src.data.questdb import QuestDBClient

    client = QuestDBClient(host="questdb", ilp_port=9009, http_port=9000)
    await client.connect()

    # Write market ticks
    await client.write_market_tick("AAPL", bid=189.50, ask=189.55, last=189.52, ...)

    # Query recent data
    rows = await client.query(
        "SELECT * FROM market_ticks WHERE ticker = 'AAPL' LIMIT 10"
    )

    await client.close()
"""

from __future__ import annotations

import asyncio
import contextlib
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import httpx
from pydantic import BaseModel, Field

from src.utils.logging import get_logger

if TYPE_CHECKING:
    import structlog

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ILP_DEFAULT_PORT: int = 9009
HTTP_DEFAULT_PORT: int = 9000
PG_DEFAULT_PORT: int = 8812
WRITE_BATCH_SIZE: int = 500
FLUSH_INTERVAL_SECONDS: float = 1.0
HTTP_TIMEOUT_SECONDS: float = 30.0


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class MarketTick(BaseModel):
    """A single market data tick for QuestDB insertion."""

    ticker: str
    bid: float = 0.0
    ask: float = 0.0
    last: float = 0.0
    volume: int = 0
    iv: float = 0.0
    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0
    vega: float = 0.0
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class GEXLevel(BaseModel):
    """A GEX snapshot for QuestDB insertion."""

    ticker: str
    spot_price: float
    net_gex: float
    call_wall: float = 0.0
    put_wall: float = 0.0
    vol_trigger: float = 0.0
    regime: str = "neutral"
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class SignalScore(BaseModel):
    """Ensemble signal score snapshot for QuestDB insertion."""

    ticker: str
    technical_score: float = 0.0
    sentiment_score: float = 0.0
    flow_score: float = 0.0
    regime_score: float = 0.0
    ensemble_score: float = 0.0
    confidence: float = 0.0
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


# ---------------------------------------------------------------------------
# QuestDB Client
# ---------------------------------------------------------------------------


class QuestDBClient:
    """Async QuestDB client using ILP for writes and HTTP for reads.

    Parameters
    ----------
    host:
        QuestDB hostname.
    ilp_port:
        InfluxDB Line Protocol TCP port (default 9009).
    http_port:
        HTTP REST API port for queries (default 9000).
    """

    def __init__(
        self,
        host: str = "questdb",
        ilp_port: int = ILP_DEFAULT_PORT,
        http_port: int = HTTP_DEFAULT_PORT,
    ) -> None:
        self._host = host
        self._ilp_port = ilp_port
        self._http_port = http_port
        self._http_url = f"http://{host}:{http_port}"
        self._writer: asyncio.StreamWriter | None = None
        self._reader: asyncio.StreamReader | None = None
        self._buffer: list[str] = []
        self._http_client: httpx.AsyncClient | None = None
        self._connected = False
        self._log: structlog.BoundLogger = get_logger("data.questdb")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self, max_retries: int = 5, base_delay: float = 2.0) -> None:
        """Open TCP connection for ILP writes and HTTP client for reads.

        Retries with exponential backoff if QuestDB is not yet available.
        """
        last_exc: Exception | None = None
        for attempt in range(1, max_retries + 1):
            try:
                self._reader, self._writer = await asyncio.open_connection(
                    self._host,
                    self._ilp_port,
                )
                self._http_client = httpx.AsyncClient(
                    base_url=self._http_url,
                    timeout=HTTP_TIMEOUT_SECONDS,
                )
                self._connected = True
                self._log.info(
                    "questdb_connected",
                    host=self._host,
                    ilp_port=self._ilp_port,
                    http_port=self._http_port,
                    attempt=attempt,
                )
                return
            except (ConnectionError, OSError) as exc:
                last_exc = exc
                delay = base_delay * (2 ** (attempt - 1))
                self._log.warning(
                    "questdb_connect_retry",
                    attempt=attempt,
                    max_retries=max_retries,
                    delay=delay,
                    error=str(exc),
                )
                if attempt < max_retries:
                    await asyncio.sleep(delay)
        msg = f"QuestDB connection failed after {max_retries} attempts"
        raise ConnectionError(msg) from last_exc

    async def close(self) -> None:
        """Flush pending writes and close all connections."""
        await self.flush()
        if self._writer is not None:
            self._writer.close()
            with contextlib.suppress(ConnectionError, OSError):
                await self._writer.wait_closed()
            self._writer = None
            self._reader = None
        if self._http_client is not None:
            await self._http_client.aclose()
            self._http_client = None
        self._connected = False
        self._log.info("questdb_closed")

    @property
    def connected(self) -> bool:
        """Return True if the client is connected."""
        return self._connected

    # ------------------------------------------------------------------
    # ILP writing
    # ------------------------------------------------------------------

    def _escape_tag(self, value: str) -> str:
        """Escape special characters in ILP tag values."""
        return value.replace(" ", r"\ ").replace(",", r"\,").replace("=", r"\=")

    def _ts_nanos(self, dt: datetime) -> int:
        """Convert a datetime to nanoseconds since epoch."""
        return int(dt.timestamp() * 1_000_000_000)

    def _enqueue(self, line: str) -> None:
        """Add an ILP line to the write buffer."""
        self._buffer.append(line)

    async def flush(self) -> None:
        """Send all buffered ILP lines to QuestDB."""
        if not self._buffer or self._writer is None:
            return

        payload = "\n".join(self._buffer) + "\n"
        try:
            self._writer.write(payload.encode("utf-8"))
            await self._writer.drain()
            count = len(self._buffer)
            self._buffer.clear()
            self._log.debug("questdb_flushed", lines=count)
        except (ConnectionError, BrokenPipeError, OSError) as exc:
            self._log.warning(
                "questdb_flush_failed",
                error=str(exc),
                lines_lost=len(self._buffer),
            )
            self._buffer.clear()
            self._connected = False

    async def _maybe_flush(self) -> None:
        """Flush if the buffer has reached the batch threshold."""
        if len(self._buffer) >= WRITE_BATCH_SIZE:
            await self.flush()

    # ------------------------------------------------------------------
    # Typed write methods
    # ------------------------------------------------------------------

    async def write_market_tick(self, tick: MarketTick) -> None:
        """Buffer a market data tick for insertion.

        Parameters
        ----------
        tick:
            Pydantic model with all tick fields.
        """
        tag = self._escape_tag(tick.ticker)
        ts = self._ts_nanos(tick.timestamp)
        line = (
            f"market_ticks,ticker={tag} "
            f"bid={tick.bid},ask={tick.ask},last={tick.last},"
            f"volume={tick.volume}i,"
            f"iv={tick.iv},delta={tick.delta},gamma={tick.gamma},"
            f"theta={tick.theta},vega={tick.vega} "
            f"{ts}"
        )
        self._enqueue(line)
        await self._maybe_flush()

    async def write_market_ticks(self, ticks: list[MarketTick]) -> None:
        """Buffer a batch of market ticks."""
        for tick in ticks:
            await self.write_market_tick(tick)

    async def write_gex_level(self, gex: GEXLevel) -> None:
        """Buffer a GEX level snapshot for insertion.

        Parameters
        ----------
        gex:
            Pydantic model with GEX data.
        """
        tag = self._escape_tag(gex.ticker)
        regime_tag = self._escape_tag(gex.regime)
        ts = self._ts_nanos(gex.timestamp)
        line = (
            f"gex_levels,ticker={tag},regime={regime_tag} "
            f"spot_price={gex.spot_price},net_gex={gex.net_gex},"
            f"call_wall={gex.call_wall},put_wall={gex.put_wall},"
            f"vol_trigger={gex.vol_trigger} "
            f"{ts}"
        )
        self._enqueue(line)
        await self._maybe_flush()

    async def write_signal_score(self, score: SignalScore) -> None:
        """Buffer a signal score snapshot for insertion.

        Parameters
        ----------
        score:
            Pydantic model with all signal scores.
        """
        tag = self._escape_tag(score.ticker)
        ts = self._ts_nanos(score.timestamp)
        line = (
            f"signal_scores,ticker={tag} "
            f"technical_score={score.technical_score},"
            f"sentiment_score={score.sentiment_score},"
            f"flow_score={score.flow_score},"
            f"regime_score={score.regime_score},"
            f"ensemble_score={score.ensemble_score},"
            f"confidence={score.confidence} "
            f"{ts}"
        )
        self._enqueue(line)
        await self._maybe_flush()

    async def write_ilp_raw(
        self,
        table: str,
        tags: dict[str, str],
        fields: dict[str, float | int],
        ts: datetime | None = None,
    ) -> None:
        """Write an arbitrary ILP line.

        Parameters
        ----------
        table:
            QuestDB table name.
        tags:
            Tag key-value pairs (indexed columns).
        fields:
            Field key-value pairs (data columns).
        ts:
            Timestamp.  Defaults to now.
        """
        tag_str = ",".join(f"{k}={self._escape_tag(str(v))}" for k, v in tags.items())
        field_parts: list[str] = []
        for k, v in fields.items():
            if isinstance(v, int):
                field_parts.append(f"{k}={v}i")
            else:
                field_parts.append(f"{k}={v}")
        field_str = ",".join(field_parts)
        ts_ns = self._ts_nanos(ts or datetime.now(UTC))
        line = (
            f"{table},{tag_str} {field_str} {ts_ns}"
            if tag_str
            else f"{table} {field_str} {ts_ns}"
        )
        self._enqueue(line)
        await self._maybe_flush()

    # ------------------------------------------------------------------
    # HTTP query methods
    # ------------------------------------------------------------------

    async def query(self, sql: str) -> list[dict[str, Any]]:
        """Execute a SQL query and return rows as list of dicts.

        Parameters
        ----------
        sql:
            SQL query string (QuestDB SQL dialect).

        Returns
        -------
        list[dict[str, Any]]
            Each row as a dictionary keyed by column name.
        """
        if self._http_client is None:
            raise RuntimeError("QuestDBClient not connected. Call connect() first.")

        resp = await self._http_client.get(
            "/exec",
            params={"query": sql, "fmt": "json"},
        )
        resp.raise_for_status()
        data = resp.json()

        columns: list[dict[str, str]] = data.get("columns", [])
        dataset: list[list[Any]] = data.get("dataset", [])
        col_names = [c["name"] for c in columns]

        return [dict(zip(col_names, row, strict=False)) for row in dataset]

    async def query_df(self, sql: str) -> Any:
        """Execute a SQL query and return a pandas DataFrame.

        Requires pandas to be installed.

        Parameters
        ----------
        sql:
            SQL query string.

        Returns
        -------
        pandas.DataFrame
            Query results as a DataFrame.
        """
        import pandas as pd

        rows = await self.query(sql)
        return pd.DataFrame(rows)

    async def count(self, table: str) -> int:
        """Return the row count for a table."""
        rows = await self.query(f"SELECT count() AS cnt FROM {table}")
        if rows:
            return int(rows[0]["cnt"])
        return 0

    # ------------------------------------------------------------------
    # Table management
    # ------------------------------------------------------------------

    async def ensure_tables(self) -> None:
        """Create the core time-series tables if they don't exist.

        This issues the DDL via the HTTP /exec endpoint.
        """
        if self._http_client is None:
            raise RuntimeError("QuestDBClient not connected.")

        ddl_statements = [
            """
            CREATE TABLE IF NOT EXISTS market_ticks (
                timestamp TIMESTAMP,
                ticker SYMBOL,
                bid DOUBLE,
                ask DOUBLE,
                last DOUBLE,
                volume LONG,
                iv DOUBLE,
                delta DOUBLE,
                gamma DOUBLE,
                theta DOUBLE,
                vega DOUBLE
            ) TIMESTAMP(timestamp) PARTITION BY DAY WAL
            """,
            """
            CREATE TABLE IF NOT EXISTS gex_levels (
                timestamp TIMESTAMP,
                ticker SYMBOL,
                spot_price DOUBLE,
                net_gex DOUBLE,
                call_wall DOUBLE,
                put_wall DOUBLE,
                vol_trigger DOUBLE,
                regime SYMBOL
            ) TIMESTAMP(timestamp) PARTITION BY DAY WAL
            """,
            """
            CREATE TABLE IF NOT EXISTS signal_scores (
                timestamp TIMESTAMP,
                ticker SYMBOL,
                technical_score DOUBLE,
                sentiment_score DOUBLE,
                flow_score DOUBLE,
                regime_score DOUBLE,
                ensemble_score DOUBLE,
                confidence DOUBLE
            ) TIMESTAMP(timestamp) PARTITION BY DAY WAL
            """,
        ]

        for ddl in ddl_statements:
            resp = await self._http_client.get(
                "/exec",
                params={"query": ddl.strip()},
            )
            if resp.status_code != 200:
                self._log.warning(
                    "questdb_ddl_failed",
                    status=resp.status_code,
                    body=resp.text[:200],
                )
        self._log.info("questdb_tables_ensured")
