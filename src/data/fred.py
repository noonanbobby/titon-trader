"""FRED (Federal Reserve Economic Data) API client for Project Titan.

Fetches macroeconomic time-series used by the cross-asset signal module:

- **DGS2** / **DGS10**: 2-Year and 10-Year Treasury yields (yield curve spread)
- **DFF**: Federal Funds Effective Rate
- **BAMLH0A0HYM2**: ICE BofA US High Yield OAS (credit stress indicator)
- **T10YIE**: 10-Year Breakeven Inflation Rate
- **VIXCLS**: CBOE VIX (daily close, for backfill / verification)

Usage::

    from src.data.fred import FREDClient

    client = FREDClient(api_key="your_fred_key")
    spread = await client.get_yield_curve_spread()
    fed_rate = await client.get_fed_funds_rate()
    hy_oas = await client.get_hy_oas()
    await client.close()
"""

from __future__ import annotations

import asyncio
from datetime import date, timedelta
from typing import TYPE_CHECKING, Any

import httpx
from pydantic import BaseModel, Field
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.utils.logging import get_logger

if TYPE_CHECKING:
    import pandas as pd
    import structlog

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE_URL: str = "https://api.stlouisfed.org/fred"
HTTP_TIMEOUT_SECONDS: float = 30.0
RATE_LIMIT_DELAY: float = 0.5  # FRED is generous but let's be respectful

# Standard FRED series IDs
SERIES_DGS2: str = "DGS2"
SERIES_DGS10: str = "DGS10"
SERIES_FED_FUNDS: str = "DFF"
SERIES_HY_OAS: str = "BAMLH0A0HYM2"
SERIES_BREAKEVEN_10Y: str = "T10YIE"
SERIES_VIX: str = "VIXCLS"


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class FREDObservation(BaseModel):
    """A single FRED data point."""

    date: date
    value: float | None = None


class MacroSnapshot(BaseModel):
    """A point-in-time snapshot of key macro indicators."""

    yield_2y: float | None = None
    yield_10y: float | None = None
    yield_spread_2s10s: float | None = None
    fed_funds_rate: float | None = None
    hy_oas: float | None = None
    breakeven_10y: float | None = None
    vix: float | None = None
    timestamp: date = Field(default_factory=date.today)


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class FREDClient:
    """Async FRED API client.

    Parameters
    ----------
    api_key:
        FRED API key.
    cache:
        Optional RedisCache for response caching.
    """

    def __init__(
        self,
        api_key: str,
        cache: Any | None = None,
    ) -> None:
        self._api_key = api_key
        self._cache = cache
        self._client: httpx.AsyncClient | None = None
        self._log: structlog.BoundLogger = get_logger("data.fred")

    async def _ensure_client(self) -> httpx.AsyncClient:
        """Lazily create the HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=BASE_URL,
                timeout=HTTP_TIMEOUT_SECONDS,
            )
        return self._client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    @retry(
        retry=retry_if_exception_type(
            (httpx.HTTPStatusError, httpx.ConnectError, httpx.TimeoutException)
        ),
        wait=wait_exponential(multiplier=1, min=1, max=30),
        stop=stop_after_attempt(4),
    )
    async def _get(
        self, path: str, params: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Execute a rate-limited GET request with retry logic."""
        client = await self._ensure_client()
        await asyncio.sleep(RATE_LIMIT_DELAY)
        full_params = {
            "api_key": self._api_key,
            "file_type": "json",
            **(params or {}),
        }
        resp = await client.get(path, params=full_params)
        resp.raise_for_status()
        return resp.json()

    # ------------------------------------------------------------------
    # Core series fetch
    # ------------------------------------------------------------------

    async def get_series(
        self,
        series_id: str,
        observation_start: date | None = None,
        observation_end: date | None = None,
        limit: int = 100,
        sort_order: str = "desc",
    ) -> list[FREDObservation]:
        """Fetch observations for a FRED series.

        Parameters
        ----------
        series_id:
            FRED series identifier (e.g. ``"DGS10"``).
        observation_start:
            Start date.  Defaults to 90 days ago.
        observation_end:
            End date.  Defaults to today.
        limit:
            Max observations to return.
        sort_order:
            ``"asc"`` or ``"desc"``.

        Returns
        -------
        list[FREDObservation]
            Data points with date and value.
        """
        if observation_end is None:
            observation_end = date.today()
        if observation_start is None:
            observation_start = observation_end - timedelta(days=90)

        cache_key = f"fred:{series_id}:{observation_start}:{observation_end}"
        if self._cache is not None:
            cached = await self._cache.get_json(cache_key)
            if cached is not None:
                return [FREDObservation(**o) for o in cached]

        data = await self._get(
            "/series/observations",
            params={
                "series_id": series_id,
                "observation_start": str(observation_start),
                "observation_end": str(observation_end),
                "limit": limit,
                "sort_order": sort_order,
            },
        )

        observations: list[FREDObservation] = []
        for obs in data.get("observations", []):
            val_str = obs.get("value", ".")
            val = float(val_str) if val_str != "." else None
            observations.append(
                FREDObservation(
                    date=date.fromisoformat(obs["date"]),
                    value=val,
                )
            )

        if self._cache is not None and observations:
            await self._cache.set_json(
                cache_key,
                [o.model_dump(mode="json") for o in observations],
                ttl=3600,
            )

        self._log.debug(
            "fred_series_fetched", series=series_id, count=len(observations)
        )
        return observations

    async def get_latest_value(self, series_id: str) -> float | None:
        """Get the most recent non-null value for a series.

        Parameters
        ----------
        series_id:
            FRED series identifier.

        Returns
        -------
        float or None
            Latest value, or None if unavailable.
        """
        observations = await self.get_series(series_id, limit=5, sort_order="desc")
        for obs in observations:
            if obs.value is not None:
                return obs.value
        return None

    # ------------------------------------------------------------------
    # Convenience methods for key indicators
    # ------------------------------------------------------------------

    async def get_yield_curve_spread(self) -> float | None:
        """Fetch the 2s10s yield curve spread (DGS10 - DGS2).

        Returns
        -------
        float or None
            Yield spread in percentage points, or None if data unavailable.
        """
        dgs10 = await self.get_latest_value(SERIES_DGS10)
        dgs2 = await self.get_latest_value(SERIES_DGS2)
        if dgs10 is not None and dgs2 is not None:
            spread = dgs10 - dgs2
            self._log.debug("fred_yield_spread", dgs10=dgs10, dgs2=dgs2, spread=spread)
            return spread
        return None

    async def get_fed_funds_rate(self) -> float | None:
        """Fetch the latest effective Federal Funds Rate."""
        return await self.get_latest_value(SERIES_FED_FUNDS)

    async def get_hy_oas(self) -> float | None:
        """Fetch the latest High Yield OAS (credit stress indicator).

        Returns
        -------
        float or None
            ICE BofA US High Yield Option-Adjusted Spread in basis points.
        """
        return await self.get_latest_value(SERIES_HY_OAS)

    async def get_breakeven_inflation(self) -> float | None:
        """Fetch the 10-Year Breakeven Inflation Rate."""
        return await self.get_latest_value(SERIES_BREAKEVEN_10Y)

    async def get_macro_snapshot(self) -> MacroSnapshot:
        """Fetch all key macro indicators in a single snapshot.

        Returns
        -------
        MacroSnapshot
            Point-in-time snapshot of all monitored macro indicators.
        """
        cache_key = "fred:macro_snapshot"
        if self._cache is not None:
            cached = await self._cache.get_json(cache_key)
            if cached is not None:
                return MacroSnapshot(**cached)

        dgs2 = await self.get_latest_value(SERIES_DGS2)
        dgs10 = await self.get_latest_value(SERIES_DGS10)
        spread = (dgs10 - dgs2) if (dgs10 is not None and dgs2 is not None) else None
        ff = await self.get_latest_value(SERIES_FED_FUNDS)
        hy = await self.get_latest_value(SERIES_HY_OAS)
        be = await self.get_latest_value(SERIES_BREAKEVEN_10Y)

        snapshot = MacroSnapshot(
            yield_2y=dgs2,
            yield_10y=dgs10,
            yield_spread_2s10s=spread,
            fed_funds_rate=ff,
            hy_oas=hy,
            breakeven_10y=be,
        )

        if self._cache is not None:
            await self._cache.set_json(
                cache_key,
                snapshot.model_dump(mode="json"),
                ttl=3600,
            )

        self._log.info("fred_macro_snapshot", spread=spread, ff=ff, hy_oas=hy)
        return snapshot

    # ------------------------------------------------------------------
    # Historical series as DataFrame
    # ------------------------------------------------------------------

    async def get_series_df(
        self,
        series_id: str,
        observation_start: date | None = None,
        observation_end: date | None = None,
    ) -> pd.DataFrame:
        """Fetch a FRED series as a pandas DataFrame.

        Parameters
        ----------
        series_id:
            FRED series identifier.
        observation_start:
            Start date.
        observation_end:
            End date.

        Returns
        -------
        pandas.DataFrame
            DataFrame with date index and value column.
        """
        import pandas as _pd

        observations = await self.get_series(
            series_id,
            observation_start=observation_start,
            observation_end=observation_end,
            limit=10000,
            sort_order="asc",
        )
        rows = [
            {"date": o.date, "value": o.value}
            for o in observations
            if o.value is not None
        ]
        df = _pd.DataFrame(rows)
        if not df.empty:
            df["date"] = _pd.to_datetime(df["date"])
            df = df.set_index("date")
        return df
