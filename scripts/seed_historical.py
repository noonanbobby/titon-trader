#!/usr/bin/env python3
"""Phase 1: Seed comprehensive historical data.

Pulls 4+ years of daily data for ML walk-forward training:
- OHLCV bars for 14 tickers via yfinance (reliable, free, full history)
- VIX daily close from FRED + yfinance backup
- 7 FRED macro series (VIX, yields, credit, USD)
- Earnings calendar for 10 primary tickers from Finnhub

Stores to QuestDB (daily_ohlcv, macro_data tables) and parquet files.

Note: Polygon.io plan only supports ~2 years of history. yfinance provides
the full 4+ years needed for walk-forward training. Polygon is reserved for
real-time options data during live trading.
"""

from __future__ import annotations

import asyncio
import os
import sys
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yfinance as yf  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

from src.data.fred import FREDClient  # noqa: E402
from src.data.questdb import QuestDBClient  # noqa: E402

load_dotenv(PROJECT_ROOT / ".env")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATA_DIR = PROJECT_ROOT / "data" / "historical"
REPORTS_DIR = PROJECT_ROOT / "reports"
START_DATE = date(2022, 1, 1)
END_DATE = date.today()

PRICE_TICKERS: list[str] = [
    "AAPL",
    "NVDA",
    "MSFT",
    "GOOGL",
    "META",
    "AMZN",
    "TSLA",
    "AMD",
    "AVGO",
    "CRM",
    "SPY",
    "QQQ",
    "IWM",
    "XLK",
]

FRED_SERIES: dict[str, str] = {
    "VIXCLS": "VIX Daily Close",
    "DGS2": "2-Year Treasury Yield",
    "DGS10": "10-Year Treasury Yield",
    "T10Y2Y": "2Y/10Y Yield Spread",
    "BAMLH0A0HYM2": "High Yield OAS",
    "DFF": "Fed Funds Effective Rate",
    "DTWEXBGS": "Trade-Weighted USD Index",
}

EARNINGS_TICKERS: list[str] = [
    "AAPL",
    "NVDA",
    "MSFT",
    "GOOGL",
    "META",
    "AMZN",
    "TSLA",
    "AMD",
    "AVGO",
    "CRM",
]


# ---------------------------------------------------------------------------
# QuestDB table creation
# ---------------------------------------------------------------------------


async def create_questdb_tables(qdb: QuestDBClient) -> None:
    """Create daily_ohlcv and macro_data tables in QuestDB."""
    if qdb._http_client is None:
        raise RuntimeError("QuestDB not connected")

    ddl_statements = [
        """
        CREATE TABLE IF NOT EXISTS daily_ohlcv (
            timestamp TIMESTAMP,
            ticker SYMBOL,
            open DOUBLE,
            high DOUBLE,
            low DOUBLE,
            close DOUBLE,
            volume LONG,
            vwap DOUBLE
        ) TIMESTAMP(timestamp) PARTITION BY MONTH WAL
        """,
        """
        CREATE TABLE IF NOT EXISTS macro_data (
            timestamp TIMESTAMP,
            series_id SYMBOL,
            value DOUBLE
        ) TIMESTAMP(timestamp) PARTITION BY MONTH WAL
        """,
    ]

    for ddl in ddl_statements:
        resp = await qdb._http_client.get("/exec", params={"query": ddl.strip()})
        if resp.status_code != 200:
            print(f"  WARNING: DDL failed: {resp.text[:200]}")
        else:
            print("  Table created/verified OK")

    # Also create the standard time-series tables
    await qdb.ensure_tables()


# ---------------------------------------------------------------------------
# 1A: Price data via yfinance
# ---------------------------------------------------------------------------


async def seed_price_data(qdb: QuestDBClient) -> dict[str, dict]:
    """Fetch OHLCV bars via yfinance and store in parquet + QuestDB.

    yfinance is synchronous, so we run it in a thread executor.
    """
    results: dict[str, dict] = {}

    def _download_ticker(ticker: str) -> pd.DataFrame:
        """Download daily OHLCV for a single ticker."""
        t = yf.Ticker(ticker)
        df = t.history(
            start=str(START_DATE),
            end=str(END_DATE + timedelta(days=1)),
            interval="1d",
            auto_adjust=True,
        )
        return df

    loop = asyncio.get_event_loop()

    for ticker in PRICE_TICKERS:
        print(f"\n[YFINANCE] Fetching {ticker}...")
        try:
            df = await loop.run_in_executor(None, _download_ticker, ticker)

            if df.empty:
                print(f"  WARNING: {ticker} returned empty")
                results[ticker] = {"rows": 0, "start": "N/A", "end": "N/A"}
                continue

            # Standardize column names
            df = df.rename(
                columns={
                    "Open": "open",
                    "High": "high",
                    "Low": "low",
                    "Close": "close",
                    "Volume": "volume",
                }
            )

            # Compute VWAP approximation: (high + low + close) / 3
            df["vwap"] = (df["high"] + df["low"] + df["close"]) / 3.0

            # Keep only needed columns
            cols = ["open", "high", "low", "close", "volume", "vwap"]
            df = df[[c for c in cols if c in df.columns]]

            # Ensure timezone-aware index
            if df.index.tzinfo is None:
                df.index = df.index.tz_localize("UTC")
            else:
                df.index = df.index.tz_convert("UTC")

            df.index.name = "timestamp"

            # Remove duplicates
            df = df[~df.index.duplicated(keep="last")]
            df = df.sort_index()

            # Save parquet
            parquet_path = DATA_DIR / f"{ticker}_daily.parquet"
            df.to_parquet(parquet_path, engine="pyarrow")

            # Write to QuestDB via ILP
            for ts, row in df.iterrows():
                await qdb.write_ilp_raw(
                    table="daily_ohlcv",
                    tags={"ticker": ticker},
                    fields={
                        "open": float(row["open"]),
                        "high": float(row["high"]),
                        "low": float(row["low"]),
                        "close": float(row["close"]),
                        "volume": int(row["volume"]),
                        "vwap": float(row["vwap"]),
                    },
                    ts=ts.to_pydatetime(),
                )

            await qdb.flush()

            results[ticker] = {
                "rows": len(df),
                "start": str(df.index.min().date()),
                "end": str(df.index.max().date()),
            }
            print(
                f"  {ticker}: {len(df)} rows, "
                f"{df.index.min().date()} -> {df.index.max().date()}"
            )

        except Exception as e:
            print(f"  ERROR fetching {ticker}: {e}")
            results[ticker] = {
                "rows": 0,
                "start": "N/A",
                "end": "N/A",
                "error": str(e),
            }

    return results


# ---------------------------------------------------------------------------
# 1B + 1C: FRED macro data
# ---------------------------------------------------------------------------


async def seed_fred_data(fred: FREDClient, qdb: QuestDBClient) -> dict[str, dict]:
    """Fetch FRED macro series and store in parquet + QuestDB."""
    results: dict[str, dict] = {}

    for series_id, name in FRED_SERIES.items():
        print(f"\n[FRED] Fetching {series_id} ({name})...")
        try:
            df = await fred.get_series_df(
                series_id=series_id,
                observation_start=START_DATE,
                observation_end=END_DATE,
            )

            if df.empty:
                print(f"  WARNING: {series_id} returned empty")
                results[series_id] = {"rows": 0, "start": "N/A", "end": "N/A"}
                continue

            # Save parquet
            parquet_path = DATA_DIR / f"fred_{series_id.lower()}.parquet"
            df.to_parquet(parquet_path, engine="pyarrow")

            # Write to QuestDB
            for ts, row in df.iterrows():
                ts_dt = ts.to_pydatetime()
                if ts_dt.tzinfo is None:
                    ts_dt = ts_dt.replace(tzinfo=UTC)
                await qdb.write_ilp_raw(
                    table="macro_data",
                    tags={"series_id": series_id},
                    fields={"value": float(row["value"])},
                    ts=ts_dt,
                )

            await qdb.flush()

            results[series_id] = {
                "rows": len(df),
                "start": str(df.index.min().date()),
                "end": str(df.index.max().date()),
            }
            print(
                f"  {series_id}: {len(df)} rows, "
                f"{df.index.min().date()} -> {df.index.max().date()}"
            )

        except Exception as e:
            print(f"  ERROR fetching {series_id}: {e}")
            results[series_id] = {
                "rows": 0,
                "start": "N/A",
                "end": "N/A",
                "error": str(e),
            }

    # Also save VIX as standalone parquet for regime detection
    vix_path = DATA_DIR / "fred_vixcls.parquet"
    if vix_path.exists():
        vix_df = pd.read_parquet(vix_path)
        vix_df.to_parquet(DATA_DIR / "vix_daily.parquet", engine="pyarrow")
        print(f"\n  VIX standalone copy saved ({len(vix_df)} rows)")

    return results


# ---------------------------------------------------------------------------
# 1D: Earnings calendar from Finnhub
# ---------------------------------------------------------------------------


async def seed_earnings_data() -> dict[str, int]:
    """Fetch earnings calendar via yfinance and store as parquet.

    yfinance provides comprehensive historical earnings dates with EPS data.
    Finnhub free tier only returns limited recent earnings.
    """
    results: dict[str, int] = {}
    all_earnings: list[dict] = []
    loop = asyncio.get_event_loop()

    def _get_earnings(ticker: str) -> pd.DataFrame:
        t = yf.Ticker(ticker)
        return t.get_earnings_dates(limit=50)

    for ticker in EARNINGS_TICKERS:
        print(f"\n[YFINANCE] Fetching earnings for {ticker}...")
        try:
            ed = await loop.run_in_executor(None, _get_earnings, ticker)

            if ed is None or ed.empty:
                print(f"  WARNING: {ticker} returned no earnings")
                results[ticker] = 0
                continue

            # Filter to our date range
            start_ts = pd.Timestamp(START_DATE, tz="America/New_York")
            end_ts = pd.Timestamp(END_DATE, tz="America/New_York")
            mask = (ed.index >= start_ts) & (ed.index <= end_ts)
            ed_filtered = ed[mask]

            for ts, row in ed_filtered.iterrows():
                ear_date = ts.date() if hasattr(ts, "date") else ts
                all_earnings.append(
                    {
                        "ticker": ticker,
                        "date": str(ear_date),
                        "hour": "amc" if ts.hour >= 16 else "bmo",
                        "eps_estimate": row.get("EPS Estimate"),
                        "eps_actual": row.get("Reported EPS"),
                        "surprise_pct": row.get("Surprise(%)"),
                    }
                )

            results[ticker] = len(ed_filtered)
            print(f"  {ticker}: {len(ed_filtered)} earnings events")

        except Exception as e:
            print(f"  ERROR: {ticker}: {e}")
            results[ticker] = 0

    if all_earnings:
        earnings_df = pd.DataFrame(all_earnings)
        earnings_df.to_parquet(
            DATA_DIR / "earnings_calendar.parquet",
            engine="pyarrow",
            index=False,
        )
        print(f"\nEarnings calendar saved: {len(all_earnings)} total events")

    return results


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def compute_max_business_day_gap(df: pd.DataFrame) -> int:
    """Compute the maximum gap in business days between consecutive rows."""
    if len(df) < 2:
        return 0
    idx = df.index.sort_values()
    gaps = []
    for i in range(1, len(idx)):
        d1 = idx[i - 1]
        d2 = idx[i]
        # Handle timezone-aware timestamps
        if hasattr(d1, "date"):
            d1 = d1.date()
        if hasattr(d2, "date"):
            d2 = d2.date()
        bdays = int(np.busday_count(d1, d2))
        gaps.append(bdays)
    return max(gaps) if gaps else 0


def validate_data(
    price_results: dict[str, dict],
    fred_results: dict[str, dict],
) -> list[str]:
    """Run all validation checks. Returns list of failures."""
    failures: list[str] = []

    for ticker in PRICE_TICKERS:
        info = price_results.get(ticker, {})
        rows = info.get("rows", 0)
        start = info.get("start", "N/A")
        end = info.get("end", "N/A")

        if rows < 1000:
            failures.append(f"FAIL: {ticker} has {rows} rows, need 1000+")
        if start != "N/A" and start > "2022-03-01":
            failures.append(f"FAIL: {ticker} starts at {start}, need 2022 or earlier")
        if end != "N/A" and end < "2026-02-01":
            failures.append(f"WARN: {ticker} ends at {end}, may not have latest data")

        # Check gap
        parquet_path = DATA_DIR / f"{ticker}_daily.parquet"
        if parquet_path.exists():
            df = pd.read_parquet(parquet_path)
            if not df.empty:
                max_gap = compute_max_business_day_gap(df)
                if max_gap > 5:
                    failures.append(f"FAIL: {ticker} has {max_gap}-business-day gap")

    # FRED data: every series must have 500+ observations
    for series_id in FRED_SERIES:
        info = fred_results.get(series_id, {})
        rows = info.get("rows", 0)
        if rows < 500:
            failures.append(f"FAIL: FRED {series_id} has {rows} rows, need 500+")

    # VIX specifically must have 1000+ rows
    vix_info = fred_results.get("VIXCLS", {})
    vix_rows = vix_info.get("rows", 0)
    if vix_rows < 1000:
        failures.append(f"FAIL: VIX has {vix_rows} rows, need 1000+")

    return failures


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def generate_report(
    price_results: dict[str, dict],
    fred_results: dict[str, dict],
    earnings_results: dict[str, int],
    failures: list[str],
) -> str:
    """Generate the Phase 1 data report."""
    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")

    lines = [
        "# Phase 1: Historical Data Seeding Report",
        f"**Generated:** {now}",
        "",
        "## Data Inventory",
        "",
        "### Price Data (yfinance)",
        "| Ticker | Rows | Start Date | End Date | Max Gap (bdays) | Status |",
        "|--------|------|------------|----------|-----------------|--------|",
    ]

    for ticker in PRICE_TICKERS:
        info = price_results.get(ticker, {})
        rows = info.get("rows", 0)
        start = info.get("start", "N/A")
        end = info.get("end", "N/A")

        gap_str = "N/A"
        parquet_path = DATA_DIR / f"{ticker}_daily.parquet"
        if parquet_path.exists():
            df = pd.read_parquet(parquet_path)
            if not df.empty:
                gap = compute_max_business_day_gap(df)
                gap_str = str(gap)

        status = "PASS" if rows >= 1000 else "FAIL"
        if "error" in info:
            status = f"ERROR: {info['error'][:30]}"

        lines.append(f"| {ticker} | {rows} | {start} | {end} | {gap_str} | {status} |")

    lines.extend(
        [
            "",
            "### FRED Macro Data",
            "| Series | Description | Rows | Start Date | End Date | Status |",
            "|--------|-------------|------|------------|----------|--------|",
        ]
    )

    for series_id, name in FRED_SERIES.items():
        info = fred_results.get(series_id, {})
        rows = info.get("rows", 0)
        start = info.get("start", "N/A")
        end = info.get("end", "N/A")
        min_rows = 1000 if series_id == "VIXCLS" else 500
        status = "PASS" if rows >= min_rows else "FAIL"
        if "error" in info:
            status = f"ERROR: {info['error'][:30]}"
        lines.append(f"| {series_id} | {name} | {rows} | {start} | {end} | {status} |")

    lines.extend(
        [
            "",
            "### Earnings Calendar (yfinance)",
            "| Ticker | Events Found | Status |",
            "|--------|-------------|--------|",
        ]
    )

    for ticker in EARNINGS_TICKERS:
        count = earnings_results.get(ticker, 0)
        status = "PASS" if count > 0 else "WARN"
        lines.append(f"| {ticker} | {count} | {status} |")

    lines.extend(
        [
            "",
            "## Storage Locations",
            "- **QuestDB tables:** `daily_ohlcv`, `macro_data`, `market_ticks`, "
            "`gex_levels`, `signal_scores`",
            "- **Parquet files:** `data/historical/`",
            "",
            "### Parquet File Inventory",
        ]
    )

    for f in sorted(DATA_DIR.glob("*.parquet")):
        df = pd.read_parquet(f)
        lines.append(f"- `{f.name}`: {len(df)} rows")

    lines.extend(
        [
            "",
            "## Validation Results",
            "",
        ]
    )

    if not failures:
        lines.append("**All checks PASSED.**")
    else:
        for f in failures:
            lines.append(f"- {f}")

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Polygon.io plan supports ~2 years of historical data. "
            "yfinance used for full 4+ year history.",
            "- Polygon reserved for real-time options data during live trading.",
            "- VWAP from yfinance is approximated as (High + Low + Close) / 3.",
            "",
        ]
    )

    report = "\n".join(lines) + "\n"

    report_path = REPORTS_DIR / "PHASE_1_DATA.md"
    report_path.write_text(report)
    print(f"\nReport saved to {report_path}")
    return report


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> None:
    """Run the full Phase 1 data seeding pipeline."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("PHASE 1: Historical Data Seeding")
    print(f"Date range: {START_DATE} -> {END_DATE}")
    print("=" * 60)

    # Initialize clients
    fred = FREDClient(api_key=os.environ["FRED_API_KEY"])
    qdb = QuestDBClient(host="localhost", ilp_port=9009, http_port=9000)

    try:
        await qdb.connect()
        print("\nQuestDB connected. Creating tables...")
        await create_questdb_tables(qdb)

        # 1A: Price data
        print("\n" + "=" * 60)
        print("1A: PRICE DATA (yfinance)")
        print("=" * 60)
        price_results = await seed_price_data(qdb)

        # 1B + 1C: FRED macro data
        print("\n" + "=" * 60)
        print("1B/1C: FRED MACRO DATA")
        print("=" * 60)
        fred_results = await seed_fred_data(fred, qdb)

        # 1D: Earnings calendar
        print("\n" + "=" * 60)
        print("1D: EARNINGS CALENDAR (yfinance)")
        print("=" * 60)
        earnings_results = await seed_earnings_data()

        # Validation
        print("\n" + "=" * 60)
        print("VALIDATION")
        print("=" * 60)
        failures = validate_data(price_results, fred_results)
        if failures:
            print("\nValidation issues:")
            for f in failures:
                print(f"  {f}")
        else:
            print("\nAll validation checks PASSED!")

        # Report
        print("\n" + "=" * 60)
        print("REPORT")
        print("=" * 60)
        generate_report(price_results, fred_results, earnings_results, failures)

    finally:
        await fred.close()
        await qdb.close()

    print("\n" + "=" * 60)
    print("PHASE 1 COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
