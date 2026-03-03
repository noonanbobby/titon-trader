# Phase 1: Historical Data Seeding Report
**Generated:** 2026-03-03 22:06:32 UTC

## Data Inventory

### Price Data (yfinance)
| Ticker | Rows | Start Date | End Date | Max Gap (bdays) | Status |
|--------|------|------------|----------|-----------------|--------|
| AAPL | 1044 | 2022-01-03 | 2026-03-03 | 2 | PASS |
| NVDA | 1044 | 2022-01-03 | 2026-03-03 | 2 | PASS |
| MSFT | 1044 | 2022-01-03 | 2026-03-03 | 2 | PASS |
| GOOGL | 1044 | 2022-01-03 | 2026-03-03 | 2 | PASS |
| META | 1044 | 2022-01-03 | 2026-03-03 | 2 | PASS |
| AMZN | 1044 | 2022-01-03 | 2026-03-03 | 2 | PASS |
| TSLA | 1044 | 2022-01-03 | 2026-03-03 | 2 | PASS |
| AMD | 1044 | 2022-01-03 | 2026-03-03 | 2 | PASS |
| AVGO | 1044 | 2022-01-03 | 2026-03-03 | 2 | PASS |
| CRM | 1044 | 2022-01-03 | 2026-03-03 | 2 | PASS |
| SPY | 1044 | 2022-01-03 | 2026-03-03 | 2 | PASS |
| QQQ | 1044 | 2022-01-03 | 2026-03-03 | 2 | PASS |
| IWM | 1044 | 2022-01-03 | 2026-03-03 | 2 | PASS |
| XLK | 1044 | 2022-01-03 | 2026-03-03 | 2 | PASS |

### FRED Macro Data
| Series | Description | Rows | Start Date | End Date | Status |
|--------|-------------|------|------------|----------|--------|
| VIXCLS | VIX Daily Close | 1072 | 2022-01-03 | 2026-03-02 | PASS |
| DGS2 | 2-Year Treasury Yield | 1038 | 2022-01-03 | 2026-03-02 | PASS |
| DGS10 | 10-Year Treasury Yield | 1038 | 2022-01-03 | 2026-03-02 | PASS |
| T10Y2Y | 2Y/10Y Yield Spread | 1039 | 2022-01-03 | 2026-03-03 | PASS |
| BAMLH0A0HYM2 | High Yield OAS | 1089 | 2022-01-03 | 2026-03-02 | PASS |
| DFF | Fed Funds Effective Rate | 1522 | 2022-01-01 | 2026-03-02 | PASS |
| DTWEXBGS | Trade-Weighted USD Index | 1039 | 2022-01-03 | 2026-02-27 | PASS |

### Earnings Calendar (yfinance)
| Ticker | Events Found | Status |
|--------|-------------|--------|
| AAPL | 17 | PASS |
| NVDA | 17 | PASS |
| MSFT | 17 | PASS |
| GOOGL | 17 | PASS |
| META | 17 | PASS |
| AMZN | 17 | PASS |
| TSLA | 17 | PASS |
| AMD | 17 | PASS |
| AVGO | 16 | PASS |
| CRM | 17 | PASS |

## Storage Locations
- **QuestDB tables:** `daily_ohlcv`, `macro_data`, `market_ticks`, `gex_levels`, `signal_scores`
- **Parquet files:** `data/historical/`

### Parquet File Inventory
- `AAPL_daily.parquet`: 1044 rows
- `AMD_daily.parquet`: 1044 rows
- `AMZN_daily.parquet`: 1044 rows
- `AVGO_daily.parquet`: 1044 rows
- `CRM_daily.parquet`: 1044 rows
- `GOOGL_daily.parquet`: 1044 rows
- `IWM_daily.parquet`: 1044 rows
- `META_daily.parquet`: 1044 rows
- `MSFT_daily.parquet`: 1044 rows
- `NVDA_daily.parquet`: 1044 rows
- `QQQ_daily.parquet`: 1044 rows
- `SPY_daily.parquet`: 1044 rows
- `TSLA_daily.parquet`: 1044 rows
- `XLK_daily.parquet`: 1044 rows
- `earnings_calendar.parquet`: 169 rows
- `fred_bamlh0a0hym2.parquet`: 1089 rows
- `fred_dff.parquet`: 1522 rows
- `fred_dgs10.parquet`: 1038 rows
- `fred_dgs2.parquet`: 1038 rows
- `fred_dtwexbgs.parquet`: 1039 rows
- `fred_t10y2y.parquet`: 1039 rows
- `fred_vixcls.parquet`: 1072 rows
- `vix_daily.parquet`: 1072 rows

## Validation Results

**All checks PASSED.**

## Notes

- Polygon.io plan supports ~2 years of historical data. yfinance used for full 4+ year history.
- Polygon reserved for real-time options data during live trading.
- VWAP from yfinance is approximated as (High + Low + Close) / 3.

