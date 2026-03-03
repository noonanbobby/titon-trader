#!/usr/bin/env python3
"""Phase 2: Build complete ML feature matrix from historical data.

Transforms raw OHLCV + FRED macro + earnings data into 96+ ML-ready features
per ticker per day. Uses TechnicalSignalGenerator for price-based features,
then adds cross-asset, calendar, and ticker-specific features.

All features use only data available at prediction time (no future leakage).

Usage:
    uv run python scripts/build_features.py
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.signals.technical import TechnicalSignalGenerator  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATA_DIR = PROJECT_ROOT / "data" / "historical"
FEATURES_DIR = PROJECT_ROOT / "data" / "features"
REPORTS_DIR = PROJECT_ROOT / "reports"

PRIMARY_TICKERS: list[str] = [
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
BENCHMARK_TICKERS: list[str] = ["SPY", "QQQ", "IWM", "XLK"]
ALL_TICKERS: list[str] = PRIMARY_TICKERS + BENCHMARK_TICKERS

TARGET_HORIZON: int = 5
TARGET_THRESHOLD: float = 0.01  # 1% forward return


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_price_data() -> dict[str, pd.DataFrame]:
    """Load OHLCV parquet files for all tickers."""
    data: dict[str, pd.DataFrame] = {}
    for ticker in ALL_TICKERS:
        path = DATA_DIR / f"{ticker}_daily.parquet"
        if not path.exists():
            print(f"  WARNING: {path} not found, skipping {ticker}")
            continue
        df = pd.read_parquet(path)
        # Ensure timezone-aware UTC index
        if df.index.tzinfo is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")
        df = df.sort_index()
        data[ticker] = df
    return data


def load_fred_data() -> dict[str, pd.DataFrame]:
    """Load FRED macro parquet files."""
    series_files = {
        "VIXCLS": "fred_vixcls.parquet",
        "DGS2": "fred_dgs2.parquet",
        "DGS10": "fred_dgs10.parquet",
        "T10Y2Y": "fred_t10y2y.parquet",
        "BAMLH0A0HYM2": "fred_bamlh0a0hym2.parquet",
        "DFF": "fred_dff.parquet",
        "DTWEXBGS": "fred_dtwexbgs.parquet",
    }
    data: dict[str, pd.DataFrame] = {}
    for series_id, filename in series_files.items():
        path = DATA_DIR / filename
        if path.exists():
            df = pd.read_parquet(path)
            if df.index.tzinfo is None:
                df.index = df.index.tz_localize("UTC")
            data[series_id] = df
        else:
            print(f"  WARNING: {path} not found")
    return data


def load_earnings_data() -> pd.DataFrame:
    """Load earnings calendar parquet."""
    path = DATA_DIR / "earnings_calendar.parquet"
    if path.exists():
        df = pd.read_parquet(path)
        df["date"] = pd.to_datetime(df["date"])
        return df
    print("  WARNING: earnings_calendar.parquet not found")
    return pd.DataFrame()


# ---------------------------------------------------------------------------
# Feature Group: Cross-Asset (15+ features)
# ---------------------------------------------------------------------------


def compute_cross_asset_features(
    df: pd.DataFrame,
    fred: dict[str, pd.DataFrame],
    spy_df: pd.DataFrame,
    qqq_df: pd.DataFrame | None,
    xlk_df: pd.DataFrame | None,
) -> pd.DataFrame:
    """Compute cross-asset features by merging FRED macro data on date."""
    out = pd.DataFrame(index=df.index)

    # Helper: merge a FRED series onto our index with forward-fill
    def _merge_fred(series_id: str, col_name: str) -> pd.Series:
        if series_id not in fred:
            return pd.Series(np.nan, index=df.index, name=col_name)
        fdf = fred[series_id].copy()
        fdf.columns = [col_name]
        # Align to trading dates via reindex + ffill
        merged = fdf.reindex(df.index, method="ffill")
        return merged[col_name]

    # VIX features
    vix = _merge_fred("VIXCLS", "vix_level")
    out["vix_level"] = vix
    vix_mean_60 = vix.rolling(60, min_periods=20).mean()
    vix_std_60 = vix.rolling(60, min_periods=20).std()
    out["vix_zscore"] = (vix - vix_mean_60) / vix_std_60.replace(0, np.nan)
    vix_sma_20 = vix.rolling(20, min_periods=10).mean()
    out["vix_term_proxy"] = vix / vix_sma_20.replace(0, np.nan)

    # IV Rank proxy using VIX
    vix_52w_high = vix.rolling(252, min_periods=60).max()
    vix_52w_low = vix.rolling(252, min_periods=60).min()
    iv_range = vix_52w_high - vix_52w_low
    out["iv_rank_proxy"] = (vix - vix_52w_low) / iv_range.replace(0, np.nan)

    # IV Percentile proxy
    out["iv_percentile_proxy"] = vix.rolling(252, min_periods=60).apply(
        lambda x: (x.iloc[:-1] < x.iloc[-1]).sum() / max(len(x) - 1, 1),
        raw=False,
    )

    # Treasury yields
    out["yield_2y"] = _merge_fred("DGS2", "yield_2y")
    out["yield_10y"] = _merge_fred("DGS10", "yield_10y")

    # Yield spread
    spread = _merge_fred("T10Y2Y", "yield_spread_2s10s")
    out["yield_spread_2s10s"] = spread
    out["yield_spread_mom_20d"] = spread.diff(20)

    # High yield OAS
    hy_oas = _merge_fred("BAMLH0A0HYM2", "hy_oas")
    out["hy_oas"] = hy_oas
    oas_mean_60 = hy_oas.rolling(60, min_periods=20).mean()
    oas_std_60 = hy_oas.rolling(60, min_periods=20).std()
    out["hy_oas_zscore"] = (hy_oas - oas_mean_60) / oas_std_60.replace(0, np.nan)

    # Fed funds rate
    ff = _merge_fred("DFF", "fed_funds_rate")
    out["fed_funds_rate"] = ff
    out["fed_funds_90d_chg"] = ff.diff(90)

    # DXY proxy
    dxy = _merge_fred("DTWEXBGS", "dxy_proxy")
    out["dxy_proxy"] = dxy
    out["dxy_mom_20d"] = dxy.diff(20) / dxy.shift(20).replace(0, np.nan)

    # Sector relative strength: XLK/SPY ratio
    if xlk_df is not None and "close" in xlk_df.columns:
        xlk_close = xlk_df["close"].reindex(df.index, method="ffill")
        spy_close = spy_df["close"].reindex(df.index, method="ffill")
        xlk_spy_ratio = xlk_close / spy_close.replace(0, np.nan)
        out["xlk_spy_ratio_chg_20d"] = xlk_spy_ratio.pct_change(20)

    # SPY 20-day return
    spy_close = spy_df["close"].reindex(df.index, method="ffill")
    out["spy_return_20d"] = spy_close.pct_change(20)

    # SPY/QQQ ratio change (growth rotation)
    if qqq_df is not None and "close" in qqq_df.columns:
        qqq_close = qqq_df["close"].reindex(df.index, method="ffill")
        spy_qqq_ratio = spy_close / qqq_close.replace(0, np.nan)
        out["spy_qqq_ratio_chg_10d"] = spy_qqq_ratio.pct_change(10)

    # HV/IV ratio: HV(20) / VIX level
    log_ret = np.log(df["close"] / df["close"].shift(1))
    hv_20 = log_ret.rolling(20, min_periods=10).std() * np.sqrt(252) * 100
    out["hv_iv_ratio"] = hv_20 / vix.replace(0, np.nan)

    return out


# ---------------------------------------------------------------------------
# Feature Group: Calendar (8+ features)
# ---------------------------------------------------------------------------


def compute_calendar_features(
    df: pd.DataFrame,
    earnings_df: pd.DataFrame,
    ticker: str,
) -> pd.DataFrame:
    """Compute calendar-based features."""
    out = pd.DataFrame(index=df.index)

    # Day of week (0=Mon, 4=Fri)
    out["day_of_week"] = df.index.dayofweek

    # Month (1-12)
    out["month"] = df.index.month

    # Is January (January effect)
    out["is_january"] = (df.index.month == 1).astype(int)

    # Quarter-end proximity
    def _days_to_quarter_end(dt: pd.Timestamp) -> int:
        q_month = ((dt.month - 1) // 3 + 1) * 3
        q_end = pd.Timestamp(
            year=dt.year, month=q_month, day=1, tz=dt.tzinfo
        ) + pd.offsets.MonthEnd(0)
        return max((q_end - dt).days, 0)

    out["days_to_quarter_end"] = pd.Series(
        [_days_to_quarter_end(ts) for ts in df.index],
        index=df.index,
    )

    # Options expiration week (3rd Friday of month)
    def _is_opex_week(dt: pd.Timestamp) -> int:
        # Third Friday: find first day of month, find first Friday, add 14 days
        first_day = dt.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        days_until_friday = (4 - first_day.weekday()) % 7
        third_friday = first_day + timedelta(days=days_until_friday + 14)
        week_start = third_friday - timedelta(days=third_friday.weekday())
        week_end = week_start + timedelta(days=4)
        dt_norm = dt.normalize()
        return 1 if week_start <= dt_norm <= week_end else 0

    out["is_opex_week"] = pd.Series(
        [_is_opex_week(ts) for ts in df.index],
        index=df.index,
    )

    # Earnings features
    if not earnings_df.empty:
        ticker_earnings = earnings_df[
            earnings_df["ticker"].str.upper() == ticker.upper()
        ].copy()
        if not ticker_earnings.empty:
            # Normalize earnings dates to tz-naive for comparison
            earnings_dates_raw = sorted(ticker_earnings["date"].tolist())
            earnings_dates_naive = [
                d.tz_localize(None) if hasattr(d, "tz_localize") and d.tzinfo else d
                for d in earnings_dates_raw
            ]
            days_to_next = []
            days_since_last = []
            for ts in df.index:
                ts_naive = ts.tz_localize(None) if ts.tzinfo else ts
                ts_date = ts_naive.normalize()
                # Days to next earnings
                future = [d for d in earnings_dates_naive if d >= ts_date]
                if future:
                    days_to_next.append((future[0] - ts_date).days)
                else:
                    days_to_next.append(90)

                # Days since last earnings
                past = [d for d in earnings_dates_naive if d < ts_date]
                if past:
                    days_since_last.append((ts_date - past[-1]).days)
                else:
                    days_since_last.append(90)

            out["days_to_earnings"] = days_to_next
            out["days_since_earnings"] = days_since_last
            out["within_5d_earnings"] = (
                pd.Series(days_to_next, index=df.index).le(5)
            ).astype(int)
        else:
            out["days_to_earnings"] = 90
            out["days_since_earnings"] = 90
            out["within_5d_earnings"] = 0
    else:
        out["days_to_earnings"] = 90
        out["days_since_earnings"] = 90
        out["within_5d_earnings"] = 0

    return out


# ---------------------------------------------------------------------------
# Feature Group: Ticker-Specific (5+ features)
# ---------------------------------------------------------------------------


def compute_ticker_specific_features(
    df: pd.DataFrame,
    spy_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute ticker-vs-benchmark features."""
    out = pd.DataFrame(index=df.index)

    spy_close = spy_df["close"].reindex(df.index, method="ffill")
    ticker_ret = df["close"].pct_change()
    spy_ret = spy_close.pct_change()

    # 20-day rolling beta vs SPY
    cov = ticker_ret.rolling(20, min_periods=10).cov(spy_ret)
    spy_var = spy_ret.rolling(20, min_periods=10).var()
    out["beta_spy_20d"] = cov / spy_var.replace(0, np.nan)

    # 20-day rolling correlation with SPY
    out["corr_spy_20d"] = ticker_ret.rolling(20, min_periods=10).corr(spy_ret)

    # Relative strength vs SPY (20-day)
    ticker_ret_20 = df["close"].pct_change(20)
    spy_ret_20 = spy_close.pct_change(20)
    out["relative_strength_spy"] = ticker_ret_20 - spy_ret_20

    # Idiosyncratic volatility (residual after removing beta)
    beta = out["beta_spy_20d"]
    resid = ticker_ret - beta * spy_ret
    out["idio_vol_20d"] = resid.rolling(20, min_periods=10).std() * np.sqrt(252)

    # Average daily dollar volume (liquidity proxy)
    if "volume" in df.columns:
        dollar_vol = df["close"] * df["volume"]
        out["avg_dollar_vol_20d"] = dollar_vol.rolling(20, min_periods=10).mean()
        # Normalize to millions for numerical stability
        out["avg_dollar_vol_20d"] = out["avg_dollar_vol_20d"] / 1e6

    return out


# ---------------------------------------------------------------------------
# Additional features not in TechnicalSignalGenerator
# ---------------------------------------------------------------------------


def compute_extra_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute additional momentum features beyond TechnicalSignalGenerator."""
    out = pd.DataFrame(index=df.index)
    close = df["close"]

    # RSI(5) — TechnicalSignalGenerator has RSI(7,14,21) but not RSI(5)
    import pandas_ta as ta

    out["rsi_5"] = ta.rsi(close, length=5)

    # Price vs SMA ratios — already in tech generator but with different names
    # These are requested with specific naming:
    sma_200 = close.rolling(200, min_periods=100).mean()
    out["price_vs_sma200"] = close / sma_200 - 1

    # SMA alignment ratios
    sma_20 = close.rolling(20, min_periods=10).mean()
    sma_50 = close.rolling(50, min_periods=25).mean()
    out["sma20_vs_sma50"] = sma_20 / sma_50 - 1
    out["sma50_vs_sma200"] = sma_50 / sma_200 - 1

    return out


def compute_extra_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute additional volatility features."""
    out = pd.DataFrame(index=df.index)
    close = df["close"]
    log_ret = np.log(close / close.shift(1))

    # HV at additional windows
    out["hv_10_ann"] = log_ret.rolling(10, min_periods=5).std() * np.sqrt(252) * 100
    out["hv_60_ann"] = log_ret.rolling(60, min_periods=30).std() * np.sqrt(252) * 100

    # HV ratio short/long
    out["hv_ratio_10_60_ann"] = out["hv_10_ann"] / out["hv_60_ann"].replace(0, np.nan)

    # Bollinger squeeze: BB width < 20th percentile of trailing 100 days
    import pandas_ta as ta

    bbands = ta.bbands(close, length=20, std=2.0)
    if bbands is not None and not bbands.empty:
        bb_width = bbands.iloc[:, 3]  # BBB column
        bb_width_pctl20 = bb_width.rolling(100, min_periods=50).quantile(0.2)
        out["bb_squeeze"] = (bb_width < bb_width_pctl20).astype(int)

    # Keltner Channel position
    high = df["high"]
    low = df["low"]
    kc = ta.kc(high, low, close, length=20, scalar=1.5)
    if kc is not None and not kc.empty:
        kc_upper = kc.iloc[:, 2]
        kc_lower = kc.iloc[:, 0]
        kc_range = kc_upper - kc_lower
        out["kc_position"] = (close - kc_lower) / kc_range.replace(0, np.nan)

    return out


def compute_extra_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute additional volume features."""
    out = pd.DataFrame(index=df.index)
    close = df["close"]
    volume = df["volume"]

    # OBV slope (5-day)
    import pandas_ta as ta

    obv = ta.obv(close, volume)
    if obv is not None:
        obv_shifted = obv.shift(5).replace(0, np.nan)
        out["obv_slope_5d"] = obv.diff(5) / obv_shifted.abs()

    # Volume z-score
    vol_mean_20 = volume.rolling(20, min_periods=10).mean()
    vol_std_20 = volume.rolling(20, min_periods=10).std()
    out["vol_zscore_20d"] = (volume - vol_mean_20) / vol_std_20.replace(0, np.nan)

    # Relative volume
    out["rel_volume_20d"] = volume / vol_mean_20.replace(0, np.nan)

    # Volume-price trend confirmation (5-day)
    price_chg = close.diff(5)
    vol_chg = volume.diff(5)
    out["vol_price_confirm_5d"] = (np.sign(price_chg) == np.sign(vol_chg)).astype(int)

    return out


# ---------------------------------------------------------------------------
# Target variable
# ---------------------------------------------------------------------------


def compute_target(df: pd.DataFrame) -> pd.Series:
    """Compute 5-day forward return binary target."""
    close = df["close"]
    forward_return = close.shift(-TARGET_HORIZON) / close - 1.0
    target = (forward_return > TARGET_THRESHOLD).astype(float)
    target.iloc[-TARGET_HORIZON:] = np.nan
    target.name = "target"
    return target


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def build_features_for_ticker(
    ticker: str,
    price_data: dict[str, pd.DataFrame],
    fred_data: dict[str, pd.DataFrame],
    earnings_df: pd.DataFrame,
    tech_gen: TechnicalSignalGenerator,
) -> pd.DataFrame | None:
    """Build complete feature matrix for a single ticker."""
    if ticker not in price_data:
        print(f"  WARNING: No price data for {ticker}")
        return None

    df = price_data[ticker].copy()
    spy_df = price_data.get("SPY")
    qqq_df = price_data.get("QQQ")
    xlk_df = price_data.get("XLK")

    if spy_df is None:
        print("  ERROR: SPY data required for ticker-specific features")
        return None

    # 1. Technical features via TechnicalSignalGenerator (120+ features)
    print("  Computing technical features...")
    tech_df = tech_gen.calculate_features(df)

    # The tech generator drops NaN rows from warmup. Align our df.
    df = df.loc[tech_df.index]

    # 2. Extra momentum features
    extra_mom = compute_extra_momentum_features(df)

    # 3. Extra volatility features
    extra_vol = compute_extra_volatility_features(df)

    # 4. Extra volume features
    extra_volm = compute_extra_volume_features(df)

    # 5. Cross-asset features
    cross_asset = compute_cross_asset_features(df, fred_data, spy_df, qqq_df, xlk_df)

    # 6. Calendar features
    calendar = compute_calendar_features(df, earnings_df, ticker)

    # 7. Ticker-specific features
    ticker_spec = compute_ticker_specific_features(df, spy_df)

    # 8. Target variable
    target = compute_target(df)

    # Combine all features
    result = tech_df.copy()
    for extra_df in [
        extra_mom,
        extra_vol,
        extra_volm,
        cross_asset,
        calendar,
        ticker_spec,
    ]:
        for col in extra_df.columns:
            if col not in result.columns:
                result[col] = extra_df[col]

    result["target"] = target

    # Add ticker column
    result["ticker"] = ticker

    # Drop rows where target is NaN (last 5 rows)
    result = result.dropna(subset=["target"])

    # Handle NaN: drop rows where >50% of features are NaN
    feature_cols = [c for c in result.columns if c not in ["ticker", "target"]]
    nan_frac = result[feature_cols].isna().mean(axis=1)
    result = result[nan_frac <= 0.5]

    # Forward-fill remaining NaN within this ticker, then fill with column median
    result[feature_cols] = result[feature_cols].ffill()
    for col in feature_cols:
        median_val = result[col].median()
        if pd.isna(median_val):
            median_val = 0.0
        result[col] = result[col].fillna(median_val)

    # Replace inf/-inf with column max/min
    for col in feature_cols:
        col_data = result[col]
        if np.isinf(col_data).any():
            finite_vals = col_data[np.isfinite(col_data)]
            if len(finite_vals) > 0:
                result[col] = col_data.replace([np.inf], finite_vals.max()).replace(
                    [-np.inf], finite_vals.min()
                )
            else:
                result[col] = 0.0

    # Ensure target is int
    result["target"] = result["target"].astype(int)

    return result


def main() -> None:
    """Run the full feature engineering pipeline."""
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("PHASE 2: Feature Engineering Pipeline")
    print("=" * 60)

    # Load data
    print("\nLoading data...")
    price_data = load_price_data()
    fred_data = load_fred_data()
    earnings_df = load_earnings_data()
    print(f"  Loaded {len(price_data)} tickers, {len(fred_data)} FRED series")

    # Initialize technical signal generator
    tech_gen = TechnicalSignalGenerator(lookback_period=252)

    # Build features for each primary ticker
    all_frames: list[pd.DataFrame] = []
    ticker_stats: dict[str, dict] = {}

    for ticker in PRIMARY_TICKERS:
        print(f"\n[{ticker}] Building features...")
        result = build_features_for_ticker(
            ticker,
            price_data,
            fred_data,
            earnings_df,
            tech_gen,
        )
        if result is not None:
            feature_cols = [c for c in result.columns if c not in ["ticker", "target"]]
            ticker_stats[ticker] = {
                "input_rows": len(price_data[ticker]),
                "output_rows": len(result),
                "n_features": len(feature_cols),
                "target_mean": round(result["target"].mean(), 4),
                "nan_remaining": int(result[feature_cols].isna().sum().sum()),
            }
            all_frames.append(result)
            print(
                f"  {ticker}: {len(result)} rows, "
                f"{len(feature_cols)} features, "
                f"target_mean={ticker_stats[ticker]['target_mean']}"
            )

    if not all_frames:
        print("\nERROR: No feature data produced!")
        return

    # Combine all tickers
    combined = pd.concat(all_frames, axis=0)

    # Identify feature columns (exclude ticker and target)
    feature_cols = sorted(
        [c for c in combined.columns if c not in ["ticker", "target"]]
    )

    print(f"\n{'=' * 60}")
    print(f"COMBINED: {len(combined)} rows, {len(feature_cols)} features")
    print(f"{'=' * 60}")

    # Remove constant features
    constant_cols = []
    for col in feature_cols:
        if combined[col].std() < 1e-10:
            constant_cols.append(col)
    if constant_cols:
        print(
            f"\nRemoving {len(constant_cols)} constant features: {constant_cols[:5]}..."
        )
        combined = combined.drop(columns=constant_cols)
        feature_cols = [c for c in feature_cols if c not in constant_cols]

    # Save to parquet
    output_path = FEATURES_DIR / "all_features.parquet"
    combined.to_parquet(output_path, engine="pyarrow")
    print(f"\nSaved to {output_path}")

    # -----------------------------------------------------------------------
    # Validation
    # -----------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("VALIDATION")
    print(f"{'=' * 60}")

    failures: list[str] = []

    # 1. Feature count
    if len(feature_cols) < 96:
        failures.append(f"FAIL: Only {len(feature_cols)} features, need 96+")
    else:
        print(f"  PASS: {len(feature_cols)} features (>= 96)")

    # 2. Row count per ticker
    for ticker in PRIMARY_TICKERS:
        rows = len(combined[combined["ticker"] == ticker])
        if rows < 800:
            failures.append(f"FAIL: {ticker} has {rows} rows, need 800+")
        else:
            print(f"  PASS: {ticker} has {rows} rows (>= 800)")

    # 3. No NaN
    nan_count = combined[feature_cols].isna().sum().sum()
    if nan_count > 0:
        failures.append(f"FAIL: {nan_count} NaN values remain")
    else:
        print("  PASS: No NaN values in features")

    # 4. No infinite values
    inf_count = np.isinf(combined[feature_cols].values).sum()
    if inf_count > 0:
        failures.append(f"FAIL: {inf_count} Inf values in features")
    else:
        print("  PASS: No Inf values in features")

    # 5. Target is binary
    target_vals = set(combined["target"].unique())
    if not target_vals.issubset({0, 1}):
        failures.append(f"FAIL: Target has values {target_vals}, expected {{0, 1}}")
    else:
        print("  PASS: Target is binary (0/1)")

    target_mean = combined["target"].mean()
    if target_mean < 0.2 or target_mean > 0.8:
        failures.append(f"FAIL: Target mean {target_mean:.3f} too imbalanced")
    else:
        print(f"  PASS: Target mean {target_mean:.3f} (balanced)")

    # 6. Leakage check
    leakage_flags: list[str] = []
    for col in feature_cols:
        corr = combined[col].corr(combined["target"])
        if abs(corr) > 0.5:
            leakage_flags.append(f"{col}: corr={corr:.3f}")
    if leakage_flags:
        for lf in leakage_flags:
            failures.append(f"LEAKAGE: {lf}")
    else:
        print("  PASS: No feature-target correlation > 0.5")

    # 7. No constant features
    const_check = []
    for col in feature_cols:
        if combined[col].std() < 1e-8:
            const_check.append(col)
    if const_check:
        failures.append(f"FAIL: Constant features: {const_check[:5]}")
    else:
        print("  PASS: No constant features")

    if failures:
        print("\nVALIDATION ISSUES:")
        for f in failures:
            print(f"  {f}")
    else:
        print("\nAll validation checks PASSED!")

    # -----------------------------------------------------------------------
    # Report
    # -----------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("GENERATING REPORT")
    print(f"{'=' * 60}")

    generate_report(
        combined,
        feature_cols,
        ticker_stats,
        failures,
        leakage_flags,
    )

    print(f"\n{'=' * 60}")
    print("PHASE 2 COMPLETE")
    print(f"{'=' * 60}")


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def generate_report(
    df: pd.DataFrame,
    feature_cols: list[str],
    ticker_stats: dict[str, dict],
    failures: list[str],
    leakage_flags: list[str],
) -> None:
    """Generate the Phase 2 feature engineering report."""
    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")

    # Categorize features
    groups = {
        "Momentum": [],
        "Volatility": [],
        "Volume": [],
        "Cross-Asset": [],
        "Calendar": [],
        "Ticker-Specific": [],
        "Trend": [],
        "Pattern": [],
        "Derived": [],
        "Other": [],
    }

    momentum_keys = [
        "rsi",
        "macd",
        "roc",
        "stoch",
        "willr",
        "cci",
        "mfi",
        "cmo",
        "ultimate",
        "return",
        "momentum",
        "aroon",
    ]
    volatility_keys = [
        "atr",
        "bb_",
        "hv_",
        "vol_ratio_10_60",
        "kc_",
        "dc_",
        "gk_vol",
        "parkinson",
        "squeeze",
        "iv_rank",
        "iv_percent",
        "hv_iv",
        "hv_ratio",
    ]
    volume_keys = [
        "obv",
        "vol_",
        "vwap",
        "cmf",
        "ad_line",
        "force_index",
        "rel_volume",
        "vol_zscore",
        "vol_price_confirm",
    ]
    cross_asset_keys = [
        "vix_",
        "yield_",
        "hy_oas",
        "fed_funds",
        "dxy_",
        "xlk_spy",
        "spy_return",
        "spy_qqq",
    ]
    calendar_keys = [
        "day_of_week",
        "month",
        "is_january",
        "days_to_earn",
        "days_since_earn",
        "within_5d",
        "quarter_end",
        "opex",
    ]
    ticker_keys = [
        "beta_spy",
        "corr_spy",
        "relative_strength",
        "idio_vol",
        "avg_dollar_vol",
    ]
    trend_keys = [
        "sma_",
        "ema_",
        "close_to_sma",
        "close_to_ema",
        "adx",
        "di_plus",
        "di_minus",
        "supertrend",
        "linreg",
        "trend_strength",
        "price_vs_sma",
        "sma20_vs",
        "sma50_vs",
    ]
    pattern_keys = ["cdl_", "inside_bar", "outside_bar", "higher_high", "lower_low"]

    for col in feature_cols:
        col_lower = col.lower()
        categorized = False
        for key in pattern_keys:
            if key in col_lower:
                groups["Pattern"].append(col)
                categorized = True
                break
        if categorized:
            continue
        for key in momentum_keys:
            if key in col_lower:
                groups["Momentum"].append(col)
                categorized = True
                break
        if categorized:
            continue
        for key in trend_keys:
            if key in col_lower:
                groups["Trend"].append(col)
                categorized = True
                break
        if categorized:
            continue
        for key in calendar_keys:
            if key in col_lower:
                groups["Calendar"].append(col)
                categorized = True
                break
        if categorized:
            continue
        for key in ticker_keys:
            if key in col_lower:
                groups["Ticker-Specific"].append(col)
                categorized = True
                break
        if categorized:
            continue
        for key in cross_asset_keys:
            if key in col_lower:
                groups["Cross-Asset"].append(col)
                categorized = True
                break
        if categorized:
            continue
        for key in volume_keys:
            if key in col_lower:
                groups["Volume"].append(col)
                categorized = True
                break
        if categorized:
            continue
        for key in volatility_keys:
            if key in col_lower:
                groups["Volatility"].append(col)
                categorized = True
                break
        if categorized:
            continue
        groups["Other"].append(col)

    lines = [
        "# Phase 2: Feature Engineering Report",
        f"**Generated:** {now}",
        "",
        "## Feature Inventory",
        "| Group | Count | Example Features |",
        "|-------|-------|-----------------|",
    ]

    total = 0
    for group_name in [
        "Trend",
        "Momentum",
        "Volatility",
        "Volume",
        "Cross-Asset",
        "Calendar",
        "Ticker-Specific",
        "Pattern",
        "Derived",
        "Other",
    ]:
        feats = groups[group_name]
        if feats:
            examples = ", ".join(feats[:4])
            if len(feats) > 4:
                examples += ", ..."
            lines.append(f"| {group_name} | {len(feats)} | {examples} |")
            total += len(feats)

    lines.append(f"| **Total** | **{total}** | |")

    lines.extend(
        [
            "",
            "## Data Quality",
            "| Ticker | Input Rows | Output Rows | Features | Target Mean |",
            "|--------|-----------|------------|----------|-------------|",
        ]
    )

    for ticker in PRIMARY_TICKERS:
        stats = ticker_stats.get(ticker, {})
        lines.append(
            f"| {ticker} | {stats.get('input_rows', 0)} "
            f"| {stats.get('output_rows', 0)} "
            f"| {stats.get('n_features', 0)} "
            f"| {stats.get('target_mean', 0)} |"
        )

    # Leakage check
    lines.extend(["", "## Leakage Check Results", ""])
    if leakage_flags:
        for lf in leakage_flags:
            lines.append(f"- WARNING: {lf}")
    else:
        lines.append("All features passed (no correlation > 0.5 with target).")

    # Feature distribution
    lines.extend(["", "## Feature Distribution Summary", ""])
    stds = df[feature_cols].std().sort_values(ascending=False)
    lines.append("**Top 10 most variable features:**")
    for feat, std_val in stds.head(10).items():
        lines.append(f"- `{feat}`: std={std_val:.4f}")
    lines.append("")
    lines.append("**Bottom 10 least variable features:**")
    for feat, std_val in stds.tail(10).items():
        lines.append(f"- `{feat}`: std={std_val:.6f}")

    # Top correlations
    lines.extend(["", "## Top Feature-Target Correlations", ""])
    corrs = df[feature_cols].corrwith(df["target"]).abs().sort_values(ascending=False)
    lines.append("| Feature | Abs Correlation |")
    lines.append("|---------|----------------|")
    for feat, corr_val in corrs.head(20).items():
        lines.append(f"| {feat} | {corr_val:.4f} |")

    # Validation
    lines.extend(["", "## Validation Results", ""])
    if not failures:
        lines.append("**All checks PASSED.**")
    else:
        for f in failures:
            lines.append(f"- {f}")

    lines.extend(
        [
            "",
            "## Output Location",
            f"- `data/features/all_features.parquet`"
            f" ({len(df)} rows x {len(feature_cols)} features)",
            "",
        ]
    )

    report = "\n".join(lines) + "\n"
    report_path = REPORTS_DIR / "PHASE_2_FEATURES.md"
    report_path.write_text(report)
    print(f"Report saved to {report_path}")


if __name__ == "__main__":
    main()
