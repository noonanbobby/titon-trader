"""Bootstrap training script for Project Titan.

Fetches historical data from Polygon.io, computes technical indicators
matching the ensemble's 48-feature layout, creates forward-return labels,
and trains an XGBoost model via walk-forward cross-validation.

Produces ``models/ensemble_xgb.json`` which the live system loads at
startup to replace the deterministic fallback model.

Usage::

    # Inside the titan container:
    uv run python scripts/train_bootstrap.py

    # Or from the host:
    docker compose exec titan uv run python scripts/train_bootstrap.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

POLYGON_API_KEY: str = os.environ.get("POLYGON_API_KEY", "")
MODELS_DIR: Path = Path("models")
MODEL_OUTPUT: Path = MODELS_DIR / "ensemble_xgb.json"
CALIBRATOR_OUTPUT: Path = MODELS_DIR / "ensemble_xgb.calibrator.pkl"

# Universe — core tickers for bootstrap training (kept small for API limits)
TICKERS: list[str] = [
    "SPY",
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "GOOGL",
    "META",
    "AMD",
    "TSLA",
    "JPM",
    "BA",
    "XOM",
    "GS",
    "BAC",
]

# Training parameters
LOOKBACK_YEARS: int = 2  # Polygon free tier limited to ~2 years
FORWARD_HORIZON_DAYS: int = 5
FORWARD_THRESHOLD: float = 0.01  # 1% forward return = positive
N_FOLDS: int = 5
EMBARGO_DAYS: int = 5

# Ensemble feature count (must match EnsembleSignalGenerator)
TOTAL_FEATURES: int = 48

# Rate limit for Polygon.io free tier: 5 requests/minute
POLYGON_RATE_DELAY: float = 13.0  # seconds between requests
MAX_RETRIES: int = 3


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------


async def fetch_polygon_bars(
    client: httpx.AsyncClient,
    ticker: str,
    from_date: str,
    to_date: str,
) -> pd.DataFrame:
    """Fetch daily bars from Polygon.io with retry on 429."""
    url = (
        f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day"
        f"/{from_date}/{to_date}"
    )
    params = {
        "adjusted": "true",
        "sort": "asc",
        "limit": "50000",
        "apiKey": POLYGON_API_KEY,
    }

    for attempt in range(MAX_RETRIES):
        resp = await client.get(url, params=params, timeout=30.0)
        if resp.status_code == 429:
            wait = POLYGON_RATE_DELAY * (attempt + 2)
            print(f"    Rate limited, waiting {wait:.0f}s (attempt {attempt + 1})")
            await asyncio.sleep(wait)
            continue
        resp.raise_for_status()
        break
    else:
        print(f"  WARNING: Failed after {MAX_RETRIES} retries for {ticker}")
        return pd.DataFrame()

    data = resp.json()

    results = data.get("results", [])
    if not results:
        print(f"  WARNING: No data for {ticker} ({from_date} to {to_date})")
        return pd.DataFrame()

    df = pd.DataFrame(results)
    df["date"] = pd.to_datetime(df["t"], unit="ms").dt.date
    df = df.rename(
        columns={
            "o": "open",
            "h": "high",
            "l": "low",
            "c": "close",
            "v": "volume",
        }
    )
    df = df.set_index("date")
    df.index = pd.DatetimeIndex(df.index)
    df = df[["open", "high", "low", "close", "volume"]].copy()
    return df


async def fetch_vix_data(
    client: httpx.AsyncClient,
    from_date: str,
    to_date: str,
) -> pd.Series:
    """Fetch VIX daily data from Polygon.io."""
    # Try I:VIX first (index), then VIX (futures proxy)
    for vix_ticker in ["I:VIX", "VIX"]:
        try:
            df = await fetch_polygon_bars(client, vix_ticker, from_date, to_date)
            if not df.empty:
                return df["close"]
        except Exception:
            continue

    # Fallback: synthesize from SPY volatility
    print("  WARNING: VIX data unavailable, synthesizing from SPY volatility")
    return pd.Series(dtype=float)


# ---------------------------------------------------------------------------
# Technical indicator computation
# ---------------------------------------------------------------------------


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index."""
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window=period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def compute_macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.Series:
    """MACD histogram."""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line - signal_line


def compute_adx(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """Average Directional Index."""
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()

    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

    plus_di = 100.0 * plus_dm.rolling(window=period).mean() / atr.replace(0, np.nan)
    minus_di = 100.0 * minus_dm.rolling(window=period).mean() / atr.replace(0, np.nan)

    dx = 100.0 * abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.rolling(window=period).mean()
    return adx


def compute_bollinger_width(
    close: pd.Series,
    period: int = 20,
    std_dev: float = 2.0,
) -> pd.Series:
    """Bollinger Band width as percentage of SMA."""
    sma = close.rolling(window=period).mean()
    std = close.rolling(window=period).std()
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    return (upper - lower) / sma.replace(0, np.nan)


def compute_atr_pct(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """ATR as percentage of close."""
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr / close.replace(0, np.nan)


def compute_stochastic(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    k_period: int = 14,
    d_period: int = 3,
) -> tuple[pd.Series, pd.Series]:
    """Stochastic %K and %D."""
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()
    denom = (highest_high - lowest_low).replace(0, np.nan)
    k = 100.0 * (close - lowest_low) / denom
    d = k.rolling(window=d_period).mean()
    return k, d


def compute_cci(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 20,
) -> pd.Series:
    """Commodity Channel Index."""
    tp = (high + low + close) / 3.0
    sma = tp.rolling(window=period).mean()
    mad = tp.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean())
    return (tp - sma) / (0.015 * mad.replace(0, np.nan))


def compute_mfi(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    period: int = 14,
) -> pd.Series:
    """Money Flow Index."""
    tp = (high + low + close) / 3.0
    mf = tp * volume
    delta = tp.diff()
    pos_mf = mf.where(delta > 0, 0.0).rolling(window=period).sum()
    neg_mf = mf.where(delta <= 0, 0.0).rolling(window=period).sum()
    mfi = 100.0 - (100.0 / (1.0 + pos_mf / neg_mf.replace(0, np.nan)))
    return mfi


def compute_williams_r(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """Williams %R."""
    highest = high.rolling(window=period).max()
    lowest = low.rolling(window=period).min()
    denom = (highest - lowest).replace(0, np.nan)
    return -100.0 * (highest - close) / denom


def compute_obv_slope(
    close: pd.Series, volume: pd.Series, period: int = 20
) -> pd.Series:
    """On-Balance Volume slope (normalized)."""
    sign = np.sign(close.diff())
    obv = (sign * volume).cumsum()
    # Slope of OBV over period, normalized by mean volume
    obv_slope = obv.diff(period) / volume.rolling(window=period).mean().replace(
        0, np.nan
    )
    return obv_slope


def build_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build the 20 technical features matching the ensemble's feature layout.

    Feature indices 0..19 in the ensemble vector:
        RSI_14, MACD_hist, ADX, BB_width, ATR_pct, trend_strength,
        SMA_20_50_cross, SMA_50_200_cross, volume_ratio, OBV_slope,
        VWAP_deviation, stoch_K, stoch_D, CCI_20, MFI_14, williams_R,
        ROC_10, TRIX, ultimate_osc, technical_composite
    """
    h, lo, c, v = df["high"], df["low"], df["close"], df["volume"]

    features = pd.DataFrame(index=df.index)

    # 0: RSI_14 — normalize to [0, 1]
    features["RSI_14"] = compute_rsi(c) / 100.0

    # 1: MACD_hist — normalize by close
    features["MACD_hist"] = compute_macd(c) / c.replace(0, np.nan)

    # 2: ADX — normalize to [0, 1]
    features["ADX"] = compute_adx(h, lo, c) / 100.0

    # 3: BB_width
    features["BB_width"] = compute_bollinger_width(c)

    # 4: ATR_pct
    features["ATR_pct"] = compute_atr_pct(h, lo, c)

    # 5: trend_strength — SMA(20) vs SMA(50) distance as pct
    sma20 = c.rolling(20).mean()
    sma50 = c.rolling(50).mean()
    sma200 = c.rolling(200).mean()
    features["trend_strength"] = (sma20 - sma50) / c.replace(0, np.nan)

    # 6: SMA_20_50_cross — 1 if SMA20 > SMA50, -1 otherwise
    features["SMA_20_50_cross"] = np.where(sma20 > sma50, 1.0, -1.0)

    # 7: SMA_50_200_cross
    features["SMA_50_200_cross"] = np.where(sma50 > sma200, 1.0, -1.0)

    # 8: volume_ratio — current vs 20-day average
    vol_avg = v.rolling(20).mean()
    features["volume_ratio"] = (v / vol_avg.replace(0, np.nan)).clip(0, 5) / 5.0

    # 9: OBV_slope
    obv_slope = compute_obv_slope(c, v)
    features["OBV_slope"] = obv_slope.clip(-5, 5) / 5.0

    # 10: VWAP_deviation — use daily VWAP proxy
    typical_price = (h + lo + c) / 3.0
    vwap_proxy = (typical_price * v).rolling(20).sum() / v.rolling(20).sum().replace(
        0, np.nan
    )
    features["VWAP_deviation"] = (c - vwap_proxy) / c.replace(0, np.nan)

    # 11-12: stoch_K, stoch_D
    k, d = compute_stochastic(h, lo, c)
    features["stoch_K"] = k / 100.0
    features["stoch_D"] = d / 100.0

    # 13: CCI_20 — normalize
    features["CCI_20"] = (compute_cci(h, lo, c) / 200.0).clip(-1, 1)

    # 14: MFI_14 — normalize to [0, 1]
    features["MFI_14"] = compute_mfi(h, lo, c, v) / 100.0

    # 15: williams_R — normalize to [-1, 0] → [0, 1]
    features["williams_R"] = (compute_williams_r(h, lo, c) + 100.0) / 100.0

    # 16: ROC_10 — rate of change
    features["ROC_10"] = c.pct_change(10).clip(-0.3, 0.3) / 0.3

    # 17: TRIX — triple exponential EMA
    ema1 = c.ewm(span=15, adjust=False).mean()
    ema2 = ema1.ewm(span=15, adjust=False).mean()
    ema3 = ema2.ewm(span=15, adjust=False).mean()
    features["TRIX"] = (ema3.pct_change() * 10000).clip(-5, 5) / 5.0

    # 18: ultimate_osc — simplified
    bp = c - pd.concat([lo, c.shift(1)], axis=1).min(axis=1)
    tr = pd.concat([h - lo, abs(h - c.shift(1)), abs(lo - c.shift(1))], axis=1).max(
        axis=1
    )
    avg7 = bp.rolling(7).sum() / tr.rolling(7).sum().replace(0, np.nan)
    avg14 = bp.rolling(14).sum() / tr.rolling(14).sum().replace(0, np.nan)
    avg28 = bp.rolling(28).sum() / tr.rolling(28).sum().replace(0, np.nan)
    features["ultimate_osc"] = ((4 * avg7 + 2 * avg14 + avg28) / 7.0).clip(0, 1)

    # 19: technical_composite — average of key indicators
    features["technical_composite"] = features[
        ["RSI_14", "stoch_K", "MFI_14", "williams_R"]
    ].mean(axis=1)

    return features


# ---------------------------------------------------------------------------
# Feature matrix builder (48 features matching ensemble layout)
# ---------------------------------------------------------------------------


def build_ensemble_features(
    ticker_data: dict[str, pd.DataFrame],
    vix_data: pd.Series,
) -> pd.DataFrame:
    """Build training feature matrix with 48 features per sample.

    For bootstrap training, only technical features (0-19) and partial
    VRP/regime features are available from price history. Signal-dependent
    features (sentiment, flow, insider, GEX, cross-asset) are filled with
    neutral defaults. The model will learn primarily from technical patterns
    and be progressively refined via weekly retraining on live signal data.
    """
    all_rows: list[dict] = []

    for ticker, df in ticker_data.items():
        if df.empty or len(df) < 250:
            print(f"  Skipping {ticker}: insufficient data ({len(df)} bars)")
            continue

        tech = build_technical_features(df)

        # Align VIX to ticker dates
        vix_aligned = vix_data.reindex(df.index, method="ffill")

        # Compute IV proxy from HV (realized vol as IV estimate)
        log_ret = np.log(df["close"] / df["close"].shift(1))
        hv20 = log_ret.rolling(20).std() * np.sqrt(252) * 100
        hv_52w_high = hv20.rolling(252).max()
        hv_52w_low = hv20.rolling(252).min()
        iv_rank = (hv20 - hv_52w_low) / (hv_52w_high - hv_52w_low).replace(0, np.nan)
        iv_rank = iv_rank.clip(0, 1)

        # Forward return for target
        fwd_ret = df["close"].shift(-FORWARD_HORIZON_DAYS) / df["close"] - 1.0
        target = (fwd_ret > FORWARD_THRESHOLD).astype(float)
        target.iloc[-FORWARD_HORIZON_DAYS:] = np.nan

        for i in range(len(df)):
            idx = df.index[i]
            if pd.isna(target.iloc[i]):
                continue

            # Skip rows with insufficient lookback
            if i < 252:
                continue

            # 48-feature vector
            row: dict[str, float] = {}

            # Features 0-19: Technical
            for j, col in enumerate(tech.columns):
                val = tech[col].iloc[i]
                row[f"f{j:02d}"] = float(val) if pd.notna(val) else 0.0

            # Features 20-22: Sentiment (unavailable — neutral)
            row["f20"] = 0.0  # sentiment_score
            row["f21"] = 0.0  # num_articles (normalized)
            row["f22"] = 0.0  # avg_confidence

            # Features 23-26: Options Flow (unavailable — neutral)
            row["f23"] = 0.0  # flow_score
            row["f24"] = 0.0  # net_premium
            row["f25"] = 0.0  # consistency
            row["f26"] = 0.0  # num_unusual

            # Features 27-31: Regime (derived from VIX level)
            vix_val = (
                float(vix_aligned.iloc[i]) if pd.notna(vix_aligned.iloc[i]) else 20.0
            )
            adx_val = (
                float(tech["ADX"].iloc[i]) * 100.0
                if pd.notna(tech["ADX"].iloc[i])
                else 20.0
            )

            # One-hot encode regime from VIX + ADX rules
            is_low_vol = 1.0 if vix_val < 18 and adx_val > 25 else 0.0
            is_high_vol = 1.0 if 18 <= vix_val < 35 and adx_val > 25 else 0.0
            is_range = 1.0 if 18 <= vix_val < 35 and adx_val < 20 else 0.0
            is_crisis = 1.0 if vix_val >= 35 else 0.0

            # Handle case where no regime matches (default to range_bound)
            if is_low_vol + is_high_vol + is_range + is_crisis == 0:
                is_range = 1.0

            row["f27"] = is_low_vol  # low_vol_trend
            row["f28"] = is_high_vol  # high_vol_trend
            row["f29"] = is_range  # range_bound
            row["f30"] = is_crisis  # crisis
            row["f31"] = 0.7  # regime_confidence (moderate default)

            # Features 32-34: GEX (unavailable — neutral)
            row["f32"] = 0.0  # gex_score
            row["f33"] = 0.0  # net_gex
            row["f34"] = 0.5  # gex_regime_binary (neutral)

            # Features 35-38: Insider (unavailable — neutral)
            row["f35"] = 0.0  # insider_score
            row["f36"] = 0.0  # num_buys
            row["f37"] = 0.0  # num_sells
            row["f38"] = 0.0  # net_value

            # Features 39-43: VRP (partially available from HV proxy)
            row["f39"] = float(iv_rank.iloc[i]) if pd.notna(iv_rank.iloc[i]) else 0.5
            row["f40"] = float(iv_rank.iloc[i]) if pd.notna(iv_rank.iloc[i]) else 0.5
            row["f41"] = 0.0  # vrp_spread (unavailable)
            hv_iv = float(hv20.iloc[i] / 100.0) if pd.notna(hv20.iloc[i]) else 0.5
            row["f42"] = min(hv_iv / 3.0, 1.0)
            row["f43"] = 0.0  # vrp_score

            # Features 44-47: Cross-Asset (partially available)
            row["f44"] = 0.0  # cross_asset_score
            row["f45"] = 0.0  # yield_curve_score
            row["f46"] = 0.0  # credit_score
            # VIX term structure proxy: normalize VIX level
            vix_norm = float(np.clip((vix_val - 20.0) / 30.0, -1.0, 1.0))
            row["f47"] = vix_norm

            # Metadata (not features)
            row["ticker"] = ticker
            row["date"] = idx
            row["target"] = float(target.iloc[i])

            all_rows.append(row)

    result = pd.DataFrame(all_rows)
    print(f"  Built {len(result)} training samples across {len(ticker_data)} tickers")
    return result


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_walk_forward(
    X: pd.DataFrame,  # noqa: N803
    y: pd.Series,
    n_folds: int = N_FOLDS,
    embargo: int = EMBARGO_DAYS,
) -> XGBClassifier:
    """Walk-forward training with purged k-fold cross-validation."""
    n = len(X)
    fold_size = n // (n_folds + 1)

    best_model: XGBClassifier | None = None
    best_auc: float = -1.0
    fold_results: list[dict] = []

    for fold in range(n_folds):
        # Train on earlier data, test on later
        train_end = fold_size * (fold + 1)
        test_start = train_end + embargo
        test_end = min(test_start + fold_size, n)

        if test_start >= n or test_end - test_start < 20:
            break

        X_train = X.iloc[:train_end]  # noqa: N806
        y_train = y.iloc[:train_end]
        X_test = X.iloc[test_start:test_end]  # noqa: N806
        y_test = y.iloc[test_start:test_end]

        model = XGBClassifier(
            max_depth=6,
            n_estimators=500,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            use_label_encoder=False,
            eval_metric="logloss",
            early_stopping_rounds=50,
            verbosity=0,
        )

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )

        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)

        from sklearn.metrics import accuracy_score, roc_auc_score

        try:
            auc = roc_auc_score(y_test, y_pred_proba)
        except ValueError:
            auc = 0.5

        acc = accuracy_score(y_test, y_pred)

        fold_results.append(
            {
                "fold": fold,
                "train_size": len(X_train),
                "test_size": len(X_test),
                "accuracy": round(acc, 4),
                "auc_roc": round(auc, 4),
            }
        )

        print(
            f"    Fold {fold}: train={len(X_train)}, test={len(X_test)}, "
            f"accuracy={acc:.4f}, AUC={auc:.4f}"
        )

        if auc > best_auc:
            best_auc = auc
            best_model = model

    print(f"\n  Best AUC: {best_auc:.4f}")

    if best_model is None:
        raise RuntimeError("No folds produced a valid model")

    return best_model


def calibrate_model(
    model: XGBClassifier,
    X_cal: pd.DataFrame,  # noqa: N803
    y_cal: pd.Series,
) -> object | None:
    """Fit isotonic calibration on held-out data."""
    try:
        from sklearn.isotonic import IsotonicRegression

        raw_probs = model.predict_proba(X_cal)[:, 1]
        calibrator = IsotonicRegression(
            y_min=0.01,
            y_max=0.99,
            out_of_bounds="clip",
        )
        calibrator.fit(raw_probs, y_cal)
        print(f"  Calibrator fitted on {len(X_cal)} samples")
        return calibrator
    except Exception as e:
        print(f"  WARNING: Calibration failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> None:
    """Bootstrap training pipeline."""
    start_time = time.monotonic()

    if not POLYGON_API_KEY:
        print("ERROR: POLYGON_API_KEY not set")
        sys.exit(1)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Date range: 4 years back from today
    end_date = date.today()
    start_date = end_date - timedelta(days=LOOKBACK_YEARS * 365)

    print("=== Project Titan: Bootstrap ML Training ===")
    print(f"Date range: {start_date} to {end_date}")
    print(f"Tickers: {len(TICKERS)}")
    print()

    # ------------------------------------------------------------------
    # Step 1: Fetch historical data
    # ------------------------------------------------------------------
    print("[1/5] Fetching historical data from Polygon.io...")

    ticker_data: dict[str, pd.DataFrame] = {}

    async with httpx.AsyncClient() as client:
        # Fetch VIX first
        print("  Fetching VIX data...")
        vix_data = await fetch_vix_data(client, str(start_date), str(end_date))

        # Fetch each ticker
        for ticker in TICKERS:
            print(f"  Fetching {ticker}...")
            try:
                df = await fetch_polygon_bars(
                    client, ticker, str(start_date), str(end_date)
                )
                if not df.empty:
                    ticker_data[ticker] = df
                    print(f"    {ticker}: {len(df)} bars")
            except httpx.HTTPStatusError as e:
                print(f"    ERROR: {ticker}: HTTP {e.response.status_code}")
            except Exception as e:
                print(f"    ERROR: {ticker}: {e}")

            await asyncio.sleep(POLYGON_RATE_DELAY)

    if not ticker_data:
        print("ERROR: No historical data retrieved")
        sys.exit(1)

    if vix_data.empty:
        # Synthesize VIX from SPY volatility
        spy_df = ticker_data.get("SPY")
        if spy_df is not None:
            log_ret = np.log(spy_df["close"] / spy_df["close"].shift(1))
            vix_data = log_ret.rolling(20).std() * np.sqrt(252) * 100
            vix_data = vix_data.clip(9, 80)
        else:
            print("ERROR: Neither VIX nor SPY data available")
            sys.exit(1)

    print(f"\n  Loaded {len(ticker_data)} tickers, VIX: {len(vix_data)} bars")

    # ------------------------------------------------------------------
    # Step 2: Build feature matrix
    # ------------------------------------------------------------------
    print("\n[2/5] Building feature matrix...")
    train_df = build_ensemble_features(ticker_data, vix_data)

    if len(train_df) < 500:
        print(f"ERROR: Only {len(train_df)} samples — need at least 500")
        sys.exit(1)

    # Sort by date for walk-forward
    train_df = train_df.sort_values("date").reset_index(drop=True)

    feature_cols = [f"f{i:02d}" for i in range(TOTAL_FEATURES)]
    X = train_df[feature_cols].astype(float)  # noqa: N806
    y = train_df["target"].astype(int)

    # Fill any remaining NaN with 0
    X = X.fillna(0.0)  # noqa: N806

    print(f"  Feature matrix: {X.shape}")
    print(f"  Target balance: {y.mean():.3f} positive rate")

    # ------------------------------------------------------------------
    # Step 3: Train walk-forward model
    # ------------------------------------------------------------------
    print("\n[3/5] Training walk-forward XGBoost model...")
    model = train_walk_forward(X, y)

    # ------------------------------------------------------------------
    # Step 4: Calibrate
    # ------------------------------------------------------------------
    print("\n[4/5] Calibrating probabilities...")
    # Use last 20% of data for calibration
    cal_start = int(len(X) * 0.8)
    X_cal = X.iloc[cal_start:]  # noqa: N806
    y_cal = y.iloc[cal_start:]
    calibrator = calibrate_model(model, X_cal, y_cal)

    # ------------------------------------------------------------------
    # Step 5: Save model
    # ------------------------------------------------------------------
    print("\n[5/5] Saving model...")

    # Save XGBoost model
    model.save_model(str(MODEL_OUTPUT))
    print(f"  Model saved: {MODEL_OUTPUT}")

    # Save calibrator
    if calibrator is not None:
        import pickle

        with open(CALIBRATOR_OUTPUT, "wb") as f:
            pickle.dump(calibrator, f)
        print(f"  Calibrator saved: {CALIBRATOR_OUTPUT}")

    # Save metadata
    meta = {
        "model_name": "titan_xgboost_ensemble",
        "version": 1,
        "trained_at": datetime.utcnow().isoformat(),
        "train_start": str(start_date),
        "train_end": str(end_date),
        "n_features": TOTAL_FEATURES,
        "feature_names": feature_cols,
        "n_samples": len(X),
        "target_positive_rate": round(float(y.mean()), 4),
        "training_type": "bootstrap",
        "note": "Bootstrap model trained on historical technical features. "
        "Sentiment, flow, insider, GEX features are neutral defaults. "
        "Model will improve with weekly retraining on live signal data.",
    }
    meta_path = MODELS_DIR / "ensemble_xgb_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)
    print(f"  Metadata saved: {meta_path}")

    elapsed = time.monotonic() - start_time
    print(f"\n=== Training complete in {elapsed:.1f}s ===")
    print(f"Model ready at: {MODEL_OUTPUT}")


if __name__ == "__main__":
    asyncio.run(main())
