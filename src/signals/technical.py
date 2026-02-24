"""Technical indicator feature engineering for Project Titan.

Calculates 120+ technical features from OHLCV data using pandas-ta and
custom implementations.  The output DataFrame is consumed by the ML
ensemble meta-learner and the feature engineering pipeline.

Feature categories:

- **Trend**: Moving averages, MACD, ADX, Aroon, Supertrend, linear regression
- **Momentum**: RSI, Stochastic, Williams %R, CCI, ROC, MFI, CMO, returns
- **Volatility**: ATR, Bollinger, Keltner, Donchian, HV, Garman-Klass, Parkinson
- **Volume**: OBV, VWAP proxy, volume ratios, A/D, CMF, Force Index
- **Pattern**: Candlestick patterns, inside/outside bars, HH/LL counts
- **Derived**: Z-scores, mean reversion, Hurst exponent, calendar features

Usage::

    from src.signals.technical import TechnicalSignalGenerator

    generator = TechnicalSignalGenerator(lookback_period=252)
    features_df = generator.calculate_features(ohlcv_df)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pandas_ta as ta

from src.utils.logging import get_logger

if TYPE_CHECKING:
    import structlog


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TRADING_DAYS_PER_YEAR: int = 252
"""Number of trading days per year, used for annualization."""

HURST_MAX_LAG: int = 20
"""Maximum lag used in the simplified Hurst exponent calculation."""


# ---------------------------------------------------------------------------
# TechnicalSignalGenerator
# ---------------------------------------------------------------------------


class TechnicalSignalGenerator:
    """Calculates all technical features used by the ML ensemble.

    Processes an OHLCV DataFrame through trend, momentum, volatility,
    volume, pattern, and derived feature pipelines to produce a
    feature-rich DataFrame for downstream model consumption.

    Args:
        lookback_period: Number of trading days to use for percentile
            and ranking calculations.  Defaults to 252 (one year).
    """

    def __init__(self, lookback_period: int = 252) -> None:
        self._lookback_period: int = lookback_period
        self._log: structlog.stdlib.BoundLogger = get_logger("signals.technical")
        self._log.info(
            "technical_signal_generator_initialized",
            lookback_period=lookback_period,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical features from an OHLCV DataFrame.

        Calls each sub-method to compute trend, momentum, volatility,
        volume, pattern, and derived features, then merges all results
        into a single DataFrame.  Rows with NaN values introduced by
        indicator lookback periods are dropped from the output.

        Args:
            df: OHLCV DataFrame with columns ``open``, ``high``, ``low``,
                ``close``, ``volume`` indexed by date.  Must contain at
                least 200 rows for all indicators to compute properly.

        Returns:
            DataFrame with the original OHLCV columns plus 120+ feature
            columns.  NaN rows from lookback periods are removed.

        Raises:
            ValueError: If required OHLCV columns are missing.
        """
        required_columns = {"open", "high", "low", "close", "volume"}
        missing = required_columns - set(df.columns)
        if missing:
            raise ValueError(f"Missing required OHLCV columns: {sorted(missing)}")

        if len(df) < 200:
            self._log.warning(
                "insufficient_data_for_features",
                rows=len(df),
                recommended_minimum=200,
            )

        # Work on a copy to avoid mutating the caller's DataFrame
        result = df.copy()

        self._log.debug(
            "calculating_features",
            input_rows=len(result),
        )

        # Calculate each feature category and merge
        trend = self._trend_features(result)
        momentum = self._momentum_features(result)
        volatility = self._volatility_features(result)
        volume = self._volume_features(result)
        pattern = self._pattern_features(result)
        derived = self._derived_features(result)

        for feature_df in (trend, momentum, volatility, volume, pattern, derived):
            for col in feature_df.columns:
                if col not in result.columns:
                    result[col] = feature_df[col]

        # Drop NaN rows produced by lookback periods
        initial_rows = len(result)
        result = result.dropna()
        dropped_rows = initial_rows - len(result)

        self._log.info(
            "features_calculated",
            total_features=len(result.columns) - len(required_columns),
            output_rows=len(result),
            dropped_nan_rows=dropped_rows,
        )

        return result

    # ------------------------------------------------------------------
    # Trend features
    # ------------------------------------------------------------------

    def _trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate trend-following indicator features.

        Computes simple and exponential moving averages, price-relative-to-MA
        ratios, MACD, ADX, Aroon, Supertrend, linear regression slope, and
        trend strength.

        Args:
            df: OHLCV DataFrame.

        Returns:
            DataFrame with trend feature columns.
        """
        out = pd.DataFrame(index=df.index)
        close = df["close"]
        high = df["high"]
        low = df["low"]

        # Simple Moving Averages
        for period in (10, 20, 50, 100, 200):
            col_name = f"sma_{period}"
            out[col_name] = ta.sma(close, length=period)
            out[f"close_to_sma_{period}"] = close / out[col_name]

        # Exponential Moving Averages
        for period in (10, 20, 50):
            col_name = f"ema_{period}"
            out[col_name] = ta.ema(close, length=period)
            out[f"close_to_ema_{period}"] = close / out[col_name]

        # MACD (12, 26, 9)
        macd_result = ta.macd(close, fast=12, slow=26, signal=9)
        if macd_result is not None and not macd_result.empty:
            out["macd"] = macd_result.iloc[:, 0]
            out["macd_histogram"] = macd_result.iloc[:, 1]
            out["macd_signal"] = macd_result.iloc[:, 2]
        else:
            out["macd"] = np.nan
            out["macd_histogram"] = np.nan
            out["macd_signal"] = np.nan

        # ADX (14)
        adx_result = ta.adx(high, low, close, length=14)
        if adx_result is not None and not adx_result.empty:
            out["adx_14"] = adx_result.iloc[:, 0]
            out["di_plus_14"] = adx_result.iloc[:, 1]
            out["di_minus_14"] = adx_result.iloc[:, 2]
        else:
            out["adx_14"] = np.nan
            out["di_plus_14"] = np.nan
            out["di_minus_14"] = np.nan

        # Aroon (14)
        aroon_result = ta.aroon(high, low, length=14)
        if aroon_result is not None and not aroon_result.empty:
            out["aroon_down_14"] = aroon_result.iloc[:, 0]
            out["aroon_up_14"] = aroon_result.iloc[:, 1]
            out["aroon_osc_14"] = aroon_result.iloc[:, 2]
        else:
            out["aroon_down_14"] = np.nan
            out["aroon_up_14"] = np.nan
            out["aroon_osc_14"] = np.nan

        # Supertrend (7, 3.0)
        supertrend_result = ta.supertrend(high, low, close, length=7, multiplier=3.0)
        if supertrend_result is not None and not supertrend_result.empty:
            # pandas-ta returns: SUPERT_7_3.0, SUPERTd_7_3.0, etc.
            out["supertrend"] = supertrend_result.iloc[:, 0]
            out["supertrend_direction"] = supertrend_result.iloc[:, 1]
        else:
            out["supertrend"] = np.nan
            out["supertrend_direction"] = np.nan

        # Linear regression slope (20-day)
        out["linreg_slope_20"] = ta.slope(close, length=20)

        # Trend strength: (close - SMA_50) / ATR_14
        atr_14 = ta.atr(high, low, close, length=14)
        sma_50 = out.get("sma_50", ta.sma(close, length=50))
        if atr_14 is not None:
            safe_atr = atr_14.replace(0, np.nan)
            out["trend_strength"] = (close - sma_50) / safe_atr
        else:
            out["trend_strength"] = np.nan

        self._log.debug(
            "trend_features_calculated",
            feature_count=len(out.columns),
        )

        return out

    # ------------------------------------------------------------------
    # Momentum features
    # ------------------------------------------------------------------

    def _momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate momentum oscillator features.

        Computes RSI at multiple timeframes, Stochastic, Williams %R, CCI,
        Rate of Change, Money Flow Index, Chande Momentum Oscillator,
        Ultimate Oscillator, and multi-period returns.

        Args:
            df: OHLCV DataFrame.

        Returns:
            DataFrame with momentum feature columns.
        """
        out = pd.DataFrame(index=df.index)
        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"]

        # RSI at multiple timeframes
        for period in (7, 14, 21):
            out[f"rsi_{period}"] = ta.rsi(close, length=period)

        # Stochastic %K/%D (14, 3, 3)
        stoch_result = ta.stoch(high, low, close, k=14, d=3, smooth_k=3)
        if stoch_result is not None and not stoch_result.empty:
            out["stoch_k"] = stoch_result.iloc[:, 0]
            out["stoch_d"] = stoch_result.iloc[:, 1]
        else:
            out["stoch_k"] = np.nan
            out["stoch_d"] = np.nan

        # Williams %R (14)
        out["willr_14"] = ta.willr(high, low, close, length=14)

        # CCI (20)
        out["cci_20"] = ta.cci(high, low, close, length=20)

        # Rate of Change at multiple periods
        for period in (5, 10, 20):
            out[f"roc_{period}"] = ta.roc(close, length=period)

        # Money Flow Index (14)
        out["mfi_14"] = ta.mfi(high, low, close, volume, length=14)

        # Chande Momentum Oscillator (14)
        out["cmo_14"] = ta.cmo(close, length=14)

        # Ultimate Oscillator
        uo_result = ta.uo(high, low, close)
        if uo_result is not None:
            out["ultimate_oscillator"] = uo_result
        else:
            out["ultimate_oscillator"] = np.nan

        # Returns at multiple horizons
        for period in (1, 5, 10, 20, 60):
            out[f"return_{period}d"] = close.pct_change(periods=period)

        self._log.debug(
            "momentum_features_calculated",
            feature_count=len(out.columns),
        )

        return out

    # ------------------------------------------------------------------
    # Volatility features
    # ------------------------------------------------------------------

    def _volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate volatility indicator features.

        Computes ATR at multiple timeframes, ATR percentile rank, Bollinger
        Bands, Keltner Channels, Donchian Channels, historical volatility
        at multiple windows, HV ratio, Garman-Klass volatility, and
        Parkinson volatility.

        Args:
            df: OHLCV DataFrame.

        Returns:
            DataFrame with volatility feature columns.
        """
        out = pd.DataFrame(index=df.index)
        close = df["close"]
        high = df["high"]
        low = df["low"]

        # ATR at multiple timeframes
        for period in (7, 14, 21):
            out[f"atr_{period}"] = ta.atr(high, low, close, length=period)

        # ATR percentile: rank of 14-day ATR over the lookback period
        atr_14 = out["atr_14"]
        out["atr_percentile"] = atr_14.rolling(
            window=self._lookback_period, min_periods=20
        ).apply(
            lambda x: _percentile_rank(x),
            raw=True,
        )

        # Bollinger Bands (20, 2)
        bbands = ta.bbands(close, length=20, std=2.0)
        if bbands is not None and not bbands.empty:
            out["bb_lower"] = bbands.iloc[:, 0]
            out["bb_mid"] = bbands.iloc[:, 1]
            out["bb_upper"] = bbands.iloc[:, 2]
            out["bb_bandwidth"] = bbands.iloc[:, 3]
            out["bb_pctb"] = bbands.iloc[:, 4]
        else:
            for col in ("bb_lower", "bb_mid", "bb_upper", "bb_bandwidth", "bb_pctb"):
                out[col] = np.nan

        # Keltner Channels (20, 1.5)
        kc = ta.kc(high, low, close, length=20, scalar=1.5)
        if kc is not None and not kc.empty:
            out["kc_lower"] = kc.iloc[:, 0]
            out["kc_mid"] = kc.iloc[:, 1]
            out["kc_upper"] = kc.iloc[:, 2]
        else:
            for col in ("kc_lower", "kc_mid", "kc_upper"):
                out[col] = np.nan

        # Donchian Channels (20)
        donchian = ta.donchian(high, low, lower_length=20, upper_length=20)
        if donchian is not None and not donchian.empty:
            out["dc_lower"] = donchian.iloc[:, 0]
            out["dc_mid"] = donchian.iloc[:, 1]
            out["dc_upper"] = donchian.iloc[:, 2]
        else:
            for col in ("dc_lower", "dc_mid", "dc_upper"):
                out[col] = np.nan
        # Donchian width
        if "dc_upper" in out.columns and "dc_lower" in out.columns:
            safe_mid = out.get("dc_mid", (out["dc_upper"] + out["dc_lower"]) / 2.0)
            safe_mid = safe_mid.replace(0, np.nan)
            out["dc_width"] = (out["dc_upper"] - out["dc_lower"]) / safe_mid
        else:
            out["dc_width"] = np.nan

        # Historical volatility at multiple windows (annualized std of log returns)
        log_returns = np.log(close / close.shift(1))
        for window in (10, 20, 30, 60, 90):
            out[f"hv_{window}"] = log_returns.rolling(window=window).std() * np.sqrt(
                TRADING_DAYS_PER_YEAR
            )

        # HV ratio: short-term vs long-term volatility
        out["hv_ratio_10_60"] = out["hv_10"] / out["hv_60"].replace(0, np.nan)

        # Garman-Klass volatility (20-day)
        out["gk_vol_20"] = self._garman_klass_volatility(df, window=20)

        # Parkinson volatility (20-day)
        out["parkinson_vol_20"] = self._parkinson_volatility(df, window=20)

        self._log.debug(
            "volatility_features_calculated",
            feature_count=len(out.columns),
        )

        return out

    # ------------------------------------------------------------------
    # Volume features
    # ------------------------------------------------------------------

    def _volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate volume-based indicator features.

        Computes On-Balance Volume, volume-weighted close proxy, volume
        moving averages, volume ratio, Accumulation/Distribution line,
        Chaikin Money Flow, and Force Index.

        Args:
            df: OHLCV DataFrame.

        Returns:
            DataFrame with volume feature columns.
        """
        out = pd.DataFrame(index=df.index)
        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"]

        # On-Balance Volume
        obv = ta.obv(close, volume)
        if obv is not None:
            out["obv"] = obv
        else:
            out["obv"] = np.nan

        # VWAP proxy for daily data: cumulative (volume * close) / cumulative volume
        # For daily bars, compute a rolling volume-weighted price
        cum_vol_price = (volume * close).rolling(window=20, min_periods=1).sum()
        cum_vol = volume.rolling(window=20, min_periods=1).sum()
        out["vwap_proxy_20"] = cum_vol_price / cum_vol.replace(0, np.nan)

        # Volume SMAs
        for period in (10, 20, 50):
            out[f"vol_sma_{period}"] = ta.sma(volume, length=period)

        # Volume ratio: current volume / SMA_20
        vol_sma_20 = out["vol_sma_20"]
        out["vol_ratio_20"] = volume / vol_sma_20.replace(0, np.nan)

        # Accumulation/Distribution line
        ad = ta.ad(high, low, close, volume)
        if ad is not None:
            out["ad_line"] = ad
        else:
            out["ad_line"] = np.nan

        # Chaikin Money Flow (20)
        cmf = ta.cmf(high, low, close, volume, length=20)
        if cmf is not None:
            out["cmf_20"] = cmf
        else:
            out["cmf_20"] = np.nan

        # Force Index (13-period EMA of volume * price change)
        raw_force = volume * close.diff()
        out["force_index_13"] = ta.ema(raw_force, length=13)

        self._log.debug(
            "volume_features_calculated",
            feature_count=len(out.columns),
        )

        return out

    # ------------------------------------------------------------------
    # Pattern features
    # ------------------------------------------------------------------

    def _pattern_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate candlestick and bar-pattern features.

        Detects common candlestick patterns via pandas-ta (doji, hammer,
        engulfing, morning star, evening star) as well as inside-bar and
        outside-bar patterns and higher-high / lower-low streak counts.

        Args:
            df: OHLCV DataFrame.

        Returns:
            DataFrame with binary pattern feature columns.
        """
        out = pd.DataFrame(index=df.index)
        open_ = df["open"]
        high = df["high"]
        low = df["low"]
        close = df["close"]

        # Candlestick patterns via pandas-ta cdl_pattern
        # pandas-ta provides a unified candlestick interface
        for pattern_name in (
            "doji",
            "hammer",
            "engulfing",
            "morningstar",
            "eveningstar",
        ):
            cdl_result = ta.cdl_pattern(open_, high, low, close, name=pattern_name)
            if cdl_result is not None and not cdl_result.empty:
                out[f"cdl_{pattern_name}"] = cdl_result.iloc[:, 0]
            else:
                out[f"cdl_{pattern_name}"] = 0

        # Inside bar: high < previous high AND low > previous low
        prev_high = high.shift(1)
        prev_low = low.shift(1)
        out["inside_bar"] = ((high < prev_high) & (low > prev_low)).astype(int)

        # Outside bar: high > previous high AND low < previous low
        out["outside_bar"] = ((high > prev_high) & (low < prev_low)).astype(int)

        # Higher highs / lower lows counts over a 5-day window
        out["higher_highs_5"] = _count_consecutive_higher(high, window=5)
        out["lower_lows_5"] = _count_consecutive_lower(low, window=5)

        self._log.debug(
            "pattern_features_calculated",
            feature_count=len(out.columns),
        )

        return out

    # ------------------------------------------------------------------
    # Derived features
    # ------------------------------------------------------------------

    def _derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate derived and calendar features.

        Computes price distance from 52-week high/low, Z-score of close
        relative to its 20-day SMA, mean-reversion signal, a simplified
        Hurst exponent, and calendar encodings (day of week, month).

        Args:
            df: OHLCV DataFrame.

        Returns:
            DataFrame with derived feature columns.
        """
        out = pd.DataFrame(index=df.index)
        close = df["close"]
        high = df["high"]
        low = df["low"]

        # 52-week (252-day) high and low distances
        rolling_high_252 = high.rolling(
            window=self._lookback_period,
            min_periods=20,
        ).max()
        rolling_low_252 = low.rolling(
            window=self._lookback_period,
            min_periods=20,
        ).min()

        out["dist_from_52w_high"] = (
            close - rolling_high_252
        ) / rolling_high_252.replace(0, np.nan)
        out["dist_from_52w_low"] = (close - rolling_low_252) / rolling_low_252.replace(
            0, np.nan
        )

        # Z-score of close relative to 20-day SMA
        sma_20 = ta.sma(close, length=20)
        std_20 = close.rolling(window=20).std()
        safe_std = std_20.replace(0, np.nan)
        out["zscore_close_sma20"] = (close - sma_20) / safe_std

        # Mean reversion signal: (close - SMA_20) / std_20
        out["mean_reversion_20"] = (close - sma_20) / safe_std

        # Hurst exponent (simplified, over 100-day rolling window)
        out["hurst_exponent_100"] = self._hurst_exponent(close, window=100)

        # Calendar features
        if isinstance(df.index, pd.DatetimeIndex):
            out["day_of_week"] = df.index.dayofweek
            out["month"] = df.index.month
        else:
            # Attempt to convert index to datetime
            try:
                dt_index = pd.to_datetime(df.index)
                out["day_of_week"] = dt_index.dayofweek
                out["month"] = dt_index.month
            except (ValueError, TypeError):
                self._log.warning("cannot_extract_calendar_features")
                out["day_of_week"] = 0
                out["month"] = 0

        self._log.debug(
            "derived_features_calculated",
            feature_count=len(out.columns),
        )

        return out

    # ------------------------------------------------------------------
    # Volatility estimator helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _garman_klass_volatility(df: pd.DataFrame, window: int = 20) -> pd.Series:
        """Calculate Garman-Klass volatility estimator.

        The Garman-Klass estimator uses open, high, low, and close prices
        to provide a more efficient volatility estimate than the
        close-to-close estimator.

        Formula per bar::

            GK = 0.5 * ln(H/L)^2 - (2*ln(2) - 1) * ln(C/O)^2

        The estimator is annualized by multiplying by sqrt(252).

        Args:
            df: OHLCV DataFrame with ``open``, ``high``, ``low``, ``close``.
            window: Rolling window size.

        Returns:
            Annualized Garman-Klass volatility series.
        """
        log_hl = np.log(df["high"] / df["low"])
        log_co = np.log(df["close"] / df["open"])

        gk_var = 0.5 * log_hl**2 - (2.0 * np.log(2.0) - 1.0) * log_co**2

        return np.sqrt(gk_var.rolling(window=window).mean() * TRADING_DAYS_PER_YEAR)

    @staticmethod
    def _parkinson_volatility(df: pd.DataFrame, window: int = 20) -> pd.Series:
        """Calculate Parkinson volatility estimator.

        The Parkinson estimator uses only high and low prices and is
        approximately 5x more efficient than the close-to-close estimator.

        Formula::

            PV = sqrt( (1 / (4 * N * ln2)) * sum(ln(H/L)^2) )

        Annualized by multiplying by sqrt(252).

        Args:
            df: OHLCV DataFrame with ``high``, ``low``.
            window: Rolling window size.

        Returns:
            Annualized Parkinson volatility series.
        """
        log_hl_sq = np.log(df["high"] / df["low"]) ** 2

        factor = 1.0 / (4.0 * np.log(2.0))
        return np.sqrt(
            factor * log_hl_sq.rolling(window=window).mean() * TRADING_DAYS_PER_YEAR
        )

    @staticmethod
    def _hurst_exponent(series: pd.Series, window: int = 100) -> pd.Series:
        """Calculate a simplified Hurst exponent over a rolling window.

        Uses the rescaled range (R/S) method with a fixed set of lag
        sub-divisions.  Values interpretation:

        - H < 0.5: mean-reverting
        - H = 0.5: random walk
        - H > 0.5: trending

        This is a simplified implementation suitable for feature
        engineering; it is not a rigorous statistical estimator.

        Args:
            series: Price series (typically close prices).
            window: Rolling window size.

        Returns:
            Rolling Hurst exponent series.
        """

        def _rs_hurst(prices: np.ndarray) -> float:
            """Compute the Hurst exponent for a fixed price window."""
            if len(prices) < HURST_MAX_LAG:
                return np.nan

            returns = np.diff(np.log(prices))
            if len(returns) < 2:
                return np.nan

            lags = range(2, min(HURST_MAX_LAG + 1, len(returns)))
            rs_values: list[float] = []
            lag_list: list[int] = []

            for lag in lags:
                # Split returns into non-overlapping sub-series
                n_subseries = len(returns) // lag
                if n_subseries < 1:
                    continue

                rs_sum = 0.0
                valid_count = 0
                for i in range(n_subseries):
                    sub = returns[i * lag : (i + 1) * lag]
                    mean_sub = np.mean(sub)
                    cumdev = np.cumsum(sub - mean_sub)
                    r = np.max(cumdev) - np.min(cumdev)
                    s = np.std(sub, ddof=1)
                    if s > 0:
                        rs_sum += r / s
                        valid_count += 1

                if valid_count > 0:
                    rs_values.append(rs_sum / valid_count)
                    lag_list.append(lag)

            if len(rs_values) < 3:
                return np.nan

            log_lags = np.log(np.array(lag_list, dtype=float))
            log_rs = np.log(np.array(rs_values, dtype=float))

            # Linear fit: log(R/S) = H * log(n) + c
            coeffs = np.polyfit(log_lags, log_rs, 1)
            hurst = coeffs[0]

            # Clamp to [0, 1] for robustness
            return float(max(0.0, min(1.0, hurst)))

        return series.rolling(window=window, min_periods=window).apply(
            _rs_hurst, raw=True
        )


# ---------------------------------------------------------------------------
# Module-level helper functions
# ---------------------------------------------------------------------------


def _percentile_rank(values: np.ndarray) -> float:
    """Calculate the percentile rank of the last value in an array.

    Returns the percentage of values in the array that are less than or
    equal to the last value.

    Args:
        values: 1-D numpy array.  The last element is the value to rank.

    Returns:
        Percentile rank as a float between 0.0 and 100.0.
    """
    if len(values) < 2:
        return 50.0
    current = values[-1]
    total = len(values)
    count_below = np.sum(values[:-1] <= current)
    return float(count_below / (total - 1) * 100.0)


def _count_consecutive_higher(series: pd.Series, window: int = 5) -> pd.Series:
    """Count consecutive higher values over a rolling window.

    For each point, counts how many of the preceding values in the
    window are sequentially higher than their predecessor.

    Args:
        series: Input price series (typically highs).
        window: Number of periods to look back.

    Returns:
        Series with the count of consecutive higher values.
    """
    diff = series.diff()
    is_higher = (diff > 0).astype(int)
    return is_higher.rolling(window=window, min_periods=1).sum()


def _count_consecutive_lower(series: pd.Series, window: int = 5) -> pd.Series:
    """Count consecutive lower values over a rolling window.

    For each point, counts how many of the preceding values in the
    window are sequentially lower than their predecessor.

    Args:
        series: Input price series (typically lows).
        window: Number of periods to look back.

    Returns:
        Series with the count of consecutive lower values.
    """
    diff = series.diff()
    is_lower = (diff < 0).astype(int)
    return is_lower.rolling(window=window, min_periods=1).sum()
