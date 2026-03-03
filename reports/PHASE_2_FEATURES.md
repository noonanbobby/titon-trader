# Phase 2: Feature Engineering Report
**Generated:** 2026-03-03 22:23:27 UTC

## Feature Inventory
| Group | Count | Example Features |
|-------|-------|-----------------|
| Trend | 29 | adx_14, close_to_ema_10, close_to_ema_20, close_to_ema_50, ... |
| Momentum | 27 | aroon_down_14, aroon_osc_14, aroon_up_14, cci_20, ... |
| Volatility | 30 | atr_14, atr_21, atr_7, atr_percentile, ... |
| Volume | 13 | ad_line, cmf_20, force_index_13, gk_vol_20, ... |
| Cross-Asset | 15 | dxy_mom_20d, dxy_proxy, fed_funds_90d_chg, fed_funds_rate, ... |
| Calendar | 8 | day_of_week, days_since_earnings, days_to_earnings, days_to_quarter_end, ... |
| Ticker-Specific | 5 | avg_dollar_vol_20d, beta_spy_20d, corr_spy_20d, idio_vol_20d, ... |
| Pattern | 5 | cdl_doji, higher_highs_5, inside_bar, lower_lows_5, ... |
| Other | 9 | close, dist_from_52w_high, dist_from_52w_low, high, ... |
| **Total** | **141** | |

## Data Quality
| Ticker | Input Rows | Output Rows | Features | Target Mean |
|--------|-----------|------------|----------|-------------|
| AAPL | 1044 | 840 | 145 | 0.4536 |
| NVDA | 1044 | 840 | 145 | 0.5393 |
| MSFT | 1044 | 840 | 145 | 0.4274 |
| GOOGL | 1044 | 840 | 145 | 0.4821 |
| META | 1044 | 840 | 145 | 0.5012 |
| AMZN | 1044 | 840 | 145 | 0.4429 |
| TSLA | 1044 | 840 | 145 | 0.4595 |
| AMD | 1044 | 840 | 145 | 0.4714 |
| AVGO | 1044 | 840 | 145 | 0.4917 |
| CRM | 1044 | 840 | 145 | 0.4524 |

## Leakage Check Results

All features passed (no correlation > 0.5 with target).

## Feature Distribution Summary

**Top 10 most variable features:**
- `obv`: std=6356124376.2701
- `ad_line`: std=5893813530.4200
- `force_index_13`: std=197573012.7357
- `volume`: std=114750408.6020
- `vol_sma_50`: std=110814908.3587
- `vol_sma_10`: std=110207802.2086
- `vol_sma_20`: std=109719701.1886
- `avg_dollar_vol_20d`: std=9928.6915
- `cci_20`: std=1629.4181
- `dc_upper`: std=149.0342

**Bottom 10 least variable features:**
- `return_5d`: std=0.058248
- `close_to_ema_20`: std=0.054035
- `close_to_sma_10`: std=0.043269
- `hurst_exponent_100`: std=0.039185
- `close_to_ema_10`: std=0.036396
- `spy_return_20d`: std=0.035024
- `xlk_spy_ratio_chg_20d`: std=0.028703
- `return_1d`: std=0.026379
- `spy_qqq_ratio_chg_10d`: std=0.015285
- `dxy_mom_20d`: std=0.013699

## Top Feature-Target Correlations

| Feature | Abs Correlation |
|---------|----------------|
| hy_oas | 0.1136 |
| vix_level | 0.1039 |
| fed_funds_90d_chg | 0.0820 |
| close | 0.0778 |
| iv_rank_proxy | 0.0777 |
| vwap | 0.0772 |
| low | 0.0771 |
| high | 0.0768 |
| open | 0.0767 |
| is_january | 0.0764 |
| sma_10 | 0.0759 |
| ema_10 | 0.0758 |
| spy_return_20d | 0.0758 |
| kc_lower | 0.0748 |
| ema_20 | 0.0747 |
| kc_mid | 0.0747 |
| kc_upper | 0.0745 |
| dc_mid | 0.0741 |
| vwap_proxy_20 | 0.0740 |
| dc_upper | 0.0740 |

## Validation Results

**All checks PASSED.**

## Output Location
- `data/features/all_features.parquet` (8400 rows × 141 features)

