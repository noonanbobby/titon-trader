# Phase 3: Walk-Forward Training Report
**Generated:** 2026-03-03 23:10:20 UTC

## Training Configuration
- Train window: 12 months
- Test window: 1 month
- Purged CV: 5-fold, 5-day embargo
- Models: XGBoost, LightGBM, CatBoost
- Features: 93 (after pruning from 141 original)

## Walk-Forward Results
| Window | Train Period | Test Period | CV AUC (mean±std) | OOS AUC | OOS Accuracy |
|--------|-------------|-------------|-------------------|---------|-------------|
| 1 | 2022-10 to 2023-09 | 2023-10 | 0.7567±0.0304 | 0.6955 | 0.5818 |
| 2 | 2022-11 to 2023-10 | 2023-11 | 0.7632±0.0368 | 0.6493 | 0.4429 |
| 3 | 2022-12 to 2023-11 | 2023-12 | 0.7757±0.0336 | 0.4665 | 0.4750 |
| 4 | 2023-01 to 2023-12 | 2024-01 | 0.7653±0.0432 | 0.4545 | 0.4095 |
| 5 | 2023-02 to 2024-01 | 2024-02 | 0.7395±0.0782 | 0.5893 | 0.5550 |
| 6 | 2023-03 to 2024-02 | 2024-03 | 0.7196±0.0825 | 0.4531 | 0.4200 |
| 7 | 2023-04 to 2024-03 | 2024-04 | 0.7104±0.0767 | 0.5995 | 0.4545 |
| 8 | 2023-05 to 2024-04 | 2024-05 | 0.7324±0.0645 | 0.6391 | 0.5545 |
| 9 | 2023-06 to 2024-05 | 2024-06 | 0.7102±0.0728 | 0.6250 | 0.4895 |
| 10 | 2023-07 to 2024-06 | 2024-07 | 0.7164±0.0762 | 0.4504 | 0.3136 |
| 11 | 2023-08 to 2024-07 | 2024-08 | 0.7303±0.0610 | 0.6090 | 0.5773 |
| 12 | 2023-09 to 2024-08 | 2024-09 | 0.7451±0.0621 | 0.6234 | 0.3550 |
| 13 | 2023-10 to 2024-09 | 2024-10 | 0.7338±0.0862 | 0.6563 | 0.6043 |
| 14 | 2023-11 to 2024-10 | 2024-11 | 0.7080±0.0702 | 0.5236 | 0.4900 |
| 15 | 2023-12 to 2024-11 | 2024-12 | 0.7004±0.0624 | 0.5299 | 0.4571 |
| 16 | 2024-01 to 2024-12 | 2025-01 | 0.6817±0.0462 | 0.5448 | 0.5750 |
| 17 | 2024-02 to 2025-01 | 2025-02 | 0.6859±0.0579 | 0.5201 | 0.5947 |
| 18 | 2024-03 to 2025-02 | 2025-03 | 0.6936±0.0773 | 0.6369 | 0.4952 |
| 19 | 2024-04 to 2025-03 | 2025-04 | 0.7211±0.0532 | 0.6590 | 0.6143 |
| 20 | 2024-05 to 2025-04 | 2025-05 | 0.7423±0.0723 | 0.6076 | 0.4714 |
| 21 | 2024-06 to 2025-05 | 2025-06 | 0.7513±0.0688 | 0.5047 | 0.4900 |
| 22 | 2024-07 to 2025-06 | 2025-07 | 0.7229±0.0875 | 0.4741 | 0.4773 |
| 23 | 2024-08 to 2025-07 | 2025-08 | 0.7000±0.0992 | 0.6180 | 0.6095 |
| 24 | 2024-09 to 2025-08 | 2025-09 | 0.6890±0.0727 | 0.4583 | 0.4667 |
| 25 | 2024-10 to 2025-09 | 2025-10 | 0.6590±0.0737 | 0.5496 | 0.5087 |
| 26 | 2024-11 to 2025-10 | 2025-11 | 0.6751±0.0697 | 0.7746 | 0.6421 |
| 27 | 2024-12 to 2025-11 | 2025-12 | 0.6944±0.0987 | 0.6680 | 0.7273 |
| 28 | 2025-01 to 2025-12 | 2026-01 | 0.6925±0.0876 | 0.5905 | 0.6050 |
| 29 | 2025-02 to 2026-01 | 2026-02 | 0.6900±0.0744 | 0.6195 | 0.5375 |

## Aggregate Metrics
| Metric | Value |
|--------|-------|
| Avg OOS AUC | 0.5790 |
| Min OOS AUC | 0.4504 |
| Max OOS AUC | 0.7746 |
| Std OOS AUC | 0.0823 |
| Avg CV AUC | 0.7174 |
| Features (post-pruning) | 93 |
| Total training samples | 72550 |
| Training time | 163.8s |

## Feature Importance (Top 20)
| Rank | Feature | Avg Importance |
|------|---------|---------------|
| 1 | spy_return_20d | 0.024133 |
| 2 | dxy_mom_20d | 0.023520 |
| 3 | hy_oas | 0.021861 |
| 4 | days_to_quarter_end | 0.020748 |
| 5 | month | 0.020482 |
| 6 | yield_2y | 0.018970 |
| 7 | vix_term_proxy | 0.018439 |
| 8 | dxy_proxy | 0.018388 |
| 9 | hy_oas_zscore | 0.017923 |
| 10 | spy_qqq_ratio_chg_10d | 0.017330 |
| 11 | yield_spread_2s10s | 0.017007 |
| 12 | days_since_earnings | 0.015855 |
| 13 | xlk_spy_ratio_chg_20d | 0.015185 |
| 14 | yield_10y | 0.014199 |
| 15 | vix_zscore | 0.013951 |
| 16 | cmf_20 | 0.013123 |
| 17 | corr_spy_20d | 0.013096 |
| 18 | hv_ratio_10_60 | 0.012895 |
| 19 | vix_level | 0.012842 |
| 20 | iv_rank_proxy | 0.012524 |

## Features Pruned (48 dropped, < 0.5% importance)
- `atr_14`
- `bb_mid`
- `bb_squeeze`
- `cdl_doji`
- `close`
- `close_to_ema_20`
- `close_to_ema_50`
- `close_to_sma_20`
- `close_to_sma_50`
- `cmo_14`
- `day_of_week`
- `dc_lower`
- `dc_mid`
- `dc_upper`
- `ema_10`
- `ema_20`
- `ema_50`
- `high`
- `higher_highs_5`
- `hv_10_ann`
- `inside_bar`
- `is_january`
- `kc_lower`
- `kc_mid`
- `kc_position`
- `kc_upper`
- `low`
- `lower_lows_5`
- `mean_reversion_20`
- `open`
- `outside_bar`
- `rel_volume_20d`
- `return_10d`
- `return_20d`
- `return_5d`
- `rsi_14`
- `rsi_21`
- `sma_100`
- `sma_20`
- `sma_200`
- `sma_50`
- `supertrend`
- `supertrend_direction`
- `vol_price_confirm_5d`
- `vwap`
- `vwap_proxy_20`
- `within_5d_earnings`
- `zscore_close_sma20`

## Per-Model Performance
| Model | Avg OOS AUC |
|-------|-------------|
| xgboost | 0.5617 |
| lightgbm | 0.5761 |
| catboost | 0.5737 |

## Assessment

Below target: avg OOS AUC 0.5790 < 0.60. Investigate data leakage, target definition, or feature engineering issues.
AUC stability needs attention (range: 0.3242 >= 0.25). Performance varies significantly across market regimes.
Ensemble (0.5790) matches or beats best single model (0.5761).

## Saved Artifacts
- `models/ensemble_xgb.json`
- `models/ensemble_lgb.pkl`
- `models/ensemble_cat.cbm`
- `models/feature_names.json`
- `models/model_metadata.json`
