# Phase 0: Forensic Audit Report
**Generated:** 2026-03-03 21:45 UTC
**System:** Project Titan on EC2 (Ubuntu 24.04, Docker Compose)
**Auditor:** Claude Opus 4.6

---

## Executive Summary

**Overall Grade: B-** — The codebase is impressively complete at 53,223 lines across 73 Python files with zero TODOs, zero stubs, and 657 passing tests. All 10 strategies, 9 signal modules, 4 AI agents, and full risk management are implemented as real production code. However, **7 of 10 strategy classes will crash at runtime** due to Pydantic model field mismatches with the base class, the circuit breaker is **not escalating despite 17.87% computed drawdown** (exceeding the 15% EMERGENCY threshold), IB Gateway connectivity is broken (risk monitor throws errors every 5 minutes), CatBoost is missing from the ML ensemble, and QuestDB has zero tables/data. The system is not ready for live or paper trading until these critical issues are resolved.

---

## Component Scoreboard

| Component | Status | Grade | Notes |
|-----------|--------|-------|-------|
| Historical Data | 0 rows on-disk, bootstrap used Polygon API | **D** | No local data, QuestDB empty, no seed script |
| Feature Engineering | 48 features (20 technical + 28 signal) | **B+** | Real implementation, dynamic feature count |
| ML Training Pipeline | Walk-forward: YES, XGBoost + LightGBM | **B** | CatBoost missing, bootstrap model has 28/48 features as zeros |
| Calibration | Isotonic regression, real | **A** | ECE/MCE/Brier evaluation, proper sklearn implementation |
| Regime Detector | HMM 3-state, 4yr window, VIX>35 override | **A** | All formulas verified correct |
| GEX Calculator | Full implementation, 5 key levels | **A** | Correct formula, dealer positioning, vol trigger |
| Signal Pipeline | 9/9 modules real, 101 functions | **A-** | Sentiment contrarian filter defined but not wired in |
| Strategy Engine | 10/10 implemented, but 7/10 have interface bugs | **D+** | 7 strategies will crash on Pydantic ValidationError |
| Risk Management | 10-layer pipeline, CB, Kelly, events | **B+** | CB not escalating due to IB disconnect; hardcoded sector map |
| AI Agents | 4 agents + LangGraph orchestrator | **A-** | Regime-strategy mapping mismatch in fallback mode |
| Broker Layer | ib_async, limit-only, ComboLeg, monthly filter | **C** | IB Gateway healthy but Titan not connected; risk monitor errors |
| Notifications | Telegram working, Twilio 401 errors | **C+** | Twilio credentials are placeholders |

---

## Critical Findings (P0 — Fix Immediately)

### 1. Seven of Ten Strategies Will Crash at Runtime
**Files:** `calendar_spread.py`, `diagonal_spread.py`, `broken_wing_butterfly.py`, `short_strangle.py`, `pmcc.py`, `ratio_spread.py`, `long_straddle.py`

These 7 strategies use an incompatible interface with the base class:

| Issue | Base Class (`base.py`) | 7 Broken Strategies |
|-------|----------------------|---------------------|
| `is_eligible()` args | `(regime, iv_rank)` — 2 args | `(regime, iv_rank, spot_price)` — 3 args → **TypeError** |
| `TradeSignal` field | `strategy_name=` | `strategy=` → **ValidationError** |
| `TradeSignal` field | `entry_reasoning=` | `reasoning=` → **ValidationError** |
| `TradeSignal` field | N/A | `net_premium=`, `regime=`, `iv_rank=` → **extra fields rejected** |
| `LegSpec` field | N/A (frozen model) | `mid_price=` → **ValidationError** |
| `check_mechanical_exit()` args | `(current_pnl_pct, dte_remaining)` — 2 args | `(trade, current_pnl, current_pnl_pct, dte_remaining)` — 4 args → **TypeError** |
| `ExitSignal` fields | `exit_type=`, `reasoning=`, `current_pnl=` | `reason=`, `details=`, `urgency=` → **ValidationError** |
| `check_entry()` types | `greeks: dict[str, float]`, `options_chain: list[dict]` | `greeks: GreeksSnapshot`, `options_chain: list[OptionData]` |

Only **bull_call_spread**, **bull_put_spread**, and **iron_condor** match the base class interface.

### 2. Circuit Breaker Not Escalating Despite 17.87% Drawdown
**File:** `circuit_breakers.py`
**Evidence:** PostgreSQL shows `level=NORMAL, total_drawdown_pct=0.1787, high_water_mark=150000.00`

The drawdown exceeds the EMERGENCY threshold (15%), yet the level is NORMAL. Root cause: IB Gateway is not connected → `accountSummary()` fails → `update_pnl()` never runs → thresholds never evaluated. The CB loaded NORMAL from DB at startup and never received a valid NLV update.

Additionally, the HWM of $150,000 may be an initialization artifact — the paper account NLV is $123,193 and may have never been at $150K. If HWM was seeded from the `ACCOUNT_SIZE` env var rather than actual NLV, the drawdown calculation is wrong.

### 3. IB Gateway Connectivity Broken
**Evidence:** Health endpoint shows `ib_connected: false`. Logs show `<IB not connected>` errors on `accountSummaryAsync()` every 5 minutes. The IB Gateway container is healthy (ports 4001/4002 open), but the Titan application is not maintaining an active connection.

**Impact:** No market data streaming, no account updates, no order placement. The entire trading pipeline is non-functional.

### 4. Health Endpoint Reports All Backends Disconnected
**Evidence:** `ib_connected: false, redis_connected: false, postgres_connected: false`

PostgreSQL IS being written to (84 circuit_breaker_state rows), yet the health check reports it as disconnected. Redis is responding to PING but reported as disconnected. The health check likely uses separate connection checks that aren't using the application's connection pools.

---

## Major Findings (P1 — Fix Before Live Trading)

### 5. CatBoost Missing from ML Ensemble
**Spec:** CLAUDE.md requires "XGBoost + LightGBM + CatBoost" ensemble.
**Reality:** CatBoost is not in `pyproject.toml` and is not installed. Only XGBoost model files exist in `models/`. The trainer supports all three backends but CatBoost cannot be used.

### 6. Bootstrap Model Has 28/48 Features as Constants
**File:** `models/ensemble_xgb_metadata.json`
The bootstrap model was trained with sentiment (3), flow (4), GEX (3), insider (4), and partial VRP/cross-asset features filled with neutral zeros. XGBoost assigns near-zero importance to constant features. When live signals populate those 28 features with real data, the model will ignore them. Weekly retraining on live data is critical.

### 7. QuestDB Has Zero Tables
**Evidence:** `SHOW TABLES` returns empty dataset.
No `market_ticks`, `gex_levels`, or `signal_scores` tables exist. Time-series data ingestion is not operational. This blocks historical analysis, signal replay, and dashboard time-series visualizations.

### 8. Twilio SMS Credentials Are Placeholders
**Evidence:** Logs show `HTTP 401 error: Authentication Error - invalid username`. The Twilio SID is `your_sid`. SMS alerting is completely non-functional. Only Telegram notifications work.

### 9. Regime-Strategy Mapping Mismatch in ML Fallback
**File:** `src/ai/agents.py:93-114`
The `REGIME_STRATEGY_MAP` used during Claude API outages does not match CLAUDE.md:
- `high_vol_trend` maps to `iron_condor, short_strangle, broken_wing_butterfly` — should be `bull_put_spread, ratio_spread`
- `low_vol_trend` includes `bull_put_spread` — should be `bull_call_spread, pmcc, diagonal_spread`

### 10. Online Learning Rate Changes Don't Apply to River Model
**File:** `src/ml/online.py`
The `INCREASE_LR` drift response updates a `current_learning_rate` property, but River's `HoeffdingAdaptiveTreeClassifier` has no learning rate parameter. The graduated response logs the change but doesn't affect model behavior.

### 11. Target Threshold Mismatch Between Bootstrap and Feature Engineer
**Files:** `scripts/train_bootstrap.py:53` vs `src/ml/features.py:71`
Bootstrap uses 1% (0.01) forward return threshold for positive labels. FeatureEngineer uses 2% (0.02). Different definitions of "positive" between training and live inference.

---

## Minor Findings (P2 — Fix When Convenient)

### 12. Hardcoded Sector Map in Risk Manager
**File:** `src/risk/manager.py:188-235` — Ticker-to-sector mapping is hardcoded instead of loaded from `config/tickers.yaml`. New tickers require code changes.

### 13. Sentiment Contrarian Filter Not Wired In
**File:** `src/signals/sentiment.py` — `_apply_contrarian_filter()` method exists but is never called from `_calculate_rolling_sentiment()`. StockTwits/Reddit sentiment is not getting the contrarian weighting specified in the blueprint.

### 14. `_get_entry_iv` Returns 0.0 in Long Straddle
**File:** `src/strategies/long_straddle.py` — The IV collapse exit condition (>20% drop) uses `_get_entry_iv()` which always returns 0.0, making the exit check dead code.

### 15. Static Economic Calendar Dates Need Annual Updates
**File:** `src/risk/event_calendar.py:67-158` — FOMC/CPI/NFP dates are hardcoded through 2026. Comment says `# UPDATE ANNUALLY`.

### 16. Calibration Set Overlaps Training Folds in Bootstrap
**File:** `scripts/train_bootstrap.py:734-737` — The last 20% used for calibration can overlap with the last walk-forward test fold.

### 17. Generic Feature Names (f00–f47)
**Files:** `train_bootstrap.py`, `ensemble.py` — Positional feature names instead of descriptive names. Creates fragile coupling and hurts interpretability.

### 18. fold_num Always 0 in FoldMetrics
**File:** `src/ml/trainer.py:722` — `fold_num` is never updated by the caller; all fold results report fold 0.

### 19. Gateway Data Type Based on Port Number
**File:** `src/broker/gateway.py:238` — Market data type (LIVE vs DELAYED) determined by `port == 4002` rather than reading `IBKR_TRADING_MODE` setting.

### 20. Risk Agent Delta Check Uses Raw Delta (500) vs Spec Dollar-Delta ($15K)
**File:** `src/ai/risk_agent.py:81` — `HARD_MAX_PORTFOLIO_DELTA = 500.0` is raw contract delta. The manager.py has a separate $15K dollar-delta check, so double-covered but semantically different.

---

## Dependencies Missing

| Package | Required By | Status |
|---------|------------|--------|
| **catboost** | ML ensemble (3-model spec) | **NOT INSTALLED, not in pyproject.toml** |
| All others | — | Installed and correct versions |

**Installed packages verified (key ones):**
- xgboost 3.2.0, lightgbm 4.6.0, hmmlearn 0.3.3, river 0.23.0, optuna 4.7.0
- anthropic 0.83.0, langgraph 1.0.9, ib-async 2.1.0
- transformers 5.2.0, torch 2.10.0, scikit-learn 1.8.0
- stable-baselines3 2.7.1, pydantic 2.12.5

---

## Data Gaps

| Data Source | Status | Rows | Gap |
|------------|--------|------|-----|
| Local disk (`data/`) | **EMPTY** | 0 | No parquet/CSV files at all |
| QuestDB time-series | **EMPTY** | 0 tables | No market_ticks, gex_levels, or signal_scores |
| PostgreSQL trades | Empty | 0 | Expected — no trades executed yet |
| PostgreSQL account_snapshots | Empty | 0 | Should be populated by risk monitor |
| PostgreSQL circuit_breaker_state | Active | 84 | Recording every 5 min |
| Bootstrap training data | Via Polygon API | 3,402 samples | 14 tickers x ~243 days. Sufficient for bootstrap. |
| `scripts/seed_historical.py` | **DOES NOT EXIST** | — | No automated data seeding pipeline |

**For 4-year walk-forward training per the spec (1,000+ rows/ticker), no local historical data infrastructure exists.** The bootstrap script fetches from Polygon on-demand but does not persist locally.

---

## ML Pipeline Gaps vs Specification

| Spec Requirement | Current State | Gap |
|-----------------|---------------|-----|
| XGBoost + LightGBM + CatBoost | XGBoost only (trained), LightGBM supported but no model, CatBoost not installed | CatBoost missing; no multi-model averaging |
| 120+ features from pandas-ta | `technical.py` computes 120+ features correctly | None |
| Walk-forward CV with embargo | Implemented: expanding window, 5-day embargo | None |
| Isotonic calibration | Real sklearn IsotonicRegression | None |
| Confidence >= 0.78 threshold | Enforced at `ensemble.py:495` | None |
| ADWIN drift detection | Wired in via River, graduated response | Learning rate change has no effect on River model |
| RL agent (SAC/PPO) | `rl_agent.py` implemented (1,934 lines) | Not trained, no model file |
| Optuna optimization | `optimizer.py` implemented (2,014 lines) | Not run yet |
| Weekly retraining | `scheduling.py` has the schedule | No trained model beyond bootstrap |

---

## Recommendations for Next Phases (Priority Order)

### Phase 1: Make the System Operational
1. **Fix IB Gateway connectivity** — Debug why Titan app can't connect despite healthy gateway container. This unblocks everything.
2. **Fix health endpoint** — Ensure it reflects actual connection state of PostgreSQL, Redis, IB.
3. **Fix circuit breaker HWM initialization** — Either seed from actual NLV or recalculate on startup.

### Phase 2: Fix Strategy Interface Mismatches
4. **Align 7 broken strategies** with `base.py` interface:
   - Change `is_eligible()` calls from 3 args to 2
   - Change `TradeSignal(strategy=...)` to `TradeSignal(strategy_name=...)`
   - Change `entry_reasoning` field name
   - Remove extra fields (`net_premium`, `regime`, `iv_rank`)
   - Remove `mid_price` from `LegSpec`
   - Fix `check_mechanical_exit()` call signatures
   - Fix `ExitSignal` field names
5. **Run all 657 tests** to verify fixes don't break existing passing tests.

### Phase 3: Complete ML Pipeline
6. **Add CatBoost** to `pyproject.toml` and implement 3-model ensemble averaging.
7. **Fix target threshold mismatch** (1% vs 2%) between bootstrap and feature engineer.
8. **Create data seeding infrastructure** — Local Polygon data cache, QuestDB ingestion.
9. **Run Optuna optimization** on bootstrap data.

### Phase 4: Data Infrastructure
10. **Initialize QuestDB tables** — Create `market_ticks`, `gex_levels`, `signal_scores`.
11. **Build `seed_historical.py`** script for local data persistence.
12. **Verify data pipeline end-to-end** — Market data → QuestDB → feature engineering.

### Phase 5: Notifications and Monitoring
13. **Configure Twilio credentials** or remove SMS alerting.
14. **Wire in sentiment contrarian filter**.
15. **Fix online learning rate** to use River-compatible mechanism.

### Phase 6: Paper Trading Validation
16. **Full smoke test** — End-to-end with paper account: signal → strategy → risk → execution.
17. **Monitor for 1 week** — Verify circuit breakers, event calendar, position management.
18. **Retrain ML models** on live signal data after 2 weeks of data collection.

---

## Appendix A: File Inventory

### Line Counts by Module

| Module | Files | Lines | Assessment |
|--------|-------|-------|------------|
| `src/main.py` | 1 | 2,663 | REAL |
| `src/ai/` | 7 | 7,764 | REAL |
| `src/broker/` | 5 | 3,669 | REAL |
| `src/data/` | 8 | 3,752 | REAL |
| `src/ml/` | 7 | 10,218 | REAL |
| `src/notifications/` | 3 | 2,310 | REAL |
| `src/risk/` | 7 | 4,362 | REAL |
| `src/signals/` | 9 | 6,857 | REAL |
| `src/strategies/` | 12 | 9,100 | REAL (7 have interface bugs) |
| `src/utils/` | 4 | 1,641 | REAL |
| **Total** | **73** | **53,223** | **73/73 REAL (no stubs)** |

### Docker Services

| Service | Image | Status | Health |
|---------|-------|--------|--------|
| titan | titan-titan (custom) | Up 3h | Healthy |
| ib-gateway | ghcr.io/gnzsnz/ib-gateway | Up 3h | Healthy |
| postgres | postgres:16 | Up 15h | Healthy |
| questdb | questdb/questdb:latest | Up 15h | Healthy |
| redis | redis:7-alpine | Up 15h | Healthy |
| prometheus | prom/prometheus:latest | Up 14h | Running |
| grafana | grafana/grafana:latest | Up 14h | Running |

### Models on Disk

| File | Size | Description |
|------|------|-------------|
| `ensemble_xgb.json` | 91KB | XGBoost model (bootstrap, 48 features, 3402 samples) |
| `ensemble_xgb.calibrator.pkl` | 472B | Isotonic calibration model |
| `ensemble_xgb_metadata.json` | 1KB | Training metadata |

---

## Appendix B: Verified Correct Components

These items were individually verified during this audit and match the CLAUDE.md specification:

- IV Rank formula (`technical.py:770`, `vrp.py:228`) — div-by-zero protected
- IV Percentile formula (`technical.py:803`, `vrp.py:287`)
- GEX formula (`gex.py:313`) — canonical sign convention
- VRP formula (`vrp.py:172`) — IV minus RV
- Kelly criterion quarter-Kelly (`position_sizer.py:366-383`) — capped at 0.25
- Position sizing `math.floor` (`position_sizer.py:279`) — never rounds up
- Circuit breaker escalate-only (`circuit_breakers.py:206-208`)
- High water mark tracking (`circuit_breakers.py:172-178`)
- Tail risk weights sum to 1.0 (`tail_risk.py:123-127`)
- Isotonic calibration (`ensemble.py:282-283`) — real sklearn IsotonicRegression
- Confidence threshold >= 0.78 (`ensemble.py:493`)
- HMM 3-state + 4yr window (`regime.py:142, 245-247`)
- VIX > 35 crisis override (`regime.py:345-347`) — unconditional
- Insider cluster 3+/30d (`insider.py:69, 72, 337-351`) — excludes 10b5-1
- Vol/OI >= 1.25 + sweeps (`options_flow.py:46, 327, 346-363`)
- FinBERT model = ProsusAI/finbert (`sentiment.py:153`)
- Risk VETO enforcement (`risk_agent.py:788-919`) — no bypass path
- LangGraph pipeline: START → analyze → risk → (execute or reject)
- ML fallback 2-min timeout (`agents.py:65, 852-918`)
- Walk-forward training with embargo (`trainer.py:205-350`)
- ADWIN drift detection wired (`online.py:815`)
- Backtester slippage 15% (`backtest.py:1005-1100`)
- Earnings exclusion 5d before / 1d after (config overrides code default of 3)
- Index contracts CBOE (`contracts.py:196-232`)
- Index routing 9 symbols (`market_data.py:57-59`)
- Limit orders only — no market orders anywhere in codebase
- Monthly expirations enforced via 3rd-Friday filter
- NonGuaranteed = "0" for guaranteed fills on combo orders
