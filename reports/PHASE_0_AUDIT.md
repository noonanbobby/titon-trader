# Phase 0: Forensic Audit Report
**Generated:** 2026-03-03T20:30:00Z
**System:** Project Titan on EC2 (paper trading mode)
**Auditor:** Claude Opus 4.6 automated forensic audit

---

## Executive Summary

**Overall Grade: B+** — Project Titan is a substantial, genuinely implemented system with 50,560 lines of production Python across 73 files, 659 passing tests, and all 7 Docker containers running healthy. The codebase is real — no stubs, no placeholders, no TODOs. However, **7 of 10 strategy implementations have interface mismatches** that will cause runtime crashes when those strategies attempt to generate trades, the health endpoint is broken due to a property/method call bug, and the ML model is a bootstrap trained only on technical features. The system is currently running in paper mode with $123K net liquidation and zero open positions, executing scheduled risk monitoring and position checks.

---

## Component Scoreboard

| Component | Status | Grade | Notes |
|-----------|--------|-------|-------|
| **Historical Data** | 0 rows in QuestDB, 0 files in data/ | **D** | Bootstrap model trained from Polygon API (3,402 samples). No persistent historical store. |
| **Feature Engineering** | 48 features, 6 cross-features | **A** | 3 selection methods, normalization, target creation, lag features, correlation removal |
| **ML Training Pipeline** | Walk-forward, 3 models, 5-day embargo | **A** | XGBoost + LightGBM + CatBoost all implemented. Purged k-fold CV. |
| **Calibration** | Real isotonic regression | **A-** | sklearn IsotonicRegression + Platt scaling. Calibrator pickle only 472B (minimal fit). |
| **Regime Detector** | GaussianHMM 3-state, 4yr window | **A** | VIX>35 crisis override unconditional. Rule-based backup. |
| **GEX Calculator** | Full implementation | **A** | Call Wall, Put Wall, Vol Trigger, Zero Gamma, Max Pain levels |
| **Signal Pipeline** | All 8 streams real (6,857 lines) | **A** | Technical, Sentiment, Flow, Regime, GEX, Insider, VRP, Cross-Asset |
| **Strategy Engine** | 10 strategies loaded, 7 have interface bugs | **D** | 3 strategies correct (bull_call, bull_put, iron_condor). 7 will crash at runtime. |
| **Risk Management** | 10-layer checks, real state machine | **A** | Circuit breakers, Kelly sizer, event calendar, tail risk, correlation, portfolio Greeks |
| **AI Agents** | 4-agent LangGraph pipeline | **A** | VETO enforcement, ML fallback, prompt caching, extended thinking |
| **Broker Layer** | Full ib_async integration | **A** | 9 CBOE indices, combo/BAG orders, rate limiting, auto-reconnect |
| **Notifications** | Telegram + SMS + HTML reports | **A-** | Rate-limited. Interactive commands. Kill requires confirmation. |
| **Health Endpoint** | Broken — always reports false | **F** | Property called as method; health providers never registered |
| **Infrastructure** | All 7 containers healthy | **A** | All 26 env vars configured. All Python packages installed. |

---

## Critical Findings (Fix Immediately — P0)

### 1. Strategy Interface Mismatch: 7 of 10 Strategies Will Crash at Runtime
**Severity: BLOCKER** — These strategies cannot generate or exit trades.

**Cohort B** (calendar_spread, diagonal_spread, broken_wing_butterfly, short_strangle, pmcc, ratio_spread, long_straddle) has **4 categories of bugs**:

| Bug | Detail | Effect |
|-----|--------|--------|
| `is_eligible()` called with 3 args | Base class accepts `(regime, iv_rank)`. Cohort B passes `(regime, iv_rank, spot_price)` | `TypeError` at entry check |
| `check_mechanical_exit()` called with 4 args | Base class accepts `(current_pnl_pct, dte_remaining)`. Cohort B passes `(trade, current_pnl, current_pnl_pct, dte_remaining)` | `TypeError` at exit check |
| `TradeSignal(strategy=...)` wrong field name | Model field is `strategy_name`, not `strategy` | Pydantic `ValidationError` |
| Extra fields on TradeSignal | `net_premium`, `regime`, `iv_rank`, `reasoning` not in model (uses `entry_reasoning`) | Pydantic `ValidationError` |

**Only 3 strategies work correctly:** bull_call_spread, bull_put_spread, iron_condor (Cohort A).

**Files affected:**
- `src/strategies/calendar_spread.py:140,309,355`
- `src/strategies/diagonal_spread.py:140,321`
- `src/strategies/broken_wing_butterfly.py:166,348`
- `src/strategies/short_strangle.py:148,215`
- `src/strategies/pmcc.py:190,269`
- `src/strategies/ratio_spread.py:152,216`
- `src/strategies/long_straddle.py:168,250`

### 2. RiskManager Field Name Mismatch
**Severity: BLOCKER** — Risk evaluation crashes for ALL strategies.

`src/risk/manager.py:377` accesses `signal.strategy` but `TradeSignal` defines `strategy_name`. Will raise `AttributeError` on every trade evaluation.

### 3. Health Endpoint Property Call Bug
**Severity: HIGH** — Monitoring is blind to actual system state.

`src/main.py:1229` calls `self._gateway.is_connected()` with parentheses, but `is_connected` is a `@property` on `GatewayManager`. This calls `True()` / `False()` which raises `TypeError`, suppressed by `contextlib.suppress(Exception)`. Result: health endpoint always reports `ib_connected: false` even when connected.

### 4. Health Providers Never Registered
**Severity: HIGH** — Health endpoint has no data sources.

`register_health_providers()` is defined in `src/utils/metrics.py:264` but **never called** from `src/main.py`. The `/health` endpoint returns hardcoded defaults for all fields (`ib_connected: false`, `redis_connected: false`, `postgres_connected: false`, `regime: "unknown"`).

---

## Major Findings (Fix Before Live Trading — P1)

### 5. Bootstrap ML Model — 5 of 8 Signal Streams Have Neutral Defaults
The deployed model (`models/ensemble_xgb.json`) was trained by `scripts/train_bootstrap.py` with only technical features. Sentiment, options flow, insider, GEX, and cross-asset features were set to neutral defaults during training. The model has:
- 3,402 training samples across 14 tickers
- Generic feature names (`f00`-`f47`) instead of semantic names
- No validation metrics (AUC, accuracy) in metadata
- Different metadata schema than `WalkForwardTrainer.save_model()` produces

**Impact:** The model cannot properly weight 5 of 8 signal streams. It's effectively a technical-only model.

### 6. Calibrator May Be Minimally Fitted
`models/ensemble_xgb.calibrator.pkl` is only 472 bytes. A properly fitted `IsotonicRegression` on 3,402 samples would typically be larger. May indicate a synthetic or trivially fitted calibrator.

### 7. No Historical Data Persistence
- QuestDB: 0 tables, 0 rows
- `data/` directory: empty (only `.gitkeep`)
- PostgreSQL: 0 rows in trades, trade_legs, agent_decisions, model_versions, account_snapshots
- Only circuit_breaker_state has data (78 rows)

The system has no accumulated market data for analysis, backtesting, or model retraining.

### 8. strategies.yaml Parameters Deviate from CLAUDE.md Spec

| Parameter | CLAUDE.md | strategies.yaml |
|-----------|-----------|-----------------|
| Bull Call IV Rank | 20-50% | 0-50% |
| Bull Put IV Rank | 40-70% | 30-100% |
| Iron Condor IV Rank | 50-70% | 25-75% |
| Calendar IV Rank | <30% | 10-50% |

The YAML is more permissive than the blueprint specification.

### 9. `except Exception:` Proliferation
125+ broad `except Exception:` catches across the codebase. While all include logging, this makes debugging harder and can mask unexpected errors. Most concentrated in `src/main.py` (60+).

---

## Minor Findings (Fix When Convenient — P2)

### 10. `pass` in 8 Exception Handlers
All in parsing/fallback chains (`journal_agent.py:1286,1295,1313,1324`, `quiver.py:257`, `main.py:1302`, `rl_agent.py:1100`, `backtest.py:1999`). Silent error suppression — should at minimum log at debug level.

### 11. RiskManager Docstring Says 7 Layers, Implementation Has 10
Layers 8 (portfolio delta dollars), 9 (portfolio vega dollars), and 10 (market hours) are undocumented.

### 12. risk_limits.yaml Uses Raw Greek Units vs CLAUDE.md Dollar Units
CLAUDE.md: `max_portfolio_delta: +/-$15K`, `max_portfolio_vega: <$5K`
risk_limits.yaml: `max_portfolio_delta: 500` (delta units), `max_portfolio_vega: 1000` (vega units)
RiskManager layer 8-9 uses dollar equivalents ($15K delta dollars, $5K vega dollars), so both are enforced.

### 13. `seed_historical.py` Does Not Exist
`scripts/seed_historical.py` is referenced conceptually but the file doesn't exist. Only `scripts/train_bootstrap.py` exists, which fetches data transiently for training.

---

## Dependencies Status

### All Required Packages: INSTALLED

| Package | Version | Status |
|---------|---------|--------|
| xgboost | 3.2.0 | Installed |
| lightgbm | 4.6.0 | Installed |
| catboost | N/A | **NOT INSTALLED** (not in pip list) |
| hmmlearn | 0.3.3 | Installed |
| river | 0.23.0 | Installed |
| optuna | 4.7.0 | Installed |
| torch | 2.10.0 | Installed |
| transformers | 5.2.0 | Installed |
| ib_async | 2.1.0 | Installed |
| structlog | 25.5.0 | Installed |
| pydantic | 2.12.5 | Installed |
| anthropic | 0.83.0 | Installed |
| langgraph | 1.0.9 | Installed |
| scikit-learn | 1.8.0 | Installed |
| scipy | 1.17.1 | Installed |
| numpy | 2.2.6 | Installed |
| pandas | 2.3.3 | Installed |
| redis | 7.2.0 | Installed |
| asyncpg | 0.31.0 | Installed |
| httpx | 0.28.1 | Installed |
| tenacity | 9.1.4 | Installed |
| python-telegram-bot | 22.6 | Installed |
| stable-baselines3 | 2.7.1 | Installed |
| quantstats | 0.0.81 | Installed |

**Missing:** CatBoost is not in the pip list. The trainer supports it but it may not be importable at runtime.

---

## Data Gaps

| Data Source | Current State | Required For |
|-------------|---------------|--------------|
| QuestDB time-series | 0 tables, 0 rows | Market ticks, GEX levels, signal scores |
| Historical OHLCV | 0 persistent files | 4-year walk-forward (need 1000+ rows/ticker) |
| Trades history | 0 rows in PostgreSQL | Journal agent, pattern learning, model retraining |
| Account snapshots | 0 rows | Performance tracking, drawdown monitoring |
| Agent decisions | 0 rows | Decision audit trail |
| Model versions | 0 rows | Model governance, A/B testing |

---

## ML Pipeline Gaps

| Spec (MasterBluePrint) | Actual Status |
|------------------------|---------------|
| XGBoost + LightGBM + CatBoost ensemble | Only XGBoost model trained. LightGBM/CatBoost implemented but unused. |
| Walk-forward with 4-year window | Bootstrap used 2-year window (Polygon free tier limit) |
| 8 signal streams feeding ensemble | Only technical features have real data. 5 streams are neutral zeros. |
| Weekly retraining | Not yet scheduled. Bootstrap is version 1. |
| Isotonic calibration on validation set | Calibrator exists (472B) but quality is uncertain |
| Optuna hyperparameter optimization | `src/ml/optimizer.py` exists (2,014 lines) but not yet run |
| RL agent | `src/ml/rl_agent.py` exists (1,934 lines) but not yet integrated |
| ADWIN online learning | `src/ml/online.py` exists (1,386 lines) but not yet wired to live pipeline |
| Backtester validation | `src/ml/backtest.py` exists (2,561 lines) but no backtest results saved |

---

## Test Coverage

- **659 tests passing** (4.22 seconds)
- Test areas: broker, ML, risk, signals, strategies, utils, trading math
- Tests run on host (not in container — `tests/` not copied to Docker image)

---

## Infrastructure Status

| Container | Status | Health |
|-----------|--------|--------|
| titan | Up 2h | Healthy (but /health endpoint broken) |
| ib-gateway | Up 2h | Healthy |
| postgres | Up 13h | Healthy (6 tables, only circuit_breaker_state has data) |
| questdb | Up 13h | Healthy (0 tables) |
| redis | Up 13h | Healthy (0 keys) |
| grafana | Up 13h | Running (dashboards provisioned) |
| prometheus | Up 13h | Running |

**Trading Mode:** `IBKR_TRADING_MODE=paper`
**Net Liquidation:** $123,193.24
**Buying Power:** $491,670.12
**Open Positions:** 0

---

## Recommendations for Next Phases

### Phase 1: Critical Bug Fixes (Do First)
1. **Fix 7 Cohort B strategy interface mismatches** — Align to base class contract or extend base class
2. **Fix RiskManager `signal.strategy` -> `signal.strategy_name`**
3. **Fix health endpoint** — Change `is_connected()` to `is_connected` (remove parens)
4. **Register health providers** in main.py startup sequence
5. **Verify CatBoost installation** — Add to pyproject.toml if missing

### Phase 2: ML Model Quality
6. **Retrain model with all 8 signal streams** once live data accumulates
7. **Run Optuna hyperparameter optimization**
8. **Validate calibrator quality** — Check ECE/Brier scores
9. **Align strategies.yaml IV ranges** with CLAUDE.md specification

### Phase 3: Data Infrastructure
10. **Create QuestDB tables** and start persisting market data
11. **Build historical data seeder** for backtesting
12. **Enable weekly retraining pipeline**
13. **Wire ADWIN online learning** to live prediction loop

### Phase 4: Hardening
14. **Reduce `except Exception:` scope** — Use specific exceptions
15. **Add integration tests** for strategy->risk->execution pipeline
16. **Backtest all 10 strategies** and validate returns
17. **Wire RL agent** for position management optimization

### Phase 5: Live Readiness
18. **Paper trade for 2+ weeks** with all strategies functional
19. **Validate order execution** end-to-end (entry, monitoring, exit)
20. **Stress test** circuit breakers and emergency stop
21. **Switch to live** only after documented validation of each component
