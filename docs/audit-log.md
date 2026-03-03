# Audit Log — Project Titan

Last updated: 2026-02-24

## Previously Reported Limitations (All Resolved)

1. **`min_recovery_days` loaded but never enforced** — Enforced in both `is_trading_allowed()` (`circuit_breakers.py:257-276`) and `_check_recovery_advance()` (`circuit_breakers.py:594-604`). HALT/EMERGENCY levels block trading until the cooling-off period elapses.

2. **Market holidays hardcoded for 2025-2026 only** — Replaced with dynamic `compute_nyse_holidays(year)` in `helpers.py:104-148` using algorithmic computation (Easter, nth-weekday, NYSE observation rules). Works for any year, LRU-cached.

3. **Test coverage was 0% for actual `src/` modules** — 657 tests now cover core `src/` modules: circuit breakers, position sizer, ensemble signals, regime detection, base strategy, helpers, broker contracts, market data routing, and ML feature engineering.

## Fixes Applied (2026-02-24 — Core Audit)

| Fix | File | Description |
|-----|------|-------------|
| Circuit breaker bypass | `manager.py:647` | Removed `await` on sync method, unpacked tuple to fix truthiness check |
| Event calendar method | `manager.py:674` | Changed non-existent `get_blocking_event()` to `is_blocked()` |
| VIX crisis override | `regime.py:345` | Added unconditional VIX > 35 crisis regime override in `predict()` |
| Agent model fallbacks | `agents.py:656-720` | Updated model from `claude-sonnet-4-5-20250929` to `claude-sonnet-4-6` |
| Analysis thinking budget | `agents.py:661` | Fixed fallback from 8192 to 16384 |
| Risk thinking budget | `agents.py:684` | Fixed fallback from 4096 to 8192 |
| Docker port mapping | `docker-compose.yml:16` | Fixed `4002:4004` to `4002:4002` |
| IB Gateway health check | `docker-compose.yml:21` | Changed port 4004 to 4002 |
| HWM seed value | `init_db.sql:139` | Changed from 0.00 to 150000.00 |
| QuestDB flush error handling | `questdb.py:191` | Added try/except for TCP write failures |
| QuestDB connect retry | `questdb.py:136` | Added exponential backoff retry (5 attempts) |
| Scheduler Prometheus metrics | `scheduling.py:361-372` | Wired SCHEDULER_JOB_DURATION and SCHEDULER_JOB_ERRORS |
| Circuit breaker async lock | `circuit_breakers.py:113` | Added asyncio.Lock to protect concurrent state modifications |
| CB level validation | `circuit_breakers.py:413` | Validate loaded level against BreakerLevel enum |
| Sentiment blocking call | `sentiment.py:252` | Wrapped FinBERT inference in `run_in_executor` |
| Telegram deprecation | `telegram.py:576` | Changed `get_event_loop()` to `get_running_loop()` |
| Position sizer min-1 | `position_sizer.py:281` | Removed dangerous minimum-1 contract override |
| CB default to HALT | `circuit_breakers.py:384` | Default to HALT when DB unreachable on startup |
| PnL reset wiring | `scheduling.py, main.py` | Wired daily/weekly/monthly PnL reset jobs |
| Earnings window | `risk_limits.yaml` | Changed days_before from 3 to 5 |
| Redis graceful degradation | `cache.py` | Added try/except to all Redis operations |
| Position check implementation | `main.py` | Implemented actual DTE/profit/stoploss exit criteria evaluation |
| Trade submission lock | `main.py` | Added asyncio.Lock to prevent overlapping scan race conditions |
| Telegram /kill wiring | `telegram.py, main.py` | Wired /kill CONFIRM to TitanApplication.request_kill() |
| Ticker+strategy dedup | `manager.py` | Added composite dedup check in position limits |
| Strategy constructors (7) | `calendar_spread.py`, `diagonal_spread.py`, `broken_wing_butterfly.py`, `short_strangle.py`, `pmcc.py`, `ratio_spread.py`, `long_straddle.py` | Fixed `__init__(config)` to `__init__(name, config)` and `super().__init__(config)` to `super().__init__(name, config)`, removed redundant `@property name` overrides |
| Signal pipeline | `main.py:1530-1800` | Implemented full `_run_signal_pipeline()`: spot price fetch, parallel signal gathering (sentiment, flow, insider, cross-asset), regime detection, VRP, ensemble scoring, confidence gate, AI orchestrator routing with fallback |
| Order status polling | `execution_agent.py:1008-1095` | Implemented `_poll_order_status()`: queries OrderManager for live Trade objects, maps ib_async status strings to canonical states |
| Orchestrator -> OrderManager | `agents.py:607, 697` | Added `order_manager` parameter pass-through from orchestrator to ExecutionAgent |

## Fixes Applied (2026-02-24 — Intelligence Layer)

| Fix | File | Description |
|-----|------|-------------|
| VIX Index contract support | `contracts.py` | Added `Index` import and `create_index()` method to `ContractFactory` (exchange default: CBOE) |
| Index routing in MarketDataManager | `market_data.py` | Added `_INDEX_SYMBOLS` frozenset (VIX, SPX, NDX, RUT, VVIX, etc.) and routing in `get_snapshot()`, `get_historical_bars()`, `get_historical_iv()` |
| `build_trade_features()` method | `features.py` | Implemented missing method for weekly retrain |
| Model path alignment | `main.py` | After `trainer.train()`, copies latest `titan_xgboost_ensemble_v*.json` to `ensemble_xgb.json` |
| Broker test coverage | `tests/test_broker/` | 26 + 35 tests for ContractFactory and MarketDataManager |
| ML features test coverage | `tests/test_ml/test_features.py` | 40 tests for FeatureEngineer |
