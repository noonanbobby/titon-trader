# CLAUDE.md — Project Titan: AI-Powered Options Trading System

## CORE STANDARDS

Production system managing **$150,000 of real capital** on Interactive Brokers. Every bug is a potential **$3,000 loss**.

- **No placeholders, no TODOs, no stubs.** Every function fully implemented.
- **No mock data in production paths.** Test mocks only.
- **No silent failures.** Every error logged with full context, operator notified via Telegram.
- **No bare `except:`.** Specific exceptions, `tenacity` retries with exponential backoff.
- **Circuit breakers are SACRED.** Never bypassed, overridden, or "temporarily disabled."
- **Always use limit orders** on spreads. Never market orders.
- **Monthly expirations only.** Better liquidity than weeklies.
- **Close spreads before expiration.** Pin risk and assignment risk are real.
- **Paper trading first.** Always `IBKR_TRADING_MODE=paper` until validated.
- **If Claude API is down**, pure ML signals must keep the system operational (2-min timeout fallback).

---

## REFERENCE DOCS

Read these before building any component:

| Document | Purpose |
|----------|---------|
| `/docs/MasterBluePrint.md` | **AUTHORITATIVE** — Strategy parameters, risk framework, ML architecture, agent design |
| `/docs/Research-1.md` | GEX mechanics, VRP evidence, insider alpha, RL/Optuna, Docker deployment |
| `/docs/Research-2.md` | IBKR API patterns, combo orders, ML accuracy expectations, FinBERT, QuestDB |
| `/docs/SetupChecklist.md` | API key acquisition, .env template, pre-live checklist |
| `/docs/audit-log.md` | Detailed history of all audit fixes applied |

---

## TRADING PARAMETERS

| Parameter | Value |
|-----------|-------|
| Account Size | $150,000 (target $200K by June 2026) |
| Max Drawdown | 15% ($22,500) — EMERGENCY STOP |
| Per-Trade Risk | 2% ($3,000) max loss |
| Concurrent Positions | 5–8 (max 8) |
| Per-Ticker Exposure | 25–30% max |
| Confidence Threshold | >= 0.78 |
| Trading Frequency | 3–5 trades/week |

### 10 Strategies (details in `config/strategies.yaml` and MasterBluePrint)

| Strategy | Regime | IV Rank | DTE | Key Parameters |
|----------|--------|---------|-----|----------------|
| Bull Call Spread | Low vol trending | 20–50% | 30–45 | Long 0.55–0.60d, Short 0.25–0.30d |
| Bull Put Spread | Uptrend + high IV | 40–70% | 45 | Sell 16–20d, Buy 5–10d |
| Iron Condor | Range-bound | 50–70% | 30–60 | 15–20d shorts, 5–10pt wings |
| Calendar Spread | Low IV | <30% | 30/60 | ATM strikes, positive vega |
| Diagonal Spread | Mild trend + IV | 30–50% | 30/60–90 | OTM front, ITM back |
| Broken Wing Butterfly | Range/mild trend | 30–50% | 21 | Deltas 32/28/20 |
| Short Strangle | Range + high IV | >50% | 30–45 | 16d calls, 16d puts |
| PMCC | Strong uptrend | 20–40% | 180+/30–45 | Long 0.70d LEAPS |
| Ratio Spread | Directional + IV | >40% | 30–45 | Buy 1 ATM, Sell 2 OTM |
| Long Straddle | Pre-catalyst + low IV | <30% | 30–45 | ATM calls + ATM puts |

### Four Regimes (GaussianHMM, 3-state, 4-year rolling window)

| Regime | VIX | ADX | Strategies |
|--------|-----|-----|------------|
| Low Vol Trending | <18 | >25 | Bull Call, PMCC, Diagonal |
| High Vol Trending | 18–35 | >25 | Bull Put, Ratio |
| Range-Bound High IV | 18–35 | <20 | Iron Condor, Strangle, BWB |
| Crisis | >35 | any | Cash, Long Straddle only |

### Risk Framework

| Layer | Controls |
|-------|----------|
| Per-Trade | 2% max loss, defined-risk only, no naked shorts |
| Portfolio | Max 8 positions, 25–30%/ticker, +/-$15K delta, <$5K vega |
| Circuit Breakers | 2% daily->halt, 5% weekly->50% reduce, 10% monthly->stop, 15%->EMERGENCY |
| Tail Risk | SKEW/VVIX composite, earnings/FOMC avoidance (5 days before, 1 day after) |

### Circuit Breaker Recovery (graduated, never instant)

1. **Stage 1 (50%)**: Half size
2. **Stage 2 (75%)**: After 3 consecutive winners at Stage 1
3. **Stage 3 (100%)**: After 3 consecutive winners at Stage 2
4. **Full Reset**: After 2-week cooling period with positive P&L at each stage

---

## SIGNAL ARCHITECTURE

| Stream | Weight | Source |
|--------|--------|--------|
| Technical/ML | 50–60% | XGBoost + LightGBM + CatBoost -> `src/signals/technical.py`, `src/ml/trainer.py` |
| Sentiment | 15–20% | FinBERT (ProsusAI/finbert) on Finnhub news -> `src/signals/sentiment.py` |
| Options Flow | 20–25% | Unusual Whales: vol/OI >= 1.25, sweeps, blocks >$500K -> `src/signals/options_flow.py` |
| Regime Context | 5–10% | HMM, VIX term structure, insider clusters (3+ in 30d), GEX |

**ML accuracy**: Unfiltered 53–60% (normal). Confidence-filtered >0.78: 70–85% at ~12% coverage. Goal is high selectivity, not coverage.

---

## CODING STANDARDS

- **Type hints** on every function signature and return type
- **Pydantic models** for all data structures
- **Async/await** for all I/O (IBKR, databases, APIs, Redis)
- **Structured logging** via `structlog` (JSON format)
- **No global state** — dependency injection via constructors
- **Constants** in `UPPER_CASE`, no magic numbers
- **Config** from environment (`.env`) or YAML only, never hardcoded
- **Tests**: `uv run pytest tests/ -v --tb=short` — strategy logic 90%, risk logic 95% coverage targets

---

## KEY TECHNICAL GOTCHAS

### ib_async (NOT ib_insync)

- ONE session per IBKR username — never log into Client Portal/TWS while bot runs
- Daily reset 00:15–01:45 ET — bot must auto-reconnect
- Rate limit: 50 msg/sec, use `asyncio.sleep(0.05)` between rapid calls
- Max 100 concurrent streaming data lines
- Options chain: `reqSecDefOptParams()` -> filter strikes (+/-20 pts) -> `qualifyContracts()` -> `reqMktData()` (tick types 10–13)
- Combo/BAG orders: `NonGuaranteed = "0"` for guaranteed fills
- Error codes: 502/504 (not connected), 1100 (connectivity lost), 1102 (restored), 2104/2106/2158 (data farm)
- Order Efficiency Ratio must stay below 20:1

### Claude API

- Prompt caching: `cache_control: {"type": "ephemeral"}` on system prompts (90% savings)
- Extended thinking: Analysis 16K tokens, Risk 8K tokens (`thinking.budget_tokens`)
- Batch API for Journal Agent (50% discount)
- Exponential backoff for 429/529

### Options Formulas

```
IV Rank = (Current_IV - 52wk_Low_IV) / (52wk_High_IV - 52wk_Low_IV)
IV Percentile = % of days in past year where IV < current
GEX = Sum(call_OI * call_gamma * 100 * spot) - Sum(put_OI * put_gamma * 100 * spot)
Kelly: f* = (p*b - q) / b, then USE QUARTER-KELLY: f*/4
```

### Databases

- **PostgreSQL**: trades, state, models (schema: `scripts/init_db.sql`). All restart-surviving state.
- **QuestDB**: time-series (market_ticks, gex_levels, signal_scores). Partitioned by DAY.
- **Redis**: cache + pub/sub + streams. 512MB `allkeys-lru`.

---

## AUDIT STATUS (2026-02-24)

All previously reported limitations are resolved. 657 tests passing. Full fix history in `/docs/audit-log.md`.

### Verified Correct (do not re-break these)

| Component | File | Status |
|-----------|------|--------|
| IV Rank formula | `technical.py:770`, `vrp.py:228` | Correct, div-by-zero protected |
| IV Percentile formula | `technical.py:803`, `vrp.py:287` | Correct |
| GEX formula | `gex.py:313` | Correct |
| VRP formula | `vrp.py:172` | Correct (IV - RV) |
| Kelly criterion (quarter-Kelly) | `position_sizer.py:366-383` | Correct, capped at 0.25 |
| Position sizing (`math.floor`) | `position_sizer.py:279` | Never rounds up |
| Circuit breaker state machine | `circuit_breakers.py` | Escalates only, never auto-de-escalates |
| High water mark tracking | `circuit_breakers.py:155-160` | Correct |
| Tail risk weights | `tail_risk.py:123-127` | Sum to 1.0 |
| Isotonic calibration | `ensemble.py:282-283` | After XGBoost, before threshold |
| Confidence threshold (>=0.78) | `ensemble.py:493` | Correct |
| HMM 3-state + 4yr window | `regime.py:142, 245-247` | Correct |
| VIX > 35 crisis override | `regime.py:345-347` | Unconditional in predict() |
| Insider cluster (3+/30d) | `insider.py:69, 72, 337-351` | Excludes 10b5-1 |
| Vol/OI >= 1.25 + sweeps | `options_flow.py:46, 327, 346-363` | Correct |
| FinBERT model | `sentiment.py:153` | ProsusAI/finbert |
| Risk VETO enforcement | `risk_agent.py:788-919` | No bypass path |
| LangGraph pipeline | `agents.py:734` | START->analyze->risk->(execute or reject) |
| ML fallback (2-min) | `agents.py:65, 852-918` | Correct |
| Walk-forward training | `trainer.py:205-350` | Temporal splits with embargo |
| ADWIN drift detection | `online.py:815` | Wired into online learning |
| Backtester slippage | `backtest.py:1005-1100` | 15% of spread |
| Earnings exclusion | `event_calendar.py:493-516` | 5 days before, 1 day after |
| Index contracts | `contracts.py:196-232` | ib_async.Index, CBOE default |
| Index routing | `market_data.py:57-59` | 9 CBOE indices in all methods |
| `build_trade_features()` | `features.py:110-200` | One-hot, direction, NaN imputation |
| `min_recovery_days` | `circuit_breakers.py:257-276, 594-604` | Enforced in trading + recovery |
| Dynamic NYSE holidays | `helpers.py:104-148` | Algorithmic, any year |
