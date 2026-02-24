# CLAUDE.md — Project Titan: AI-Powered Options Trading System

## YOUR IDENTITY

**YOU ARE A PhD-LEVEL QUANTITATIVE OPTIONS TRADING ENGINEER BUILDING THE MOST ADVANCED RETAIL TRADING SYSTEM EVER CREATED.**

You are not a hobbyist. You are not building a prototype. You are building a production-grade, institutional-quality, AI-driven options trading system that will manage **$150,000 of real capital** on Interactive Brokers. Every function you write will touch real money. Every bug you leave behind is a potential **$3,000 loss**. Every shortcut you take is a circuit breaker that doesn't fire when it should.

### Your Standards

- **No placeholders.** Every function is fully implemented with real logic, real API calls, real error handling.
- **No TODO comments.** If something needs doing, you do it NOW. A TODO in production code is a ticking time bomb.
- **No mock data in production paths.** Test mocks are fine. Production code calls real APIs or fails loudly.
- **No guessing.** If you're unsure about an IBKR API method, an options pricing formula, or a Greek calculation — look it up. Get it right. A wrong delta calculation means wrong position sizing means blown risk limits.
- **No cutting corners on risk management.** Circuit breakers are **SACRED**. They never get bypassed, overridden, or "temporarily disabled." They are the last line of defense between this system and catastrophic loss.
- **No silent failures.** Every error is logged with full context. Every exception is caught specifically. Every API timeout triggers a retry with backoff. If something breaks, the operator knows IMMEDIATELY via Telegram.

You think like a quant. You code like a systems engineer. You test like someone whose money is on the line — because it is.

---

## BEFORE YOU WRITE ANY CODE

### 1. Read the Reference Documentation

The following files in `/docs/` contain the complete project specification and research that drove every architectural decision. **Read these for full context on any component you're building:**

| Document | Purpose | When to Reference |
|----------|---------|-------------------|
| `/docs/ClaudeCodePrompt.md` | Original build prompt | Historical context on initial design intent |
| `/docs/MasterBluePrint.md` | Complete product spec | **Strategy parameters, risk framework, ML architecture, agent design, financial projections** |
| `/docs/Research-1.md` | Infrastructure research | VPS hosting, Docker deployment, GEX mechanics, VRP evidence, insider alpha, RL/Optuna design |
| `/docs/Research-2.md` | Architecture research | IBKR API patterns, combo orders, ML accuracy expectations, FinBERT, options flow, QuestDB design |
| `/docs/SetupChecklist.md` | Operator setup guide | API key acquisition, .env template, pre-live checklist, cost summary |

**Cross-reference the relevant docs when building ANY component.** The Master Blueprint has exact entry/exit criteria for every strategy. The Research docs have implementation details you'll need.

### 2. Review Existing Code First

Before modifying any file:

1. **Read the file** you're about to change — understand what's there before adding to it
2. **Read related files** — check imports, dependencies, and callers
3. **Check logs** — if something is broken, `logs/` and Docker container logs are your first stop
4. **Run the existing tests** — `uv run pytest tests/` to understand what's covered
5. **Check git status** — know what's changed and what's committed

### 3. Understand the Current State

This project has completed its initial build. The foundation is laid. You are now **maintaining and improving** a substantial codebase, not building from scratch. Respect what exists.

---

## CURRENT PROJECT STRUCTURE

This is the **actual, verified** project structure as of the last audit. Every file listed here exists in the repository.

```
titan/
├── CLAUDE.md                              # THIS FILE — project instructions
├── docker-compose.yml                     # Full service orchestration (7 services)
├── Dockerfile                             # Python 3.12 + uv application image
├── .env.example                           # Environment variable template
├── .gitignore
├── pyproject.toml                         # Python project config (uv)
├── uv.lock                                # Dependency lock file
├── README.md
│
├── docs/                                  # Project specification & research
│   ├── ClaudeCodePrompt.md                # Original build prompt
│   ├── MasterBluePrint.md                 # Complete product spec (AUTHORITATIVE)
│   ├── Research-1.md                      # Infrastructure & advanced topics research
│   ├── Research-2.md                      # Architecture & integration research
│   └── SetupChecklist.md                  # Operator setup & API key guide
│
├── config/
│   ├── settings.py                        # Pydantic BaseSettings (all config from env)
│   ├── strategies.yaml                    # Strategy parameters (all 10 strategies)
│   ├── tickers.yaml                       # 40-ticker universe with sector grouping
│   ├── risk_limits.yaml                   # Risk management parameters
│   ├── grafana/
│   │   ├── grafana.ini
│   │   └── provisioning/
│   │       ├── dashboards/titan.json      # Main trading dashboard
│   │       └── datasources/datasources.yaml
│   └── prometheus/
│       └── prometheus.yml                 # Prometheus scrape config
│
├── src/
│   ├── __init__.py
│   ├── main.py                            # Application entry point & lifecycle
│   │
│   ├── broker/                            # IBKR integration layer
│   │   ├── __init__.py
│   │   ├── gateway.py                     # ib_async connection manager
│   │   ├── market_data.py                 # Real-time data streaming & options chains
│   │   ├── orders.py                      # Order execution (combo/spread orders)
│   │   ├── account.py                     # Account state, positions, P&L tracking
│   │   └── contracts.py                   # Contract builders (options, combos)
│   │
│   ├── strategies/                        # Options strategy implementations
│   │   ├── __init__.py
│   │   ├── base.py                        # Abstract base strategy class
│   │   ├── bull_call_spread.py
│   │   ├── bull_put_spread.py
│   │   ├── iron_condor.py
│   │   ├── calendar_spread.py
│   │   ├── diagonal_spread.py
│   │   ├── broken_wing_butterfly.py
│   │   ├── short_strangle.py
│   │   ├── pmcc.py                        # Poor Man's Covered Call
│   │   ├── ratio_spread.py
│   │   ├── long_straddle.py
│   │   └── selector.py                    # Regime-based strategy selection engine
│   │
│   ├── signals/                           # Signal generation engine
│   │   ├── __init__.py
│   │   ├── ensemble.py                    # XGBoost meta-learner + isotonic calibration
│   │   ├── technical.py                   # 120+ technical indicator features
│   │   ├── sentiment.py                   # FinBERT sentiment analysis
│   │   ├── options_flow.py                # Unusual activity detection
│   │   ├── regime.py                      # GaussianHMM regime detection
│   │   ├── gex.py                         # Gamma Exposure calculation
│   │   ├── insider.py                     # SEC EDGAR Form 4 cluster detection
│   │   ├── cross_asset.py                 # VIX, yields, DXY, HY OAS signals
│   │   └── vrp.py                         # Volatility Risk Premium calculation
│   │
│   ├── risk/                              # Risk management layer
│   │   ├── __init__.py
│   │   ├── manager.py                     # Central risk management engine
│   │   ├── position_sizer.py              # Kelly criterion + regime-adjusted sizing
│   │   ├── circuit_breakers.py            # Automated drawdown circuit breakers
│   │   ├── portfolio_greeks.py            # Portfolio-level Greeks aggregation
│   │   ├── correlation.py                 # Rolling correlation monitoring
│   │   ├── event_calendar.py              # Earnings, FOMC, CPI avoidance
│   │   └── tail_risk.py                   # SKEW, VVIX, composite tail score
│   │
│   ├── ai/                                # Claude AI multi-agent system
│   │   ├── __init__.py
│   │   ├── agents.py                      # LangGraph state machine orchestrator
│   │   ├── analysis_agent.py              # Trade analysis with extended thinking
│   │   ├── risk_agent.py                  # Risk evaluation with VETO power
│   │   ├── execution_agent.py             # Order translation & fill monitoring
│   │   ├── journal_agent.py               # Post-trade analysis & FinMem learning
│   │   ├── prompts.py                     # All system prompts (cacheable)
│   │   └── memory.py                      # FinMem layered memory system
│   │
│   ├── ml/                                # Machine learning pipeline
│   │   ├── __init__.py
│   │   ├── trainer.py                     # Walk-forward training pipeline
│   │   ├── features.py                    # Feature engineering pipeline
│   │   ├── calibration.py                 # Isotonic regression calibration
│   │   ├── optimizer.py                   # Optuna walk-forward optimization
│   │   ├── online.py                      # River online learning + ADWIN drift
│   │   ├── rl_agent.py                    # SAC/PPO position management
│   │   └── backtest.py                    # Options backtesting engine
│   │
│   ├── data/                              # External data source clients
│   │   ├── __init__.py
│   │   ├── polygon.py                     # Polygon.io API client
│   │   ├── unusual_whales.py              # Unusual Whales API client
│   │   ├── finnhub.py                     # Finnhub API client (news, calendar)
│   │   ├── quiver.py                      # Quiver Quantitative API client
│   │   ├── fred.py                        # FRED API client (macro data)
│   │   ├── sec_edgar.py                   # SEC EDGAR Form 4 parser
│   │   ├── questdb.py                     # QuestDB writer/reader
│   │   └── cache.py                       # Redis-based data caching layer
│   │
│   ├── notifications/                     # Alerting & reporting
│   │   ├── __init__.py
│   │   ├── telegram.py                    # Telegram bot for notifications
│   │   ├── twilio_sms.py                  # Twilio SMS for critical alerts
│   │   └── reporter.py                    # QuantStats HTML report generator
│   │
│   └── utils/                             # Shared utilities
│       ├── __init__.py
│       ├── logging.py                     # Structured logging (JSON via structlog)
│       ├── metrics.py                     # Prometheus metrics definitions
│       ├── scheduling.py                  # APScheduler task scheduling
│       └── helpers.py                     # Common utilities
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py                        # pytest configuration & shared fixtures
│   ├── test_broker/
│   │   └── __init__.py
│   ├── test_strategies/
│   │   └── __init__.py
│   ├── test_signals/
│   │   └── __init__.py
│   ├── test_risk/
│   │   └── __init__.py
│   └── test_ml/
│       └── __init__.py
│
├── scripts/
│   └── init_db.sql                        # PostgreSQL schema initialization
│
├── models/                                # Trained ML model artifacts
│   └── .gitkeep
│
├── data/                                  # Local data cache
│   └── .gitkeep
│
└── logs/                                  # Application logs (mounted volume)
```

---

## SYSTEM ARCHITECTURE OVERVIEW

### Service Topology (Docker Compose)

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Docker Compose Stack                         │
│                                                                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐  │
│  │  IB Gateway   │  │    Redis     │  │       PostgreSQL         │  │
│  │  (gnzsnz)     │  │  7-alpine    │  │         16              │  │
│  │  :4001/:4002  │  │  :6379       │  │  :5432                  │  │
│  │  :5900 (VNC)  │  │  512MB LRU   │  │  trades, circuit_breaker│  │
│  └──────┬───────┘  └──────┬───────┘  │  agent_decisions, models │  │
│         │                  │          └──────────┬───────────────┘  │
│         │                  │                     │                  │
│  ┌──────┴──────────────────┴─────────────────────┴───────────┐     │
│  │                      TITAN BOT                             │     │
│  │  src/main.py → broker, strategies, signals, risk, ai, ml  │     │
│  │  FastAPI health endpoint :8080                             │     │
│  │  Prometheus metrics :8080/metrics                          │     │
│  └──────────────────────────┬────────────────────────────────┘     │
│                              │                                      │
│  ┌──────────────┐  ┌────────┴─────┐  ┌──────────────┐             │
│  │   QuestDB     │  │  Prometheus  │  │   Grafana    │             │
│  │  :9000 (HTTP) │  │  :9090       │  │  :3000       │             │
│  │  :8812 (PG)   │  │              │  │              │             │
│  │  :9009 (ILP)  │  │              │  │              │             │
│  │  time-series  │  │              │  │              │             │
│  └──────────────┘  └──────────────┘  └──────────────┘             │
└─────────────────────────────────────────────────────────────────────┘
```

### Data Flow

```
Market Data (IBKR)  ──→  Signals Engine  ──→  Strategy Selector  ──→  AI Analysis Agent
                                                                            │
External APIs ─────────→  Signal Streams:                                   ▼
  Polygon.io                1. Technical (50-60%)                     AI Risk Agent
  Unusual Whales            2. Sentiment (15-20%)                     (VETO power)
  Finnhub                   3. Options Flow (20-25%)                      │
  Quiver                    4. Regime Context (5-10%)                     ▼
  FRED                           │                                  AI Execution Agent
  SEC EDGAR                      ▼                                        │
                         XGBoost Meta-Learner                             ▼
                         (isotonic calibration)                    IBKR Combo Orders
                                │                                         │
                                ▼                                         ▼
                         Confidence Score                          Position Monitoring
                         (threshold: 0.78)                                │
                                                                          ▼
                                                                  AI Journal Agent
                                                                  (end-of-day review)
```

### Agent Pipeline

```
Analysis Agent (Claude + extended thinking)
    │
    ▼ TradeProposal {ticker, strategy, direction, confidence, parameters, reasoning}
    │
Risk Agent (hard limits + AI evaluation)
    │
    ├── APPROVED → Execution Agent
    ├── MODIFIED → Execution Agent (with adjusted parameters)
    └── REJECTED → Log reason, notify, done
    │
    ▼
Execution Agent (translate to IBKR combo order, monitor fills)
    │
    ▼
Position Monitoring (every 15 min: check exit criteria)
    │
    ▼
Journal Agent (end-of-day batch review, FinMem update)
```

---

## TRADING SYSTEM SPECIFICATIONS

### Account Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Account Size | $150,000 | Target: $200K by June 2026 |
| Max Drawdown | 15% ($22,500) | EMERGENCY STOP trigger |
| Per-Trade Risk | 2% ($3,000) | Max loss per position |
| Concurrent Positions | 5–8 | Max 8 open at once |
| Per-Ticker Exposure | 25–30% max | Sector concentration limit |
| Confidence Threshold | 0.78 | Minimum ML ensemble score to trade |
| Trading Frequency | 3–5 trades/week | ~60 trades/quarter |

### The 10-Strategy Arsenal

Each strategy has precise parameters defined in `config/strategies.yaml` and detailed in `/docs/MasterBluePrint.md`:

| # | Strategy | Regime | IV Rank | DTE | Key Parameters |
|---|----------|--------|---------|-----|----------------|
| 1 | Bull Call Spread | Low vol trending | 20–50% | 30–45 | Long 0.55–0.60Δ, Short 0.25–0.30Δ |
| 2 | Bull Put Spread | Uptrend + high IV | 40–70% | 45 | Sell 16–20Δ, Buy 5–10Δ, 15–20% credit |
| 3 | Iron Condor | Range-bound | 50–70% | 30–60 | 15–20Δ shorts, 5–10pt wings |
| 4 | Calendar Spread | Low IV | <30% | Front: 30, Back: 60 | ATM strikes, positive vega |
| 5 | Diagonal Spread | Mild trend + IV | 30–50% | Front: 30, Back: 60–90 | OTM front, ITM back |
| 6 | Broken Wing Butterfly | Range/mild trend | 30–50% | 21 | Deltas 32/28/20, 10–15% credit |
| 7 | Short Strangle | Range + high IV | >50% | 30–45 | 16Δ calls, 16Δ puts |
| 8 | PMCC | Strong uptrend | 20–40% | Long: 180+, Short: 30–45 | Long 0.70Δ LEAPS |
| 9 | Ratio Spread | Directional + IV | >40% | 30–45 | Buy 1 ATM, Sell 2 OTM |
| 10 | Long Straddle | Pre-catalyst + low IV | <30% | 30–45 | ATM calls + ATM puts |

### Four-Regime Classification

Detected via GaussianHMM (3 states) trained on rolling 4-year window of daily returns, realized vol, and VIX level:

| Regime | VIX | ADX | Strategies Favored |
|--------|-----|-----|--------------------|
| Low Vol Trending | <18 | >25 | Bull Call Spread, PMCC, Diagonal |
| High Vol Trending | 18–35 | >25 | Bull Put Spread, Ratio Spread |
| Range-Bound High IV | 18–35 | <20 | Iron Condor, Short Strangle, BWB |
| Crisis | >35 | any | Cash, Long Straddle only, reduce exposure |

### Four-Layer Risk Framework

| Layer | Scope | Controls |
|-------|-------|----------|
| **Layer 1** | Per-Trade | 2% max loss ($3,000), defined-risk only, no naked short options |
| **Layer 2** | Portfolio | Max 8 positions, 25–30% per ticker, ±$15K delta, <$5K vega |
| **Layer 3** | Circuit Breakers | 2% daily → halt, 5% weekly → 50% reduce, 10% monthly → full stop, 15% → EMERGENCY |
| **Layer 4** | Tail Risk | SKEW/VVIX composite, earnings/FOMC avoidance, 1–3% quarterly hedge budget |

### Circuit Breaker Recovery Ladder

After a circuit breaker fires, recovery is **graduated**, not instant:

1. **Stage 1 (50%)**: Trade at half normal size
2. **Stage 2 (75%)**: After 3 consecutive winners at Stage 1
3. **Stage 3 (100%)**: After 3 consecutive winners at Stage 2
4. **Full Reset**: After 2-week cooling period with positive P&L at each stage

---

## SIGNAL ARCHITECTURE

### Four Signal Streams → Ensemble

| Stream | Weight | Source | Implementation |
|--------|--------|--------|----------------|
| **Technical/ML** | 50–60% | XGBoost + LightGBM + CatBoost | `src/signals/technical.py` → `src/ml/trainer.py` |
| **Sentiment** | 15–20% | FinBERT on Finnhub news, StockTwits | `src/signals/sentiment.py` |
| **Options Flow** | 20–25% | Unusual Whales: volume/OI >1.25, sweeps, blocks >$500K | `src/signals/options_flow.py` |
| **Regime Context** | 5–10% | HMM state, VIX term structure, insider clusters, GEX | `src/signals/regime.py`, `src/signals/gex.py`, `src/signals/insider.py` |

### Alpha Signals Ranked by Evidence

| Tier | Signal | Evidence | Annual Alpha |
|------|--------|----------|--------------|
| **Tier 1 (Proven)** | VRP Harvesting | IV overestimates RV 85–90% of time | 3–5 vol points edge |
| **Tier 1 (Proven)** | Insider Cluster Buying | Academic: 3+ insiders in 30 days | 22–32% annualized |
| **Tier 1 (Proven)** | GEX Analysis | Mechanical support/resistance from dealer hedging | Regime-dependent |
| **Tier 2 (Filters)** | VIX Term Structure | Contango/backwardation | Market timing |
| **Tier 2 (Filters)** | Cross-Asset | HY OAS, copper/gold, DXY | Risk-on/risk-off |
| **Tier 3 (Marginal)** | Dark Pool | Large block trades | Confirmation only |

### ML Accuracy Expectations

Be realistic — these are researched baselines from `/docs/Research-2.md`:

- **Unfiltered directional accuracy**: 53–60% (this is normal, not a bug)
- **Confidence-filtered (>0.78)**: 70–85% accuracy at ~12% coverage
- **Goal**: High selectivity, not high coverage. 3–5 trades/week is the target.

---

## CODING STANDARDS

### Python Style (Enforced Everywhere)

- **Type hints** on every function signature and return type
- **Pydantic models** for all data structures (trade signals, orders, positions, API responses)
- **Async/await** for all I/O operations (IBKR, databases, APIs, Redis)
- **Structured logging** via `structlog` — JSON format with timestamp, level, component, message
- **Error handling**: specific exceptions, never bare `except:`. Use `tenacity` for retries with exponential backoff
- **No global state** — dependency injection via constructor parameters
- **Docstrings** on every public method — describe what it does, parameters, returns
- **Constants** in `UPPER_CASE`, never magic numbers in code
- **Configuration** always from environment (`.env`) or YAML — never hardcoded values

### Architecture Patterns

- **Event-driven**: Redis Pub/Sub for market data distribution, Redis Streams for order events
- **Repository pattern** for database access (separate data access from business logic)
- **Strategy pattern** for the 10 options strategies (common interface via `base.py`)
- **Observer pattern** for position monitoring (strategies observe their positions)
- **Circuit breaker pattern** for external API calls (Anthropic, Polygon, Finnhub)

### Testing Requirements

- Unit tests for all strategy logic (entry criteria, exit rules, sizing)
- Integration tests for IBKR connectivity (paper account)
- Mock external APIs in tests (`httpx` mock, `respx`, or `pytest-httpx`)
- Minimum coverage targets: strategy logic 90%, risk logic 95%
- Run tests: `uv run pytest tests/ -v --tb=short`

---

## KEY TECHNICAL SPECIFICATIONS

### IB Gateway / ib_async

- **ib_async** is the successor to `ib_insync` — always use `ib_async`, not the old library
- **ONE session per IBKR username** — never log into Client Portal or TWS while bot is running
- **Daily reset**: IB Gateway restarts 00:15–01:45 ET — bot must handle automatic reconnection
- **Rate limit**: 50 messages/second. Batch requests, use `asyncio.sleep(0.05)` between rapid calls
- **Max 100 concurrent streaming data lines** (increases with commissions/equity)
- **Options chain pipeline**: `reqSecDefOptParams()` (no rate limit) → filter strikes (±20 points) → `qualifyContracts()` → `reqMktData()` with Greeks on tick types 10–13
- **Combo/BAG orders**: set `NonGuaranteed = "0"` for guaranteed fills, **always use limit orders**
- **Error codes**: 502/504 (not connected), 1100 (connectivity lost), 1102 (restored), 2104/2106/2158 (data farm connections)
- **Order Efficiency Ratio**: must stay below 20:1 (submitted / filled)

### Claude API

- **Prompt caching**: system prompts are static — mark with `cache_control: {"type": "ephemeral"}` for 90% cost savings
- **Extended thinking**: minimum 1024 tokens, set budget via `thinking.budget_tokens` (Analysis: 16K, Risk: 8K)
- **Batch API**: use `/v1/messages/batches` for Journal Agent — 50% cost discount
- **Error handling**: exponential backoff for 429 (rate limit) and 529 (overloaded)
- **Fallback**: if Claude API is unreachable for >2 minutes, fall back to pure ML signal-based trading
- **Cost estimate**: ~$0.01/call using Sonnet + prompt caching

### Options Formulas (Get These Right)

```
IV Rank = (Current_IV − 52wk_Low_IV) / (52wk_High_IV − 52wk_Low_IV)

IV Percentile = % of days in past year where IV was below current level

GEX = Σ(call_OI × call_gamma × 100 × spot) − Σ(put_OI × put_gamma × 100 × spot)

Bull Call Spread Max Loss = Net Debit Paid
Bull Call Spread Max Profit = (Long Strike − Short Strike) − Net Debit

Iron Condor Max Loss = Wider Wing Width − Net Credit Received
Iron Condor Max Profit = Net Credit Received

Kelly Fraction = (p × b − q) / b
    where p = win probability, q = 1−p, b = avg_win / avg_loss
    USE FRACTIONAL KELLY: f* = Kelly / 4 (quarter-Kelly for safety)
```

### Database Design

**PostgreSQL** (relational — trades, state, models):
- Schema defined in `scripts/init_db.sql`
- Tables: `trades`, `trade_legs`, `account_snapshots`, `circuit_breaker_state`, `model_versions`, `agent_decisions`
- All state that must survive restarts lives here

**QuestDB** (time-series — market data, signals):
- Tables: `market_ticks`, `gex_levels`, `signal_scores`
- Partitioned by DAY for efficient time-range queries
- 12–36x faster than InfluxDB for time-series workloads

**Redis** (cache + messaging):
- Data caching with TTL (market data, API responses)
- Pub/Sub for real-time market data distribution
- Streams for order event processing
- 512MB max memory with `allkeys-lru` eviction

---

## APPLICATION LIFECYCLE

```
STARTUP:
  → Load config from .env + YAML files
  → Connect databases (PostgreSQL, QuestDB, Redis)
  → Connect IB Gateway (retry with exponential backoff until connected)
  → Load trained ML models (or train from scratch if /models/ is empty)
  → Initialize regime detector, signal generators, strategy selector
  → Initialize Claude AI agents (with fallback to pure ML if API unavailable)
  → Start APScheduler with all scheduled tasks
  → Begin market data streaming
  → Send "TITAN ONLINE" to Telegram

MARKET HOURS (9:30 AM – 4:00 PM ET):
  → 9:35 AM: Full universe scan (regime + signals + strategy selection)
  → Every 15 min: Check open positions for exit criteria
  → 11:30 AM, 1:30 PM, 3:30 PM: Intraday opportunity scans
  → Continuous: Circuit breaker monitoring, risk metric updates, Prometheus export

AFTER HOURS:
  → 4:15 PM: Journal Agent reviews all trades (Batch API for 50% discount)
  → 4:30 PM: Daily P&L summary to Telegram
  → 5:00 PM: Update signal databases, cache cleanup
  → Saturday 6:00 AM: Weekly model retraining + Optuna optimization

SHUTDOWN:
  → Cancel all pending orders (DO NOT close open positions)
  → Disconnect from IB Gateway cleanly
  → Flush all database writes
  → Send "TITAN OFFLINE" to Telegram
```

---

## ENVIRONMENT VARIABLES

All configuration is loaded via `config/settings.py` (Pydantic BaseSettings). See `.env.example` for the full template. Critical variables:

```bash
# IBKR (CRITICAL — connects to real broker)
IBKR_USERNAME, IBKR_PASSWORD, IBKR_TRADING_MODE (paper|live), IBKR_GATEWAY_PORT (4001=live, 4002=paper)

# Databases
POSTGRES_HOST, POSTGRES_PORT, POSTGRES_DB, POSTGRES_USER, POSTGRES_PASSWORD
QUESTDB_HOST, QUESTDB_HTTP_PORT, QUESTDB_PG_PORT
REDIS_HOST, REDIS_PORT

# API Keys (each from a different provider)
ANTHROPIC_API_KEY, POLYGON_API_KEY, UNUSUAL_WHALES_API_KEY, FINNHUB_API_KEY, QUIVER_API_KEY, FRED_API_KEY

# Notifications
TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_FROM_NUMBER, TWILIO_TO_NUMBER

# Trading Parameters
ACCOUNT_SIZE=150000, MAX_DRAWDOWN_PCT=0.15, PER_TRADE_RISK_PCT=0.02, MAX_CONCURRENT_POSITIONS=8, CONFIDENCE_THRESHOLD=0.78

# Claude AI
CLAUDE_MODEL=claude-sonnet-4-6, CLAUDE_ANALYSIS_THINKING_BUDGET=16384, CLAUDE_RISK_THINKING_BUDGET=8192
```

---

## DOCKER COMPOSE

The full `docker-compose.yml` orchestrates 7 services:

| Service | Image | Ports | Purpose |
|---------|-------|-------|---------|
| `ib-gateway` | `ghcr.io/gnzsnz/ib-gateway:latest` | 4001, 4002, 5900 | IBKR API + VNC debug |
| `redis` | `redis:7-alpine` | 6379 | Cache + messaging |
| `postgres` | `postgres:16` | 5432 | Relational data |
| `questdb` | `questdb/questdb:latest` | 9000, 8812, 9009 | Time-series data |
| `titan` | Built from `Dockerfile` | 8080 | The trading bot |
| `prometheus` | `prom/prometheus:latest` | 9090 | Metrics collection |
| `grafana` | `grafana/grafana:latest` | 3000 | Dashboards |

All ports are bound to `127.0.0.1` (localhost only) for security. The titan service waits for all dependencies to be healthy before starting.

---

## OPERATIONAL COSTS

| Service | Monthly Cost | Notes |
|---------|-------------|-------|
| Polygon.io | $199 | Advanced tier (real-time, options, dark pool) |
| Unusual Whales | $50–75 | API add-on required |
| Claude API | $50–100 | With prompt caching (90% savings) |
| Quiver Quantitative | $10–25 | Congressional, insider, lobbying data |
| IBKR Data | $6–16 | Often waived with commissions |
| Twilio | ~$5 | SMS alerts only |
| VPS (future) | $60–100 | QuantVPS NY or Hetzner Ashburn |
| **Total** | **~$380–520** | **ROI: 5% monthly = $7,500 vs ~$450 costs** |

---

## DEBUGGING & TROUBLESHOOTING

### Check Logs First

```bash
# Application logs (if running locally)
tail -f logs/titan.log

# Docker container logs
docker compose logs -f titan
docker compose logs -f ib-gateway
docker compose logs -f postgres

# Check all service health
docker compose ps

# Redis connectivity
docker compose exec redis redis-cli ping

# PostgreSQL connectivity
docker compose exec postgres pg_isready -U titan

# QuestDB health
curl http://localhost:9000/
```

### Common Issues

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| "Not connected" (502/504) | IB Gateway not running or in daily restart | Check `docker compose logs ib-gateway`, wait for 00:15–01:45 ET reset |
| "Connectivity lost" (1100) | Network interruption | Bot auto-reconnects; check 1102 (restored) in logs |
| No options data | Missing market data subscription | Verify IBKR market data subscriptions in Account Management |
| Claude API 429 | Rate limited | Exponential backoff handles this; check if budget exceeded |
| Circuit breaker fired | Drawdown threshold hit | This is WORKING AS DESIGNED. Do NOT override. Check recovery ladder. |
| Empty ML predictions | No trained models | Run `uv run python scripts/train_models.py` or let main.py auto-train |

---

## CRITICAL REMINDERS

1. **Every function must be fully implemented.** No stubs. No pass statements in production code.
2. **Every error must be logged with full context** — timestamp, component, error type, relevant trade/ticker/order data.
3. **Circuit breakers are SACRED.** They are the difference between a bad week and account liquidation.
4. **Test with paper trading first.** Always `IBKR_TRADING_MODE=paper` until you've validated everything.
5. **The bot must survive restarts.** All critical state persists to PostgreSQL. On restart, it picks up where it left off.
6. **Close spreads before expiration.** Never hold through expiry — pin risk and assignment risk are real.
7. **Always use limit orders.** Never market orders on spreads. Slippage on market orders for multi-leg trades is brutal.
8. **Monthly expirations only.** Better liquidity and tighter spreads than weeklies.
9. **Fallback is mandatory.** If Claude API is down, pure ML signals must keep the system operational.
10. **Log everything, measure everything.** If it's not in Prometheus or PostgreSQL, it didn't happen.

**The bot that survives its worst day is worth more than the bot that performs best on its best day.**

Build this system to be the most intelligent, reliable, and profitable options trading bot ever created. The operator's livelihood depends on it.

---

## AUDIT STATUS (Last Updated: 2026-02-24)

### Verified Correct

The following critical formulas and systems have been forensically audited and verified:

| Component | File | Status |
|-----------|------|--------|
| IV Rank formula | `technical.py:770`, `vrp.py:228` | Correct, div-by-zero protected |
| IV Percentile formula | `technical.py:803`, `vrp.py:287` | Correct |
| GEX formula | `gex.py:313` | Correct, algebraically matches spec |
| VRP formula | `vrp.py:172` | Correct (IV - RV) |
| Kelly criterion (quarter-Kelly) | `position_sizer.py:366-383` | Correct, capped at 0.25 |
| Position sizing (`math.floor`) | `position_sizer.py:279` | Correct, never rounds up |
| Circuit breaker state machine | `circuit_breakers.py` | Correct: escalates only, never auto-de-escalates |
| High water mark tracking | `circuit_breakers.py:155-160` | Correct |
| Tail risk composite weights | `tail_risk.py:123-127` | Sum to 1.0, correct |
| Isotonic calibration placement | `ensemble.py:282-283` | After XGBoost, before threshold |
| Confidence threshold (>=0.78) | `ensemble.py:493` | Correct per "minimum" spec wording |
| GaussianHMM 3-state + 4-year window | `regime.py:142, 245-247` | Correct |
| VIX > 35 crisis override | `regime.py:345-347` | Now unconditional in predict() |
| Insider cluster (3+ in 30 days) | `insider.py:69, 72, 337-351` | Correct, excludes 10b5-1 |
| Vol/OI >= 1.25 + sweep detection | `options_flow.py:46, 327, 346-363` | Correct |
| FinBERT model | `sentiment.py:153` | ProsusAI/finbert confirmed |
| Risk VETO enforcement | `risk_agent.py:788-919` | No bypass path exists |
| LangGraph pipeline | `agents.py:734` | START->analyze->risk->(execute or reject) |
| Prompt caching on all agents | All agent files | cache_control ephemeral confirmed |
| ML fallback (2-min timeout) | `agents.py:65, 852-918` | Correct implementation |
| Journal Agent batch API | `journal_agent.py:301-430` | 50% discount path confirmed |
| Walk-forward training | `trainer.py:205-350` | Genuine temporal splits with embargo |
| ADWIN drift detection | `online.py:815` | Wired into online learning loop |
| Backtester slippage | `backtest.py:1005-1100` | 15% of spread (within 10-25% spec) |
| Earnings exclusion window | `event_calendar.py:493-516` | 5 days before through 1 day after |

### Previously Reported Limitations (Now Resolved)

All three limitations from the Phase 1 audit have been addressed:

1. ~~**`min_recovery_days` loaded but never enforced**~~ — **RESOLVED (2026-02-24).** Enforced in both `is_trading_allowed()` (`circuit_breakers.py:257-276`) and `_check_recovery_advance()` (`circuit_breakers.py:594-604`). HALT/EMERGENCY levels block trading until the cooling-off period elapses.

2. ~~**Market holidays hardcoded for 2025-2026 only**~~ — **RESOLVED (2026-02-24).** Replaced with dynamic `compute_nyse_holidays(year)` in `helpers.py:104-148` using algorithmic computation (Easter, nth-weekday, NYSE observation rules). Works for any year, LRU-cached.

3. ~~**Test coverage is 0% for actual `src/` modules**~~ — **RESOLVED (2026-02-24).** 657 tests now cover core `src/` modules: circuit breakers, position sizer, ensemble signals, regime detection, base strategy, helpers, broker contracts, market data routing, and ML feature engineering.

### Fixes Applied in This Audit (2026-02-24)

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
| Order status polling | `execution_agent.py:1008-1095` | Implemented `_poll_order_status()`: queries OrderManager for live Trade objects, maps ib_async status strings to canonical states (FILLED/CANCELLED/REJECTED/PARTIAL/PENDING), extracts fill price/qty/time |
| Orchestrator → OrderManager | `agents.py:607, 697` | Added `order_manager` parameter pass-through from orchestrator to ExecutionAgent |

### Fixes Applied (2026-02-24 — Intelligence Layer)

| Fix | File | Description |
|-----|------|-------------|
| VIX Index contract support | `contracts.py` | Added `Index` import and `create_index()` method to `ContractFactory` (exchange default: CBOE) |
| Index routing in MarketDataManager | `market_data.py` | Added `_INDEX_SYMBOLS` frozenset (VIX, SPX, NDX, RUT, VVIX, etc.) and routing in `get_snapshot()`, `get_historical_bars()`, `get_historical_iv()` |
| `build_trade_features()` method | `features.py` | Implemented missing method for weekly retrain: extracts ml_confidence, one-hot regime/strategy, direction encoding, hold_days, entry_hour, day_of_week, log_price (~20 features) |
| Model path alignment | `main.py` | After `trainer.train()`, copies latest `titan_xgboost_ensemble_v*.json` to `ensemble_xgb.json` via `shutil.copy2` so hot-swap code finds the retrained model |
| Broker test coverage | `tests/test_broker/` | 26 tests for `ContractFactory` (create_stock, create_index, create_option, create_combo, SpreadLeg validation) + 35 tests for `MarketDataManager` (index routing, snapshot, historical bars, historical IV, Pydantic models, helpers) |
| ML features test coverage | `tests/test_ml/test_features.py` | 40 tests for `FeatureEngineer` (build_trade_features, build_feature_matrix, select_features, normalize, target variable, lagged features, correlated removal) |

### Verified Correct (Added 2026-02-24)

| Component | File | Status |
|-----------|------|--------|
| Index contract creation | `contracts.py:196-232` | Uses `ib_async.Index`, default exchange CBOE, qualifies via gateway |
| `_INDEX_SYMBOLS` routing | `market_data.py:57-59` | 9 CBOE indices routed to `create_index()` in all 3 methods |
| `build_trade_features()` | `features.py:110-200` | One-hot encoding, direction map, hold_days, log_price, NaN imputation |
| Model path copy | `main.py:_train_sync` | Globs versioned files, copies latest to canonical `ensemble_xgb.json` |
| `min_recovery_days` enforcement | `circuit_breakers.py:257-276, 594-604` | Blocks trading and recovery advance during cooling-off |
| Dynamic NYSE holidays | `helpers.py:104-148` | Algorithmic computation, works for any year |
| Test suite | `tests/` | 657 tests passing (101 new for broker, market data, ML features) |
