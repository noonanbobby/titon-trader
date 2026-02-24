# CLAUDE.md — Project Titan: AI-Powered Options Trading System

## YOU ARE BUILDING THE MOST ADVANCED RETAIL OPTIONS TRADING BOT EVER CREATED

This file contains complete instructions for building Project Titan — a fully automated, AI-driven options trading system for Interactive Brokers. Read this ENTIRE file before writing any code. Every section matters. No shortcuts. No placeholders. No TODO comments. Every component must be production-grade.

---

## PROJECT STRUCTURE

```
titan/
├── CLAUDE.md                          # This file
├── docker-compose.yml                 # Full service orchestration
├── docker-compose.dev.yml             # Development overrides
├── Dockerfile                         # Main Python application image
├── Dockerfile.gateway                 # IB Gateway image (extends gnzsnz)
├── .env.example                       # Template for environment variables
├── .gitignore
├── pyproject.toml                     # Python project config (use uv)
├── requirements.txt                   # Pinned dependencies
├── README.md
│
├── config/
│   ├── settings.py                    # Pydantic Settings (all config from env)
│   ├── strategies.yaml                # Strategy parameters (all 10 strategies)
│   ├── tickers.yaml                   # Universe definition
│   ├── risk_limits.yaml               # Risk management parameters
│   ├── grafana/
│   │   ├── provisioning/
│   │   │   ├── dashboards/
│   │   │   │   └── titan.json         # Main trading dashboard
│   │   │   └── datasources/
│   │   │       └── datasources.yaml   # PostgreSQL + QuestDB + Prometheus
│   │   └── grafana.ini
│   ├── prometheus/
│   │   └── prometheus.yml
│   └── ibc/
│       └── config.ini                 # IBC configuration for IB Gateway
│
├── src/
│   ├── __init__.py
│   ├── main.py                        # Application entry point and lifecycle
│   │
│   ├── broker/
│   │   ├── __init__.py
│   │   ├── gateway.py                 # IB Gateway connection manager (ib_async)
│   │   ├── market_data.py             # Real-time data streaming and options chains
│   │   ├── orders.py                  # Order execution (combo/spread orders)
│   │   ├── account.py                 # Account state, positions, P&L tracking
│   │   └── contracts.py               # Contract builders (options, combos)
│   │
│   ├── strategies/
│   │   ├── __init__.py
│   │   ├── base.py                    # Abstract base strategy class
│   │   ├── bull_call_spread.py
│   │   ├── bull_put_spread.py
│   │   ├── iron_condor.py
│   │   ├── calendar_spread.py
│   │   ├── diagonal_spread.py
│   │   ├── broken_wing_butterfly.py
│   │   ├── short_strangle.py
│   │   ├── pmcc.py
│   │   ├── ratio_spread.py
│   │   ├── long_straddle.py
│   │   └── selector.py                # Regime-based strategy selection engine
│   │
│   ├── signals/
│   │   ├── __init__.py
│   │   ├── ensemble.py                # Meta-learner combining all signal streams
│   │   ├── technical.py               # Technical indicator feature engineering
│   │   ├── sentiment.py               # FinBERT sentiment analysis
│   │   ├── options_flow.py            # Unusual activity detection
│   │   ├── regime.py                  # HMM regime detection
│   │   ├── gex.py                     # Gamma Exposure calculation
│   │   ├── insider.py                 # Form 4 insider cluster detection
│   │   ├── cross_asset.py             # VIX, yields, DXY, HY OAS signals
│   │   └── vrp.py                     # Volatility Risk Premium calculation
│   │
│   ├── risk/
│   │   ├── __init__.py
│   │   ├── manager.py                 # Central risk management engine
│   │   ├── position_sizer.py          # Kelly criterion + regime-adjusted sizing
│   │   ├── circuit_breakers.py        # Automated drawdown circuit breakers
│   │   ├── portfolio_greeks.py        # Portfolio-level Greeks aggregation
│   │   ├── correlation.py             # Rolling correlation monitoring
│   │   ├── event_calendar.py          # Earnings, FOMC, CPI avoidance
│   │   └── tail_risk.py               # SKEW, VVIX, composite tail score
│   │
│   ├── ai/
│   │   ├── __init__.py
│   │   ├── agents.py                  # LangGraph multi-agent orchestration
│   │   ├── analysis_agent.py          # Trade analysis with extended thinking
│   │   ├── risk_agent.py              # Risk evaluation with veto power
│   │   ├── execution_agent.py         # Order translation and monitoring
│   │   ├── journal_agent.py           # Post-trade analysis and learning
│   │   ├── prompts.py                 # All system prompts (cacheable)
│   │   └── memory.py                  # FinMem layered memory system
│   │
│   ├── ml/
│   │   ├── __init__.py
│   │   ├── trainer.py                 # Walk-forward training pipeline
│   │   ├── features.py                # Feature engineering pipeline
│   │   ├── calibration.py             # Isotonic regression calibration
│   │   ├── optimizer.py               # Optuna walk-forward optimization
│   │   ├── online.py                  # River online learning + ADWIN drift
│   │   ├── rl_agent.py                # SAC/PPO position management
│   │   └── backtest.py                # Options backtesting engine
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── polygon.py                 # Polygon.io API client
│   │   ├── unusual_whales.py          # Unusual Whales API client
│   │   ├── finnhub.py                 # Finnhub API client (news, calendar)
│   │   ├── quiver.py                  # Quiver Quantitative API client
│   │   ├── fred.py                    # FRED API client (macro data)
│   │   ├── sec_edgar.py               # SEC EDGAR Form 4 parser
│   │   ├── questdb.py                 # QuestDB writer/reader
│   │   └── cache.py                   # Redis-based data caching layer
│   │
│   ├── notifications/
│   │   ├── __init__.py
│   │   ├── telegram.py                # Telegram bot for notifications
│   │   ├── twilio_sms.py              # Twilio SMS for critical alerts
│   │   └── reporter.py                # QuantStats HTML report generator
│   │
│   └── utils/
│       ├── __init__.py
│       ├── logging.py                 # Structured logging (JSON format)
│       ├── metrics.py                 # Prometheus metrics definitions
│       ├── scheduling.py              # APScheduler task scheduling
│       └── helpers.py                 # Common utilities
│
├── tests/
│   ├── conftest.py
│   ├── test_broker/
│   ├── test_strategies/
│   ├── test_signals/
│   ├── test_risk/
│   └── test_ml/
│
├── scripts/
│   ├── setup_db.py                    # Database initialization
│   ├── seed_historical.py             # Download historical data for training
│   ├── train_models.py                # Run initial model training
│   ├── backtest.py                    # Run backtests
│   └── health_check.py               # System health verification
│
├── models/                            # Trained ML model artifacts
│   └── .gitkeep
│
└── data/                              # Local data cache
    └── .gitkeep
```

---

## CRITICAL DEPENDENCIES (requirements.txt)

```
# IBKR API
ib_async>=2.1.0

# Data & Analysis
pandas>=2.2.0
numpy>=1.26.0
pandas-ta>=0.3.14b
scipy>=1.12.0
py_vollib>=1.0.1
QuantLib-Python>=1.33

# Machine Learning
xgboost>=2.0.0
lightgbm>=4.3.0
catboost>=1.2.0
scikit-learn>=1.4.0
optuna>=4.7.0
optuna-dashboard>=0.15.0
stable-baselines3>=2.3.0
gymnasium>=0.29.0
river>=0.21.0
hmmlearn>=0.3.0
transformers>=4.37.0  # for FinBERT
torch>=2.2.0

# Claude AI
anthropic>=0.40.0
langgraph>=0.2.0

# Database
asyncpg>=0.29.0
psycopg2-binary>=2.9.9
redis>=5.0.0
questdb>=2.0.0

# Web & API
httpx>=0.27.0
aiohttp>=3.9.0
websockets>=12.0
fastapi>=0.110.0
uvicorn>=0.27.0

# Monitoring
prometheus_client>=0.20.0
quantstats>=0.0.62
streamlit>=1.30.0

# Notifications
python-telegram-bot>=21.0
twilio>=9.0.0

# Infrastructure
python-dotenv>=1.0.0
pydantic>=2.6.0
pydantic-settings>=2.1.0
pyyaml>=6.0.0
APScheduler>=3.10.0
structlog>=24.1.0
tenacity>=8.2.0
```

---

## ENVIRONMENT VARIABLES (.env)

```bash
# IBKR Configuration
IBKR_USERNAME=your_username
IBKR_PASSWORD=your_password
IBKR_TRADING_MODE=paper  # paper or live
IBKR_GATEWAY_PORT=4002   # 4001=live, 4002=paper
IBKR_CLIENT_ID=1

# Database
POSTGRES_HOST=postgres
POSTGRES_PORT=5432
POSTGRES_DB=titan
POSTGRES_USER=titan
POSTGRES_PASSWORD=generate_secure_password_here

QUESTDB_HOST=questdb
QUESTDB_HTTP_PORT=9000
QUESTDB_PG_PORT=8812

REDIS_HOST=redis
REDIS_PORT=6379

# API Keys
ANTHROPIC_API_KEY=sk-ant-...
POLYGON_API_KEY=your_polygon_key
UNUSUAL_WHALES_API_KEY=your_uw_key
FINNHUB_API_KEY=your_finnhub_key
QUIVER_API_KEY=your_quiver_key
FRED_API_KEY=your_fred_key

# Notifications
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
TWILIO_ACCOUNT_SID=your_sid
TWILIO_AUTH_TOKEN=your_token
TWILIO_FROM_NUMBER=+1234567890
TWILIO_TO_NUMBER=+1234567890

# Trading Parameters
ACCOUNT_SIZE=150000
MAX_DRAWDOWN_PCT=0.15
PER_TRADE_RISK_PCT=0.02
MAX_CONCURRENT_POSITIONS=8
CONFIDENCE_THRESHOLD=0.78

# Claude AI
CLAUDE_MODEL=claude-sonnet-4-5-20250929
CLAUDE_ANALYSIS_THINKING_BUDGET=8192
CLAUDE_RISK_THINKING_BUDGET=4096
```

---

## DOCKER-COMPOSE.YML

Build this EXACTLY as specified. All services must work together seamlessly:

```yaml
version: '3.8'

services:
  ib-gateway:
    image: ghcr.io/gnzsnz/ib-gateway:latest
    restart: always
    environment:
      TWS_USERID: ${IBKR_USERNAME}
      TWS_PASSWORD: ${IBKR_PASSWORD}
      TRADING_MODE: ${IBKR_TRADING_MODE:-paper}
      TWS_SETTINGS_PATH: /root/Jts
      TWOFA_TIMEOUT_ACTION: restart
      AUTO_RESTART_TIME: "23:45"
      RELOGIN_AFTER_2FA_TIMEOUT: "yes"
    ports:
      - "127.0.0.1:4001:4001"
      - "127.0.0.1:4002:4002"
      - "127.0.0.1:5900:5900"  # VNC for debugging
    volumes:
      - ib-gateway-settings:/root/Jts
    healthcheck:
      test: ["CMD", "nc", "-z", "localhost", "${IBKR_GATEWAY_PORT:-4002}"]
      interval: 10s
      timeout: 5s
      retries: 5
      start_period: 60s

  redis:
    image: redis:7-alpine
    restart: always
    command: redis-server --maxmemory 512mb --maxmemory-policy allkeys-lru
    ports:
      - "127.0.0.1:6379:6379"
    volumes:
      - redis-data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 3s
      retries: 3

  postgres:
    image: postgres:16
    restart: always
    environment:
      POSTGRES_DB: ${POSTGRES_DB:-titan}
      POSTGRES_USER: ${POSTGRES_USER:-titan}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
    ports:
      - "127.0.0.1:5432:5432"
    volumes:
      - postgres-data:/var/lib/postgresql/data
      - ./scripts/init_db.sql:/docker-entrypoint-initdb.d/init.sql
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${POSTGRES_USER:-titan}"]
      interval: 10s
      timeout: 3s
      retries: 3

  questdb:
    image: questdb/questdb:latest
    restart: always
    ports:
      - "127.0.0.1:9000:9000"   # HTTP/REST
      - "127.0.0.1:8812:8812"   # PostgreSQL wire protocol
      - "127.0.0.1:9009:9009"   # InfluxDB line protocol
    volumes:
      - questdb-data:/var/lib/questdb
    environment:
      QDB_HTTP_ENABLED: "true"
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9000/"]
      interval: 10s
      timeout: 3s
      retries: 3

  titan:
    build:
      context: .
      dockerfile: Dockerfile
    restart: unless-stopped
    depends_on:
      ib-gateway:
        condition: service_healthy
      redis:
        condition: service_healthy
      postgres:
        condition: service_healthy
      questdb:
        condition: service_healthy
    env_file: .env
    volumes:
      - ./config:/app/config
      - ./models:/app/models
      - ./data:/app/data
      - ./logs:/app/logs
    healthcheck:
      test: ["CMD", "python", "-c", "import httpx; httpx.get('http://localhost:8080/health')"]
      interval: 30s
      timeout: 10s
      retries: 3

  prometheus:
    image: prom/prometheus:latest
    restart: always
    ports:
      - "127.0.0.1:9090:9090"
    volumes:
      - ./config/prometheus/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus-data:/prometheus

  grafana:
    image: grafana/grafana:latest
    restart: always
    ports:
      - "127.0.0.1:3000:3000"
    environment:
      GF_SECURITY_ADMIN_PASSWORD: ${GRAFANA_PASSWORD:-admin}
      GF_INSTALL_PLUGINS: grafana-clock-panel
    volumes:
      - ./config/grafana/provisioning:/etc/grafana/provisioning
      - grafana-data:/var/lib/grafana
    depends_on:
      - prometheus
      - postgres
      - questdb

volumes:
  ib-gateway-settings:
  redis-data:
  postgres-data:
  questdb-data:
  prometheus-data:
  grafana-data:
```

---

## IMPLEMENTATION PRIORITIES — BUILD IN THIS EXACT ORDER

### Phase 1: Foundation (Day 1)
1. Create the full project structure above
2. Set up `pyproject.toml` and install dependencies
3. Create `config/settings.py` with Pydantic Settings loading all env vars
4. Create `docker-compose.yml` exactly as specified above
5. Build `src/broker/gateway.py` — connection manager using `ib_async`:
   - Async connection to IB Gateway on configurable port
   - Automatic reconnection with exponential backoff
   - Error handling for codes 502, 504, 1100, 1102
   - Connection health monitoring
   - Clean shutdown handling
6. Build `src/broker/market_data.py`:
   - Stream real-time quotes for ticker universe
   - Options chain retrieval: `reqSecDefOptParams()` → filter strikes → `qualifyContracts()` → `reqMktData()`
   - Greeks extraction from tick types 10–13
   - IV surface data collection
7. Build `src/broker/contracts.py`:
   - Stock contract builder
   - Option contract builder (with proper exchange, currency, multiplier)
   - Combo/BAG contract builder for multi-leg spreads
   - Validate all contracts via `qualifyContracts()`
8. Build `src/broker/orders.py`:
   - Limit order builder for spreads (net debit/credit)
   - Combo order builder for bull call spreads, iron condors, etc.
   - Order status monitoring via callbacks
   - Fill tracking and slippage calculation
9. Build `src/broker/account.py`:
   - Account summary retrieval (NetLiquidation, BuyingPower, ExcessLiquidity)
   - Position tracking with live P&L
   - Margin calculation for spread positions
10. Test: Connect to IB Gateway paper account, stream data, place a test bull call spread

### Phase 2: Strategy Engine (Day 2)
1. Build `src/strategies/base.py`:
   - Abstract base class with methods: `check_entry()`, `check_exit()`, `construct_order()`, `calculate_greeks()`, `calculate_max_loss()`, `calculate_max_profit()`
   - All strategies inherit from this
2. Implement ALL TEN STRATEGIES as specified in the Master Blueprint
   - Each strategy class must implement all abstract methods
   - Entry criteria are configurable via `config/strategies.yaml`
   - Exit rules are mechanical and automated — no discretion
3. Build `src/strategies/selector.py`:
   - Takes regime state, IV rank, trend strength, ML confidence as inputs
   - Filters eligible strategies per regime
   - Scores candidates by expected value
   - Returns ranked list of (strategy, ticker, parameters, score)
4. Build `src/risk/manager.py`:
   - Pre-trade risk checks (all Layer 1 + Layer 2 controls)
   - Position size calculation (fractional Kelly)
   - Portfolio exposure checks (sector, ticker, Greeks limits)
   - Returns approved/rejected/modified verdict
5. Build `src/risk/circuit_breakers.py`:
   - Track daily, weekly, monthly, and total P&L
   - Automated triggers at 2%, 5%, 10%, 15% thresholds
   - Recovery ladder implementation
   - Persist state to PostgreSQL (survives restarts)
6. Build `src/risk/event_calendar.py`:
   - Fetch earnings dates from Finnhub
   - Fetch FOMC, CPI, NFP dates from Finnhub economic calendar
   - Block entries within event exclusion windows

### Phase 3: Signal Generation (Day 3)
1. Build `src/signals/technical.py`:
   - Calculate all 120+ features using pandas-ta
   - IV Rank, IV Percentile (from IBKR historical vol data)
   - HV/IV ratio, Bollinger width, ATR percentile
   - Output: DataFrame with all features per ticker per timestamp
2. Build `src/signals/sentiment.py`:
   - Load FinBERT model (ProsusAI/finbert)
   - Batch process news from Finnhub every 15 minutes
   - 24-hour rolling sentiment score per ticker
   - StockTwits sentiment (contrarian-weighted for WSB-like content)
3. Build `src/signals/options_flow.py`:
   - Unusual Whales API integration
   - Detect: volume/OI > 1.25, sweeps, blocks > $500K
   - Multi-day consistency scoring (3+ days same direction)
   - Net directional premium flow
4. Build `src/signals/regime.py`:
   - GaussianHMM with 3 states from hmmlearn
   - Features: rolling 20-day returns, rolling 20-day realized vol, VIX level
   - Training: rolling 4-year window
   - Classification: low_vol_trend, high_vol_trend, range_bound, crisis
   - Backup: VIX threshold + ADX rules for redundancy
5. Build `src/signals/gex.py`:
   - Calculate per-strike gamma exposure from options chain OI + Greeks
   - Identify Call Wall, Put Wall, Volatility Trigger
   - Determine positive/negative GEX regime
6. Build `src/signals/insider.py`:
   - SEC EDGAR Form 4 XML parser (free, real-time)
   - Cluster detection: 3+ insiders buying within 30 days
   - Weight by seniority (CEO/CFO > VP > Director) and dollar amount
   - Filter out 10b5-1 plan trades
7. Build `src/signals/vrp.py`:
   - Calculate IV − RV spread (implied minus realized vol)
   - IV Rank and IV Percentile calculation
   - VRP regime: rich (IV >> RV), fair, cheap (IV << RV)
8. Build `src/signals/cross_asset.py`:
   - FRED API: 2Y/10Y spread, Fed funds rate, HY OAS (BAMLH0A0HYM2)
   - VIX term structure: VIX/VIX3M ratio
   - DXY, copper/gold ratio from Polygon
9. Build `src/signals/ensemble.py`:
   - Collect calibrated outputs from all 4 streams
   - XGBoost meta-learner
   - Isotonic regression calibration
   - Output: final confidence score 0.0–1.0
   - Trade signal only when score > CONFIDENCE_THRESHOLD (0.78)
10. Build `src/ml/features.py`:
    - Full feature engineering pipeline
    - Feature selection via importance thresholding (> 0.5%)
    - Standardization/normalization as needed
11. Build `src/ml/trainer.py`:
    - Walk-forward training loop
    - Purged k-fold cross-validation (5 folds, 5-day embargo)
    - Model serialization to /models directory
12. Build `src/ml/calibration.py`:
    - Isotonic regression calibration wrapper
    - Calibration curve visualization

### Phase 4: AI Agents (Day 4)
1. Build `src/ai/prompts.py`:
   - Analysis Agent system prompt (cacheable, detailed with all strategy rules)
   - Risk Agent system prompt (portfolio limits, circuit breaker rules)
   - Execution Agent system prompt (IBKR order types, combo construction)
   - Journal Agent system prompt (trade review criteria, pattern detection)
2. Build `src/ai/analysis_agent.py`:
   - Claude API call with extended thinking enabled
   - Input: ML scores, regime, GEX, sentiment, news headlines, options chain snapshot
   - Output: Structured JSON — {ticker, strategy, direction, confidence, parameters, reasoning}
   - Thinking budget: configurable via env var
3. Build `src/ai/risk_agent.py`:
   - Evaluates Analysis Agent proposals
   - Checks all risk limits
   - Returns: APPROVED, REJECTED (with reason), MODIFIED (with adjustments)
4. Build `src/ai/execution_agent.py`:
   - Translates approved proposals to IBKR combo orders
   - Monitors fill status
   - Handles partial fills and order modifications
5. Build `src/ai/journal_agent.py`:
   - Batch API calls (50% discount) for end-of-day
   - Reviews all closed trades
   - FinMem memory: short (5 trades), medium (30 trades), long (regime patterns)
   - Generates insights for Analysis Agent system prompt updates
6. Build `src/ai/agents.py`:
   - LangGraph state machine orchestrating all 4 agents
   - Conditional routing: Analysis → Risk (approve?) → Execution
   - Error handling and retries
   - Fallback: if Claude API unavailable, fall back to pure ML signals

### Phase 5: Infrastructure (Day 5)
1. Build `src/notifications/telegram.py`:
   - Trade entry/exit messages with P&L
   - Daily summary: trades, P&L, drawdown, regime
   - Circuit breaker alerts
   - Command handlers: /status, /positions, /kill (emergency stop)
2. Build `src/notifications/twilio_sms.py`:
   - SMS for: connectivity loss > 5 min, circuit breaker trigger, emergency stop
   - Rate limit: max 1 SMS per condition per hour
3. Build `src/notifications/reporter.py`:
   - Weekly QuantStats HTML tear sheet generation
   - Monthly PDF report
   - Send via Telegram as file attachment
4. Build `src/utils/metrics.py`:
   - Prometheus metrics: trade_count, win_rate, avg_pnl, drawdown_pct, api_latency, regime_state, confidence_score, positions_open
   - Expose on port 8080 via FastAPI
5. Build Grafana dashboard JSON:
   - P&L curve (PostgreSQL query)
   - Open positions table with Greeks
   - Regime state indicator
   - ML confidence distribution
   - Circuit breaker status
   - System health (CPU, memory, API latency from Prometheus)
6. Build `src/utils/scheduling.py`:
   - APScheduler-based task scheduling
   - Market open scan (9:35 AM ET)
   - Intraday scans (every 2 hours: 11:30, 1:30, 3:30)
   - Position exit checks (every 15 minutes during market hours)
   - End-of-day journal (4:15 PM ET)
   - Weekly model retraining (Saturday 6 AM ET)
   - Earnings/event calendar refresh (daily 8 AM ET)

### Phase 6: Advanced ML (Day 6)
1. Build `src/ml/optimizer.py`:
   - Optuna walk-forward optimization
   - Parameters: entry thresholds, DTE targets, delta targets, profit targets, stop loss levels
   - Multi-objective: maximize Sharpe + minimize max drawdown
   - PostgreSQL-backed study storage
2. Build `src/ml/online.py`:
   - River online learning for real-time model updates
   - ADWIN drift detection on strategy error rates
   - Trigger actions: log warning → increase LR → reset weights → full retrain
3. Build `src/ml/rl_agent.py`:
   - SAC via Stable-Baselines3
   - State: price, portfolio value, Greeks, IV, time to expiry, VIX, regime
   - Action: scale factor [-1, +1]
   - Reward: Sharpe over rolling 20-trade window
   - Training: offline on historical trade outcomes
4. Build `src/ml/backtest.py`:
   - Options-specific backtesting engine
   - Realistic fill modeling: mid price + 10–25% of spread as slippage
   - Multi-leg execution simulation
   - Walk-forward validation framework

---

## CODING STANDARDS — ENFORCE THESE EVERYWHERE

### Python Style
- **Type hints everywhere** — every function signature, every return type
- **Pydantic models** for all data structures (trade signals, orders, positions)
- **Async/await** for all I/O operations (IBKR, databases, APIs, Redis)
- **Structured logging** via structlog — JSON format with timestamp, level, component, message
- **Error handling**: specific exceptions, never bare `except:`. Use `tenacity` for retries
- **No global state** — dependency injection via constructor parameters
- **Docstrings** on every public method — describe what it does, parameters, returns
- **Constants** in UPPER_CASE, never magic numbers in code
- **Configuration** always from environment or YAML — never hardcoded

### Architecture Patterns
- **Event-driven**: Redis Pub/Sub for market data distribution, Redis Streams for order events
- **Repository pattern** for database access (separate data access from business logic)
- **Strategy pattern** for the 10 options strategies (common interface, different implementations)
- **Observer pattern** for position monitoring (strategies observe their positions)
- **Circuit breaker pattern** for external API calls (Anthropic, Polygon, Finnhub)

### Testing
- Unit tests for all strategy logic (entry criteria, exit rules, sizing)
- Integration tests for IBKR connectivity (paper account)
- Mock external APIs in tests (httpx mock)
- Minimum coverage: strategy logic 90%, risk logic 95%

---

## DATABASE SCHEMAS

### PostgreSQL Tables

```sql
-- Trades table
CREATE TABLE trades (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    ticker VARCHAR(10) NOT NULL,
    strategy VARCHAR(50) NOT NULL,
    direction VARCHAR(10) NOT NULL,  -- LONG, SHORT
    status VARCHAR(20) NOT NULL DEFAULT 'PENDING',  -- PENDING, OPEN, CLOSED, CANCELLED
    entry_time TIMESTAMPTZ,
    exit_time TIMESTAMPTZ,
    entry_price DECIMAL(10,4),
    exit_price DECIMAL(10,4),
    quantity INTEGER NOT NULL,
    max_profit DECIMAL(10,2),
    max_loss DECIMAL(10,2),
    realized_pnl DECIMAL(10,2),
    commission DECIMAL(10,2),
    ml_confidence DECIMAL(5,4),
    regime VARCHAR(30),
    entry_iv_rank DECIMAL(5,2),
    entry_reasoning TEXT,  -- Claude's analysis
    exit_reasoning TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Trade legs table (for multi-leg spreads)
CREATE TABLE trade_legs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    trade_id UUID REFERENCES trades(id),
    leg_type VARCHAR(10) NOT NULL,  -- BUY, SELL
    option_type VARCHAR(4) NOT NULL,  -- CALL, PUT
    strike DECIMAL(10,2) NOT NULL,
    expiry DATE NOT NULL,
    quantity INTEGER NOT NULL,
    fill_price DECIMAL(10,4),
    ib_order_id INTEGER,
    ib_con_id INTEGER,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Account snapshots
CREATE TABLE account_snapshots (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    net_liquidation DECIMAL(12,2),
    buying_power DECIMAL(12,2),
    excess_liquidity DECIMAL(12,2),
    realized_pnl_day DECIMAL(10,2),
    unrealized_pnl DECIMAL(10,2),
    total_positions INTEGER,
    regime VARCHAR(30)
);

-- Circuit breaker state
CREATE TABLE circuit_breaker_state (
    id SERIAL PRIMARY KEY,
    level VARCHAR(20) NOT NULL,  -- NORMAL, CAUTION, WARNING, HALT, EMERGENCY
    triggered_at TIMESTAMPTZ,
    daily_pnl DECIMAL(10,2),
    weekly_pnl DECIMAL(10,2),
    monthly_pnl DECIMAL(10,2),
    total_drawdown_pct DECIMAL(5,4),
    high_water_mark DECIMAL(12,2),
    recovery_stage INTEGER DEFAULT 0,  -- 0=normal, 1=50%, 2=75%, 3=100%
    consecutive_winners INTEGER DEFAULT 0,
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- ML model metadata
CREATE TABLE model_versions (
    id SERIAL PRIMARY KEY,
    model_name VARCHAR(50) NOT NULL,
    version INTEGER NOT NULL,
    trained_at TIMESTAMPTZ DEFAULT NOW(),
    train_start DATE,
    train_end DATE,
    val_accuracy DECIMAL(5,4),
    val_sharpe DECIMAL(6,3),
    features_json JSONB,
    hyperparams_json JSONB,
    model_path VARCHAR(255),
    is_active BOOLEAN DEFAULT FALSE
);

-- Agent decisions log
CREATE TABLE agent_decisions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    agent VARCHAR(30) NOT NULL,  -- ANALYSIS, RISK, EXECUTION, JOURNAL
    trade_id UUID REFERENCES trades(id),
    decision VARCHAR(20) NOT NULL,  -- RECOMMEND, APPROVE, REJECT, MODIFY, EXECUTE
    reasoning TEXT,
    confidence DECIMAL(5,4),
    thinking_tokens INTEGER,
    latency_ms INTEGER,
    cost_usd DECIMAL(6,4)
);

CREATE INDEX idx_trades_status ON trades(status);
CREATE INDEX idx_trades_ticker ON trades(ticker);
CREATE INDEX idx_trades_strategy ON trades(strategy);
CREATE INDEX idx_trades_entry_time ON trades(entry_time);
CREATE INDEX idx_account_snapshots_ts ON account_snapshots(timestamp);
CREATE INDEX idx_agent_decisions_ts ON agent_decisions(timestamp);
```

### QuestDB Tables (Time-Series)

```sql
-- Market data ticks
CREATE TABLE IF NOT EXISTS market_ticks (
    timestamp TIMESTAMP,
    ticker SYMBOL,
    bid DOUBLE,
    ask DOUBLE,
    last DOUBLE,
    volume LONG,
    iv DOUBLE,
    delta DOUBLE,
    gamma DOUBLE,
    theta DOUBLE,
    vega DOUBLE
) TIMESTAMP(timestamp) PARTITION BY DAY;

-- GEX levels
CREATE TABLE IF NOT EXISTS gex_levels (
    timestamp TIMESTAMP,
    ticker SYMBOL,
    spot_price DOUBLE,
    net_gex DOUBLE,
    call_wall DOUBLE,
    put_wall DOUBLE,
    vol_trigger DOUBLE,
    regime SYMBOL
) TIMESTAMP(timestamp) PARTITION BY DAY;

-- Signal scores
CREATE TABLE IF NOT EXISTS signal_scores (
    timestamp TIMESTAMP,
    ticker SYMBOL,
    technical_score DOUBLE,
    sentiment_score DOUBLE,
    flow_score DOUBLE,
    regime_score DOUBLE,
    ensemble_score DOUBLE,
    confidence DOUBLE
) TIMESTAMP(timestamp) PARTITION BY DAY;
```

---

## MAIN APPLICATION ENTRY POINT (src/main.py)

The main application must:
1. Load all configuration from environment and YAML
2. Initialize all database connections (PostgreSQL, QuestDB, Redis)
3. Connect to IB Gateway with health monitoring
4. Load trained ML models (or train from scratch if none exist)
5. Initialize regime detector, signal generators, strategy selector
6. Initialize Claude AI agents (with fallback to pure ML if API unavailable)
7. Set up APScheduler with all scheduled tasks
8. Start the event loop:
   - Stream market data via ib_async
   - Process signals on schedule
   - Evaluate trade opportunities
   - Execute approved trades
   - Monitor open positions for exits
   - Track P&L and risk metrics
   - Export Prometheus metrics
9. Handle graceful shutdown (close positions? → NO, just disconnect cleanly)
10. Expose FastAPI health endpoint on port 8080

### Application Lifecycle
```
STARTUP:
  → Load config
  → Connect databases
  → Connect IB Gateway (retry until connected)
  → Load/train ML models
  → Initialize agents
  → Start schedulers
  → Begin market data streaming
  → Log "TITAN ONLINE" to Telegram

MARKET HOURS (9:30 AM - 4:00 PM ET):
  → 9:35 AM: Full universe scan (regime + signals + strategy selection)
  → Every 15 min: Check open positions for exit criteria
  → 11:30 AM, 1:30 PM, 3:30 PM: Intraday opportunity scans
  → Continuous: Circuit breaker monitoring, risk metric updates

AFTER HOURS:
  → 4:15 PM: Journal Agent reviews all trades
  → 4:30 PM: Daily P&L summary to Telegram
  → 5:00 PM: Update signal databases, cache cleanup
  → Saturday 6 AM: Weekly model retraining + Optuna optimization

SHUTDOWN:
  → Cancel all pending orders
  → Disconnect from IB Gateway cleanly
  → Flush all database writes
  → Log "TITAN OFFLINE" to Telegram
```

---

## CRITICAL IMPLEMENTATION NOTES

### IB Gateway Specifics
- **ib_async** is the successor to ib_insync — use it, not the old library
- Only ONE session per IBKR username — never log into Client Portal or TWS while bot is running
- IB Gateway resets daily 00:15–01:45 ET — bot must handle reconnection
- Rate limit: 50 messages/second. Batch requests, use `sleep(0.05)` between rapid calls
- Max 100 concurrent streaming data lines (increases with commissions/equity)
- Options chains: call `reqSecDefOptParams()` first (no rate limit), then filter strikes before requesting data
- Combo/BAG orders: set `NonGuaranteed = "0"` for guaranteed fills, always use limit orders
- Error codes to handle: 502 (not connected), 504 (not connected), 1100 (connectivity lost), 1102 (restored), 2104/2106/2158 (data farm connections)
- Order Efficiency Ratio must stay below 20:1 (submitted÷filled)

### Claude API Specifics
- Use prompt caching: system prompt is static, mark with `cache_control: {"type": "ephemeral"}`
- Extended thinking: minimum 1024 tokens, set budget via `thinking.budget_tokens`
- Batch API: set `custom_id` and use `/v1/messages/batches` endpoint for Journal Agent
- Error handling: implement exponential backoff for 429 (rate limit) and 529 (overloaded)
- Fallback: if Claude API is unreachable for > 2 minutes, fall back to pure ML signal-based trading (no agent reasoning)

### Options Trading Specifics
- IV Rank = (Current IV − 52-week Low IV) ÷ (52-week High IV − 52-week Low IV)
- IV Percentile = % of days in past year where IV was below current level
- GEX = Σ(call_OI × call_gamma × 100 × spot) − Σ(put_OI × put_gamma × 100 × spot)
- Bull call spread max loss = net debit paid
- Iron condor max loss = wider wing width − net credit received
- Always use monthly expirations for better liquidity (not weeklies)
- Close spreads before expiration — never hold through expiry

---

## REMEMBER

- **No placeholders.** Every function must be fully implemented.
- **No TODO comments.** If something needs doing, do it now.
- **No mock data in production code.** Use real API calls everywhere.
- **Test with paper trading first.** Use IBKR_TRADING_MODE=paper.
- **Log everything.** Every trade decision, every API call, every error.
- **The bot must survive restarts.** All state persists to PostgreSQL.
- **Circuit breakers are sacred.** They NEVER get overridden.

Build this system to be the most intelligent, reliable, and profitable options trading bot ever created. The operator's livelihood depends on it.
