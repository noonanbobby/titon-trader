# Project Titan — AI-Powered Options Trading System

An institutional-quality, AI-driven options trading system built to manage $150,000 of real capital on Interactive Brokers. Combines machine learning signal generation, Claude AI multi-agent decision-making, and a four-layer risk management framework to execute 10 defined-risk options strategies across a 40-ticker universe.

## Architecture

```
IB Gateway ─── Titan Bot ─── PostgreSQL
                  │              (trades, state)
Redis ────────────┤
  (cache, pubsub) │           QuestDB
                  │              (time-series)
Prometheus ───────┤
                  │           Grafana
                  │              (dashboards)
```

**7 Docker services** orchestrated via `docker-compose.yml`, all ports bound to `127.0.0.1`.

## Signal Pipeline

Four signal streams feed an XGBoost meta-learner with isotonic calibration:

| Stream | Weight | Source |
|--------|--------|--------|
| Technical/ML | 50-60% | 120+ indicators via XGBoost + LightGBM |
| Sentiment | 15-20% | FinBERT on Finnhub news |
| Options Flow | 20-25% | Unusual Whales: vol/OI, sweeps, blocks |
| Regime Context | 5-10% | GaussianHMM, VIX structure, insider clusters, GEX |

Only signals exceeding the **0.78 confidence threshold** proceed to the AI agent pipeline.

## AI Agent Pipeline

```
Analysis Agent (Claude + extended thinking, 16K budget)
       │
       ▼
Risk Agent (hard limits + AI evaluation, VETO power)
       │
       ├── APPROVED → Execution Agent → IBKR combo orders
       ├── MODIFIED → Execution Agent (adjusted parameters)
       └── REJECTED → logged, done

Journal Agent runs end-of-day via Batch API (50% cost savings)
```

When Claude API is unavailable for >2 minutes, the system falls back to pure ML signal-based trading.

## 10 Options Strategies

| # | Strategy | Regime | IV Rank |
|---|----------|--------|---------|
| 1 | Bull Call Spread | Low vol trending | 20-50% |
| 2 | Bull Put Spread | Uptrend + high IV | 40-70% |
| 3 | Iron Condor | Range-bound | 50-70% |
| 4 | Calendar Spread | Low IV | <30% |
| 5 | Diagonal Spread | Mild trend | 30-50% |
| 6 | Broken Wing Butterfly | Range/mild trend | 30-50% |
| 7 | Short Strangle | Range + high IV | >50% |
| 8 | PMCC | Strong uptrend | 20-40% |
| 9 | Ratio Spread | Directional + IV | >40% |
| 10 | Long Straddle | Pre-catalyst | <30% |

Strategy selection is driven by a 4-regime classifier (GaussianHMM): low_vol_trend, high_vol_trend, range_bound, crisis. VIX > 35 unconditionally triggers crisis mode.

## Risk Management

**Four-layer framework** protecting $150,000 of capital:

| Layer | Scope | Controls |
|-------|-------|----------|
| Layer 1 | Per-Trade | 2% max loss ($3,000), defined-risk only |
| Layer 2 | Portfolio | Max 8 positions, 25-30% per ticker, Greek limits |
| Layer 3 | Circuit Breakers | 2% daily halt, 5% weekly reduce, 10% monthly stop, 15% emergency |
| Layer 4 | Tail Risk | SKEW/VVIX composite, earnings/FOMC avoidance |

Circuit breaker recovery is graduated: 50% → 75% → 100% sizing, requiring consecutive winners at each stage.

## Quick Start

### Prerequisites

- Docker and Docker Compose
- Interactive Brokers account (paper or live)
- API keys: Anthropic, Polygon.io, Unusual Whales, Finnhub, Quiver, FRED
- Telegram bot token (for notifications)

### Setup

```bash
# Clone the repository
git clone <repo-url> titan && cd titan

# Copy environment template and fill in your API keys
cp .env.example .env
# Edit .env with your credentials

# Initialize the database
docker compose up -d postgres
docker compose exec postgres psql -U titan -f /docker-entrypoint-initdb.d/init.sql

# Start all services
docker compose up -d

# Check health
docker compose ps
docker compose logs -f titan
```

### Paper Trading (Required First)

Set `IBKR_TRADING_MODE=paper` in your `.env` file. The system connects to IB Gateway port 4002 (paper) by default. Monitor via:

- **Grafana**: http://localhost:3000 (admin/admin)
- **Prometheus**: http://localhost:9090
- **Titan health**: http://localhost:8080/health (internal to Docker)
- **IB Gateway VNC**: localhost:5900

### Running Tests

```bash
uv run pytest tests/ -v --tb=short
```

**657 tests** covering circuit breakers, position sizing, ensemble signals, regime detection, base strategy, helpers, broker contracts, market data routing, and ML feature engineering.

## Project Structure

```
titan/
├── src/
│   ├── main.py              # Application entry point & lifecycle
│   ├── broker/              # IBKR integration (ib_async)
│   ├── strategies/          # 10 options strategy implementations
│   ├── signals/             # Signal generation (technical, sentiment, flow, regime, GEX, insider, VRP)
│   ├── risk/                # Risk management (circuit breakers, position sizing, Greeks, correlation)
│   ├── ai/                  # Claude AI agents (LangGraph pipeline)
│   ├── ml/                  # ML pipeline (walk-forward training, calibration, Optuna, RL, backtesting)
│   ├── data/                # External API clients (Polygon, UW, Finnhub, Quiver, FRED, SEC EDGAR, QuestDB, Redis)
│   ├── notifications/       # Telegram bot, Twilio SMS, QuantStats reports
│   └── utils/               # Logging, Prometheus metrics, scheduling, helpers
├── config/                  # Settings, strategy params, risk limits, Grafana/Prometheus config
├── scripts/                 # Database init, health check, setup
├── tests/                   # Test suite
├── docs/                    # Project specification & research documents
├── models/                  # Trained ML model artifacts
└── docker-compose.yml       # 7-service orchestration
```

## Key Technologies

- **Python 3.12** with async/await throughout
- **ib_async** for IBKR integration (successor to ib_insync)
- **Claude API** with extended thinking, prompt caching, Batch API
- **LangGraph** for AI agent state machine
- **XGBoost** + LightGBM ensemble with isotonic calibration
- **GaussianHMM** (hmmlearn) for regime detection
- **FinBERT** (ProsusAI/finbert) for sentiment analysis
- **stable-baselines3** for RL position management (SAC/PPO)
- **River** for online learning with ADWIN drift detection
- **Optuna** for walk-forward hyperparameter optimization
- **PostgreSQL 16** for relational data
- **QuestDB** for time-series data (ILP ingestion)
- **Redis 7** for caching and pub/sub
- **Prometheus** + **Grafana** for monitoring
- **APScheduler** for task scheduling (US/Eastern timezone)
- **structlog** for JSON structured logging
- **Pydantic** for all data models
- **tenacity** for retry logic with exponential backoff

## Documentation

| Document | Purpose |
|----------|---------|
| `CLAUDE.md` | Complete project instructions, coding standards, and audit status |
| `docs/MasterBluePrint.md` | Authoritative product specification |
| `docs/Research-1.md` | Infrastructure research (VPS, Docker, GEX, VRP, insider alpha) |
| `docs/Research-2.md` | Architecture research (IBKR API, ML accuracy, FinBERT, QuestDB) |
| `docs/SetupChecklist.md` | Operator setup guide with API key acquisition |

## Operational Costs

| Service | Monthly Cost |
|---------|-------------|
| Polygon.io | $199 |
| Unusual Whales | $50-75 |
| Claude API | $50-100 |
| Quiver Quantitative | $10-25 |
| IBKR Data | $6-16 |
| Twilio | ~$5 |
| **Total** | **~$380-520** |

## License

Proprietary. All rights reserved.
