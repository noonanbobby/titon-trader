# PROJECT TITAN: The Definitive AI-Powered Options Trading System
## Master Blueprint & Complete Product Specification
### Version 2.0 — February 2026

---

## EXECUTIVE SUMMARY

Project Titan is a fully automated, AI-driven options trading system designed to trade a $150,000 Interactive Brokers (IBKR) account with the goal of reaching $200,000 by June 2026. The system combines institutional-grade infrastructure, a ten-strategy regime-adaptive options engine, ensemble machine learning signal generation, Claude AI multi-agent reasoning, and a four-layer risk management framework. It runs 24/7 on a Linux environment (WSL Ubuntu for development, VPS for production), connecting to IBKR via headless IB Gateway in Docker containers.

**What makes Titan different from every other retail trading bot:**

1. **It doesn't just trade one strategy.** It maintains a ten-strategy options arsenal and autonomously selects the optimal structure (bull call spreads, iron condors, credit spreads, calendars, butterflies, PMCCs, etc.) based on real-time market regime detection.

2. **It thinks before it trades.** Claude AI agents with extended thinking analyze multi-factor situations, synthesize quantitative signals with macro context and news sentiment, and produce reasoned trade theses — not just signal-driven orders.

3. **It improves itself.** Reinforcement learning optimizes position management. Bayesian optimization tunes strategy parameters via walk-forward windows. Online learning detects concept drift and triggers retraining. A Journal Agent reviews every closed trade and feeds pattern improvements back into the system.

4. **It sees what others don't.** GEX (Gamma Exposure) analysis reveals mechanical support/resistance from dealer hedging flows. Insider cluster buying signals detect smart money positioning. Volatility risk premium harvesting exploits the structural tendency of implied volatility to overstate realized volatility.

5. **It protects capital above all.** A four-layer risk framework with automated circuit breakers ensures the 15% ($22,500) maximum drawdown is never breached, with graduated position reduction at 2%, 5%, 10%, and 15% thresholds.

---

## SYSTEM SPECIFICATIONS

### Account Parameters
- **Broker:** Interactive Brokers (IBKR)
- **Account Size:** $150,000
- **Target:** $200,000 by June 2026 (~33% return in ~4 months)
- **Max Drawdown:** 15% ($22,500)
- **Per-Trade Risk:** 2% ($3,000) maximum loss
- **Primary Universe:** Large-cap tech (AAPL, NVDA, MSFT, GOOGL, META, AMZN, TSLA, AMD, AVGO, CRM)
- **Secondary Universe:** SPX/NDX index options (for 60/40 tax treatment)
- **Strategy Holding Period:** 1–45 days (short-term swing and premium selling)

### Technology Stack
| Component | Technology | Purpose |
|-----------|-----------|---------|
| Language | Python 3.12+ | Core application |
| IBKR API | ib_async (v2.x) | Broker connectivity |
| IB Gateway | gnzsnz/ib-gateway-docker | Headless broker gateway |
| Gateway Manager | IBC (v3.17+) | Auto-login, 2FA, daily restart |
| Message Bus | Redis 7+ (Streams + Pub/Sub) | Inter-component messaging |
| Trade Database | PostgreSQL 16 | Orders, positions, audit logs |
| Time-Series DB | QuestDB | Market data, tick history |
| ML Framework | XGBoost + LightGBM + CatBoost | Ensemble signal generation |
| Sentiment | FinBERT (ProsusAI/finbert) | Financial news NLP |
| Regime Detection | hmmlearn (GaussianHMM) | Market regime classification |
| RL Framework | Stable-Baselines3 (SAC/PPO) | Position management optimization |
| Optimization | Optuna (v4.7+) | Bayesian hyperparameter tuning |
| Online Learning | River (v0.21+) | Incremental model updates |
| Options Pricing | QuantLib-Python + py_vollib | Greeks, IV surface, pricing |
| Technical Analysis | pandas-ta | 130+ indicators |
| AI Reasoning | Claude API (Sonnet 4.5) | Multi-agent trade analysis |
| AI Orchestration | LangGraph | Multi-agent state machine |
| Dashboard | Grafana + Prometheus | Real-time monitoring |
| Analysis UI | Streamlit | Interactive strategy analysis |
| Notifications | Telegram Bot API + Twilio | Alerts and trade notifications |
| Containerization | Docker Compose | Service orchestration |
| Dev Environment | WSL Ubuntu 24.04 | Windows 11 development |
| Production | QuantVPS / Hetzner (US East) | Always-on VPS deployment |

### Data Sources
| Source | Cost | Data Provided |
|--------|------|---------------|
| IBKR Market Data | ~$6/mo (OPRA + equities) | Real-time quotes, options chains, Greeks |
| Polygon.io | $199/mo (Advanced) | Tick data, options history, dark pool prints |
| Unusual Whales | $50/mo + API | Options flow, sweeps, congressional trades |
| Finnhub | Free (60 calls/min) | Earnings calendar, economic calendar, news |
| Quiver Quantitative | $25/mo | Insider filings, 13F, patents, lobbying |
| FRED API | Free | Macro indicators, yield curves, Fed data |
| SEC EDGAR | Free | Form 4 insider filings, 13F institutional |
| StockTwits API | Free | Retail sentiment |
| **Total Monthly** | **~$280/mo** | |

---

## THE TEN-STRATEGY OPTIONS ARSENAL

Titan does not rely on a single strategy. It maintains ten distinct options strategies and deploys them based on quantitative regime detection. Each strategy has precise entry criteria, position sizing rules, and mechanical exit parameters.

### Strategy 1: Bull Call Spread (Directional Bullish — Debit)
- **When:** Trending up (ADX > 25), IV Rank 20–50%, ML confidence > 0.78
- **Construction:** Buy 0.55–0.60 delta call, sell 0.25–0.30 delta call
- **DTE:** 30–45 days, monthly expirations preferred
- **Entry Filters:** Price > 20 SMA and 50 SMA, RSI 40–60, no earnings within DTE, OI > 50 both strikes, bid-ask < 10% of option price
- **Position Size:** Max risk 2% of account ($3,000). For $5 net debit = 6 contracts max
- **Exit:** Close at 50–65% of max profit, OR at 21 DTE, OR if spread value drops to 25–50% of entry (stop loss)
- **Rolling:** Only if thesis intact and roll cost < 50% of new spread max profit. Never roll more than once

### Strategy 2: Bull Put Spread / Credit Put Spread (Directional Bullish — Credit)
- **When:** Trending up + high IV (IV Rank > 50%), strong support identified
- **Construction:** Sell 16–20 delta put, buy 5–10 delta put (same expiry)
- **DTE:** 45 days optimal
- **Credit Target:** 15–20% of spread width
- **Position Size:** Max risk = spread width minus credit × contracts ≤ $3,000
- **Exit:** Close at 50% of credit received, OR at 21 DTE, OR if loss reaches 2× credit

### Strategy 3: Iron Condor (Neutral — Credit)
- **When:** Range-bound (ADX < 20), IV Rank ≥ 50–70%, no catalyst expected
- **Construction:** Sell 15–20 delta call and put, buy wings 5–10 points wider
- **DTE:** 30–60 days
- **Credit Target:** 25–33% of wing width
- **Position Size:** Max risk per side ≤ $3,000
- **Exit:** Close at 50% of total credit, OR at 21 DTE. Defend tested side by rolling untested side closer for additional credit

### Strategy 4: Calendar Spread (Volatility Play — Debit)
- **When:** IV Rank < 30% (expect IV expansion), stable underlying
- **Construction:** Sell front-month ATM option (25–40 DTE), buy back-month same strike (50–90 DTE)
- **Best Use:** Pre-earnings (sell inflated near-term, buy calmer far-term)
- **Position Size:** Max debit ≤ $3,000
- **Exit:** Close when front-month expires or at 50% profit. Close immediately if underlying moves > 1 standard deviation from strike

### Strategy 5: Diagonal Spread (Directional + Volatility)
- **When:** Moderate directional bias + IV expansion expected
- **Construction:** Buy ITM back-month option, sell OTM front-month option
- **DTE:** Long leg 60–90 days, short leg 25–35 days
- **Exit:** Roll short leg monthly. Close when long leg reaches 50% profit

### Strategy 6: Broken Wing Butterfly (Precise Target — Credit)
- **When:** High confidence in price target, IV Rank > 40%
- **Construction:** Buy 1 ITM option, sell 2 ATM options, buy 1 OTM option (asymmetric wings)
- **DTE:** 21 days (Carl Allen methodology)
- **Entry Deltas:** 32/28/20 for puts
- **Credit Target:** Entered for net credit (10–15% of narrow wing width)
- **Exit:** Close at 2% of narrow wing width profit, OR at 7 DTE

### Strategy 7: Short Strangle (Premium Selling — Undefined Risk)
- **When:** IV Rank 50–100%, expected range-bound, high conviction
- **Construction:** Sell 16-delta call and 16-delta put (no wings)
- **DTE:** 45 days
- **Position Size:** SMALL — max 2–3% of portfolio per strangle (undefined risk)
- **Exit:** Close at 50% of credit. Close immediately if tested side breaches short strike. Never hold through earnings

### Strategy 8: Poor Man's Covered Call / PMCC (Income — Debit)
- **When:** Long-term bullish on underlying, want capital-efficient income
- **Construction:** Buy deep ITM LEAPS call (70–80 delta, 12+ months), sell 30-delta monthly calls
- **Position Size:** LEAPS cost ≤ 10% of account per position
- **Exit:** Roll LEAPS when < 6 months remaining. Roll short calls monthly. Close entire position if LEAPS drops to 50% of entry value

### Strategy 9: Ratio Spread (Skew Exploitation — Mixed)
- **When:** Volatility skew exceeds 1 SD from historical average
- **Construction:** Buy 1 ATM call, sell 2 OTM calls (1:2 ratio)
- **DTE:** 30–45 days
- **CAUTION:** Unlimited upside risk — position size must be tiny (max 1% of account)
- **Exit:** Close if underlying approaches short strikes. Hard stop at 50% of max theoretical loss

### Strategy 10: Long Straddle (Event Play — Debit)
- **When:** Pre-earnings with IV Rank < 30%, expect large move, direction uncertain
- **Construction:** Buy ATM call + ATM put (same strike, same expiry)
- **DTE:** Enter 7–14 days before event, expiry 1–2 weeks after event
- **Position Size:** Max debit ≤ 2% of account
- **Exit:** Close immediately after event (capture IV crush avoidance by early entry). Target 30% profit

---

## REGIME DETECTION AND STRATEGY SELECTION

### The Four-Regime Model

A Hidden Markov Model (GaussianHMM with 3 states) trained on rolling 4-year daily returns, realized volatility, and VIX classifies the market into regimes. A simpler VIX-threshold + ADX overlay provides redundancy.

| Regime | VIX Range | ADX | Character | Primary Strategies |
|--------|-----------|-----|-----------|-------------------|
| **Low Vol Trending** | < 18 | > 25 | Calm trend | Bull call spreads, PMCCs, diagonals |
| **High Vol Trending** | 18–35 | > 25 | Volatile trend | Credit put spreads, narrow bull call spreads |
| **Range-Bound High IV** | 18–35 | < 20 | Choppy, premium-rich | Iron condors, short strangles, BWB |
| **Crisis** | > 35 | Any | Panic | NO new positions. Protective puts only. Reduce all exposure 50% |

### Strategy Selection Logic (Pseudocode)
```
FOR each ticker in universe:
    regime = hmm_model.predict(current_features)
    iv_rank = calculate_iv_rank(ticker, lookback=252)
    trend = calculate_trend_strength(ticker)  # ADX, SMA alignment
    ml_confidence = ensemble_model.predict_proba(ticker_features)
    
    IF regime == CRISIS:
        SKIP (no new positions)
    
    eligible_strategies = filter_by_regime(regime, all_strategies)
    
    FOR each strategy in eligible_strategies:
        IF strategy.entry_criteria_met(ticker, iv_rank, trend, ml_confidence):
            risk_check = risk_manager.evaluate(strategy, ticker, current_portfolio)
            IF risk_check.approved:
                score = calculate_expected_value(strategy, ticker)
                candidates.append((strategy, ticker, score))
    
    SORT candidates by score DESC
    EXECUTE top candidate (if any pass all filters)
```

---

## ENSEMBLE ML SIGNAL GENERATION

### Architecture: Four Signal Streams → Meta-Learner

**Stream 1: Gradient Boosted Tree Ensemble (50–60% weight)**
- Models: XGBoost + LightGBM + CatBoost (3 models, averaged)
- Target: 5-day forward return classification (UP > 1%, DOWN < -1%, NEUTRAL)
- Features (per ticker, ~120 total):
  - Price-based: RSI(14), MACD, Bollinger %B, ATR(14), Keltner Channel position, rate of change (5/10/20 day), price vs SMA(20/50/200), SMA crossover signals
  - Volume-based: OBV slope, VWAP deviation, volume z-score (20-day), relative volume
  - Volatility: IV Rank, IV Percentile, HV(20)/IV ratio, Bollinger Band width, ATR percentile
  - Options-specific: Put/call OI ratio, put/call volume ratio, max pain distance, GEX polarity
  - Cross-asset: VIX level, VIX term structure slope (VIX/VIX3M), 2Y/10Y spread, DXY, XLK relative strength, high-yield OAS spread
  - Calendar: Day of week, days to next FOMC, days to next CPI, days to earnings
- Training: Walk-forward (12-month train, 1-month test, roll monthly)
- Anti-overfitting: Purged k-fold CV (5 folds, 5-day embargo), feature importance pruning (drop features with < 0.5% importance)

**Stream 2: FinBERT Sentiment Engine (15–20% weight)**
- Model: ProsusAI/finbert (pre-trained, no fine-tuning needed)
- Sources: Finnhub news API, StockTwits API, SEC 8-K filings
- Processing: Batch inference every 15 minutes during market hours
- Output: Sentiment score (-1 to +1) per ticker, 24h rolling average
- Special handling: Reddit/WSB sentiment is CONTRARIAN (negative weight)

**Stream 3: Options Flow Signal (20–25% weight)**
- Source: Unusual Whales API + IBKR options chain data
- Signals:
  - Volume/OI ratio > 1.25 (unusual activity)
  - Sweep orders detected (urgency signal)
  - Block trades > $500K premium (institutional)
  - Multi-day consistency: same-direction flow 3+ consecutive days (strongest signal)
  - Net directional premium: bullish premium − bearish premium (dollar-weighted)
- Output: Flow score (-1 to +1) per ticker

**Stream 4: Regime Context (5–10% weight)**
- HMM regime state (categorical: low_vol_trend, high_vol_trend, range_bound, crisis)
- VIX term structure (contango = bullish, backwardation = bearish)
- Fed funds futures implied probability of next rate move
- Insider cluster buying score (from Quiver/SEC EDGAR)
- GEX polarity (positive = mean-reverting, negative = trend-amplifying)

**Meta-Learner: Calibrated XGBoost**
- Input: Calibrated probability outputs from all 4 streams
- Calibration: Isotonic regression (sklearn CalibratedClassifierCV)
- Output: Final confidence score 0.0–1.0
- Trade threshold: τ = 0.78 (only trade when confidence exceeds this)
- Retraining: Weekly on walk-forward schedule

---

## ALPHA SIGNALS: RANKED BY EVIDENCE

### Tier 1: Proven, Implementable, High-Value
1. **Volatility Risk Premium (VRP) Harvesting** — IV overestimates RV ~85–90% of the time. Systematic premium selling (iron condors, credit spreads) captures 3–5 vol points of edge. Requires dynamic hedging for tail risk
2. **Insider Cluster Buying** — Multiple Form 4 purchase filings within 30 days, weighted by seniority and dollar value. Academic evidence: 22–32% annualized alpha
3. **GEX (Gamma Exposure) Analysis** — Dealer hedging flows create mechanical support/resistance. Positive GEX = mean-reverting (sell wings wider). Negative GEX = trend-amplifying (tighten stops, smaller size)

### Tier 2: Useful as Filters, Not Standalone Alpha
4. **VIX Term Structure** — Backwardation is strong contrarian buy signal. Contango is normal
5. **Cross-Asset Signals** — HY OAS spread, copper/gold ratio, 2Y/10Y curve
6. **13F Institutional Positioning** — 45-day lag limits alpha but useful for conviction overlay
7. **Short Interest / Days-to-Cover** — > 5 days notable, > 10 days squeeze risk

### Tier 3: Marginal or Inaccessible for Retail
8. Dark pool data (delayed, can't distinguish accumulation from rebalancing)
9. Congressional trading post-STOCK Act (45-day disclosure delay, diminished alpha)
10. 0DTE flow as directional indicator (CBOE research shows balanced customer activity)
11. Satellite/credit card data ($50K+/year, institutional only)

---

## FOUR-LAYER RISK MANAGEMENT FRAMEWORK

### Layer 1: Per-Trade Risk Controls
- Maximum loss per position: 2% of account ($3,000)
- Defined-risk strategies only (spreads, no naked options except tiny strangles)
- Position sizing formula: Max Contracts = $3,000 ÷ (spread width × 100)
- Use fractional Kelly criterion (quarter-Kelly) for sizing optimization
- Always use limit orders. Never market orders on multi-leg spreads
- No entries during first 15 minutes or last 15 minutes of market day

### Layer 2: Portfolio-Level Controls
- Maximum concurrent positions: 5–8 spreads
- Maximum single-ticker exposure: 25–30% of total risk ($5,625–$6,750)
- Maximum single-sector exposure: 40–50% of total risk
- Portfolio beta-weighted delta: capped at ±$15,000
- Net portfolio vega: under $5,000
- Total capital deployed: 30–50% of account, never exceeding 70%
- Correlation monitoring: reduce if rolling pairwise correlation > 0.7 across > 50% of positions

### Layer 3: Drawdown Circuit Breakers (AUTOMATED, NON-NEGOTIABLE)
| Trigger | Action |
|---------|--------|
| Daily loss > 2% ($3,000) | Halt all trading for remainder of day |
| Weekly loss > 5% ($7,500) | Reduce position sizes by 50%, no new positions for 2 trading days |
| Monthly loss > 10% ($15,000) | Full stop. Close all positions. Mandatory strategy review |
| Total drawdown > 15% ($22,500) | EMERGENCY STOP. Close everything. Minimum 2-week trading halt |

**Recovery Ladder:**
- Resume at 50% normal position size
- Require 3 consecutive winners → move to 75%
- Require 3 more consecutive winners → return to 100%
- Only restore full sizing when account recovers to within 5% of high-water mark

### Layer 4: Event Risk & Tail Protection
- Close or reduce positions 5–7 days before earnings of underlying
- Avoid new entries 1–2 days before FOMC, CPI, NFP
- Monitor ex-dividend dates for early assignment risk on short calls
- Hedge budget: 1–3% per quarter for SPY puts (2–5% OTM, 45–90 DTE) or VIX calls
- Tail risk composite score: CBOE SKEW + VVIX + put/call skew + HY OAS spread
- When composite z-score > 2.0: reduce all exposure by 50%
- Kill switch: immediate liquidation on API connectivity loss > 5 minutes during market hours

---

## CLAUDE AI MULTI-AGENT ARCHITECTURE

### Four Specialized Agents via LangGraph

**Agent 1: Analysis Agent (Claude Sonnet 4.5 with Extended Thinking)**
- Role: Primary trade analyst
- Thinking budget: 4K–8K tokens for routine, 16K for complex multi-asset
- Inputs: ML ensemble scores, regime state, GEX levels, sentiment, news, options chain data, cross-asset signals
- Output: Structured trade recommendation with confidence score, reasoning chain, and specific spread parameters
- Frequency: Runs at market open (9:35 AM ET) and every 2 hours during market hours

**Agent 2: Risk Agent (Claude Sonnet 4.5, Medium Effort)**
- Role: Portfolio risk guardian with VETO POWER
- Evaluates: Portfolio Greeks exposure, correlation risk, drawdown proximity, event calendar, circuit breaker status
- Can: Approve, reject, or modify (reduce size) any proposed trade
- Frequency: Evaluates every trade proposal + continuous portfolio monitoring every 30 minutes

**Agent 3: Execution Agent (Claude Sonnet 4.5, Low Effort)**
- Role: Translates approved trades into IBKR orders
- Selects: Order type (limit at mid, limit at 10% above mid, etc.), combo order construction
- Monitors: Fill status, slippage, partial fills
- Handles: Order modifications if not filled within 5 minutes (adjust limit by 1 tick)

**Agent 4: Journal Agent (Claude Sonnet via Batch API — 50% discount)**
- Role: End-of-day trade analysis and pattern learning
- Reviews: Every closed trade with P&L, entry/exit conditions, market context
- Identifies: Winning patterns, losing patterns, parameter improvements
- Memory: FinMem layered architecture (short-term: last 5 trades; medium-term: last 30 trades; long-term: regime-level patterns)
- Output: Updated system prompt additions for Analysis Agent, suggested parameter adjustments for Optuna

### Cost Management
- Prompt caching: 90% savings on static system prompts
- Tiered model usage: Haiku for quick classification, Sonnet for analysis, extended thinking only when needed
- Batch API: 50% discount for end-of-day journaling
- Estimated monthly API cost: $50–100 depending on trade frequency

---

## SELF-IMPROVING ML PIPELINE

### Reinforcement Learning (Position Management)
- Algorithm: SAC (Soft Actor-Critic) via Stable-Baselines3
- State space: Underlying price, option portfolio value, all Greeks, IV, time to expiry, moneyness, VIX, regime
- Action space: Continuous — scale factor [-1 (close all) to +1 (add to position)]
- Reward: Sharpe-based over rolling 20-trade windows
- Training: Offline on historical data, updated monthly with new trade outcomes
- Deployment: Suggests hold/close/scale/roll decisions, subject to Risk Agent approval

### Bayesian Optimization (Strategy Parameters)
- Engine: Optuna v4.7 with TPESampler
- Optimizes: Entry/exit thresholds, DTE targets, delta targets, confidence threshold τ
- Multi-objective: Maximize Sharpe, minimize max drawdown (Pareto front)
- Schedule: Weekly walk-forward re-optimization
- Storage: PostgreSQL-backed Optuna study for persistence
- Dashboard: optuna-dashboard for visualization

### Online Learning (Drift Detection)
- Engine: River library with ADWIN detector
- Monitors: Strategy error rate, feature distribution shifts
- Action on drift: Increase learning rate → reset model weights → trigger full Optuna re-optimization
- Concept drift detection: Page-Hinkley test + ADWIN ensemble

---

## INFRASTRUCTURE AND DEPLOYMENT

### Development Environment (WSL Ubuntu)
```
Windows 11
└── WSL2 (Ubuntu 24.04)
    └── Docker Desktop (WSL2 backend)
        ├── ib-gateway (gnzsnz/ib-gateway-docker)
        ├── titan-bot (Python 3.12, custom image)
        ├── redis (7-alpine)
        ├── postgres (16)
        ├── questdb (latest)
        ├── grafana (latest)
        └── prometheus (latest)
```

### Production Environment (VPS — future)
- Provider: QuantVPS or Hetzner (US East Coast, sub-5ms to NYSE)
- Specs: 4+ vCPU, 16GB+ RAM, 100GB+ NVMe
- OS: Ubuntu 24.04 LTS
- Same Docker Compose stack as development
- GitHub Actions CI/CD for deployment
- Cloudflare Tunnel for secure remote Grafana access
- Automated backups: PostgreSQL → S3, QuestDB → S3

### Monitoring & Alerting
- **Grafana dashboards:** Real-time P&L curve, open positions with Greeks, regime state, ML model confidence, circuit breaker status, system health (CPU, memory, API latency)
- **Prometheus metrics:** Custom Python metrics via prometheus_client for trade counts, win rate, average P&L, drawdown percentage, API response times
- **Telegram bot:** Trade entry/exit notifications, daily P&L summary, circuit breaker alerts
- **Twilio SMS:** Critical failures only — connectivity loss, circuit breaker triggers, emergency stop
- **QuantStats HTML reports:** Weekly automated tear sheets with Sharpe, Sortino, max drawdown, monthly heatmap

---

## ACCELERATED GO-LIVE TIMELINE

| Day | Milestone |
|-----|-----------|
| Day 1 | WSL Ubuntu + Docker environment configured. IB Gateway connecting to paper account. Basic bot framework running |
| Day 2 | Market data streaming. Options chain retrieval working. Order execution tested on paper (single bull call spread) |
| Day 3 | ML pipeline with historical data. Backtesting framework operational. First walk-forward backtest complete |
| Day 4 | All 10 strategies coded. Regime detection active. Risk management framework tested |
| Day 5 | Claude AI agents integrated. Telegram notifications working. Grafana dashboards live |
| Day 6 | Full paper trading smoke test — all systems coordinated. Monitor for bugs |
| Day 7 | **GO LIVE** with 1-2 contracts per position. Full monitoring active |
| Week 2 | Scale to normal position sizing if performance matches expectations |
| Week 3 | Push to GitHub. Begin VPS setup for always-on deployment |
| Week 4+ | Full production on VPS. Continuous improvement loop active |

---

## FINANCIAL PROJECTIONS (CONSERVATIVE)

**Assumptions:**
- 3–5 trades per week (confidence-filtered)
- 70–75% win rate (after confidence threshold filtering)
- Average winner: $4,500 (50–65% of max profit on spreads with 2:1+ reward)
- Average loser: $3,000 (2% max loss per trade)
- 60 trades over 16 weeks

**Expected Value Per Trade:**
E[V] = (0.72 × $4,500) − (0.28 × $3,000) = $3,240 − $840 = **+$2,400**

**Projected Total P&L:**
60 trades × $2,400 = **$144,000 theoretical**

**After Costs:**
- Data subscriptions: ~$1,200 (4 months)
- Claude API: ~$400 (4 months)  
- VPS (when deployed): ~$300 (4 months)
- Slippage/commissions: ~$3,000 (estimated)
- **Net: ~$139,000 theoretical**

**Reality-Adjusted Target:** Even at 50% of theoretical (accounting for model imperfection, execution gaps, and the learning curve of the first month), **$50,000+ profit is achievable** — hitting the $200K target.

---

*This document serves as the complete product specification for Project Titan. The accompanying CLAUDE.md file contains the comprehensive prompt for Claude Code to build the entire system. The SETUP_CHECKLIST.md contains parallel tasks for the operator.*
