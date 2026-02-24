# AI-driven options trading bot: complete architectural blueprint

**A production-grade bull call spread system targeting 33% returns on a $150K Interactive Brokers account is achievable with the right architecture, but demands disciplined risk management and realistic ML expectations.** The plan below integrates IBKR API connectivity via IB Gateway, Claude AI decision support through custom MCP servers, ensemble ML signal generation, and a layered risk framework designed to protect against the 15% max drawdown constraint. This blueprint covers every component — from data ingestion to order execution to tax compliance — forming a complete system specification ready for implementation.

---

## System architecture and core technology stack

The system follows an event-driven microservice pattern running on Windows 11, built primarily in Python with Redis as the internal message bus. Python is the clear choice here: IBKR's own API latency of **10–50ms** makes Python's execution speed irrelevant as a bottleneck, while Python's ecosystem for ML, options pricing, and IBKR integration is unmatched (42% of all algo-trading repositories on GitHub use Python).

**Core stack decisions:**

- **IBKR connectivity:** IB Gateway (not TWS) using the `ib_async` library (v2.1.0, successor to `ib_insync`, actively maintained at github.com/ib-api-reloaded/ib_async). IB Gateway consumes ~40% fewer resources than TWS while providing identical socket API functionality on port 4001 (live) or 4002 (paper). Use IBC (github.com/IbcAlpha/IBC) for automated login, 2FA handling, and daily restart management.
- **Databases:** Dual-database architecture — **PostgreSQL** for relational trade management (orders, positions, account state, ACID compliance) and **QuestDB** for time-series market data (12–36× faster ingestion than InfluxDB, native `ASOF JOIN` for temporal queries, SQL-compatible).
- **Message bus:** Redis with Streams for inter-component communication. Redis Pub/Sub distributes real-time market data to strategy engines; Redis Streams with consumer groups handle order events requiring acknowledgment. Sub-millisecond latency with built-in persistence.
- **Dashboard:** Grafana for always-on P&L and system health monitoring (connects natively to both PostgreSQL and QuestDB), supplemented by Streamlit for interactive strategy analysis and ad-hoc reporting.
- **Windows deployment:** Servy (modern Windows service manager, available via `winget install servy`) wraps the Python trading bot as an NT service with auto-recovery, health checks, and crash notifications. IBC manages IB Gateway lifecycle independently.

The process architecture runs as coordinated services:

```
Windows Services (Servy)
├── IBC → IB Gateway (auto-login, daily restart at 23:45 ET)
├── Trading Bot Main Process
│   ├── Market Data Handler (ib_async, async event loop)
│   ├── Feature Engine (pandas-ta, py_vollib)
│   ├── ML Signal Generator (XGBoost/LightGBM ensemble)
│   ├── Strategy Engine (scoring, threshold, spread construction)
│   └── Order Manager (combo orders, position tracking)
├── Redis Server (message bus + cache)
├── QuestDB (time-series data store)
├── PostgreSQL (trade management)
├── Grafana (monitoring dashboard)
└── Streamlit (analysis dashboard)
```

---

## IBKR API integration and order execution

The IBKR integration layer handles three critical functions: market data streaming, options chain retrieval with Greeks, and multi-leg spread order execution. The `ib_async` library implements the full IBKR binary protocol internally — it does not require the official `ibapi` package — and provides both synchronous and asyncio interfaces.

**Options chain and Greeks retrieval** follows a non-throttled pipeline. Call `reqSecDefOptParams()` to get available strikes and expirations without hitting rate limits, filter to relevant strikes (±20 points from underlying), build `Option` contracts, qualify them via `qualifyContracts()`, then request streaming data with `reqMktData()`. Greeks (delta, gamma, theta, vega, implied volatility) arrive automatically via tick types 10–13 (`tickOptionComputation` callback). This requires active market data subscriptions for both OPRA options ($1.50/month) and the underlying equity exchanges (~$4.50/month for Networks A+B+C), though fees are waived above $5/month in commissions.

**Bull call spread execution** uses IBKR's combo/BAG contract type. The spread is defined as a single contract with two `ComboLeg` objects — BUY the lower-strike call, SELL the higher-strike call — submitted as a guaranteed fill (`NonGuaranteed = "0"`). Always use **limit orders** with the net debit price, never market orders. The reference project `aicheung/0dte-trader` on GitHub provides working implementations of options combo orders with delta-based strike selection that can serve as a starting template.

**Connection reliability** requires handling IBKR's mandatory daily system reset between **00:15–01:45 ET**. IBC automates re-authentication after restarts. The bot must implement exponential backoff reconnection, monitor for error codes 502/504 (not connected) and 1100/1102 (connectivity lost/restored), and re-subscribe to all market data streams after reconnection. Critical constraint: only one brokerage session per username — never log into Client Portal while the bot is running.

**Rate limits to respect:** 50 messages/second across all API calls, maximum 100 concurrent streaming data lines (scales with commissions/equity), and an Order Efficiency Ratio (orders submitted ÷ orders filled) that must stay below 20:1 to avoid IBKR warnings.

---

## Claude AI and MCP server integration strategy

Claude AI serves two roles in this system: a development accelerator via Claude Code and a real-time analytical engine via the Claude API and MCP servers. The MCP ecosystem now includes **multiple purpose-built IBKR servers** and **financial data servers** that can be composed into a powerful intelligence layer.

**IBKR MCP servers** (choose based on deployment needs):

- `xiao81/IBKR-MCP-Server` — TWS API-based, provides `get_portfolio()`, `get_account_summary()`, `get_option_price()`, requires local TWS/Gateway. Best for this Windows setup.
- `code-rabi/interactive-brokers-mcp` — Full trading via `npx -y interactive-brokers-mcp`, supports paper trading mode, Flex Query integration.
- `rcontesti/IB_MCP` — Web API approach via Docker, no TWS required, better for cloud deployments.

**Financial data MCP servers** to configure in Claude Code:

- **Alpha Vantage MCP** (`uvx av-mcp YOUR_API_KEY`) — Real-time options chains, technical indicators (RSI, MACD, Bollinger), fundamentals. Free API key. Widely regarded as the most versatile finance MCP server.
- **Financial Datasets MCP** (remote at `mcp.financialdatasets.ai/mcp`) — Company facts, SEC filings, insider trades, institutional ownership, market news.
- **Massive.com/Polygon.io MCP** — Full options/equity data including dark pool trades, news via Benzinga, analyst ratings.

**Building a custom MCP server** for this system is straightforward using FastMCP (now part of the official MCP Python SDK). The recommended approach wraps `ib_async` calls as MCP tools:

```python
from mcp.server.fastmcp import FastMCP
mcp = FastMCP("ibkr-options-server")

@mcp.tool()
async def analyze_spread(symbol: str, long_strike: float, 
                         short_strike: float, expiry: str) -> dict:
    """Analyze bull call spread: P&L profile, Greeks, probability of profit."""
    # ib_async calls for option data, QuantLib for analytics
    ...
```

**Claude API for trade decisions** operates cost-effectively. Using Sonnet at $3/$15 per million tokens with prompt caching (90% savings on cached system prompts), analyzing a trade signal costs roughly **$0.01 per call**. At 100 signals/day, that's $1.35/day. The system should use prompt caching for the static system prompt containing strategy rules, risk parameters, and market context, while sending only new signal data as dynamic input. Use the **Batch API** (50% discount) for end-of-day analysis, screening, and model evaluation. Extended thinking (minimum 1,024 tokens) should be enabled for complex multi-factor trade decisions.

**Claude Code on Windows 11** is now natively supported (`winget install Anthropic.ClaudeCode`). Use `CLAUDE.md` files in the project root to provide architecture context, trading rules, and code conventions. MCP servers are configured per-project in `settings.json`. The Max plan ($100/month) provides 5× Pro usage, Opus access, and 1M context window — recommended for active development.

---

## ML signal generation: a realistic, high-probability approach

Achieving the target 90%+ win rate requires extreme trade selectivity — the system will generate signals for roughly **2–5 high-conviction trades per week** across the six target stocks, not dozens. Research consistently shows that unfiltered directional accuracy on large-cap equities maxes out at **53–60%**, but confidence-filtered predictions (only trading when model confidence exceeds 0.80) can reach **70–85% accuracy** at ~12% coverage. The system compensates for fewer trades by sizing positions appropriately within the $150K account.

**Ensemble model architecture** combines four signal streams through a calibrated meta-learner:

1. **Gradient boosted tree ensemble** (XGBoost + LightGBM + CatBoost) trained on tabular features: technical indicators across multiple timeframes, IV rank/percentile, HV/IV ratio, put/call ratios, volume/OI analysis, cross-asset features (VIX, XLK momentum, yield curve, DXY). These models handle nonlinear feature interactions natively and are fast to train and inference (<10ms).

2. **Sentiment engine** using FinBERT (`ProsusAI/finbert` on HuggingFace) applied to financial news from Benzinga/Finnhub APIs plus social media from StockTwits. Key insight: Reddit/WSB sentiment works best as a **contrarian** indicator (academic research shows WSB attention reduces holding-period returns by 8.5%). Run sentiment inference in batches every 15–30 minutes, not real-time.

3. **Options flow signal** tracking unusual activity: volume/OI ratio >1.25, sweep orders, block trades >$500K premium, and net directional premium flow. Multi-day consistency (repeated call buying over 3+ days) is a stronger signal than single-day spikes. Source from Unusual Whales API ($50/month + API add-on).

4. **Regime context** via Hidden Markov Model (2–3 states using `hmmlearn` GaussianHMM on rolling returns + VIX). The regime determines which parameter set the strategy engine uses and whether to trade at all (crisis regime = no new positions).

**The meta-learner** (XGBoost) takes calibrated probability outputs from all four streams and produces a final confidence score. Calibration uses isotonic regression (`sklearn.calibration.CalibratedClassifierCV`). Only scores above **τ = 0.78** trigger trade evaluation. This threshold is optimized on walk-forward validation data to maximize expected profit per trade.

**Critical anti-overfitting measures** follow Marcos López de Prado's framework from *Advances in Financial Machine Learning*: purged k-fold cross-validation (removing training observations whose labels overlap with test periods), embargo periods after each fold, combinatorial purged cross-validation (CPCV) for multiple backtest paths, and tri-state labeling (up/down/hold). The `mlfinlab` library implements these methods. Models retrain weekly using walk-forward optimization.

**Recommended feature weights** (starting point, to be optimized empirically): Technical/Quantitative signals 50–60%, Options flow 20–25%, Sentiment 15–20%, Macro context 5–10%.

---

## Bull call spread parameter optimization

The strategy engine translates high-confidence bullish signals into specific spread parameters. These parameters adapt to market regime, IV environment, and the individual stock's characteristics.

**Strike selection uses delta-based targeting.** Buy the call at **0.55–0.60 delta** (slightly ITM, higher probability of profit) and sell the call at **0.25–0.30 delta** (sufficient premium offset, realistic profit target). This creates a net spread delta of 0.30–0.35 — providing meaningful directional exposure while limiting risk to the net debit. In trending markets with high conviction, widen the spread for greater profit potential; in choppy or uncertain conditions, narrow the spread to reduce cost.

**Optimal DTE is 30–45 days at entry**, the consensus sweet spot from tastytrade, projectoption, and Nasdaq research. A 27-DTE spread that goes ITM captures **44% of max profit** versus only 19% at 174 DTE. Always prefer monthly expirations over weeklies for better liquidity and tighter bid-ask spreads. In high-IV environments (IV Rank >50), shorten DTE to 30 days to capture faster decay; in normal IV (rank 20–50), extend to 45 days for more time.

**Entry criteria** require multiple confirmations: ML confidence score above threshold, IV Rank between **20–50** (avoid paying inflated premium above 50), open interest >50 on both strikes with bid-ask spread <10% of option price, no earnings within the DTE window, no FOMC/CPI within 2 days, price above 20-day and 50-day SMA, and RSI between 40–60. Enter preferably on Tuesday–Wednesday to avoid Monday gaps and Friday decay.

**Exit rules are mechanistic:** Close at **50–65% of max profit** (captures the favorable portion of the probability curve before diminishing returns), close at **21 DTE** regardless of P&L (gamma risk inflection point), and close immediately if the spread value drops to **25–50% of entry price** (75–50% loss on debit paid). Never hold through expiration unless both legs are deeply ITM.

**Rolling is the exception, not the rule.** Roll out in time only when the thesis remains intact and the roll costs less than 50% of the new spread's max profit. Never roll more than once. If the thesis is invalidated, take the loss cleanly.

---

## Risk management: the layered defense framework

Risk management is the single most critical component for hitting the 33% return target without exceeding the 15% drawdown limit. The framework operates at four levels: per-trade, portfolio, drawdown circuit breakers, and Greeks-based hedging.

**Per-trade risk** caps at **2% of account ($3,000) maximum loss per position**. For a $5.00 net debit spread, that's 6 contracts maximum. For a $3.00 spread, 10 contracts. This allows approximately 7 consecutive losing trades before hitting the max drawdown — a sufficient buffer given the system's confidence-filtered entry criteria. Use fractional Kelly criterion (quarter-Kelly) for optimal position sizing, which captures ~75% of optimal growth while cutting drawdown risk by roughly half.

**Portfolio-level controls** maintain these constraints simultaneously:

- Maximum **5–8 concurrent spread positions** (diversified across tickers and expirations)
- Maximum **25–30% of total risk** in any single stock ($5,625–$6,750 at risk in one name)
- Maximum **40–50% of total risk** in any single sector (critical for an all-tech portfolio)
- Portfolio beta-weighted delta capped at **±$15,000** (10% of account)
- Net portfolio vega under **$5,000** (a 1-point IV move changes portfolio by <$5K)
- Total capital deployed: 30–50% of account ($45K–$75K), never exceeding 70%

**Drawdown circuit breakers** trigger automatically:

| Trigger | Action |
|---------|--------|
| Daily loss >2% ($3,000) | Halt trading for remainder of day |
| Weekly loss >5% ($7,500) | Reduce position sizes by 50%, no new positions for 2 days |
| Monthly loss >10% ($15,000) | Full stop, close all positions, complete strategy review |
| Total drawdown >15% ($22,500) | Full stop, minimum 2-week break, paper trade before resuming |

**Recovery follows a disciplined ladder:** Resume at 50% normal position size, require 3 consecutive winners before moving to 75%, then 3 more before returning to 100%. Only restore full sizing when the account recovers to within 5% of its high-water mark.

**Event risk protocol:** Close or reduce all positions **5–7 days before earnings** of the underlying (IV crush on debit spreads is destructive). Avoid new entries 1–2 days before FOMC/CPI releases. Monitor ex-dividend dates — if the short call's extrinsic value drops below the dividend amount, early assignment risk is elevated (though large-cap tech dividends are small: AAPL ~$0.25, MSFT ~$0.83 per quarter).

**Portfolio hedging budget:** Allocate **1–3% per quarter** ($1,500–$4,500) for tail-risk protection via SPY puts (2–5% OTM, 45–90 DTE) or VIX calls (20%+ OTM, 3–4 months out). This insurance protects against correlated sell-offs in the tech-heavy portfolio.

---

## Data pipeline and external intelligence sources

The real-time data pipeline feeds the ML models, regime detector, and strategy engine with structured, low-latency information from multiple sources.

**Recommended data stack** (estimated $350–500/month total):

- **Polygon.io/Massive.com** ($199/month Advanced tier) — Real-time options data, tick-level quotes, dark pool trades, Benzinga news add-on ($99/month). Full options history back to 2014 for backtesting.
- **Unusual Whales** ($50/month + API add-on) — Options flow, sweep detection, congressional trades, dark pool data. Best value in the options flow space.
- **Finnhub** (free tier, 60 calls/minute) — Earnings calendar, economic calendar (FOMC, CPI, jobs), news with basic sentiment, financial statements. Generous free tier eliminates need for paid alternative.
- **Theta Data** ($80/month Standard) — Historical options data for backtesting, 4–12 years depending on tier.

**Volatility surface construction** uses QuantLib-Python's `BlackVarianceSurface` with market strike/expiry/IV data, interpolated via bicubic method. For fast bulk IV calculations, `py_vollib` implements Jäckel's LetsBeRational algorithm — orders of magnitude faster than traditional Newton-Raphson. Monitor IV term structure (ATM IV across expirations) and skew (IV across strikes at each expiration) for regime signals.

**Market regime detection** runs a 2–3 state GaussianHMM from `hmmlearn` on rolling daily returns and VIX levels. The regime output adjusts strategy parameters: low-volatility regime (VIX <15) uses wider spreads and longer DTE; elevated volatility (VIX 25–35) uses narrower spreads and reduced position sizes; crisis regime (VIX >35) halts new entries entirely.

---

## Backtesting and paper trading validation path

Before any live capital deployment, the system must pass three validation gates: historical backtesting, walk-forward optimization, and paper trading.

**Historical backtesting** uses QuantConnect's Lean engine (open-source, ~10K GitHub stars, full options support including chains, Greeks, and assignment modeling) with Theta Data or QuantConnect's built-in options data (2010–present, minute-level). The critical requirement is **realistic fill modeling**: entry at mid-price plus 10–25% of spread width as slippage, spread orders not individual legs, and proper handling of pin risk near expiration.

**Walk-forward optimization** splits data into rolling in-sample (parameter optimization) and out-of-sample (validation) windows — train on months 1–12, test month 13; train on 2–13, test 14; and so on. Combine with combinatorial purged cross-validation (CPCV) to generate multiple backtest paths and measure Probability of Backtest Overfitting. This is the single most important step for preventing live losses from overfitting.

**IBKR paper trading** runs for a minimum of **4 weeks** across different market conditions before live deployment. Paper trading uses the same API on port 7497 with real-time market data. Key limitations to understand: paper fills are simulated at NBBO without modeling queue position or partial fills, so live fills will be slightly worse. Track paper performance versus theoretical backtest expectations; if paper results deviate more than 15% from backtested performance, investigate before proceeding.

**Transition to live:** Start with **1 contract per position** for the first 2 weeks of live trading, regardless of what position sizing models suggest. Scale to normal sizing only after confirming that live fills, slippage, and system behavior match paper trading results within acceptable variance.

The open-source project **ThetaGang** (2.4K GitHub stars, actively maintained through 2026) provides excellent reference architecture for IBKR integration patterns including regime-aware rebalancing and VIX hedging. **OptionLab** (477 stars, v1.5.1) handles strategy P&L modeling and probability-of-profit calculation. **FinRL** (10K+ stars, AI4Finance Foundation) demonstrates the cutting edge of reinforcement learning for trading, though options-specific RL remains nascent.

---

## Regulatory compliance and account configuration

A **$150K IBKR account** clears all regulatory thresholds comfortably. The account is well above the $25,000 PDT minimum, allowing unlimited day trading. It also exceeds the **$110,000 Portfolio Margin threshold**, which can significantly reduce margin requirements for diversified positions — though starting with Reg-T margin is recommended until the system proves itself.

**Bull call spreads require IBKR Level 2 options permissions** (Long Call Spread classification). Ensure the account has "Good" or "Extensive" options knowledge designation and at least 2 years of trading experience reported. Under Reg-T, the margin requirement for a bull call spread equals the **net debit paid** — no additional margin beyond the premium. This means nearly the full $150K account is available for spreading.

**No SEC or FINRA registration is required** for individual retail traders running personal automated strategies. FINRA's algorithmic trading registration (Series 57) applies only to associated persons of FINRA member firms. However, the bot must respect IBKR's Order Efficiency Ratio — keep the ratio of submitted/revised/cancelled orders to executed orders **below 20:1** to avoid warnings. Use limit orders, avoid excessive order modifications, and leverage IBKR's server-side order types (trailing stops, pegged orders) which don't count toward OER.

**Tax implications are significant** for frequent options trading. Equity options on individual stocks (AAPL, NVDA, etc.) are taxed as **short-term capital gains** at ordinary income rates — no Section 1256 benefit. Wash sale rules apply aggressively: rolling positions or re-entering the same underlying within 30 days after a loss triggers wash sale treatment. Consider using **SPX/NDX options** for some positions to access Section 1256's 60/40 tax treatment and wash sale exemption, though liquidity on individual strikes differs from equity options.

**IBKR performs real-time auto-liquidation** with no margin call warning period. The bot must monitor Excess Liquidity and SMA continuously. Implement a software-level margin buffer that triggers position reduction well before IBKR's auto-liquidation threshold.

---

## Achieving the $150K → $200K target: a quantitative assessment

The target of $200K by early summer 2026 requires ~$50,000 in profit (~33% return) over approximately 16 weeks. With 2–5 high-conviction trades per week, the system needs roughly 40–80 total trades during this period.

At **2% risk per trade ($3,000 max loss)** with a realistic **70–75% win rate** (confidence-filtered) and an average **reward-to-risk of 1.5:1** (targeting 50–65% of max profit on spreads that offer 2:1+ max reward), the expected value per trade is approximately:

**E[V] = (0.72 × $4,500) − (0.28 × $3,000) = $3,240 − $840 = +$2,400 per trade**

Over 60 trades: **60 × $2,400 = $144,000 theoretical**. Even after accounting for slippage, data costs (~$2,000 for the period), and Claude API costs (~$500), the mathematical path to $50,000 profit exists with significant margin of safety — provided the system achieves and maintains the confidence-filtered win rate through disciplined trade selection and strict risk management.

The greatest risks to this plan are correlated tech sector drawdowns (mitigated by portfolio hedging and sector concentration limits), model degradation in a shifting regime (mitigated by weekly retraining and regime detection), and execution discipline failures (mitigated by fully automated rules-based exits). The circuit breaker framework ensures that even worst-case scenarios cap losses at the $22,500 maximum drawdown.

## Conclusion

This blueprint specifies a complete, production-grade system where every component has been selected for a specific reason: `ib_async` for proven IBKR integration, XGBoost ensembles for tabular financial prediction where they consistently outperform deep learning, confidence-threshold filtering to convert mediocre directional accuracy into high win rates, and a four-layer risk framework that mechanistically prevents catastrophic losses. The most important architectural insight is that **trade selectivity — not model sophistication — drives the system's edge**. By trading only 2–5 times per week at extreme confidence levels while managing risk with automated circuit breakers, the system converts a modest statistical edge into consistent compounding. Begin with the IBKR paper trading integration and backtesting pipeline, validate the ML signal system on historical data with proper walk-forward methodology, and only deploy live capital after 4+ weeks of paper trading confirmation.