"""LangGraph multi-agent orchestration for Project Titan.

Implements the trading pipeline as a stateful graph of four cooperative
AI agents:

1. **Analysis Agent** -- evaluates market data and ML signals to propose
   trades using Claude with extended thinking.
2. **Risk Agent** -- evaluates proposals against portfolio limits and
   circuit-breaker rules; can approve, reject, or modify.
3. **Execution Agent** -- translates approved proposals into IBKR combo
   order specifications.
4. **Journal Agent** -- reviews closed trades at end-of-day to extract
   lessons and update the FinMem memory system.

The graph routes conditionally after the Risk node: approved proposals
flow to Execution, rejected proposals are logged and the pipeline ends.

When the Claude API is unreachable the system degrades gracefully into
**fallback mode**, generating trade proposals from ML signals alone and
applying hard-coded risk checks that do not require LLM reasoning.

Usage::

    from config.settings import get_settings
    from src.ai.agents import TradingAgentOrchestrator, AgentState

    settings = get_settings()
    orch = TradingAgentOrchestrator(
        api_key=settings.api_keys.anthropic_api_key.get_secret_value(),
        settings=settings.claude,
    )
    state = orch.create_initial_state(ticker="AAPL", ...)
    result = await orch.run_with_fallback(state)
"""

from __future__ import annotations

import time
from datetime import UTC
from typing import TYPE_CHECKING, Any, TypedDict

from langgraph.graph import END, START, StateGraph
from prometheus_client import Counter, Gauge, Histogram
from pydantic import BaseModel, Field
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.utils.logging import get_logger
from src.utils.metrics import API_LATENCY

if TYPE_CHECKING:
    import structlog

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CONFIDENCE_THRESHOLD: float = 0.78
"""Minimum ensemble confidence required to generate a trade signal."""

CLAUDE_API_TIMEOUT_SECONDS: float = 120.0
"""Maximum seconds to wait before declaring the Claude API unavailable."""

MAX_PIPELINE_RETRIES: int = 3
"""Number of retries for transient API errors within a single node."""

RETRY_MIN_WAIT_SECONDS: float = 1.0
"""Minimum wait between retries (exponential backoff base)."""

RETRY_MAX_WAIT_SECONDS: float = 30.0
"""Maximum wait between retries (exponential backoff cap)."""

# Token estimates for cost calculation (Claude Sonnet 3.5 pricing)
ANALYSIS_INPUT_TOKENS: int = 2000
ANALYSIS_OUTPUT_TOKENS: int = 1000
ANALYSIS_THINKING_TOKENS: int = 8192
RISK_INPUT_TOKENS: int = 1500
RISK_OUTPUT_TOKENS: int = 500
RISK_THINKING_TOKENS: int = 4096
EXECUTION_INPUT_TOKENS: int = 500
EXECUTION_OUTPUT_TOKENS: int = 300

# Per-token pricing in USD (Sonnet 3.5 with prompt caching)
INPUT_COST_PER_TOKEN: float = 3.0 / 1_000_000
CACHED_INPUT_COST_PER_TOKEN: float = 0.30 / 1_000_000
OUTPUT_COST_PER_TOKEN: float = 15.0 / 1_000_000

# Regime-to-strategy mapping for ML-only fallback mode
REGIME_STRATEGY_MAP: dict[str, list[str]] = {
    "low_vol_trend": [
        "bull_call_spread",
        "bull_put_spread",
        "pmcc",
        "diagonal_spread",
    ],
    "high_vol_trend": [
        "iron_condor",
        "short_strangle",
        "broken_wing_butterfly",
    ],
    "range_bound": [
        "iron_condor",
        "calendar_spread",
        "short_strangle",
    ],
    "crisis": [
        "long_straddle",
        "ratio_spread",
    ],
}

# IV rank thresholds that influence strategy selection in fallback mode
IV_RANK_LOW: float = 30.0
IV_RANK_HIGH: float = 50.0

logger: structlog.stdlib.BoundLogger = get_logger("ai.agents")

# ---------------------------------------------------------------------------
# Prometheus metrics specific to the AI pipeline
# ---------------------------------------------------------------------------

AGENT_LATENCY: Histogram = Histogram(
    "titan_agent_latency_seconds",
    "Latency of individual agent steps in the trading pipeline",
    labelnames=["agent"],
    buckets=(
        0.1,
        0.25,
        0.5,
        1.0,
        2.5,
        5.0,
        10.0,
        20.0,
        30.0,
        60.0,
    ),
)

PIPELINE_LATENCY: Histogram = Histogram(
    "titan_pipeline_latency_seconds",
    "End-to-end latency of the full trading pipeline",
    buckets=(
        0.5,
        1.0,
        2.5,
        5.0,
        10.0,
        20.0,
        30.0,
        60.0,
        120.0,
    ),
)

PROPOSALS_TOTAL: Counter = Counter(
    "titan_proposals_total",
    "Total number of trade proposals generated",
    labelnames=["source"],
)

REJECTIONS_TOTAL: Counter = Counter(
    "titan_rejections_total",
    "Total number of trade proposals rejected by the risk agent",
    labelnames=["reason"],
)

APPROVALS_TOTAL: Counter = Counter(
    "titan_approvals_total",
    "Total number of trade proposals approved by the risk agent",
)

FALLBACK_MODE_ACTIVE: Gauge = Gauge(
    "titan_fallback_mode_active",
    "1 when the system is operating in ML-only fallback mode",
)

PIPELINE_ERRORS: Counter = Counter(
    "titan_pipeline_errors_total",
    "Total errors encountered in the agent pipeline",
    labelnames=["agent"],
)


# ---------------------------------------------------------------------------
# LangGraph state definition
# ---------------------------------------------------------------------------


class AgentState(TypedDict, total=False):
    """Shared state flowing through the LangGraph trading pipeline.

    Each node reads from and writes to this dict.  Fields are populated
    incrementally as the pipeline progresses through analysis, risk
    evaluation, execution, and journaling stages.
    """

    # -- Market context (populated before pipeline starts) --
    ticker: str
    ml_scores: dict[str, Any]
    regime: str
    iv_rank: float
    sentiment_score: float
    gex_data: dict[str, Any]
    options_chain: list[dict[str, Any]]
    account_summary: dict[str, Any]
    current_positions: list[dict[str, Any]]
    portfolio_greeks: dict[str, Any]
    circuit_breaker_state: dict[str, Any]
    event_calendar: dict[str, Any]
    correlation_data: dict[str, Any]

    # -- Pipeline outputs (populated by agents) --
    proposals: list[dict[str, Any]]
    risk_evaluations: list[dict[str, Any]]
    execution_results: list[dict[str, Any]]
    journal_entries: list[dict[str, Any]]

    # -- Control flow --
    errors: list[str]
    step: str
    should_execute: bool
    fallback_mode: bool


# ---------------------------------------------------------------------------
# Pydantic models for structured agent I/O
# ---------------------------------------------------------------------------


class AnalysisInput(BaseModel):
    """Input payload for the Analysis Agent.

    Aggregates all market context needed for Claude to reason about
    potential trade opportunities.
    """

    ticker: str = Field(
        ...,
        min_length=1,
        max_length=10,
        description="Underlying symbol",
    )
    ml_scores: dict[str, Any] = Field(
        default_factory=dict,
        description="ML ensemble scores keyed by signal type",
    )
    regime: str = Field(
        default="unknown",
        description="Current market regime identifier",
    )
    iv_rank: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Current IV Rank (0-100)",
    )
    sentiment_score: float = Field(
        default=0.0,
        ge=-1.0,
        le=1.0,
        description="FinBERT rolling sentiment score",
    )
    gex_data: dict[str, Any] = Field(
        default_factory=dict,
        description="Gamma Exposure levels and regime",
    )
    options_chain: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Snapshot of the options chain",
    )
    account_summary: dict[str, Any] = Field(
        default_factory=dict,
        description="Account equity and buying power",
    )
    current_positions: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Currently open positions",
    )
    event_calendar: dict[str, Any] = Field(
        default_factory=dict,
        description="Upcoming earnings, FOMC, CPI events",
    )


class RiskContext(BaseModel):
    """Input payload for the Risk Agent.

    Combines the proposed trade with current portfolio state so the
    risk agent can check all Layer 1 and Layer 2 controls.
    """

    proposal: dict[str, Any] = Field(
        ...,
        description="Trade proposal from the Analysis Agent",
    )
    account_summary: dict[str, Any] = Field(
        default_factory=dict,
        description="Account equity and buying power",
    )
    current_positions: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Currently open positions",
    )
    portfolio_greeks: dict[str, Any] = Field(
        default_factory=dict,
        description="Aggregate portfolio Greeks",
    )
    circuit_breaker_state: dict[str, Any] = Field(
        default_factory=dict,
        description="Current circuit-breaker level and P&L",
    )
    correlation_data: dict[str, Any] = Field(
        default_factory=dict,
        description="Rolling correlation matrix data",
    )
    event_calendar: dict[str, Any] = Field(
        default_factory=dict,
        description="Upcoming events that may affect risk",
    )


class TradeProposal(BaseModel):
    """Structured trade proposal emitted by the Analysis Agent."""

    ticker: str
    strategy: str
    direction: str
    confidence: float = Field(ge=0.0, le=1.0)
    parameters: dict[str, Any] = Field(default_factory=dict)
    reasoning: str = ""
    legs: list[dict[str, Any]] = Field(default_factory=list)
    max_profit: float = 0.0
    max_loss: float = 0.0
    reward_risk_ratio: float = 0.0


class RiskEvaluation(BaseModel):
    """Structured risk evaluation emitted by the Risk Agent."""

    proposal_index: int = Field(
        ge=0,
        description="Index into the proposals list",
    )
    decision: str = Field(
        description="APPROVED, REJECTED, or MODIFIED",
    )
    reason: str = ""
    modifications: dict[str, Any] = Field(default_factory=dict)
    risk_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Overall risk score (0=safe, 1=extreme)",
    )


class ExecutionResult(BaseModel):
    """Structured execution result from the Execution Agent."""

    proposal_index: int = Field(ge=0)
    status: str = Field(
        description="PLANNED, SUBMITTED, FILLED, FAILED",
    )
    order_spec: dict[str, Any] = Field(default_factory=dict)
    error: str = ""


class JournalEntry(BaseModel):
    """Structured journal entry from the Journal Agent."""

    date: str
    trades_reviewed: int = 0
    total_pnl: float = 0.0
    win_rate: float = 0.0
    insights: list[str] = Field(default_factory=list)
    patterns_detected: list[str] = Field(default_factory=list)
    memory_updates: dict[str, Any] = Field(default_factory=dict)
    recommendations: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Helper functions (module-level)
# ---------------------------------------------------------------------------


def create_ml_fallback_proposal(state: AgentState) -> dict[str, Any]:
    """Create a basic trade proposal from ML signals when Claude is unavailable.

    Uses the ensemble confidence score directly and selects a strategy
    based on a hard-coded regime-to-strategy mapping combined with IV
    rank filtering.

    Parameters
    ----------
    state:
        The current pipeline state containing market context.

    Returns
    -------
    dict
        A simplified proposal dict compatible with the risk evaluation
        step, or an empty dict if conditions are not met.
    """
    ml_scores = state.get("ml_scores", {})
    ml_confidence = float(ml_scores.get("ensemble_score", 0.0))
    regime = state.get("regime", "unknown")
    iv_rank = state.get("iv_rank", 0.0)
    ticker = state.get("ticker", "")

    if ml_confidence < CONFIDENCE_THRESHOLD:
        logger.info(
            "fallback_below_threshold",
            ticker=ticker,
            ml_confidence=round(ml_confidence, 4),
            threshold=CONFIDENCE_THRESHOLD,
        )
        return {}

    # Select eligible strategies for the current regime
    eligible_strategies = REGIME_STRATEGY_MAP.get(regime, [])
    if not eligible_strategies:
        logger.warning(
            "fallback_no_strategies_for_regime",
            ticker=ticker,
            regime=regime,
        )
        return {}

    # Apply IV rank filtering to narrow strategy choice
    if iv_rank >= IV_RANK_HIGH:
        # High IV: prefer credit strategies
        preferred = [
            s
            for s in eligible_strategies
            if s
            in {
                "iron_condor",
                "short_strangle",
                "bull_put_spread",
                "broken_wing_butterfly",
            }
        ]
    elif iv_rank <= IV_RANK_LOW:
        # Low IV: prefer debit strategies
        preferred = [
            s
            for s in eligible_strategies
            if s
            in {
                "bull_call_spread",
                "calendar_spread",
                "long_straddle",
                "diagonal_spread",
                "pmcc",
            }
        ]
    else:
        preferred = eligible_strategies

    # Fall back to the full eligible list if filtering was too strict
    if not preferred:
        preferred = eligible_strategies

    selected_strategy = preferred[0]

    # Determine direction based on strategy type
    credit_strategies = {
        "iron_condor",
        "short_strangle",
        "bull_put_spread",
        "broken_wing_butterfly",
    }
    direction = "SHORT" if selected_strategy in credit_strategies else "LONG"

    proposal = {
        "ticker": ticker,
        "strategy": selected_strategy,
        "direction": direction,
        "confidence": ml_confidence,
        "parameters": {
            "regime": regime,
            "iv_rank": iv_rank,
        },
        "reasoning": (
            f"ML fallback: ensemble confidence {ml_confidence:.4f} "
            f"exceeds threshold {CONFIDENCE_THRESHOLD} in "
            f"{regime} regime with IV rank {iv_rank:.1f}. "
            f"Selected {selected_strategy} based on regime-strategy "
            f"mapping and IV rank filtering."
        ),
        "legs": [],
        "max_profit": 0.0,
        "max_loss": 0.0,
        "reward_risk_ratio": 0.0,
        "source": "ml_fallback",
    }

    logger.info(
        "fallback_proposal_created",
        ticker=ticker,
        strategy=selected_strategy,
        direction=direction,
        confidence=round(ml_confidence, 4),
    )

    return proposal


def estimate_pipeline_cost(state: AgentState) -> float:
    """Estimate the total Claude API cost for running the full pipeline.

    Assumes the system prompt is cached after the first call, reducing
    input costs for the risk and execution steps.

    Parameters
    ----------
    state:
        The current pipeline state (used for context but cost is a
        static estimate based on average token usage).

    Returns
    -------
    float
        Estimated cost in USD for processing one ticker through the
        full analysis-risk-execution pipeline.
    """
    # Analysis Agent: first call, system prompt not yet cached
    analysis_cost = (
        ANALYSIS_INPUT_TOKENS * INPUT_COST_PER_TOKEN
        + ANALYSIS_OUTPUT_TOKENS * OUTPUT_COST_PER_TOKEN
        + ANALYSIS_THINKING_TOKENS * OUTPUT_COST_PER_TOKEN
    )

    # Risk Agent: system prompt cached from analysis call
    risk_cost = (
        RISK_INPUT_TOKENS * CACHED_INPUT_COST_PER_TOKEN
        + RISK_OUTPUT_TOKENS * OUTPUT_COST_PER_TOKEN
        + RISK_THINKING_TOKENS * OUTPUT_COST_PER_TOKEN
    )

    # Execution Agent: system prompt cached, no thinking
    execution_cost = (
        EXECUTION_INPUT_TOKENS * CACHED_INPUT_COST_PER_TOKEN
        + EXECUTION_OUTPUT_TOKENS * OUTPUT_COST_PER_TOKEN
    )

    total = analysis_cost + risk_cost + execution_cost

    logger.debug(
        "pipeline_cost_estimate",
        analysis_usd=round(analysis_cost, 6),
        risk_usd=round(risk_cost, 6),
        execution_usd=round(execution_cost, 6),
        total_usd=round(total, 6),
    )

    return round(total, 6)


def _safe_get_list(
    state: AgentState,
    key: str,
) -> list[Any]:
    """Retrieve a list from state, defaulting to empty list."""
    value = state.get(key, [])
    if value is None:
        return []
    return list(value)


def _safe_get_dict(
    state: AgentState,
    key: str,
) -> dict[str, Any]:
    """Retrieve a dict from state, defaulting to empty dict."""
    value = state.get(key, {})
    if value is None:
        return {}
    return dict(value)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


class TradingAgentOrchestrator:
    """LangGraph-based orchestrator for the four-agent trading pipeline.

    Manages the lifecycle of Analysis, Risk, Execution, and Journal
    agents, wiring them into a directed graph with conditional routing
    after the risk evaluation step.

    Parameters
    ----------
    api_key:
        Anthropic API key for Claude calls.
    settings:
        Claude AI model and thinking budget configuration.
    """

    def __init__(
        self,
        api_key: str,
        settings: Any,
    ) -> None:
        self._api_key: str = api_key
        self._settings: Any = settings
        self._log: structlog.stdlib.BoundLogger = get_logger(
            "ai.orchestrator",
        )
        self._graph: StateGraph | None = None
        self._compiled: Any = None

        # Track Claude API availability for fallback decisions
        self._last_api_failure: float = 0.0
        self._consecutive_api_failures: int = 0

        # Lazily initialised agent instances
        self._analysis_agent: Any = None
        self._risk_agent: Any = None
        self._execution_agent: Any = None
        self._journal_agent: Any = None

        self._init_agents()
        self._log.info(
            "orchestrator_initialised",
            model=getattr(settings, "claude_model", "unknown"),
        )

    # ------------------------------------------------------------------
    # Agent initialisation
    # ------------------------------------------------------------------

    def _init_agents(self) -> None:
        """Lazily import and instantiate all four agent classes.

        If the agent modules are not yet available (e.g. during early
        development), the orchestrator logs a warning and leaves the
        agent slots as ``None``.  Pipeline methods check for ``None``
        and fall back to ML-only mode when an agent is missing.
        """
        try:
            from src.ai.analysis_agent import AnalysisAgent

            self._analysis_agent = AnalysisAgent(
                api_key=self._api_key,
                model=getattr(
                    self._settings,
                    "claude_model",
                    "claude-sonnet-4-5-20250929",
                ),
                thinking_budget=getattr(
                    self._settings,
                    "claude_analysis_thinking_budget",
                    8192,
                ),
            )
            self._log.info("analysis_agent_loaded")
        except ImportError:
            self._log.warning(
                "analysis_agent_not_available",
                reason="module not found",
            )

        try:
            from src.ai.risk_agent import RiskAgent

            self._risk_agent = RiskAgent(
                api_key=self._api_key,
                model=getattr(
                    self._settings,
                    "claude_model",
                    "claude-sonnet-4-5-20250929",
                ),
                thinking_budget=getattr(
                    self._settings,
                    "claude_risk_thinking_budget",
                    4096,
                ),
            )
            self._log.info("risk_agent_loaded")
        except ImportError:
            self._log.warning(
                "risk_agent_not_available",
                reason="module not found",
            )

        try:
            from src.ai.execution_agent import ExecutionAgent

            self._execution_agent = ExecutionAgent(
                api_key=self._api_key,
                model=getattr(
                    self._settings,
                    "claude_model",
                    "claude-sonnet-4-5-20250929",
                ),
            )
            self._log.info("execution_agent_loaded")
        except ImportError:
            self._log.warning(
                "execution_agent_not_available",
                reason="module not found",
            )

        try:
            from src.ai.journal_agent import JournalAgent

            self._journal_agent = JournalAgent(
                api_key=self._api_key,
                model=getattr(
                    self._settings,
                    "claude_model",
                    "claude-sonnet-4-5-20250929",
                ),
            )
            self._log.info("journal_agent_loaded")
        except ImportError:
            self._log.warning(
                "journal_agent_not_available",
                reason="module not found",
            )

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def build_graph(self) -> StateGraph:
        """Build and return the LangGraph state machine.

        Graph topology::

            START -> analyze -> evaluate_risk
                evaluate_risk --(approved)--> execute -> END
                evaluate_risk --(rejected)--> log_rejection -> END

        Returns
        -------
        StateGraph
            The compiled LangGraph graph ready for invocation.
        """
        graph = StateGraph(AgentState)

        # Register nodes
        graph.add_node("analyze", self._run_analysis)
        graph.add_node("evaluate_risk", self._run_risk_evaluation)
        graph.add_node("execute", self._run_execution)
        graph.add_node("log_rejection", self._log_rejection)

        # Wire edges
        graph.add_edge(START, "analyze")
        graph.add_edge("analyze", "evaluate_risk")
        graph.add_conditional_edges(
            "evaluate_risk",
            self._route_after_risk,
            {
                "execute": "execute",
                "log_rejection": "log_rejection",
            },
        )
        graph.add_edge("execute", END)
        graph.add_edge("log_rejection", END)

        self._graph = graph
        self._log.info("graph_built")
        return graph

    def _get_compiled_graph(self) -> Any:
        """Return a compiled graph, building it if necessary.

        Returns
        -------
        CompiledGraph
            A LangGraph compiled graph ready for ``ainvoke``.
        """
        if self._compiled is None:
            graph = self.build_graph()
            self._compiled = graph.compile()
        return self._compiled

    # ------------------------------------------------------------------
    # Pipeline execution
    # ------------------------------------------------------------------

    async def run_trading_pipeline(
        self,
        state: AgentState,
    ) -> AgentState:
        """Execute the full trading pipeline through the LangGraph graph.

        Compiles the graph (if not already compiled) and runs it with
        the provided state.  Total latency is tracked via Prometheus.

        Parameters
        ----------
        state:
            Initial pipeline state populated with market context.

        Returns
        -------
        AgentState
            The final state after all agents have executed.
        """
        start_time = time.monotonic()
        self._log.info(
            "pipeline_started",
            ticker=state.get("ticker", "unknown"),
            fallback_mode=state.get("fallback_mode", False),
        )

        try:
            compiled = self._get_compiled_graph()
            result: AgentState = await compiled.ainvoke(state)

            elapsed = time.monotonic() - start_time
            PIPELINE_LATENCY.observe(elapsed)

            self._log.info(
                "pipeline_completed",
                ticker=state.get("ticker", "unknown"),
                elapsed_seconds=round(elapsed, 3),
                proposals=len(result.get("proposals", [])),
                approved=result.get("should_execute", False),
                errors=len(result.get("errors", [])),
            )
            return result

        except Exception as exc:
            elapsed = time.monotonic() - start_time
            PIPELINE_LATENCY.observe(elapsed)
            PIPELINE_ERRORS.labels(agent="pipeline").inc()

            self._log.error(
                "pipeline_failed",
                ticker=state.get("ticker", "unknown"),
                elapsed_seconds=round(elapsed, 3),
                error=str(exc),
                exc_info=True,
            )

            errors = list(state.get("errors", []))
            errors.append(f"Pipeline failure: {exc}")
            state["errors"] = errors
            return state

    async def run_with_fallback(
        self,
        state: AgentState,
    ) -> AgentState:
        """Execute the pipeline with automatic fallback to ML-only mode.

        Attempts the full Claude-powered pipeline first.  If it fails
        (e.g. Claude API is down for more than 2 minutes), the system
        switches to fallback mode where:

        - Trade proposals come from ML ensemble scores and hard-coded
          regime-strategy mappings (no Claude analysis).
        - Risk checks use deterministic rules (no Claude reasoning).
        - Execution plans are generated mechanically.

        Parameters
        ----------
        state:
            Initial pipeline state populated with market context.

        Returns
        -------
        AgentState
            The final state, with ``fallback_mode`` set to ``True``
            if the fallback path was used.
        """
        # Initialise mutable state fields
        state.setdefault("proposals", [])
        state.setdefault("risk_evaluations", [])
        state.setdefault("execution_results", [])
        state.setdefault("journal_entries", [])
        state.setdefault("errors", [])
        state.setdefault("step", "init")
        state.setdefault("should_execute", False)
        state.setdefault("fallback_mode", False)

        # Check if we should go straight to fallback mode
        if self._is_api_unavailable():
            self._log.warning(
                "api_unavailable_entering_fallback",
                consecutive_failures=(self._consecutive_api_failures),
            )
            state["fallback_mode"] = True
            FALLBACK_MODE_ACTIVE.set(1)
            return await self._run_fallback_pipeline(state)

        try:
            result = await self.run_trading_pipeline(state)

            # Reset failure tracking on success
            self._consecutive_api_failures = 0
            FALLBACK_MODE_ACTIVE.set(0)
            return result

        except Exception as exc:
            self._last_api_failure = time.monotonic()
            self._consecutive_api_failures += 1

            self._log.warning(
                "pipeline_failed_trying_fallback",
                error=str(exc),
                consecutive_failures=(self._consecutive_api_failures),
            )

            state["fallback_mode"] = True
            FALLBACK_MODE_ACTIVE.set(1)
            return await self._run_fallback_pipeline(state)

    # ------------------------------------------------------------------
    # Node implementations
    # ------------------------------------------------------------------

    async def _run_analysis(
        self,
        state: AgentState,
    ) -> dict[str, Any]:
        """Analysis node: generate trade proposals.

        Delegates to the Analysis Agent for Claude-powered reasoning.
        Falls back to ML-only proposals if the agent is unavailable
        or the API call fails.

        Parameters
        ----------
        state:
            Current pipeline state with market context.

        Returns
        -------
        dict
            State update containing ``proposals`` and ``step``.
        """
        start_time = time.monotonic()
        state_update: dict[str, Any] = {"step": "analyze"}
        proposals: list[dict[str, Any]] = []

        try:
            if self._analysis_agent is not None and not state.get(
                "fallback_mode", False
            ):
                analysis_input = AnalysisInput(
                    ticker=state.get("ticker", ""),
                    ml_scores=_safe_get_dict(state, "ml_scores"),
                    regime=state.get("regime", "unknown"),
                    iv_rank=state.get("iv_rank", 0.0),
                    sentiment_score=state.get(
                        "sentiment_score",
                        0.0,
                    ),
                    gex_data=_safe_get_dict(state, "gex_data"),
                    options_chain=_safe_get_list(
                        state,
                        "options_chain",
                    ),
                    account_summary=_safe_get_dict(
                        state,
                        "account_summary",
                    ),
                    current_positions=_safe_get_list(
                        state,
                        "current_positions",
                    ),
                    event_calendar=_safe_get_dict(
                        state,
                        "event_calendar",
                    ),
                )

                raw_proposals = await self._call_analysis_agent(
                    analysis_input,
                )

                if raw_proposals:
                    for p in raw_proposals:
                        p["source"] = "claude_analysis"
                    proposals.extend(raw_proposals)
                    PROPOSALS_TOTAL.labels(
                        source="claude",
                    ).inc(len(proposals))
            else:
                # Fallback: generate proposal from ML signals
                fallback = create_ml_fallback_proposal(state)
                if fallback:
                    proposals.append(fallback)
                    PROPOSALS_TOTAL.labels(
                        source="ml_fallback",
                    ).inc()

        except Exception as exc:
            PIPELINE_ERRORS.labels(agent="analysis").inc()
            self._log.error(
                "analysis_node_error",
                ticker=state.get("ticker", "unknown"),
                error=str(exc),
                exc_info=True,
            )

            errors = list(state.get("errors", []))
            errors.append(f"Analysis error: {exc}")
            state_update["errors"] = errors

            # Attempt ML fallback on analysis failure
            fallback = create_ml_fallback_proposal(state)
            if fallback:
                proposals.append(fallback)
                PROPOSALS_TOTAL.labels(
                    source="ml_fallback",
                ).inc()

        elapsed = time.monotonic() - start_time
        AGENT_LATENCY.labels(agent="analysis").observe(elapsed)

        self._log.info(
            "analysis_complete",
            ticker=state.get("ticker", "unknown"),
            proposals_count=len(proposals),
            elapsed_seconds=round(elapsed, 3),
        )

        state_update["proposals"] = proposals
        return state_update

    async def _run_risk_evaluation(
        self,
        state: AgentState,
    ) -> dict[str, Any]:
        """Risk evaluation node: approve, reject, or modify proposals.

        Delegates to the Risk Agent for Claude-powered evaluation.
        Falls back to deterministic risk checks when Claude is
        unavailable.

        Parameters
        ----------
        state:
            Current pipeline state containing proposals to evaluate.

        Returns
        -------
        dict
            State update containing ``risk_evaluations``,
            ``should_execute``, and ``step``.
        """
        start_time = time.monotonic()
        state_update: dict[str, Any] = {"step": "evaluate_risk"}
        evaluations: list[dict[str, Any]] = []
        any_approved = False

        proposals = _safe_get_list(state, "proposals")
        if not proposals:
            self._log.info(
                "risk_no_proposals",
                ticker=state.get("ticker", "unknown"),
            )
            state_update["risk_evaluations"] = []
            state_update["should_execute"] = False
            return state_update

        for idx, proposal in enumerate(proposals):
            try:
                evaluation = await self._evaluate_single_proposal(
                    idx,
                    proposal,
                    state,
                )
                evaluations.append(evaluation)

                if evaluation.get("decision") == "APPROVED":
                    any_approved = True
                    APPROVALS_TOTAL.inc()
                elif evaluation.get("decision") == "REJECTED":
                    reason = evaluation.get("reason", "unknown")
                    REJECTIONS_TOTAL.labels(reason=reason).inc()
                elif evaluation.get("decision") == "MODIFIED":
                    any_approved = True
                    APPROVALS_TOTAL.inc()

            except Exception as exc:
                PIPELINE_ERRORS.labels(agent="risk").inc()
                self._log.error(
                    "risk_evaluation_error",
                    proposal_index=idx,
                    error=str(exc),
                    exc_info=True,
                )

                # Reject on error -- safety first
                evaluations.append(
                    {
                        "proposal_index": idx,
                        "decision": "REJECTED",
                        "reason": f"Risk evaluation error: {exc}",
                        "modifications": {},
                        "risk_score": 1.0,
                    }
                )
                REJECTIONS_TOTAL.labels(
                    reason="evaluation_error",
                ).inc()

        elapsed = time.monotonic() - start_time
        AGENT_LATENCY.labels(agent="risk").observe(elapsed)

        self._log.info(
            "risk_evaluation_complete",
            ticker=state.get("ticker", "unknown"),
            total_proposals=len(proposals),
            approved=sum(
                1 for e in evaluations if e.get("decision") in ("APPROVED", "MODIFIED")
            ),
            rejected=sum(1 for e in evaluations if e.get("decision") == "REJECTED"),
            elapsed_seconds=round(elapsed, 3),
        )

        state_update["risk_evaluations"] = evaluations
        state_update["should_execute"] = any_approved
        return state_update

    def _route_after_risk(
        self,
        state: AgentState,
    ) -> str:
        """Conditional edge: route to execution or rejection logging.

        Parameters
        ----------
        state:
            Current pipeline state after risk evaluation.

        Returns
        -------
        str
            ``"execute"`` if any proposal was approved, otherwise
            ``"log_rejection"``.
        """
        should_execute = state.get("should_execute", False)
        self._log.debug(
            "routing_after_risk",
            should_execute=should_execute,
        )
        if should_execute:
            return "execute"
        return "log_rejection"

    async def _run_execution(
        self,
        state: AgentState,
    ) -> dict[str, Any]:
        """Execution node: build order specs for approved proposals.

        Delegates to the Execution Agent for Claude-powered order
        translation.  Falls back to mechanical order construction
        when Claude is unavailable.

        Parameters
        ----------
        state:
            Current pipeline state containing approved proposals.

        Returns
        -------
        dict
            State update containing ``execution_results`` and ``step``.
        """
        start_time = time.monotonic()
        state_update: dict[str, Any] = {"step": "execute"}
        results: list[dict[str, Any]] = []

        proposals = _safe_get_list(state, "proposals")
        evaluations = _safe_get_list(state, "risk_evaluations")

        # Build a map of approved proposal indices
        approved_indices: set[int] = set()
        modifications_by_index: dict[int, dict[str, Any]] = {}
        for evaluation in evaluations:
            decision = evaluation.get("decision", "")
            idx = evaluation.get("proposal_index", -1)
            if decision == "APPROVED":
                approved_indices.add(idx)
            elif decision == "MODIFIED":
                approved_indices.add(idx)
                modifications_by_index[idx] = evaluation.get(
                    "modifications",
                    {},
                )

        for idx in sorted(approved_indices):
            if idx >= len(proposals):
                self._log.warning(
                    "execution_index_out_of_range",
                    proposal_index=idx,
                    total_proposals=len(proposals),
                )
                continue

            proposal = dict(proposals[idx])

            # Apply modifications from risk agent if any
            if idx in modifications_by_index:
                mods = modifications_by_index[idx]
                proposal["parameters"] = {
                    **proposal.get("parameters", {}),
                    **mods,
                }
                self._log.info(
                    "proposal_modified_by_risk",
                    proposal_index=idx,
                    modifications=mods,
                )

            try:
                result = await self._execute_single_proposal(
                    idx,
                    proposal,
                    state,
                )
                results.append(result)
            except Exception as exc:
                PIPELINE_ERRORS.labels(agent="execution").inc()
                self._log.error(
                    "execution_error",
                    proposal_index=idx,
                    error=str(exc),
                    exc_info=True,
                )
                results.append(
                    {
                        "proposal_index": idx,
                        "status": "FAILED",
                        "order_spec": {},
                        "error": str(exc),
                    }
                )

        elapsed = time.monotonic() - start_time
        AGENT_LATENCY.labels(agent="execution").observe(elapsed)

        planned = sum(1 for r in results if r.get("status") == "PLANNED")
        failed = sum(1 for r in results if r.get("status") == "FAILED")
        self._log.info(
            "execution_complete",
            ticker=state.get("ticker", "unknown"),
            planned=planned,
            failed=failed,
            elapsed_seconds=round(elapsed, 3),
        )

        state_update["execution_results"] = results
        return state_update

    async def _log_rejection(
        self,
        state: AgentState,
    ) -> dict[str, Any]:
        """Rejection logging node: record all rejected proposals.

        Logs each rejection reason, updates Prometheus counters, and
        appends human-readable messages to the state errors list.

        Parameters
        ----------
        state:
            Current pipeline state after risk evaluation rejected
            all proposals.

        Returns
        -------
        dict
            State update containing ``errors`` and ``step``.
        """
        state_update: dict[str, Any] = {"step": "log_rejection"}
        rejection_errors: list[str] = list(
            state.get("errors", []),
        )

        evaluations = _safe_get_list(state, "risk_evaluations")
        proposals = _safe_get_list(state, "proposals")

        for evaluation in evaluations:
            decision = evaluation.get("decision", "")
            if decision != "REJECTED":
                continue

            idx = evaluation.get("proposal_index", -1)
            reason = evaluation.get("reason", "unknown")
            risk_score = evaluation.get("risk_score", 0.0)

            ticker = "unknown"
            strategy = "unknown"
            if 0 <= idx < len(proposals):
                ticker = proposals[idx].get("ticker", "unknown")
                strategy = proposals[idx].get(
                    "strategy",
                    "unknown",
                )

            msg = (
                f"Proposal {idx} rejected: {ticker} "
                f"{strategy} -- {reason} "
                f"(risk_score={risk_score:.2f})"
            )
            rejection_errors.append(msg)

            self._log.info(
                "proposal_rejected",
                proposal_index=idx,
                ticker=ticker,
                strategy=strategy,
                reason=reason,
                risk_score=round(risk_score, 4),
            )

        state_update["errors"] = rejection_errors
        return state_update

    # ------------------------------------------------------------------
    # Journal (separate method, not part of the main graph)
    # ------------------------------------------------------------------

    async def run_journal(
        self,
        closed_trades: list[dict[str, Any]],
    ) -> JournalEntry:
        """Run the Journal Agent on closed trades (end-of-day batch).

        This method is invoked separately from the main trading
        pipeline, typically at 4:15 PM ET after market close.

        Parameters
        ----------
        closed_trades:
            List of trade dicts representing positions closed today.

        Returns
        -------
        JournalEntry
            Structured journal entry with insights, patterns, and
            memory updates.
        """
        start_time = time.monotonic()
        self._log.info(
            "journal_started",
            trades_count=len(closed_trades),
        )

        try:
            if self._journal_agent is not None and not self._is_api_unavailable():
                raw_entry = await self._call_journal_agent(
                    closed_trades,
                )
                entry = JournalEntry(**raw_entry)
            else:
                # Mechanical journal when Claude is unavailable
                entry = self._create_mechanical_journal(
                    closed_trades,
                )

        except Exception as exc:
            PIPELINE_ERRORS.labels(agent="journal").inc()
            self._log.error(
                "journal_error",
                error=str(exc),
                exc_info=True,
            )
            entry = self._create_mechanical_journal(closed_trades)

        elapsed = time.monotonic() - start_time
        AGENT_LATENCY.labels(agent="journal").observe(elapsed)

        self._log.info(
            "journal_complete",
            trades_reviewed=entry.trades_reviewed,
            total_pnl=round(entry.total_pnl, 2),
            insights_count=len(entry.insights),
            elapsed_seconds=round(elapsed, 3),
        )

        return entry

    # ------------------------------------------------------------------
    # Internal helpers: agent delegation
    # ------------------------------------------------------------------

    @retry(
        retry=retry_if_exception_type((TimeoutError, ConnectionError)),
        stop=stop_after_attempt(MAX_PIPELINE_RETRIES),
        wait=wait_exponential(
            min=RETRY_MIN_WAIT_SECONDS,
            max=RETRY_MAX_WAIT_SECONDS,
        ),
        reraise=True,
    )
    async def _call_analysis_agent(
        self,
        analysis_input: AnalysisInput,
    ) -> list[dict[str, Any]]:
        """Call the Analysis Agent with retry logic.

        Parameters
        ----------
        analysis_input:
            Structured input for the analysis agent.

        Returns
        -------
        list[dict]
            List of trade proposal dicts from Claude's analysis.

        Raises
        ------
        TimeoutError
            If the Claude API does not respond within the timeout.
        ConnectionError
            If the Claude API is unreachable.
        """
        with API_LATENCY.labels(api="claude_analysis").time():
            result = await self._analysis_agent.analyze(
                analysis_input.model_dump(),
            )

        if isinstance(result, dict):
            return [result]
        if isinstance(result, list):
            return result
        return []

    @retry(
        retry=retry_if_exception_type((TimeoutError, ConnectionError)),
        stop=stop_after_attempt(MAX_PIPELINE_RETRIES),
        wait=wait_exponential(
            min=RETRY_MIN_WAIT_SECONDS,
            max=RETRY_MAX_WAIT_SECONDS,
        ),
        reraise=True,
    )
    async def _call_risk_agent(
        self,
        risk_context: RiskContext,
    ) -> dict[str, Any]:
        """Call the Risk Agent with retry logic.

        Parameters
        ----------
        risk_context:
            Structured input for the risk agent.

        Returns
        -------
        dict
            Risk evaluation dict with decision, reason, and
            optional modifications.
        """
        with API_LATENCY.labels(api="claude_risk").time():
            result = await self._risk_agent.evaluate(
                risk_context.model_dump(),
            )

        if isinstance(result, dict):
            return result
        return {
            "decision": "REJECTED",
            "reason": "Invalid risk agent response",
            "risk_score": 1.0,
        }

    @retry(
        retry=retry_if_exception_type((TimeoutError, ConnectionError)),
        stop=stop_after_attempt(MAX_PIPELINE_RETRIES),
        wait=wait_exponential(
            min=RETRY_MIN_WAIT_SECONDS,
            max=RETRY_MAX_WAIT_SECONDS,
        ),
        reraise=True,
    )
    async def _call_execution_agent(
        self,
        proposal: dict[str, Any],
    ) -> dict[str, Any]:
        """Call the Execution Agent with retry logic.

        Parameters
        ----------
        proposal:
            Approved trade proposal to translate into an order spec.

        Returns
        -------
        dict
            Execution result with order specification.
        """
        with API_LATENCY.labels(api="claude_execution").time():
            result = await self._execution_agent.plan_execution(
                proposal,
            )

        if isinstance(result, dict):
            return result
        return {
            "status": "FAILED",
            "error": "Invalid execution agent response",
        }

    @retry(
        retry=retry_if_exception_type((TimeoutError, ConnectionError)),
        stop=stop_after_attempt(MAX_PIPELINE_RETRIES),
        wait=wait_exponential(
            min=RETRY_MIN_WAIT_SECONDS,
            max=RETRY_MAX_WAIT_SECONDS,
        ),
        reraise=True,
    )
    async def _call_journal_agent(
        self,
        closed_trades: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Call the Journal Agent with retry logic.

        Parameters
        ----------
        closed_trades:
            List of closed trade dicts to review.

        Returns
        -------
        dict
            Journal entry dict with insights and memory updates.
        """
        with API_LATENCY.labels(api="claude_journal").time():
            result = await self._journal_agent.review_trades(
                closed_trades,
            )

        if isinstance(result, dict):
            return result
        return {
            "date": "",
            "trades_reviewed": 0,
            "insights": [],
        }

    # ------------------------------------------------------------------
    # Internal helpers: per-proposal evaluation & execution
    # ------------------------------------------------------------------

    async def _evaluate_single_proposal(
        self,
        idx: int,
        proposal: dict[str, Any],
        state: AgentState,
    ) -> dict[str, Any]:
        """Evaluate a single proposal through the Risk Agent or fallback.

        Parameters
        ----------
        idx:
            Index of the proposal in the proposals list.
        proposal:
            The trade proposal dict.
        state:
            Current pipeline state.

        Returns
        -------
        dict
            Risk evaluation dict.
        """
        if self._risk_agent is not None and not state.get("fallback_mode", False):
            risk_context = RiskContext(
                proposal=proposal,
                account_summary=_safe_get_dict(
                    state,
                    "account_summary",
                ),
                current_positions=_safe_get_list(
                    state,
                    "current_positions",
                ),
                portfolio_greeks=_safe_get_dict(
                    state,
                    "portfolio_greeks",
                ),
                circuit_breaker_state=_safe_get_dict(
                    state,
                    "circuit_breaker_state",
                ),
                correlation_data=_safe_get_dict(
                    state,
                    "correlation_data",
                ),
                event_calendar=_safe_get_dict(
                    state,
                    "event_calendar",
                ),
            )
            raw_eval = await self._call_risk_agent(risk_context)
            raw_eval.setdefault("proposal_index", idx)
            return raw_eval

        # Deterministic fallback risk checks
        return self._deterministic_risk_check(idx, proposal, state)

    async def _execute_single_proposal(
        self,
        idx: int,
        proposal: dict[str, Any],
        state: AgentState,
    ) -> dict[str, Any]:
        """Execute a single approved proposal via the Execution Agent.

        Parameters
        ----------
        idx:
            Index of the proposal.
        proposal:
            The (possibly modified) trade proposal.
        state:
            Current pipeline state.

        Returns
        -------
        dict
            Execution result dict.
        """
        if self._execution_agent is not None and not state.get("fallback_mode", False):
            raw_result = await self._call_execution_agent(proposal)
            raw_result.setdefault("proposal_index", idx)
            return raw_result

        # Mechanical execution plan
        return self._mechanical_execution_plan(idx, proposal)

    # ------------------------------------------------------------------
    # Fallback pipeline (no Claude)
    # ------------------------------------------------------------------

    async def _run_fallback_pipeline(
        self,
        state: AgentState,
    ) -> AgentState:
        """Execute a simplified pipeline using only ML signals and rules.

        Bypasses Claude entirely and uses:
        - ``create_ml_fallback_proposal`` for analysis
        - ``_deterministic_risk_check`` for risk
        - ``_mechanical_execution_plan`` for execution

        Parameters
        ----------
        state:
            Initial pipeline state.

        Returns
        -------
        AgentState
            Final state after the fallback pipeline completes.
        """
        start_time = time.monotonic()
        self._log.info(
            "fallback_pipeline_started",
            ticker=state.get("ticker", "unknown"),
        )

        # Step 1: Generate fallback proposal
        proposal = create_ml_fallback_proposal(state)
        proposals: list[dict[str, Any]] = [proposal] if proposal else []
        state["proposals"] = proposals
        state["step"] = "analyze"

        if not proposals:
            state["should_execute"] = False
            state["risk_evaluations"] = []
            state["execution_results"] = []
            elapsed = time.monotonic() - start_time
            PIPELINE_LATENCY.observe(elapsed)
            self._log.info(
                "fallback_no_proposals",
                ticker=state.get("ticker", "unknown"),
                elapsed_seconds=round(elapsed, 3),
            )
            return state

        # Step 2: Deterministic risk check
        evaluations: list[dict[str, Any]] = []
        any_approved = False
        for idx, prop in enumerate(proposals):
            evaluation = self._deterministic_risk_check(
                idx,
                prop,
                state,
            )
            evaluations.append(evaluation)
            if evaluation.get("decision") in (
                "APPROVED",
                "MODIFIED",
            ):
                any_approved = True
                APPROVALS_TOTAL.inc()
            else:
                REJECTIONS_TOTAL.labels(
                    reason=evaluation.get("reason", "unknown"),
                ).inc()

        state["risk_evaluations"] = evaluations
        state["should_execute"] = any_approved
        state["step"] = "evaluate_risk"

        # Step 3: Mechanical execution if approved
        if any_approved:
            results: list[dict[str, Any]] = []
            for evaluation in evaluations:
                if evaluation.get("decision") in (
                    "APPROVED",
                    "MODIFIED",
                ):
                    idx = evaluation.get("proposal_index", 0)
                    if idx < len(proposals):
                        result = self._mechanical_execution_plan(
                            idx,
                            proposals[idx],
                        )
                        results.append(result)
            state["execution_results"] = results
            state["step"] = "execute"
        else:
            state["execution_results"] = []
            state["step"] = "log_rejection"

        elapsed = time.monotonic() - start_time
        PIPELINE_LATENCY.observe(elapsed)

        self._log.info(
            "fallback_pipeline_complete",
            ticker=state.get("ticker", "unknown"),
            proposals=len(proposals),
            approved=any_approved,
            elapsed_seconds=round(elapsed, 3),
        )

        return state

    # ------------------------------------------------------------------
    # Deterministic risk checks (Claude-free)
    # ------------------------------------------------------------------

    def _deterministic_risk_check(
        self,
        idx: int,
        proposal: dict[str, Any],
        state: AgentState,
    ) -> dict[str, Any]:
        """Apply hard-coded risk rules without Claude.

        Checks performed:
        1. Circuit breaker is not in HALT or EMERGENCY.
        2. Number of open positions is below the maximum.
        3. Confidence score exceeds the threshold.
        4. No conflicting earnings event in the event calendar.

        Parameters
        ----------
        idx:
            Proposal index.
        proposal:
            Trade proposal dict.
        state:
            Current pipeline state.

        Returns
        -------
        dict
            Risk evaluation dict with decision and reason.
        """
        cb_state = _safe_get_dict(state, "circuit_breaker_state")
        cb_level = cb_state.get("level", "NORMAL")
        positions = _safe_get_list(state, "current_positions")
        confidence = float(proposal.get("confidence", 0.0))
        ticker = proposal.get("ticker", "")

        # Check 1: Circuit breaker
        if cb_level in ("HALT", "EMERGENCY"):
            self._log.info(
                "deterministic_reject_circuit_breaker",
                proposal_index=idx,
                cb_level=cb_level,
            )
            return {
                "proposal_index": idx,
                "decision": "REJECTED",
                "reason": (
                    f"Circuit breaker at {cb_level} -- no new positions allowed"
                ),
                "modifications": {},
                "risk_score": 1.0,
            }

        # Check 2: Position limit
        max_positions = 8  # from TradingSettings default
        account_summary = _safe_get_dict(state, "account_summary")
        max_pos = int(
            account_summary.get(
                "max_concurrent_positions",
                max_positions,
            ),
        )
        if len(positions) >= max_pos:
            self._log.info(
                "deterministic_reject_position_limit",
                proposal_index=idx,
                current_positions=len(positions),
                max_positions=max_pos,
            )
            return {
                "proposal_index": idx,
                "decision": "REJECTED",
                "reason": (f"Position limit reached: {len(positions)}/{max_pos}"),
                "modifications": {},
                "risk_score": 0.8,
            }

        # Check 3: Confidence threshold
        if confidence < CONFIDENCE_THRESHOLD:
            self._log.info(
                "deterministic_reject_low_confidence",
                proposal_index=idx,
                confidence=round(confidence, 4),
                threshold=CONFIDENCE_THRESHOLD,
            )
            return {
                "proposal_index": idx,
                "decision": "REJECTED",
                "reason": (
                    f"Confidence {confidence:.4f} below "
                    f"threshold {CONFIDENCE_THRESHOLD}"
                ),
                "modifications": {},
                "risk_score": 0.6,
            }

        # Check 4: Event calendar conflict
        event_cal = _safe_get_dict(state, "event_calendar")
        blocked_tickers = event_cal.get("blocked_tickers", [])
        if ticker in blocked_tickers:
            self._log.info(
                "deterministic_reject_event_conflict",
                proposal_index=idx,
                ticker=ticker,
            )
            return {
                "proposal_index": idx,
                "decision": "REJECTED",
                "reason": (
                    f"Event conflict: {ticker} has upcoming earnings or macro event"
                ),
                "modifications": {},
                "risk_score": 0.7,
            }

        # Check 5: Reduce size if circuit breaker is CAUTION/WARNING
        if cb_level in ("CAUTION", "WARNING"):
            size_factor = 0.5 if cb_level == "CAUTION" else 0.25
            self._log.info(
                "deterministic_modify_size",
                proposal_index=idx,
                cb_level=cb_level,
                size_factor=size_factor,
            )
            return {
                "proposal_index": idx,
                "decision": "MODIFIED",
                "reason": (
                    f"Circuit breaker at {cb_level} -- "
                    f"reducing size to {size_factor:.0%}"
                ),
                "modifications": {
                    "size_factor": size_factor,
                },
                "risk_score": 0.5,
            }

        # All checks passed
        return {
            "proposal_index": idx,
            "decision": "APPROVED",
            "reason": "All deterministic risk checks passed",
            "modifications": {},
            "risk_score": 0.2,
        }

    # ------------------------------------------------------------------
    # Mechanical execution plan (Claude-free)
    # ------------------------------------------------------------------

    def _mechanical_execution_plan(
        self,
        idx: int,
        proposal: dict[str, Any],
    ) -> dict[str, Any]:
        """Build a basic execution plan without Claude.

        Creates an order specification from the proposal's strategy
        and parameters.  The actual order placement is handled
        downstream by the broker layer.

        Parameters
        ----------
        idx:
            Proposal index.
        proposal:
            Approved trade proposal.

        Returns
        -------
        dict
            Execution result with order spec.
        """
        order_spec = {
            "ticker": proposal.get("ticker", ""),
            "strategy": proposal.get("strategy", ""),
            "direction": proposal.get("direction", ""),
            "legs": proposal.get("legs", []),
            "order_type": "LMT",
            "tif": "DAY",
            "source": "mechanical_fallback",
        }

        params = proposal.get("parameters", {})
        if params.get("size_factor"):
            order_spec["size_factor"] = params["size_factor"]

        self._log.info(
            "mechanical_execution_plan",
            proposal_index=idx,
            ticker=order_spec["ticker"],
            strategy=order_spec["strategy"],
        )

        return {
            "proposal_index": idx,
            "status": "PLANNED",
            "order_spec": order_spec,
            "error": "",
        }

    # ------------------------------------------------------------------
    # Mechanical journal (Claude-free)
    # ------------------------------------------------------------------

    def _create_mechanical_journal(
        self,
        closed_trades: list[dict[str, Any]],
    ) -> JournalEntry:
        """Create a basic journal entry from trade statistics.

        When Claude is unavailable, the journal captures quantitative
        metrics without qualitative AI insights.

        Parameters
        ----------
        closed_trades:
            List of closed trade dicts.

        Returns
        -------
        JournalEntry
            A mechanical journal entry with basic statistics.
        """
        from datetime import datetime

        total_pnl = sum(float(t.get("realized_pnl", 0.0)) for t in closed_trades)
        winners = sum(1 for t in closed_trades if float(t.get("realized_pnl", 0.0)) > 0)
        total = len(closed_trades)
        win_rate = winners / total if total > 0 else 0.0

        insights: list[str] = []
        if total > 0:
            avg_pnl = total_pnl / total
            insights.append(
                f"Average P&L per trade: ${avg_pnl:.2f}",
            )
            insights.append(
                f"Win rate: {win_rate:.1%} ({winners}/{total})",
            )
            if total_pnl > 0:
                insights.append("Net positive day")
            elif total_pnl < 0:
                insights.append("Net negative day")

        # Detect simple patterns
        patterns: list[str] = []
        strategies_used = set(t.get("strategy", "") for t in closed_trades)
        for strategy in strategies_used:
            strat_trades = [t for t in closed_trades if t.get("strategy") == strategy]
            strat_pnl = sum(float(t.get("realized_pnl", 0.0)) for t in strat_trades)
            if strat_pnl < 0 and len(strat_trades) >= 2:
                patterns.append(
                    f"{strategy}: {len(strat_trades)} trades, "
                    f"net loss ${strat_pnl:.2f}",
                )

        return JournalEntry(
            date=datetime.now(UTC).strftime("%Y-%m-%d"),
            trades_reviewed=total,
            total_pnl=round(total_pnl, 2),
            win_rate=round(win_rate, 4),
            insights=insights,
            patterns_detected=patterns,
            memory_updates={},
            recommendations=[
                "Claude API unavailable -- journal generated "
                "mechanically. Review manually.",
            ],
        )

    # ------------------------------------------------------------------
    # API availability tracking
    # ------------------------------------------------------------------

    def _is_api_unavailable(self) -> bool:
        """Check whether the Claude API is considered unavailable.

        The API is considered unavailable if it has failed consecutively
        and the last failure was within the timeout window.

        Returns
        -------
        bool
            ``True`` if the system should use fallback mode.
        """
        if self._consecutive_api_failures == 0:
            return False

        elapsed = time.monotonic() - self._last_api_failure
        if elapsed > CLAUDE_API_TIMEOUT_SECONDS:
            # Enough time has passed; try the API again
            self._log.info(
                "api_cooldown_expired",
                elapsed_seconds=round(elapsed, 1),
                resetting_failure_count=True,
            )
            self._consecutive_api_failures = 0
            return False

        # API has been failing and we're still within the timeout
        return self._consecutive_api_failures >= 2

    # ------------------------------------------------------------------
    # State factory
    # ------------------------------------------------------------------

    @staticmethod
    def create_initial_state(
        ticker: str,
        ml_scores: dict[str, Any] | None = None,
        regime: str = "unknown",
        iv_rank: float = 0.0,
        sentiment_score: float = 0.0,
        gex_data: dict[str, Any] | None = None,
        options_chain: list[dict[str, Any]] | None = None,
        account_summary: dict[str, Any] | None = None,
        current_positions: list[dict[str, Any]] | None = None,
        portfolio_greeks: dict[str, Any] | None = None,
        circuit_breaker_state: dict[str, Any] | None = None,
        event_calendar: dict[str, Any] | None = None,
        correlation_data: dict[str, Any] | None = None,
    ) -> AgentState:
        """Create a properly initialised AgentState dict.

        Convenience factory that ensures all required fields are
        present with sensible defaults so callers do not need to
        populate every key manually.

        Parameters
        ----------
        ticker:
            Underlying symbol to analyse.
        ml_scores:
            ML ensemble scores.
        regime:
            Current market regime.
        iv_rank:
            Current IV Rank (0-100).
        sentiment_score:
            FinBERT rolling sentiment (-1 to 1).
        gex_data:
            Gamma Exposure data.
        options_chain:
            Options chain snapshot.
        account_summary:
            Account equity and buying power.
        current_positions:
            Open positions.
        portfolio_greeks:
            Aggregate portfolio Greeks.
        circuit_breaker_state:
            Circuit-breaker level and P&L.
        event_calendar:
            Upcoming event data.
        correlation_data:
            Rolling correlation matrix.

        Returns
        -------
        AgentState
            A fully initialised state dict ready for the pipeline.
        """
        return AgentState(
            ticker=ticker,
            ml_scores=ml_scores or {},
            regime=regime,
            iv_rank=iv_rank,
            sentiment_score=sentiment_score,
            gex_data=gex_data or {},
            options_chain=options_chain or [],
            account_summary=account_summary or {},
            current_positions=current_positions or [],
            portfolio_greeks=portfolio_greeks or {},
            circuit_breaker_state=circuit_breaker_state or {},
            event_calendar=event_calendar or {},
            correlation_data=correlation_data or {},
            proposals=[],
            risk_evaluations=[],
            execution_results=[],
            journal_entries=[],
            errors=[],
            step="init",
            should_execute=False,
            fallback_mode=False,
        )
