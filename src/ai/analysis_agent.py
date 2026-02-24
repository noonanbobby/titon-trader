"""Analysis Agent for Project Titan.

Uses the Anthropic Claude API with extended thinking to evaluate trade
opportunities.  The agent receives a structured snapshot of market
conditions -- ML confidence scores, regime state, IV rank, sentiment,
GEX regime, and an options-chain summary -- and returns zero or more
:class:`TradeProposal` objects describing recommended positions.

Prompt caching is enabled on the system message so that repeated calls
within a five-minute window benefit from reduced input-token costs.

Usage::

    from src.ai.analysis_agent import AnalysisAgent, AnalysisInput

    agent = AnalysisAgent(api_key="sk-ant-...")
    result = await agent.analyze(AnalysisInput(
        ticker="AAPL",
        ml_confidence=0.85,
        regime="low_vol_trend",
        ...
    ))
    for proposal in result.proposals:
        print(proposal.ticker, proposal.strategy, proposal.confidence)
"""

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING, Any

from anthropic import AsyncAnthropic
from pydantic import BaseModel, Field
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from src.ai.prompts import ANALYSIS_AGENT_SYSTEM_PROMPT
from src.utils.logging import get_logger
from src.utils.metrics import API_LATENCY

if TYPE_CHECKING:
    import structlog
    from anthropic.types import Message

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Claude Sonnet pricing (per million tokens)
PRICE_INPUT_PER_M: float = 3.0
PRICE_CACHED_INPUT_PER_M: float = 0.30
PRICE_OUTPUT_PER_M: float = 15.0
PRICE_THINKING_PER_M: float = 15.0

# Default limits
DEFAULT_MODEL: str = "claude-sonnet-4-6"
DEFAULT_THINKING_BUDGET: int = 16384
DEFAULT_MAX_TOKENS: int = 4096
BATCH_DELAY_SECONDS: float = 1.0

# Valid strategy names that Claude may recommend
VALID_STRATEGIES: frozenset[str] = frozenset(
    {
        "bull_call_spread",
        "bull_put_spread",
        "iron_condor",
        "calendar_spread",
        "diagonal_spread",
        "broken_wing_butterfly",
        "short_strangle",
        "pmcc",
        "ratio_spread",
        "long_straddle",
    }
)


# ---------------------------------------------------------------------------
# Retry predicate
# ---------------------------------------------------------------------------


def _is_transient_error(exc: BaseException) -> bool:
    """Return ``True`` for HTTP 429 (rate limit) and 529 (overloaded).

    These are the two transient error codes documented by the Anthropic
    API that should be retried with exponential backoff.
    """
    from anthropic import APIStatusError

    if isinstance(exc, APIStatusError):
        return exc.status_code in (429, 529)
    return False


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class TradeProposal(BaseModel):
    """A single trade recommendation produced by the Analysis Agent.

    Attributes:
        ticker: Underlying symbol (e.g. ``"AAPL"``).
        strategy: Canonical strategy name (one of the ten strategies).
        direction: ``LONG`` (debit) or ``SHORT`` (credit).
        confidence: Agent's confidence in the trade (0.0--1.0).
        strikes: Mapping of leg roles to strike prices.
        expiry: Target expiration date in ISO-8601 format.
        quantity: Number of spread units to trade.
        expected_credit_debit: Expected net premium (positive = debit,
            negative = credit).
        max_profit: Maximum theoretical profit in dollars.
        max_loss: Maximum theoretical loss in dollars (positive number).
        reward_risk_ratio: ``max_profit / max_loss``.
        reasoning: Natural-language analysis reasoning from Claude.
        thinking_summary: Summary of the extended thinking process.
    """

    ticker: str = Field(
        ...,
        min_length=1,
        max_length=10,
        description="Underlying ticker symbol",
    )
    strategy: str = Field(
        ...,
        description="Canonical strategy name",
    )
    direction: str = Field(
        ...,
        description="LONG (debit) or SHORT (credit)",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Agent confidence in the trade (0.0-1.0)",
    )
    strikes: dict[str, float] = Field(
        default_factory=dict,
        description="Leg role to strike price mapping",
    )
    expiry: str = Field(
        ...,
        description="Target expiration date (ISO-8601 format)",
    )
    quantity: int = Field(
        ...,
        ge=1,
        description="Number of spread units to trade",
    )
    expected_credit_debit: float = Field(
        ...,
        description=("Expected net premium per unit (positive=debit, negative=credit)"),
    )
    max_profit: float = Field(
        ...,
        description="Maximum theoretical profit in dollars",
    )
    max_loss: float = Field(
        ...,
        gt=0,
        description="Maximum theoretical loss in dollars",
    )
    reward_risk_ratio: float = Field(
        ...,
        ge=0.0,
        description="Reward-to-risk ratio (max_profit / max_loss)",
    )
    reasoning: str = Field(
        default="",
        description="Natural-language analysis reasoning",
    )
    thinking_summary: str = Field(
        default="",
        description="Summary of the extended thinking process",
    )


class AnalysisInput(BaseModel):
    """Structured input provided to the Analysis Agent for evaluation.

    Attributes:
        ticker: Underlying symbol to analyze.
        ml_confidence: Ensemble ML confidence score (0.0--1.0).
        regime: Current market regime identifier.
        iv_rank: Current IV Rank (0--100).
        sentiment_score: FinBERT rolling sentiment (-1.0 to 1.0).
        gex_regime: GEX regime (``"positive"`` or ``"negative"``).
        options_chain_summary: Key statistics from the options chain.
        account_equity: Current account net liquidation value.
        current_positions: List of dicts describing open positions.
    """

    ticker: str = Field(
        ...,
        min_length=1,
        max_length=10,
        description="Ticker symbol to analyze",
    )
    ml_confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Ensemble ML confidence score",
    )
    regime: str = Field(
        ...,
        description="Current market regime identifier",
    )
    iv_rank: float = Field(
        ...,
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
    gex_regime: str = Field(
        default="neutral",
        description="GEX regime (positive, negative, neutral)",
    )
    options_chain_summary: dict[str, Any] = Field(
        default_factory=dict,
        description="Key statistics from the options chain",
    )
    account_equity: float = Field(
        default=150_000.0,
        gt=0,
        description="Current account net liquidation value",
    )
    current_positions: list[dict[str, Any]] = Field(
        default_factory=list,
        description="List of currently open positions",
    )
    memory_context: str = Field(
        default="",
        description="FinMem context with recent outcomes and regime patterns",
    )


class AnalysisResult(BaseModel):
    """Complete output from a single Analysis Agent invocation.

    Attributes:
        proposals: Zero or more trade proposals.
        raw_response: The full text block from the API response.
        thinking_text: The full extended thinking text.
        tokens_used: Total tokens consumed (input + output).
        thinking_tokens: Tokens used for extended thinking.
        latency_ms: Round-trip API latency in milliseconds.
        cost_estimate: Estimated cost of the API call in USD.
    """

    proposals: list[TradeProposal] = Field(
        default_factory=list,
        description="Trade proposals generated by the agent",
    )
    raw_response: str = Field(
        default="",
        description="Full text block from Claude response",
    )
    thinking_text: str = Field(
        default="",
        description="Extended thinking text from Claude",
    )
    tokens_used: int = Field(
        default=0,
        ge=0,
        description="Total tokens consumed (input + output)",
    )
    thinking_tokens: int = Field(
        default=0,
        ge=0,
        description="Tokens used for extended thinking",
    )
    latency_ms: int = Field(
        default=0,
        ge=0,
        description="Round-trip API latency in milliseconds",
    )
    cost_estimate: float = Field(
        default=0.0,
        ge=0.0,
        description="Estimated API call cost in USD",
    )


# ---------------------------------------------------------------------------
# AnalysisAgent
# ---------------------------------------------------------------------------


class AnalysisAgent:
    """Claude-powered trade analysis agent with extended thinking.

    Sends structured market data to the Anthropic API and parses the
    response into typed :class:`TradeProposal` objects.  Extended
    thinking is always enabled so Claude can reason through complex
    multi-factor trade decisions before committing to a recommendation.

    Args:
        api_key: Anthropic API key.
        model: Claude model identifier.  Defaults to
            ``claude-sonnet-4-5-20250929``.
        thinking_budget: Maximum tokens allocated to the extended
            thinking block.  Defaults to ``8192``.
    """

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        thinking_budget: int = DEFAULT_THINKING_BUDGET,
    ) -> None:
        self._client: AsyncAnthropic = AsyncAnthropic(api_key=api_key)
        self._model: str = model
        self._thinking_budget: int = thinking_budget
        self._log: structlog.stdlib.BoundLogger = get_logger(
            "ai.analysis_agent",
        )
        self._log.info(
            "analysis_agent_initialized",
            model=self._model,
            thinking_budget=self._thinking_budget,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def analyze(self, input_data: AnalysisInput) -> AnalysisResult:
        """Analyze a single ticker and return trade proposals.

        Builds a user message from the supplied :class:`AnalysisInput`,
        calls the Anthropic API with extended thinking, and parses the
        response into :class:`TradeProposal` objects.

        Args:
            input_data: Structured market snapshot for one ticker.

        Returns:
            An :class:`AnalysisResult` containing proposals, raw
            response text, thinking text, token usage, latency, and
            estimated cost.
        """
        self._log.info(
            "analysis_started",
            ticker=input_data.ticker,
            ml_confidence=input_data.ml_confidence,
            regime=input_data.regime,
            iv_rank=input_data.iv_rank,
        )

        user_message = self._format_user_message(input_data)
        start_time = time.monotonic()

        try:
            response = await self._call_api(user_message)
        except Exception as exc:
            elapsed_ms = int(
                (time.monotonic() - start_time) * 1000,
            )
            self._log.error(
                "analysis_api_error",
                ticker=input_data.ticker,
                error=str(exc),
                latency_ms=elapsed_ms,
            )
            return AnalysisResult(latency_ms=elapsed_ms)

        elapsed_ms = int((time.monotonic() - start_time) * 1000)

        # Extract content blocks by type
        thinking_text = self._extract_thinking(response)
        raw_text = self._extract_text(response)

        # Token accounting
        usage = response.usage
        input_tokens = usage.input_tokens
        output_tokens = usage.output_tokens
        # cache_read_input_tokens may not exist on all SDK versions
        cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
        thinking_tokens = getattr(usage, "thinking_tokens", 0) or 0

        total_tokens = input_tokens + output_tokens
        cost = self._estimate_cost(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            thinking_tokens=thinking_tokens,
            cached_input_tokens=cache_read,
        )

        # Parse proposals from the text response
        proposals = self._parse_response(raw_text, input_data.ticker)

        # Inject a thinking summary into each proposal
        thinking_summary = (
            thinking_text[:500] + "..." if len(thinking_text) > 500 else thinking_text
        )
        proposals = [
            p.model_copy(update={"thinking_summary": thinking_summary})
            for p in proposals
        ]

        # Record latency metric
        API_LATENCY.labels(api="claude_analysis").observe(
            elapsed_ms / 1000.0,
        )

        self._log.info(
            "analysis_completed",
            ticker=input_data.ticker,
            proposals_count=len(proposals),
            tokens_used=total_tokens,
            thinking_tokens=thinking_tokens,
            latency_ms=elapsed_ms,
            cost_usd=round(cost, 4),
        )

        return AnalysisResult(
            proposals=proposals,
            raw_response=raw_text,
            thinking_text=thinking_text,
            tokens_used=total_tokens,
            thinking_tokens=thinking_tokens,
            latency_ms=elapsed_ms,
            cost_estimate=round(cost, 6),
        )

    async def analyze_batch(
        self,
        inputs: list[AnalysisInput],
    ) -> list[AnalysisResult]:
        """Analyze multiple tickers sequentially.

        Processes each ticker one at a time with a small delay between
        calls to respect Anthropic rate limits.

        Args:
            inputs: List of analysis inputs, one per ticker.

        Returns:
            List of :class:`AnalysisResult` in the same order as
            *inputs*.
        """
        import asyncio

        results: list[AnalysisResult] = []
        for idx, input_data in enumerate(inputs):
            self._log.info(
                "batch_analysis_item",
                index=idx,
                total=len(inputs),
                ticker=input_data.ticker,
            )
            result = await self.analyze(input_data)
            results.append(result)

            # Delay between calls to avoid rate limiting
            if idx < len(inputs) - 1:
                await asyncio.sleep(BATCH_DELAY_SECONDS)

        self._log.info(
            "batch_analysis_completed",
            total=len(inputs),
            total_proposals=sum(len(r.proposals) for r in results),
        )
        return results

    # ------------------------------------------------------------------
    # Internal: API call with retry
    # ------------------------------------------------------------------

    @retry(
        retry=retry_if_exception(_is_transient_error),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        stop=stop_after_attempt(4),
        reraise=True,
    )
    async def _call_api(self, user_message: str) -> Message:
        """Send a message to the Anthropic API with extended thinking.

        The system prompt is sent with ``cache_control`` set to
        ``ephemeral`` so repeated calls within the cache window
        benefit from reduced input-token costs.

        Args:
            user_message: Formatted user message containing all
                market data for the analysis.

        Returns:
            The raw Anthropic :class:`Message` response object.

        Raises:
            anthropic.APIStatusError: On non-transient API errors.
        """
        response: Message = await self._client.messages.create(
            model=self._model,
            max_tokens=DEFAULT_MAX_TOKENS,
            thinking={
                "type": "enabled",
                "budget_tokens": self._thinking_budget,
            },
            system=[
                {
                    "type": "text",
                    "text": ANALYSIS_AGENT_SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                },
            ],
            messages=[
                {"role": "user", "content": user_message},
            ],
        )
        return response

    # ------------------------------------------------------------------
    # Internal: message formatting
    # ------------------------------------------------------------------

    @staticmethod
    def _format_user_message(input_data: AnalysisInput) -> str:
        """Build the user message from an AnalysisInput.

        Produces a structured text block that the Analysis Agent system
        prompt expects: clearly delimited sections for ML scores, regime,
        options chain data, account state, and existing positions.

        Args:
            input_data: The analysis input to format.

        Returns:
            Formatted user message string.
        """
        positions_text = "None"
        if input_data.current_positions:
            position_lines: list[str] = []
            for pos in input_data.current_positions:
                line = (
                    f"  - {pos.get('ticker', '?')} | "
                    f"{pos.get('strategy', '?')} | "
                    f"{pos.get('direction', '?')} | "
                    f"P&L: ${pos.get('unrealized_pnl', 0):.2f}"
                )
                position_lines.append(line)
            positions_text = "\n".join(position_lines)

        chain_json = json.dumps(
            input_data.options_chain_summary,
            indent=2,
            default=str,
        )

        return (
            f"## Trade Analysis Request\n\n"
            f"**Ticker:** {input_data.ticker}\n"
            f"**ML Ensemble Confidence:** "
            f"{input_data.ml_confidence:.4f}\n"
            f"**Market Regime:** {input_data.regime}\n"
            f"**IV Rank:** {input_data.iv_rank:.1f}\n"
            f"**Sentiment Score:** "
            f"{input_data.sentiment_score:.4f}\n"
            f"**GEX Regime:** {input_data.gex_regime}\n\n"
            f"### Options Chain Summary\n"
            f"```json\n{chain_json}\n```\n\n"
            f"### Account State\n"
            f"- Account Equity: "
            f"${input_data.account_equity:,.2f}\n"
            f"- Open Positions: "
            f"{len(input_data.current_positions)}\n\n"
            f"### Current Positions\n"
            f"{positions_text}\n\n"
            + (
                f"### Trading Memory (FinMem)\n{input_data.memory_context}\n\n"
                if input_data.memory_context
                else ""
            )
            + "Analyze this ticker and return your trade "
            "recommendation(s) as a JSON array. If no trade is "
            "warranted, return an empty array `[]`."
        )

    # ------------------------------------------------------------------
    # Internal: response parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_thinking(response: Message) -> str:
        """Extract thinking text from the response content blocks.

        Args:
            response: The Anthropic API response.

        Returns:
            Concatenated thinking text, or empty string if none.
        """
        parts: list[str] = []
        for block in response.content:
            if block.type == "thinking":
                parts.append(block.thinking)
        return "\n".join(parts)

    @staticmethod
    def _extract_text(response: Message) -> str:
        """Extract text content from the response content blocks.

        Args:
            response: The Anthropic API response.

        Returns:
            Concatenated text content, or empty string if none.
        """
        parts: list[str] = []
        for block in response.content:
            if block.type == "text":
                parts.append(block.text)
        return "\n".join(parts)

    def _parse_response(
        self,
        raw_text: str,
        fallback_ticker: str,
    ) -> list[TradeProposal]:
        """Parse trade proposals from the Claude text response.

        Handles JSON wrapped in markdown code fences (````` ```json ```)
        as well as bare JSON arrays.  Each element in the parsed array
        is validated against the :class:`TradeProposal` Pydantic model.
        Invalid elements are logged and skipped.

        Args:
            raw_text: The raw text block from the API response.
            fallback_ticker: Ticker to use if not present in a
                proposal.

        Returns:
            List of validated :class:`TradeProposal` instances.
            Returns an empty list if parsing fails entirely.
        """
        json_str = self._extract_json_from_text(raw_text)
        if not json_str:
            self._log.warning(
                "no_json_found_in_response",
                raw_text_length=len(raw_text),
            )
            return []

        try:
            parsed = json.loads(json_str)
        except json.JSONDecodeError as exc:
            self._log.error(
                "json_parse_error",
                error=str(exc),
                json_str=json_str[:500],
            )
            return []

        # Normalise to a list
        if isinstance(parsed, dict):
            parsed = [parsed]
        elif not isinstance(parsed, list):
            self._log.warning(
                "unexpected_json_type",
                json_type=type(parsed).__name__,
            )
            return []

        proposals: list[TradeProposal] = []
        for idx, item in enumerate(parsed):
            if not isinstance(item, dict):
                self._log.warning(
                    "skipping_non_dict_proposal",
                    index=idx,
                    item_type=type(item).__name__,
                )
                continue

            proposal = self._validate_proposal(item, idx, fallback_ticker)
            if proposal is not None:
                proposals.append(proposal)

        self._log.debug(
            "proposals_parsed",
            total_parsed=len(parsed),
            valid_proposals=len(proposals),
        )
        return proposals

    def _validate_proposal(
        self,
        item: dict[str, Any],
        index: int,
        fallback_ticker: str,
    ) -> TradeProposal | None:
        """Validate and construct a single TradeProposal from a dict.

        Applies defaults for missing fields and validates the strategy
        name against the known set.

        Args:
            item: Raw dict parsed from the JSON response.
            index: Position in the proposals array (for logging).
            fallback_ticker: Ticker to inject if missing.

        Returns:
            A validated :class:`TradeProposal`, or ``None`` if
            validation fails.
        """
        # Inject ticker if missing
        if "ticker" not in item:
            item["ticker"] = fallback_ticker

        # Validate strategy name
        strategy = item.get("strategy", "")
        if strategy not in VALID_STRATEGIES:
            self._log.warning(
                "invalid_strategy_in_proposal",
                index=index,
                strategy=strategy,
                valid=list(VALID_STRATEGIES),
            )
            return None

        # Validate direction
        direction = item.get("direction", "").upper()
        if direction not in ("LONG", "SHORT"):
            self._log.warning(
                "invalid_direction_in_proposal",
                index=index,
                direction=item.get("direction"),
            )
            return None
        item["direction"] = direction

        # Ensure required numeric fields have defaults
        item.setdefault("confidence", 0.5)
        item.setdefault("strikes", {})
        item.setdefault("quantity", 1)
        item.setdefault("expected_credit_debit", 0.0)
        item.setdefault("max_profit", 0.0)
        item.setdefault("max_loss", 1.0)
        item.setdefault("reward_risk_ratio", 0.0)
        item.setdefault("reasoning", "")
        item.setdefault("thinking_summary", "")
        item.setdefault("expiry", "")

        try:
            return TradeProposal.model_validate(item)
        except Exception as exc:
            self._log.error(
                "proposal_validation_error",
                index=index,
                error=str(exc),
                item_keys=list(item.keys()),
            )
            return None

    @staticmethod
    def _extract_json_from_text(text: str) -> str:
        """Extract a JSON string from text that may contain code fences.

        Handles three patterns:
        1. ````` ```json ... ``` `````
        2. ````` ``` ... ``` `````
        3. Bare JSON starting with ``[`` or ``{``

        Args:
            text: Raw text that may contain JSON.

        Returns:
            Extracted JSON string, or empty string if not found.
        """
        stripped = text.strip()

        # Pattern 1 & 2: fenced code blocks
        if "```" in stripped:
            # Find the opening fence
            start_idx = stripped.find("```")
            # Skip past the opening fence line
            newline_after_fence = stripped.find("\n", start_idx)
            if newline_after_fence == -1:
                return ""
            content_start = newline_after_fence + 1

            # Find the closing fence
            close_idx = stripped.find("```", content_start)
            if close_idx == -1:
                # No closing fence -- take everything after opening
                return stripped[content_start:].strip()
            return stripped[content_start:close_idx].strip()

        # Pattern 3: bare JSON
        for char_idx, char in enumerate(stripped):
            if char in ("[", "{"):
                return stripped[char_idx:]

        return ""

    # ------------------------------------------------------------------
    # Internal: cost estimation
    # ------------------------------------------------------------------

    @staticmethod
    def _estimate_cost(
        input_tokens: int,
        output_tokens: int,
        thinking_tokens: int,
        cached_input_tokens: int = 0,
    ) -> float:
        """Estimate the USD cost of an API call.

        Uses Claude Sonnet pricing:
        - Input: $3.00 / million tokens
        - Cached input: $0.30 / million tokens
        - Output: $15.00 / million tokens
        - Thinking: $15.00 / million tokens

        Args:
            input_tokens: Total input tokens (including cached).
            output_tokens: Output tokens generated.
            thinking_tokens: Tokens used in the thinking block.
            cached_input_tokens: Subset of input_tokens served from
                prompt cache.

        Returns:
            Estimated cost in USD.
        """
        # Non-cached input tokens
        fresh_input = max(0, input_tokens - cached_input_tokens)

        cost = (
            (fresh_input / 1_000_000.0) * PRICE_INPUT_PER_M
            + (cached_input_tokens / 1_000_000.0) * PRICE_CACHED_INPUT_PER_M
            + (output_tokens / 1_000_000.0) * PRICE_OUTPUT_PER_M
            + (thinking_tokens / 1_000_000.0) * PRICE_THINKING_PER_M
        )
        return cost
