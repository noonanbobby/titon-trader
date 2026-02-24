"""Risk Agent for Project Titan.

Evaluates trade proposals from the Analysis Agent and renders a binding
risk decision: **APPROVED**, **REJECTED**, or **MODIFIED**.  The agent
combines Claude's extended-thinking reasoning with hard-coded risk
limits that can **never** be overridden -- not by the AI, not by ML
confidence, and not by manual intervention during live trading.

The evaluation pipeline works in two stages:

1. **AI evaluation** -- Claude reviews the proposal against portfolio
   context, upcoming events, correlation data, and circuit-breaker
   state.  It may approve, reject, or suggest modifications.
2. **Hard-limit enforcement** -- a deterministic layer checks sacred
   risk limits (per-trade risk, position count, circuit-breaker level,
   portfolio delta).  If *any* hard limit is violated the decision is
   forced to ``REJECTED`` regardless of what the AI recommended.

Usage::

    from src.ai.risk_agent import RiskAgent, RiskContext

    agent = RiskAgent(api_key="sk-ant-...")
    evaluation = await agent.evaluate(
        proposal=proposal_dict,
        context=RiskContext(
            proposal=proposal_dict,
            current_positions=[...],
            ...
        ),
    )
    if evaluation.decision == "APPROVED":
        # proceed to execution
        ...
"""

from __future__ import annotations

import json
import time
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from anthropic import AsyncAnthropic
from pydantic import BaseModel, Field
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from src.ai.prompts import RISK_AGENT_SYSTEM_PROMPT
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
DEFAULT_THINKING_BUDGET: int = 8192
DEFAULT_MAX_TOKENS: int = 4096
BATCH_DELAY_SECONDS: float = 1.0

# Hard risk limits -- these are sacred and cannot be overridden
HARD_MAX_RISK_PCT: float = 0.02
HARD_MAX_RISK_DOLLARS: float = 3000.0
HARD_MAX_CONCURRENT_POSITIONS: int = 8
HARD_MAX_PORTFOLIO_DELTA: float = 500.0

# Circuit-breaker levels that block new trades
BLOCKED_CB_LEVELS: frozenset[str] = frozenset({"HALT", "EMERGENCY"})


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


class RiskDecision(StrEnum):
    """Possible outcomes of a risk evaluation."""

    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    MODIFIED = "MODIFIED"


class RiskModification(BaseModel):
    """A single modification applied to a trade proposal.

    Attributes:
        field: Name of the field that was modified.
        original_value: The value before modification.
        modified_value: The value after modification.
        reason: Explanation for the modification.
    """

    field: str = Field(
        ...,
        description="Name of the modified field",
    )
    original_value: Any = Field(
        ...,
        description="Value before modification",
    )
    modified_value: Any = Field(
        ...,
        description="Value after modification",
    )
    reason: str = Field(
        ...,
        description="Explanation for the modification",
    )


class RiskEvaluation(BaseModel):
    """Complete result of a risk evaluation.

    Attributes:
        decision: Final risk decision (APPROVED, REJECTED, MODIFIED).
        reason: Human-readable explanation of the decision.
        modifications: List of modifications (empty if APPROVED or
            REJECTED).
        risk_score: Composite risk score (0.0 = no risk, 1.0 = max).
        checks_passed: Names of risk checks that passed.
        checks_failed: Names of risk checks that failed.
        thinking_summary: Summary of the extended thinking process.
        tokens_used: Total tokens consumed (input + output).
        latency_ms: Round-trip API latency in milliseconds.
    """

    decision: RiskDecision = Field(
        ...,
        description="Final risk decision",
    )
    reason: str = Field(
        default="",
        description="Human-readable explanation",
    )
    modifications: list[RiskModification] = Field(
        default_factory=list,
        description="Modifications applied to the proposal",
    )
    risk_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Composite risk score (0=safe, 1=maximum)",
    )
    checks_passed: list[str] = Field(
        default_factory=list,
        description="Names of risk checks that passed",
    )
    checks_failed: list[str] = Field(
        default_factory=list,
        description="Names of risk checks that failed",
    )
    thinking_summary: str = Field(
        default="",
        description="Summary of extended thinking process",
    )
    tokens_used: int = Field(
        default=0,
        ge=0,
        description="Total tokens consumed",
    )
    latency_ms: int = Field(
        default=0,
        ge=0,
        description="Round-trip API latency in milliseconds",
    )


class RiskContext(BaseModel):
    """Full context provided to the Risk Agent for evaluation.

    Includes the trade proposal, current portfolio state, circuit
    breaker status, P&L history, upcoming events, and correlation data.

    Attributes:
        proposal: The trade proposal as a dictionary.
        current_positions: List of dicts describing open positions.
        portfolio_greeks: Aggregate portfolio Greeks.
        circuit_breaker_level: Current circuit breaker level.
        recovery_stage: Current recovery stage (0 = normal).
        daily_pnl: Today's realized + unrealized P&L.
        weekly_pnl: This week's cumulative P&L.
        monthly_pnl: This month's cumulative P&L.
        upcoming_events: Upcoming events within exclusion windows.
        account_equity: Current account net liquidation value.
        buying_power: Available buying power.
        correlation_data: Pairwise correlations with existing
            positions.
    """

    proposal: dict[str, Any] = Field(
        ...,
        description="The trade proposal to evaluate",
    )
    current_positions: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Currently open positions",
    )
    portfolio_greeks: dict[str, float] = Field(
        default_factory=dict,
        description="Aggregate portfolio Greeks",
    )
    circuit_breaker_level: str = Field(
        default="NORMAL",
        description="Current circuit breaker level",
    )
    recovery_stage: int = Field(
        default=0,
        ge=0,
        description="Current recovery stage (0=normal)",
    )
    daily_pnl: float = Field(
        default=0.0,
        description="Today's P&L in dollars",
    )
    weekly_pnl: float = Field(
        default=0.0,
        description="This week's cumulative P&L in dollars",
    )
    monthly_pnl: float = Field(
        default=0.0,
        description="This month's cumulative P&L in dollars",
    )
    upcoming_events: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Events within exclusion windows",
    )
    account_equity: float = Field(
        default=150_000.0,
        gt=0,
        description="Current net liquidation value",
    )
    buying_power: float = Field(
        default=0.0,
        description="Available buying power",
    )
    correlation_data: dict[str, float] = Field(
        default_factory=dict,
        description="Pairwise correlations with existing positions",
    )


# ---------------------------------------------------------------------------
# RiskAgent
# ---------------------------------------------------------------------------


class RiskAgent:
    """Claude-powered risk evaluation agent with veto authority.

    Combines AI-driven reasoning with deterministic hard-limit
    enforcement.  The hard limits are **sacred** and cannot be
    overridden by the AI under any circumstances.

    Args:
        api_key: Anthropic API key.
        model: Claude model identifier.  Defaults to
            ``claude-sonnet-4-5-20250929``.
        thinking_budget: Maximum tokens allocated to the extended
            thinking block.  Defaults to ``4096``.
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
            "ai.risk_agent",
        )
        self._log.info(
            "risk_agent_initialized",
            model=self._model,
            thinking_budget=self._thinking_budget,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def evaluate(
        self,
        proposal: dict[str, Any],
        context: RiskContext,
    ) -> RiskEvaluation:
        """Evaluate a trade proposal against risk limits.

        Runs the two-stage evaluation pipeline:

        1. AI evaluation via Claude with extended thinking.
        2. Hard-limit enforcement that can override the AI decision.

        Args:
            proposal: The trade proposal as a dictionary (from
                :class:`TradeProposal`).
            context: Full risk context including portfolio state.

        Returns:
            A :class:`RiskEvaluation` with the final decision.
        """
        self._log.info(
            "risk_evaluation_started",
            ticker=proposal.get("ticker", "?"),
            strategy=proposal.get("strategy", "?"),
            circuit_breaker=context.circuit_breaker_level,
        )

        # Stage 1: AI evaluation
        user_message = self._format_context_message(proposal, context)
        start_time = time.monotonic()

        ai_evaluation = await self._get_ai_evaluation(user_message)

        elapsed_ms = int((time.monotonic() - start_time) * 1000)
        ai_evaluation = ai_evaluation.model_copy(update={"latency_ms": elapsed_ms})

        # Record latency metric
        API_LATENCY.labels(api="claude_risk").observe(
            elapsed_ms / 1000.0,
        )

        # Stage 2: hard-limit enforcement
        final_evaluation = self._apply_hard_limits(proposal, context, ai_evaluation)

        self._log.info(
            "risk_evaluation_completed",
            ticker=proposal.get("ticker", "?"),
            decision=final_evaluation.decision.value,
            risk_score=final_evaluation.risk_score,
            checks_passed=len(final_evaluation.checks_passed),
            checks_failed=len(final_evaluation.checks_failed),
            latency_ms=final_evaluation.latency_ms,
        )

        return final_evaluation

    async def evaluate_batch(
        self,
        proposals: list[dict[str, Any]],
        context: RiskContext,
    ) -> list[RiskEvaluation]:
        """Evaluate multiple proposals sequentially.

        Each proposal is evaluated against the same context.  A small
        delay is inserted between calls to respect rate limits.

        Args:
            proposals: List of proposal dictionaries.
            context: Shared risk context.

        Returns:
            List of :class:`RiskEvaluation` in the same order as
            *proposals*.
        """
        import asyncio

        results: list[RiskEvaluation] = []
        for idx, proposal in enumerate(proposals):
            self._log.info(
                "batch_risk_item",
                index=idx,
                total=len(proposals),
                ticker=proposal.get("ticker", "?"),
            )
            result = await self.evaluate(proposal, context)
            results.append(result)

            if idx < len(proposals) - 1:
                await asyncio.sleep(BATCH_DELAY_SECONDS)

        self._log.info(
            "batch_risk_completed",
            total=len(proposals),
            approved=sum(1 for r in results if r.decision == RiskDecision.APPROVED),
            rejected=sum(1 for r in results if r.decision == RiskDecision.REJECTED),
            modified=sum(1 for r in results if r.decision == RiskDecision.MODIFIED),
        )
        return results

    # ------------------------------------------------------------------
    # Internal: AI evaluation
    # ------------------------------------------------------------------

    async def _get_ai_evaluation(
        self,
        user_message: str,
    ) -> RiskEvaluation:
        """Obtain the AI-driven risk evaluation from Claude.

        On API failure, returns a conservative REJECTED evaluation so
        the system never proceeds with a trade when the risk agent is
        unavailable.

        Args:
            user_message: Formatted context message for the API.

        Returns:
            A :class:`RiskEvaluation` parsed from Claude's response,
            or a fallback REJECTED evaluation on error.
        """
        try:
            response = await self._call_api(user_message)
        except Exception as exc:
            self._log.error(
                "risk_api_error",
                error=str(exc),
            )
            return RiskEvaluation(
                decision=RiskDecision.REJECTED,
                reason=(
                    f"Risk Agent API unavailable: {exc}. "
                    "Defaulting to REJECTED for safety."
                ),
                risk_score=1.0,
                checks_failed=["api_availability"],
            )

        thinking_text = self._extract_thinking(response)
        raw_text = self._extract_text(response)

        # Token accounting
        usage = response.usage
        total_tokens = usage.input_tokens + usage.output_tokens

        # Parse AI evaluation from response
        evaluation = self._parse_ai_response(raw_text, thinking_text, total_tokens)
        return evaluation

    @retry(
        retry=retry_if_exception(_is_transient_error),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        stop=stop_after_attempt(4),
        reraise=True,
    )
    async def _call_api(self, user_message: str) -> Message:
        """Send a message to the Anthropic API with extended thinking.

        The system prompt uses ``cache_control`` for cost efficiency.

        Args:
            user_message: Formatted context message.

        Returns:
            The raw Anthropic :class:`Message` response.

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
                    "text": RISK_AGENT_SYSTEM_PROMPT,
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
    def _format_context_message(
        proposal: dict[str, Any],
        context: RiskContext,
    ) -> str:
        """Build the user message from the proposal and risk context.

        Produces a structured text block with clearly delimited sections
        for the proposal, portfolio state, circuit breaker status, P&L,
        upcoming events, and correlation data.

        Args:
            proposal: The trade proposal dictionary.
            context: Full risk context.

        Returns:
            Formatted user message string.
        """
        proposal_json = json.dumps(proposal, indent=2, default=str)

        positions_text = "None"
        if context.current_positions:
            pos_lines: list[str] = []
            for pos in context.current_positions:
                line = (
                    f"  - {pos.get('ticker', '?')} | "
                    f"{pos.get('strategy', '?')} | "
                    f"Delta: {pos.get('delta', 0):.2f} | "
                    f"P&L: ${pos.get('unrealized_pnl', 0):.2f}"
                )
                pos_lines.append(line)
            positions_text = "\n".join(pos_lines)

        greeks_text = ", ".join(
            f"{k}: {v:.2f}" for k, v in context.portfolio_greeks.items()
        )

        events_text = "None"
        if context.upcoming_events:
            event_lines: list[str] = []
            for evt in context.upcoming_events:
                line = (
                    f"  - {evt.get('type', '?')}: "
                    f"{evt.get('ticker', '?')} on "
                    f"{evt.get('date', '?')}"
                )
                event_lines.append(line)
            events_text = "\n".join(event_lines)

        corr_text = "None"
        if context.correlation_data:
            corr_lines: list[str] = []
            for pair, value in context.correlation_data.items():
                corr_lines.append(f"  - {pair}: {value:.4f}")
            corr_text = "\n".join(corr_lines)

        return (
            f"## Risk Evaluation Request\n\n"
            f"### Trade Proposal\n"
            f"```json\n{proposal_json}\n```\n\n"
            f"### Portfolio State\n"
            f"- Account Equity: "
            f"${context.account_equity:,.2f}\n"
            f"- Buying Power: "
            f"${context.buying_power:,.2f}\n"
            f"- Open Positions: "
            f"{len(context.current_positions)}\n"
            f"- Portfolio Greeks: {greeks_text}\n\n"
            f"### Current Positions\n"
            f"{positions_text}\n\n"
            f"### Circuit Breaker\n"
            f"- Level: {context.circuit_breaker_level}\n"
            f"- Recovery Stage: {context.recovery_stage}\n\n"
            f"### P&L\n"
            f"- Daily: ${context.daily_pnl:,.2f}\n"
            f"- Weekly: ${context.weekly_pnl:,.2f}\n"
            f"- Monthly: ${context.monthly_pnl:,.2f}\n\n"
            f"### Upcoming Events\n"
            f"{events_text}\n\n"
            f"### Correlation Data\n"
            f"{corr_text}\n\n"
            f"Evaluate this proposal and return your risk "
            f"decision as a JSON object with keys: decision, "
            f"reason, modifications, risk_score, checks_passed, "
            f"checks_failed."
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
            Concatenated thinking text, or empty string.
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
            Concatenated text content, or empty string.
        """
        parts: list[str] = []
        for block in response.content:
            if block.type == "text":
                parts.append(block.text)
        return "\n".join(parts)

    def _parse_ai_response(
        self,
        raw_text: str,
        thinking_text: str,
        tokens_used: int,
    ) -> RiskEvaluation:
        """Parse a RiskEvaluation from Claude's text response.

        Extracts JSON from the response text (handling code fences),
        maps the parsed ``decision`` field to a :class:`RiskDecision`
        enum, and builds a :class:`RiskEvaluation` instance.

        On parse failure, returns a conservative REJECTED evaluation.

        Args:
            raw_text: The text block from the API response.
            thinking_text: The thinking block text.
            tokens_used: Total tokens consumed by the API call.

        Returns:
            A :class:`RiskEvaluation` instance.
        """
        json_str = self._extract_json_from_text(raw_text)
        if not json_str:
            self._log.warning(
                "no_json_in_risk_response",
                raw_text_length=len(raw_text),
            )
            return RiskEvaluation(
                decision=RiskDecision.REJECTED,
                reason=(
                    "Failed to parse risk evaluation response. "
                    "Defaulting to REJECTED for safety."
                ),
                risk_score=1.0,
                checks_failed=["response_parsing"],
                tokens_used=tokens_used,
            )

        try:
            parsed = json.loads(json_str)
        except json.JSONDecodeError as exc:
            self._log.error(
                "risk_json_parse_error",
                error=str(exc),
                json_str=json_str[:500],
            )
            return RiskEvaluation(
                decision=RiskDecision.REJECTED,
                reason=(f"JSON parse error: {exc}. Defaulting to REJECTED for safety."),
                risk_score=1.0,
                checks_failed=["json_parsing"],
                tokens_used=tokens_used,
            )

        if not isinstance(parsed, dict):
            self._log.warning(
                "unexpected_risk_json_type",
                json_type=type(parsed).__name__,
            )
            return RiskEvaluation(
                decision=RiskDecision.REJECTED,
                reason="Unexpected JSON structure in response.",
                risk_score=1.0,
                checks_failed=["json_structure"],
                tokens_used=tokens_used,
            )

        # Map decision string to enum
        decision_str = str(parsed.get("decision", "REJECTED")).upper()
        try:
            decision = RiskDecision(decision_str)
        except ValueError:
            self._log.warning(
                "unknown_risk_decision",
                decision=decision_str,
            )
            decision = RiskDecision.REJECTED

        # Parse modifications
        modifications: list[RiskModification] = []
        raw_mods = parsed.get("modifications", [])
        if isinstance(raw_mods, list):
            for mod_dict in raw_mods:
                if isinstance(mod_dict, dict):
                    try:
                        modifications.append(RiskModification.model_validate(mod_dict))
                    except Exception as exc:
                        self._log.warning(
                            "invalid_modification",
                            error=str(exc),
                        )

        # Build thinking summary
        thinking_summary = (
            thinking_text[:500] + "..." if len(thinking_text) > 500 else thinking_text
        )

        risk_score = float(parsed.get("risk_score", 0.5))
        risk_score = max(0.0, min(1.0, risk_score))

        return RiskEvaluation(
            decision=decision,
            reason=str(parsed.get("reason", "")),
            modifications=modifications,
            risk_score=risk_score,
            checks_passed=list(parsed.get("checks_passed", [])),
            checks_failed=list(parsed.get("checks_failed", [])),
            thinking_summary=thinking_summary,
            tokens_used=tokens_used,
        )

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
            start_idx = stripped.find("```")
            newline_after_fence = stripped.find("\n", start_idx)
            if newline_after_fence == -1:
                return ""
            content_start = newline_after_fence + 1

            close_idx = stripped.find("```", content_start)
            if close_idx == -1:
                return stripped[content_start:].strip()
            return stripped[content_start:close_idx].strip()

        # Pattern 3: bare JSON
        for char_idx, char in enumerate(stripped):
            if char in ("[", "{"):
                return stripped[char_idx:]

        return ""

    # ------------------------------------------------------------------
    # Internal: hard-limit enforcement
    # ------------------------------------------------------------------

    def _apply_hard_limits(
        self,
        proposal: dict[str, Any],
        context: RiskContext,
        ai_evaluation: RiskEvaluation,
    ) -> RiskEvaluation:
        """Enforce sacred risk limits that can never be overridden.

        These checks are deterministic and run *after* the AI
        evaluation.  If any hard limit is violated, the decision is
        forced to REJECTED regardless of what the AI recommended.

        Hard limits checked:
        1. Circuit breaker level is not HALT or EMERGENCY.
        2. Per-trade risk does not exceed max_risk_pct (2%) or
           max_risk_dollars ($3,000).
        3. Total position count is below max_concurrent_positions (8).
        4. Portfolio delta stays within limits after the new trade.

        Args:
            proposal: The trade proposal dictionary.
            context: Full risk context.
            ai_evaluation: The evaluation from Claude.

        Returns:
            The final :class:`RiskEvaluation`, potentially overridden
            to REJECTED.
        """
        hard_checks_passed: list[str] = []
        hard_checks_failed: list[str] = []
        rejection_reasons: list[str] = []

        # -- Check 1: Circuit breaker level --
        cb_level = context.circuit_breaker_level.upper()
        if cb_level in BLOCKED_CB_LEVELS:
            hard_checks_failed.append("circuit_breaker_level")
            rejection_reasons.append(
                f"Circuit breaker at {cb_level} -- no new trades permitted"
            )
        else:
            hard_checks_passed.append("circuit_breaker_level")

        # -- Check 2: Per-trade risk --
        max_loss = float(proposal.get("max_loss", 0.0))
        quantity = int(proposal.get("quantity", 1))
        total_risk = max_loss * quantity

        # Percentage-based limit
        risk_pct = (
            total_risk / context.account_equity if context.account_equity > 0 else 1.0
        )
        if risk_pct > HARD_MAX_RISK_PCT:
            hard_checks_failed.append("per_trade_risk_pct")
            rejection_reasons.append(
                f"Per-trade risk {risk_pct:.2%} exceeds limit "
                f"of {HARD_MAX_RISK_PCT:.2%}"
            )
        else:
            hard_checks_passed.append("per_trade_risk_pct")

        # Dollar-based limit
        if total_risk > HARD_MAX_RISK_DOLLARS:
            hard_checks_failed.append("per_trade_risk_dollars")
            rejection_reasons.append(
                f"Per-trade risk ${total_risk:,.2f} exceeds "
                f"limit of ${HARD_MAX_RISK_DOLLARS:,.2f}"
            )
        else:
            hard_checks_passed.append("per_trade_risk_dollars")

        # -- Check 3: Position count --
        current_count = len(context.current_positions)
        if current_count >= HARD_MAX_CONCURRENT_POSITIONS:
            hard_checks_failed.append("max_concurrent_positions")
            rejection_reasons.append(
                f"Position count {current_count} already at "
                f"limit of {HARD_MAX_CONCURRENT_POSITIONS}"
            )
        else:
            hard_checks_passed.append("max_concurrent_positions")

        # -- Check 4: Portfolio delta --
        # Use algebraic sum (not absolute) so hedging trades that reduce
        # net delta are not incorrectly rejected.
        portfolio_delta = float(context.portfolio_greeks.get("delta", 0.0))
        proposal_delta = float(proposal.get("expected_delta", 0.0))
        projected_delta = abs(portfolio_delta + proposal_delta)
        if projected_delta > HARD_MAX_PORTFOLIO_DELTA:
            hard_checks_failed.append("portfolio_delta")
            rejection_reasons.append(
                f"Projected portfolio delta "
                f"{projected_delta:.1f} exceeds limit of "
                f"{HARD_MAX_PORTFOLIO_DELTA:.1f}"
            )
        else:
            hard_checks_passed.append("portfolio_delta")

        # -- Merge with AI evaluation --
        all_passed = ai_evaluation.checks_passed + hard_checks_passed
        all_failed = ai_evaluation.checks_failed + hard_checks_failed

        # If any hard limit failed, override to REJECTED
        if hard_checks_failed:
            combined_reason = "HARD LIMIT VIOLATION: " + "; ".join(rejection_reasons)
            if ai_evaluation.reason:
                combined_reason += f" | AI assessment: {ai_evaluation.reason}"

            self._log.warning(
                "hard_limit_override",
                ticker=proposal.get("ticker", "?"),
                ai_decision=ai_evaluation.decision.value,
                hard_checks_failed=hard_checks_failed,
                rejection_reasons=rejection_reasons,
            )

            return RiskEvaluation(
                decision=RiskDecision.REJECTED,
                reason=combined_reason,
                modifications=[],
                risk_score=max(ai_evaluation.risk_score, 0.9),
                checks_passed=all_passed,
                checks_failed=all_failed,
                thinking_summary=ai_evaluation.thinking_summary,
                tokens_used=ai_evaluation.tokens_used,
                latency_ms=ai_evaluation.latency_ms,
            )

        # All hard limits passed -- keep AI decision
        return ai_evaluation.model_copy(
            update={
                "checks_passed": all_passed,
                "checks_failed": all_failed,
            }
        )

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
        fresh_input = max(0, input_tokens - cached_input_tokens)

        cost = (
            (fresh_input / 1_000_000.0) * PRICE_INPUT_PER_M
            + (cached_input_tokens / 1_000_000.0) * PRICE_CACHED_INPUT_PER_M
            + (output_tokens / 1_000_000.0) * PRICE_OUTPUT_PER_M
            + (thinking_tokens / 1_000_000.0) * PRICE_THINKING_PER_M
        )
        return cost
