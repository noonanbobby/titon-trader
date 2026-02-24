"""Journal Agent for Project Titan.

Reviews closed trades at end-of-day, assigns quality grades, extracts
lessons learned, and manages the FinMem layered memory system.  Uses
the Anthropic Batch API for cost savings (50% discount) when reviewing
multiple trades simultaneously.

Usage::

    from src.ai.journal_agent import JournalAgent

    agent = JournalAgent(api_key="sk-ant-...", model="claude-sonnet-4-5-20250929")
    entry = await agent.review_trades(closed_trades)
    print(entry.summary.total_pnl)
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from anthropic import AsyncAnthropic
from pydantic import BaseModel, Field
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.ai.prompts import JOURNAL_AGENT_SYSTEM_PROMPT
from src.utils.logging import get_logger

if TYPE_CHECKING:
    import structlog

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_API_TOKENS: int = 4096
BATCH_POLL_INTERVAL_SECONDS: float = 10.0
BATCH_POLL_MAX_WAIT_SECONDS: float = 3600.0

# Grade thresholds for the rubric
GRADE_A_MIN_PNL_PCT: float = 0.0
GRADE_B_MIN_PNL_PCT: float = -0.10
GRADE_C_MIN_PNL_PCT: float = -0.25
GRADE_D_MIN_PNL_PCT: float = -0.50

# Entry/exit quality thresholds (0.0 to 1.0 scale)
QUALITY_GOOD_THRESHOLD: float = 0.65
QUALITY_REASONABLE_THRESHOLD: float = 0.40

# FinMem layer sizes
FINMEM_SHORT_WINDOW: int = 5
FINMEM_MEDIUM_WINDOW: int = 30

# Cost estimation: approximate per-token pricing (batch = 50% off)
COST_PER_INPUT_TOKEN: float = 0.0000015
COST_PER_OUTPUT_TOKEN: float = 0.0000075
BATCH_DISCOUNT: float = 0.50


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class TradeReview(BaseModel):
    """Graded review of a single closed trade.

    Attributes:
        trade_id: UUID of the trade from the ``trades`` table.
        ticker: Underlying symbol.
        strategy: Strategy that was used.
        grade: Letter grade (A through F).
        pnl: Realised profit or loss in dollars.
        pnl_pct: P&L as a percentage of max risk.
        entry_quality: Description of entry timing and conditions.
        exit_quality: Description of exit timing and conditions.
        lessons: Key takeaways from this trade.
        pattern_observations: Observed market patterns.
        would_take_again: Whether the trade setup was sound.
        improvement_suggestions: Actionable improvements.
    """

    trade_id: str = Field(..., description="UUID of the reviewed trade")
    ticker: str = Field(..., description="Underlying symbol")
    strategy: str = Field(..., description="Strategy used")
    grade: str = Field(
        ...,
        description="Letter grade: A, B, C, D, or F",
    )
    pnl: float = Field(..., description="Realised P&L in dollars")
    pnl_pct: float = Field(..., description="P&L as percentage of max risk")
    entry_quality: str = Field(
        ...,
        description="Description of entry timing and conditions",
    )
    exit_quality: str = Field(
        ...,
        description="Description of exit timing and conditions",
    )
    lessons: list[str] = Field(
        default_factory=list,
        description="Key takeaways from this trade",
    )
    pattern_observations: list[str] = Field(
        default_factory=list,
        description="Observed market patterns during the trade",
    )
    would_take_again: bool = Field(
        ...,
        description="Whether the trade setup was sound",
    )
    improvement_suggestions: list[str] = Field(
        default_factory=list,
        description="Actionable improvement suggestions",
    )


class DailySummary(BaseModel):
    """Aggregated summary of all trades reviewed in a session.

    Attributes:
        date: ISO date string for the trading day.
        total_trades: Number of trades reviewed.
        winners: Number of profitable trades.
        losers: Number of losing trades.
        total_pnl: Aggregate P&L in dollars.
        best_trade: Highest-graded trade review.
        worst_trade: Lowest-graded trade review.
        regime: Market regime during the trading day.
        key_observations: Cross-trade patterns and insights.
        strategy_performance: Performance breakdown by strategy.
        memory_updates: FinMem updates to persist.
    """

    date: str = Field(..., description="ISO date string (YYYY-MM-DD)")
    total_trades: int = Field(..., ge=0, description="Number of trades reviewed")
    winners: int = Field(..., ge=0, description="Number of profitable trades")
    losers: int = Field(..., ge=0, description="Number of losing trades")
    total_pnl: float = Field(..., description="Aggregate P&L in dollars")
    best_trade: TradeReview | None = Field(
        default=None,
        description="Highest-graded trade review",
    )
    worst_trade: TradeReview | None = Field(
        default=None,
        description="Lowest-graded trade review",
    )
    regime: str = Field(
        default="unknown",
        description="Market regime during the trading day",
    )
    key_observations: list[str] = Field(
        default_factory=list,
        description="Cross-trade patterns and insights",
    )
    strategy_performance: dict[str, dict[str, Any]] = Field(
        default_factory=dict,
        description=("Performance by strategy: {strategy: {trades, pnl, win_rate}}"),
    )
    memory_updates: list[dict[str, Any]] = Field(
        default_factory=list,
        description="FinMem memory layer updates to persist",
    )


class JournalEntry(BaseModel):
    """Complete journal output from a review session.

    Attributes:
        reviews: Individual trade reviews.
        summary: Aggregated daily summary.
        tokens_used: Total API tokens consumed.
        cost_estimate: Estimated API cost in USD.
    """

    reviews: list[TradeReview] = Field(..., description="Individual trade reviews")
    summary: DailySummary = Field(..., description="Aggregated daily summary")
    tokens_used: int = Field(
        default=0,
        description="Total API tokens consumed",
    )
    cost_estimate: float = Field(
        default=0.0,
        description="Estimated API cost in USD",
    )


# ---------------------------------------------------------------------------
# Grade ordering for comparisons
# ---------------------------------------------------------------------------

_GRADE_ORDER: dict[str, int] = {
    "A": 5,
    "B": 4,
    "C": 3,
    "D": 2,
    "F": 1,
}


# ---------------------------------------------------------------------------
# JournalAgent
# ---------------------------------------------------------------------------


class JournalAgent:
    """Reviews closed trades and manages learning via FinMem.

    Uses the Anthropic Batch API for cost-efficient bulk reviews and
    maintains a layered memory system with short (5 trades), medium
    (30 trades), and long-term (regime patterns) horizons.

    Args:
        api_key: Anthropic API key.
        model: Claude model identifier for API calls.
    """

    def __init__(self, api_key: str, model: str) -> None:
        self._client: AsyncAnthropic = AsyncAnthropic(api_key=api_key)
        self._model: str = model
        self._log: structlog.stdlib.BoundLogger = get_logger("ai.journal_agent")
        self._total_tokens: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def review_trades(self, closed_trades: list[dict[str, Any]]) -> JournalEntry:
        """Review a batch of closed trades and produce a journal entry.

        Uses the Anthropic Batch API when multiple trades need review
        for 50% cost savings.  Falls back to sequential single-trade
        reviews if batch creation fails.

        Args:
            closed_trades: List of trade dicts, each containing at
                minimum ``trade_id``, ``ticker``, ``strategy``,
                ``direction``, ``entry_price``, ``exit_price``,
                ``realized_pnl``, ``max_loss``, ``entry_reasoning``,
                ``exit_reasoning``, ``regime``, and ``entry_time``.

        Returns:
            A :class:`JournalEntry` with individual reviews and an
            aggregated summary.
        """
        if not closed_trades:
            self._log.info("no_trades_to_review")
            today = datetime.now(tz=UTC).strftime("%Y-%m-%d")
            return JournalEntry(
                reviews=[],
                summary=DailySummary(
                    date=today,
                    total_trades=0,
                    winners=0,
                    losers=0,
                    total_pnl=0.0,
                ),
                tokens_used=0,
                cost_estimate=0.0,
            )

        self._log.info(
            "reviewing_trades",
            count=len(closed_trades),
        )
        self._total_tokens = 0

        # Attempt batch review for cost savings
        reviews = await self._review_trades_batch(closed_trades)

        summary = await self._generate_summary(reviews, closed_trades)

        cost = self._estimate_cost(self._total_tokens)

        entry = JournalEntry(
            reviews=reviews,
            summary=summary,
            tokens_used=self._total_tokens,
            cost_estimate=round(cost, 4),
        )

        self._log.info(
            "journal_entry_complete",
            total_trades=len(reviews),
            total_pnl=summary.total_pnl,
            tokens_used=self._total_tokens,
            cost_estimate=cost,
        )

        return entry

    async def create_batch_request(self, trades: list[dict[str, Any]]) -> str:
        """Create an Anthropic batch API request for trade reviews.

        Builds individual review requests with unique custom IDs and
        submits them as a batch.  Returns the batch ID for status
        polling.

        Args:
            trades: List of trade dicts to review.

        Returns:
            Batch ID string for polling.

        Raises:
            anthropic.APIError: On batch creation failure.
        """
        requests = []
        for trade in trades:
            custom_id = f"review-{trade.get('trade_id', uuid.uuid4().hex)}"
            formatted = self._format_trade_for_review(trade)

            requests.append(
                {
                    "custom_id": custom_id,
                    "params": {
                        "model": self._model,
                        "max_tokens": MAX_API_TOKENS,
                        "system": [
                            {
                                "type": "text",
                                "text": JOURNAL_AGENT_SYSTEM_PROMPT,
                                "cache_control": {"type": "ephemeral"},
                            },
                        ],
                        "messages": [
                            {
                                "role": "user",
                                "content": formatted,
                            },
                        ],
                    },
                }
            )

        self._log.info(
            "creating_batch_request",
            num_requests=len(requests),
        )

        batch = await self._client.messages.batches.create(requests=requests)

        self._log.info(
            "batch_created",
            batch_id=batch.id,
            num_requests=len(requests),
        )

        return batch.id

    async def poll_batch_status(self, batch_id: str) -> list[dict[str, Any]]:
        """Poll a batch until completion and return results.

        Polls the Anthropic batch endpoint at regular intervals until
        the batch reaches a terminal state (ended).  Collects all
        results and returns them.

        Args:
            batch_id: Batch identifier from :meth:`create_batch_request`.

        Returns:
            List of result dicts, each containing ``custom_id`` and
            the response content.

        Raises:
            TimeoutError: If the batch does not complete within the
                maximum wait time.
        """
        self._log.info("polling_batch", batch_id=batch_id)

        start_time = time.monotonic()

        while (time.monotonic() - start_time) < BATCH_POLL_MAX_WAIT_SECONDS:
            batch = await self._client.messages.batches.retrieve(batch_id)

            self._log.debug(
                "batch_poll_status",
                batch_id=batch_id,
                status=batch.processing_status,
            )

            if batch.processing_status == "ended":
                break

            await asyncio.sleep(BATCH_POLL_INTERVAL_SECONDS)
        else:
            raise TimeoutError(
                f"Batch {batch_id} did not complete within "
                f"{BATCH_POLL_MAX_WAIT_SECONDS}s"
            )

        results: list[dict[str, Any]] = []
        async for result in self._client.messages.batches.results(batch_id):
            result_dict: dict[str, Any] = {
                "custom_id": result.custom_id,
            }
            if result.result.type == "succeeded":
                message = result.result.message
                text_parts: list[str] = []
                for block in message.content:
                    if hasattr(block, "text"):
                        text_parts.append(block.text)
                result_dict["content"] = "\n".join(text_parts)
                result_dict["success"] = True
                input_tokens = message.usage.input_tokens
                output_tokens = message.usage.output_tokens
                self._total_tokens += input_tokens + output_tokens
            else:
                result_dict["content"] = ""
                result_dict["success"] = False
                result_dict["error"] = str(result.result.type)

            results.append(result_dict)

        self._log.info(
            "batch_results_collected",
            batch_id=batch_id,
            num_results=len(results),
        )

        return results

    # ------------------------------------------------------------------
    # Internal: review pipelines
    # ------------------------------------------------------------------

    async def _review_trades_batch(
        self, trades: list[dict[str, Any]]
    ) -> list[TradeReview]:
        """Review trades using the batch API with fallback.

        Attempts to use the batch API for cost savings.  If batch
        creation or polling fails, falls back to sequential
        single-trade reviews.

        Args:
            trades: List of trade dicts.

        Returns:
            List of :class:`TradeReview` instances.
        """
        try:
            batch_id = await self.create_batch_request(trades)
            results = await self.poll_batch_status(batch_id)
            return self._parse_batch_results(results, trades)
        except Exception:
            self._log.warning(
                "batch_review_failed_falling_back_to_sequential",
                exc_info=True,
            )
            return await self._review_trades_sequential(trades)

    async def _review_trades_sequential(
        self, trades: list[dict[str, Any]]
    ) -> list[TradeReview]:
        """Review trades one at a time as a fallback.

        Args:
            trades: List of trade dicts.

        Returns:
            List of :class:`TradeReview` instances.
        """
        reviews: list[TradeReview] = []
        for trade in trades:
            review = await self._review_single_trade(trade)
            reviews.append(review)
        return reviews

    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        reraise=True,
    )
    async def _review_single_trade(self, trade: dict[str, Any]) -> TradeReview:
        """Review a single trade via the Claude API.

        Builds a detailed trade context, calls Claude for analysis,
        and parses the response into a structured review.

        Args:
            trade: Trade dict with full execution details.

        Returns:
            A graded :class:`TradeReview`.
        """
        trade_id = trade.get("trade_id", uuid.uuid4().hex)
        ticker = trade.get("ticker", "UNKNOWN")
        strategy = trade.get("strategy", "unknown")

        self._log.info(
            "reviewing_single_trade",
            trade_id=trade_id,
            ticker=ticker,
            strategy=strategy,
        )

        formatted = self._format_trade_for_review(trade)

        response = await self._client.messages.create(
            model=self._model,
            max_tokens=MAX_API_TOKENS,
            system=[
                {
                    "type": "text",
                    "text": JOURNAL_AGENT_SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                },
            ],
            messages=[
                {"role": "user", "content": formatted},
            ],
        )

        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        self._total_tokens += input_tokens + output_tokens

        return self._parse_review_response(response, trade)

    async def _generate_summary(
        self,
        reviews: list[TradeReview],
        trades: list[dict[str, Any]],
    ) -> DailySummary:
        """Generate an aggregated daily summary from individual reviews.

        Aggregates review data, calls Claude for cross-trade pattern
        analysis, and compiles strategy-level performance metrics.

        Args:
            reviews: Individual trade reviews.
            trades: Original trade dicts for additional context.

        Returns:
            A :class:`DailySummary` with aggregate statistics and
            insights.
        """
        today = datetime.now(tz=UTC).strftime("%Y-%m-%d")

        if not reviews:
            return DailySummary(
                date=today,
                total_trades=0,
                winners=0,
                losers=0,
                total_pnl=0.0,
            )

        total_pnl = sum(r.pnl for r in reviews)
        winners = sum(1 for r in reviews if r.pnl > 0)
        losers = sum(1 for r in reviews if r.pnl <= 0)

        # Find best and worst by grade, then by P&L
        sorted_by_grade = sorted(
            reviews,
            key=lambda r: (
                _GRADE_ORDER.get(r.grade, 0),
                r.pnl,
            ),
            reverse=True,
        )
        best_trade = sorted_by_grade[0] if sorted_by_grade else None
        worst_trade = sorted_by_grade[-1] if sorted_by_grade else None

        # Strategy performance aggregation
        strategy_perf = self._aggregate_strategy_performance(reviews)

        # Determine regime from trades
        regimes = [t.get("regime", "unknown") for t in trades if t.get("regime")]
        regime = max(set(regimes), key=regimes.count) if regimes else "unknown"

        # Generate cross-trade observations via Claude
        key_observations = await self._generate_observations(
            reviews, strategy_perf, regime
        )

        # Build FinMem memory updates
        memory_updates = self._build_memory_updates(
            reviews, strategy_perf, regime, today
        )

        return DailySummary(
            date=today,
            total_trades=len(reviews),
            winners=winners,
            losers=losers,
            total_pnl=round(total_pnl, 2),
            best_trade=best_trade,
            worst_trade=worst_trade,
            regime=regime,
            key_observations=key_observations,
            strategy_performance=strategy_perf,
            memory_updates=memory_updates,
        )

    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=2, max=15),
        reraise=True,
    )
    async def _generate_observations(
        self,
        reviews: list[TradeReview],
        strategy_perf: dict[str, dict[str, Any]],
        regime: str,
    ) -> list[str]:
        """Generate cross-trade pattern observations via Claude.

        Args:
            reviews: All trade reviews for the day.
            strategy_perf: Aggregated strategy performance dict.
            regime: Market regime string.

        Returns:
            List of observation strings.
        """
        review_summaries = "\n".join(
            f"- {r.ticker} ({r.strategy}): grade={r.grade}, "
            f"pnl=${r.pnl:.2f} ({r.pnl_pct:.1f}%), "
            f"lessons={'; '.join(r.lessons[:2])}"
            for r in reviews
        )

        perf_text = "\n".join(
            f"- {strat}: {data.get('trades', 0)} trades, "
            f"pnl=${data.get('pnl', 0):.2f}, "
            f"win_rate={data.get('win_rate', 0):.1%}"
            for strat, data in strategy_perf.items()
        )

        prompt = (
            f"Analyse the following trading day and identify "
            f"cross-trade patterns, regime-specific insights, and "
            f"actionable observations.\n\n"
            f"Market Regime: {regime}\n\n"
            f"Trade Reviews:\n{review_summaries}\n\n"
            f"Strategy Performance:\n{perf_text}\n\n"
            f"Respond with a JSON array of 3-5 concise observation "
            f'strings. Example: ["observation 1", "observation 2"]'
        )

        try:
            response = await self._client.messages.create(
                model=self._model,
                max_tokens=1024,
                system=[
                    {
                        "type": "text",
                        "text": JOURNAL_AGENT_SYSTEM_PROMPT,
                        "cache_control": {"type": "ephemeral"},
                    },
                ],
                messages=[
                    {"role": "user", "content": prompt},
                ],
            )

            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens
            self._total_tokens += input_tokens + output_tokens

            text = self._extract_text(response)
            observations = self._parse_json_array(text)
            if observations:
                return observations
        except Exception:
            self._log.warning(
                "observation_generation_failed",
                exc_info=True,
            )

        # Fallback: generate basic observations from data
        return self._generate_fallback_observations(reviews, strategy_perf)

    # ------------------------------------------------------------------
    # Grading
    # ------------------------------------------------------------------

    def _calculate_grade(
        self,
        pnl_pct: float,
        entry_quality: float,
        exit_quality: float,
    ) -> str:
        """Assign a letter grade based on P&L and execution quality.

        Grading rubric:
          - A: Profitable with good entry and exit execution.
          - B: Profitable, or good analysis despite a loss.
          - C: Breakeven or small loss with reasonable analysis.
          - D: Loss due to poor execution or timing.
          - F: Loss due to violated rules or ignored signals.

        Args:
            pnl_pct: Realised P&L as a fraction of max risk.
            entry_quality: Entry quality score (0.0 to 1.0).
            exit_quality: Exit quality score (0.0 to 1.0).

        Returns:
            Letter grade string.
        """
        is_profitable = pnl_pct > GRADE_A_MIN_PNL_PCT
        good_entry = entry_quality >= QUALITY_GOOD_THRESHOLD
        good_exit = exit_quality >= QUALITY_GOOD_THRESHOLD
        reasonable_entry = entry_quality >= QUALITY_REASONABLE_THRESHOLD
        reasonable_exit = exit_quality >= QUALITY_REASONABLE_THRESHOLD

        if is_profitable and good_entry and good_exit:
            return "A"

        if is_profitable or (good_entry and good_exit):
            return "B"

        if pnl_pct >= GRADE_C_MIN_PNL_PCT and (reasonable_entry or reasonable_exit):
            return "C"

        if pnl_pct >= GRADE_D_MIN_PNL_PCT:
            return "D"

        return "F"

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    def _parse_review_response(
        self, response: Any, trade: dict[str, Any]
    ) -> TradeReview:
        """Parse a Claude response into a TradeReview.

        Attempts to extract structured JSON from the response.  Falls
        back to heuristic parsing if JSON extraction fails.

        Args:
            response: Anthropic API response object.
            trade: Original trade dict for fallback data.

        Returns:
            A populated :class:`TradeReview`.
        """
        text = self._extract_text(response)
        trade_id = str(trade.get("trade_id", uuid.uuid4().hex))
        ticker = trade.get("ticker", "UNKNOWN")
        strategy = trade.get("strategy", "unknown")
        pnl = float(trade.get("realized_pnl", 0.0))
        max_loss = float(trade.get("max_loss", 1.0))
        pnl_pct = (pnl / max_loss * 100.0) if max_loss != 0 else 0.0

        # Attempt structured JSON parse
        parsed = self._try_parse_json(text)
        if parsed:
            entry_quality_score = parsed.get("entry_quality_score", 0.5)
            exit_quality_score = parsed.get("exit_quality_score", 0.5)
            grade = self._calculate_grade(
                pnl_pct / 100.0,
                entry_quality_score,
                exit_quality_score,
            )

            return TradeReview(
                trade_id=trade_id,
                ticker=ticker,
                strategy=strategy,
                grade=parsed.get("grade", grade),
                pnl=pnl,
                pnl_pct=round(pnl_pct, 2),
                entry_quality=parsed.get("entry_quality", "Analysis pending"),
                exit_quality=parsed.get("exit_quality", "Analysis pending"),
                lessons=parsed.get("lessons", []),
                pattern_observations=parsed.get("pattern_observations", []),
                would_take_again=parsed.get("would_take_again", pnl > 0),
                improvement_suggestions=parsed.get("improvement_suggestions", []),
            )

        # Fallback: use heuristics and raw text
        entry_quality_score = 0.5
        exit_quality_score = 0.5
        grade = self._calculate_grade(
            pnl_pct / 100.0,
            entry_quality_score,
            exit_quality_score,
        )

        return TradeReview(
            trade_id=trade_id,
            ticker=ticker,
            strategy=strategy,
            grade=grade,
            pnl=pnl,
            pnl_pct=round(pnl_pct, 2),
            entry_quality=self._extract_section(text, "entry"),
            exit_quality=self._extract_section(text, "exit"),
            lessons=self._extract_list_section(text, "lessons"),
            pattern_observations=self._extract_list_section(text, "patterns"),
            would_take_again=pnl > 0,
            improvement_suggestions=self._extract_list_section(text, "improvements"),
        )

    def _parse_batch_results(
        self,
        results: list[dict[str, Any]],
        trades: list[dict[str, Any]],
    ) -> list[TradeReview]:
        """Parse batch API results into TradeReview instances.

        Matches batch results to their corresponding trades by
        custom_id and parses each into a structured review.

        Args:
            results: Batch result dicts with ``custom_id`` and
                ``content`` keys.
            trades: Original trade dicts.

        Returns:
            List of :class:`TradeReview` instances.
        """
        trade_map: dict[str, dict[str, Any]] = {}
        for trade in trades:
            tid = trade.get("trade_id", "")
            trade_map[f"review-{tid}"] = trade

        reviews: list[TradeReview] = []
        for result in results:
            custom_id = result.get("custom_id", "")
            trade = trade_map.get(custom_id, {})

            if not trade:
                # Try matching by index order
                idx = results.index(result)
                if idx < len(trades):
                    trade = trades[idx]

            if not result.get("success", False):
                self._log.warning(
                    "batch_result_failed",
                    custom_id=custom_id,
                    error=result.get("error"),
                )
                # Create a minimal review from trade data
                reviews.append(self._create_fallback_review(trade))
                continue

            content = result.get("content", "")
            review = self._parse_text_to_review(content, trade)
            reviews.append(review)

        return reviews

    def _parse_text_to_review(
        self,
        text: str,
        trade: dict[str, Any],
    ) -> TradeReview:
        """Parse raw text response into a TradeReview.

        Args:
            text: Claude response text.
            trade: Original trade dict.

        Returns:
            A populated :class:`TradeReview`.
        """
        trade_id = str(trade.get("trade_id", uuid.uuid4().hex))
        ticker = trade.get("ticker", "UNKNOWN")
        strategy = trade.get("strategy", "unknown")
        pnl = float(trade.get("realized_pnl", 0.0))
        max_loss = float(trade.get("max_loss", 1.0))
        pnl_pct = (pnl / max_loss * 100.0) if max_loss != 0 else 0.0

        parsed = self._try_parse_json(text)
        if parsed:
            entry_quality_score = parsed.get("entry_quality_score", 0.5)
            exit_quality_score = parsed.get("exit_quality_score", 0.5)
            grade = self._calculate_grade(
                pnl_pct / 100.0,
                entry_quality_score,
                exit_quality_score,
            )

            return TradeReview(
                trade_id=trade_id,
                ticker=ticker,
                strategy=strategy,
                grade=parsed.get("grade", grade),
                pnl=pnl,
                pnl_pct=round(pnl_pct, 2),
                entry_quality=parsed.get("entry_quality", "Analysis pending"),
                exit_quality=parsed.get("exit_quality", "Analysis pending"),
                lessons=parsed.get("lessons", []),
                pattern_observations=parsed.get("pattern_observations", []),
                would_take_again=parsed.get("would_take_again", pnl > 0),
                improvement_suggestions=parsed.get("improvement_suggestions", []),
            )

        entry_quality_score = 0.5
        exit_quality_score = 0.5
        grade = self._calculate_grade(
            pnl_pct / 100.0,
            entry_quality_score,
            exit_quality_score,
        )

        return TradeReview(
            trade_id=trade_id,
            ticker=ticker,
            strategy=strategy,
            grade=grade,
            pnl=pnl,
            pnl_pct=round(pnl_pct, 2),
            entry_quality=self._extract_section(text, "entry"),
            exit_quality=self._extract_section(text, "exit"),
            lessons=self._extract_list_section(text, "lessons"),
            pattern_observations=self._extract_list_section(text, "patterns"),
            would_take_again=pnl > 0,
            improvement_suggestions=self._extract_list_section(text, "improvements"),
        )

    def _create_fallback_review(self, trade: dict[str, Any]) -> TradeReview:
        """Create a minimal review when API review fails.

        Args:
            trade: Trade dict with basic trade information.

        Returns:
            A :class:`TradeReview` with heuristic grading.
        """
        trade_id = str(trade.get("trade_id", uuid.uuid4().hex))
        pnl = float(trade.get("realized_pnl", 0.0))
        max_loss = float(trade.get("max_loss", 1.0))
        pnl_pct = (pnl / max_loss * 100.0) if max_loss != 0 else 0.0
        grade = self._calculate_grade(pnl_pct / 100.0, 0.5, 0.5)

        return TradeReview(
            trade_id=trade_id,
            ticker=trade.get("ticker", "UNKNOWN"),
            strategy=trade.get("strategy", "unknown"),
            grade=grade,
            pnl=pnl,
            pnl_pct=round(pnl_pct, 2),
            entry_quality="Review unavailable (API fallback)",
            exit_quality="Review unavailable (API fallback)",
            lessons=["Manual review recommended"],
            pattern_observations=[],
            would_take_again=pnl > 0,
            improvement_suggestions=["Review this trade manually"],
        )

    # ------------------------------------------------------------------
    # Formatting
    # ------------------------------------------------------------------

    def _format_trade_for_review(self, trade: dict[str, Any]) -> str:
        """Format trade details into a clean text block for Claude.

        Organises all available trade data into a structured prompt
        that enables thorough analysis.

        Args:
            trade: Trade dict with execution details.

        Returns:
            Formatted multi-line string for the API message.
        """
        ticker = trade.get("ticker", "UNKNOWN")
        strategy = trade.get("strategy", "unknown")
        direction = trade.get("direction", "UNKNOWN")
        entry_price = trade.get("entry_price", 0.0)
        exit_price = trade.get("exit_price", 0.0)
        pnl = trade.get("realized_pnl", 0.0)
        max_loss = trade.get("max_loss", 0.0)
        max_profit = trade.get("max_profit", 0.0)
        pnl_pct = (pnl / max_loss * 100.0) if max_loss != 0 else 0.0
        entry_time = trade.get("entry_time", "N/A")
        exit_time = trade.get("exit_time", "N/A")
        regime_entry = trade.get("regime", "unknown")
        regime_exit = trade.get("regime_at_exit", regime_entry)
        iv_rank = trade.get("entry_iv_rank", "N/A")
        ml_confidence = trade.get("ml_confidence", "N/A")
        entry_reasoning = trade.get("entry_reasoning", "No reasoning recorded")
        exit_reasoning = trade.get("exit_reasoning", "No exit reasoning recorded")
        exit_type = trade.get("exit_type", "N/A")
        commission = trade.get("commission", 0.0)

        # Build legs description if available
        legs_text = "N/A"
        legs = trade.get("legs", [])
        if legs:
            leg_lines = []
            for leg in legs:
                leg_lines.append(
                    f"  {leg.get('action', '?')} "
                    f"{leg.get('quantity', '?')}x "
                    f"{leg.get('strike', '?')} "
                    f"{leg.get('right', '?')} "
                    f"exp {leg.get('expiry', '?')}"
                )
            legs_text = "\n".join(leg_lines)

        # Options chain snapshot if available
        chain_text = "N/A"
        chain = trade.get("options_chain_snapshot", {})
        if chain:
            chain_text = json.dumps(chain, indent=2)

        return (
            f"Review the following closed trade and provide a "
            f"detailed analysis.\n\n"
            f"=== TRADE DETAILS ===\n"
            f"Trade ID: {trade.get('trade_id', 'N/A')}\n"
            f"Ticker: {ticker}\n"
            f"Strategy: {strategy}\n"
            f"Direction: {direction}\n"
            f"Entry Price: ${entry_price:.4f}\n"
            f"Exit Price: ${exit_price:.4f}\n"
            f"Max Profit: ${max_profit:.2f}\n"
            f"Max Loss: ${max_loss:.2f}\n"
            f"Realised P&L: ${pnl:.2f} ({pnl_pct:.1f}% "
            f"of max risk)\n"
            f"Commission: ${commission:.2f}\n"
            f"Exit Type: {exit_type}\n\n"
            f"=== TIMING ===\n"
            f"Entry: {entry_time}\n"
            f"Exit: {exit_time}\n\n"
            f"=== MARKET CONDITIONS ===\n"
            f"Regime at Entry: {regime_entry}\n"
            f"Regime at Exit: {regime_exit}\n"
            f"IV Rank at Entry: {iv_rank}\n"
            f"ML Confidence: {ml_confidence}\n\n"
            f"=== LEGS ===\n{legs_text}\n\n"
            f"=== ENTRY REASONING ===\n"
            f"{entry_reasoning}\n\n"
            f"=== EXIT REASONING ===\n"
            f"{exit_reasoning}\n\n"
            f"=== OPTIONS CHAIN AT ENTRY ===\n"
            f"{chain_text}\n\n"
            f"Respond with a JSON object containing:\n"
            f'{{"grade": "A/B/C/D/F", '
            f'"entry_quality": "description", '
            f'"entry_quality_score": 0.0-1.0, '
            f'"exit_quality": "description", '
            f'"exit_quality_score": 0.0-1.0, '
            f'"lessons": ["lesson1", ...], '
            f'"pattern_observations": ["pattern1", ...], '
            f'"would_take_again": true/false, '
            f'"improvement_suggestions": ["suggestion1", ...]}}'
        )

    # ------------------------------------------------------------------
    # Memory (FinMem)
    # ------------------------------------------------------------------

    def _build_memory_updates(
        self,
        reviews: list[TradeReview],
        strategy_perf: dict[str, dict[str, Any]],
        regime: str,
        date_str: str,
    ) -> list[dict[str, Any]]:
        """Build FinMem layered memory updates.

        Three memory layers:
          - Short-term (5 trades): recent trade outcomes and lessons
          - Medium-term (30 trades): strategy performance trends
          - Long-term: regime-specific pattern observations

        Args:
            reviews: Trade reviews from this session.
            strategy_perf: Aggregated strategy performance.
            regime: Current market regime.
            date_str: ISO date string.

        Returns:
            List of memory update dicts with ``layer``, ``content``,
            and ``timestamp`` keys.
        """
        updates: list[dict[str, Any]] = []

        # Short-term memory: latest trade outcomes
        recent = reviews[-FINMEM_SHORT_WINDOW:]
        for review in recent:
            updates.append(
                {
                    "layer": "short",
                    "type": "trade_outcome",
                    "content": {
                        "trade_id": review.trade_id,
                        "ticker": review.ticker,
                        "strategy": review.strategy,
                        "grade": review.grade,
                        "pnl": review.pnl,
                        "lessons": review.lessons[:3],
                    },
                    "timestamp": date_str,
                }
            )

        # Medium-term memory: strategy performance summary
        for strat, perf in strategy_perf.items():
            updates.append(
                {
                    "layer": "medium",
                    "type": "strategy_performance",
                    "content": {
                        "strategy": strat,
                        "regime": regime,
                        "trades": perf.get("trades", 0),
                        "pnl": perf.get("pnl", 0.0),
                        "win_rate": perf.get("win_rate", 0.0),
                    },
                    "timestamp": date_str,
                }
            )

        # Long-term memory: regime pattern observations
        all_patterns: list[str] = []
        for review in reviews:
            all_patterns.extend(review.pattern_observations)

        if all_patterns:
            updates.append(
                {
                    "layer": "long",
                    "type": "regime_patterns",
                    "content": {
                        "regime": regime,
                        "patterns": list(set(all_patterns)),
                        "num_trades": len(reviews),
                        "total_pnl": sum(r.pnl for r in reviews),
                    },
                    "timestamp": date_str,
                }
            )

        return updates

    # ------------------------------------------------------------------
    # Aggregation helpers
    # ------------------------------------------------------------------

    def _aggregate_strategy_performance(
        self, reviews: list[TradeReview]
    ) -> dict[str, dict[str, Any]]:
        """Aggregate performance metrics by strategy.

        Args:
            reviews: List of trade reviews.

        Returns:
            Dict mapping strategy names to performance dicts
            containing ``trades``, ``pnl``, and ``win_rate``.
        """
        perf: dict[str, dict[str, Any]] = {}

        for review in reviews:
            strat = review.strategy
            if strat not in perf:
                perf[strat] = {
                    "trades": 0,
                    "pnl": 0.0,
                    "winners": 0,
                }

            perf[strat]["trades"] += 1
            perf[strat]["pnl"] += review.pnl
            if review.pnl > 0:
                perf[strat]["winners"] += 1

        # Calculate win rates
        for strat_data in perf.values():
            total = strat_data["trades"]
            strat_data["win_rate"] = strat_data["winners"] / total if total > 0 else 0.0
            strat_data["pnl"] = round(strat_data["pnl"], 2)

        return perf

    def _generate_fallback_observations(
        self,
        reviews: list[TradeReview],
        strategy_perf: dict[str, dict[str, Any]],
    ) -> list[str]:
        """Generate basic observations without Claude API.

        Used as a fallback when the API call for observations fails.

        Args:
            reviews: Trade reviews.
            strategy_perf: Aggregated strategy performance.

        Returns:
            List of observation strings.
        """
        observations: list[str] = []

        total_pnl = sum(r.pnl for r in reviews)
        if total_pnl > 0:
            observations.append(
                f"Net positive day with ${total_pnl:.2f} total P&L "
                f"across {len(reviews)} trades."
            )
        elif total_pnl < 0:
            observations.append(
                f"Net negative day with ${total_pnl:.2f} total P&L "
                f"across {len(reviews)} trades."
            )

        # Best/worst strategy
        if strategy_perf:
            best_strat = max(
                strategy_perf.items(),
                key=lambda x: x[1].get("pnl", 0),
            )
            worst_strat = min(
                strategy_perf.items(),
                key=lambda x: x[1].get("pnl", 0),
            )
            if best_strat[1].get("pnl", 0) > 0:
                observations.append(
                    f"Best performing strategy: "
                    f"{best_strat[0]} with "
                    f"${best_strat[1]['pnl']:.2f} P&L."
                )
            if worst_strat[1].get("pnl", 0) < 0:
                observations.append(
                    f"Worst performing strategy: "
                    f"{worst_strat[0]} with "
                    f"${worst_strat[1]['pnl']:.2f} P&L."
                )

        # Grade distribution
        grade_counts: dict[str, int] = {}
        for review in reviews:
            grade_counts[review.grade] = grade_counts.get(review.grade, 0) + 1
        if grade_counts:
            dist = ", ".join(f"{g}:{c}" for g, c in sorted(grade_counts.items()))
            observations.append(f"Grade distribution: {dist}.")

        return observations

    # ------------------------------------------------------------------
    # Text extraction and JSON parsing
    # ------------------------------------------------------------------

    def _extract_text(self, response: Any) -> str:
        """Extract text content from a Claude API response.

        Args:
            response: Anthropic API response object.

        Returns:
            Concatenated text from all text blocks.
        """
        parts: list[str] = []
        for block in response.content:
            if hasattr(block, "text"):
                parts.append(block.text)
        return "\n".join(parts) if parts else ""

    def _try_parse_json(self, text: str) -> dict[str, Any] | None:
        """Attempt to extract and parse JSON from response text.

        Handles both pure JSON responses and responses with JSON
        embedded in markdown code blocks.

        Args:
            text: Raw response text.

        Returns:
            Parsed dict or ``None`` if parsing fails.
        """
        # Try direct parse
        try:
            return json.loads(text.strip())
        except (json.JSONDecodeError, ValueError):
            pass

        # Try extracting from code block
        json_start = text.find("{")
        json_end = text.rfind("}") + 1
        if json_start >= 0 and json_end > json_start:
            try:
                return json.loads(text[json_start:json_end])
            except (json.JSONDecodeError, ValueError):
                pass

        return None

    def _parse_json_array(self, text: str) -> list[str]:
        """Parse a JSON array of strings from response text.

        Args:
            text: Raw response text.

        Returns:
            List of strings, or empty list if parsing fails.
        """
        try:
            result = json.loads(text.strip())
            if isinstance(result, list):
                return [str(item) for item in result]
        except (json.JSONDecodeError, ValueError):
            pass

        # Try extracting from code block
        arr_start = text.find("[")
        arr_end = text.rfind("]") + 1
        if arr_start >= 0 and arr_end > arr_start:
            try:
                result = json.loads(text[arr_start:arr_end])
                if isinstance(result, list):
                    return [str(item) for item in result]
            except (json.JSONDecodeError, ValueError):
                pass

        return []

    def _extract_section(self, text: str, keyword: str) -> str:
        """Extract a text section following a keyword header.

        Searches for lines containing the keyword and returns the
        following content until the next section header.

        Args:
            text: Full response text.
            keyword: Section keyword to search for.

        Returns:
            Extracted section text, or a default message.
        """
        lines = text.split("\n")
        capturing = False
        captured: list[str] = []

        for line in lines:
            lower_line = line.lower().strip()
            if keyword.lower() in lower_line and (
                lower_line.startswith("#")
                or lower_line.startswith("**")
                or lower_line.endswith(":")
            ):
                capturing = True
                continue
            if capturing:
                if (
                    line.strip().startswith("#") or line.strip().startswith("**")
                ) and captured:
                    break
                if line.strip():
                    captured.append(line.strip())

        return " ".join(captured) if captured else "Analysis pending"

    def _extract_list_section(self, text: str, keyword: str) -> list[str]:
        """Extract a bulleted list section from response text.

        Args:
            text: Full response text.
            keyword: Section keyword to search for.

        Returns:
            List of extracted items.
        """
        section = self._extract_section(text, keyword)
        if section == "Analysis pending":
            return []

        items: list[str] = []
        for part in section.split("\n"):
            cleaned = part.strip().lstrip("-*").strip()
            if cleaned:
                items.append(cleaned)

        if not items and section != "Analysis pending":
            items = [section]

        return items

    # ------------------------------------------------------------------
    # Cost estimation
    # ------------------------------------------------------------------

    def _estimate_cost(self, total_tokens: int) -> float:
        """Estimate API cost for the review session.

        Uses approximate per-token pricing with the 50% batch
        discount applied.

        Args:
            total_tokens: Total tokens consumed (input + output).

        Returns:
            Estimated cost in USD.
        """
        # Rough split: 70% input, 30% output
        input_tokens = int(total_tokens * 0.70)
        output_tokens = total_tokens - input_tokens

        input_cost = input_tokens * COST_PER_INPUT_TOKEN
        output_cost = output_tokens * COST_PER_OUTPUT_TOKEN
        total = (input_cost + output_cost) * BATCH_DISCOUNT

        return round(total, 4)
