"""FinMem layered memory system for Project Titan AI agents.

Implements a three-tier memory architecture inspired by the FinMem paper:

1. **Short-term** (capacity: 5 trades) -- Full details of the most recent
   trade outcomes.  When the buffer fills, the oldest entry is evicted to
   medium-term.
2. **Medium-term** (capacity: 30 trades) -- Summarised patterns covering
   roughly one month of trading activity, aggregated by strategy-regime
   combination.
3. **Long-term** (unlimited) -- Regime-level performance statistics that
   accumulate over the system's entire lifetime and serve as institutional
   knowledge.

Persistence is handled via a PostgreSQL ``agent_memory`` table.  Each
memory layer is stored as a JSON blob so that the full state survives
application restarts.

Usage::

    from src.ai.memory import FinMemory, TradeMemory

    memory = FinMemory()
    await memory.load_from_db(pool)

    await memory.add_trade(TradeMemory(
        ticker="AAPL", strategy="bull_call_spread", ...
    ))

    context = await memory.get_context_for_analysis(regime="low_vol_trend")
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from src.utils.logging import get_logger

if TYPE_CHECKING:
    import asyncpg
    import structlog

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SHORT_TERM_CAPACITY: int = 5
MEDIUM_TERM_CAPACITY: int = 30

_ENSURE_TABLE_SQL: str = """\
CREATE TABLE IF NOT EXISTS agent_memory (
    id SERIAL PRIMARY KEY,
    layer VARCHAR(20) NOT NULL UNIQUE,
    data_json JSONB NOT NULL DEFAULT '[]'::jsonb,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
"""

_UPSERT_SQL: str = """\
INSERT INTO agent_memory (layer, data_json, updated_at)
VALUES ($1, $2::jsonb, NOW())
ON CONFLICT (layer)
DO UPDATE SET data_json = EXCLUDED.data_json, updated_at = NOW();
"""

_SELECT_SQL: str = """\
SELECT layer, data_json FROM agent_memory WHERE layer = $1;
"""

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class TradeMemory(BaseModel):
    """Full record of a single trade for short-term memory storage.

    Attributes:
        ticker: Underlying symbol.
        strategy: Strategy name that produced the trade.
        direction: ``LONG`` or ``SHORT``.
        entry_date: ISO-formatted entry timestamp.
        exit_date: ISO-formatted exit timestamp.
        pnl: Realised P&L in dollars.
        regime: Market regime at time of entry.
        confidence: ML ensemble confidence score at entry.
        grade: Journal Agent grade (``A`` through ``F``).
        lessons: List of lessons extracted by the Journal Agent.
    """

    ticker: str = Field(..., description="Underlying symbol")
    strategy: str = Field(..., description="Strategy name")
    direction: str = Field(..., description="LONG or SHORT")
    entry_date: str = Field(
        ...,
        description="ISO-formatted entry timestamp",
    )
    exit_date: str = Field(
        ...,
        description="ISO-formatted exit timestamp",
    )
    pnl: float = Field(..., description="Realised P&L in USD")
    regime: str = Field(..., description="Market regime at entry")
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="ML ensemble confidence at entry",
    )
    grade: str = Field(
        default="C",
        description="Journal Agent grade (A-F)",
    )
    lessons: list[str] = Field(
        default_factory=list,
        description="Lessons extracted from this trade",
    )


class PatternMemory(BaseModel):
    """Summarised pattern observation for medium-term memory.

    Attributes:
        pattern_type: Category of pattern (e.g. ``strategy_regime``,
            ``timing``, ``signal``, ``behavioral``, ``execution``).
        description: Human-readable description of the pattern.
        frequency: How often observed -- ``one_off``, ``recurring``,
            or ``confirmed``.
        regime: Market regime the pattern relates to, if applicable.
        strategies_affected: List of strategies involved.
        last_seen: ISO-formatted date of last observation.
        win_count: Number of winning trades in this pattern.
        loss_count: Number of losing trades in this pattern.
        total_pnl: Cumulative P&L in dollars.
    """

    pattern_type: str = Field(
        ...,
        description="Category: strategy_regime, timing, signal, behavioral, execution",
    )
    description: str = Field(
        ...,
        description="Human-readable pattern description",
    )
    frequency: str = Field(
        default="one_off",
        description="one_off, recurring, or confirmed",
    )
    regime: str = Field(
        default="",
        description="Market regime if applicable",
    )
    strategies_affected: list[str] = Field(
        default_factory=list,
        description="Strategies involved in this pattern",
    )
    last_seen: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat(),
        description="ISO-formatted date of last observation",
    )
    win_count: int = Field(
        default=0,
        description="Number of winning trades in pattern",
    )
    loss_count: int = Field(
        default=0,
        description="Number of losing trades in pattern",
    )
    total_pnl: float = Field(
        default=0.0,
        description="Cumulative P&L for trades in this pattern",
    )


class RegimeMemory(BaseModel):
    """Long-term regime-level performance statistics.

    Attributes:
        regime: Market regime identifier.
        avg_win_rate: Average win rate (0.0 -- 1.0) across all trades
            in this regime.
        avg_pnl: Average realised P&L per trade in dollars.
        best_strategies: Strategies ranked by performance in this regime
            (list of dicts with ``name``, ``win_rate``, ``avg_pnl``).
        worst_strategies: Strategies ranked worst to best.
        total_trades: Total number of trades in this regime.
        total_wins: Number of winning trades.
        total_losses: Number of losing trades.
        cumulative_pnl: Cumulative P&L in dollars.
        last_updated: ISO-formatted last-update timestamp.
    """

    regime: str = Field(..., description="Market regime identifier")
    avg_win_rate: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Average win rate (0.0-1.0)",
    )
    avg_pnl: float = Field(
        default=0.0,
        description="Average P&L per trade in USD",
    )
    best_strategies: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Strategies ranked best to worst with name, win_rate, avg_pnl",
    )
    worst_strategies: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Strategies ranked worst to best",
    )
    total_trades: int = Field(
        default=0,
        description="Total trades in this regime",
    )
    total_wins: int = Field(
        default=0,
        description="Total winning trades",
    )
    total_losses: int = Field(
        default=0,
        description="Total losing trades",
    )
    cumulative_pnl: float = Field(
        default=0.0,
        description="Cumulative P&L in USD",
    )
    last_updated: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat(),
        description="ISO-formatted last-update timestamp",
    )


class MemorySnapshot(BaseModel):
    """Complete snapshot of all three memory layers.

    Attributes:
        short_term: List of recent :class:`TradeMemory` entries.
        medium_term: List of :class:`PatternMemory` observations.
        long_term: List of :class:`RegimeMemory` statistics.
        summary: Human-readable summary of overall memory state.
    """

    short_term: list[TradeMemory] = Field(
        default_factory=list,
        description="Short-term trade memories (max 5)",
    )
    medium_term: list[PatternMemory] = Field(
        default_factory=list,
        description="Medium-term pattern observations (max 30)",
    )
    long_term: list[RegimeMemory] = Field(
        default_factory=list,
        description="Long-term regime statistics",
    )
    summary: str = Field(
        default="",
        description="Human-readable summary of memory state",
    )


# ---------------------------------------------------------------------------
# FinMemory implementation
# ---------------------------------------------------------------------------


class FinMemory:
    """Three-layer memory system for Project Titan AI agents.

    The memory layers are:

    * **short_term** -- up to :data:`SHORT_TERM_CAPACITY` (5) most recent
      :class:`TradeMemory` entries with full trade details.
    * **medium_term** -- up to :data:`MEDIUM_TERM_CAPACITY` (30)
      :class:`PatternMemory` observations summarising strategy/regime
      performance patterns.
    * **long_term** -- unlimited list of :class:`RegimeMemory` entries
      tracking cumulative per-regime statistics.

    All persistence is through PostgreSQL via an ``asyncpg`` connection
    pool passed to :meth:`save_to_db` and :meth:`load_from_db`.
    """

    def __init__(self) -> None:
        self._log: structlog.stdlib.BoundLogger = get_logger("ai.memory")
        self._short_term: list[TradeMemory] = []
        self._medium_term: list[PatternMemory] = []
        self._long_term: list[RegimeMemory] = []

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def short_term(self) -> list[TradeMemory]:
        """Return a copy of the short-term memory layer."""
        return list(self._short_term)

    @property
    def medium_term(self) -> list[PatternMemory]:
        """Return a copy of the medium-term memory layer."""
        return list(self._medium_term)

    @property
    def long_term(self) -> list[RegimeMemory]:
        """Return a copy of the long-term memory layer."""
        return list(self._long_term)

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    async def add_trade(self, trade_memory: TradeMemory) -> None:
        """Add a trade to short-term memory, rolling to medium-term if full.

        When the short-term buffer is at capacity, the oldest entry is
        removed and merged into the medium-term pattern layer via
        :meth:`_roll_to_medium_term`.

        Parameters
        ----------
        trade_memory:
            Completed trade to store.
        """
        if len(self._short_term) >= SHORT_TERM_CAPACITY:
            evicted = self._short_term.pop(0)
            self._log.info(
                "short_term_eviction",
                evicted_ticker=evicted.ticker,
                evicted_strategy=evicted.strategy,
                evicted_pnl=evicted.pnl,
            )
            await self._roll_to_medium_term(evicted)

        self._short_term.append(trade_memory)
        self._log.info(
            "trade_added_to_short_term",
            ticker=trade_memory.ticker,
            strategy=trade_memory.strategy,
            pnl=trade_memory.pnl,
            grade=trade_memory.grade,
            short_term_size=len(self._short_term),
        )

    async def consolidate(self) -> None:
        """Summarise medium-term patterns and update long-term regime stats.

        This method should be called periodically (e.g. weekly or after
        medium-term reaches capacity) to:

        1. Aggregate medium-term pattern data by regime.
        2. Update long-term :class:`RegimeMemory` statistics with the
           aggregated results.
        3. Prune the medium-term layer of patterns that have been
           fully absorbed into long-term memory.
        """
        self._log.info(
            "consolidation_started",
            medium_term_size=len(self._medium_term),
            long_term_regimes=len(self._long_term),
        )

        # Aggregate medium-term patterns by regime
        regime_aggregations: dict[str, dict[str, Any]] = {}
        for pattern in self._medium_term:
            regime = pattern.regime or "unknown"
            if regime not in regime_aggregations:
                regime_aggregations[regime] = {
                    "total_wins": 0,
                    "total_losses": 0,
                    "total_pnl": 0.0,
                    "strategy_stats": {},
                }

            agg = regime_aggregations[regime]
            agg["total_wins"] += pattern.win_count
            agg["total_losses"] += pattern.loss_count
            agg["total_pnl"] += pattern.total_pnl

            for strategy in pattern.strategies_affected:
                if strategy not in agg["strategy_stats"]:
                    agg["strategy_stats"][strategy] = {
                        "wins": 0,
                        "losses": 0,
                        "pnl": 0.0,
                    }
                ss = agg["strategy_stats"][strategy]
                ss["wins"] += pattern.win_count
                ss["losses"] += pattern.loss_count
                ss["pnl"] += pattern.total_pnl

        # Merge aggregations into long-term memory
        for regime, agg_data in regime_aggregations.items():
            regime_mem = self._find_or_create_regime(regime)

            regime_mem.total_wins += agg_data["total_wins"]
            regime_mem.total_losses += agg_data["total_losses"]
            regime_mem.cumulative_pnl += agg_data["total_pnl"]
            regime_mem.total_trades = regime_mem.total_wins + regime_mem.total_losses

            if regime_mem.total_trades > 0:
                regime_mem.avg_win_rate = (
                    regime_mem.total_wins / regime_mem.total_trades
                )
                regime_mem.avg_pnl = regime_mem.cumulative_pnl / regime_mem.total_trades

            # Update per-strategy rankings
            strategy_rankings: list[dict[str, Any]] = []
            for strat_name, strat_data in agg_data["strategy_stats"].items():
                total = strat_data["wins"] + strat_data["losses"]
                win_rate = strat_data["wins"] / total if total > 0 else 0.0
                avg_pnl = strat_data["pnl"] / total if total > 0 else 0.0
                strategy_rankings.append(
                    {
                        "name": strat_name,
                        "win_rate": round(win_rate, 4),
                        "avg_pnl": round(avg_pnl, 2),
                        "total_trades": total,
                    }
                )

            # Merge with existing strategy rankings
            existing_map: dict[str, dict[str, Any]] = {
                s["name"]: s for s in regime_mem.best_strategies
            }
            for new_strat in strategy_rankings:
                name = new_strat["name"]
                if name in existing_map:
                    old = existing_map[name]
                    combined_total = (
                        old.get("total_trades", 0) + new_strat["total_trades"]
                    )
                    if combined_total > 0:
                        combined_wins = (
                            old.get("win_rate", 0) * old.get("total_trades", 0)
                            + new_strat["win_rate"] * new_strat["total_trades"]
                        )
                        combined_pnl = (
                            old.get("avg_pnl", 0) * old.get("total_trades", 0)
                            + new_strat["avg_pnl"] * new_strat["total_trades"]
                        )
                        existing_map[name] = {
                            "name": name,
                            "win_rate": round(combined_wins / combined_total, 4),
                            "avg_pnl": round(combined_pnl / combined_total, 2),
                            "total_trades": combined_total,
                        }
                else:
                    existing_map[name] = new_strat

            sorted_strategies = sorted(
                existing_map.values(),
                key=lambda s: (s.get("win_rate", 0), s.get("avg_pnl", 0)),
                reverse=True,
            )
            regime_mem.best_strategies = sorted_strategies
            regime_mem.worst_strategies = list(reversed(sorted_strategies))
            regime_mem.last_updated = datetime.now(UTC).isoformat()

        # Prune confirmed patterns that have been absorbed
        confirmed_count = sum(
            1 for p in self._medium_term if p.frequency == "confirmed"
        )
        if confirmed_count > MEDIUM_TERM_CAPACITY // 2:
            self._medium_term = [
                p for p in self._medium_term if p.frequency != "confirmed"
            ]
            self._log.info(
                "pruned_confirmed_patterns",
                removed=confirmed_count,
                remaining=len(self._medium_term),
            )

        self._log.info(
            "consolidation_complete",
            regimes_updated=len(regime_aggregations),
            medium_term_remaining=len(self._medium_term),
        )

    async def get_context_for_analysis(self, regime: str) -> str:
        """Format relevant memories for the Analysis Agent's context.

        Returns a text block summarizing recent trade outcomes,
        applicable patterns for the given regime, and long-term
        strategy performance in this regime.

        Parameters
        ----------
        regime:
            Current market regime identifier.

        Returns
        -------
        str
            Formatted memory context for injection into the Analysis
            Agent prompt.
        """
        sections: list[str] = []

        # Short-term: recent trade outcomes
        if self._short_term:
            sections.append("=== RECENT TRADE OUTCOMES (Short-Term Memory) ===")
            for trade in self._short_term:
                outcome = "WIN" if trade.pnl > 0 else "LOSS"
                sections.append(
                    f"  {trade.ticker} | {trade.strategy} | {trade.direction} | "
                    f"{outcome} ${trade.pnl:+,.2f} | Regime: {trade.regime} | "
                    f"Confidence: {trade.confidence:.2f} | Grade: {trade.grade}"
                )
                if trade.lessons:
                    for lesson in trade.lessons:
                        sections.append(f"    Lesson: {lesson}")
            sections.append("")

        # Medium-term: relevant patterns for the current regime
        regime_patterns = [
            p for p in self._medium_term if p.regime == regime or p.regime == ""
        ]
        if regime_patterns:
            sections.append(
                f"=== RELEVANT PATTERNS FOR {regime.upper()} (Medium-Term Memory) ==="
            )
            for pattern in regime_patterns:
                total = pattern.win_count + pattern.loss_count
                win_rate = pattern.win_count / total if total > 0 else 0.0
                sections.append(
                    f"  [{pattern.frequency.upper()}] {pattern.description} "
                    f"| WR: {win_rate:.0%} ({total} trades) "
                    f"| P&L: ${pattern.total_pnl:+,.2f}"
                )
            sections.append("")

        # Long-term: regime-specific performance
        regime_mem = self._find_regime(regime)
        if regime_mem and regime_mem.total_trades > 0:
            sections.append(
                f"=== {regime.upper()} REGIME STATISTICS (Long-Term Memory) ==="
            )
            sections.append(
                f"  Total trades: {regime_mem.total_trades} | "
                f"Win rate: {regime_mem.avg_win_rate:.1%} | "
                f"Avg P&L: ${regime_mem.avg_pnl:+,.2f} | "
                f"Cumulative: ${regime_mem.cumulative_pnl:+,.2f}"
            )
            if regime_mem.best_strategies:
                sections.append("  Best strategies:")
                for strat in regime_mem.best_strategies[:3]:
                    sections.append(
                        f"    {strat['name']}: "
                        f"WR {strat.get('win_rate', 0):.0%}, "
                        f"Avg P&L ${strat.get('avg_pnl', 0):+,.2f} "
                        f"({strat.get('total_trades', 0)} trades)"
                    )
            if regime_mem.worst_strategies:
                sections.append("  Worst strategies:")
                for strat in regime_mem.worst_strategies[:2]:
                    sections.append(
                        f"    {strat['name']}: "
                        f"WR {strat.get('win_rate', 0):.0%}, "
                        f"Avg P&L ${strat.get('avg_pnl', 0):+,.2f} "
                        f"({strat.get('total_trades', 0)} trades)"
                    )
            sections.append("")

        if not sections:
            return "No trading memory available yet. This is a fresh start."

        return "\n".join(sections)

    async def get_context_for_journal(self, recent_trades: list[dict[str, Any]]) -> str:
        """Format memory context for the Journal Agent's review.

        Provides the Journal Agent with recent short-term memories and
        medium-term patterns so it can identify evolving trends and
        compare the new trades against recent history.

        Parameters
        ----------
        recent_trades:
            List of recently closed trade dicts to be reviewed.  Each
            should have ``"ticker"``, ``"strategy"``, ``"pnl"``,
            ``"regime"`` at minimum.

        Returns
        -------
        str
            Formatted context for the Journal Agent.
        """
        sections: list[str] = []

        # Short-term context
        if self._short_term:
            sections.append("=== RECENT TRADE HISTORY ===")
            for trade in self._short_term:
                outcome = "WIN" if trade.pnl > 0 else "LOSS"
                sections.append(
                    f"  {trade.ticker} | {trade.strategy} | {outcome} "
                    f"${trade.pnl:+,.2f} | Grade: {trade.grade} | "
                    f"Regime: {trade.regime}"
                )
            sections.append("")

        # Medium-term pattern context
        if self._medium_term:
            sections.append("=== ACTIVE PATTERNS ===")
            for pattern in self._medium_term[-10:]:
                sections.append(
                    f"  [{pattern.frequency.upper()}] {pattern.description}"
                )
            sections.append("")

        # Stats across recent trades being reviewed
        if recent_trades:
            total_pnl = sum(t.get("pnl", 0) for t in recent_trades)
            wins = sum(1 for t in recent_trades if t.get("pnl", 0) > 0)
            losses = len(recent_trades) - wins
            sections.append("=== TODAY'S TRADES FOR REVIEW ===")
            sections.append(
                f"  Total trades: {len(recent_trades)} | "
                f"Wins: {wins} | Losses: {losses} | "
                f"Net P&L: ${total_pnl:+,.2f}"
            )
            for trade in recent_trades:
                sections.append(
                    f"  {trade.get('ticker', '?')} | "
                    f"{trade.get('strategy', '?')} | "
                    f"${trade.get('pnl', 0):+,.2f} | "
                    f"Regime: {trade.get('regime', '?')}"
                )
            sections.append("")

        # Long-term regime stats for context
        if self._long_term:
            sections.append("=== REGIME PERFORMANCE SUMMARY ===")
            for regime_mem in self._long_term:
                if regime_mem.total_trades > 0:
                    sections.append(
                        f"  {regime_mem.regime}: "
                        f"{regime_mem.total_trades} trades, "
                        f"WR {regime_mem.avg_win_rate:.1%}, "
                        f"Avg P&L ${regime_mem.avg_pnl:+,.2f}"
                    )
            sections.append("")

        if not sections:
            return "No prior trading memory. This is the first review session."

        return "\n".join(sections)

    async def save_to_db(self, pool: asyncpg.Pool) -> None:
        """Persist all three memory layers to PostgreSQL.

        Creates the ``agent_memory`` table if it does not exist, then
        upserts each layer as a JSON blob.

        Parameters
        ----------
        pool:
            An asyncpg connection pool connected to the Titan database.
        """
        async with pool.acquire() as conn:
            await conn.execute(_ENSURE_TABLE_SQL)

            short_json = json.dumps([t.model_dump() for t in self._short_term])
            medium_json = json.dumps([p.model_dump() for p in self._medium_term])
            long_json = json.dumps([r.model_dump() for r in self._long_term])

            await conn.execute(_UPSERT_SQL, "short_term", short_json)
            await conn.execute(_UPSERT_SQL, "medium_term", medium_json)
            await conn.execute(_UPSERT_SQL, "long_term", long_json)

        self._log.info(
            "memory_saved_to_db",
            short_term_count=len(self._short_term),
            medium_term_count=len(self._medium_term),
            long_term_count=len(self._long_term),
        )

    async def load_from_db(self, pool: asyncpg.Pool) -> None:
        """Restore all memory layers from PostgreSQL.

        Creates the ``agent_memory`` table if it does not exist.
        If no data is found for a layer, the in-memory list remains
        empty.

        Parameters
        ----------
        pool:
            An asyncpg connection pool connected to the Titan database.
        """
        async with pool.acquire() as conn:
            await conn.execute(_ENSURE_TABLE_SQL)

            # Short-term
            row = await conn.fetchrow(_SELECT_SQL, "short_term")
            if row and row["data_json"]:
                raw = row["data_json"]
                parsed = json.loads(raw) if isinstance(raw, str) else raw
                self._short_term = [TradeMemory.model_validate(item) for item in parsed]

            # Medium-term
            row = await conn.fetchrow(_SELECT_SQL, "medium_term")
            if row and row["data_json"]:
                raw = row["data_json"]
                parsed = json.loads(raw) if isinstance(raw, str) else raw
                self._medium_term = [
                    PatternMemory.model_validate(item) for item in parsed
                ]

            # Long-term
            row = await conn.fetchrow(_SELECT_SQL, "long_term")
            if row and row["data_json"]:
                raw = row["data_json"]
                parsed = json.loads(raw) if isinstance(raw, str) else raw
                self._long_term = [RegimeMemory.model_validate(item) for item in parsed]

        self._log.info(
            "memory_loaded_from_db",
            short_term_count=len(self._short_term),
            medium_term_count=len(self._medium_term),
            long_term_count=len(self._long_term),
        )

    def get_snapshot(self) -> MemorySnapshot:
        """Return a complete snapshot of all three memory layers.

        Returns
        -------
        MemorySnapshot
            Frozen copy of the current memory state with a generated
            summary string.
        """
        # Build summary
        total_short = len(self._short_term)
        total_medium = len(self._medium_term)
        total_long_regimes = len(self._long_term)
        total_long_trades = sum(r.total_trades for r in self._long_term)

        recent_pnl = sum(t.pnl for t in self._short_term)
        recent_wins = sum(1 for t in self._short_term if t.pnl > 0)
        recent_losses = total_short - recent_wins

        summary_parts: list[str] = [
            f"Memory State: {total_short}/{SHORT_TERM_CAPACITY} short-term, "
            f"{total_medium}/{MEDIUM_TERM_CAPACITY} medium-term, "
            f"{total_long_regimes} regimes tracked.",
        ]

        if total_short > 0:
            summary_parts.append(
                f"Recent: {recent_wins}W/{recent_losses}L, P&L ${recent_pnl:+,.2f}."
            )

        if total_long_trades > 0:
            overall_pnl = sum(r.cumulative_pnl for r in self._long_term)
            overall_wr = sum(r.total_wins for r in self._long_term) / total_long_trades
            summary_parts.append(
                f"Lifetime: {total_long_trades} trades, "
                f"WR {overall_wr:.1%}, "
                f"Cumulative P&L ${overall_pnl:+,.2f}."
            )

        return MemorySnapshot(
            short_term=list(self._short_term),
            medium_term=list(self._medium_term),
            long_term=list(self._long_term),
            summary=" ".join(summary_parts),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _roll_to_medium_term(self, trade: TradeMemory) -> None:
        """Merge an evicted short-term trade into the medium-term layer.

        Looks for an existing :class:`PatternMemory` entry matching the
        trade's strategy-regime combination.  If found, updates its
        statistics.  If not, creates a new pattern entry.

        If the medium-term layer exceeds :data:`MEDIUM_TERM_CAPACITY`,
        a consolidation is triggered automatically.

        Parameters
        ----------
        trade:
            The trade being evicted from short-term memory.
        """
        pattern_key = f"{trade.strategy}_{trade.regime}"
        is_win = trade.pnl > 0

        existing_pattern: PatternMemory | None = None
        for pattern in self._medium_term:
            if (
                pattern.pattern_type == "strategy_regime"
                and pattern.regime == trade.regime
                and trade.strategy in pattern.strategies_affected
            ):
                existing_pattern = pattern
                break

        if existing_pattern is not None:
            if is_win:
                existing_pattern.win_count += 1
            else:
                existing_pattern.loss_count += 1
            existing_pattern.total_pnl += trade.pnl
            existing_pattern.last_seen = datetime.now(UTC).isoformat()

            total = existing_pattern.win_count + existing_pattern.loss_count
            win_rate = existing_pattern.win_count / total if total > 0 else 0.0
            existing_pattern.description = (
                f"{trade.strategy} in {trade.regime}: "
                f"{existing_pattern.win_count}W/{existing_pattern.loss_count}L "
                f"({win_rate:.0%} win rate), "
                f"cumulative P&L ${existing_pattern.total_pnl:+,.2f}"
            )

            # Upgrade frequency based on observation count
            if total >= 10:
                existing_pattern.frequency = "confirmed"
            elif total >= 3:
                existing_pattern.frequency = "recurring"

            self._log.debug(
                "medium_term_pattern_updated",
                pattern_key=pattern_key,
                win_count=existing_pattern.win_count,
                loss_count=existing_pattern.loss_count,
                total_pnl=existing_pattern.total_pnl,
            )
        else:
            outcome_word = "winning" if is_win else "losing"
            new_pattern = PatternMemory(
                pattern_type="strategy_regime",
                description=(
                    f"{trade.strategy} in {trade.regime}: "
                    f"1 {outcome_word} trade, "
                    f"P&L ${trade.pnl:+,.2f}"
                ),
                frequency="one_off",
                regime=trade.regime,
                strategies_affected=[trade.strategy],
                last_seen=datetime.now(UTC).isoformat(),
                win_count=1 if is_win else 0,
                loss_count=0 if is_win else 1,
                total_pnl=trade.pnl,
            )
            self._medium_term.append(new_pattern)
            self._log.debug(
                "medium_term_pattern_created",
                pattern_key=pattern_key,
                pnl=trade.pnl,
            )

        # Auto-consolidate if medium-term is full
        if len(self._medium_term) >= MEDIUM_TERM_CAPACITY:
            self._log.info(
                "auto_consolidation_triggered",
                medium_term_size=len(self._medium_term),
            )
            await self.consolidate()

    def _find_regime(self, regime: str) -> RegimeMemory | None:
        """Find a long-term regime memory by name.

        Parameters
        ----------
        regime:
            Regime identifier to search for.

        Returns
        -------
        RegimeMemory | None
            The matching regime memory, or ``None`` if not found.
        """
        for rm in self._long_term:
            if rm.regime == regime:
                return rm
        return None

    def _find_or_create_regime(self, regime: str) -> RegimeMemory:
        """Find or create a long-term regime memory entry.

        Parameters
        ----------
        regime:
            Regime identifier.

        Returns
        -------
        RegimeMemory
            Existing or newly created regime memory entry.
        """
        existing = self._find_regime(regime)
        if existing is not None:
            return existing

        new_regime = RegimeMemory(regime=regime)
        self._long_term.append(new_regime)
        self._log.info(
            "regime_memory_created",
            regime=regime,
        )
        return new_regime

    def __repr__(self) -> str:
        return (
            f"<FinMemory("
            f"short_term={len(self._short_term)}/{SHORT_TERM_CAPACITY}, "
            f"medium_term={len(self._medium_term)}/{MEDIUM_TERM_CAPACITY}, "
            f"long_term_regimes={len(self._long_term)}"
            f")>"
        )
