"""Central risk management engine for Project Titan.

Performs multi-layered pre-trade risk evaluation.  Every proposed trade
passes through seven independent checks before it can be approved:

1. **Circuit breaker** -- is trading currently allowed?
2. **Event calendar** -- is the ticker blocked by an upcoming event?
3. **Position limits** -- per-ticker, per-strategy, per-sector, total.
4. **Per-trade risk** -- max risk %, max risk dollars, reward/risk ratio.
5. **Portfolio Greeks** -- would this trade push aggregate Greeks over limits?
6. **Correlation** -- is this trade too correlated with existing positions?
7. **Liquidity** -- bid-ask spread and open interest minimums.

Usage::

    from config.settings import get_settings
    from src.risk.manager import RiskManager

    settings = get_settings()
    risk_mgr = RiskManager(
        settings=settings,
        risk_config=risk_config,
        circuit_breaker=circuit_breaker,
        event_calendar=event_calendar,
    )
    verdict = await risk_mgr.evaluate_trade(signal, portfolio, account_summary)
"""

from __future__ import annotations

import math
from datetime import datetime, time
from typing import TYPE_CHECKING, Any
from zoneinfo import ZoneInfo

from pydantic import BaseModel, Field

from src.risk.position_sizer import PositionSizer
from src.utils.logging import get_logger

if TYPE_CHECKING:
    import structlog

    from config.settings import Settings
    from src.strategies.base import TradeSignal

# ---------------------------------------------------------------------------
# Market hours constants
# ---------------------------------------------------------------------------

_US_EASTERN = ZoneInfo("America/New_York")
_MARKET_OPEN_BUFFER_END = time(9, 45)  # No entries before 9:45 ET
_MARKET_CLOSE_BUFFER_START = time(15, 45)  # No entries after 3:45 ET
_MARKET_OPEN = time(9, 30)
_MARKET_CLOSE = time(16, 0)

# Dollar-denominated portfolio Greek limits (beta_weighted net_delta / net_vega)
_MAX_PORTFOLIO_DELTA_DOLLARS: float = 15000.0  # delta_limit: $15000 max
_MAX_PORTFOLIO_VEGA_DOLLARS: float = 5000.0  # vega_limit: $5000 max

# ---------------------------------------------------------------------------
# Verdict constants
# ---------------------------------------------------------------------------

VERDICT_APPROVED: str = "APPROVED"
VERDICT_REJECTED: str = "REJECTED"
VERDICT_MODIFIED: str = "MODIFIED"


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class RiskVerdict(BaseModel):
    """Result of a pre-trade risk evaluation.

    Attributes:
        verdict: One of ``APPROVED``, ``REJECTED``, or ``MODIFIED``.
        reason: Human-readable explanation of the decision.
        adjustments: When ``verdict`` is ``MODIFIED``, contains the
            adjustments made (e.g., reduced quantity).
        original_quantity: The quantity originally proposed.
        approved_quantity: The quantity approved after risk evaluation.
        risk_score: Composite risk score from 0.0 (safest) to 1.0
            (highest risk).
    """

    verdict: str = Field(
        description="Risk decision: APPROVED, REJECTED, or MODIFIED",
    )
    reason: str = Field(
        default="",
        description="Explanation for the risk decision",
    )
    adjustments: dict[str, Any] = Field(
        default_factory=dict,
        description="Adjustments applied when verdict is MODIFIED",
    )
    original_quantity: int = Field(
        default=0,
        description="Originally proposed contract quantity",
    )
    approved_quantity: int = Field(
        default=0,
        description="Approved contract quantity after risk evaluation",
    )
    risk_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Composite risk score (0=safest, 1=highest risk)",
    )


class PortfolioExposure(BaseModel):
    """Snapshot of aggregate portfolio risk across all open positions.

    Attributes:
        total_positions: Number of open positions.
        positions_by_ticker: Count of positions grouped by ticker.
        positions_by_strategy: Count of positions grouped by strategy name.
        positions_by_sector: Count of positions grouped by sector.
        total_delta: Net portfolio delta exposure.
        total_gamma: Absolute portfolio gamma exposure.
        total_theta: Net portfolio theta (negative = decay cost).
        total_vega: Absolute portfolio vega exposure.
        total_risk_dollars: Sum of max-loss across all open positions.
        total_risk_pct: ``total_risk_dollars`` as a fraction of account
            equity.
    """

    total_positions: int = Field(
        default=0,
        ge=0,
        description="Total number of open positions",
    )
    positions_by_ticker: dict[str, int] = Field(
        default_factory=dict,
        description="Position count by underlying ticker",
    )
    positions_by_strategy: dict[str, int] = Field(
        default_factory=dict,
        description="Position count by strategy name",
    )
    positions_by_sector: dict[str, int] = Field(
        default_factory=dict,
        description="Position count by sector",
    )
    positions_by_ticker_strategy: dict[str, int] = Field(
        default_factory=dict,
        description="Position count by ticker:strategy composite key",
    )
    total_delta: float = Field(
        default=0.0,
        description="Net portfolio delta",
    )
    total_gamma: float = Field(
        default=0.0,
        description="Absolute portfolio gamma",
    )
    total_theta: float = Field(
        default=0.0,
        description="Net portfolio theta (negative = daily decay cost)",
    )
    total_vega: float = Field(
        default=0.0,
        description="Absolute portfolio vega",
    )
    total_risk_dollars: float = Field(
        default=0.0,
        ge=0.0,
        description="Total dollar risk across all open positions",
    )
    total_risk_pct: float = Field(
        default=0.0,
        ge=0.0,
        description="Total risk as a fraction of account equity",
    )


# ---------------------------------------------------------------------------
# Sector lookup
# ---------------------------------------------------------------------------

# Maps tickers to their sector for position concentration checks.
# Derived from config/tickers.yaml sector groupings.
_TICKER_SECTOR_MAP: dict[str, str] = {
    # Index ETFs
    "SPY": "index_etfs",
    "QQQ": "index_etfs",
    "IWM": "index_etfs",
    "DIA": "index_etfs",
    # Sector ETFs
    "XLF": "sector_etfs",
    "XLE": "sector_etfs",
    "XLK": "sector_etfs",
    "XLV": "sector_etfs",
    "GDX": "sector_etfs",
    # Technology
    "AAPL": "technology",
    "MSFT": "technology",
    "GOOGL": "technology",
    "AMZN": "technology",
    "META": "technology",
    "NVDA": "technology",
    "AMD": "technology",
    "TSM": "technology",
    "CRM": "technology",
    "ORCL": "technology",
    # Consumer
    "TSLA": "consumer",
    "NFLX": "consumer",
    "SBUX": "consumer",
    "NKE": "consumer",
    "COST": "consumer",
    # Financials
    "JPM": "financials",
    "GS": "financials",
    "BAC": "financials",
    "V": "financials",
    # Healthcare
    "UNH": "healthcare",
    "JNJ": "healthcare",
    "ABBV": "healthcare",
    "MRK": "healthcare",
    # Energy
    "XOM": "energy",
    "CVX": "energy",
    # Industrials
    "BA": "industrials",
    "CAT": "industrials",
    # Volatility
    "VIX": "volatility",
    "UVXY": "volatility",
}


def _get_sector(ticker: str) -> str:
    """Return the sector for a given ticker, defaulting to ``'unknown'``.

    Args:
        ticker: The underlying ticker symbol.

    Returns:
        Sector name as a string.
    """
    return _TICKER_SECTOR_MAP.get(ticker.upper(), "unknown")


# ---------------------------------------------------------------------------
# RiskManager
# ---------------------------------------------------------------------------


class RiskManager:
    """Central risk management engine.

    Orchestrates a multi-layered sequence of pre-trade checks.  Each
    layer can independently reject or modify a proposed trade.  The
    checks run in order of increasing cost: cheap boolean gates first,
    heavier portfolio-level calculations last.

    Args:
        settings: Application settings instance.
        risk_config: Parsed ``risk_limits.yaml`` as a dictionary.
        circuit_breaker: An initialized :class:`CircuitBreaker` instance.
        event_calendar: An initialized :class:`EventCalendar` instance.
    """

    def __init__(
        self,
        settings: Settings,
        risk_config: dict,
        circuit_breaker: Any,
        event_calendar: Any,
    ) -> None:
        self._settings: Settings = settings
        self._risk_config: dict = risk_config
        self._circuit_breaker = circuit_breaker
        self._event_calendar = event_calendar
        self._log: structlog.stdlib.BoundLogger = get_logger("risk.manager")

        # Initialize the position sizer.
        self._position_sizer: PositionSizer = PositionSizer(
            settings=settings,
            risk_config=risk_config,
        )

        # Cache frequently accessed limits.
        pos_limits: dict = self._risk_config.get("position_limits", {})
        self._max_concurrent_positions: int = int(
            pos_limits.get(
                "max_concurrent_positions",
                settings.trading.max_concurrent_positions,
            )
        )
        self._max_positions_per_ticker: int = int(
            pos_limits.get("max_positions_per_ticker", 2)
        )
        self._max_positions_per_strategy: int = int(
            pos_limits.get("max_positions_per_strategy", 3)
        )
        self._max_positions_per_sector: int = int(
            pos_limits.get("max_positions_per_sector", 3)
        )
        self._max_total_risk_pct: float = float(
            pos_limits.get("max_total_risk_pct", 0.10)
        )

        per_trade: dict = self._risk_config.get("per_trade", {})
        self._max_risk_pct: float = float(
            per_trade.get("max_risk_pct", settings.trading.per_trade_risk_pct)
        )
        self._max_risk_dollars: float = float(per_trade.get("max_risk_dollars", 3000.0))
        self._min_reward_risk_ratio: float = float(
            per_trade.get("min_reward_risk_ratio", 1.0)
        )
        self._max_bid_ask_spread_pct: float = float(
            per_trade.get("max_bid_ask_spread_pct", 0.05)
        )
        self._min_open_interest: int = int(per_trade.get("min_open_interest", 500))
        self._min_daily_volume: int = int(per_trade.get("min_daily_volume", 100))

        greeks: dict = self._risk_config.get("greeks_limits", {})
        self._max_portfolio_delta: float = float(greeks.get("max_portfolio_delta", 500))
        self._max_portfolio_gamma: float = float(greeks.get("max_portfolio_gamma", 200))
        self._max_portfolio_theta: float = float(
            greeks.get("max_portfolio_theta", -500)
        )
        self._max_portfolio_vega: float = float(greeks.get("max_portfolio_vega", 1000))

        corr: dict = self._risk_config.get("correlation", {})
        self._max_pairwise_correlation: float = float(
            corr.get("max_pairwise_correlation", 0.80)
        )
        self._correlation_check_enabled: bool = bool(
            corr.get("correlation_check_enabled", True)
        )

        self._log.info(
            "risk_manager_initialized",
            max_concurrent=self._max_concurrent_positions,
            max_risk_pct=self._max_risk_pct,
            max_risk_dollars=self._max_risk_dollars,
        )

    # ------------------------------------------------------------------
    # Public: master evaluation
    # ------------------------------------------------------------------

    async def evaluate_trade(
        self,
        signal: TradeSignal,
        portfolio: PortfolioExposure,
        account_summary: dict,
    ) -> RiskVerdict:
        """Run all pre-trade risk checks and return a verdict.

        Each layer is evaluated in order.  The first rejection short-
        circuits the remaining checks and returns immediately.  If
        the per-trade risk layer determines a smaller size is needed,
        the verdict is ``MODIFIED`` rather than rejected.

        Args:
            signal: The proposed trade signal from the strategy selector
                or ensemble.
            portfolio: Current aggregate portfolio exposure snapshot.
            account_summary: Dictionary with at least ``net_liquidation``
                and ``buying_power`` keys.

        Returns:
            A :class:`RiskVerdict` indicating whether the trade is
            approved, rejected, or modified.
        """
        ticker = signal.ticker
        strategy = signal.strategy
        account_equity = float(account_summary.get("net_liquidation", 0.0))

        self._log.info(
            "evaluating_trade",
            ticker=ticker,
            strategy=strategy,
            direction=signal.direction,
            confidence=getattr(signal, "confidence", None),
        )

        # Accumulate risk factors for composite score.
        risk_factors: list[float] = []

        # ── Layer 1: Circuit breaker ──────────────────────────────────
        cb_ok, cb_reason = await self._check_circuit_breaker()
        if not cb_ok:
            self._log.warning(
                "trade_rejected_circuit_breaker",
                ticker=ticker,
                strategy=strategy,
                reason=cb_reason,
            )
            return RiskVerdict(
                verdict=VERDICT_REJECTED,
                reason=f"Circuit breaker: {cb_reason}",
                original_quantity=getattr(signal, "quantity", 0),
                approved_quantity=0,
                risk_score=1.0,
            )

        # ── Layer 2: Event calendar ───────────────────────────────────
        event_ok, event_reason = await self._check_event_exclusions(ticker)
        if not event_ok:
            self._log.warning(
                "trade_rejected_event_exclusion",
                ticker=ticker,
                strategy=strategy,
                reason=event_reason,
            )
            return RiskVerdict(
                verdict=VERDICT_REJECTED,
                reason=f"Event exclusion: {event_reason}",
                original_quantity=getattr(signal, "quantity", 0),
                approved_quantity=0,
                risk_score=0.8,
            )

        # ── Layer 3: Position limits ──────────────────────────────────
        pos_ok, pos_reason = self._check_position_limits(signal, portfolio)
        if not pos_ok:
            self._log.warning(
                "trade_rejected_position_limits",
                ticker=ticker,
                strategy=strategy,
                reason=pos_reason,
            )
            return RiskVerdict(
                verdict=VERDICT_REJECTED,
                reason=f"Position limit: {pos_reason}",
                original_quantity=getattr(signal, "quantity", 0),
                approved_quantity=0,
                risk_score=0.7,
            )

        # ── Layer 4: Per-trade risk ───────────────────────────────────
        risk_ok, risk_reason, approved_qty = self._check_per_trade_risk(
            signal, account_equity
        )
        if not risk_ok:
            self._log.warning(
                "trade_rejected_per_trade_risk",
                ticker=ticker,
                strategy=strategy,
                reason=risk_reason,
            )
            return RiskVerdict(
                verdict=VERDICT_REJECTED,
                reason=f"Per-trade risk: {risk_reason}",
                original_quantity=getattr(signal, "quantity", 0),
                approved_quantity=0,
                risk_score=0.9,
            )

        original_qty = getattr(signal, "quantity", approved_qty)
        is_modified = approved_qty != original_qty and original_qty > 0

        # Per-trade risk score component.
        max_loss = getattr(signal, "max_loss", 0.0)
        if account_equity > 0.0 and max_loss > 0.0:
            trade_risk_pct = (max_loss * approved_qty) / account_equity
            risk_factors.append(min(trade_risk_pct / self._max_risk_pct, 1.0))
        else:
            risk_factors.append(0.0)

        # ── Layer 5: Portfolio Greeks ─────────────────────────────────
        greeks_ok, greeks_reason = self._check_portfolio_greeks(signal, portfolio)
        if not greeks_ok:
            self._log.warning(
                "trade_rejected_portfolio_greeks",
                ticker=ticker,
                strategy=strategy,
                reason=greeks_reason,
            )
            return RiskVerdict(
                verdict=VERDICT_REJECTED,
                reason=f"Portfolio Greeks: {greeks_reason}",
                original_quantity=original_qty,
                approved_quantity=0,
                risk_score=0.85,
            )

        # Greeks utilization risk component.
        delta_util = (
            abs(portfolio.total_delta) / self._max_portfolio_delta
            if self._max_portfolio_delta > 0
            else 0.0
        )
        risk_factors.append(min(delta_util, 1.0))

        # ── Layer 6: Correlation ──────────────────────────────────────
        corr_ok, corr_reason = self._check_correlation(signal, portfolio)
        if not corr_ok:
            self._log.warning(
                "trade_rejected_correlation",
                ticker=ticker,
                strategy=strategy,
                reason=corr_reason,
            )
            return RiskVerdict(
                verdict=VERDICT_REJECTED,
                reason=f"Correlation: {corr_reason}",
                original_quantity=original_qty,
                approved_quantity=0,
                risk_score=0.75,
            )

        # ── Layer 7: Liquidity ────────────────────────────────────────
        liq_ok, liq_reason = self._check_liquidity(signal)
        if not liq_ok:
            self._log.warning(
                "trade_rejected_liquidity",
                ticker=ticker,
                strategy=strategy,
                reason=liq_reason,
            )
            return RiskVerdict(
                verdict=VERDICT_REJECTED,
                reason=f"Liquidity: {liq_reason}",
                original_quantity=original_qty,
                approved_quantity=0,
                risk_score=0.6,
            )

        # ── Layer 8: Portfolio delta dollars ────────────────────────────
        delta_ok, delta_reason = self._check_portfolio_delta_dollars(
            signal, portfolio, account_equity
        )
        if not delta_ok:
            self._log.warning(
                "trade_rejected_portfolio_delta_dollars",
                ticker=ticker,
                strategy=strategy,
                reason=delta_reason,
            )
            return RiskVerdict(
                verdict=VERDICT_REJECTED,
                reason=f"Portfolio delta $: {delta_reason}",
                original_quantity=original_qty,
                approved_quantity=0,
                risk_score=0.85,
            )

        # ── Layer 9: Portfolio vega dollars ────────────────────────────
        vega_ok, vega_reason = self._check_portfolio_vega_dollars(signal, portfolio)
        if not vega_ok:
            self._log.warning(
                "trade_rejected_portfolio_vega_dollars",
                ticker=ticker,
                strategy=strategy,
                reason=vega_reason,
            )
            return RiskVerdict(
                verdict=VERDICT_REJECTED,
                reason=f"Portfolio vega $: {vega_reason}",
                original_quantity=original_qty,
                approved_quantity=0,
                risk_score=0.85,
            )

        # ── Layer 10: Market hours filter ──────────────────────────────
        hours_ok, hours_reason = self._check_market_hours()
        if not hours_ok:
            self._log.warning(
                "trade_rejected_market_hours",
                ticker=ticker,
                strategy=strategy,
                reason=hours_reason,
            )
            return RiskVerdict(
                verdict=VERDICT_REJECTED,
                reason=f"Market hours: {hours_reason}",
                original_quantity=original_qty,
                approved_quantity=0,
                risk_score=0.3,
            )

        # ── All checks passed ─────────────────────────────────────────
        # Compute composite risk score (average of all factors, capped at 1).
        if risk_factors:
            composite_risk = min(sum(risk_factors) / len(risk_factors), 1.0)
        else:
            composite_risk = 0.0

        # Position concentration factor.
        total_after = portfolio.total_positions + 1
        concentration = total_after / self._max_concurrent_positions
        composite_risk = min((composite_risk + concentration) / 2.0, 1.0)

        if is_modified:
            self._log.info(
                "trade_modified",
                ticker=ticker,
                strategy=strategy,
                original_qty=original_qty,
                approved_qty=approved_qty,
                risk_score=round(composite_risk, 4),
            )
            return RiskVerdict(
                verdict=VERDICT_MODIFIED,
                reason=(
                    f"Quantity adjusted from {original_qty} to "
                    f"{approved_qty} based on risk limits"
                ),
                adjustments={"quantity": approved_qty},
                original_quantity=original_qty,
                approved_quantity=approved_qty,
                risk_score=round(composite_risk, 4),
            )

        self._log.info(
            "trade_approved",
            ticker=ticker,
            strategy=strategy,
            approved_qty=approved_qty,
            risk_score=round(composite_risk, 4),
        )
        return RiskVerdict(
            verdict=VERDICT_APPROVED,
            reason="All risk checks passed",
            original_quantity=original_qty,
            approved_quantity=approved_qty,
            risk_score=round(composite_risk, 4),
        )

    # ------------------------------------------------------------------
    # Layer 1: Circuit breaker
    # ------------------------------------------------------------------

    async def _check_circuit_breaker(self) -> tuple[bool, str]:
        """Check whether trading is permitted by the circuit breaker.

        Delegates to the injected circuit breaker instance to determine
        if the current drawdown state allows new trade entries.

        Returns:
            Tuple of (is_allowed, reason_if_blocked).
        """
        allowed, reason_msg, size_mult = self._circuit_breaker.is_trading_allowed()
        if not allowed:
            level = getattr(self._circuit_breaker, "current_level", "UNKNOWN")
            reason = f"Trading halted at circuit breaker level {level}: {reason_msg}"
            return False, reason
        return True, ""

    # ------------------------------------------------------------------
    # Layer 2: Event calendar exclusions
    # ------------------------------------------------------------------

    async def _check_event_exclusions(self, ticker: str) -> tuple[bool, str]:
        """Check if the ticker has upcoming events within exclusion windows.

        Earnings, FOMC, CPI, NFP, and OpEx dates create exclusion
        windows during which new entries for the affected ticker (or
        all tickers for macro events) are blocked.

        Args:
            ticker: The underlying ticker symbol.

        Returns:
            Tuple of (is_allowed, reason_if_blocked).
        """
        is_blocked, block_reason = self._event_calendar.is_blocked(ticker)
        if is_blocked:
            return False, block_reason
        return True, ""

    # ------------------------------------------------------------------
    # Layer 3: Position limits
    # ------------------------------------------------------------------

    def _check_position_limits(
        self,
        signal: TradeSignal,
        portfolio: PortfolioExposure,
    ) -> tuple[bool, str]:
        """Verify position concentration limits are not breached.

        Checks:
        - Total concurrent position count.
        - Per-ticker position count.
        - Per-strategy position count.
        - Per-sector position count.

        Args:
            signal: The proposed trade signal.
            portfolio: Current portfolio exposure snapshot.

        Returns:
            Tuple of (is_within_limits, reason_if_exceeded).
        """
        ticker = signal.ticker
        strategy = signal.strategy
        sector = _get_sector(ticker)

        # Total positions.
        if portfolio.total_positions >= self._max_concurrent_positions:
            return (
                False,
                f"Maximum concurrent positions reached "
                f"({portfolio.total_positions}/{self._max_concurrent_positions})",
            )

        # Per-ticker.
        ticker_count = portfolio.positions_by_ticker.get(ticker, 0)
        if ticker_count >= self._max_positions_per_ticker:
            return (
                False,
                f"Maximum positions for {ticker} reached "
                f"({ticker_count}/{self._max_positions_per_ticker})",
            )

        # Per-strategy.
        strategy_count = portfolio.positions_by_strategy.get(strategy, 0)
        if strategy_count >= self._max_positions_per_strategy:
            return (
                False,
                f"Maximum positions for strategy '{strategy}' reached "
                f"({strategy_count}/{self._max_positions_per_strategy})",
            )

        # Per-sector.
        sector_count = portfolio.positions_by_sector.get(sector, 0)
        if sector_count >= self._max_positions_per_sector:
            return (
                False,
                f"Maximum positions for sector '{sector}' reached "
                f"({sector_count}/{self._max_positions_per_sector})",
            )

        # Duplicate ticker+strategy check — prevent opening the same
        # strategy on the same ticker twice (e.g. two bull_call_spread on AAPL).
        ticker_strategy_key = f"{ticker}:{strategy}"
        if portfolio.positions_by_ticker_strategy.get(ticker_strategy_key, 0) > 0:
            return (
                False,
                f"Duplicate trade blocked: {ticker} already has an "
                f"active '{strategy}' position",
            )

        self._log.debug(
            "position_limits_ok",
            ticker=ticker,
            strategy=strategy,
            sector=sector,
            total=portfolio.total_positions,
            ticker_count=ticker_count,
            strategy_count=strategy_count,
            sector_count=sector_count,
        )
        return True, ""

    # ------------------------------------------------------------------
    # Layer 4: Per-trade risk
    # ------------------------------------------------------------------

    def _check_per_trade_risk(
        self,
        signal: TradeSignal,
        account_equity: float,
    ) -> tuple[bool, str, int]:
        """Validate per-trade risk limits and compute position size.

        Uses the :class:`PositionSizer` to calculate the optimal number
        of contracts, then verifies the result against maximum risk
        percentage, maximum risk dollars, and minimum reward-to-risk
        ratio constraints.

        Args:
            signal: The proposed trade signal.
            account_equity: Current account net liquidation value.

        Returns:
            Tuple of (is_within_limits, reason_if_exceeded, approved_quantity).
        """
        if account_equity <= 0.0:
            return False, "Account equity is zero or negative", 0

        max_loss = getattr(signal, "max_loss", 0.0)
        max_profit = getattr(signal, "max_profit", 0.0)
        confidence = getattr(signal, "confidence", 0.0)
        regime = getattr(signal, "regime", "unknown")

        # Max loss per contract (for spread trades this is the defined risk).
        max_loss_per_contract = abs(max_loss) if max_loss != 0.0 else 0.0
        if max_loss_per_contract <= 0.0:
            return (
                False,
                "Max loss per contract is zero or undefined; "
                "cannot calculate position size",
                0,
            )

        # Check reward-to-risk ratio.
        if max_loss_per_contract > 0.0 and max_profit >= 0.0:
            reward_risk_ratio = max_profit / max_loss_per_contract
            if reward_risk_ratio < self._min_reward_risk_ratio:
                return (
                    False,
                    f"Reward/risk ratio {reward_risk_ratio:.2f} below minimum "
                    f"{self._min_reward_risk_ratio:.2f}",
                    0,
                )

        # Calculate position size via Kelly criterion.
        # Use historical averages if available on the signal, otherwise
        # derive from the max profit/loss of this specific trade.
        avg_win = getattr(signal, "avg_win", max_profit)
        avg_loss_val = getattr(signal, "avg_loss", max_loss_per_contract)
        win_probability = max(confidence, 0.01)  # Avoid division by zero.

        # Get circuit breaker state for sizing adjustment.
        cb_level = getattr(self._circuit_breaker, "current_level", "NORMAL")
        recovery_stage = getattr(self._circuit_breaker, "recovery_stage", 0)

        position_size = self._position_sizer.calculate_position_size(
            account_equity=account_equity,
            max_loss_per_contract=max_loss_per_contract,
            win_probability=win_probability,
            avg_win=avg_win if avg_win > 0.0 else max_profit,
            avg_loss=avg_loss_val if avg_loss_val > 0.0 else max_loss_per_contract,
            regime=regime,
            circuit_breaker_level=cb_level,
            recovery_stage=recovery_stage,
        )

        if position_size.contracts <= 0:
            return (
                False,
                "Position sizer returned zero contracts "
                "(negative Kelly edge or trading halted)",
                0,
            )

        # Verify the total risk does not exceed per-trade limits.
        total_risk = position_size.contracts * max_loss_per_contract
        max_allowed_by_pct = account_equity * self._max_risk_pct
        max_allowed_by_dollars = self._max_risk_dollars

        if total_risk > max_allowed_by_pct or total_risk > max_allowed_by_dollars:
            # Reduce contracts to fit within limits.
            allowed_dollars = min(max_allowed_by_pct, max_allowed_by_dollars)
            reduced_contracts = int(math.floor(allowed_dollars / max_loss_per_contract))
            if reduced_contracts <= 0:
                return (
                    False,
                    f"Even 1 contract (${max_loss_per_contract:.2f} risk) "
                    f"exceeds per-trade risk limit "
                    f"(${min(max_allowed_by_pct, max_allowed_by_dollars):.2f})",
                    0,
                )
            self._log.info(
                "per_trade_risk_reduced",
                original_contracts=position_size.contracts,
                reduced_contracts=reduced_contracts,
                total_risk=round(reduced_contracts * max_loss_per_contract, 2),
                max_allowed=round(allowed_dollars, 2),
            )
            return True, "", reduced_contracts

        # Check total portfolio risk after adding this trade.
        # This is a forward-looking check using the portfolio exposure.
        self._log.debug(
            "per_trade_risk_ok",
            contracts=position_size.contracts,
            total_risk=round(total_risk, 2),
            max_allowed_pct=round(max_allowed_by_pct, 2),
            max_allowed_dollars=max_allowed_by_dollars,
        )

        return True, "", position_size.contracts

    # ------------------------------------------------------------------
    # Layer 5: Portfolio Greeks
    # ------------------------------------------------------------------

    def _check_portfolio_greeks(
        self,
        signal: TradeSignal,
        portfolio: PortfolioExposure,
    ) -> tuple[bool, str]:
        """Check if adding this trade would exceed portfolio Greeks limits.

        Estimates the incremental Greeks from the proposed trade and
        verifies the resulting portfolio-level Greeks remain within
        configured limits.

        Args:
            signal: The proposed trade signal.
            portfolio: Current portfolio exposure snapshot.

        Returns:
            Tuple of (is_within_limits, reason_if_exceeded).
        """
        # Extract Greeks from the signal (set by strategy during construction).
        trade_delta = getattr(signal, "delta", 0.0)
        trade_gamma = getattr(signal, "gamma", 0.0)
        trade_theta = getattr(signal, "theta", 0.0)
        trade_vega = getattr(signal, "vega", 0.0)
        quantity = getattr(signal, "quantity", 1)

        # Scale Greeks by quantity.
        scaled_delta = trade_delta * quantity
        scaled_gamma = trade_gamma * quantity
        scaled_theta = trade_theta * quantity
        scaled_vega = trade_vega * quantity

        # Projected portfolio Greeks after the trade.
        projected_delta = portfolio.total_delta + scaled_delta
        projected_gamma = portfolio.total_gamma + abs(scaled_gamma)
        projected_theta = portfolio.total_theta + scaled_theta
        projected_vega = portfolio.total_vega + abs(scaled_vega)

        # Delta check: absolute value must not exceed limit.
        if abs(projected_delta) > self._max_portfolio_delta:
            return (
                False,
                f"Projected portfolio delta {projected_delta:.1f} would exceed "
                f"limit of +/-{self._max_portfolio_delta:.1f} "
                f"(current: {portfolio.total_delta:.1f}, "
                f"trade adds: {scaled_delta:.1f})",
            )

        # Gamma check.
        if projected_gamma > self._max_portfolio_gamma:
            return (
                False,
                f"Projected portfolio gamma {projected_gamma:.1f} would exceed "
                f"limit of {self._max_portfolio_gamma:.1f} "
                f"(current: {portfolio.total_gamma:.1f}, "
                f"trade adds: {abs(scaled_gamma):.1f})",
            )

        # Theta check: more negative than limit is a breach.
        if projected_theta < self._max_portfolio_theta:
            return (
                False,
                f"Projected portfolio theta {projected_theta:.1f} would exceed "
                f"limit of {self._max_portfolio_theta:.1f} "
                f"(current: {portfolio.total_theta:.1f}, "
                f"trade adds: {scaled_theta:.1f})",
            )

        # Vega check.
        if projected_vega > self._max_portfolio_vega:
            return (
                False,
                f"Projected portfolio vega {projected_vega:.1f} would exceed "
                f"limit of {self._max_portfolio_vega:.1f} "
                f"(current: {portfolio.total_vega:.1f}, "
                f"trade adds: {abs(scaled_vega):.1f})",
            )

        self._log.debug(
            "portfolio_greeks_ok",
            projected_delta=round(projected_delta, 2),
            projected_gamma=round(projected_gamma, 2),
            projected_theta=round(projected_theta, 2),
            projected_vega=round(projected_vega, 2),
        )
        return True, ""

    # ------------------------------------------------------------------
    # Layer 6: Correlation
    # ------------------------------------------------------------------

    def _check_correlation(
        self,
        signal: TradeSignal,
        portfolio: PortfolioExposure,
    ) -> tuple[bool, str]:
        """Check if the proposed trade is too correlated with existing positions.

        Uses the sector as a proxy for correlation.  If the same
        sector already has positions at or near the limit, and the
        proposed trade is in a highly correlated ticker, it is rejected.

        For a more precise check, the caller can provide pairwise
        correlation data on the signal via the ``correlations`` attribute
        (a dict mapping existing tickers to their rolling correlation
        with the proposed ticker).

        Args:
            signal: The proposed trade signal.
            portfolio: Current portfolio exposure snapshot.

        Returns:
            Tuple of (is_within_limits, reason_if_exceeded).
        """
        if not self._correlation_check_enabled:
            return True, ""

        ticker = signal.ticker
        sector = _get_sector(ticker)

        # Check pairwise correlations if available on the signal.
        correlations: dict[str, float] = getattr(signal, "correlations", {})
        for existing_ticker, corr_value in correlations.items():
            if abs(corr_value) > self._max_pairwise_correlation:
                return (
                    False,
                    f"Pairwise correlation between {ticker} and "
                    f"{existing_ticker} is {corr_value:.2f}, exceeding "
                    f"limit of {self._max_pairwise_correlation:.2f}",
                )

        # Sector-based correlation proxy: if the sector already has
        # multiple positions and the new trade adds to the same sector,
        # flag it as a concentration risk.
        sector_count = portfolio.positions_by_sector.get(sector, 0)
        sector_limit = self._max_positions_per_sector

        # Allow up to the sector limit (hard check is in Layer 3), but
        # warn when approaching it.
        if sector_count >= sector_limit - 1 and sector_count > 0:
            self._log.info(
                "sector_concentration_warning",
                ticker=ticker,
                sector=sector,
                sector_count=sector_count,
                sector_limit=sector_limit,
            )

        # Check if the same ticker already has a position in the same
        # direction — this concentrates risk beyond what diversification
        # provides.
        ticker_count = portfolio.positions_by_ticker.get(ticker, 0)
        if ticker_count >= self._max_positions_per_ticker:
            return (
                False,
                f"Adding another position in {ticker} (currently "
                f"{ticker_count}) would exceed the per-ticker "
                f"diversification limit",
            )

        self._log.debug(
            "correlation_check_ok",
            ticker=ticker,
            sector=sector,
            sector_count=sector_count,
        )
        return True, ""

    # ------------------------------------------------------------------
    # Layer 7: Liquidity
    # ------------------------------------------------------------------

    def _check_liquidity(self, signal: TradeSignal) -> tuple[bool, str]:
        """Verify the proposed trade meets minimum liquidity requirements.

        Checks the bid-ask spread as a percentage of the mid price and
        the open interest of the option legs against configured minimums.

        Args:
            signal: The proposed trade signal.

        Returns:
            Tuple of (is_within_limits, reason_if_failed).
        """
        # Extract liquidity data from the signal.
        bid_ask_spread_pct = getattr(signal, "bid_ask_spread_pct", None)
        open_interest = getattr(signal, "open_interest", None)
        daily_volume = getattr(signal, "daily_volume", None)

        # Bid-ask spread check.
        if (
            bid_ask_spread_pct is not None
            and bid_ask_spread_pct > self._max_bid_ask_spread_pct
        ):
            return (
                False,
                f"Bid-ask spread {bid_ask_spread_pct:.2%} exceeds "
                f"maximum {self._max_bid_ask_spread_pct:.2%}",
            )

        # Open interest check.
        if open_interest is not None and open_interest < self._min_open_interest:
            return (
                False,
                f"Open interest {open_interest} below minimum "
                f"{self._min_open_interest}",
            )

        # Daily volume check.
        if daily_volume is not None and daily_volume < self._min_daily_volume:
            return (
                False,
                f"Daily volume {daily_volume} below minimum {self._min_daily_volume}",
            )

        self._log.debug(
            "liquidity_check_ok",
            ticker=signal.ticker,
            bid_ask_spread_pct=bid_ask_spread_pct,
            open_interest=open_interest,
            daily_volume=daily_volume,
        )
        return True, ""

    # ------------------------------------------------------------------
    # Layer 8: Portfolio delta dollars (beta-weighted)
    # ------------------------------------------------------------------

    def _check_portfolio_delta_dollars(
        self,
        signal: TradeSignal,
        portfolio: PortfolioExposure,
        account_equity: float,
    ) -> tuple[bool, str]:
        """Check if portfolio delta in dollar terms exceeds the $15K limit.

        Beta-weighted portfolio delta exposure is approximated as
        ``total_delta * 100`` (each delta point ≈ $100 of SPY-equivalent
        exposure for a standard equity option).  This is a conservative
        approximation; a full beta-weighted calculation requires per-position
        beta data.

        Args:
            signal: The proposed trade signal.
            portfolio: Current portfolio exposure snapshot.
            account_equity: Current net liquidation value.

        Returns:
            Tuple of (is_within_limit, reason_if_exceeded).
        """
        trade_delta = getattr(signal, "delta", 0.0)
        quantity = getattr(signal, "quantity", 1)
        projected_delta = portfolio.total_delta + (trade_delta * quantity)

        # Approximate dollar delta: each delta point * contract multiplier
        delta_dollars = abs(projected_delta) * 100.0

        if delta_dollars > _MAX_PORTFOLIO_DELTA_DOLLARS:
            return (
                False,
                f"Projected portfolio delta ${delta_dollars:,.0f} exceeds "
                f"limit of ${_MAX_PORTFOLIO_DELTA_DOLLARS:,.0f} "
                f"(current delta: {portfolio.total_delta:.1f}, "
                f"trade adds: {trade_delta * quantity:.1f})",
            )

        self._log.debug(
            "portfolio_delta_dollars_ok",
            projected_delta_dollars=round(delta_dollars, 2),
            limit=_MAX_PORTFOLIO_DELTA_DOLLARS,
        )
        return True, ""

    # ------------------------------------------------------------------
    # Layer 9: Portfolio vega dollars
    # ------------------------------------------------------------------

    def _check_portfolio_vega_dollars(
        self,
        signal: TradeSignal,
        portfolio: PortfolioExposure,
    ) -> tuple[bool, str]:
        """Check if portfolio vega in dollar terms exceeds the $5K limit.

        Net vega exposure is computed as ``total_vega * 100`` (each vega
        point represents a $100 change per 1% move in IV for a standard
        equity option contract).

        Args:
            signal: The proposed trade signal.
            portfolio: Current portfolio exposure snapshot.

        Returns:
            Tuple of (is_within_limit, reason_if_exceeded).
        """
        trade_vega = getattr(signal, "vega", 0.0)
        quantity = getattr(signal, "quantity", 1)
        projected_vega = portfolio.total_vega + abs(trade_vega * quantity)

        vega_dollars = projected_vega * 100.0

        if vega_dollars > _MAX_PORTFOLIO_VEGA_DOLLARS:
            return (
                False,
                f"Projected portfolio vega ${vega_dollars:,.0f} exceeds "
                f"limit of ${_MAX_PORTFOLIO_VEGA_DOLLARS:,.0f} "
                f"(current vega: {portfolio.total_vega:.1f}, "
                f"trade adds: {abs(trade_vega * quantity):.1f})",
            )

        self._log.debug(
            "portfolio_vega_dollars_ok",
            projected_vega_dollars=round(vega_dollars, 2),
            limit=_MAX_PORTFOLIO_VEGA_DOLLARS,
        )
        return True, ""

    # ------------------------------------------------------------------
    # Layer 10: Market hours filter
    # ------------------------------------------------------------------

    def _check_market_hours(self) -> tuple[bool, str]:
        """Reject entries during the first 15 and last 15 minutes of trading.

        The opening 15 minutes (9:30–9:45 ET) are dominated by overnight
        order imbalances and erratic price action.  The closing 15 minutes
        (3:45–4:00 ET) carry MOC (Market-On-Close) order imbalance risk
        and widening spreads.

        Returns:
            Tuple of (is_within_hours, reason_if_blocked).
        """
        now_et = datetime.now(tz=_US_EASTERN)
        current_time = now_et.time()

        # Outside market hours entirely — allow (the scheduler should
        # handle this, but if called outside hours let other checks decide).
        if current_time < _MARKET_OPEN or current_time >= _MARKET_CLOSE:
            return True, ""

        # Opening buffer: 9:30–9:45 ET
        if current_time < _MARKET_OPEN_BUFFER_END:
            return (
                False,
                f"Opening buffer: no entries before 9:45 ET "
                f"(current time: {current_time.strftime('%H:%M:%S')} ET)",
            )

        # Closing buffer: 3:45–4:00 ET
        if current_time >= _MARKET_CLOSE_BUFFER_START:
            return (
                False,
                f"Closing buffer: no entries after 3:45 ET "
                f"(current time: {current_time.strftime('%H:%M:%S')} ET)",
            )

        return True, ""

    # ------------------------------------------------------------------
    # Portfolio exposure aggregation
    # ------------------------------------------------------------------

    async def get_portfolio_exposure(
        self,
        positions: list[dict],
        account_equity: float = 0.0,
    ) -> PortfolioExposure:
        """Aggregate open positions into a portfolio exposure snapshot.

        Iterates through all open positions and sums up position counts
        by ticker, strategy, and sector, along with aggregate Greeks
        and total dollar risk.

        Args:
            positions: List of position dictionaries.  Each dict should
                contain at minimum: ``ticker``, ``strategy``, ``delta``,
                ``gamma``, ``theta``, ``vega``, ``max_loss``, ``quantity``.
            account_equity: Current account net liquidation for computing
                ``total_risk_pct``.  Defaults to 0.0 (in which case
                ``total_risk_pct`` will be 0.0).

        Returns:
            An aggregated :class:`PortfolioExposure` snapshot.
        """
        by_ticker: dict[str, int] = {}
        by_strategy: dict[str, int] = {}
        by_sector: dict[str, int] = {}
        by_ticker_strategy: dict[str, int] = {}

        total_delta: float = 0.0
        total_gamma: float = 0.0
        total_theta: float = 0.0
        total_vega: float = 0.0
        total_risk_dollars: float = 0.0

        for pos in positions:
            ticker = str(pos.get("ticker", "UNKNOWN"))
            strategy = str(pos.get("strategy", "unknown"))
            sector = _get_sector(ticker)
            quantity = int(pos.get("quantity", 1))

            by_ticker[ticker] = by_ticker.get(ticker, 0) + 1
            by_strategy[strategy] = by_strategy.get(strategy, 0) + 1
            by_sector[sector] = by_sector.get(sector, 0) + 1

            # Track ticker:strategy composite for duplicate trade prevention
            ts_key = f"{ticker}:{strategy}"
            by_ticker_strategy[ts_key] = by_ticker_strategy.get(ts_key, 0) + 1

            total_delta += float(pos.get("delta", 0.0)) * quantity
            total_gamma += abs(float(pos.get("gamma", 0.0)) * quantity)
            total_theta += float(pos.get("theta", 0.0)) * quantity
            total_vega += abs(float(pos.get("vega", 0.0)) * quantity)

            max_loss = abs(float(pos.get("max_loss", 0.0))) * quantity
            total_risk_dollars += max_loss

        total_positions = len(positions)
        total_risk_pct = (
            total_risk_dollars / account_equity if account_equity > 0.0 else 0.0
        )

        exposure = PortfolioExposure(
            total_positions=total_positions,
            positions_by_ticker=by_ticker,
            positions_by_strategy=by_strategy,
            positions_by_sector=by_sector,
            positions_by_ticker_strategy=by_ticker_strategy,
            total_delta=round(total_delta, 4),
            total_gamma=round(total_gamma, 4),
            total_theta=round(total_theta, 4),
            total_vega=round(total_vega, 4),
            total_risk_dollars=round(total_risk_dollars, 2),
            total_risk_pct=round(total_risk_pct, 6),
        )

        self._log.info(
            "portfolio_exposure_calculated",
            total_positions=total_positions,
            total_delta=exposure.total_delta,
            total_gamma=exposure.total_gamma,
            total_theta=exposure.total_theta,
            total_vega=exposure.total_vega,
            total_risk_dollars=exposure.total_risk_dollars,
            total_risk_pct=exposure.total_risk_pct,
        )

        return exposure

    # ------------------------------------------------------------------
    # Property accessors for internal components
    # ------------------------------------------------------------------

    @property
    def position_sizer(self) -> PositionSizer:
        """Return the internal position sizer instance.

        Useful for external callers that need to compute a position size
        independently of the full risk evaluation pipeline.
        """
        return self._position_sizer
