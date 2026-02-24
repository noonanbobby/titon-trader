"""Options-specific backtesting engine for Project Titan.

Provides a production-grade backtesting framework designed for multi-leg
options strategies.  The engine simulates realistic order fills with
configurable slippage, tracks open positions with daily mark-to-market
P&L, enforces mechanical exit rules (profit target, stop loss, DTE
expiry), and computes comprehensive performance metrics.

A walk-forward validation framework is included for out-of-sample
testing with purged embargo gaps between train and test periods.

Usage::

    from src.ml.backtest import OptionsBacktester, BacktestConfig

    config = BacktestConfig(
        initial_capital=150_000.0,
        slippage_pct=0.15,
    )
    bt = OptionsBacktester(config=config)
    result = bt.run(signals, price_data, options_data, "iron_condor")
    report = bt.generate_report(result)
"""

from __future__ import annotations

import math
import time
import uuid
from datetime import date, datetime, timedelta
from enum import StrEnum
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from src.utils.logging import get_logger

if TYPE_CHECKING:
    import structlog


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TRADING_DAYS_PER_YEAR: int = 252
RISK_FREE_RATE: float = 0.05
DEFAULT_SLIPPAGE_PCT: float = 0.15
DEFAULT_COMMISSION_PER_CONTRACT: float = 0.65
DEFAULT_MAX_SPREAD_PCT: float = 0.05
DEFAULT_CLOSE_BEFORE_EXPIRY_DTE: int = 5
DEFAULT_INITIAL_CAPITAL: float = 150_000.0
MAX_CONCURRENT_POSITIONS: int = 8
MAX_POSITIONS_PER_TICKER: int = 2
MAX_TOTAL_RISK_PCT: float = 0.10
ASSIGNMENT_RISK_DTE: int = 3
CONTRACT_MULTIPLIER: int = 100


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class FillSide(StrEnum):
    """Whether the fill is a buy or sell from the trader's perspective."""

    BUY = "BUY"
    SELL = "SELL"


class PositionStatus(StrEnum):
    """Lifecycle status of a backtest position."""

    OPEN = "OPEN"
    CLOSED = "CLOSED"
    EXPIRED = "EXPIRED"
    ASSIGNED = "ASSIGNED"


class ExitReason(StrEnum):
    """Why a backtest position was closed."""

    PROFIT_TARGET = "PROFIT_TARGET"
    STOP_LOSS = "STOP_LOSS"
    DTE_LIMIT = "DTE_LIMIT"
    ASSIGNMENT_RISK = "ASSIGNMENT_RISK"
    SIGNAL_EXIT = "SIGNAL_EXIT"
    END_OF_DATA = "END_OF_DATA"
    EXPIRED = "EXPIRED"


class BarFrequency(StrEnum):
    """Supported bar frequencies for backtesting."""

    DAILY = "daily"
    INTRADAY = "intraday"


# ---------------------------------------------------------------------------
# Pydantic data models
# ---------------------------------------------------------------------------


class FillSimulation(BaseModel):
    """Result of simulating a single-leg option fill.

    Attributes:
        strike: Strike price of the option.
        right: Option right -- ``C`` for call, ``P`` for put.
        expiry: Expiration date of the option.
        side: ``BUY`` or ``SELL`` from the trader's perspective.
        bid: Best bid at the time of the fill.
        ask: Best ask at the time of the fill.
        mid_price: Midpoint of bid and ask.
        fill_price: Simulated execution price after slippage.
        slippage: Dollar amount of slippage applied.
        spread_pct: Bid-ask spread as a percentage of mid price.
        rejected: Whether the fill was rejected due to liquidity.
        reject_reason: Human-readable rejection explanation.
        quantity: Number of contracts filled.
        commission: Total commission for this leg.
    """

    strike: float = Field(description="Strike price")
    right: str = Field(description="C or P")
    expiry: date = Field(description="Expiration date")
    side: str = Field(description="BUY or SELL")
    bid: float = Field(description="Best bid at fill time")
    ask: float = Field(description="Best ask at fill time")
    mid_price: float = Field(description="Midpoint of bid and ask")
    fill_price: float = Field(description="Simulated fill price")
    slippage: float = Field(description="Slippage in dollars")
    spread_pct: float = Field(
        description="Bid-ask spread as pct of mid",
    )
    rejected: bool = Field(
        default=False,
        description="Whether the fill was rejected",
    )
    reject_reason: str = Field(
        default="",
        description="Reason for rejection if any",
    )
    quantity: int = Field(default=1, description="Contracts filled")
    commission: float = Field(
        default=0.0,
        description="Total commission for this leg",
    )


class BacktestLeg(BaseModel):
    """A single leg within a backtest position.

    Attributes:
        strike: Strike price.
        right: ``C`` or ``P``.
        expiry: Expiration date.
        side: ``BUY`` or ``SELL``.
        quantity: Number of contracts.
        entry_price: Fill price at entry.
        current_price: Most recent mark-to-market price.
        delta: Option delta at entry.
        gamma: Option gamma at entry.
        theta: Option theta at entry.
        vega: Option vega at entry.
    """

    strike: float = Field(description="Strike price")
    right: str = Field(description="C or P")
    expiry: date = Field(description="Expiration date")
    side: str = Field(description="BUY or SELL")
    quantity: int = Field(default=1, description="Contracts")
    entry_price: float = Field(description="Fill price at entry")
    current_price: float = Field(
        default=0.0,
        description="Current mark-to-market price",
    )
    delta: float = Field(default=0.0, description="Delta at entry")
    gamma: float = Field(default=0.0, description="Gamma at entry")
    theta: float = Field(default=0.0, description="Theta at entry")
    vega: float = Field(default=0.0, description="Vega at entry")


class BacktestPosition(BaseModel):
    """An open or closed position tracked during the backtest.

    Attributes:
        position_id: Unique identifier for this position.
        ticker: Underlying symbol.
        strategy: Strategy name that generated this trade.
        direction: ``LONG`` (debit) or ``SHORT`` (credit).
        legs: List of legs composing the spread.
        entry_date: Date the position was opened.
        exit_date: Date the position was closed (None if open).
        entry_net_premium: Net premium at entry (positive = debit).
        exit_net_premium: Net premium at exit.
        max_profit: Theoretical max profit in dollars.
        max_loss: Theoretical max loss in dollars.
        quantity: Number of spread units.
        total_commission: Sum of all leg commissions.
        realized_pnl: Final P&L after closing.
        unrealized_pnl: Current mark-to-market P&L.
        status: Position lifecycle status.
        exit_reason: Why the position was closed.
        ml_confidence: ML confidence score at entry.
        profit_target_pct: Fraction of max profit to target.
        stop_loss_pct: Fraction of max loss for stop.
        net_delta: Aggregate delta of all legs.
        net_gamma: Aggregate gamma of all legs.
        net_theta: Aggregate theta of all legs.
        net_vega: Aggregate vega of all legs.
    """

    position_id: str = Field(description="Unique position ID")
    ticker: str = Field(description="Underlying symbol")
    strategy: str = Field(description="Strategy name")
    direction: str = Field(description="LONG or SHORT")
    legs: list[BacktestLeg] = Field(
        default_factory=list,
        description="Legs composing the spread",
    )
    entry_date: date = Field(description="Date position was opened")
    exit_date: date | None = Field(
        default=None,
        description="Date position was closed",
    )
    entry_net_premium: float = Field(
        description="Net premium at entry (positive=debit)",
    )
    exit_net_premium: float = Field(
        default=0.0,
        description="Net premium at exit",
    )
    max_profit: float = Field(description="Theoretical max profit")
    max_loss: float = Field(
        description="Theoretical max loss (positive)",
    )
    quantity: int = Field(default=1, description="Spread units")
    total_commission: float = Field(
        default=0.0,
        description="Total commissions paid",
    )
    realized_pnl: float = Field(
        default=0.0,
        description="Final realized P&L",
    )
    unrealized_pnl: float = Field(
        default=0.0,
        description="Current unrealized P&L",
    )
    status: str = Field(
        default=PositionStatus.OPEN,
        description="Position status",
    )
    exit_reason: str = Field(
        default="",
        description="Reason for closing",
    )
    ml_confidence: float = Field(
        default=0.0,
        description="ML confidence at entry",
    )
    profit_target_pct: float = Field(
        default=0.50,
        description="Profit target as fraction of max profit",
    )
    stop_loss_pct: float = Field(
        default=1.00,
        description="Stop loss as multiple of max loss",
    )
    net_delta: float = Field(default=0.0, description="Net delta")
    net_gamma: float = Field(default=0.0, description="Net gamma")
    net_theta: float = Field(default=0.0, description="Net theta")
    net_vega: float = Field(default=0.0, description="Net vega")


class BacktestTrade(BaseModel):
    """Summary record of a completed trade for reporting.

    Attributes:
        trade_id: Unique trade identifier.
        ticker: Underlying symbol.
        strategy: Strategy name.
        direction: ``LONG`` or ``SHORT``.
        entry_date: Entry date.
        exit_date: Exit date.
        entry_net_premium: Net premium paid/received at entry.
        exit_net_premium: Net premium paid/received at exit.
        max_profit: Theoretical max profit.
        max_loss: Theoretical max loss.
        quantity: Number of spread units.
        realized_pnl: Realized profit or loss.
        commission: Total commissions.
        pnl_pct: P&L as percentage of max loss.
        holding_days: Calendar days the trade was open.
        exit_reason: Why the trade was closed.
        ml_confidence: ML confidence at entry.
        n_legs: Number of legs in the spread.
    """

    trade_id: str = Field(description="Unique trade ID")
    ticker: str = Field(description="Underlying symbol")
    strategy: str = Field(description="Strategy name")
    direction: str = Field(description="LONG or SHORT")
    entry_date: date = Field(description="Entry date")
    exit_date: date = Field(description="Exit date")
    entry_net_premium: float = Field(description="Net premium at entry")
    exit_net_premium: float = Field(description="Net premium at exit")
    max_profit: float = Field(description="Theoretical max profit")
    max_loss: float = Field(description="Theoretical max loss")
    quantity: int = Field(default=1, description="Spread units")
    realized_pnl: float = Field(description="Realized P&L")
    commission: float = Field(description="Total commissions")
    pnl_pct: float = Field(description="P&L as pct of max loss")
    holding_days: int = Field(description="Days held")
    exit_reason: str = Field(description="Exit reason")
    ml_confidence: float = Field(default=0.0, description="ML score")
    n_legs: int = Field(default=2, description="Number of legs")


class BacktestMetrics(BaseModel):
    """Comprehensive performance metrics from a backtest run.

    Attributes:
        total_pnl: Total realized profit/loss in dollars.
        total_return_pct: Total return as a percentage of capital.
        win_rate: Fraction of trades that were profitable.
        profit_factor: Gross profits divided by gross losses.
        sharpe_ratio: Annualized Sharpe ratio of daily returns.
        sortino_ratio: Annualized Sortino ratio of daily returns.
        calmar_ratio: Annualized return divided by max drawdown.
        max_drawdown_pct: Maximum drawdown as a percentage.
        max_drawdown_dollars: Maximum drawdown in dollars.
        max_drawdown_duration_days: Longest drawdown in calendar days.
        avg_trade_pnl: Average P&L per trade.
        avg_winner: Average P&L of winning trades.
        avg_loser: Average P&L of losing trades.
        best_trade: P&L of the best single trade.
        worst_trade: P&L of the worst single trade.
        total_trades: Number of completed trades.
        winning_trades: Number of winning trades.
        losing_trades: Number of losing trades.
        total_commissions: Sum of all commissions paid.
        expectancy: Expected value per trade.
        avg_holding_days: Mean holding period in calendar days.
        max_concurrent_positions: Peak number of simultaneous positions.
        equity_curve: List of (date_str, equity) tuples.
    """

    total_pnl: float = Field(description="Total realized P&L")
    total_return_pct: float = Field(
        description="Total return as pct of initial capital",
    )
    win_rate: float = Field(description="Fraction of winners")
    profit_factor: float = Field(
        description="Gross profits / gross losses",
    )
    sharpe_ratio: float = Field(description="Annualized Sharpe")
    sortino_ratio: float = Field(description="Annualized Sortino")
    calmar_ratio: float = Field(description="Return / max drawdown")
    max_drawdown_pct: float = Field(
        description="Max drawdown percentage",
    )
    max_drawdown_dollars: float = Field(
        description="Max drawdown in dollars",
    )
    max_drawdown_duration_days: int = Field(
        description="Longest drawdown in days",
    )
    avg_trade_pnl: float = Field(description="Average P&L per trade")
    avg_winner: float = Field(description="Average winning trade")
    avg_loser: float = Field(description="Average losing trade")
    best_trade: float = Field(description="Best single trade P&L")
    worst_trade: float = Field(description="Worst single trade P&L")
    total_trades: int = Field(description="Total completed trades")
    winning_trades: int = Field(description="Count of winners")
    losing_trades: int = Field(description="Count of losers")
    total_commissions: float = Field(description="Total commissions")
    expectancy: float = Field(
        description="Expected value per trade",
    )
    avg_holding_days: float = Field(
        description="Average holding period in days",
    )
    max_concurrent_positions: int = Field(
        description="Peak simultaneous positions",
    )
    equity_curve: list[tuple[str, float]] = Field(
        default_factory=list,
        description="(date, equity) tuples",
    )


class BacktestConfig(BaseModel):
    """Configuration parameters for the backtesting engine.

    Attributes:
        initial_capital: Starting account equity in dollars.
        commission_per_contract: Commission per contract per leg.
        slippage_pct: Slippage as fraction of bid-ask spread.
        max_spread_pct: Reject fills where spread exceeds this
            fraction of mid price.
        max_concurrent_positions: Maximum open positions at once.
        max_positions_per_ticker: Maximum positions on one ticker.
        max_total_risk_pct: Maximum total account risk percentage.
        close_before_expiry_dte: Close positions at this DTE.
        assignment_risk_dte: Flag assignment risk below this DTE.
        bar_frequency: Bar data frequency.
        profit_target_pct: Default profit target fraction.
        stop_loss_pct: Default stop loss multiple.
        contract_multiplier: Options contract multiplier.
    """

    initial_capital: float = Field(
        default=DEFAULT_INITIAL_CAPITAL,
        gt=0,
        description="Starting account equity",
    )
    commission_per_contract: float = Field(
        default=DEFAULT_COMMISSION_PER_CONTRACT,
        ge=0,
        description="Commission per contract per leg",
    )
    slippage_pct: float = Field(
        default=DEFAULT_SLIPPAGE_PCT,
        ge=0.0,
        le=1.0,
        description="Slippage as fraction of bid-ask spread",
    )
    max_spread_pct: float = Field(
        default=DEFAULT_MAX_SPREAD_PCT,
        gt=0.0,
        le=1.0,
        description="Max bid-ask spread pct for fill acceptance",
    )
    max_concurrent_positions: int = Field(
        default=MAX_CONCURRENT_POSITIONS,
        ge=1,
        description="Maximum simultaneous open positions",
    )
    max_positions_per_ticker: int = Field(
        default=MAX_POSITIONS_PER_TICKER,
        ge=1,
        description="Maximum positions per underlying",
    )
    max_total_risk_pct: float = Field(
        default=MAX_TOTAL_RISK_PCT,
        gt=0.0,
        le=1.0,
        description="Maximum total risk as pct of equity",
    )
    close_before_expiry_dte: int = Field(
        default=DEFAULT_CLOSE_BEFORE_EXPIRY_DTE,
        ge=0,
        description="Close positions at this many DTE",
    )
    assignment_risk_dte: int = Field(
        default=ASSIGNMENT_RISK_DTE,
        ge=0,
        description="Assignment risk warning below this DTE",
    )
    bar_frequency: str = Field(
        default=BarFrequency.DAILY,
        description="Bar data frequency: daily or intraday",
    )
    profit_target_pct: float = Field(
        default=0.50,
        gt=0.0,
        description="Default profit target fraction of max profit",
    )
    stop_loss_pct: float = Field(
        default=1.00,
        gt=0.0,
        description="Default stop loss multiple of max loss",
    )
    contract_multiplier: int = Field(
        default=CONTRACT_MULTIPLIER,
        gt=0,
        description="Options contract multiplier",
    )


class BacktestResult(BaseModel):
    """Complete output from a backtest run.

    Attributes:
        strategy_name: Name of the strategy backtested.
        config: Configuration used for the backtest.
        trades: All completed trades.
        metrics: Performance metrics summary.
        start_date: First date in the backtest period.
        end_date: Last date in the backtest period.
        elapsed_seconds: Wall-clock time for the backtest run.
        n_signals_received: Total entry signals evaluated.
        n_signals_rejected: Signals rejected (risk, liquidity, etc.).
    """

    strategy_name: str = Field(description="Strategy backtested")
    config: BacktestConfig = Field(description="Backtest configuration")
    trades: list[BacktestTrade] = Field(
        default_factory=list,
        description="All completed trades",
    )
    metrics: BacktestMetrics = Field(
        description="Performance metrics",
    )
    start_date: str = Field(description="Backtest start date")
    end_date: str = Field(description="Backtest end date")
    elapsed_seconds: float = Field(
        description="Wall-clock run time in seconds",
    )
    n_signals_received: int = Field(
        default=0,
        description="Total entry signals evaluated",
    )
    n_signals_rejected: int = Field(
        default=0,
        description="Signals rejected by risk or liquidity",
    )


class WalkForwardFoldResult(BaseModel):
    """Result from a single walk-forward fold.

    Attributes:
        fold_num: Zero-based fold index.
        train_start: Start date of the training period.
        train_end: End date of the training period.
        test_start: Start date of the test period.
        test_end: End date of the test period.
        n_train_days: Trading days in training set.
        n_test_days: Trading days in test set.
        result: Full backtest result for the test period.
    """

    fold_num: int = Field(description="Zero-based fold index")
    train_start: str = Field(description="Training period start")
    train_end: str = Field(description="Training period end")
    test_start: str = Field(description="Test period start")
    test_end: str = Field(description="Test period end")
    n_train_days: int = Field(description="Training days")
    n_test_days: int = Field(description="Test days")
    result: BacktestResult = Field(description="Fold backtest result")


class WalkForwardResult(BaseModel):
    """Aggregated results from walk-forward validation.

    Attributes:
        n_folds: Number of walk-forward folds.
        embargo_days: Embargo gap between train and test periods.
        fold_results: Per-fold results.
        aggregate_metrics: Metrics aggregated across all test periods.
        total_elapsed_seconds: Total wall-clock time.
    """

    n_folds: int = Field(description="Number of folds")
    embargo_days: int = Field(description="Embargo gap in days")
    fold_results: list[WalkForwardFoldResult] = Field(
        default_factory=list,
        description="Per-fold results",
    )
    aggregate_metrics: BacktestMetrics = Field(
        description="Aggregated metrics across folds",
    )
    total_elapsed_seconds: float = Field(
        description="Total run time in seconds",
    )


# ---------------------------------------------------------------------------
# OptionsBacktester
# ---------------------------------------------------------------------------


class OptionsBacktester:
    """Options-specific backtesting engine with realistic fill simulation.

    Simulates multi-leg options trades with configurable slippage,
    enforces mechanical exit rules, tracks portfolio Greeks across
    all open positions, and computes comprehensive performance
    metrics including Sharpe, Sortino, Calmar ratios, and drawdown
    statistics.

    Args:
        config: Backtest configuration. Uses defaults if not provided.
        initial_capital: Starting account equity (overrides config).
        commission_per_contract: Per-contract commission (overrides
            config).
    """

    def __init__(
        self,
        config: BacktestConfig | None = None,
        initial_capital: float | None = None,
        commission_per_contract: float | None = None,
    ) -> None:
        self._config: BacktestConfig = config or BacktestConfig()

        if initial_capital is not None:
            self._config = self._config.model_copy(
                update={"initial_capital": initial_capital},
            )
        if commission_per_contract is not None:
            self._config = self._config.model_copy(
                update={
                    "commission_per_contract": commission_per_contract,
                },
            )

        self._log: structlog.stdlib.BoundLogger = get_logger(
            "ml.backtest",
        )
        self._open_positions: list[BacktestPosition] = []
        self._closed_positions: list[BacktestPosition] = []
        self._equity: float = self._config.initial_capital
        self._high_water_mark: float = self._equity
        self._equity_curve: list[tuple[str, float]] = []
        self._daily_returns: list[float] = []
        self._signals_received: int = 0
        self._signals_rejected: int = 0
        self._peak_concurrent: int = 0

        self._log.info(
            "backtester_initialized",
            initial_capital=self._config.initial_capital,
            slippage_pct=self._config.slippage_pct,
            commission=self._config.commission_per_contract,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        signals: pd.DataFrame,
        price_data: pd.DataFrame,
        options_data: pd.DataFrame,
        strategy_name: str,
    ) -> BacktestResult:
        """Run a full backtest over the provided data.

        Iterates through each bar in *price_data*, evaluates entry
        signals, manages open positions (mark-to-market, exit checks),
        and records all completed trades.

        Args:
            signals: DataFrame indexed by date with at minimum a
                ``confidence`` column and optionally ``direction``,
                ``ticker``, ``profit_target_pct``, ``stop_loss_pct``,
                ``max_profit``, and ``max_loss`` columns.
            price_data: OHLCV DataFrame indexed by date with columns
                ``open``, ``high``, ``low``, ``close``, ``volume``.
            options_data: DataFrame indexed by date containing options
                chain snapshots.  Expected columns: ``strike``,
                ``right``, ``expiry``, ``bid``, ``ask``, ``delta``,
                ``gamma``, ``theta``, ``vega``, ``open_interest``.
                Can also be a MultiIndex ``(date, strike, right)``.
            strategy_name: Name of the strategy being backtested.

        Returns:
            A :class:`BacktestResult` with all trades, metrics, and
            the equity curve.
        """
        start_time = time.monotonic()
        self._reset()

        self._log.info(
            "backtest_started",
            strategy=strategy_name,
            price_rows=len(price_data),
            signal_rows=len(signals),
            options_rows=len(options_data),
        )

        dates = self._extract_sorted_dates(price_data)
        if len(dates) == 0:
            self._log.error("no_dates_in_price_data")
            return self._build_empty_result(
                strategy_name,
                start_time,
            )

        prev_equity = self._equity

        for bar_date in dates:
            # 1. Mark open positions to market
            self._mark_positions_to_market(
                bar_date,
                price_data,
                options_data,
            )

            # 2. Check exits on open positions
            self._check_exits(bar_date, price_data, options_data)

            # 3. Evaluate new entry signals
            if bar_date in signals.index:
                self._evaluate_signals(
                    bar_date,
                    signals,
                    price_data,
                    options_data,
                    strategy_name,
                )

            # 4. Record daily equity and return
            total_unrealized = sum(p.unrealized_pnl for p in self._open_positions)
            daily_equity = self._equity + total_unrealized
            self._equity_curve.append(
                (bar_date.isoformat(), daily_equity),
            )

            if prev_equity > 0:
                daily_ret = (daily_equity - prev_equity) / prev_equity
                self._daily_returns.append(daily_ret)

            if daily_equity > self._high_water_mark:
                self._high_water_mark = daily_equity
            prev_equity = daily_equity

            # 5. Track peak concurrent positions
            n_open = len(self._open_positions)
            if n_open > self._peak_concurrent:
                self._peak_concurrent = n_open

        # Close any remaining open positions at end of data
        self._close_remaining_positions(dates[-1] if dates else None)

        elapsed = time.monotonic() - start_time

        trades = self._build_trade_records()
        metrics = self._compute_metrics(trades)

        result = BacktestResult(
            strategy_name=strategy_name,
            config=self._config,
            trades=trades,
            metrics=metrics,
            start_date=dates[0].isoformat() if dates else "",
            end_date=dates[-1].isoformat() if dates else "",
            elapsed_seconds=round(elapsed, 3),
            n_signals_received=self._signals_received,
            n_signals_rejected=self._signals_rejected,
        )

        self._log.info(
            "backtest_complete",
            strategy=strategy_name,
            total_trades=metrics.total_trades,
            total_pnl=round(metrics.total_pnl, 2),
            sharpe=round(metrics.sharpe_ratio, 4),
            win_rate=round(metrics.win_rate, 4),
            max_dd_pct=round(metrics.max_drawdown_pct, 4),
            elapsed_s=round(elapsed, 3),
        )

        return result

    def run_walk_forward(
        self,
        signals: pd.DataFrame,
        price_data: pd.DataFrame,
        options_data: pd.DataFrame,
        strategy_name: str,
        n_folds: int = 5,
        embargo_days: int = 5,
    ) -> WalkForwardResult:
        """Run walk-forward validation across multiple folds.

        Splits the data chronologically into *n_folds* train/test
        periods, runs a backtest on each test period, and aggregates
        results.  An embargo gap of *embargo_days* separates each
        training set from its corresponding test set to prevent
        information leakage.

        Args:
            signals: Signal DataFrame indexed by date.
            price_data: OHLCV DataFrame indexed by date.
            options_data: Options chain DataFrame indexed by date.
            strategy_name: Strategy being backtested.
            n_folds: Number of walk-forward folds.
            embargo_days: Calendar days to embargo between train
                and test sets.

        Returns:
            A :class:`WalkForwardResult` with per-fold results and
            aggregated metrics across all test periods.
        """
        wf_start = time.monotonic()

        self._log.info(
            "walk_forward_started",
            strategy=strategy_name,
            n_folds=n_folds,
            embargo_days=embargo_days,
        )

        all_dates = self._extract_sorted_dates(price_data)
        if len(all_dates) < (n_folds + 1) * 20:
            raise ValueError(
                f"Insufficient data for {n_folds} folds: "
                f"only {len(all_dates)} trading days available"
            )

        fold_size = len(all_dates) // (n_folds + 1)
        fold_results: list[WalkForwardFoldResult] = []
        all_fold_trades: list[BacktestTrade] = []
        all_fold_equity: list[tuple[str, float]] = []

        for fold_idx in range(n_folds):
            train_end_idx = fold_size * (fold_idx + 1)
            test_start_idx = train_end_idx + embargo_days
            test_end_idx = min(
                test_start_idx + fold_size,
                len(all_dates),
            )

            if test_start_idx >= len(all_dates):
                self._log.warning(
                    "skipping_fold_insufficient_data",
                    fold=fold_idx,
                )
                continue

            train_start_date = all_dates[0]
            train_end_date = all_dates[train_end_idx - 1]
            test_start_date = all_dates[test_start_idx]
            test_end_date = all_dates[test_end_idx - 1]

            self._log.info(
                "walk_forward_fold",
                fold=fold_idx,
                train_range=(
                    train_start_date.isoformat(),
                    train_end_date.isoformat(),
                ),
                test_range=(
                    test_start_date.isoformat(),
                    test_end_date.isoformat(),
                ),
            )

            # Slice data for this fold's test period
            test_mask_price = (price_data.index >= test_start_date) & (
                price_data.index <= test_end_date
            )
            test_mask_signals = (signals.index >= test_start_date) & (
                signals.index <= test_end_date
            )
            test_mask_options = (options_data.index >= test_start_date) & (
                options_data.index <= test_end_date
            )

            fold_price = price_data.loc[test_mask_price]
            fold_signals = signals.loc[test_mask_signals]
            fold_options = options_data.loc[test_mask_options]

            # Run backtest on the test slice
            fold_result = self.run(
                fold_signals,
                fold_price,
                fold_options,
                strategy_name,
            )

            fold_results.append(
                WalkForwardFoldResult(
                    fold_num=fold_idx,
                    train_start=train_start_date.isoformat(),
                    train_end=train_end_date.isoformat(),
                    test_start=test_start_date.isoformat(),
                    test_end=test_end_date.isoformat(),
                    n_train_days=train_end_idx,
                    n_test_days=test_end_idx - test_start_idx,
                    result=fold_result,
                ),
            )

            all_fold_trades.extend(fold_result.trades)
            all_fold_equity.extend(fold_result.metrics.equity_curve)

        # Aggregate metrics across all folds
        aggregate_metrics = self._aggregate_fold_metrics(
            fold_results,
            all_fold_trades,
            all_fold_equity,
        )

        wf_elapsed = time.monotonic() - wf_start

        wf_result = WalkForwardResult(
            n_folds=len(fold_results),
            embargo_days=embargo_days,
            fold_results=fold_results,
            aggregate_metrics=aggregate_metrics,
            total_elapsed_seconds=round(wf_elapsed, 3),
        )

        self._log.info(
            "walk_forward_complete",
            n_folds=len(fold_results),
            total_trades=aggregate_metrics.total_trades,
            total_pnl=round(aggregate_metrics.total_pnl, 2),
            sharpe=round(aggregate_metrics.sharpe_ratio, 4),
            elapsed_s=round(wf_elapsed, 3),
        )

        return wf_result

    def generate_report(self, result: BacktestResult) -> str:
        """Generate a formatted text summary of backtest results.

        Args:
            result: The backtest result to summarize.

        Returns:
            Multi-line formatted report string.
        """
        m = result.metrics
        lines: list[str] = [
            "=" * 68,
            f"  BACKTEST REPORT: {result.strategy_name}",
            "=" * 68,
            f"  Period:    {result.start_date} to {result.end_date}",
            f"  Capital:   ${result.config.initial_capital:,.2f}",
            f"  Slippage:  {result.config.slippage_pct:.0%} of bid-ask spread",
            f"  Commission: ${result.config.commission_per_contract}/contract/leg",
            "-" * 68,
            "  PERFORMANCE SUMMARY",
            "-" * 68,
            f"  Total P&L:          ${m.total_pnl:>12,.2f}"
            f"  ({m.total_return_pct:>+.2f}%)",
            f"  Total Trades:       {m.total_trades:>12d}",
            f"  Win Rate:           {m.win_rate:>12.2%}",
            f"  Profit Factor:      {m.profit_factor:>12.2f}",
            f"  Expectancy:         ${m.expectancy:>12,.2f}",
            "-" * 68,
            "  RISK METRICS",
            "-" * 68,
            f"  Sharpe Ratio:       {m.sharpe_ratio:>12.4f}",
            f"  Sortino Ratio:      {m.sortino_ratio:>12.4f}",
            f"  Calmar Ratio:       {m.calmar_ratio:>12.4f}",
            f"  Max Drawdown:       {m.max_drawdown_pct:>12.2%}"
            f"  (${m.max_drawdown_dollars:>,.2f})",
            f"  Max DD Duration:    {m.max_drawdown_duration_days:>12d} days",
            "-" * 68,
            "  TRADE STATISTICS",
            "-" * 68,
            f"  Avg Trade P&L:      ${m.avg_trade_pnl:>12,.2f}",
            f"  Avg Winner:         ${m.avg_winner:>12,.2f}",
            f"  Avg Loser:          ${m.avg_loser:>12,.2f}",
            f"  Best Trade:         ${m.best_trade:>12,.2f}",
            f"  Worst Trade:        ${m.worst_trade:>12,.2f}",
            f"  Avg Holding Days:   {m.avg_holding_days:>12.1f}",
            f"  Total Commissions:  ${m.total_commissions:>12,.2f}",
            f"  Peak Positions:     {m.max_concurrent_positions:>12d}",
            "-" * 68,
            "  SIGNAL STATS",
            "-" * 68,
            f"  Signals Received:   {result.n_signals_received:>12d}",
            f"  Signals Rejected:   {result.n_signals_rejected:>12d}",
            f"  Conversion Rate:    "
            f"{self._safe_div(m.total_trades, max(result.n_signals_received, 1)):>12.2%}",  # noqa: E501
            f"  Run Time:           {result.elapsed_seconds:>12.3f}s",
            "=" * 68,
        ]

        return "\n".join(lines)

    def to_dataframe(
        self,
        result: BacktestResult,
    ) -> pd.DataFrame:
        """Convert backtest trades to a pandas DataFrame.

        Args:
            result: The backtest result containing trades.

        Returns:
            DataFrame with one row per completed trade and all
            :class:`BacktestTrade` fields as columns.
        """
        if not result.trades:
            return pd.DataFrame()

        records = [trade.model_dump() for trade in result.trades]
        df = pd.DataFrame(records)

        # Convert date columns
        for col in ("entry_date", "exit_date"):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])

        return df

    # ------------------------------------------------------------------
    # Fill simulation
    # ------------------------------------------------------------------

    def _simulate_fill(
        self,
        bid: float,
        ask: float,
        side: str,
        strike: float,
        right: str,
        expiry: date,
        quantity: int = 1,
    ) -> FillSimulation:
        """Simulate an options fill with realistic slippage.

        For BUY orders, the fill price is mid + slippage (worse for
        buyer).  For SELL orders, the fill price is mid - slippage
        (worse for seller).  The fill is rejected if the bid-ask
        spread exceeds *max_spread_pct* of the mid price.

        Args:
            bid: Best bid price.
            ask: Best ask price.
            side: ``BUY`` or ``SELL``.
            strike: Strike price of the option.
            right: ``C`` or ``P``.
            expiry: Expiration date.
            quantity: Number of contracts.

        Returns:
            A :class:`FillSimulation` describing the execution.
        """
        spread = ask - bid
        mid = (bid + ask) / 2.0

        # Reject if mid is non-positive
        if mid <= 0:
            return FillSimulation(
                strike=strike,
                right=right,
                expiry=expiry,
                side=side,
                bid=bid,
                ask=ask,
                mid_price=mid,
                fill_price=0.0,
                slippage=0.0,
                spread_pct=0.0,
                rejected=True,
                reject_reason="Non-positive mid price",
                quantity=quantity,
                commission=0.0,
            )

        spread_pct = spread / mid

        # Liquidity filter: reject wide spreads
        if spread_pct > self._config.max_spread_pct:
            return FillSimulation(
                strike=strike,
                right=right,
                expiry=expiry,
                side=side,
                bid=bid,
                ask=ask,
                mid_price=round(mid, 4),
                fill_price=0.0,
                slippage=0.0,
                spread_pct=round(spread_pct, 6),
                rejected=True,
                reject_reason=(
                    f"Spread {spread_pct:.2%} exceeds "
                    f"max {self._config.max_spread_pct:.2%}"
                ),
                quantity=quantity,
                commission=0.0,
            )

        slippage_amount = spread * self._config.slippage_pct

        if side == FillSide.BUY:
            # Buyer pays more than mid
            fill_price = mid + slippage_amount
        else:
            # Seller receives less than mid
            fill_price = mid - slippage_amount

        # Ensure fill price does not cross the market
        fill_price = max(fill_price, 0.01)

        commission = quantity * self._config.commission_per_contract

        return FillSimulation(
            strike=strike,
            right=right,
            expiry=expiry,
            side=side,
            bid=bid,
            ask=ask,
            mid_price=round(mid, 4),
            fill_price=round(fill_price, 4),
            slippage=round(slippage_amount, 4),
            spread_pct=round(spread_pct, 6),
            rejected=False,
            reject_reason="",
            quantity=quantity,
            commission=round(commission, 2),
        )

    def _execute_spread(
        self,
        legs_data: list[dict[str, Any]],
        direction: str,
        quantity: int = 1,
    ) -> tuple[list[FillSimulation], float, float]:
        """Simulate execution of a multi-leg spread order.

        Each leg is filled independently with its own slippage
        calculation.  If any leg is rejected, the entire spread is
        rejected.

        Args:
            legs_data: List of dicts describing each leg.  Required
                keys: ``bid``, ``ask``, ``strike``, ``right``,
                ``expiry``, ``side``.  Optional: ``delta``, ``gamma``,
                ``theta``, ``vega``, ``quantity``.
            direction: ``LONG`` (debit) or ``SHORT`` (credit) for
                the overall spread.
            quantity: Number of spread units.

        Returns:
            Tuple of (fills, net_premium, total_commission) where
            *net_premium* is positive for debits and negative for
            credits, and *total_commission* is the sum of all leg
            commissions.
        """
        fills: list[FillSimulation] = []
        net_premium: float = 0.0
        total_commission: float = 0.0

        for leg in legs_data:
            leg_qty = leg.get("quantity", 1) * quantity
            fill = self._simulate_fill(
                bid=leg["bid"],
                ask=leg["ask"],
                side=leg["side"],
                strike=leg["strike"],
                right=leg["right"],
                expiry=self._parse_expiry(leg["expiry"]),
                quantity=leg_qty,
            )
            fills.append(fill)

            if fill.rejected:
                return fills, 0.0, 0.0

            # BUY legs cost money (positive premium outflow)
            # SELL legs generate credit (negative premium outflow)
            multiplier = self._config.contract_multiplier
            if fill.side == FillSide.BUY:
                net_premium += fill.fill_price * fill.quantity * multiplier
            else:
                net_premium -= fill.fill_price * fill.quantity * multiplier

            total_commission += fill.commission

        return (
            fills,
            round(net_premium, 2),
            round(total_commission, 2),
        )

    # ------------------------------------------------------------------
    # Position management
    # ------------------------------------------------------------------

    def _evaluate_signals(
        self,
        bar_date: date,
        signals: pd.DataFrame,
        price_data: pd.DataFrame,
        options_data: pd.DataFrame,
        strategy_name: str,
    ) -> None:
        """Evaluate entry signals for a given bar date.

        Extracts the signal row for *bar_date*, checks risk limits,
        attempts to fill the spread, and opens a new position if
        successful.

        Args:
            bar_date: Current bar date.
            signals: Signal DataFrame.
            price_data: Price DataFrame.
            options_data: Options chain DataFrame.
            strategy_name: Strategy being backtested.
        """
        signal_row = signals.loc[bar_date]

        # Handle case where multiple signals exist for same date
        if isinstance(signal_row, pd.DataFrame):
            for _, row in signal_row.iterrows():
                self._process_single_signal(
                    bar_date,
                    row,
                    price_data,
                    options_data,
                    strategy_name,
                )
        else:
            self._process_single_signal(
                bar_date,
                signal_row,
                price_data,
                options_data,
                strategy_name,
            )

    def _process_single_signal(
        self,
        bar_date: date,
        signal: pd.Series,
        price_data: pd.DataFrame,
        options_data: pd.DataFrame,
        strategy_name: str,
    ) -> None:
        """Process a single entry signal, checking risk limits.

        Args:
            bar_date: Current bar date.
            signal: Signal Series with trade parameters.
            price_data: Price DataFrame.
            options_data: Options chain DataFrame.
            strategy_name: Strategy being backtested.
        """
        self._signals_received += 1

        confidence = float(
            signal.get("confidence", 0.0)
            if hasattr(signal, "get")
            else getattr(signal, "confidence", 0.0)
        )

        # Skip low-confidence signals
        if confidence <= 0.0:
            self._signals_rejected += 1
            return

        # Check position limits
        if not self._check_position_limits(signal, strategy_name):
            self._signals_rejected += 1
            return

        # Check total risk budget
        if not self._check_risk_budget():
            self._signals_rejected += 1
            return

        # Extract legs from signal or options data
        legs_data = self._extract_legs_from_signal(
            bar_date,
            signal,
            options_data,
        )
        if not legs_data:
            self._signals_rejected += 1
            self._log.debug(
                "no_legs_extracted",
                date=bar_date.isoformat(),
            )
            return

        direction = str(
            signal.get("direction", "LONG")
            if hasattr(signal, "get")
            else getattr(signal, "direction", "LONG")
        )

        # Execute the spread
        fills, net_premium, total_commission = self._execute_spread(
            legs_data, direction
        )

        # Check if any leg was rejected
        if any(f.rejected for f in fills):
            self._signals_rejected += 1
            self._log.debug(
                "spread_fill_rejected",
                date=bar_date.isoformat(),
                reasons=[f.reject_reason for f in fills if f.rejected],
            )
            return

        # Build position from fills
        max_profit = float(
            signal.get("max_profit", abs(net_premium))
            if hasattr(signal, "get")
            else getattr(signal, "max_profit", abs(net_premium))
        )
        max_loss = float(
            signal.get("max_loss", abs(net_premium))
            if hasattr(signal, "get")
            else getattr(signal, "max_loss", abs(net_premium))
        )
        profit_target_pct = float(
            signal.get(
                "profit_target_pct",
                self._config.profit_target_pct,
            )
            if hasattr(signal, "get")
            else getattr(
                signal,
                "profit_target_pct",
                self._config.profit_target_pct,
            )
        )
        stop_loss_pct = float(
            signal.get(
                "stop_loss_pct",
                self._config.stop_loss_pct,
            )
            if hasattr(signal, "get")
            else getattr(
                signal,
                "stop_loss_pct",
                self._config.stop_loss_pct,
            )
        )

        # Ensure max_loss is positive and non-zero
        max_loss = max(abs(max_loss), 0.01)
        max_profit = max(abs(max_profit), 0.01)

        # Build leg records
        bt_legs: list[BacktestLeg] = []
        for fill, leg_info in zip(fills, legs_data, strict=False):
            bt_legs.append(
                BacktestLeg(
                    strike=fill.strike,
                    right=fill.right,
                    expiry=fill.expiry,
                    side=fill.side,
                    quantity=fill.quantity,
                    entry_price=fill.fill_price,
                    current_price=fill.fill_price,
                    delta=float(leg_info.get("delta", 0.0)),
                    gamma=float(leg_info.get("gamma", 0.0)),
                    theta=float(leg_info.get("theta", 0.0)),
                    vega=float(leg_info.get("vega", 0.0)),
                ),
            )

        ticker = str(
            signal.get("ticker", strategy_name)
            if hasattr(signal, "get")
            else getattr(signal, "ticker", strategy_name)
        )

        # Compute aggregate Greeks
        net_delta, net_gamma, net_theta, net_vega = self._compute_position_greeks(
            bt_legs
        )

        position = BacktestPosition(
            position_id=str(uuid.uuid4()),
            ticker=ticker,
            strategy=strategy_name,
            direction=direction,
            legs=bt_legs,
            entry_date=bar_date,
            entry_net_premium=net_premium,
            max_profit=max_profit,
            max_loss=max_loss,
            quantity=1,
            total_commission=total_commission,
            status=PositionStatus.OPEN,
            ml_confidence=confidence,
            profit_target_pct=profit_target_pct,
            stop_loss_pct=stop_loss_pct,
            net_delta=net_delta,
            net_gamma=net_gamma,
            net_theta=net_theta,
            net_vega=net_vega,
        )

        # Deduct commission from equity immediately
        self._equity -= total_commission

        self._open_positions.append(position)

        self._log.debug(
            "position_opened",
            position_id=position.position_id[:8],
            ticker=ticker,
            strategy=strategy_name,
            direction=direction,
            net_premium=net_premium,
            n_legs=len(bt_legs),
            confidence=round(confidence, 4),
        )

    def _mark_positions_to_market(
        self,
        bar_date: date,
        price_data: pd.DataFrame,
        options_data: pd.DataFrame,
    ) -> None:
        """Update unrealized P&L for all open positions.

        Uses the options data to mark each leg to its current mid
        price.  Falls back to a simple decay estimate when options
        data is unavailable for the bar date.

        Args:
            bar_date: Current bar date.
            price_data: Price DataFrame.
            options_data: Options chain DataFrame.
        """
        for position in self._open_positions:
            total_current_value: float = 0.0
            total_entry_value: float = 0.0
            mult = self._config.contract_multiplier

            for leg in position.legs:
                current_price = self._get_option_price(
                    bar_date,
                    leg.strike,
                    leg.right,
                    leg.expiry,
                    options_data,
                    price_data,
                    leg.entry_price,
                )
                leg.current_price = current_price

                if leg.side == FillSide.BUY:
                    total_current_value += current_price * leg.quantity * mult
                    total_entry_value += leg.entry_price * leg.quantity * mult
                else:
                    total_current_value -= current_price * leg.quantity * mult
                    total_entry_value -= leg.entry_price * leg.quantity * mult

            # Unrealized P&L = change in spread value
            # For debit positions: profit when current > entry
            # For credit positions: profit when |current| < |entry|
            position.unrealized_pnl = round(
                total_entry_value - total_current_value,
                2,
            )

    def _check_exits(
        self,
        bar_date: date,
        price_data: pd.DataFrame,
        options_data: pd.DataFrame,
    ) -> None:
        """Check all open positions for exit conditions.

        Evaluates profit target, stop loss, DTE limit, and
        assignment risk for each position.  Closed positions are
        moved from the open list to the closed list.

        Args:
            bar_date: Current bar date.
            price_data: Price DataFrame.
            options_data: Options chain DataFrame.
        """
        positions_to_close: list[tuple[int, str]] = []

        for idx, position in enumerate(self._open_positions):
            # Calculate P&L as fraction of max profit/loss
            pnl = position.unrealized_pnl
            max_profit = position.max_profit
            max_loss = position.max_loss

            # For profit target: check against max_profit
            profit_pct = pnl / max_profit if max_profit > 0 else 0.0

            # For stop loss: check against max_loss
            loss_pct = -pnl / max_loss if max_loss > 0 else 0.0

            # Minimum DTE across all legs
            min_dte = self._min_leg_dte(position, bar_date)

            # 1. Stop loss check
            if loss_pct >= position.stop_loss_pct:
                positions_to_close.append(
                    (idx, ExitReason.STOP_LOSS),
                )
                continue

            # 2. Profit target check
            if profit_pct >= position.profit_target_pct:
                positions_to_close.append(
                    (idx, ExitReason.PROFIT_TARGET),
                )
                continue

            # 3. DTE limit check
            if min_dte <= self._config.close_before_expiry_dte:
                positions_to_close.append(
                    (idx, ExitReason.DTE_LIMIT),
                )
                continue

            # 4. Assignment risk check
            if min_dte <= self._config.assignment_risk_dte:
                # Check if any short leg is ITM
                has_assignment_risk = False
                spot = self._get_spot_price(bar_date, price_data)
                if spot is not None:
                    for leg in position.legs:
                        if leg.side == FillSide.SELL and self._is_itm(
                            leg.strike,
                            leg.right,
                            spot,
                        ):
                            has_assignment_risk = True
                            break

                if has_assignment_risk:
                    positions_to_close.append(
                        (idx, ExitReason.ASSIGNMENT_RISK),
                    )
                    continue

        # Close positions in reverse index order to avoid shifting
        for idx, exit_reason in sorted(
            positions_to_close,
            key=lambda x: x[0],
            reverse=True,
        ):
            self._close_position(idx, bar_date, exit_reason)

    def _close_position(
        self,
        idx: int,
        bar_date: date,
        exit_reason: str,
    ) -> None:
        """Close a position and move it to the closed list.

        Calculates exit commission, final P&L, and updates equity.

        Args:
            idx: Index into _open_positions.
            bar_date: Date the position is closed.
            exit_reason: Why the position is being closed.
        """
        position = self._open_positions.pop(idx)

        # Calculate exit commission
        n_contracts = sum(leg.quantity for leg in position.legs)
        exit_commission = n_contracts * self._config.commission_per_contract
        position.total_commission += exit_commission

        # Calculate exit net premium from current prices
        exit_net_premium: float = 0.0
        mult = self._config.contract_multiplier
        for leg in position.legs:
            if leg.side == FillSide.BUY:
                exit_net_premium += leg.current_price * leg.quantity * mult
            else:
                exit_net_premium -= leg.current_price * leg.quantity * mult

        position.exit_net_premium = round(exit_net_premium, 2)
        position.exit_date = bar_date
        position.status = PositionStatus.CLOSED
        position.exit_reason = exit_reason

        # Realized P&L = unrealized P&L - exit commission
        realized_pnl = position.unrealized_pnl - exit_commission
        position.realized_pnl = round(realized_pnl, 2)

        # Update equity
        self._equity += realized_pnl
        self._equity -= exit_commission

        self._closed_positions.append(position)

        self._log.debug(
            "position_closed",
            position_id=position.position_id[:8],
            exit_reason=exit_reason,
            realized_pnl=round(realized_pnl, 2),
            holding_days=(bar_date - position.entry_date).days,
        )

    def _close_remaining_positions(
        self,
        final_date: date | None,
    ) -> None:
        """Close all remaining open positions at end of backtest.

        Args:
            final_date: Last date in the backtest data.
        """
        if final_date is None:
            return

        while self._open_positions:
            self._close_position(
                0,
                final_date,
                ExitReason.END_OF_DATA,
            )

    # ------------------------------------------------------------------
    # Risk checks
    # ------------------------------------------------------------------

    def _check_position_limits(
        self,
        signal: pd.Series,
        strategy_name: str,
    ) -> bool:
        """Check whether position limits allow a new trade.

        Args:
            signal: Signal Series (may contain ``ticker``).
            strategy_name: Strategy being backtested.

        Returns:
            ``True`` if limits permit a new position.
        """
        max_pos = self._config.max_concurrent_positions
        if len(self._open_positions) >= max_pos:
            self._log.debug(
                "position_limit_reached",
                current=len(self._open_positions),
                max=max_pos,
            )
            return False

        # Per-ticker limit
        ticker = str(
            signal.get("ticker", strategy_name)
            if hasattr(signal, "get")
            else getattr(signal, "ticker", strategy_name)
        )
        ticker_count = sum(1 for p in self._open_positions if p.ticker == ticker)
        if ticker_count >= self._config.max_positions_per_ticker:
            self._log.debug(
                "per_ticker_limit_reached",
                ticker=ticker,
                current=ticker_count,
                max=self._config.max_positions_per_ticker,
            )
            return False

        return True

    def _check_risk_budget(self) -> bool:
        """Check whether total risk budget allows a new trade.

        Returns:
            ``True`` if the total at-risk capital is below the limit.
        """
        total_risk = sum(p.max_loss for p in self._open_positions)
        max_allowed = self._equity * self._config.max_total_risk_pct

        if total_risk >= max_allowed:
            self._log.debug(
                "risk_budget_exceeded",
                total_risk=round(total_risk, 2),
                max_allowed=round(max_allowed, 2),
            )
            return False

        return True

    # ------------------------------------------------------------------
    # Data extraction helpers
    # ------------------------------------------------------------------

    def _extract_sorted_dates(
        self,
        price_data: pd.DataFrame,
    ) -> list[date]:
        """Extract and sort unique dates from the price data index.

        Args:
            price_data: DataFrame with a date-like index.

        Returns:
            Sorted list of unique dates.
        """
        if isinstance(price_data.index, pd.DatetimeIndex):
            dates = sorted(set(price_data.index.date))
        elif isinstance(price_data.index, pd.MultiIndex):
            level_0 = price_data.index.get_level_values(0)
            if hasattr(level_0, "date"):
                dates = sorted(set(level_0.date))
            else:
                dates = sorted(set(pd.to_datetime(level_0).date))
        else:
            try:
                dt_index = pd.to_datetime(price_data.index)
                dates = sorted(set(dt_index.date))
            except (ValueError, TypeError):
                self._log.error("cannot_parse_dates_from_index")
                return []

        return dates

    def _extract_legs_from_signal(
        self,
        bar_date: date,
        signal: pd.Series,
        options_data: pd.DataFrame,
    ) -> list[dict[str, Any]]:
        """Extract leg definitions from a signal and options data.

        If the signal contains ``legs`` as a list of dicts, those are
        used directly.  Otherwise, the method attempts to build a
        simple two-leg spread from the options data.

        Args:
            bar_date: Current bar date.
            signal: Signal Series.
            options_data: Options chain DataFrame.

        Returns:
            List of leg dicts suitable for ``_execute_spread``.
        """
        # Check if signal contains pre-built legs
        legs_raw = (
            signal.get("legs", None)
            if hasattr(signal, "get")
            else getattr(signal, "legs", None)
        )

        if legs_raw is not None and isinstance(legs_raw, list):
            return self._normalize_legs(legs_raw)

        # Try to build legs from options data for this date
        return self._build_legs_from_options(
            bar_date,
            signal,
            options_data,
        )

    def _build_legs_from_options(
        self,
        bar_date: date,
        signal: pd.Series,
        options_data: pd.DataFrame,
    ) -> list[dict[str, Any]]:
        """Build spread legs from options chain data.

        Attempts to find suitable options from the chain snapshot
        for *bar_date*.  Uses strike and expiry fields from the
        signal if available, otherwise selects contracts near ATM.

        Args:
            bar_date: Current bar date.
            signal: Signal Series.
            options_data: Options chain DataFrame.

        Returns:
            List of leg dicts, or empty list if construction fails.
        """
        # Filter options data for this date
        try:
            if isinstance(options_data.index, pd.DatetimeIndex):
                bar_dt = pd.Timestamp(bar_date)
                date_options = options_data.loc[options_data.index == bar_dt]
            elif isinstance(
                options_data.index,
                pd.MultiIndex,
            ):
                try:
                    date_options = options_data.loc[bar_date]
                except KeyError:
                    return []
            else:
                date_options = options_data[options_data.index == bar_date]
        except (KeyError, TypeError):
            return []

        if date_options.empty:
            return []

        # Extract required columns
        required_cols = {"strike", "right", "expiry", "bid", "ask"}
        available = set(date_options.columns)
        if not required_cols.issubset(available):
            missing = required_cols - available
            self._log.debug(
                "missing_options_columns",
                missing=list(missing),
            )
            return []

        legs: list[dict[str, Any]] = []
        for _, row in date_options.iterrows():
            leg: dict[str, Any] = {
                "strike": float(row["strike"]),
                "right": str(row["right"]),
                "expiry": row["expiry"],
                "bid": float(row["bid"]),
                "ask": float(row["ask"]),
                "side": str(row.get("side", FillSide.BUY)),
                "quantity": int(row.get("quantity", 1)),
                "delta": float(row.get("delta", 0.0)),
                "gamma": float(row.get("gamma", 0.0)),
                "theta": float(row.get("theta", 0.0)),
                "vega": float(row.get("vega", 0.0)),
            }
            legs.append(leg)

        return legs

    def _normalize_legs(
        self,
        legs_raw: list[Any],
    ) -> list[dict[str, Any]]:
        """Normalize leg data into a consistent dict format.

        Args:
            legs_raw: List of leg dicts or objects.

        Returns:
            List of normalized leg dicts.
        """
        normalized: list[dict[str, Any]] = []
        for leg in legs_raw:
            if isinstance(leg, dict):
                normalized.append(leg)
            elif hasattr(leg, "model_dump"):
                normalized.append(leg.model_dump())
            elif hasattr(leg, "__dict__"):
                normalized.append(vars(leg))
            else:
                self._log.warning(
                    "unrecognized_leg_format",
                    leg_type=type(leg).__name__,
                )
        return normalized

    def _get_option_price(
        self,
        bar_date: date,
        strike: float,
        right: str,
        expiry: date,
        options_data: pd.DataFrame,
        price_data: pd.DataFrame,
        fallback_price: float,
    ) -> float:
        """Get the current market price for a specific option.

        Looks up the option in the chain data for *bar_date*.
        Falls back to a time-decay adjusted estimate if the exact
        contract is not found.

        Args:
            bar_date: Current date.
            strike: Strike price to look up.
            right: ``C`` or ``P``.
            expiry: Expiration date.
            options_data: Options chain DataFrame.
            price_data: Price DataFrame (for decay fallback).
            fallback_price: Price to use if lookup fails.

        Returns:
            Estimated mid-price of the option.
        """
        try:
            if isinstance(options_data.index, pd.DatetimeIndex):
                bar_dt = pd.Timestamp(bar_date)
                date_opts = options_data.loc[options_data.index == bar_dt]
            elif isinstance(
                options_data.index,
                pd.MultiIndex,
            ):
                try:
                    date_opts = options_data.loc[bar_date]
                except KeyError:
                    return self._decay_estimate(
                        fallback_price,
                        bar_date,
                        expiry,
                    )
            else:
                date_opts = options_data[options_data.index == bar_date]

            if date_opts.empty:
                return self._decay_estimate(
                    fallback_price,
                    bar_date,
                    expiry,
                )

            # Filter for exact strike and right
            mask = (date_opts["strike"] == strike) & (date_opts["right"] == right)
            matched = date_opts.loc[mask]

            if matched.empty:
                return self._decay_estimate(
                    fallback_price,
                    bar_date,
                    expiry,
                )

            row = matched.iloc[0]
            bid = float(row.get("bid", 0.0))
            ask = float(row.get("ask", 0.0))
            mid = (bid + ask) / 2.0

            return round(mid, 4) if mid > 0 else fallback_price

        except (KeyError, TypeError, IndexError):
            return self._decay_estimate(
                fallback_price,
                bar_date,
                expiry,
            )

    def _decay_estimate(
        self,
        entry_price: float,
        current_date: date,
        expiry: date,
    ) -> float:
        """Estimate an option price using simple time decay.

        Applies a square-root time decay model as a rough
        approximation when market data is unavailable.

        Args:
            entry_price: Option price at entry.
            current_date: Current date.
            expiry: Expiration date.

        Returns:
            Estimated option price.
        """
        if current_date >= expiry:
            return 0.01

        total_dte = max((expiry - current_date).days, 1)
        # Rough square-root decay: price proportional to sqrt(DTE)
        # Normalize against an assumed 45-DTE entry
        assumed_entry_dte = 45
        decay_factor = math.sqrt(total_dte) / math.sqrt(
            assumed_entry_dte,
        )
        estimated = entry_price * min(decay_factor, 1.0)

        return round(max(estimated, 0.01), 4)

    def _get_spot_price(
        self,
        bar_date: date,
        price_data: pd.DataFrame,
    ) -> float | None:
        """Retrieve the spot price for a given date.

        Args:
            bar_date: Date to look up.
            price_data: OHLCV DataFrame.

        Returns:
            Close price for the date, or ``None``.
        """
        try:
            if isinstance(price_data.index, pd.DatetimeIndex):
                bar_dt = pd.Timestamp(bar_date)
                if bar_dt in price_data.index:
                    return float(price_data.loc[bar_dt, "close"])
            elif bar_date in price_data.index:
                row = price_data.loc[bar_date]
                if isinstance(row, pd.DataFrame):
                    return float(row.iloc[0]["close"])
                return float(row["close"])
        except (KeyError, TypeError, IndexError):
            pass
        return None

    def _parse_expiry(self, expiry_raw: Any) -> date:
        """Parse an expiry value into a date object.

        Args:
            expiry_raw: Expiry as a date, datetime, string, or
                pandas Timestamp.

        Returns:
            Expiration date.
        """
        if isinstance(expiry_raw, date) and not isinstance(
            expiry_raw,
            datetime,
        ):
            return expiry_raw
        if isinstance(expiry_raw, datetime):
            return expiry_raw.date()
        if isinstance(expiry_raw, pd.Timestamp):
            return expiry_raw.date()
        if isinstance(expiry_raw, str):
            if len(expiry_raw) == 8 and expiry_raw.isdigit():
                return datetime.strptime(
                    expiry_raw,
                    "%Y%m%d",
                ).date()
            return datetime.fromisoformat(expiry_raw).date()
        return date.today() + timedelta(days=45)

    # ------------------------------------------------------------------
    # Greeks and utility helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_position_greeks(
        legs: list[BacktestLeg],
    ) -> tuple[float, float, float, float]:
        """Compute aggregate Greeks for a position.

        Args:
            legs: List of position legs.

        Returns:
            Tuple of ``(net_delta, net_gamma, net_theta, net_vega)``.
        """
        net_delta: float = 0.0
        net_gamma: float = 0.0
        net_theta: float = 0.0
        net_vega: float = 0.0

        for leg in legs:
            sign = 1.0 if leg.side == FillSide.BUY else -1.0
            qty = leg.quantity
            net_delta += sign * leg.delta * qty
            net_gamma += sign * leg.gamma * qty
            net_theta += sign * leg.theta * qty
            net_vega += sign * leg.vega * qty

        return (
            round(net_delta, 4),
            round(net_gamma, 4),
            round(net_theta, 4),
            round(net_vega, 4),
        )

    @staticmethod
    def _min_leg_dte(
        position: BacktestPosition,
        bar_date: date,
    ) -> int:
        """Find the minimum DTE across all legs in a position.

        Args:
            position: The backtest position.
            bar_date: Current date.

        Returns:
            Minimum days to expiration (can be negative if past).
        """
        if not position.legs:
            return 0
        return min((leg.expiry - bar_date).days for leg in position.legs)

    @staticmethod
    def _is_itm(
        strike: float,
        right: str,
        spot: float,
    ) -> bool:
        """Check whether an option is in-the-money.

        Args:
            strike: Strike price.
            right: ``C`` or ``P``.
            spot: Current spot price.

        Returns:
            ``True`` if the option is ITM.
        """
        if right == "C":
            return spot > strike
        return spot < strike

    @staticmethod
    def _safe_div(
        numerator: float,
        denominator: float,
    ) -> float:
        """Safe division returning 0.0 on zero denominator.

        Args:
            numerator: Top value.
            denominator: Bottom value.

        Returns:
            Result of division, or 0.0.
        """
        if denominator == 0:
            return 0.0
        return numerator / denominator

    # ------------------------------------------------------------------
    # Metrics computation
    # ------------------------------------------------------------------

    def _build_trade_records(self) -> list[BacktestTrade]:
        """Convert closed positions into BacktestTrade records.

        Returns:
            List of completed trade records.
        """
        trades: list[BacktestTrade] = []

        for pos in self._closed_positions:
            holding_days = (pos.exit_date - pos.entry_date).days if pos.exit_date else 0
            pnl_pct = self._safe_div(
                pos.realized_pnl,
                pos.max_loss,
            )

            trades.append(
                BacktestTrade(
                    trade_id=pos.position_id,
                    ticker=pos.ticker,
                    strategy=pos.strategy,
                    direction=pos.direction,
                    entry_date=pos.entry_date,
                    exit_date=pos.exit_date or pos.entry_date,
                    entry_net_premium=pos.entry_net_premium,
                    exit_net_premium=pos.exit_net_premium,
                    max_profit=pos.max_profit,
                    max_loss=pos.max_loss,
                    quantity=pos.quantity,
                    realized_pnl=pos.realized_pnl,
                    commission=pos.total_commission,
                    pnl_pct=round(pnl_pct, 4),
                    holding_days=holding_days,
                    exit_reason=pos.exit_reason,
                    ml_confidence=pos.ml_confidence,
                    n_legs=len(pos.legs),
                ),
            )

        return trades

    def _compute_metrics(
        self,
        trades: list[BacktestTrade],
    ) -> BacktestMetrics:
        """Compute comprehensive backtest metrics from trade records.

        Args:
            trades: List of completed trade records.

        Returns:
            A :class:`BacktestMetrics` with all computed statistics.
        """
        initial_capital = self._config.initial_capital

        if not trades:
            return self._empty_metrics()

        pnls = np.array([t.realized_pnl for t in trades])
        total_pnl = float(np.sum(pnls))
        total_return_pct = (total_pnl / initial_capital) * 100.0

        winners = pnls[pnls > 0]
        losers = pnls[pnls < 0]
        n_winners = len(winners)
        n_losers = len(losers)
        n_total = len(trades)
        win_rate = n_winners / n_total if n_total > 0 else 0.0

        gross_profit = float(np.sum(winners)) if n_winners > 0 else 0.0
        gross_loss = float(np.abs(np.sum(losers))) if n_losers > 0 else 0.0
        profit_factor = self._safe_div(gross_profit, gross_loss)

        avg_pnl = float(np.mean(pnls))
        avg_winner = float(np.mean(winners)) if n_winners > 0 else 0.0
        avg_loser = float(np.mean(losers)) if n_losers > 0 else 0.0
        best_trade = float(np.max(pnls))
        worst_trade = float(np.min(pnls))

        total_commissions = sum(t.commission for t in trades)
        avg_holding = float(
            np.mean([t.holding_days for t in trades]),
        )

        # Expectancy: avg_win * win_rate - avg_loss * loss_rate
        loss_rate = 1.0 - win_rate
        expectancy = avg_winner * win_rate + avg_loser * loss_rate

        # Sharpe, Sortino, Calmar from daily returns
        sharpe = self._compute_sharpe(self._daily_returns)
        sortino = self._compute_sortino(self._daily_returns)

        # Drawdown analysis
        dd_pct, dd_dollars, dd_duration = self._compute_drawdown(self._equity_curve)

        # Calmar = annualized return / max drawdown
        n_days = len(self._equity_curve)
        if n_days > 1 and dd_pct > 0:
            ann_return = total_return_pct / 100.0 * TRADING_DAYS_PER_YEAR / n_days
            calmar = ann_return / dd_pct
        else:
            calmar = 0.0

        return BacktestMetrics(
            total_pnl=round(total_pnl, 2),
            total_return_pct=round(total_return_pct, 2),
            win_rate=round(win_rate, 4),
            profit_factor=round(profit_factor, 4),
            sharpe_ratio=round(sharpe, 4),
            sortino_ratio=round(sortino, 4),
            calmar_ratio=round(calmar, 4),
            max_drawdown_pct=round(dd_pct, 4),
            max_drawdown_dollars=round(dd_dollars, 2),
            max_drawdown_duration_days=dd_duration,
            avg_trade_pnl=round(avg_pnl, 2),
            avg_winner=round(avg_winner, 2),
            avg_loser=round(avg_loser, 2),
            best_trade=round(best_trade, 2),
            worst_trade=round(worst_trade, 2),
            total_trades=n_total,
            winning_trades=n_winners,
            losing_trades=n_losers,
            total_commissions=round(total_commissions, 2),
            expectancy=round(expectancy, 2),
            avg_holding_days=round(avg_holding, 1),
            max_concurrent_positions=self._peak_concurrent,
            equity_curve=list(self._equity_curve),
        )

    def _empty_metrics(self) -> BacktestMetrics:
        """Return zeroed-out metrics when no trades were executed.

        Returns:
            BacktestMetrics with all values at zero/defaults.
        """
        return BacktestMetrics(
            total_pnl=0.0,
            total_return_pct=0.0,
            win_rate=0.0,
            profit_factor=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            calmar_ratio=0.0,
            max_drawdown_pct=0.0,
            max_drawdown_dollars=0.0,
            max_drawdown_duration_days=0,
            avg_trade_pnl=0.0,
            avg_winner=0.0,
            avg_loser=0.0,
            best_trade=0.0,
            worst_trade=0.0,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            total_commissions=0.0,
            expectancy=0.0,
            avg_holding_days=0.0,
            max_concurrent_positions=0,
            equity_curve=list(self._equity_curve),
        )

    @staticmethod
    def _compute_sharpe(daily_returns: list[float]) -> float:
        """Compute the annualized Sharpe ratio from daily returns.

        Uses excess returns over the risk-free rate divided by
        standard deviation, annualized by sqrt(252).

        Args:
            daily_returns: List of daily portfolio returns.

        Returns:
            Annualized Sharpe ratio.
        """
        if len(daily_returns) < 2:
            return 0.0

        returns = np.array(daily_returns)
        daily_rf = RISK_FREE_RATE / TRADING_DAYS_PER_YEAR
        excess = returns - daily_rf

        mean_excess = float(np.mean(excess))
        std_excess = float(np.std(excess, ddof=1))

        if std_excess == 0:
            return 0.0

        return float(
            (mean_excess / std_excess)
            * np.sqrt(
                TRADING_DAYS_PER_YEAR,
            )
        )

    @staticmethod
    def _compute_sortino(daily_returns: list[float]) -> float:
        """Compute the annualized Sortino ratio from daily returns.

        Uses only downside deviation (negative returns) in the
        denominator.

        Args:
            daily_returns: List of daily portfolio returns.

        Returns:
            Annualized Sortino ratio.
        """
        if len(daily_returns) < 2:
            return 0.0

        returns = np.array(daily_returns)
        daily_rf = RISK_FREE_RATE / TRADING_DAYS_PER_YEAR
        excess = returns - daily_rf

        mean_excess = float(np.mean(excess))
        downside = excess[excess < 0]

        if len(downside) == 0:
            # No negative returns: infinite Sortino is capped
            return 99.99 if mean_excess > 0 else 0.0

        downside_std = float(np.std(downside, ddof=1))
        if downside_std == 0:
            return 0.0

        return float((mean_excess / downside_std) * np.sqrt(TRADING_DAYS_PER_YEAR))

    @staticmethod
    def _compute_drawdown(
        equity_curve: list[tuple[str, float]],
    ) -> tuple[float, float, int]:
        """Compute maximum drawdown from the equity curve.

        Args:
            equity_curve: List of ``(date_str, equity)`` tuples.

        Returns:
            Tuple of ``(max_dd_pct, max_dd_dollars, max_dd_days)``.
        """
        if len(equity_curve) < 2:
            return 0.0, 0.0, 0

        equities = np.array(
            [e for _, e in equity_curve],
            dtype=float,
        )

        peak = equities[0]
        max_dd_pct: float = 0.0
        max_dd_dollars: float = 0.0

        # For duration tracking
        peak_idx: int = 0
        max_dd_duration: int = 0
        current_dd_start: int = 0
        in_drawdown: bool = False

        for i, equity in enumerate(equities):
            if equity > peak:
                peak = equity
                peak_idx = i
                if in_drawdown:
                    duration = i - current_dd_start
                    if duration > max_dd_duration:
                        max_dd_duration = duration
                    in_drawdown = False
            else:
                dd_dollars = peak - equity
                dd_pct = dd_dollars / peak if peak > 0 else 0.0

                if not in_drawdown:
                    current_dd_start = peak_idx
                    in_drawdown = True

                if dd_pct > max_dd_pct:
                    max_dd_pct = dd_pct

                if dd_dollars > max_dd_dollars:
                    max_dd_dollars = dd_dollars

        # Check final drawdown duration
        if in_drawdown:
            duration = len(equities) - 1 - current_dd_start
            if duration > max_dd_duration:
                max_dd_duration = duration

        return max_dd_pct, max_dd_dollars, max_dd_duration

    # ------------------------------------------------------------------
    # Walk-forward aggregation
    # ------------------------------------------------------------------

    def _aggregate_fold_metrics(
        self,
        fold_results: list[WalkForwardFoldResult],
        all_trades: list[BacktestTrade],
        all_equity: list[tuple[str, float]],
    ) -> BacktestMetrics:
        """Aggregate metrics across all walk-forward folds.

        Args:
            fold_results: Per-fold result objects.
            all_trades: Concatenated list of all trades.
            all_equity: Concatenated equity curve.

        Returns:
            Aggregated :class:`BacktestMetrics`.
        """
        if not all_trades:
            return self._empty_metrics()

        pnls = np.array([t.realized_pnl for t in all_trades])
        total_pnl = float(np.sum(pnls))
        initial_capital = self._config.initial_capital
        total_return_pct = (total_pnl / initial_capital) * 100.0

        winners = pnls[pnls > 0]
        losers = pnls[pnls < 0]
        n_winners = len(winners)
        n_losers = len(losers)
        n_total = len(all_trades)
        win_rate = n_winners / n_total if n_total > 0 else 0.0

        gross_profit = float(np.sum(winners)) if n_winners > 0 else 0.0
        gross_loss = float(np.abs(np.sum(losers))) if n_losers > 0 else 0.0
        profit_factor = self._safe_div(gross_profit, gross_loss)

        avg_pnl = float(np.mean(pnls))
        avg_winner = float(np.mean(winners)) if n_winners > 0 else 0.0
        avg_loser = float(np.mean(losers)) if n_losers > 0 else 0.0
        best_trade = float(np.max(pnls))
        worst_trade = float(np.min(pnls))
        total_commissions = sum(t.commission for t in all_trades)
        avg_holding = float(
            np.mean([t.holding_days for t in all_trades]),
        )

        loss_rate = 1.0 - win_rate
        expectancy = avg_winner * win_rate + avg_loser * loss_rate

        # Reconstruct daily returns from equity curve
        daily_returns: list[float] = []
        for i in range(1, len(all_equity)):
            prev_eq = all_equity[i - 1][1]
            curr_eq = all_equity[i][1]
            if prev_eq > 0:
                daily_returns.append(
                    (curr_eq - prev_eq) / prev_eq,
                )

        sharpe = self._compute_sharpe(daily_returns)
        sortino = self._compute_sortino(daily_returns)

        dd_pct, dd_dollars, dd_duration = self._compute_drawdown(
            all_equity,
        )

        n_days = len(all_equity)
        if n_days > 1 and dd_pct > 0:
            ann_return = total_return_pct / 100.0 * TRADING_DAYS_PER_YEAR / n_days
            calmar = ann_return / dd_pct
        else:
            calmar = 0.0

        peak_concurrent = max(
            (fr.result.metrics.max_concurrent_positions for fr in fold_results),
            default=0,
        )

        return BacktestMetrics(
            total_pnl=round(total_pnl, 2),
            total_return_pct=round(total_return_pct, 2),
            win_rate=round(win_rate, 4),
            profit_factor=round(profit_factor, 4),
            sharpe_ratio=round(sharpe, 4),
            sortino_ratio=round(sortino, 4),
            calmar_ratio=round(calmar, 4),
            max_drawdown_pct=round(dd_pct, 4),
            max_drawdown_dollars=round(dd_dollars, 2),
            max_drawdown_duration_days=dd_duration,
            avg_trade_pnl=round(avg_pnl, 2),
            avg_winner=round(avg_winner, 2),
            avg_loser=round(avg_loser, 2),
            best_trade=round(best_trade, 2),
            worst_trade=round(worst_trade, 2),
            total_trades=n_total,
            winning_trades=n_winners,
            losing_trades=n_losers,
            total_commissions=round(total_commissions, 2),
            expectancy=round(expectancy, 2),
            avg_holding_days=round(avg_holding, 1),
            max_concurrent_positions=peak_concurrent,
            equity_curve=all_equity,
        )

    # ------------------------------------------------------------------
    # Internal state management
    # ------------------------------------------------------------------

    def _reset(self) -> None:
        """Reset all internal state for a fresh backtest run."""
        self._open_positions = []
        self._closed_positions = []
        self._equity = self._config.initial_capital
        self._high_water_mark = self._equity
        self._equity_curve = []
        self._daily_returns = []
        self._signals_received = 0
        self._signals_rejected = 0
        self._peak_concurrent = 0

    def _build_empty_result(
        self,
        strategy_name: str,
        start_time: float,
    ) -> BacktestResult:
        """Build an empty BacktestResult when no data is available.

        Args:
            strategy_name: Name of the strategy.
            start_time: Monotonic start timestamp.

        Returns:
            BacktestResult with empty trades and zero metrics.
        """
        elapsed = time.monotonic() - start_time
        return BacktestResult(
            strategy_name=strategy_name,
            config=self._config,
            trades=[],
            metrics=self._empty_metrics(),
            start_date="",
            end_date="",
            elapsed_seconds=round(elapsed, 3),
            n_signals_received=0,
            n_signals_rejected=0,
        )
