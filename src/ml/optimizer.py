"""Optuna walk-forward optimization engine for Project Titan.

Performs multi-objective hyperparameter optimization of strategy parameters
using Optuna with NSGA-II sampling.  Each trial runs a full walk-forward
backtest across purged k-fold splits with embargo periods, evaluating
candidate parameter sets on both Sharpe ratio and maximum drawdown.

Optimizable parameter categories:
    - Entry thresholds (ML confidence required per strategy)
    - DTE targets (days-to-expiration per strategy)
    - Delta targets (delta ranges for short and long legs)
    - Profit target percentages (when to close winners)
    - Stop loss levels (when to close losers)
    - Position sizing multipliers

Results are stored in PostgreSQL via Optuna's RDB storage, allowing
dashboard visualization through ``optuna-dashboard`` and persistence
across restarts.

Usage::

    from src.ml.optimizer import WalkForwardOptimizer

    optimizer = WalkForwardOptimizer(
        postgres_dsn="postgresql://titan:secret@localhost:5432/titan",
        n_trials=200,
        n_splits=5,
        study_name="titan_iron_condor_v1",
    )
    result = await optimizer.optimize(
        X=feature_matrix,
        y=target_series,
        trade_data=historical_trades_df,
        strategy="iron_condor",
    )
"""

from __future__ import annotations

import asyncio
import time
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import yaml
from pydantic import BaseModel, Field

from src.utils.logging import get_logger

if TYPE_CHECKING:
    import optuna
    import structlog


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ANNUALIZATION_FACTOR: float = np.sqrt(252)
"""Square root of trading days per year for Sharpe annualization."""

MIN_SAMPLES_PER_FOLD: int = 30
"""Minimum validation samples required per walk-forward fold."""

MIN_TRADES_FOR_SHARPE: int = 5
"""Minimum number of simulated trades to compute a Sharpe ratio."""

DEFAULT_N_TRIALS: int = 200
"""Default number of Optuna trials if not specified."""

DEFAULT_N_SPLITS: int = 5
"""Default number of walk-forward folds."""

DEFAULT_EMBARGO_DAYS: int = 5
"""Default embargo period between train and validation sets."""

MEDIAN_PRUNER_N_STARTUP: int = 10
"""Number of completed trials before pruning begins."""

MEDIAN_PRUNER_N_WARMUP: int = 2
"""Number of reported steps before pruning a trial."""

PENALTY_SHARPE: float = -10.0
"""Sharpe penalty for degenerate or failed trials."""

PENALTY_DRAWDOWN: float = 1.0
"""Max-drawdown penalty (100%) for degenerate or failed trials."""

RISK_FREE_RATE: float = 0.05
"""Annualized risk-free rate for excess-return Sharpe calculation."""

DAILY_RISK_FREE: float = RISK_FREE_RATE / 252.0
"""Daily risk-free rate derived from the annualized figure."""


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class StrategyType(StrEnum):
    """Supported options strategy types for parameter optimization."""

    BULL_CALL_SPREAD = "bull_call_spread"
    BULL_PUT_SPREAD = "bull_put_spread"
    IRON_CONDOR = "iron_condor"
    CALENDAR_SPREAD = "calendar_spread"
    DIAGONAL_SPREAD = "diagonal_spread"
    BROKEN_WING_BUTTERFLY = "broken_wing_butterfly"
    SHORT_STRANGLE = "short_strangle"
    PMCC = "pmcc"
    RATIO_SPREAD = "ratio_spread"
    LONG_STRADDLE = "long_straddle"


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class ParameterSpace(BaseModel):
    """Defines the search boundaries for a single optimizable parameter.

    Attributes:
        name: Human-readable parameter name.
        param_type: Optuna suggest type (float, int, categorical).
        low: Lower bound for numeric parameters.
        high: Upper bound for numeric parameters.
        step: Step size for discrete numeric parameters.
        choices: Valid values for categorical parameters.
        log: Whether to sample in log scale.
    """

    name: str = Field(description="Parameter name")
    param_type: str = Field(description="Optuna type: 'float', 'int', or 'categorical'")
    low: float | None = Field(
        default=None,
        description="Lower bound for numeric parameters",
    )
    high: float | None = Field(
        default=None,
        description="Upper bound for numeric parameters",
    )
    step: float | None = Field(
        default=None,
        description="Step size for discrete numeric parameters",
    )
    choices: list[Any] | None = Field(
        default=None,
        description="Valid values for categorical parameters",
    )
    log: bool = Field(
        default=False,
        description="Whether to sample in log scale",
    )


class TrialResult(BaseModel):
    """Metrics and parameters from a single Optuna trial.

    Attributes:
        trial_number: Optuna trial index.
        params: Dictionary of parameter names to suggested values.
        sharpe_ratio: Annualized Sharpe ratio across walk-forward folds.
        max_drawdown: Maximum drawdown percentage (0.0 to 1.0).
        avg_return: Average per-trade return across folds.
        win_rate: Fraction of profitable simulated trades.
        n_trades: Total number of simulated trades across folds.
        fold_sharpes: Per-fold Sharpe ratios.
        fold_drawdowns: Per-fold maximum drawdowns.
        duration_seconds: Wall-clock time for the trial.
        pruned: Whether the trial was pruned early.
    """

    trial_number: int = Field(description="Optuna trial index")
    params: dict[str, Any] = Field(
        description="Suggested parameter values",
    )
    sharpe_ratio: float = Field(
        description="Annualized Sharpe ratio across folds",
    )
    max_drawdown: float = Field(
        description="Maximum drawdown (0.0 to 1.0)",
    )
    avg_return: float = Field(
        description="Average per-trade return",
    )
    win_rate: float = Field(
        description="Fraction of profitable trades",
    )
    n_trades: int = Field(
        description="Total simulated trades across folds",
    )
    fold_sharpes: list[float] = Field(
        default_factory=list,
        description="Per-fold Sharpe ratios",
    )
    fold_drawdowns: list[float] = Field(
        default_factory=list,
        description="Per-fold maximum drawdowns",
    )
    duration_seconds: float = Field(
        description="Trial wall-clock time in seconds",
    )
    pruned: bool = Field(
        default=False,
        description="Whether the trial was pruned",
    )


class OptimizationConfig(BaseModel):
    """Configuration for a walk-forward optimization run.

    Attributes:
        study_name: Optuna study name (also used as DB key).
        strategy: Options strategy being optimized.
        n_trials: Number of Optuna trials to run.
        n_splits: Number of walk-forward folds.
        embargo_days: Embargo period between train and test sets.
        postgres_dsn: PostgreSQL connection string for study storage.
        parameter_spaces: Per-parameter search boundaries.
        confidence_threshold: Minimum ML confidence for trade entry.
        assumed_slippage_pct: Slippage as fraction of bid-ask spread.
    """

    study_name: str = Field(
        description="Optuna study name / DB key",
    )
    strategy: str = Field(
        description="Strategy type to optimize",
    )
    n_trials: int = Field(
        default=DEFAULT_N_TRIALS,
        ge=1,
        description="Number of Optuna trials",
    )
    n_splits: int = Field(
        default=DEFAULT_N_SPLITS,
        ge=2,
        description="Number of walk-forward folds",
    )
    embargo_days: int = Field(
        default=DEFAULT_EMBARGO_DAYS,
        ge=0,
        description="Embargo days between train and test",
    )
    postgres_dsn: str = Field(
        description="PostgreSQL DSN for Optuna storage",
    )
    parameter_spaces: list[ParameterSpace] = Field(
        default_factory=list,
        description="Custom parameter search spaces",
    )
    confidence_threshold: float = Field(
        default=0.78,
        ge=0.0,
        le=1.0,
        description="Minimum ML confidence to trigger a trade",
    )
    assumed_slippage_pct: float = Field(
        default=0.15,
        ge=0.0,
        le=1.0,
        description="Slippage as fraction of bid-ask spread",
    )


class OptimizationResult(BaseModel):
    """Aggregated results from a completed optimization run.

    Attributes:
        study_name: Optuna study name.
        strategy: Strategy that was optimized.
        best_params: Best parameter set from the Pareto front.
        best_sharpe: Sharpe ratio of the best trial.
        best_max_drawdown: Max drawdown of the best trial.
        n_trials_completed: Number of trials that ran to completion.
        n_trials_pruned: Number of trials pruned early.
        pareto_front: List of trial results on the Pareto front.
        all_trial_results: Full list of trial results.
        total_duration_seconds: Total optimization wall-clock time.
        completed_at: UTC timestamp of completion.
    """

    study_name: str = Field(description="Optuna study name")
    strategy: str = Field(description="Optimized strategy type")
    best_params: dict[str, Any] = Field(
        description="Best parameter set from Pareto front",
    )
    best_sharpe: float = Field(
        description="Sharpe ratio of the best trial",
    )
    best_max_drawdown: float = Field(
        description="Max drawdown of the best trial",
    )
    n_trials_completed: int = Field(
        description="Trials that completed without pruning",
    )
    n_trials_pruned: int = Field(
        description="Trials pruned early by the pruner",
    )
    pareto_front: list[TrialResult] = Field(
        default_factory=list,
        description="Pareto-optimal trial results",
    )
    all_trial_results: list[TrialResult] = Field(
        default_factory=list,
        description="All trial results",
    )
    total_duration_seconds: float = Field(
        description="Total wall-clock time in seconds",
    )
    completed_at: datetime = Field(
        description="UTC timestamp of optimization completion",
    )


# ---------------------------------------------------------------------------
# Strategy parameter space definitions
# ---------------------------------------------------------------------------

# Each strategy type has a default set of optimizable parameter ranges.
# These are used when no custom ParameterSpace list is provided.

_COMMON_PARAM_SPACES: list[dict[str, Any]] = [
    {
        "name": "confidence_threshold",
        "param_type": "float",
        "low": 0.60,
        "high": 0.95,
        "step": 0.01,
    },
    {
        "name": "profit_target_pct",
        "param_type": "float",
        "low": 0.20,
        "high": 0.80,
        "step": 0.05,
    },
    {
        "name": "stop_loss_pct",
        "param_type": "float",
        "low": 0.50,
        "high": 3.00,
        "step": 0.10,
    },
    {
        "name": "position_size_multiplier",
        "param_type": "float",
        "low": 0.25,
        "high": 2.00,
        "step": 0.05,
    },
]

_STRATEGY_PARAM_SPACES: dict[str, list[dict[str, Any]]] = {
    StrategyType.BULL_CALL_SPREAD: [
        *_COMMON_PARAM_SPACES,
        {
            "name": "target_dte",
            "param_type": "int",
            "low": 21,
            "high": 75,
            "step": 1,
        },
        {
            "name": "long_leg_delta",
            "param_type": "float",
            "low": 0.45,
            "high": 0.80,
            "step": 0.01,
        },
        {
            "name": "short_leg_delta",
            "param_type": "float",
            "low": 0.20,
            "high": 0.55,
            "step": 0.01,
        },
        {
            "name": "wing_width",
            "param_type": "int",
            "low": 2,
            "high": 15,
            "step": 1,
        },
    ],
    StrategyType.BULL_PUT_SPREAD: [
        *_COMMON_PARAM_SPACES,
        {
            "name": "target_dte",
            "param_type": "int",
            "low": 21,
            "high": 75,
            "step": 1,
        },
        {
            "name": "short_leg_delta",
            "param_type": "float",
            "low": -0.45,
            "high": -0.10,
            "step": 0.01,
        },
        {
            "name": "long_leg_delta",
            "param_type": "float",
            "low": -0.25,
            "high": -0.02,
            "step": 0.01,
        },
        {
            "name": "wing_width",
            "param_type": "int",
            "low": 2,
            "high": 15,
            "step": 1,
        },
    ],
    StrategyType.IRON_CONDOR: [
        *_COMMON_PARAM_SPACES,
        {
            "name": "target_dte",
            "param_type": "int",
            "low": 21,
            "high": 75,
            "step": 1,
        },
        {
            "name": "short_put_delta",
            "param_type": "float",
            "low": -0.30,
            "high": -0.05,
            "step": 0.01,
        },
        {
            "name": "short_call_delta",
            "param_type": "float",
            "low": 0.05,
            "high": 0.30,
            "step": 0.01,
        },
        {
            "name": "wing_width",
            "param_type": "int",
            "low": 2,
            "high": 15,
            "step": 1,
        },
    ],
    StrategyType.CALENDAR_SPREAD: [
        *_COMMON_PARAM_SPACES,
        {
            "name": "front_month_dte",
            "param_type": "int",
            "low": 14,
            "high": 55,
            "step": 1,
        },
        {
            "name": "back_month_dte",
            "param_type": "int",
            "low": 35,
            "high": 120,
            "step": 1,
        },
        {
            "name": "strike_delta",
            "param_type": "float",
            "low": 0.35,
            "high": 0.65,
            "step": 0.01,
        },
    ],
    StrategyType.DIAGONAL_SPREAD: [
        *_COMMON_PARAM_SPACES,
        {
            "name": "front_month_dte",
            "param_type": "int",
            "low": 14,
            "high": 55,
            "step": 1,
        },
        {
            "name": "back_month_dte",
            "param_type": "int",
            "low": 35,
            "high": 120,
            "step": 1,
        },
        {
            "name": "long_leg_delta",
            "param_type": "float",
            "low": 0.50,
            "high": 0.85,
            "step": 0.01,
        },
        {
            "name": "short_leg_delta",
            "param_type": "float",
            "low": 0.15,
            "high": 0.50,
            "step": 0.01,
        },
    ],
    StrategyType.BROKEN_WING_BUTTERFLY: [
        *_COMMON_PARAM_SPACES,
        {
            "name": "target_dte",
            "param_type": "int",
            "low": 21,
            "high": 75,
            "step": 1,
        },
        {
            "name": "body_delta",
            "param_type": "float",
            "low": -0.45,
            "high": -0.15,
            "step": 0.01,
        },
        {
            "name": "narrow_wing_width",
            "param_type": "int",
            "low": 2,
            "high": 10,
            "step": 1,
        },
        {
            "name": "wide_wing_width",
            "param_type": "int",
            "low": 5,
            "high": 20,
            "step": 1,
        },
    ],
    StrategyType.SHORT_STRANGLE: [
        *_COMMON_PARAM_SPACES,
        {
            "name": "target_dte",
            "param_type": "int",
            "low": 21,
            "high": 75,
            "step": 1,
        },
        {
            "name": "short_put_delta",
            "param_type": "float",
            "low": -0.30,
            "high": -0.08,
            "step": 0.01,
        },
        {
            "name": "short_call_delta",
            "param_type": "float",
            "low": 0.08,
            "high": 0.30,
            "step": 0.01,
        },
    ],
    StrategyType.PMCC: [
        *_COMMON_PARAM_SPACES,
        {
            "name": "leaps_dte",
            "param_type": "int",
            "low": 180,
            "high": 600,
            "step": 5,
        },
        {
            "name": "short_call_dte",
            "param_type": "int",
            "low": 14,
            "high": 55,
            "step": 1,
        },
        {
            "name": "leaps_delta",
            "param_type": "float",
            "low": 0.65,
            "high": 0.90,
            "step": 0.01,
        },
        {
            "name": "short_call_delta",
            "param_type": "float",
            "low": 0.15,
            "high": 0.45,
            "step": 0.01,
        },
        {
            "name": "roll_short_at_dte",
            "param_type": "int",
            "low": 3,
            "high": 14,
            "step": 1,
        },
    ],
    StrategyType.RATIO_SPREAD: [
        *_COMMON_PARAM_SPACES,
        {
            "name": "target_dte",
            "param_type": "int",
            "low": 21,
            "high": 75,
            "step": 1,
        },
        {
            "name": "long_leg_delta",
            "param_type": "float",
            "low": 0.45,
            "high": 0.75,
            "step": 0.01,
        },
        {
            "name": "short_leg_delta",
            "param_type": "float",
            "low": 0.15,
            "high": 0.45,
            "step": 0.01,
        },
        {
            "name": "ratio_short_to_long",
            "param_type": "int",
            "low": 2,
            "high": 3,
            "step": 1,
        },
    ],
    StrategyType.LONG_STRADDLE: [
        *_COMMON_PARAM_SPACES,
        {
            "name": "target_dte",
            "param_type": "int",
            "low": 14,
            "high": 60,
            "step": 1,
        },
        {
            "name": "call_delta",
            "param_type": "float",
            "low": 0.40,
            "high": 0.60,
            "step": 0.01,
        },
        {
            "name": "put_delta",
            "param_type": "float",
            "low": -0.60,
            "high": -0.40,
            "step": 0.01,
        },
    ],
}


# ---------------------------------------------------------------------------
# WalkForwardOptimizer
# ---------------------------------------------------------------------------


class WalkForwardOptimizer:
    """Optuna-based walk-forward hyperparameter optimizer.

    Runs multi-objective optimization (maximize Sharpe, minimize drawdown)
    over strategy parameters using NSGA-II sampling, purged k-fold
    walk-forward validation, and PostgreSQL-backed study persistence.

    Args:
        postgres_dsn: PostgreSQL connection string for Optuna RDB storage.
        n_trials: Number of optimization trials to run.
        n_splits: Number of walk-forward cross-validation folds.
        study_name: Unique name for the Optuna study.
        embargo_days: Number of trading days to embargo between the end
            of the training set and the start of the validation set.
    """

    def __init__(
        self,
        postgres_dsn: str,
        n_trials: int = DEFAULT_N_TRIALS,
        n_splits: int = DEFAULT_N_SPLITS,
        study_name: str = "titan_strategy_optimization",
        embargo_days: int = DEFAULT_EMBARGO_DAYS,
    ) -> None:
        self._postgres_dsn: str = postgres_dsn
        self._n_trials: int = n_trials
        self._n_splits: int = n_splits
        self._study_name: str = study_name
        self._embargo_days: int = embargo_days
        self._log: structlog.stdlib.BoundLogger = get_logger(
            "ml.optimizer",
        )
        self._trial_results: list[TrialResult] = []

        self._log.info(
            "walk_forward_optimizer_initialized",
            study_name=study_name,
            n_trials=n_trials,
            n_splits=n_splits,
            embargo_days=embargo_days,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def optimize(
        self,
        X: pd.DataFrame,  # noqa: N803
        y: pd.Series,
        trade_data: pd.DataFrame,
        strategy: str,
    ) -> OptimizationResult:
        """Run multi-objective walk-forward optimization.

        Creates an Optuna study with NSGA-II sampler and MedianPruner,
        runs the specified number of trials, and returns aggregated
        results including the Pareto front.

        Args:
            X: Feature matrix with DatetimeIndex or MultiIndex
                (ticker, timestamp).
            y: Binary target variable aligned with X.
            trade_data: Historical trade outcome DataFrame with columns
                ``return_pct``, ``entry_date``, ``exit_date``,
                ``ml_confidence``, and ``strategy``.
            strategy: Strategy type to optimize (must be a valid
                :class:`StrategyType` value).

        Returns:
            An :class:`OptimizationResult` with the best parameters,
            Pareto front, and all trial metrics.

        Raises:
            ValueError: If the strategy is unknown or data is
                insufficient for the requested number of folds.
        """
        import optuna

        self._validate_inputs(X, y, trade_data, strategy)

        self._log.info(
            "optimization_started",
            strategy=strategy,
            n_trials=self._n_trials,
            n_samples=len(X),
            n_trades=len(trade_data),
        )

        start_time = time.monotonic()
        self._trial_results = []

        # Create RDB storage backed by PostgreSQL
        storage = self._create_storage()

        # Build the Optuna study
        study = optuna.create_study(
            study_name=self._study_name,
            storage=storage,
            directions=["maximize", "minimize"],
            sampler=optuna.samplers.NSGAIISampler(
                seed=42,
                population_size=50,
            ),
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=MEDIAN_PRUNER_N_STARTUP,
                n_warmup_steps=MEDIAN_PRUNER_N_WARMUP,
            ),
            load_if_exists=True,
        )

        self._log.info(
            "optuna_study_created",
            study_name=self._study_name,
            directions=["maximize_sharpe", "minimize_drawdown"],
            sampler="NSGAIISampler",
            pruner="MedianPruner",
        )

        # Run the optimization in a thread pool to avoid blocking
        # the async event loop
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None,
            lambda: study.optimize(
                lambda trial: self._objective(trial, X, y, trade_data, strategy),
                n_trials=self._n_trials,
                show_progress_bar=False,
                gc_after_trial=True,
            ),
        )

        elapsed = time.monotonic() - start_time

        # Extract Pareto front
        pareto_trials = study.best_trials
        pareto_results = self._extract_pareto_results(
            pareto_trials,
        )

        # Select the single best trial: highest Sharpe among those
        # with drawdown below 20%, falling back to the trial with
        # best Sharpe on the full Pareto front
        best_params, best_sharpe, best_dd = self._select_best_trial(
            pareto_trials,
        )

        # Count completed vs pruned trials
        n_completed = len(
            [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        )
        n_pruned = len(
            [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
        )

        result = OptimizationResult(
            study_name=self._study_name,
            strategy=strategy,
            best_params=best_params,
            best_sharpe=round(best_sharpe, 4),
            best_max_drawdown=round(best_dd, 4),
            n_trials_completed=n_completed,
            n_trials_pruned=n_pruned,
            pareto_front=pareto_results,
            all_trial_results=self._trial_results,
            total_duration_seconds=round(elapsed, 2),
            completed_at=datetime.now(UTC),
        )

        self._log.info(
            "optimization_complete",
            study_name=self._study_name,
            strategy=strategy,
            best_sharpe=result.best_sharpe,
            best_max_drawdown=result.best_max_drawdown,
            n_completed=n_completed,
            n_pruned=n_pruned,
            duration_seconds=result.total_duration_seconds,
        )

        return result

    async def get_best_params(
        self,
        study_name: str | None = None,
    ) -> dict[str, Any]:
        """Retrieve the best parameters from a stored Optuna study.

        Loads the study from PostgreSQL and selects the single best
        trial from the Pareto front (highest Sharpe with drawdown
        below 20%).

        Args:
            study_name: Optuna study name.  Defaults to the instance's
                configured study name.

        Returns:
            Dictionary of parameter names to optimal values.

        Raises:
            ValueError: If the study has no completed trials.
        """
        import optuna

        name = study_name or self._study_name
        storage = self._create_storage()

        loop = asyncio.get_running_loop()
        study = await loop.run_in_executor(
            None,
            lambda: optuna.load_study(
                study_name=name,
                storage=storage,
            ),
        )

        if not study.best_trials:
            raise ValueError(f"Study '{name}' has no completed trials")

        best_params, best_sharpe, best_dd = self._select_best_trial(
            study.best_trials,
        )

        self._log.info(
            "best_params_retrieved",
            study_name=name,
            best_sharpe=round(best_sharpe, 4),
            best_max_drawdown=round(best_dd, 4),
            n_params=len(best_params),
        )

        return best_params

    async def export_params_to_yaml(
        self,
        params: dict[str, Any],
        output_path: str,
    ) -> str:
        """Export optimized parameters to a YAML file.

        Writes the parameter dictionary to a YAML file suitable for
        loading into the strategy configuration at startup.

        Args:
            params: Dictionary of optimized parameter values.
            output_path: Path to the output YAML file.

        Returns:
            Absolute path to the written file.
        """
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        # Wrap the params in a structure matching strategies.yaml
        export_data: dict[str, Any] = {
            "optimized_params": {
                "study_name": self._study_name,
                "exported_at": datetime.now(UTC).isoformat(),
                "parameters": {},
            },
        }

        # Convert numpy types to native Python types for YAML
        for key, value in params.items():
            export_data["optimized_params"]["parameters"][key] = self._to_native_type(
                value
            )

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None,
            lambda: self._write_yaml(out, export_data),
        )

        resolved = str(out.resolve())

        self._log.info(
            "params_exported_to_yaml",
            path=resolved,
            n_params=len(params),
        )

        return resolved

    # ------------------------------------------------------------------
    # Objective function
    # ------------------------------------------------------------------

    def _objective(
        self,
        trial: optuna.trial.Trial,
        X: pd.DataFrame,  # noqa: N803
        y: pd.Series,
        trade_data: pd.DataFrame,
        strategy: str,
    ) -> tuple[float, float]:
        """Single Optuna trial: suggest params and evaluate.

        Suggests hyperparameters via the Optuna trial, runs walk-forward
        evaluation, and returns the two objectives (Sharpe and drawdown).

        Args:
            trial: Active Optuna trial object.
            X: Feature matrix.
            y: Binary target variable.
            trade_data: Historical trade outcome data.
            strategy: Strategy type being optimized.

        Returns:
            Tuple of (sharpe_ratio, max_drawdown) as the two objectives.
        """
        trial_start = time.monotonic()

        # Suggest parameters for this trial
        params = self._apply_params_to_strategy(trial, strategy)

        self._log.debug(
            "trial_started",
            trial_number=trial.number,
            strategy=strategy,
            params=params,
        )

        try:
            # Run walk-forward evaluation
            fold_sharpes, fold_drawdowns, fold_returns, fold_trades = (
                self._walk_forward_evaluate(params, X, y, trade_data)
            )
        except Exception:
            self._log.warning(
                "trial_evaluation_failed",
                trial_number=trial.number,
                exc_info=True,
            )
            # Return penalty values for failed trials
            trial_result = TrialResult(
                trial_number=trial.number,
                params=params,
                sharpe_ratio=PENALTY_SHARPE,
                max_drawdown=PENALTY_DRAWDOWN,
                avg_return=0.0,
                win_rate=0.0,
                n_trades=0,
                fold_sharpes=[],
                fold_drawdowns=[],
                duration_seconds=round(time.monotonic() - trial_start, 2),
                pruned=False,
            )
            self._trial_results.append(trial_result)
            return PENALTY_SHARPE, PENALTY_DRAWDOWN

        # Aggregate results across folds
        if not fold_sharpes:
            self._log.warning(
                "trial_no_valid_folds",
                trial_number=trial.number,
            )
            return PENALTY_SHARPE, PENALTY_DRAWDOWN

        avg_sharpe = float(np.mean(fold_sharpes))
        worst_drawdown = float(np.max(fold_drawdowns))

        # Aggregate per-trade metrics
        all_returns = []
        for fr in fold_returns:
            all_returns.extend(fr)

        all_returns_arr = np.array(all_returns) if all_returns else np.array([0.0])
        avg_return = float(np.mean(all_returns_arr))
        win_rate = (
            float(np.mean(all_returns_arr > 0)) if len(all_returns_arr) > 0 else 0.0
        )
        n_trades = sum(fold_trades)

        # Report intermediate value for pruning (Sharpe at each fold)
        for fold_idx, fold_sharpe in enumerate(fold_sharpes):
            trial.report(fold_sharpe, fold_idx)
            if trial.should_prune():
                self._log.debug(
                    "trial_pruned",
                    trial_number=trial.number,
                    pruned_at_fold=fold_idx,
                    fold_sharpe=round(fold_sharpe, 4),
                )
                trial_result = TrialResult(
                    trial_number=trial.number,
                    params=params,
                    sharpe_ratio=avg_sharpe,
                    max_drawdown=worst_drawdown,
                    avg_return=avg_return,
                    win_rate=round(win_rate, 4),
                    n_trades=n_trades,
                    fold_sharpes=[round(s, 4) for s in fold_sharpes],
                    fold_drawdowns=[round(d, 4) for d in fold_drawdowns],
                    duration_seconds=round(time.monotonic() - trial_start, 2),
                    pruned=True,
                )
                self._trial_results.append(trial_result)

                import optuna

                raise optuna.TrialPruned()

        trial_elapsed = time.monotonic() - trial_start

        trial_result = TrialResult(
            trial_number=trial.number,
            params=params,
            sharpe_ratio=round(avg_sharpe, 4),
            max_drawdown=round(worst_drawdown, 4),
            avg_return=round(avg_return, 6),
            win_rate=round(win_rate, 4),
            n_trades=n_trades,
            fold_sharpes=[round(s, 4) for s in fold_sharpes],
            fold_drawdowns=[round(d, 4) for d in fold_drawdowns],
            duration_seconds=round(trial_elapsed, 2),
            pruned=False,
        )
        self._trial_results.append(trial_result)

        self._log.debug(
            "trial_complete",
            trial_number=trial.number,
            sharpe=round(avg_sharpe, 4),
            max_drawdown=round(worst_drawdown, 4),
            n_trades=n_trades,
            win_rate=round(win_rate, 4),
            duration_seconds=round(trial_elapsed, 2),
        )

        return avg_sharpe, worst_drawdown

    # ------------------------------------------------------------------
    # Walk-forward evaluation
    # ------------------------------------------------------------------

    def _walk_forward_evaluate(
        self,
        params: dict[str, Any],
        X: pd.DataFrame,  # noqa: N803
        y: pd.Series,
        trade_data: pd.DataFrame,
    ) -> tuple[
        list[float],
        list[float],
        list[list[float]],
        list[int],
    ]:
        """Evaluate parameters across walk-forward folds.

        Splits the data into chronological folds with embargo, simulates
        trades on each validation fold using the candidate parameters,
        and computes per-fold Sharpe ratios and drawdowns.

        Args:
            params: Candidate parameter set.
            X: Feature matrix.
            y: Binary target.
            trade_data: Historical trade data.

        Returns:
            Tuple of:
                - fold_sharpes: Per-fold Sharpe ratios.
                - fold_drawdowns: Per-fold max drawdowns.
                - fold_returns: Per-fold lists of per-trade returns.
                - fold_trade_counts: Per-fold trade counts.
        """
        splits = self._purged_kfold_split(X, y)

        fold_sharpes: list[float] = []
        fold_drawdowns: list[float] = []
        fold_returns: list[list[float]] = []
        fold_trade_counts: list[int] = []

        confidence_threshold = params.get("confidence_threshold", 0.78)
        profit_target_pct = params.get("profit_target_pct", 0.50)
        stop_loss_pct = params.get("stop_loss_pct", 2.00)
        position_size_mult = params.get("position_size_multiplier", 1.00)

        for fold_idx, (_train_idx, val_idx) in enumerate(splits):
            X_val = X.iloc[val_idx]  # noqa: N806
            y_val = y.iloc[val_idx]

            # Filter trade data to the validation window
            val_trades = self._filter_trades_to_window(trade_data, X_val)

            # Simulate trades using the candidate parameters
            trade_returns = self._simulate_fold_trades(
                X_val=X_val,
                y_val=y_val,
                val_trades=val_trades,
                confidence_threshold=confidence_threshold,
                profit_target_pct=profit_target_pct,
                stop_loss_pct=stop_loss_pct,
                position_size_multiplier=position_size_mult,
            )

            if len(trade_returns) < MIN_TRADES_FOR_SHARPE:
                self._log.debug(
                    "fold_insufficient_trades",
                    fold=fold_idx,
                    n_trades=len(trade_returns),
                    min_required=MIN_TRADES_FOR_SHARPE,
                )
                # Use penalty values for sparse folds
                fold_sharpes.append(PENALTY_SHARPE)
                fold_drawdowns.append(PENALTY_DRAWDOWN)
                fold_returns.append(trade_returns)
                fold_trade_counts.append(len(trade_returns))
                continue

            returns_arr = np.array(trade_returns)
            sharpe = self._calculate_sharpe(returns_arr)
            drawdown = self._calculate_max_drawdown(returns_arr)

            fold_sharpes.append(sharpe)
            fold_drawdowns.append(drawdown)
            fold_returns.append(trade_returns)
            fold_trade_counts.append(len(trade_returns))

            self._log.debug(
                "fold_evaluated",
                fold=fold_idx,
                sharpe=round(sharpe, 4),
                max_drawdown=round(drawdown, 4),
                n_trades=len(trade_returns),
            )

        return (
            fold_sharpes,
            fold_drawdowns,
            fold_returns,
            fold_trade_counts,
        )

    def _simulate_fold_trades(
        self,
        X_val: pd.DataFrame,  # noqa: N803
        y_val: pd.Series,
        val_trades: pd.DataFrame,
        confidence_threshold: float,
        profit_target_pct: float,
        stop_loss_pct: float,
        position_size_multiplier: float,
    ) -> list[float]:
        """Simulate trades on a validation fold using candidate params.

        If historical trade data is available for the fold window, uses
        actual trade outcomes adjusted by the candidate parameters.
        Otherwise, falls back to ML-signal-based simulation using the
        feature matrix and target.

        Args:
            X_val: Validation feature matrix.
            y_val: Validation target variable.
            val_trades: Historical trades in the validation window.
            confidence_threshold: Min ML confidence to enter.
            profit_target_pct: Fraction of max profit to target.
            stop_loss_pct: Fraction of max loss to stop out.
            position_size_multiplier: Scale factor for position sizing.

        Returns:
            List of per-trade return percentages.
        """
        returns: list[float] = []

        # Prefer actual trade data when available
        if (
            not val_trades.empty
            and "return_pct" in val_trades.columns
            and "ml_confidence" in val_trades.columns
        ):
            returns = self._simulate_from_trade_data(
                val_trades=val_trades,
                confidence_threshold=confidence_threshold,
                profit_target_pct=profit_target_pct,
                stop_loss_pct=stop_loss_pct,
                position_size_multiplier=position_size_multiplier,
            )
        else:
            # Fall back to signal-based simulation
            returns = self._simulate_from_signals(
                X_val=X_val,
                y_val=y_val,
                confidence_threshold=confidence_threshold,
                profit_target_pct=profit_target_pct,
                stop_loss_pct=stop_loss_pct,
                position_size_multiplier=position_size_multiplier,
            )

        return returns

    def _simulate_from_trade_data(
        self,
        val_trades: pd.DataFrame,
        confidence_threshold: float,
        profit_target_pct: float,
        stop_loss_pct: float,
        position_size_multiplier: float,
    ) -> list[float]:
        """Simulate using historical trade outcome data.

        Filters trades by the candidate confidence threshold, then
        applies profit target and stop loss adjustments to the
        historical returns.

        Args:
            val_trades: Historical trades with ``return_pct`` and
                ``ml_confidence`` columns.
            confidence_threshold: Min ML confidence for entry.
            profit_target_pct: Profit target as fraction of max profit.
            stop_loss_pct: Stop loss as fraction of max loss.
            position_size_multiplier: Position size scaling factor.

        Returns:
            List of adjusted per-trade returns.
        """
        returns: list[float] = []

        for _, trade in val_trades.iterrows():
            confidence = float(trade.get("ml_confidence", 0.0))
            raw_return = float(trade.get("return_pct", 0.0))

            # Only take trades meeting the confidence threshold
            if confidence < confidence_threshold:
                continue

            # Apply profit target cap
            max_profit_return = float(
                trade.get("max_profit_pct", abs(raw_return) * 2.0)
            )
            capped_profit = max_profit_return * profit_target_pct
            if raw_return > 0:
                adjusted_return = min(raw_return, capped_profit)
            else:
                # Apply stop loss floor
                max_loss_return = float(
                    trade.get(
                        "max_loss_pct",
                        abs(raw_return) * 1.5,
                    )
                )
                loss_floor = -abs(max_loss_return) * stop_loss_pct
                adjusted_return = max(raw_return, loss_floor)

            # Scale by position size multiplier
            adjusted_return *= position_size_multiplier

            # Apply slippage deduction
            slippage_cost = abs(adjusted_return) * 0.005
            if adjusted_return > 0:
                adjusted_return -= slippage_cost
            else:
                adjusted_return -= slippage_cost

            returns.append(adjusted_return)

        return returns

    def _simulate_from_signals(
        self,
        X_val: pd.DataFrame,  # noqa: N803
        y_val: pd.Series,
        confidence_threshold: float,
        profit_target_pct: float,
        stop_loss_pct: float,
        position_size_multiplier: float,
    ) -> list[float]:
        """Simulate trades from ML signals when trade data unavailable.

        Uses the ensemble confidence score from the feature matrix as
        a proxy for ML confidence.  Trades are entered when confidence
        exceeds the threshold, and the actual outcome (target variable)
        determines the return direction.

        Args:
            X_val: Validation feature matrix.
            y_val: Validation binary target.
            confidence_threshold: Min confidence for entry.
            profit_target_pct: Profit target fraction.
            stop_loss_pct: Stop loss fraction.
            position_size_multiplier: Size multiplier.

        Returns:
            List of simulated per-trade returns.
        """
        returns: list[float] = []

        # Look for an ensemble confidence column
        confidence_col = None
        for col_name in (
            "signal_ensemble_score",
            "ensemble_score",
            "signal_confidence",
            "confidence",
        ):
            if col_name in X_val.columns:
                confidence_col = col_name
                break

        valid_mask = y_val.notna()
        X_valid = X_val.loc[valid_mask]  # noqa: N806
        y_valid = y_val.loc[valid_mask]

        for idx in X_valid.index:
            # Determine confidence from the feature matrix
            if confidence_col is not None:
                confidence = float(X_valid.loc[idx, confidence_col])
            else:
                # Use a hash-based proxy when no confidence column
                row_values = X_valid.loc[idx].values
                confidence = float(
                    np.clip(
                        np.mean(np.abs(row_values)) % 1.0,
                        0.0,
                        1.0,
                    )
                )

            if confidence < confidence_threshold:
                continue

            actual_outcome = float(y_valid.loc[idx])

            # Generate a realistic return based on outcome
            # Base return: profitable trades earn a portion of
            # premium; losing trades incur a loss
            if actual_outcome == 1.0:
                # Winning trade: assume profit proportional to
                # confidence and profit target
                base_return = profit_target_pct * 0.10
                adjusted_return = base_return * position_size_multiplier
            else:
                # Losing trade: loss proportional to stop loss
                base_loss = stop_loss_pct * 0.05
                adjusted_return = -base_loss * position_size_multiplier

            # Apply slippage
            slippage = abs(adjusted_return) * 0.005
            adjusted_return -= slippage

            returns.append(adjusted_return)

        return returns

    # ------------------------------------------------------------------
    # Parameter suggestion
    # ------------------------------------------------------------------

    def _apply_params_to_strategy(
        self,
        trial: optuna.trial.Trial,
        strategy: str,
    ) -> dict[str, Any]:
        """Map Optuna trial suggestions to strategy parameters.

        Looks up the default parameter space for the strategy type
        and uses the Optuna trial to suggest values within the defined
        ranges.

        Args:
            trial: Active Optuna trial.
            strategy: Strategy type being optimized.

        Returns:
            Dictionary of parameter names to suggested values.
        """
        # Get parameter spaces for this strategy
        param_spaces = self._get_param_spaces(strategy)
        params: dict[str, Any] = {}

        for space in param_spaces:
            name = space["name"]
            ptype = space["param_type"]

            if ptype == "float":
                low = float(space["low"])
                high = float(space["high"])
                step_val = space.get("step")
                use_log = space.get("log", False)

                # Optuna does not allow step and log simultaneously
                if step_val is not None and not use_log:
                    params[name] = trial.suggest_float(
                        name,
                        low,
                        high,
                        step=float(step_val),
                    )
                elif use_log:
                    params[name] = trial.suggest_float(
                        name,
                        low,
                        high,
                        log=True,
                    )
                else:
                    params[name] = trial.suggest_float(
                        name,
                        low,
                        high,
                    )

            elif ptype == "int":
                low = int(space["low"])
                high = int(space["high"])
                step_val = space.get("step")

                if step_val is not None:
                    params[name] = trial.suggest_int(
                        name,
                        low,
                        high,
                        step=int(step_val),
                    )
                else:
                    params[name] = trial.suggest_int(
                        name,
                        low,
                        high,
                    )

            elif ptype == "categorical":
                choices = space.get("choices", [])
                if choices:
                    params[name] = trial.suggest_categorical(name, choices)

        # Enforce logical constraints between parameters
        params = self._enforce_param_constraints(params, strategy)

        return params

    def _enforce_param_constraints(
        self,
        params: dict[str, Any],
        strategy: str,
    ) -> dict[str, Any]:
        """Enforce logical constraints between related parameters.

        Ensures that parameter combinations are valid (e.g., long leg
        delta is always further from zero than short leg delta for
        bull call spreads, back month DTE > front month DTE for
        calendar spreads).

        Args:
            params: Suggested parameter values.
            strategy: Strategy type.

        Returns:
            Params with constraints applied.
        """
        result = dict(params)

        if strategy == StrategyType.BULL_CALL_SPREAD:
            # Long leg delta > short leg delta for bull call spread
            if ("long_leg_delta" in result and "short_leg_delta" in result) and result[
                "long_leg_delta"
            ] <= result["short_leg_delta"]:
                result["long_leg_delta"], result["short_leg_delta"] = (
                    result["short_leg_delta"] + 0.05,
                    result["long_leg_delta"],
                )

        elif strategy == StrategyType.BULL_PUT_SPREAD:
            # Short leg delta closer to ATM (more negative)
            if ("short_leg_delta" in result and "long_leg_delta" in result) and abs(
                result["short_leg_delta"]
            ) <= abs(result["long_leg_delta"]):
                result["short_leg_delta"], result["long_leg_delta"] = (
                    result["long_leg_delta"] - 0.05,
                    result["short_leg_delta"],
                )

        elif strategy in (
            StrategyType.CALENDAR_SPREAD,
            StrategyType.DIAGONAL_SPREAD,
        ):
            # Back month DTE must exceed front month DTE
            if ("front_month_dte" in result and "back_month_dte" in result) and result[
                "back_month_dte"
            ] <= result["front_month_dte"]:
                result["back_month_dte"] = result["front_month_dte"] + 15

        elif strategy == StrategyType.IRON_CONDOR:
            # Short put delta should be more negative than short call
            if "short_put_delta" in result and "short_call_delta" in result:
                if result["short_put_delta"] >= 0:
                    result["short_put_delta"] = -abs(result["short_put_delta"])
                if result["short_call_delta"] <= 0:
                    result["short_call_delta"] = abs(result["short_call_delta"])

        elif strategy == StrategyType.BROKEN_WING_BUTTERFLY:
            # Wide wing must be wider than narrow wing
            if ("narrow_wing_width" in result and "wide_wing_width" in result) and (
                result["wide_wing_width"] <= result["narrow_wing_width"]
            ):
                result["wide_wing_width"] = result["narrow_wing_width"] + 3

        elif strategy == StrategyType.PMCC:  # noqa: SIM102
            # LEAPS DTE must exceed short call DTE significantly
            if (
                "leaps_dte" in result
                and "short_call_dte" in result
                and result["leaps_dte"] < result["short_call_dte"] * 4
            ):
                result["leaps_dte"] = result["short_call_dte"] * 4

        return result

    def _get_param_spaces(
        self,
        strategy: str,
    ) -> list[dict[str, Any]]:
        """Get the parameter search space for a strategy type.

        Returns the strategy-specific parameter space definitions,
        falling back to the common parameter spaces if the strategy
        is not recognized.

        Args:
            strategy: Strategy type string.

        Returns:
            List of parameter space definitions.
        """
        return _STRATEGY_PARAM_SPACES.get(
            strategy,
            _COMMON_PARAM_SPACES,
        )

    # ------------------------------------------------------------------
    # Sharpe and drawdown calculations
    # ------------------------------------------------------------------

    @staticmethod
    def _calculate_sharpe(returns: np.ndarray) -> float:
        """Calculate annualized Sharpe ratio from trade returns.

        Uses excess returns over the daily risk-free rate and
        annualizes by the square root of 252 trading days.

        Args:
            returns: Array of per-trade return percentages.

        Returns:
            Annualized Sharpe ratio.  Returns the penalty value if
            there are insufficient trades or zero standard deviation.
        """
        if len(returns) < MIN_TRADES_FOR_SHARPE:
            return PENALTY_SHARPE

        excess_returns = returns - DAILY_RISK_FREE
        mean_excess = float(np.mean(excess_returns))
        std_returns = float(np.std(returns, ddof=1))

        if std_returns < 1e-10:
            return 0.0

        daily_sharpe = mean_excess / std_returns
        annualized = daily_sharpe * ANNUALIZATION_FACTOR

        return float(annualized)

    @staticmethod
    def _calculate_max_drawdown(returns: np.ndarray) -> float:
        """Calculate maximum drawdown from a sequence of trade returns.

        Computes the cumulative equity curve from the trade returns
        and measures the largest peak-to-trough decline as a fraction
        of the peak value.

        Args:
            returns: Array of per-trade return percentages.

        Returns:
            Maximum drawdown as a positive fraction (0.0 to 1.0).
            Returns 0.0 if there are no returns.
        """
        if len(returns) == 0:
            return 0.0

        # Build cumulative equity curve starting at 1.0
        equity = np.cumprod(1.0 + returns)

        # Running maximum
        running_max = np.maximum.accumulate(equity)

        # Drawdown at each point
        drawdowns = (running_max - equity) / np.where(running_max > 0, running_max, 1.0)

        max_dd = float(np.max(drawdowns))

        # Clip to [0, 1] range
        return float(np.clip(max_dd, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Walk-forward splitting
    # ------------------------------------------------------------------

    def _purged_kfold_split(
        self,
        X: pd.DataFrame,  # noqa: N803
        y: pd.Series,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Generate purged k-fold splits for time-series data.

        Produces non-overlapping, chronologically ordered folds where
        each validation set follows the corresponding training set.
        An embargo period is excluded between training and validation
        to prevent information leakage.

        Args:
            X: Feature matrix (used for length and index).
            y: Target variable (used for length).

        Returns:
            List of (train_indices, val_indices) tuples.

        Raises:
            ValueError: If no valid folds can be created.
        """
        n_samples = len(X)
        fold_size = n_samples // (self._n_splits + 1)

        splits: list[tuple[np.ndarray, np.ndarray]] = []

        for i in range(self._n_splits):
            train_end = fold_size * (i + 1)
            val_start = train_end + self._embargo_days
            val_end = min(val_start + fold_size, n_samples)

            if val_start >= n_samples:
                self._log.debug(
                    "skipping_fold_insufficient_data",
                    fold=i,
                    val_start=val_start,
                    n_samples=n_samples,
                )
                continue

            train_idx = np.arange(0, train_end)
            val_idx = np.arange(val_start, val_end)

            if len(val_idx) < MIN_SAMPLES_PER_FOLD:
                self._log.debug(
                    "fold_validation_too_small",
                    fold=i,
                    val_size=len(val_idx),
                    min_required=MIN_SAMPLES_PER_FOLD,
                )
                continue

            splits.append((train_idx, val_idx))

        if not splits:
            raise ValueError(
                f"Cannot create valid folds from {n_samples} "
                f"samples with {self._n_splits} splits and "
                f"{self._embargo_days}-day embargo"
            )

        return splits

    # ------------------------------------------------------------------
    # Trade filtering
    # ------------------------------------------------------------------

    def _filter_trades_to_window(
        self,
        trade_data: pd.DataFrame,
        X_val: pd.DataFrame,  # noqa: N803
    ) -> pd.DataFrame:
        """Filter trade data to match the validation fold window.

        Identifies the date range of the validation fold and returns
        only trades whose entry dates fall within that range.

        Args:
            trade_data: Full historical trade DataFrame.
            X_val: Validation feature matrix (used for date range).

        Returns:
            Filtered trade DataFrame for the validation window.
        """
        if trade_data.empty:
            return trade_data

        # Extract date range from validation data
        try:
            if isinstance(X_val.index, pd.MultiIndex):
                timestamps = X_val.index.get_level_values(-1)
            else:
                timestamps = X_val.index

            if isinstance(timestamps, pd.DatetimeIndex):
                start_date = timestamps.min()
                end_date = timestamps.max()
            else:
                dt_index = pd.to_datetime(timestamps)
                start_date = dt_index.min()
                end_date = dt_index.max()
        except (ValueError, TypeError):
            self._log.debug(
                "cannot_extract_dates_from_val_index",
            )
            return trade_data

        # Filter trades by entry_date
        if "entry_date" in trade_data.columns:
            date_col = pd.to_datetime(trade_data["entry_date"])
            mask = (date_col >= start_date) & (date_col <= end_date)
            return trade_data.loc[mask].copy()

        if "entry_time" in trade_data.columns:
            date_col = pd.to_datetime(trade_data["entry_time"])
            mask = (date_col >= start_date) & (date_col <= end_date)
            return trade_data.loc[mask].copy()

        return trade_data

    # ------------------------------------------------------------------
    # Storage and I/O
    # ------------------------------------------------------------------

    def _create_storage(self) -> str:
        """Create the Optuna RDB storage connection string.

        Formats the PostgreSQL DSN as an Optuna-compatible storage URL.

        Returns:
            Optuna RDB storage URL string.
        """
        return self._postgres_dsn

    # ------------------------------------------------------------------
    # Result extraction
    # ------------------------------------------------------------------

    def _extract_pareto_results(
        self,
        pareto_trials: list[optuna.trial.FrozenTrial],
    ) -> list[TrialResult]:
        """Convert Optuna Pareto front trials to TrialResult models.

        Args:
            pareto_trials: Optuna frozen trial objects from the
                Pareto front.

        Returns:
            List of TrialResult instances.
        """
        results: list[TrialResult] = []

        for trial in pareto_trials:
            # Find matching internal result if available
            matching = [
                r for r in self._trial_results if r.trial_number == trial.number
            ]

            if matching:
                results.append(matching[0])
            else:
                # Reconstruct from Optuna trial
                sharpe = trial.values[0] if trial.values else 0.0
                drawdown = trial.values[1] if len(trial.values) > 1 else 0.0
                results.append(
                    TrialResult(
                        trial_number=trial.number,
                        params=dict(trial.params),
                        sharpe_ratio=round(sharpe, 4),
                        max_drawdown=round(drawdown, 4),
                        avg_return=0.0,
                        win_rate=0.0,
                        n_trades=0,
                        duration_seconds=round(
                            trial.duration.total_seconds() if trial.duration else 0.0,
                            2,
                        ),
                    )
                )

        return results

    def _select_best_trial(
        self,
        pareto_trials: list[optuna.trial.FrozenTrial],
    ) -> tuple[dict[str, Any], float, float]:
        """Select the single best trial from the Pareto front.

        Prefers trials with Sharpe above zero and drawdown below 20%.
        Among qualifying trials, selects the one with the highest
        Sharpe ratio.  Falls back to the trial with the best Sharpe
        if no trials meet the drawdown constraint.

        Args:
            pareto_trials: Optuna frozen trials on the Pareto front.

        Returns:
            Tuple of (best_params, best_sharpe, best_drawdown).
        """
        if not pareto_trials:
            self._log.warning("no_pareto_trials_found")
            return {}, PENALTY_SHARPE, PENALTY_DRAWDOWN

        # Prefer trials with drawdown below 20%
        max_acceptable_drawdown: float = 0.20
        qualified = [
            t
            for t in pareto_trials
            if len(t.values) >= 2
            and t.values[1] <= max_acceptable_drawdown
            and t.values[0] > 0
        ]

        if qualified:
            # Highest Sharpe among qualified
            best = max(qualified, key=lambda t: t.values[0])
        else:
            # Fall back to highest Sharpe on full front
            best = max(pareto_trials, key=lambda t: t.values[0])

        best_params = dict(best.params)
        best_sharpe = best.values[0] if best.values else 0.0
        best_dd = best.values[1] if len(best.values) > 1 else 0.0

        return best_params, best_sharpe, best_dd

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_inputs(
        self,
        X: pd.DataFrame,  # noqa: N803
        y: pd.Series,
        trade_data: pd.DataFrame,
        strategy: str,
    ) -> None:
        """Validate optimization inputs.

        Args:
            X: Feature matrix.
            y: Target variable.
            trade_data: Historical trade data.
            strategy: Strategy type string.

        Raises:
            ValueError: If any inputs are invalid.
        """
        if len(X) != len(y):
            raise ValueError(f"X and y must have the same length: {len(X)} != {len(y)}")

        min_samples = (
            self._n_splits + 1
        ) * MIN_SAMPLES_PER_FOLD + self._n_splits * self._embargo_days
        if len(X) < min_samples:
            raise ValueError(
                f"Insufficient data for {self._n_splits} folds: "
                f"need at least {min_samples} samples, "
                f"got {len(X)}"
            )

        valid_strategies = {st.value for st in StrategyType}
        if strategy not in valid_strategies:
            raise ValueError(
                f"Unknown strategy '{strategy}'. Supported: {sorted(valid_strategies)}"
            )

        if not isinstance(trade_data, pd.DataFrame):
            raise ValueError("trade_data must be a pandas DataFrame")

    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------

    @staticmethod
    def _to_native_type(value: Any) -> Any:
        """Convert numpy or other types to native Python types.

        Args:
            value: Value to convert.

        Returns:
            Native Python type suitable for YAML serialization.
        """
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (np.bool_,)):
            return bool(value)
        return value

    @staticmethod
    def _write_yaml(path: Path, data: dict[str, Any]) -> None:
        """Write data to a YAML file.

        Args:
            path: Output file path.
            data: Dictionary to serialize as YAML.
        """
        with open(path, "w") as f:
            yaml.dump(
                data,
                f,
                default_flow_style=False,
                sort_keys=False,
                allow_unicode=True,
            )


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------


def create_optimizer(
    postgres_dsn: str,
    strategy: str,
    n_trials: int = DEFAULT_N_TRIALS,
    n_splits: int = DEFAULT_N_SPLITS,
    embargo_days: int = DEFAULT_EMBARGO_DAYS,
) -> WalkForwardOptimizer:
    """Create a pre-configured WalkForwardOptimizer instance.

    Generates a descriptive study name from the strategy and timestamp,
    and initializes the optimizer with sensible defaults.

    Args:
        postgres_dsn: PostgreSQL connection string for Optuna storage.
        strategy: Strategy type to optimize.
        n_trials: Number of optimization trials.
        n_splits: Number of walk-forward folds.
        embargo_days: Embargo period between train and test sets.

    Returns:
        Configured :class:`WalkForwardOptimizer` instance.
    """
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    study_name = f"titan_{strategy}_{timestamp}"

    return WalkForwardOptimizer(
        postgres_dsn=postgres_dsn,
        n_trials=n_trials,
        n_splits=n_splits,
        study_name=study_name,
        embargo_days=embargo_days,
    )
