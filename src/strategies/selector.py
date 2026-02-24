"""Regime-based strategy selection engine for Project Titan.

The :class:`StrategySelector` is the central decision point that determines
which options strategies are eligible and most attractive for a given
ticker at a given moment.  It combines regime alignment, IV-rank fit,
ML confidence, and historical performance into a single composite score,
then returns a ranked list of :class:`ScoredCandidate` objects for the
risk manager and AI agents to evaluate.

Usage::

    from src.strategies.selector import StrategySelector, load_strategies_config

    strategies = load_strategies_config("config/strategies.yaml")
    selector = StrategySelector(
        strategies=strategies, config_path="config/strategies.yaml"
    )

    candidates = await selector.select_strategies(
        ticker="AAPL",
        spot_price=185.0,
        iv_rank=42.0,
        regime="low_vol_trend",
        ml_confidence=0.82,
        greeks={"delta": 0.1, "gamma": 0.02, "theta": -0.5, "vega": 1.2},
        options_chain=[...],
        open_positions=[...],
    )
    for candidate in candidates:
        print(f"{candidate.strategy_name}: score={candidate.score:.3f}")
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml
from pydantic import BaseModel, Field

from src.strategies.base import (
    BaseStrategy,
    StrategyConfig,
    TradeSignal,
)
from src.utils.logging import get_logger

if TYPE_CHECKING:
    import structlog

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Mapping from strategy name (YAML key) to the module path where the
# concrete strategy class lives.  The class name is derived by converting
# the snake_case key to PascalCase (e.g. "bull_call_spread" -> module
# "src.strategies.bull_call_spread", class "BullCallSpread").
_STRATEGY_MODULE_PREFIX: str = "src.strategies"

# Scoring weights for the composite candidate score.
_WEIGHT_IV_FIT: float = 0.30
_WEIGHT_ML_CONFIDENCE: float = 0.35
_WEIGHT_REGIME_ALIGNMENT: float = 0.20
_WEIGHT_HISTORICAL_WIN_RATE: float = 0.15

# Default historical win rate when no track record is available.  Using a
# neutral 50 % ensures new strategies are not penalised.
_DEFAULT_WIN_RATE: float = 0.50

# Regime alignment bonuses: strategies that thrive in the current regime
# receive a boost.  These bonuses are additive multipliers applied to the
# regime component of the score.
_REGIME_BONUS: dict[str, dict[str, float]] = {
    # Strategy name -> {regime -> bonus multiplier}
    # A multiplier of 1.0 means neutral; > 1.0 is a bonus.
    "bull_call_spread": {
        "low_vol_trend": 1.2,
        "high_vol_trend": 1.0,
    },
    "bull_put_spread": {
        "low_vol_trend": 1.1,
        "range_bound": 1.2,
    },
    "iron_condor": {
        "low_vol_trend": 1.1,
        "range_bound": 1.3,
    },
    "short_strangle": {
        "low_vol_trend": 1.0,
        "range_bound": 1.3,
    },
    "calendar_spread": {
        "range_bound": 1.3,
        "low_vol_trend": 1.1,
    },
    "diagonal_spread": {
        "low_vol_trend": 1.2,
        "high_vol_trend": 1.1,
    },
    "broken_wing_butterfly": {
        "range_bound": 1.2,
        "low_vol_trend": 1.1,
    },
    "long_straddle": {
        "high_vol_trend": 1.2,
        "crisis": 1.3,
    },
    "pmcc": {
        "low_vol_trend": 1.3,
        "high_vol_trend": 1.0,
    },
    "ratio_spread": {
        "high_vol_trend": 1.2,
        "range_bound": 1.1,
    },
}

# IV Rank sweet-spot midpoints per strategy.  The IV fit score measures
# how close the current IV rank is to the strategy's ideal operating
# point.  These are derived from the midpoint of each strategy's
# [min_iv_rank, max_iv_rank] range but can be tuned independently.
_IV_SWEET_SPOT: dict[str, float] = {
    "bull_call_spread": 25.0,
    "bull_put_spread": 55.0,
    "iron_condor": 50.0,
    "short_strangle": 65.0,
    "calendar_spread": 30.0,
    "diagonal_spread": 35.0,
    "broken_wing_butterfly": 55.0,
    "long_straddle": 15.0,
    "pmcc": 25.0,
    "ratio_spread": 70.0,
}

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class ScoredCandidate(BaseModel):
    """A strategy candidate scored and ranked by the selector.

    Attributes:
        strategy_name: Canonical strategy name.
        ticker: Underlying symbol the candidate was evaluated for.
        score: Composite score in [0.0, 1.0].  Higher is better.
        signal: The :class:`TradeSignal` produced by the strategy's
            ``check_entry`` method, or ``None`` if scoring was done
            without a full entry check.
        reasoning: Human-readable explanation of the score components.
    """

    strategy_name: str
    ticker: str
    score: float = Field(..., ge=0.0, le=1.0)
    signal: TradeSignal | None = None
    reasoning: str = ""


# ---------------------------------------------------------------------------
# Strategy selector
# ---------------------------------------------------------------------------


class StrategySelector:
    """Regime-based strategy selection and scoring engine.

    Given market conditions and ML confidence for a ticker, the selector:

    1. Filters strategies that are eligible for the current regime and
       IV rank.
    2. Excludes strategies that have reached their per-strategy position
       limit.
    3. Scores each remaining strategy by a weighted composite of IV fit,
       ML confidence, regime alignment, and historical win rate.
    4. Invokes ``check_entry`` on eligible strategies to obtain concrete
       :class:`TradeSignal` objects.
    5. Returns a list of :class:`ScoredCandidate` sorted by descending
       score.

    Args:
        strategies: Mapping of strategy name to instantiated
            :class:`BaseStrategy` objects.
        config_path: Path to ``strategies.yaml`` for loading global
            defaults.
        historical_win_rates: Optional dict mapping strategy names to
            their observed win rates (0.0 -- 1.0).  Strategies without
            an entry default to ``_DEFAULT_WIN_RATE``.
    """

    def __init__(
        self,
        strategies: dict[str, BaseStrategy],
        config_path: str = "config/strategies.yaml",
        historical_win_rates: dict[str, float] | None = None,
    ) -> None:
        self._strategies: dict[str, BaseStrategy] = strategies
        self._config_path: str = config_path
        self._historical_win_rates: dict[str, float] = (
            historical_win_rates if historical_win_rates is not None else {}
        )
        self._log: structlog.stdlib.BoundLogger = get_logger("strategy.selector")

        # Load global defaults from YAML (min OI, bid-ask limits, etc.)
        self._global_defaults: dict[str, Any] = self._load_global_defaults()

        self._log.info(
            "selector_initialised",
            num_strategies=len(self._strategies),
            strategy_names=sorted(self._strategies.keys()),
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def select_strategies(
        self,
        ticker: str,
        spot_price: float,
        iv_rank: float,
        regime: str,
        ml_confidence: float,
        greeks: dict[str, float],
        options_chain: list[dict[str, Any]],
        open_positions: list[dict[str, Any]],
    ) -> list[ScoredCandidate]:
        """Select and rank eligible strategies for a ticker.

        This is the primary entry point for the selection pipeline.  It
        runs through filtering, scoring, and optional entry evaluation
        to produce a ranked list of candidates.

        Args:
            ticker: Underlying symbol (e.g. ``"AAPL"``).
            spot_price: Current price of the underlying.
            iv_rank: Current IV Rank (0--100).
            regime: Current market regime identifier (e.g.
                ``"low_vol_trend"``).
            ml_confidence: Ensemble ML confidence score (0.0 -- 1.0).
            greeks: Aggregate Greeks dict for the underlying.
            options_chain: Available options with Greeks and prices.
            open_positions: List of currently open position dicts.  Each
                dict must contain at minimum a ``"strategy"`` key.

        Returns:
            List of :class:`ScoredCandidate` sorted by descending score.
            May be empty if no strategies are eligible.
        """
        self._log.info(
            "selection_started",
            ticker=ticker,
            spot_price=spot_price,
            iv_rank=iv_rank,
            regime=regime,
            ml_confidence=round(ml_confidence, 4),
            open_positions_count=len(open_positions),
        )

        candidates: list[ScoredCandidate] = []

        for name, strategy in self._strategies.items():
            # 1. Eligibility: regime + IV rank
            if not strategy.is_eligible(regime, iv_rank):
                self._log.debug(
                    "strategy_filtered_eligibility",
                    strategy=name,
                    ticker=ticker,
                )
                continue

            # 2. Position limits
            if not self._check_position_limits(name, open_positions):
                self._log.debug(
                    "strategy_filtered_position_limit",
                    strategy=name,
                    ticker=ticker,
                )
                continue

            # 3. Score the candidate
            score = self._score_candidate(
                strategy=strategy,
                iv_rank=iv_rank,
                ml_confidence=ml_confidence,
                regime=regime,
            )

            # 4. Attempt entry check to get a concrete TradeSignal
            signal: TradeSignal | None = None
            try:
                signal = await strategy.check_entry(
                    ticker=ticker,
                    spot_price=spot_price,
                    iv_rank=iv_rank,
                    regime=regime,
                    greeks=greeks,
                    options_chain=options_chain,
                )
            except Exception as exc:
                self._log.warning(
                    "check_entry_failed",
                    strategy=name,
                    ticker=ticker,
                    error=str(exc),
                    error_type=type(exc).__name__,
                )
                continue

            if signal is None:
                self._log.debug(
                    "strategy_no_signal",
                    strategy=name,
                    ticker=ticker,
                )
                continue

            # Build reasoning string
            reasoning = self._build_reasoning(
                strategy_name=name,
                iv_rank=iv_rank,
                ml_confidence=ml_confidence,
                regime=regime,
                score=score,
            )

            candidates.append(
                ScoredCandidate(
                    strategy_name=name,
                    ticker=ticker,
                    score=score,
                    signal=signal,
                    reasoning=reasoning,
                )
            )

        # Sort by score descending
        candidates.sort(key=lambda c: c.score, reverse=True)

        self._log.info(
            "selection_complete",
            ticker=ticker,
            eligible_count=len(candidates),
            top_strategy=(candidates[0].strategy_name if candidates else None),
            top_score=(round(candidates[0].score, 4) if candidates else None),
        )

        return candidates

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _score_candidate(
        self,
        strategy: BaseStrategy,
        iv_rank: float,
        ml_confidence: float,
        regime: str,
    ) -> float:
        """Compute a composite score for a strategy candidate.

        The score is a weighted combination of four components, each
        normalised to ``[0.0, 1.0]``:

        1. **IV rank fit** (weight 0.30): How close the current IV rank
           is to the strategy's sweet spot.  Measured as
           ``1 - |iv_rank - sweet_spot| / 100``.
        2. **ML confidence** (weight 0.35): The raw ensemble confidence
           score, passed through directly.
        3. **Regime alignment** (weight 0.20): A base score of ``1.0``
           multiplied by any regime-specific bonus, then clamped to
           ``[0.0, 1.0]``.
        4. **Historical win rate** (weight 0.15): The observed win rate
           for this strategy, defaulting to ``_DEFAULT_WIN_RATE`` when
           no history is available.

        Args:
            strategy: The strategy being scored.
            iv_rank: Current IV Rank (0--100).
            ml_confidence: Ensemble confidence (0.0 -- 1.0).
            regime: Current market regime identifier.

        Returns:
            Composite score in ``[0.0, 1.0]``.
        """
        name = strategy.name

        # 1. IV rank fit
        sweet_spot = _IV_SWEET_SPOT.get(name, 50.0)
        iv_distance = abs(iv_rank - sweet_spot) / 100.0
        iv_fit_score = max(0.0, 1.0 - iv_distance)

        # 2. ML confidence (direct pass-through, already 0-1)
        ml_score = max(0.0, min(1.0, ml_confidence))

        # 3. Regime alignment
        regime_bonuses = _REGIME_BONUS.get(name, {})
        regime_multiplier = regime_bonuses.get(regime, 1.0)
        # Base regime score is 1.0 (strategy is eligible by definition),
        # boosted by the multiplier.  Normalise by dividing by the max
        # possible multiplier (1.3) so the result stays in [0, 1].
        regime_score = min(1.0, regime_multiplier / 1.3)

        # 4. Historical win rate
        win_rate = self._historical_win_rates.get(name, _DEFAULT_WIN_RATE)
        win_rate_score = max(0.0, min(1.0, win_rate))

        # Weighted composite
        composite = (
            _WEIGHT_IV_FIT * iv_fit_score
            + _WEIGHT_ML_CONFIDENCE * ml_score
            + _WEIGHT_REGIME_ALIGNMENT * regime_score
            + _WEIGHT_HISTORICAL_WIN_RATE * win_rate_score
        )

        # Clamp to [0, 1]
        composite = max(0.0, min(1.0, composite))

        self._log.debug(
            "candidate_scored",
            strategy=name,
            iv_fit=round(iv_fit_score, 4),
            ml_score=round(ml_score, 4),
            regime_score=round(regime_score, 4),
            win_rate_score=round(win_rate_score, 4),
            composite=round(composite, 4),
        )

        return round(composite, 4)

    # ------------------------------------------------------------------
    # Position limits
    # ------------------------------------------------------------------

    def _check_position_limits(
        self,
        strategy_name: str,
        open_positions: list[dict[str, Any]],
    ) -> bool:
        """Check if adding another position of this strategy exceeds limits.

        Counts how many open positions share the same strategy name and
        compares against the strategy's ``max_positions`` configuration.

        Args:
            strategy_name: Canonical strategy name to check.
            open_positions: Currently open positions (each dict must have
                a ``"strategy"`` key).

        Returns:
            ``True`` if a new position may be opened, ``False`` if the
            limit would be exceeded.
        """
        strategy = self._strategies.get(strategy_name)
        if strategy is None:
            self._log.warning(
                "unknown_strategy_in_limit_check",
                strategy_name=strategy_name,
            )
            return False

        max_allowed = strategy.config.max_positions
        current_count = sum(
            1 for pos in open_positions if pos.get("strategy") == strategy_name
        )

        if current_count >= max_allowed:
            self._log.debug(
                "position_limit_reached",
                strategy=strategy_name,
                current=current_count,
                max_allowed=max_allowed,
            )
            return False

        return True

    # ------------------------------------------------------------------
    # Reasoning builder
    # ------------------------------------------------------------------

    def _build_reasoning(
        self,
        strategy_name: str,
        iv_rank: float,
        ml_confidence: float,
        regime: str,
        score: float,
    ) -> str:
        """Construct a human-readable reasoning string for a candidate.

        Args:
            strategy_name: Strategy name.
            iv_rank: Current IV Rank.
            ml_confidence: ML confidence score.
            regime: Current regime.
            score: Composite score.

        Returns:
            Multi-line reasoning string.
        """
        sweet_spot = _IV_SWEET_SPOT.get(strategy_name, 50.0)
        iv_distance = abs(iv_rank - sweet_spot)
        win_rate = self._historical_win_rates.get(strategy_name, _DEFAULT_WIN_RATE)

        lines = [
            f"Strategy: {strategy_name}",
            f"Composite score: {score:.4f}",
            f"Regime: {regime} (eligible)",
            f"IV Rank: {iv_rank:.1f} (sweet spot: {sweet_spot:.1f}, "
            f"distance: {iv_distance:.1f})",
            f"ML confidence: {ml_confidence:.4f}",
            f"Historical win rate: {win_rate:.2%}",
        ]
        return " | ".join(lines)

    # ------------------------------------------------------------------
    # Configuration loading
    # ------------------------------------------------------------------

    def _load_global_defaults(self) -> dict[str, Any]:
        """Load the ``defaults`` section from strategies.yaml.

        Returns:
            Dict of global default values, or an empty dict if the file
            or section is missing.
        """
        config_file = Path(self._config_path)
        if not config_file.exists():
            self._log.warning(
                "strategies_yaml_not_found",
                path=str(config_file),
            )
            return {}

        try:
            with open(config_file) as f:
                raw = yaml.safe_load(f)
        except Exception as exc:
            self._log.error(
                "strategies_yaml_parse_error",
                path=str(config_file),
                error=str(exc),
            )
            return {}

        if not isinstance(raw, dict):
            return {}

        defaults = raw.get("defaults", {})
        self._log.debug(
            "global_defaults_loaded",
            defaults=defaults,
        )
        return defaults if isinstance(defaults, dict) else {}


# ---------------------------------------------------------------------------
# Strategy loading from YAML
# ---------------------------------------------------------------------------


def _snake_to_pascal(name: str) -> str:
    """Convert a snake_case name to PascalCase.

    Args:
        name: Snake-case string (e.g. ``"bull_call_spread"``).

    Returns:
        PascalCase string (e.g. ``"BullCallSpread"``).
    """
    return "".join(word.capitalize() for word in name.split("_"))


def load_strategies_from_config(
    config_path: str = "config/strategies.yaml",
) -> dict[str, BaseStrategy]:
    """Load and instantiate all enabled strategies from ``strategies.yaml``.

    For each strategy entry in the YAML file:

    1. Parse its configuration into a :class:`StrategyConfig`.
    2. Dynamically import the corresponding module from
       ``src.strategies.<name>`` (e.g. ``src.strategies.bull_call_spread``).
    3. Locate the PascalCase class within that module (e.g.
       ``BullCallSpread``).
    4. Instantiate the class with ``(name=<name>, config=<config>)``.

    Only strategies with ``enabled: true`` are instantiated.  Strategies
    whose module cannot be imported are logged as warnings and skipped.

    Args:
        config_path: Path to ``strategies.yaml``.

    Returns:
        Dict mapping strategy name to instantiated :class:`BaseStrategy`.

    Raises:
        FileNotFoundError: If *config_path* does not exist.
    """
    log = get_logger("strategy.loader")

    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Strategy configuration not found: {config_file}")

    with open(config_file) as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        log.error("strategies_yaml_invalid", path=str(config_file))
        return {}

    strategies_section = raw.get("strategies", {})
    if not isinstance(strategies_section, dict):
        log.error("strategies_section_invalid", path=str(config_file))
        return {}

    loaded: dict[str, BaseStrategy] = {}

    for name, params in strategies_section.items():
        if not isinstance(params, dict):
            log.warning(
                "strategy_config_invalid",
                strategy=name,
                reason="not a dict",
            )
            continue

        # Parse configuration
        try:
            config = StrategyConfig(**params)
        except Exception as exc:
            log.warning(
                "strategy_config_parse_error",
                strategy=name,
                error=str(exc),
            )
            continue

        if not config.enabled:
            log.info("strategy_disabled", strategy=name)
            continue

        # Dynamic import
        module_path = f"{_STRATEGY_MODULE_PREFIX}.{name}"
        class_name = _snake_to_pascal(name)

        try:
            module = importlib.import_module(module_path)
        except ModuleNotFoundError:
            log.warning(
                "strategy_module_not_found",
                strategy=name,
                module_path=module_path,
                class_name=class_name,
            )
            continue
        except Exception as exc:
            log.warning(
                "strategy_module_import_error",
                strategy=name,
                module_path=module_path,
                error=str(exc),
            )
            continue

        # Locate the class
        strategy_class = getattr(module, class_name, None)
        if strategy_class is None:
            log.warning(
                "strategy_class_not_found",
                strategy=name,
                module_path=module_path,
                class_name=class_name,
            )
            continue

        # Verify it is a subclass of BaseStrategy
        if not (
            isinstance(strategy_class, type)
            and issubclass(strategy_class, BaseStrategy)
        ):
            log.warning(
                "strategy_class_not_subclass",
                strategy=name,
                class_name=class_name,
                actual_type=type(strategy_class).__name__,
            )
            continue

        # Instantiate
        try:
            instance = strategy_class(name=name, config=config)
        except Exception as exc:
            log.warning(
                "strategy_instantiation_error",
                strategy=name,
                class_name=class_name,
                error=str(exc),
            )
            continue

        loaded[name] = instance
        log.info(
            "strategy_loaded",
            strategy=name,
            class_name=class_name,
            regimes=config.regimes,
            iv_range=f"[{config.min_iv_rank}, {config.max_iv_rank}]",
        )

    log.info(
        "strategies_loading_complete",
        total_loaded=len(loaded),
        strategy_names=sorted(loaded.keys()),
    )

    return loaded
