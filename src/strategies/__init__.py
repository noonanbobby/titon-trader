"""Options trading strategy subsystem for Project Titan.

Public API:

- :class:`BaseStrategy` -- abstract base class all strategies inherit from.
- :class:`StrategyConfig` -- Pydantic model for strategy YAML configuration.
- :class:`TradeSignal` -- signal emitted when entry criteria are met.
- :class:`ExitSignal` -- signal emitted when a position should be closed.
- :class:`LegSpec` -- specification for a single leg of a multi-leg trade.
- :class:`OptionData` -- typed snapshot of a single option contract.
- :class:`GreeksSnapshot` -- aggregate Greeks for an underlying or position.
- :class:`TradeRecord` -- lightweight open-trade representation.
- :class:`ScoredCandidate` -- scored strategy candidate from the selector.
- :class:`StrategySelector` -- regime-based strategy selection engine.
- :func:`load_strategies_from_config` -- load and instantiate strategies
  from ``strategies.yaml``.
"""

from src.strategies.base import (
    BaseStrategy,
    Direction,
    ExitReason,
    ExitSignal,
    ExitType,
    GreeksSnapshot,
    LegAction,
    LegSpec,
    OptionData,
    OptionRight,
    StrategyConfig,
    TradeRecord,
    TradeSignal,
)
from src.strategies.selector import (
    ScoredCandidate,
    StrategySelector,
    load_strategies_from_config,
)

__all__ = [
    "BaseStrategy",
    "Direction",
    "ExitReason",
    "ExitSignal",
    "ExitType",
    "GreeksSnapshot",
    "LegAction",
    "LegSpec",
    "OptionData",
    "OptionRight",
    "ScoredCandidate",
    "StrategyConfig",
    "StrategySelector",
    "TradeRecord",
    "TradeSignal",
    "load_strategies_from_config",
]
