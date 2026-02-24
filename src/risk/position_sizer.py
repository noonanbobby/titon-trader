"""Kelly criterion position sizing with regime and circuit breaker adjustments.

Calculates optimal position sizes using a fractional Kelly criterion
approach.  The raw Kelly fraction is capped at quarter-Kelly for safety,
then further reduced based on the current market regime and circuit
breaker recovery stage.

Usage::

    from config.settings import get_settings
    from src.risk.position_sizer import PositionSizer

    settings = get_settings()
    sizer = PositionSizer(settings=settings, risk_config=risk_config)
    result = sizer.calculate_position_size(
        account_equity=150_000.0,
        max_loss_per_contract=500.0,
        win_probability=0.62,
        avg_win=300.0,
        avg_loss=250.0,
        regime="low_vol_trend",
    )
    print(result.contracts, result.dollar_risk)
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from src.utils.logging import get_logger

if TYPE_CHECKING:
    import structlog

    from config.settings import Settings

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Quarter-Kelly is used for safety: full Kelly is far too aggressive for
# real-world options trading where distribution tails are fat and
# correlations spike in crises.
_KELLY_FRACTION_MULTIPLIER: float = 0.25

# Absolute cap on the Kelly fraction before any other adjustments.
# Even a highly favorable edge should not risk more than 25% of equity
# in a single position.
_MAX_KELLY_FRACTION: float = 0.25

# Regime adjustment factors: multiply the base position size by these
# factors depending on the detected market regime.
_REGIME_FACTORS: dict[str, float] = {
    "low_vol_trend": 1.00,
    "range_bound": 0.75,
    "high_vol_trend": 0.50,
    "crisis": 0.25,
}

# Default regime factor when the regime label is unrecognized.
_DEFAULT_REGIME_FACTOR: float = 0.50

# Circuit breaker level size multipliers.  When trading is fully halted
# or in emergency, the multiplier is 0 — no new positions allowed.
_CB_LEVEL_MULTIPLIERS: dict[str, float] = {
    "NORMAL": 1.00,
    "CAUTION": 0.50,
    "WARNING": 0.25,
    "HALT": 0.00,
    "EMERGENCY": 0.00,
}

# Recovery stage size multipliers (indexed by stage number).
# Stage 0 = not in recovery (normal trading).
_RECOVERY_STAGE_MULTIPLIERS: dict[int, float] = {
    0: 1.00,
    1: 0.50,
    2: 0.75,
    3: 1.00,
}


# ---------------------------------------------------------------------------
# Pydantic model
# ---------------------------------------------------------------------------


class PositionSize(BaseModel):
    """Result of a position sizing calculation.

    Attributes:
        contracts: Number of contracts to trade (always >= 0).
        dollar_risk: Total dollar risk for the position.
        kelly_fraction: Raw Kelly fraction before adjustments.
        regime_factor: Multiplier applied for the current market regime.
        cb_factor: Multiplier applied for the circuit breaker level and
            recovery stage.
        risk_per_contract: Maximum loss per single contract.
    """

    contracts: int = Field(
        ge=0,
        description="Number of contracts to trade",
    )
    dollar_risk: float = Field(
        ge=0.0,
        description="Total dollar risk for the calculated position",
    )
    kelly_fraction: float = Field(
        ge=0.0,
        description="Raw Kelly fraction (before regime/CB adjustments)",
    )
    regime_factor: float = Field(
        ge=0.0,
        le=1.0,
        description="Regime-based size multiplier applied",
    )
    cb_factor: float = Field(
        ge=0.0,
        le=1.0,
        description="Circuit breaker / recovery stage multiplier applied",
    )
    risk_per_contract: float = Field(
        ge=0.0,
        description="Maximum loss per single contract",
    )


# ---------------------------------------------------------------------------
# PositionSizer
# ---------------------------------------------------------------------------


class PositionSizer:
    """Calculates risk-adjusted position sizes using fractional Kelly criterion.

    The sizing pipeline proceeds through these steps:

    1. Compute the raw Kelly fraction from win probability, average win,
       and average loss.
    2. Apply quarter-Kelly scaling for safety.
    3. Derive the dollar risk from account equity times the fractional Kelly.
    4. Cap at the per-trade ``max_risk_pct`` and ``max_risk_dollars`` limits.
    5. Apply a regime adjustment factor (crisis markets trade smaller).
    6. Apply a circuit breaker adjustment factor (drawdown states trade
       smaller or not at all).
    7. Convert the adjusted dollar risk to a contract count.
    8. Enforce minimum of 1 contract (if trading is allowed) and maximum
       by buying power.

    Args:
        settings: Application settings instance.
        risk_config: Parsed ``risk_limits.yaml`` as a dictionary.
    """

    def __init__(
        self,
        settings: Settings,
        risk_config: dict,
    ) -> None:
        self._settings: Settings = settings
        self._risk_config: dict = risk_config
        self._log: structlog.stdlib.BoundLogger = get_logger("risk.position_sizer")

        # Cache frequently accessed per-trade limits.
        per_trade: dict = self._risk_config.get("per_trade", {})
        self._max_risk_pct: float = float(
            per_trade.get("max_risk_pct", settings.trading.per_trade_risk_pct)
        )
        self._max_risk_dollars: float = float(per_trade.get("max_risk_dollars", 3000.0))

        # Load recovery stage multipliers from config if present.
        self._recovery_multipliers: dict[int, float] = dict(_RECOVERY_STAGE_MULTIPLIERS)
        recovery_cfg = self._risk_config.get("recovery", {})
        for stage_def in recovery_cfg.get("stages", []):
            stage_num = int(stage_def.get("stage", 0))
            size_pct = float(stage_def.get("size_pct", 1.0))
            self._recovery_multipliers[stage_num] = size_pct

        # Load circuit breaker level multipliers from config if present.
        self._cb_multipliers: dict[str, float] = dict(_CB_LEVEL_MULTIPLIERS)
        cb_cfg = self._risk_config.get("circuit_breakers", {})
        for level_def in cb_cfg.get("levels", []):
            name = str(level_def.get("name", ""))
            if "size_multiplier" in level_def:
                self._cb_multipliers[name] = float(level_def["size_multiplier"])

        self._log.info(
            "position_sizer_initialized",
            max_risk_pct=self._max_risk_pct,
            max_risk_dollars=self._max_risk_dollars,
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def calculate_position_size(
        self,
        account_equity: float,
        max_loss_per_contract: float,
        win_probability: float,
        avg_win: float,
        avg_loss: float,
        regime: str,
        circuit_breaker_level: str = "NORMAL",
        recovery_stage: int = 0,
    ) -> PositionSize:
        """Calculate the optimal number of contracts for a trade.

        Combines the Kelly criterion with regime and circuit breaker
        adjustments to arrive at a risk-appropriate position size.

        Args:
            account_equity: Current account net liquidation value in USD.
            max_loss_per_contract: Maximum possible loss on a single
                contract (e.g., the debit paid for a bull call spread,
                or the wing width minus credit for an iron condor).
            win_probability: Estimated probability of the trade being
                profitable (0.0 to 1.0).
            avg_win: Average dollar profit on winning trades.
            avg_loss: Average dollar loss on losing trades (positive number).
            regime: Current market regime label (e.g., ``"low_vol_trend"``).
            circuit_breaker_level: Current circuit breaker level name
                (e.g., ``"NORMAL"``, ``"CAUTION"``).
            recovery_stage: Current recovery ladder stage (0 = normal,
                1/2/3 = graduated recovery).

        Returns:
            A :class:`PositionSize` with the calculated contract count
            and supporting detail.
        """
        # Guard against invalid inputs.
        if account_equity <= 0.0:
            self._log.warning("zero_or_negative_equity", equity=account_equity)
            return self._zero_position(max_loss_per_contract)

        if max_loss_per_contract <= 0.0:
            self._log.warning(
                "zero_or_negative_max_loss",
                max_loss_per_contract=max_loss_per_contract,
            )
            return self._zero_position(max_loss_per_contract)

        # Step 1 & 2: Kelly fraction (quarter-Kelly).
        kelly = self.calculate_kelly_fraction(win_probability, avg_win, avg_loss)
        if kelly <= 0.0:
            self._log.info(
                "negative_kelly_no_trade",
                kelly=kelly,
                win_prob=win_probability,
                avg_win=avg_win,
                avg_loss=avg_loss,
            )
            return self._zero_position(max_loss_per_contract, kelly_fraction=0.0)

        # Step 3: Dollar risk from equity * fractional Kelly.
        dollar_risk = account_equity * kelly

        # Step 4: Cap at per-trade limits.
        max_risk_by_pct = account_equity * self._max_risk_pct
        max_risk_by_dollars = self._max_risk_dollars
        dollar_risk = min(dollar_risk, max_risk_by_pct, max_risk_by_dollars)

        # Step 5: Regime adjustment.
        regime_factor = self.apply_regime_adjustment(1.0, regime)
        dollar_risk *= regime_factor

        # Step 6: Circuit breaker adjustment.
        cb_factor = self.apply_circuit_breaker_adjustment(
            1.0, circuit_breaker_level, recovery_stage
        )
        dollar_risk *= cb_factor

        # Step 7: Convert to contracts.
        contracts = int(math.floor(dollar_risk / max_loss_per_contract))

        # Step 8: Reject trade if risk per contract exceeds allowable risk.
        # Never force 1 contract when max_loss_per_contract > dollar_risk —
        # that would exceed the per-trade risk budget.
        if contracts < 1:
            self._log.warning(
                "position_size_zero_contracts",
                dollar_risk=round(dollar_risk, 2),
                max_loss_per_contract=round(max_loss_per_contract, 2),
                reason="spread too wide for current risk budget",
            )
            contracts = 0

        # Ensure dollar_risk is consistent with the actual contract count.
        actual_dollar_risk = contracts * max_loss_per_contract

        self._log.info(
            "position_size_calculated",
            contracts=contracts,
            dollar_risk=round(actual_dollar_risk, 2),
            kelly_fraction=round(kelly, 6),
            regime=regime,
            regime_factor=regime_factor,
            cb_level=circuit_breaker_level,
            cb_factor=cb_factor,
            recovery_stage=recovery_stage,
            account_equity=round(account_equity, 2),
            max_loss_per_contract=round(max_loss_per_contract, 2),
        )

        return PositionSize(
            contracts=contracts,
            dollar_risk=round(actual_dollar_risk, 2),
            kelly_fraction=round(kelly, 6),
            regime_factor=regime_factor,
            cb_factor=cb_factor,
            risk_per_contract=round(max_loss_per_contract, 2),
        )

    def calculate_kelly_fraction(
        self,
        win_probability: float,
        avg_win: float,
        avg_loss: float,
    ) -> float:
        """Compute the fractional Kelly criterion bet size.

        The Kelly criterion formula:

            f* = (p * b - q) / b

        where:
            p = probability of winning
            q = probability of losing = 1 - p
            b = ratio of average win to average loss (odds)

        This method applies quarter-Kelly scaling (multiply by 0.25) and
        caps the result at :data:`_MAX_KELLY_FRACTION` (25%).

        Args:
            win_probability: Estimated probability of the trade being
                profitable (0.0 to 1.0).
            avg_win: Average dollar profit on winning trades.
            avg_loss: Average dollar loss on losing trades (positive number).

        Returns:
            The fractional Kelly fraction, or 0.0 if the edge is negative.
        """
        # Validate inputs.
        if win_probability <= 0.0 or win_probability >= 1.0:
            self._log.debug(
                "kelly_edge_case_probability",
                win_probability=win_probability,
            )
            if win_probability <= 0.0:
                return 0.0
            # win_probability >= 1.0 is unrealistic but handle gracefully.

        if avg_win <= 0.0 or avg_loss <= 0.0:
            self._log.debug(
                "kelly_invalid_win_loss",
                avg_win=avg_win,
                avg_loss=avg_loss,
            )
            return 0.0

        p = win_probability
        q = 1.0 - p
        b = avg_win / avg_loss  # Win/loss ratio (the "odds").

        # Kelly fraction: f* = (p * b - q) / b
        kelly_full = (p * b - q) / b

        if kelly_full <= 0.0:
            self._log.debug(
                "negative_kelly",
                kelly_full=round(kelly_full, 6),
                win_prob=p,
                odds=round(b, 4),
            )
            return 0.0

        # Quarter-Kelly for safety.
        kelly_fractional = kelly_full * _KELLY_FRACTION_MULTIPLIER

        # Hard cap.
        kelly_fractional = min(kelly_fractional, _MAX_KELLY_FRACTION)

        self._log.debug(
            "kelly_calculated",
            kelly_full=round(kelly_full, 6),
            kelly_fractional=round(kelly_fractional, 6),
            win_prob=p,
            odds=round(b, 4),
        )

        return kelly_fractional

    def apply_regime_adjustment(self, base_size: float, regime: str) -> float:
        """Apply a market-regime-based multiplier to the base size.

        More volatile or crisis regimes reduce position sizes to limit
        exposure during unfavorable conditions.

        Args:
            base_size: The base size value to adjust (can be a dollar
                amount or a dimensionless factor).
            regime: Current market regime label.

        Returns:
            The adjusted size after applying the regime factor.
        """
        factor = _REGIME_FACTORS.get(regime, _DEFAULT_REGIME_FACTOR)

        if regime not in _REGIME_FACTORS:
            self._log.warning(
                "unknown_regime_using_default",
                regime=regime,
                default_factor=_DEFAULT_REGIME_FACTOR,
            )

        return base_size * factor

    def apply_circuit_breaker_adjustment(
        self,
        base_size: float,
        cb_level: str,
        recovery_stage: int,
    ) -> float:
        """Apply circuit breaker and recovery stage multipliers.

        When the system is in a drawdown state, position sizes are reduced
        or eliminated.  During recovery, a graduated ladder slowly restores
        full sizing as the system proves profitable again.

        Args:
            base_size: The base size value to adjust.
            cb_level: Current circuit breaker level name.
            recovery_stage: Current recovery ladder stage (0 = normal).

        Returns:
            The adjusted size after applying both the circuit breaker and
            recovery stage factors.
        """
        # Circuit breaker level multiplier.
        cb_multiplier = self._cb_multipliers.get(
            cb_level, _CB_LEVEL_MULTIPLIERS.get(cb_level, 0.0)
        )

        # If trading is halted at the circuit breaker level, override
        # any recovery stage — no trading permitted.
        if cb_multiplier <= 0.0:
            return 0.0

        # Recovery stage multiplier.
        recovery_multiplier = self._recovery_multipliers.get(recovery_stage, 0.50)

        combined = cb_multiplier * recovery_multiplier

        self._log.debug(
            "cb_adjustment_applied",
            cb_level=cb_level,
            cb_multiplier=cb_multiplier,
            recovery_stage=recovery_stage,
            recovery_multiplier=recovery_multiplier,
            combined_factor=round(combined, 4),
        )

        return base_size * combined

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _zero_position(
        self,
        max_loss_per_contract: float,
        kelly_fraction: float = 0.0,
    ) -> PositionSize:
        """Return a :class:`PositionSize` indicating no trade should be taken.

        Args:
            max_loss_per_contract: The risk per contract (preserved for
                reference even though zero contracts are recommended).
            kelly_fraction: The calculated Kelly fraction to include in
                the result.

        Returns:
            A :class:`PositionSize` with ``contracts=0``.
        """
        return PositionSize(
            contracts=0,
            dollar_risk=0.0,
            kelly_fraction=kelly_fraction,
            regime_factor=0.0,
            cb_factor=0.0,
            risk_per_contract=round(max(max_loss_per_contract, 0.0), 2),
        )
