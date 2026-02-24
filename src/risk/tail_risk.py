"""Tail risk monitoring for Project Titan.

Computes a composite tail risk score from VIX, CBOE SKEW, VVIX,
put/call ratio, and high-yield credit spreads.  When the composite
score exceeds the configured threshold, trading is halted to protect
the portfolio from extreme market events.

Usage::

    from src.risk.tail_risk import TailRiskMonitor

    monitor = TailRiskMonitor(risk_config=config["tail_risk"])
    score = monitor.calculate_tail_score(
        vix=28.5, skew=135.0, vvix=115.0,
        put_call_ratio=1.2, credit_spread=5.5,
    )
    in_danger, reason = monitor.is_danger_zone(score)
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from src.utils.logging import get_logger

if TYPE_CHECKING:
    import structlog

# ---------------------------------------------------------------------------
# Pydantic model
# ---------------------------------------------------------------------------


class TailRiskScore(BaseModel):
    """Composite tail risk score with per-component breakdown.

    All component scores are normalized to the 0.0--1.0 range where
    higher values indicate greater tail risk.  The ``composite`` score
    is the weighted sum of all components.

    Attributes:
        composite: Weighted composite tail risk score (0.0--1.0).
        vix_component: Normalized VIX component (0.0--1.0).
        skew_component: Normalized CBOE SKEW component (0.0--1.0).
        vvix_component: Normalized VVIX component (0.0--1.0).
        pcr_component: Normalized put/call ratio component (0.0--1.0).
        credit_component: Normalized credit spread component (0.0--1.0).
        is_danger: ``True`` when composite exceeds the halt threshold.
        timestamp: When the score was calculated.
    """

    composite: float = 0.0
    vix_component: float = 0.0
    skew_component: float = 0.0
    vvix_component: float = 0.0
    pcr_component: float = 0.0
    credit_component: float = 0.0
    is_danger: bool = False
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


# ---------------------------------------------------------------------------
# Normalization reference levels
#
# Each input metric is normalized to [0, 1] using a baseline (calm) and
# danger level.  Values at or below baseline map to 0.0; values at or
# above the danger level map to 1.0; intermediate values are linearly
# interpolated.
# ---------------------------------------------------------------------------

# VIX: baseline ~12 (calm), danger at config level (default 30)
_VIX_BASELINE: float = 12.0

# CBOE SKEW: baseline ~110 (normal), danger at config level (default 140)
_SKEW_BASELINE: float = 110.0

# VVIX: baseline ~80 (calm), danger at config level (default 120)
_VVIX_BASELINE: float = 80.0

# Put/Call ratio: baseline ~0.70 (normal), danger at ~1.50 (extreme fear)
_PCR_BASELINE: float = 0.70
_PCR_DANGER: float = 1.50

# HY Credit spread (OAS): baseline ~3.0% (normal), danger at ~8.0% (stress)
_CREDIT_BASELINE: float = 3.0
_CREDIT_DANGER: float = 8.0


# ---------------------------------------------------------------------------
# TailRiskMonitor
# ---------------------------------------------------------------------------


class TailRiskMonitor:
    """Monitors tail risk indicators and computes a composite danger score.

    Normalizes VIX, CBOE SKEW, VVIX, put/call ratio, and high-yield
    credit spreads to a 0--1 scale, then combines them using configurable
    weights to produce a single composite tail risk score.

    Args:
        risk_config: The ``tail_risk`` section from ``risk_limits.yaml``.
    """

    def __init__(self, risk_config: dict[str, Any]) -> None:
        self._log: structlog.stdlib.BoundLogger = get_logger("risk.tail_risk")
        self._config = risk_config

        # Danger levels from config
        self._vix_danger: float = float(self._config.get("vix_danger_level", 30))
        self._vix_crisis: float = float(self._config.get("vix_crisis_level", 40))
        self._skew_danger: float = float(self._config.get("skew_danger_level", 140))
        self._vvix_danger: float = float(self._config.get("vvix_danger_level", 120))
        self._composite_halt: float = float(
            self._config.get("composite_tail_score_halt", 0.80)
        )

        # Component weights from config
        weights = self._config.get("weights", {})
        self._weight_vix: float = float(weights.get("vix_component", 0.30))
        self._weight_skew: float = float(weights.get("skew_component", 0.25))
        self._weight_vvix: float = float(weights.get("vvix_component", 0.20))
        self._weight_pcr: float = float(weights.get("put_call_ratio_component", 0.15))
        self._weight_credit: float = float(weights.get("credit_spread_component", 0.10))

        self._log.info(
            "tail_risk_monitor_initialized",
            vix_danger=self._vix_danger,
            vix_crisis=self._vix_crisis,
            skew_danger=self._skew_danger,
            vvix_danger=self._vvix_danger,
            composite_halt=self._composite_halt,
            weights={
                "vix": self._weight_vix,
                "skew": self._weight_skew,
                "vvix": self._weight_vvix,
                "pcr": self._weight_pcr,
                "credit": self._weight_credit,
            },
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def calculate_tail_score(
        self,
        vix: float,
        skew: float,
        vvix: float,
        put_call_ratio: float,
        credit_spread: float,
    ) -> TailRiskScore:
        """Compute the composite tail risk score from market indicators.

        Each component is normalized to the 0.0--1.0 range using linear
        interpolation between its baseline (calm market) and danger level.
        The composite is the weighted sum of all components, clamped to
        [0, 1].

        Args:
            vix: Current VIX index level.
            skew: Current CBOE SKEW index level.
            vvix: Current VVIX (VIX of VIX) level.
            put_call_ratio: Current equity put/call ratio.
            credit_spread: Current high-yield credit spread (OAS) in
                percentage points.

        Returns:
            A :class:`TailRiskScore` with per-component and composite
            scores.
        """
        # Normalize each component to [0, 1]
        vix_score = _normalize(vix, _VIX_BASELINE, self._vix_danger)
        skew_score = _normalize(skew, _SKEW_BASELINE, self._skew_danger)
        vvix_score = _normalize(vvix, _VVIX_BASELINE, self._vvix_danger)
        pcr_score = _normalize(put_call_ratio, _PCR_BASELINE, _PCR_DANGER)
        credit_score = _normalize(credit_spread, _CREDIT_BASELINE, _CREDIT_DANGER)

        # Weighted composite
        composite = (
            self._weight_vix * vix_score
            + self._weight_skew * skew_score
            + self._weight_vvix * vvix_score
            + self._weight_pcr * pcr_score
            + self._weight_credit * credit_score
        )
        composite = max(0.0, min(1.0, composite))

        is_danger = composite >= self._composite_halt

        score = TailRiskScore(
            composite=round(composite, 4),
            vix_component=round(vix_score, 4),
            skew_component=round(skew_score, 4),
            vvix_component=round(vvix_score, 4),
            pcr_component=round(pcr_score, 4),
            credit_component=round(credit_score, 4),
            is_danger=is_danger,
        )

        self._log.info(
            "tail_risk_score_calculated",
            composite=score.composite,
            vix=vix,
            vix_component=score.vix_component,
            skew=skew,
            skew_component=score.skew_component,
            vvix=vvix,
            vvix_component=score.vvix_component,
            pcr=put_call_ratio,
            pcr_component=score.pcr_component,
            credit_spread=credit_spread,
            credit_component=score.credit_component,
            is_danger=score.is_danger,
        )

        return score

    def is_danger_zone(self, tail_score: TailRiskScore) -> tuple[bool, str]:
        """Determine if the tail risk score warrants halting trading.

        Args:
            tail_score: The computed tail risk score to evaluate.

        Returns:
            A tuple of ``(in_danger, reason)``.  ``in_danger`` is ``True``
            when the composite score exceeds the configured halt threshold.
        """
        if tail_score.is_danger:
            reason = (
                f"Composite tail risk score {tail_score.composite:.2f} exceeds "
                f"halt threshold {self._composite_halt:.2f} — "
                f"VIX={tail_score.vix_component:.2f}, "
                f"SKEW={tail_score.skew_component:.2f}, "
                f"VVIX={tail_score.vvix_component:.2f}, "
                f"PCR={tail_score.pcr_component:.2f}, "
                f"Credit={tail_score.credit_component:.2f}"
            )
            self._log.warning(
                "tail_risk_danger_zone",
                composite=tail_score.composite,
                threshold=self._composite_halt,
            )
            return True, reason

        return False, ""

    def get_regime_override(self, vix: float) -> str | None:
        """Return a regime override based on the current VIX level.

        Extreme VIX levels override the HMM-based regime classification
        to ensure the system responds appropriately to crisis conditions.

        Args:
            vix: Current VIX index level.

        Returns:
            ``"crisis"`` if VIX exceeds the crisis level,
            ``"high_vol_trend"`` if VIX exceeds the danger level,
            or ``None`` if no override is warranted.
        """
        if vix >= self._vix_crisis:
            self._log.warning(
                "regime_override_crisis",
                vix=vix,
                crisis_level=self._vix_crisis,
            )
            return "crisis"

        if vix >= self._vix_danger:
            self._log.info(
                "regime_override_high_vol",
                vix=vix,
                danger_level=self._vix_danger,
            )
            return "high_vol_trend"

        return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalize(value: float, baseline: float, danger: float) -> float:
    """Normalize a value to the [0, 1] range using linear interpolation.

    Values at or below ``baseline`` map to 0.0.  Values at or above
    ``danger`` map to 1.0.  Intermediate values are linearly interpolated.

    Args:
        value: The raw input value.
        baseline: The level representing calm/normal conditions (maps to 0).
        danger: The level representing danger conditions (maps to 1).

    Returns:
        A float clamped to [0.0, 1.0].
    """
    if danger <= baseline:
        # Avoid division by zero; treat as fully dangerous
        return 1.0 if value >= baseline else 0.0

    normalized = (value - baseline) / (danger - baseline)
    return max(0.0, min(1.0, normalized))
