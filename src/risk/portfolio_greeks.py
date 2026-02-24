"""Portfolio-level Greeks aggregation and limit enforcement.

Aggregates delta, gamma, theta, and vega across all open positions and
checks them against configurable limits.  Provides pre-trade validation
to ensure that adding a new position would not push portfolio Greeks
beyond acceptable bounds.

Usage::

    from src.risk.portfolio_greeks import PortfolioGreeksMonitor

    monitor = PortfolioGreeksMonitor(risk_config=config["greeks_limits"])
    greeks = monitor.calculate_portfolio_greeks(positions)
    violations = monitor.check_limits(greeks)
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from src.utils.logging import get_logger

if TYPE_CHECKING:
    import structlog

# ---------------------------------------------------------------------------
# Contract multiplier for standard US equity options
# ---------------------------------------------------------------------------

DEFAULT_OPTION_MULTIPLIER: int = 100


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class PortfolioGreeks(BaseModel):
    """Aggregate Greeks for the entire options portfolio.

    Attributes:
        net_delta: Sum of position-weighted deltas across all positions.
        net_gamma: Sum of position-weighted gammas across all positions.
        net_theta: Sum of position-weighted thetas (daily decay) across
            all positions.  Negative means the portfolio loses money from
            time decay each day.
        net_vega: Sum of position-weighted vegas across all positions.
        timestamp: When the calculation was performed.
    """

    net_delta: float = 0.0
    net_gamma: float = 0.0
    net_theta: float = 0.0
    net_vega: float = 0.0
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class GreeksViolation(BaseModel):
    """A single Greek limit violation.

    Attributes:
        greek_name: The name of the Greek that exceeded its limit
            (``"delta"``, ``"gamma"``, ``"theta"``, or ``"vega"``).
        current_value: The current portfolio-level value of this Greek.
        limit_value: The configured limit that was exceeded.
        excess: How much the current value exceeds the limit.
    """

    greek_name: str
    current_value: float
    limit_value: float
    excess: float


# ---------------------------------------------------------------------------
# PortfolioGreeksMonitor
# ---------------------------------------------------------------------------


class PortfolioGreeksMonitor:
    """Monitors aggregate portfolio Greeks and enforces exposure limits.

    Calculates net delta, gamma, theta, and vega across all open positions
    and checks them against configurable limits from ``risk_limits.yaml``.

    Args:
        risk_config: The ``greeks_limits`` section from ``risk_limits.yaml``.
    """

    def __init__(self, risk_config: dict[str, Any]) -> None:
        self._log: structlog.stdlib.BoundLogger = get_logger("risk.portfolio_greeks")
        self._config = risk_config

        # Load limits from configuration
        self._max_delta: float = float(self._config.get("max_portfolio_delta", 500))
        self._max_gamma: float = float(self._config.get("max_portfolio_gamma", 200))
        self._max_theta: float = float(self._config.get("max_portfolio_theta", -500))
        self._max_vega: float = float(self._config.get("max_portfolio_vega", 1000))
        self._delta_hedge_threshold: float = float(
            self._config.get("delta_hedge_threshold", 300)
        )

        self._log.info(
            "portfolio_greeks_monitor_initialized",
            max_delta=self._max_delta,
            max_gamma=self._max_gamma,
            max_theta=self._max_theta,
            max_vega=self._max_vega,
            delta_hedge_threshold=self._delta_hedge_threshold,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def calculate_portfolio_greeks(
        self,
        positions: list[dict[str, Any]],
    ) -> PortfolioGreeks:
        """Aggregate Greeks across all open positions.

        Each position dictionary must contain at minimum the keys:
        ``quantity``, ``delta``, ``gamma``, ``theta``, ``vega``.
        Optionally, ``multiplier`` may be provided (defaults to 100).

        The quantity should be positive for long positions and negative
        for short positions.  Greeks are multiplied by quantity and the
        option multiplier.

        Args:
            positions: List of position dictionaries with per-contract
                Greeks and signed quantities.

        Returns:
            A :class:`PortfolioGreeks` with the aggregate values.
        """
        net_delta = 0.0
        net_gamma = 0.0
        net_theta = 0.0
        net_vega = 0.0

        for pos in positions:
            quantity = pos.get("quantity", 0)
            multiplier = pos.get("multiplier", DEFAULT_OPTION_MULTIPLIER)
            delta = pos.get("delta", 0.0)
            gamma = pos.get("gamma", 0.0)
            theta = pos.get("theta", 0.0)
            vega = pos.get("vega", 0.0)

            net_delta += quantity * delta * multiplier
            net_gamma += quantity * gamma * multiplier
            net_theta += quantity * theta * multiplier
            net_vega += quantity * vega * multiplier

        greeks = PortfolioGreeks(
            net_delta=round(net_delta, 4),
            net_gamma=round(net_gamma, 4),
            net_theta=round(net_theta, 4),
            net_vega=round(net_vega, 4),
        )

        self._log.debug(
            "portfolio_greeks_calculated",
            net_delta=greeks.net_delta,
            net_gamma=greeks.net_gamma,
            net_theta=greeks.net_theta,
            net_vega=greeks.net_vega,
            position_count=len(positions),
        )

        return greeks

    def check_limits(self, greeks: PortfolioGreeks) -> list[GreeksViolation]:
        """Check aggregate Greeks against configured portfolio limits.

        Args:
            greeks: The current portfolio-level Greeks to validate.

        Returns:
            A list of :class:`GreeksViolation` objects.  An empty list
            means all Greeks are within limits.
        """
        violations: list[GreeksViolation] = []

        # Delta: absolute value must not exceed max
        if abs(greeks.net_delta) > self._max_delta:
            excess = abs(greeks.net_delta) - self._max_delta
            violations.append(
                GreeksViolation(
                    greek_name="delta",
                    current_value=greeks.net_delta,
                    limit_value=self._max_delta,
                    excess=round(excess, 4),
                )
            )
            self._log.warning(
                "greeks_limit_exceeded",
                greek="delta",
                current=greeks.net_delta,
                limit=self._max_delta,
                excess=round(excess, 4),
            )

        # Gamma: absolute value must not exceed max
        if abs(greeks.net_gamma) > self._max_gamma:
            excess = abs(greeks.net_gamma) - self._max_gamma
            violations.append(
                GreeksViolation(
                    greek_name="gamma",
                    current_value=greeks.net_gamma,
                    limit_value=self._max_gamma,
                    excess=round(excess, 4),
                )
            )
            self._log.warning(
                "greeks_limit_exceeded",
                greek="gamma",
                current=greeks.net_gamma,
                limit=self._max_gamma,
                excess=round(excess, 4),
            )

        # Theta: max_portfolio_theta is a negative number (e.g. -500).
        # A violation occurs when net_theta is MORE negative than the limit.
        if greeks.net_theta < self._max_theta:
            excess = self._max_theta - greeks.net_theta
            violations.append(
                GreeksViolation(
                    greek_name="theta",
                    current_value=greeks.net_theta,
                    limit_value=self._max_theta,
                    excess=round(excess, 4),
                )
            )
            self._log.warning(
                "greeks_limit_exceeded",
                greek="theta",
                current=greeks.net_theta,
                limit=self._max_theta,
                excess=round(excess, 4),
            )

        # Vega: absolute value must not exceed max
        if abs(greeks.net_vega) > self._max_vega:
            excess = abs(greeks.net_vega) - self._max_vega
            violations.append(
                GreeksViolation(
                    greek_name="vega",
                    current_value=greeks.net_vega,
                    limit_value=self._max_vega,
                    excess=round(excess, 4),
                )
            )
            self._log.warning(
                "greeks_limit_exceeded",
                greek="vega",
                current=greeks.net_vega,
                limit=self._max_vega,
                excess=round(excess, 4),
            )

        if not violations:
            self._log.debug("greeks_within_limits")

        return violations

    def would_exceed_limits(
        self,
        greeks: PortfolioGreeks,
        new_trade_greeks: dict[str, Any],
    ) -> tuple[bool, list[str]]:
        """Check if adding a new trade would push portfolio Greeks over limits.

        Projects the post-trade Greeks by adding the new trade's contribution
        to the current portfolio totals, then checks each against its limit.

        The ``new_trade_greeks`` dictionary should contain the *total*
        Greek contribution of the proposed trade (quantity * per-contract
        Greek * multiplier already applied).

        Args:
            greeks: The current portfolio-level Greeks.
            new_trade_greeks: Dictionary with keys ``delta``, ``gamma``,
                ``theta``, ``vega`` representing the new trade's total
                Greek contribution.

        Returns:
            A tuple of ``(would_exceed, reasons)`` where ``would_exceed``
            is ``True`` if any limit would be breached, and ``reasons``
            is a list of human-readable explanations.
        """
        projected_delta = greeks.net_delta + new_trade_greeks.get("delta", 0.0)
        projected_gamma = greeks.net_gamma + new_trade_greeks.get("gamma", 0.0)
        projected_theta = greeks.net_theta + new_trade_greeks.get("theta", 0.0)
        projected_vega = greeks.net_vega + new_trade_greeks.get("vega", 0.0)

        reasons: list[str] = []

        if abs(projected_delta) > self._max_delta:
            reasons.append(
                f"Delta would be {projected_delta:.1f} "
                f"(limit: +/-{self._max_delta:.0f})"
            )

        if abs(projected_gamma) > self._max_gamma:
            reasons.append(
                f"Gamma would be {projected_gamma:.1f} "
                f"(limit: +/-{self._max_gamma:.0f})"
            )

        if projected_theta < self._max_theta:
            reasons.append(
                f"Theta would be {projected_theta:.1f} (limit: {self._max_theta:.0f})"
            )

        if abs(projected_vega) > self._max_vega:
            reasons.append(
                f"Vega would be {projected_vega:.1f} (limit: +/-{self._max_vega:.0f})"
            )

        if reasons:
            self._log.info(
                "trade_would_exceed_greeks_limits",
                reasons=reasons,
                projected_delta=round(projected_delta, 2),
                projected_gamma=round(projected_gamma, 2),
                projected_theta=round(projected_theta, 2),
                projected_vega=round(projected_vega, 2),
            )

        return bool(reasons), reasons

    def needs_delta_hedge(
        self,
        greeks: PortfolioGreeks,
    ) -> tuple[bool, float]:
        """Determine if the portfolio needs a delta hedge.

        A delta hedge is recommended when the absolute net delta exceeds
        the configured ``delta_hedge_threshold``.

        Args:
            greeks: The current portfolio-level Greeks.

        Returns:
            A tuple of ``(needs_hedge, hedge_amount)`` where
            ``needs_hedge`` is ``True`` when net delta exceeds the
            threshold, and ``hedge_amount`` is the number of delta
            units to offset (negative means sell, positive means buy).
        """
        if abs(greeks.net_delta) > self._delta_hedge_threshold:
            # Hedge amount is the inverse of the excess delta
            hedge_amount = -greeks.net_delta
            self._log.info(
                "delta_hedge_needed",
                net_delta=greeks.net_delta,
                threshold=self._delta_hedge_threshold,
                hedge_amount=round(hedge_amount, 2),
            )
            return True, round(hedge_amount, 4)

        return False, 0.0
