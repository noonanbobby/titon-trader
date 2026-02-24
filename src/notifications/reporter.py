"""Performance report generation for Project Titan.

Produces weekly and monthly HTML tear sheets using QuantStats, plus
custom strategy-level reports.  All blocking QuantStats operations
are executed in a thread executor so the async event loop is never
starved.

Usage::

    from src.notifications.reporter import ReportGenerator

    reporter = ReportGenerator(output_dir="data/reports")
    path = await reporter.generate_weekly_report(trades, snapshots)
"""

from __future__ import annotations

import asyncio
from datetime import UTC, date, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import quantstats as qs

from src.utils.logging import get_logger

if TYPE_CHECKING:
    import structlog

logger: structlog.stdlib.BoundLogger = get_logger("notifications.reporter")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CSS = """\
<style>
  body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
         margin: 0; padding: 20px; background: #0d1117; color: #c9d1d9; }
  .header { background: linear-gradient(135deg, #161b22, #1f2937);
             padding: 24px 32px; border-radius: 8px;
             margin-bottom: 24px; }
  .header h1 { margin: 0 0 4px 0; font-size: 28px; color: #58a6ff; }
  .header p  { margin: 0; color: #8b949e; font-size: 14px; }
  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
           gap: 16px; margin-bottom: 24px; }
  .card { background: #161b22; border: 1px solid #30363d;
           border-radius: 8px; padding: 16px; }
  .card .label { font-size: 12px; text-transform: uppercase;
                  color: #8b949e; margin-bottom: 4px; }
  .card .value { font-size: 24px; font-weight: 700; }
  .positive { color: #3fb950; }
  .negative { color: #f85149; }
  .neutral  { color: #c9d1d9; }
  table { width: 100%; border-collapse: collapse; margin-bottom: 24px; }
  th { text-align: left; padding: 10px 12px; background: #161b22;
       color: #8b949e; font-size: 12px; text-transform: uppercase;
       border-bottom: 2px solid #30363d; }
  td { padding: 10px 12px; border-bottom: 1px solid #21262d;
       font-size: 14px; }
  tr:hover td { background: #161b22; }
  .section { margin-bottom: 32px; }
  .section h2 { font-size: 20px; color: #58a6ff;
                  margin: 0 0 12px 0; }
  .qs-embed { background: #fff; border-radius: 8px;
               padding: 16px; margin-top: 16px; }
  .qs-embed img { max-width: 100%; }
</style>
"""

_STRATEGY_DISPLAY_NAMES: dict[str, str] = {
    "bull_call_spread": "Bull Call Spread",
    "bull_put_spread": "Bull Put Spread",
    "iron_condor": "Iron Condor",
    "calendar_spread": "Calendar Spread",
    "diagonal_spread": "Diagonal Spread",
    "broken_wing_butterfly": "Broken-Wing Butterfly",
    "short_strangle": "Short Strangle",
    "pmcc": "Poor Man's Covered Call",
    "ratio_spread": "Ratio Spread",
    "long_straddle": "Long Straddle",
}


def _pnl_class(value: float) -> str:
    """Return the CSS class name for a P&L value."""
    if value > 0:
        return "positive"
    if value < 0:
        return "negative"
    return "neutral"


def _fmt_dollar(value: float) -> str:
    """Format a dollar value with sign and thousands separator."""
    sign = "+" if value > 0 else ""
    return f"{sign}${value:,.2f}"


def _fmt_pct(value: float) -> str:
    """Format a percentage value with sign."""
    sign = "+" if value > 0 else ""
    return f"{sign}{value:.2f}%"


def _safe_div(
    numerator: float,
    denominator: float,
    default: float = 0.0,
) -> float:
    """Divide without raising on zero denominators."""
    if denominator == 0:
        return default
    return numerator / denominator


class ReportGenerator:
    """Produces HTML performance reports using QuantStats metrics.

    Parameters
    ----------
    output_dir:
        Directory where generated HTML files are saved.  Created
        automatically if it does not exist.
    """

    def __init__(self, output_dir: str = "data/reports") -> None:
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def generate_weekly_report(
        self,
        trades: list[dict[str, Any]],
        account_history: list[dict[str, Any]],
    ) -> str:
        """Generate a weekly HTML tear sheet.

        Combines QuantStats analytics with a custom trade summary
        section.  Blocking QuantStats work runs inside the default
        thread executor.

        Parameters
        ----------
        trades:
            List of trade dictionaries (matching the ``trades`` table
            schema).
        account_history:
            List of account snapshot dictionaries (matching the
            ``account_snapshots`` table schema).

        Returns
        -------
        str
            Absolute file path of the generated HTML report.
        """
        today = date.today()
        filename = f"weekly_{today.isoformat()}.html"
        filepath = self._output_dir / filename

        returns = self._build_returns_series(account_history)
        summary = self._build_trade_summary(trades)

        qs_html = await self._generate_qs_html(returns, "Weekly")

        sections: list[dict[str, str]] = [
            {
                "title": "Summary Statistics",
                "content": self._render_summary_cards(summary),
            },
            {
                "title": "Trade Log",
                "content": self._render_trade_table(trades),
            },
            {
                "title": "Strategy Breakdown",
                "content": self._render_strategy_table(summary.get("per_strategy", {})),
            },
        ]

        if qs_html:
            sections.append(
                {
                    "title": "QuantStats Tear Sheet",
                    "content": (f'<div class="qs-embed">{qs_html}</div>'),
                }
            )

        html = self._create_html_report(
            title=f"Weekly Report - {today.isoformat()}",
            sections=sections,
        )

        await asyncio.to_thread(filepath.write_text, html)
        logger.info(
            "weekly report generated",
            path=str(filepath),
            trades=len(trades),
        )
        return str(filepath.resolve())

    async def generate_monthly_report(
        self,
        trades: list[dict[str, Any]],
        account_history: list[dict[str, Any]],
    ) -> str:
        """Generate a comprehensive monthly HTML report.

        Includes everything in the weekly report plus regime analysis
        and Greeks exposure history.

        Parameters
        ----------
        trades:
            List of trade dictionaries.
        account_history:
            List of account snapshot dictionaries.

        Returns
        -------
        str
            Absolute file path of the generated HTML report.
        """
        today = date.today()
        filename = f"monthly_{today.isoformat()}.html"
        filepath = self._output_dir / filename

        returns = self._build_returns_series(account_history)
        summary = self._build_trade_summary(trades)

        qs_html = await self._generate_qs_html(returns, "Monthly")

        sections: list[dict[str, str]] = [
            {
                "title": "Summary Statistics",
                "content": self._render_summary_cards(summary),
            },
            {
                "title": "Strategy Breakdown",
                "content": self._render_strategy_table(summary.get("per_strategy", {})),
            },
            {
                "title": "Regime Analysis",
                "content": self._render_regime_table(summary.get("per_regime", {})),
            },
            {
                "title": "Greeks Exposure History",
                "content": self._render_greeks_history(account_history),
            },
            {
                "title": "Trade Log",
                "content": self._render_trade_table(trades),
            },
        ]

        if qs_html:
            sections.append(
                {
                    "title": "QuantStats Tear Sheet",
                    "content": (f'<div class="qs-embed">{qs_html}</div>'),
                }
            )

        html = self._create_html_report(
            title=f"Monthly Report - {today.isoformat()}",
            sections=sections,
        )

        await asyncio.to_thread(filepath.write_text, html)
        logger.info(
            "monthly report generated",
            path=str(filepath),
            trades=len(trades),
        )
        return str(filepath.resolve())

    async def generate_strategy_report(
        self,
        trades: list[dict[str, Any]],
        strategy: str,
    ) -> str:
        """Generate a performance report for a single strategy.

        Parameters
        ----------
        trades:
            List of trade dictionaries (all strategies; filtered
            internally).
        strategy:
            Strategy identifier to filter on (e.g.
            ``"iron_condor"``).

        Returns
        -------
        str
            Absolute file path of the generated HTML report.
        """
        today = date.today()
        filename = f"strategy_{strategy}_{today.isoformat()}.html"
        filepath = self._output_dir / filename

        filtered = [t for t in trades if t.get("strategy") == strategy]
        summary = self._build_trade_summary(filtered)

        display_name = _STRATEGY_DISPLAY_NAMES.get(strategy, strategy)

        sections: list[dict[str, str]] = [
            {
                "title": f"{display_name} Performance",
                "content": self._render_summary_cards(summary),
            },
            {
                "title": "Regime Breakdown",
                "content": self._render_regime_table(summary.get("per_regime", {})),
            },
            {
                "title": "Trade Log",
                "content": self._render_trade_table(filtered),
            },
        ]

        html = self._create_html_report(
            title=(f"Strategy Report: {display_name} - {today.isoformat()}"),
            sections=sections,
        )

        await asyncio.to_thread(filepath.write_text, html)
        logger.info(
            "strategy report generated",
            path=str(filepath),
            strategy=strategy,
            trades=len(filtered),
        )
        return str(filepath.resolve())

    # ------------------------------------------------------------------
    # Returns series construction
    # ------------------------------------------------------------------

    @staticmethod
    def _build_returns_series(
        account_history: list[dict[str, Any]],
    ) -> pd.Series:
        """Convert account snapshots to a daily returns series.

        Snapshots are grouped by calendar date (taking the last value
        per day), then pct_change is computed on ``net_liquidation``.
        Missing trading days (weekends, holidays) are forward-filled
        before computing returns so that no spurious gaps appear.

        Parameters
        ----------
        account_history:
            List of dicts with at least ``timestamp`` and
            ``net_liquidation`` keys.

        Returns
        -------
        pd.Series
            A float64 Series indexed by ``DatetimeIndex`` with daily
            simple returns.  Empty series is returned when input is
            insufficient.
        """
        if len(account_history) < 2:
            return pd.Series(dtype="float64")

        df = pd.DataFrame(account_history)

        # Normalise timestamp to pandas datetime
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        else:
            return pd.Series(dtype="float64")

        if "net_liquidation" not in df.columns:
            return pd.Series(dtype="float64")

        df["net_liquidation"] = pd.to_numeric(df["net_liquidation"], errors="coerce")
        df = df.dropna(subset=["net_liquidation"])

        if df.empty:
            return pd.Series(dtype="float64")

        # Take the last snapshot per calendar day
        df = df.set_index("timestamp")
        daily = df["net_liquidation"].resample("1D").last().dropna()

        # Forward-fill weekends/holidays, then compute returns
        daily = daily.asfreq("1D", method="ffill")
        returns = daily.pct_change().dropna()
        returns.name = "returns"
        # Strip timezone for QuantStats compatibility
        returns.index = returns.index.tz_localize(None)
        return returns

    # ------------------------------------------------------------------
    # Trade summary
    # ------------------------------------------------------------------

    @staticmethod
    def _build_trade_summary(
        trades: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Aggregate trade statistics across multiple dimensions.

        Returns a dict with keys:

        - ``total_pnl``, ``win_rate``, ``avg_win``, ``avg_loss``
        - ``largest_win``, ``largest_loss``, ``total_trades``
        - ``winners``, ``losers``
        - ``sharpe``, ``max_drawdown``
        - ``per_strategy`` — dict keyed by strategy name
        - ``per_regime`` — dict keyed by regime label

        Parameters
        ----------
        trades:
            List of trade dictionaries with at least ``realized_pnl``
            and ``status`` keys.

        Returns
        -------
        dict
            Aggregated statistics dictionary.
        """
        closed = [
            t
            for t in trades
            if t.get("status") == "CLOSED" and t.get("realized_pnl") is not None
        ]

        result: dict[str, Any] = {
            "total_pnl": 0.0,
            "win_rate": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "largest_win": 0.0,
            "largest_loss": 0.0,
            "total_trades": len(closed),
            "winners": 0,
            "losers": 0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "per_strategy": {},
            "per_regime": {},
        }

        if not closed:
            return result

        pnls = [float(t["realized_pnl"]) for t in closed]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        result["total_pnl"] = sum(pnls)
        result["winners"] = len(wins)
        result["losers"] = len(losses)
        result["win_rate"] = _safe_div(len(wins), len(closed)) * 100
        result["avg_win"] = _safe_div(sum(wins), len(wins))
        result["avg_loss"] = _safe_div(sum(losses), len(losses))
        result["largest_win"] = max(pnls) if pnls else 0.0
        result["largest_loss"] = min(pnls) if pnls else 0.0

        # Sharpe ratio from trade returns
        if len(pnls) >= 2:
            pnl_arr = np.array(pnls, dtype=np.float64)
            std = float(np.std(pnl_arr, ddof=1))
            mean = float(np.mean(pnl_arr))
            result["sharpe"] = _safe_div(mean, std)

        # Max drawdown from cumulative P&L
        cum_pnl = np.cumsum(pnls)
        peak = np.maximum.accumulate(cum_pnl)
        drawdown = cum_pnl - peak
        result["max_drawdown"] = float(np.min(drawdown)) if len(drawdown) > 0 else 0.0

        # Per-strategy breakdown
        strategies: dict[str, list[float]] = {}
        for t in closed:
            key = t.get("strategy", "unknown")
            strategies.setdefault(key, []).append(float(t["realized_pnl"]))

        for strat, strat_pnls in strategies.items():
            strat_wins = [p for p in strat_pnls if p > 0]
            strat_losses = [p for p in strat_pnls if p <= 0]
            result["per_strategy"][strat] = {
                "total_pnl": sum(strat_pnls),
                "trades": len(strat_pnls),
                "winners": len(strat_wins),
                "losers": len(strat_losses),
                "win_rate": _safe_div(len(strat_wins), len(strat_pnls)) * 100,
                "avg_pnl": _safe_div(sum(strat_pnls), len(strat_pnls)),
            }

        # Per-regime breakdown
        regimes: dict[str, list[float]] = {}
        for t in closed:
            key = t.get("regime") or "unknown"
            regimes.setdefault(key, []).append(float(t["realized_pnl"]))

        for regime, regime_pnls in regimes.items():
            regime_wins = [p for p in regime_pnls if p > 0]
            regime_losses = [p for p in regime_pnls if p <= 0]
            result["per_regime"][regime] = {
                "total_pnl": sum(regime_pnls),
                "trades": len(regime_pnls),
                "winners": len(regime_wins),
                "losers": len(regime_losses),
                "win_rate": _safe_div(len(regime_wins), len(regime_pnls)) * 100,
                "avg_pnl": _safe_div(sum(regime_pnls), len(regime_pnls)),
            }

        return result

    # ------------------------------------------------------------------
    # QuantStats integration
    # ------------------------------------------------------------------

    async def _generate_qs_html(
        self,
        returns: pd.Series,
        title: str,
    ) -> str:
        """Run QuantStats HTML generation in a thread executor.

        Returns an empty string if the returns series is too short to
        produce a meaningful report.

        Parameters
        ----------
        returns:
            Daily simple returns series.
        title:
            Title string passed to QuantStats.

        Returns
        -------
        str
            Raw HTML fragment from QuantStats, or ``""`` on failure.
        """
        if returns.empty or len(returns) < 3:
            logger.warning(
                "skipping quantstats: insufficient data",
                data_points=len(returns),
            )
            return ""

        def _blocking_qs() -> str:
            try:
                html_str: str = qs.reports.html(
                    returns,
                    benchmark=None,
                    title=f"Titan {title} Report",
                    output=None,
                    download_filename=None,
                )
                return html_str if isinstance(html_str, str) else ""
            except Exception:
                logger.exception("quantstats report generation failed")
                return ""

        return await asyncio.to_thread(_blocking_qs)

    # ------------------------------------------------------------------
    # QuantStats metrics helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_qs_metrics(
        returns: pd.Series,
    ) -> dict[str, float]:
        """Extract key metrics from QuantStats for card display.

        Parameters
        ----------
        returns:
            Daily simple returns series.

        Returns
        -------
        dict
            Dictionary of named metric values.
        """
        if returns.empty or len(returns) < 2:
            return {}

        metrics: dict[str, float] = {}

        try:
            metrics["sharpe"] = float(qs.stats.sharpe(returns))
        except Exception:
            metrics["sharpe"] = 0.0

        try:
            metrics["sortino"] = float(qs.stats.sortino(returns))
        except Exception:
            metrics["sortino"] = 0.0

        try:
            metrics["calmar"] = float(qs.stats.calmar(returns))
        except Exception:
            metrics["calmar"] = 0.0

        try:
            metrics["max_drawdown"] = float(qs.stats.max_drawdown(returns)) * 100
        except Exception:
            metrics["max_drawdown"] = 0.0

        try:
            metrics["var_95"] = float(qs.stats.var(returns)) * 100
        except Exception:
            metrics["var_95"] = 0.0

        try:
            cum = float(qs.stats.comp(returns)) * 100
            metrics["cumulative_return"] = cum
        except Exception:
            metrics["cumulative_return"] = 0.0

        return metrics

    # ------------------------------------------------------------------
    # HTML rendering helpers
    # ------------------------------------------------------------------

    def _create_html_report(
        self,
        title: str,
        sections: list[dict[str, str]],
    ) -> str:
        """Build a complete HTML document from section fragments.

        Parameters
        ----------
        title:
            Document and header title.
        sections:
            Each dict must have ``title`` and ``content`` keys.

        Returns
        -------
        str
            Complete HTML string ready to write to disk.
        """
        section_html = ""
        for sec in sections:
            section_html += (
                f'<div class="section"><h2>{sec["title"]}</h2>{sec["content"]}</div>\n'
            )

        now_utc = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")

        return (
            "<!DOCTYPE html>\n"
            '<html lang="en">\n'
            "<head>\n"
            '  <meta charset="utf-8">\n'
            '  <meta name="viewport" '
            'content="width=device-width, initial-scale=1">\n'
            f"  <title>{title}</title>\n"
            f"{_CSS}\n"
            "</head>\n"
            "<body>\n"
            '<div class="header">\n'
            f"  <h1>{title}</h1>\n"
            f"  <p>Generated {now_utc} by Project Titan</p>\n"
            "</div>\n"
            f"{section_html}"
            "</body>\n"
            "</html>"
        )

    @staticmethod
    def _render_summary_cards(summary: dict[str, Any]) -> str:
        """Render key summary metrics as dashboard cards.

        Parameters
        ----------
        summary:
            Output from :meth:`_build_trade_summary`.

        Returns
        -------
        str
            HTML fragment containing the card grid.
        """
        total_pnl = summary.get("total_pnl", 0.0)
        win_rate = summary.get("win_rate", 0.0)
        total_trades = summary.get("total_trades", 0)
        avg_win = summary.get("avg_win", 0.0)
        avg_loss = summary.get("avg_loss", 0.0)
        largest_win = summary.get("largest_win", 0.0)
        largest_loss = summary.get("largest_loss", 0.0)
        sharpe = summary.get("sharpe", 0.0)
        max_dd = summary.get("max_drawdown", 0.0)

        cards = [
            (
                "Total P&L",
                _fmt_dollar(total_pnl),
                _pnl_class(total_pnl),
            ),
            ("Total Trades", str(total_trades), "neutral"),
            (
                "Win Rate",
                f"{win_rate:.1f}%",
                "positive" if win_rate >= 50 else "negative",
            ),
            (
                "Avg Win",
                _fmt_dollar(avg_win),
                "positive",
            ),
            (
                "Avg Loss",
                _fmt_dollar(avg_loss),
                "negative",
            ),
            (
                "Largest Win",
                _fmt_dollar(largest_win),
                "positive",
            ),
            (
                "Largest Loss",
                _fmt_dollar(largest_loss),
                "negative",
            ),
            (
                "Sharpe Ratio",
                f"{sharpe:.2f}",
                "positive" if sharpe > 0 else "negative",
            ),
            (
                "Max Drawdown",
                _fmt_dollar(max_dd),
                "negative" if max_dd < 0 else "neutral",
            ),
        ]

        html = '<div class="grid">\n'
        for label, value, css_cls in cards:
            html += (
                f'  <div class="card">\n'
                f'    <div class="label">{label}</div>\n'
                f'    <div class="value {css_cls}">{value}</div>\n'
                f"  </div>\n"
            )
        html += "</div>"
        return html

    @staticmethod
    def _render_trade_table(
        trades: list[dict[str, Any]],
    ) -> str:
        """Render a table of individual trades.

        Parameters
        ----------
        trades:
            List of trade dictionaries.

        Returns
        -------
        str
            HTML ``<table>`` fragment.
        """
        if not trades:
            return "<p>No trades in this period.</p>"

        rows = ""
        for t in trades:
            pnl = float(t.get("realized_pnl") or 0)
            css = _pnl_class(pnl)
            ticker = t.get("ticker", "")
            strategy = _STRATEGY_DISPLAY_NAMES.get(
                t.get("strategy", ""), t.get("strategy", "")
            )
            direction = t.get("direction", "")
            status = t.get("status", "")
            entry = t.get("entry_time", "")
            if hasattr(entry, "strftime"):
                entry = entry.strftime("%Y-%m-%d %H:%M")
            confidence = t.get("ml_confidence")
            conf_str = f"{float(confidence):.2f}" if confidence is not None else "-"
            rows += (
                f"<tr>"
                f"<td>{ticker}</td>"
                f"<td>{strategy}</td>"
                f"<td>{direction}</td>"
                f"<td>{status}</td>"
                f"<td>{entry}</td>"
                f'<td class="{css}">{_fmt_dollar(pnl)}</td>'
                f"<td>{conf_str}</td>"
                f"</tr>\n"
            )

        return (
            "<table>\n"
            "<thead><tr>"
            "<th>Ticker</th>"
            "<th>Strategy</th>"
            "<th>Direction</th>"
            "<th>Status</th>"
            "<th>Entry</th>"
            "<th>P&L</th>"
            "<th>Confidence</th>"
            "</tr></thead>\n"
            f"<tbody>\n{rows}</tbody>\n"
            "</table>"
        )

    @staticmethod
    def _render_strategy_table(
        per_strategy: dict[str, dict[str, Any]],
    ) -> str:
        """Render per-strategy performance breakdown table.

        Parameters
        ----------
        per_strategy:
            Dict keyed by strategy name from ``_build_trade_summary``.

        Returns
        -------
        str
            HTML ``<table>`` fragment.
        """
        if not per_strategy:
            return "<p>No strategy data available.</p>"

        rows = ""
        for strat, data in sorted(
            per_strategy.items(),
            key=lambda kv: kv[1].get("total_pnl", 0),
            reverse=True,
        ):
            pnl = data.get("total_pnl", 0.0)
            css = _pnl_class(pnl)
            display = _STRATEGY_DISPLAY_NAMES.get(strat, strat)
            rows += (
                f"<tr>"
                f"<td>{display}</td>"
                f"<td>{data.get('trades', 0)}</td>"
                f"<td>{data.get('winners', 0)}</td>"
                f"<td>{data.get('losers', 0)}</td>"
                f"<td>{data.get('win_rate', 0):.1f}%</td>"
                f"<td>{_fmt_dollar(data.get('avg_pnl', 0))}</td>"
                f'<td class="{css}">'
                f"{_fmt_dollar(pnl)}</td>"
                f"</tr>\n"
            )

        return (
            "<table>\n"
            "<thead><tr>"
            "<th>Strategy</th>"
            "<th>Trades</th>"
            "<th>Winners</th>"
            "<th>Losers</th>"
            "<th>Win Rate</th>"
            "<th>Avg P&L</th>"
            "<th>Total P&L</th>"
            "</tr></thead>\n"
            f"<tbody>\n{rows}</tbody>\n"
            "</table>"
        )

    @staticmethod
    def _render_regime_table(
        per_regime: dict[str, dict[str, Any]],
    ) -> str:
        """Render per-regime performance breakdown table.

        Parameters
        ----------
        per_regime:
            Dict keyed by regime label from ``_build_trade_summary``.

        Returns
        -------
        str
            HTML ``<table>`` fragment.
        """
        if not per_regime:
            return "<p>No regime data available.</p>"

        rows = ""
        for regime, data in sorted(
            per_regime.items(),
            key=lambda kv: kv[1].get("total_pnl", 0),
            reverse=True,
        ):
            pnl = data.get("total_pnl", 0.0)
            css = _pnl_class(pnl)
            rows += (
                f"<tr>"
                f"<td>{regime}</td>"
                f"<td>{data.get('trades', 0)}</td>"
                f"<td>{data.get('winners', 0)}</td>"
                f"<td>{data.get('losers', 0)}</td>"
                f"<td>{data.get('win_rate', 0):.1f}%</td>"
                f"<td>{_fmt_dollar(data.get('avg_pnl', 0))}</td>"
                f'<td class="{css}">'
                f"{_fmt_dollar(pnl)}</td>"
                f"</tr>\n"
            )

        return (
            "<table>\n"
            "<thead><tr>"
            "<th>Regime</th>"
            "<th>Trades</th>"
            "<th>Winners</th>"
            "<th>Losers</th>"
            "<th>Win Rate</th>"
            "<th>Avg P&L</th>"
            "<th>Total P&L</th>"
            "</tr></thead>\n"
            f"<tbody>\n{rows}</tbody>\n"
            "</table>"
        )

    @staticmethod
    def _render_greeks_history(
        account_history: list[dict[str, Any]],
    ) -> str:
        """Render a table of Greeks exposure from account snapshots.

        If the account snapshots do not carry per-snapshot Greek
        values, a notice is displayed instead.

        Parameters
        ----------
        account_history:
            List of account snapshot dicts.  Expected optional keys:
            ``portfolio_delta``, ``portfolio_gamma``,
            ``portfolio_theta``, ``portfolio_vega``.

        Returns
        -------
        str
            HTML ``<table>`` fragment.
        """
        has_greeks = any("portfolio_delta" in snap for snap in account_history)
        if not has_greeks or not account_history:
            return (
                "<p>Greeks exposure data is tracked via Prometheus "
                "metrics. View the Grafana dashboard for time-series "
                "charts of portfolio delta, gamma, theta, and "
                "vega.</p>"
            )

        rows = ""
        for snap in account_history[-30:]:
            ts = snap.get("timestamp", "")
            if hasattr(ts, "strftime"):
                ts = ts.strftime("%Y-%m-%d %H:%M")
            delta = snap.get("portfolio_delta", 0.0)
            gamma = snap.get("portfolio_gamma", 0.0)
            theta = snap.get("portfolio_theta", 0.0)
            vega = snap.get("portfolio_vega", 0.0)
            rows += (
                f"<tr>"
                f"<td>{ts}</td>"
                f"<td>{float(delta):+.4f}</td>"
                f"<td>{float(gamma):+.6f}</td>"
                f"<td>{float(theta):+.2f}</td>"
                f"<td>{float(vega):+.2f}</td>"
                f"</tr>\n"
            )

        return (
            "<table>\n"
            "<thead><tr>"
            "<th>Timestamp</th>"
            "<th>Delta</th>"
            "<th>Gamma</th>"
            "<th>Theta</th>"
            "<th>Vega</th>"
            "</tr></thead>\n"
            f"<tbody>\n{rows}</tbody>\n"
            "</table>"
        )
