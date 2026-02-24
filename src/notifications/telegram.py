"""Telegram bot for trade notifications and system control.

Provides real-time trade alerts, daily summaries, circuit-breaker
notifications, and interactive command handlers (``/status``,
``/portfolio``, ``/start``, ``/stop``, ``/kill``, ``/help``) via the
Telegram Bot API.

Usage::

    from src.notifications.telegram import TelegramNotifier

    notifier = TelegramNotifier(
        bot_token="123456:ABC-...",
        chat_id="-100123456789",
    )
    await notifier.start()
    await notifier.send_system_alert("TITAN ONLINE", severity="INFO")
    await notifier.stop()
"""

from __future__ import annotations

import asyncio
import contextlib
import re
from typing import TYPE_CHECKING, Any, Protocol

from pydantic import BaseModel, Field
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
)

from src.utils.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Callable

    import structlog
    from telegram import Update

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Telegram limits.
_MAX_MESSAGE_LENGTH: int = 4096
_MAX_MESSAGES_PER_SECOND: int = 30

# Emojis for message formatting.
_EMOJI_GREEN: str = "\U0001f7e2"  # Green circle
_EMOJI_RED: str = "\U0001f534"  # Red circle
_EMOJI_WARNING: str = "\u26a0\ufe0f"  # Warning sign
_EMOJI_SIREN: str = "\U0001f6a8"  # Rotating light
_EMOJI_CHART: str = "\U0001f4ca"  # Chart
_EMOJI_ROBOT: str = "\U0001f916"  # Robot
_EMOJI_STOP: str = "\U0001f6d1"  # Stop sign
_EMOJI_INFO: str = "\u2139\ufe0f"  # Information
_EMOJI_MONEY: str = "\U0001f4b0"  # Money bag
_EMOJI_SKULL: str = "\U0001f480"  # Skull (emergency)
_EMOJI_PAUSE: str = "\u23f8\ufe0f"  # Pause
_EMOJI_PLAY: str = "\u25b6\ufe0f"  # Play
_EMOJI_PORTFOLIO: str = "\U0001f4bc"  # Briefcase
_EMOJI_CHECK: str = "\u2705"  # Check mark
_EMOJI_CLOCK: str = "\U0001f552"  # Clock

# Severity-to-emoji mapping.
_SEVERITY_EMOJI: dict[str, str] = {
    "INFO": _EMOJI_INFO,
    "WARNING": _EMOJI_WARNING,
    "ERROR": _EMOJI_SIREN,
    "CRITICAL": _EMOJI_SKULL,
}

# Characters that must be escaped in MarkdownV2.
_MARKDOWNV2_SPECIAL_CHARS: str = r"_*[]()~`>#+-=|{}.!"


class SystemStateProvider(Protocol):
    """Protocol for the callable that returns current system state."""

    def __call__(self) -> dict[str, Any]: ...


class TradeEntryMessage(BaseModel):
    """Structured data for a trade entry notification."""

    ticker: str = Field(description="Underlying ticker symbol")
    strategy: str = Field(description="Strategy name")
    direction: str = Field(description="LONG or SHORT")
    quantity: int = Field(description="Number of contracts")
    entry_price: float = Field(description="Net entry price per unit")
    max_profit: float = Field(description="Maximum profit in USD")
    max_loss: float = Field(description="Maximum loss in USD")
    ml_confidence: float = Field(description="ML ensemble confidence score 0.0-1.0")
    regime: str = Field(description="Current market regime")


class TradeExitMessage(BaseModel):
    """Structured data for a trade exit notification."""

    ticker: str = Field(description="Underlying ticker symbol")
    strategy: str = Field(description="Strategy name")
    direction: str = Field(description="LONG or SHORT")
    quantity: int = Field(description="Number of contracts")
    entry_price: float = Field(description="Net entry price per unit")
    exit_price: float = Field(description="Net exit price per unit")
    realized_pnl: float = Field(description="Realized P&L in USD")
    commission: float = Field(default=0.0, description="Total commission in USD")
    hold_days: int = Field(default=0, description="Number of days position was held")


class DailySummary(BaseModel):
    """Structured data for the end-of-day summary notification."""

    total_trades: int = Field(description="Total trades executed today")
    winners: int = Field(description="Winning trades today")
    losers: int = Field(description="Losing trades today")
    daily_pnl: float = Field(description="Net P&L for the day")
    weekly_pnl: float = Field(description="Running weekly P&L")
    monthly_pnl: float = Field(description="Running monthly P&L")
    current_drawdown_pct: float = Field(description="Current drawdown as a percentage")
    circuit_breaker_level: str = Field(description="Current circuit breaker level")
    regime: str = Field(description="Current market regime")
    open_positions: int = Field(description="Number of open positions")


class TelegramNotifier:
    """Telegram bot for Project Titan trade notifications and control.

    Sends formatted trade alerts, daily summaries, circuit-breaker
    warnings, and general system alerts.  Also registers interactive
    command handlers so operators can query system state in real time.

    Args:
        bot_token: Telegram Bot API token from @BotFather.
        chat_id: Target chat or channel ID for outgoing messages.
    """

    def __init__(self, bot_token: str, chat_id: str) -> None:
        self._bot_token: str = bot_token
        self._chat_id: str = chat_id

        self._log: structlog.stdlib.BoundLogger = get_logger("notifications.telegram")

        # python-telegram-bot Application (initialized in start()).
        self._app: Application | None = None  # type: ignore[type-arg]

        # Rate-limiting queue: (coroutine, future) pairs.
        self._send_queue: asyncio.Queue[tuple[str, asyncio.Future[None]]] = (
            asyncio.Queue()
        )
        self._sender_task: asyncio.Task[None] | None = None

        # System state provider injected by the main application.
        self._state_provider: SystemStateProvider | None = None

        # Emergency kill flag.
        self._kill_flag: bool = False

        # Kill callback — injected by the main application so /kill
        # actually triggers TitanApplication.request_kill().
        self._kill_callback: Callable[[], None] | None = None

        # Pause/resume callbacks — injected by main application.
        self._pause_callback: Callable[[], None] | None = None
        self._resume_callback: Callable[[], None] | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def kill_requested(self) -> bool:
        """Return True if an operator issued /kill CONFIRM."""
        return self._kill_flag

    def set_system_state(self, provider: SystemStateProvider) -> None:
        """Inject a callable that returns the current system state dict.

        The provider is invoked by command handlers (``/status``,
        ``/positions``) to build their responses.

        Args:
            provider: A callable returning ``dict[str, Any]`` with keys
                such as ``connected``, ``regime``, ``positions``,
                ``daily_pnl``, ``circuit_breaker_level``.
        """
        self._state_provider = provider

    def set_kill_callback(self, callback: Callable[[], None]) -> None:
        """Inject a callback invoked when /kill CONFIRM is received.

        Args:
            callback: Typically ``TitanApplication.request_kill`` which
                sets the shutdown event and kill flag.
        """
        self._kill_callback = callback

    def set_pause_callback(self, callback: Callable[[], None]) -> None:
        """Inject a callback invoked when /stop is received.

        Args:
            callback: Typically ``TitanApplication.pause_trading`` which
                sets the trading-paused flag.
        """
        self._pause_callback = callback

    def set_resume_callback(self, callback: Callable[[], None]) -> None:
        """Inject a callback invoked when /start is received.

        Args:
            callback: Typically ``TitanApplication.resume_trading`` which
                clears the trading-paused flag.
        """
        self._resume_callback = callback

    async def start(self) -> None:
        """Initialize the bot application, register commands, and begin
        polling for incoming updates."""
        self._log.info("starting_telegram_bot")

        builder = Application.builder().token(self._bot_token)
        self._app = builder.build()

        # Register command handlers.
        self._app.add_handler(CommandHandler("status", self._handle_status))
        self._app.add_handler(CommandHandler("positions", self._handle_positions))
        self._app.add_handler(CommandHandler("portfolio", self._handle_portfolio))
        self._app.add_handler(CommandHandler("start", self._handle_start))
        self._app.add_handler(CommandHandler("stop", self._handle_stop))
        self._app.add_handler(CommandHandler("kill", self._handle_kill))
        self._app.add_handler(CommandHandler("help", self._handle_help))

        # Start the background message sender for rate limiting.
        self._sender_task = asyncio.create_task(
            self._rate_limited_sender(),
            name="titan-telegram-sender",
        )

        # Initialize and start polling in a non-blocking way.
        await self._app.initialize()
        await self._app.start()
        await self._app.updater.start_polling(  # type: ignore[union-attr]
            drop_pending_updates=True,
        )

        self._log.info("telegram_bot_started")

    async def stop(self) -> None:
        """Stop the bot and background sender task gracefully."""
        self._log.info("stopping_telegram_bot")

        # Cancel the background sender.
        if self._sender_task is not None and not self._sender_task.done():
            self._sender_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._sender_task
            self._sender_task = None

        # Stop the telegram application.
        if self._app is not None:
            if self._app.updater and self._app.updater.running:
                await self._app.updater.stop()
            if self._app.running:
                await self._app.stop()
            await self._app.shutdown()
            self._app = None

        self._log.info("telegram_bot_stopped")

    async def send_trade_entry(self, trade: dict[str, Any]) -> None:
        """Send a formatted trade entry notification.

        Args:
            trade: Dictionary with trade details. Expected keys:
                ``ticker``, ``strategy``, ``direction``, ``quantity``,
                ``entry_price``, ``max_profit``, ``max_loss``,
                ``ml_confidence``, ``regime``.
        """
        entry = TradeEntryMessage(**trade)
        max_loss_abs = abs(entry.max_loss)
        risk_reward = (
            round(entry.max_profit / max_loss_abs, 2) if max_loss_abs > 0 else 0.0
        )

        lines = [
            f"{_EMOJI_GREEN} NEW TRADE OPENED",
            "",
            f"Ticker: {entry.ticker} | Strategy: {entry.strategy}",
            (f"Direction: {entry.direction} | Quantity: {entry.quantity}"),
            (
                f"Entry Price: {self._format_currency(entry.entry_price)}"
                f" | Max Profit: "
                f"{self._format_currency(entry.max_profit)}"
            ),
            (f"Max Loss: {self._format_currency(entry.max_loss)} | R:R: {risk_reward}"),
            (f"ML Confidence: {entry.ml_confidence:.2f} | Regime: {entry.regime}"),
        ]

        await self._enqueue_message("\n".join(lines))
        self._log.info(
            "trade_entry_sent",
            ticker=entry.ticker,
            strategy=entry.strategy,
        )

    async def send_trade_exit(self, trade: dict[str, Any]) -> None:
        """Send a formatted trade exit notification with P&L.

        Args:
            trade: Dictionary with trade details. Expected keys:
                ``ticker``, ``strategy``, ``direction``, ``quantity``,
                ``entry_price``, ``exit_price``, ``realized_pnl``,
                ``commission`` (optional), ``hold_days`` (optional).
        """
        exit_msg = TradeExitMessage(**trade)
        is_winner = exit_msg.realized_pnl >= 0
        pnl_emoji = _EMOJI_GREEN if is_winner else _EMOJI_RED
        result_label = "WIN" if is_winner else "LOSS"

        lines = [
            f"{pnl_emoji} TRADE CLOSED — {result_label}",
            "",
            f"Ticker: {exit_msg.ticker} | Strategy: {exit_msg.strategy}",
            (f"Direction: {exit_msg.direction} | Quantity: {exit_msg.quantity}"),
            (
                f"Entry: "
                f"{self._format_currency(exit_msg.entry_price)}"
                f" -> Exit: "
                f"{self._format_currency(exit_msg.exit_price)}"
            ),
            (
                f"P&L: "
                f"{self._format_currency(exit_msg.realized_pnl)}"
                f" | Commission: "
                f"{self._format_currency(exit_msg.commission)}"
            ),
            f"Hold Time: {exit_msg.hold_days} day(s)",
        ]

        await self._enqueue_message("\n".join(lines))
        self._log.info(
            "trade_exit_sent",
            ticker=exit_msg.ticker,
            realized_pnl=exit_msg.realized_pnl,
        )

    async def send_daily_summary(self, summary: dict[str, Any]) -> None:
        """Send end-of-day performance summary.

        Args:
            summary: Dictionary with summary fields. Expected keys:
                ``total_trades``, ``winners``, ``losers``,
                ``daily_pnl``, ``weekly_pnl``, ``monthly_pnl``,
                ``current_drawdown_pct``, ``circuit_breaker_level``,
                ``regime``, ``open_positions``.
        """
        data = DailySummary(**summary)
        win_rate = (
            round(data.winners / data.total_trades * 100, 1)
            if data.total_trades > 0
            else 0.0
        )

        daily_emoji = _EMOJI_GREEN if data.daily_pnl >= 0 else _EMOJI_RED

        lines = [
            f"{_EMOJI_CHART} DAILY SUMMARY",
            "",
            (
                f"Trades: {data.total_trades}"
                f" | W: {data.winners}"
                f" | L: {data.losers}"
                f" | Win Rate: {win_rate}%"
            ),
            (f"{daily_emoji} Daily P&L: {self._format_currency(data.daily_pnl)}"),
            (
                f"Weekly P&L: "
                f"{self._format_currency(data.weekly_pnl)}"
                f" | Monthly: "
                f"{self._format_currency(data.monthly_pnl)}"
            ),
            (
                f"Drawdown: {data.current_drawdown_pct:.2%}"
                f" | CB Level: {data.circuit_breaker_level}"
            ),
            (f"Regime: {data.regime} | Open Positions: {data.open_positions}"),
        ]

        await self._enqueue_message("\n".join(lines))
        self._log.info(
            "daily_summary_sent",
            daily_pnl=data.daily_pnl,
            total_trades=data.total_trades,
        )

    async def send_circuit_breaker_alert(
        self, level: str, details: dict[str, Any]
    ) -> None:
        """Send an urgent circuit-breaker trigger notification.

        Args:
            level: The triggered level (e.g. ``CAUTION``, ``WARNING``,
                ``HALT``, ``EMERGENCY``).
            details: Dict with ``drawdown_pct``, ``daily_pnl``,
                ``recommended_action``, and any other relevant fields.
        """
        drawdown = details.get("drawdown_pct", 0.0)
        daily_pnl = details.get("daily_pnl", 0.0)
        action = details.get("recommended_action", "Review and assess.")

        lines = [
            f"{_EMOJI_SIREN} CIRCUIT BREAKER TRIGGERED",
            "",
            f"Level: {level}",
            f"Drawdown: {drawdown:.2%}",
            f"Daily P&L: {self._format_currency(daily_pnl)}",
            f"Action: {action}",
        ]

        await self._enqueue_message("\n".join(lines))
        self._log.warning(
            "circuit_breaker_alert_sent",
            level=level,
            drawdown_pct=drawdown,
        )

    async def send_system_alert(self, message: str, severity: str = "INFO") -> None:
        """Send a general system alert message.

        Args:
            message: The alert text.
            severity: One of ``INFO``, ``WARNING``, ``ERROR``,
                ``CRITICAL``.
        """
        emoji = _SEVERITY_EMOJI.get(severity.upper(), _EMOJI_INFO)
        text = f"{emoji} [{severity.upper()}] {message}"

        await self._enqueue_message(text)
        self._log.info(
            "system_alert_sent",
            severity=severity,
            message=message,
        )

    async def send_startup_summary(self, state: dict[str, Any]) -> None:
        """Send a rich startup notification with account and portfolio details.

        Called once during application startup after all subsystems are
        initialized and account data is available.

        Args:
            state: Dictionary with keys: ``trading_mode``, ``net_liquidation``,
                ``buying_power``, ``excess_liquidity``, ``daily_pnl``,
                ``unrealized_pnl``, ``open_positions``, ``positions``,
                ``circuit_breaker_level``, ``regime``, ``tickers_count``.
        """
        mode = state.get("trading_mode", "unknown").upper()
        mode_emoji = _EMOJI_WARNING if mode == "LIVE" else _EMOJI_INFO

        fmt = self._format_currency
        nl = fmt(state.get("net_liquidation", 0.0))
        bp = fmt(state.get("buying_power", 0.0))
        el = fmt(state.get("excess_liquidity", 0.0))
        dp = fmt(state.get("daily_pnl", 0.0))
        up = fmt(state.get("unrealized_pnl", 0.0))

        lines = [
            f"{_EMOJI_ROBOT} TITAN ONLINE {mode_emoji} [{mode}]",
            "",
            f"{_EMOJI_PORTFOLIO} ACCOUNT",
            f"  Net Liq:    {nl}",
            f"  Buying Pwr: {bp}",
            f"  Excess Liq: {el}",
            "",
            f"{_EMOJI_CHART} P&L",
            f"  Daily P&L:  {dp}",
            f"  Unrealized: {up}",
            "",
            f"{_EMOJI_INFO} SYSTEM",
            f"  Circuit Breaker: {state.get('circuit_breaker_level', 'NORMAL')}",
            f"  Regime:          {state.get('regime', 'unknown')}",
            f"  Tickers:         {state.get('tickers_count', 0)}",
            f"  Trading:  {'PAUSED' if state.get('trading_paused') else 'ACTIVE'}",
        ]

        # Append position summary
        positions: list[dict[str, Any]] = state.get("positions", [])
        open_count = state.get("open_positions", len(positions))
        lines.append("")
        if positions:
            lines.append(f"{_EMOJI_MONEY} POSITIONS ({open_count})")
            for pos in positions:
                pnl = pos.get("unrealized_pnl", 0.0)
                pnl_emoji = _EMOJI_GREEN if pnl >= 0 else _EMOJI_RED
                ticker = pos.get("ticker", "???")
                strategy = pos.get("strategy", pos.get("sec_type", "???"))
                qty = pos.get("quantity", 0)
                lines.append(
                    f"  {pnl_emoji} {ticker} | {strategy}"
                    f" | Qty: {qty}"
                    f" | P&L: {self._format_currency(pnl)}"
                )
        else:
            lines.append(f"{_EMOJI_INFO} No open positions")

        lines.append("")
        lines.append(f"{_EMOJI_CHECK} All systems operational")

        await self._enqueue_message("\n".join(lines))
        self._log.info("startup_summary_sent")

    # ------------------------------------------------------------------
    # Command handlers
    # ------------------------------------------------------------------

    async def _handle_status(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> None:
        """Handle the /status command.

        Returns current system state: connectivity, regime, account
        summary, P&L, circuit-breaker level, and trading status.
        """
        state = self._get_system_state()

        connected = state.get("connected", False)
        conn_icon = _EMOJI_GREEN if connected else _EMOJI_RED

        trading_paused = state.get("trading_paused", False)
        trading_label = (
            f"{_EMOJI_PAUSE} PAUSED" if trading_paused else f"{_EMOJI_PLAY} ACTIVE"
        )

        cb_level = state.get("circuit_breaker_level", "NORMAL")
        cb_icon = _EMOJI_GREEN if cb_level == "NORMAL" else _EMOJI_WARNING
        if cb_level in ("HALT", "EMERGENCY"):
            cb_icon = _EMOJI_SIREN

        fmt = self._format_currency
        dp = fmt(state.get("daily_pnl", 0.0))
        up = fmt(state.get("unrealized_pnl", 0.0))
        nl = fmt(state.get("net_liquidation", 0.0))
        bp = fmt(state.get("buying_power", 0.0))
        pos_cur = state.get("open_positions", 0)
        pos_max = state.get("max_positions", 8)

        lines = [
            f"{_EMOJI_ROBOT} TITAN STATUS",
            "",
            f"{conn_icon} Connected: {'Yes' if connected else 'No'}",
            f"  Mode:    {state.get('trading_mode', 'unknown').upper()}",
            f"  Trading: {trading_label}",
            "",
            f"{_EMOJI_CHART} P&L",
            f"  Daily:      {dp}",
            f"  Unrealized: {up}",
            "",
            f"{_EMOJI_PORTFOLIO} Account",
            f"  Net Liq:     {nl}",
            f"  Buying Power: {bp}",
            "",
            f"{cb_icon} Circuit Breaker: {cb_level}",
            f"  Regime: {state.get('regime', 'unknown')}",
            f"  Positions: {pos_cur}/{pos_max}",
        ]

        if self._kill_flag:
            lines.append(f"\n{_EMOJI_SKULL} KILL FLAG ACTIVE")

        await self._safe_reply(update, "\n".join(lines))

    async def _handle_positions(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> None:
        """Handle the /positions command.

        Returns a formatted list of all open positions with ticker,
        strategy, P&L, and days held.
        """
        state = self._get_system_state()
        positions: list[dict[str, Any]] = state.get("positions", [])

        if not positions:
            await self._safe_reply(update, f"{_EMOJI_INFO} No open positions.")
            return

        lines = [f"{_EMOJI_MONEY} OPEN POSITIONS ({len(positions)})"]
        lines.append("")

        for pos in positions:
            pnl = pos.get("unrealized_pnl", 0.0)
            pnl_emoji = _EMOJI_GREEN if pnl >= 0 else _EMOJI_RED
            lines.append(
                f"{pnl_emoji} {pos.get('ticker', '???')}"
                f" | {pos.get('strategy', '???')}"
                f" | P&L: {self._format_currency(pnl)}"
                f" | {pos.get('days_held', 0)}d"
            )

        await self._safe_reply(update, "\n".join(lines))

    async def _handle_portfolio(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> None:
        """Handle the /portfolio command.

        Returns full account overview with positions, P&L, and margin.
        Combines the information from /status and /positions into a
        single comprehensive view.
        """
        state = self._get_system_state()

        fmt = self._format_currency
        nl = fmt(state.get("net_liquidation", 0.0))
        bp = fmt(state.get("buying_power", 0.0))
        el = fmt(state.get("excess_liquidity", 0.0))
        mm = fmt(state.get("maint_margin", 0.0))

        lines = [
            f"{_EMOJI_PORTFOLIO} PORTFOLIO OVERVIEW",
            "",
            "ACCOUNT",
            f"  Net Liq:      {nl}",
            f"  Buying Pwr:   {bp}",
            f"  Excess Liq:   {el}",
            f"  Maint Margin: {mm}",
            "",
            "P&L",
            f"  Daily:      {self._format_currency(state.get('daily_pnl', 0.0))}",
            f"  Unrealized: {self._format_currency(state.get('unrealized_pnl', 0.0))}",
            f"  Realized:   {self._format_currency(state.get('realized_pnl', 0.0))}",
        ]

        positions: list[dict[str, Any]] = state.get("positions", [])
        lines.append("")
        if positions:
            total_unrealized = 0.0
            lines.append(f"POSITIONS ({len(positions)})")
            for pos in positions:
                pnl = pos.get("unrealized_pnl", 0.0)
                total_unrealized += pnl
                pnl_emoji = _EMOJI_GREEN if pnl >= 0 else _EMOJI_RED
                ticker = pos.get("ticker", "???")
                strategy = pos.get("strategy", pos.get("sec_type", "???"))
                qty = pos.get("quantity", 0)
                mkt_val = pos.get("market_value", 0.0)
                lines.append(
                    f"  {pnl_emoji} {ticker} | {strategy}"
                    f" | Qty: {qty}"
                    f" | Val: {self._format_currency(mkt_val)}"
                    f" | P&L: {self._format_currency(pnl)}"
                )
            lines.append(
                f"\n  Total Unrealized: {self._format_currency(total_unrealized)}"
            )
        else:
            lines.append(f"{_EMOJI_INFO} No open positions")

        lines.append("")
        cb_level = state.get("circuit_breaker_level", "NORMAL")
        lines.append(f"CB: {cb_level} | Regime: {state.get('regime', 'unknown')}")

        await self._safe_reply(update, "\n".join(lines))

    async def _handle_start(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> None:
        """Handle the /start command — resume trading.

        Clears the trading-paused flag so scheduled scans will execute
        trade proposals again.
        """
        if self._resume_callback is not None:
            try:
                self._resume_callback()
                self._log.info("trading_resumed_via_telegram")
                await self._safe_reply(
                    update,
                    f"{_EMOJI_PLAY} Trading RESUMED\nScans will execute normally.",
                )
            except Exception:
                self._log.exception("resume_callback_failed")
                await self._safe_reply(
                    update,
                    f"{_EMOJI_RED} Failed to resume trading. Check logs.",
                )
        else:
            await self._safe_reply(
                update,
                f"{_EMOJI_WARNING} Resume callback not wired."
                " System may not support pause/resume.",
            )

    async def _handle_stop(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> None:
        """Handle the /stop command — pause trading.

        Sets the trading-paused flag so scheduled scans will skip trade
        proposals. Does NOT cancel existing positions or shut down the
        system. Use /kill CONFIRM for emergency shutdown.
        """
        if self._pause_callback is not None:
            try:
                self._pause_callback()
                self._log.info("trading_paused_via_telegram")
                await self._safe_reply(
                    update,
                    (
                        f"{_EMOJI_PAUSE} Trading PAUSED\n"
                        f"Existing positions are untouched.\n"
                        f"Risk monitoring continues.\n"
                        f"Use /start to resume."
                    ),
                )
            except Exception:
                self._log.exception("pause_callback_failed")
                await self._safe_reply(
                    update,
                    f"{_EMOJI_RED} Failed to pause trading. Check logs.",
                )
        else:
            await self._safe_reply(
                update,
                f"{_EMOJI_WARNING} Pause callback not wired."
                " System may not support pause/resume.",
            )

    async def _handle_kill(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> None:
        """Handle the /kill command (emergency stop).

        Requires the operator to type ``/kill CONFIRM`` to activate.
        Sets the kill flag that the main application loop checks.
        """
        if not update.message:
            return

        raw_text = update.message.text or ""
        # Extract arguments after /kill.
        parts = raw_text.strip().split(maxsplit=1)
        confirmation = parts[1] if len(parts) > 1 else ""

        if confirmation != "CONFIRM":
            await self._safe_reply(
                update,
                (
                    f"{_EMOJI_WARNING} Emergency stop requires "
                    f"confirmation.\n"
                    f"Type: /kill CONFIRM"
                ),
            )
            return

        self._kill_flag = True
        self._log.warning("kill_switch_activated")

        # Invoke the application-level kill to trigger actual shutdown.
        if self._kill_callback is not None:
            try:
                self._kill_callback()
                self._log.info("kill_callback_invoked")
            except Exception:
                self._log.exception("kill_callback_failed")
        else:
            self._log.warning(
                "kill_callback_not_set",
                reason="kill flag set but no shutdown callback wired",
            )

        await self._safe_reply(
            update,
            (
                f"{_EMOJI_STOP} EMERGENCY STOP ACTIVATED\n"
                f"Kill flag set. The system will halt trading and "
                f"cancel pending orders."
            ),
        )

    async def _handle_help(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> None:
        """Handle the /help command.

        Lists all available bot commands.
        """
        lines = [
            f"{_EMOJI_ROBOT} TITAN BOT COMMANDS",
            "",
            "/status     — System status, P&L, connection",
            "/portfolio  — Full account & positions overview",
            "/positions  — List open positions with P&L",
            "/start      — Resume trading (after /stop)",
            "/stop       — Pause new trades (keeps positions)",
            "/kill CONFIRM — Emergency shutdown",
            "/help       — Show this message",
        ]

        await self._safe_reply(update, "\n".join(lines))

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_system_state(self) -> dict[str, Any]:
        """Retrieve system state from the injected provider.

        Returns an empty dict if no provider has been set.
        """
        if self._state_provider is None:
            return {}
        try:
            return self._state_provider()
        except Exception:
            self._log.exception("system_state_provider_error")
            return {}

    async def _enqueue_message(self, text: str) -> None:
        """Enqueue a message for rate-limited delivery.

        Args:
            text: The message text (plain text, not Markdown).
        """
        text = self._truncate_message(text)
        future: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        await self._send_queue.put((text, future))
        # Fire-and-forget: callers do not wait for delivery.

    async def _rate_limited_sender(self) -> None:
        """Background task that drains the send queue respecting
        Telegram's rate limit of 30 messages per second."""
        interval = 1.0 / _MAX_MESSAGES_PER_SECOND
        while True:
            text, future = await self._send_queue.get()
            try:
                await self._send_raw_message(text)
                if not future.done():
                    future.set_result(None)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if not future.done():
                    future.set_exception(exc)
            finally:
                self._send_queue.task_done()
            # Respect rate limit between messages.
            await asyncio.sleep(interval)

    async def _send_raw_message(self, text: str) -> None:
        """Send a plain-text message via the Telegram Bot API.

        Never raises — all errors are logged and swallowed so that a
        notification failure cannot crash the trading system.

        Args:
            text: Plain-text message to send.
        """
        if self._app is None:
            self._log.warning("send_skipped", reason="bot_not_started")
            return

        try:
            await self._app.bot.send_message(
                chat_id=self._chat_id,
                text=text,
            )
        except Exception:
            self._log.exception(
                "telegram_send_failed",
                chat_id=self._chat_id,
                text_length=len(text),
            )

    async def _safe_reply(self, update: Update, text: str) -> None:
        """Reply to an incoming update, swallowing errors.

        Args:
            update: The incoming Telegram update.
            text: Reply text.
        """
        if update.message is None:
            return

        text = self._truncate_message(text)
        try:
            await update.message.reply_text(text)
        except Exception:
            self._log.exception(
                "telegram_reply_failed",
                text_length=len(text),
            )

    @staticmethod
    def _format_currency(amount: float) -> str:
        """Format a numeric amount as a USD currency string.

        Positive amounts are shown with a ``+`` prefix, negatives with
        ``-``.

        Args:
            amount: The dollar amount to format.

        Returns:
            A string like ``+$1,234.56`` or ``-$789.00``.
        """
        sign = "+" if amount >= 0 else "-"
        return f"{sign}${abs(amount):,.2f}"

    @staticmethod
    def _truncate_message(msg: str, max_len: int = _MAX_MESSAGE_LENGTH) -> str:
        """Truncate a message to fit within Telegram's size limit.

        If the message exceeds *max_len*, it is cut and an ellipsis
        marker is appended.

        Args:
            msg: The raw message text.
            max_len: Maximum allowed length (default 4096).

        Returns:
            The (possibly truncated) message string.
        """
        if len(msg) <= max_len:
            return msg
        suffix = "\n... (truncated)"
        return msg[: max_len - len(suffix)] + suffix

    @staticmethod
    def _escape_markdown(text: str) -> str:
        """Escape special characters for Telegram MarkdownV2 format.

        Args:
            text: Raw text that may contain MarkdownV2 special chars.

        Returns:
            The escaped string safe for MarkdownV2 ``parse_mode``.
        """
        return re.sub(
            r"([" + re.escape(_MARKDOWNV2_SPECIAL_CHARS) + r"])",
            r"\\\1",
            text,
        )
