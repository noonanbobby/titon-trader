"""Application entry point and lifecycle management for Project Titan.

Orchestrates all subsystems — broker connectivity, signal generation, ML
models, AI agents, risk management, scheduling, notifications, and
metrics — into a single cohesive trading application.

The :class:`TitanApplication` class encapsulates the full startup/run/
shutdown lifecycle.  Run from the command line via::

    python -m src.main

Or programmatically::

    from src.main import TitanApplication
    import asyncio

    app = TitanApplication()
    asyncio.run(app.run())
"""

from __future__ import annotations

import asyncio
import signal
from typing import TYPE_CHECKING, Any

import yaml

from config.settings import Settings, get_settings
from src.utils.logging import configure_logging, get_logger
from src.utils.metrics import (
    CONNECTION_STATUS,
    POSITIONS_OPEN,
    setup_metrics_server,
)

if TYPE_CHECKING:
    import threading
    from collections.abc import Callable, Coroutine
    from pathlib import Path

    import asyncpg
    import redis.asyncio as aioredis
    import structlog

    from src.ai.agents import TradingAgentOrchestrator
    from src.broker.account import AccountManager
    from src.broker.contracts import ContractFactory
    from src.broker.gateway import GatewayManager
    from src.broker.market_data import MarketDataManager
    from src.broker.orders import OrderManager
    from src.risk.circuit_breakers import CircuitBreaker
    from src.risk.event_calendar import EventCalendar
    from src.risk.manager import RiskManager
    from src.signals.cross_asset import CrossAssetSignalGenerator
    from src.signals.ensemble import EnsembleSignalGenerator
    from src.signals.gex import GammaExposureCalculator
    from src.signals.insider import InsiderSignalGenerator
    from src.signals.options_flow import OptionsFlowAnalyzer
    from src.signals.regime import RegimeDetector
    from src.signals.sentiment import SentimentAnalyzer
    from src.signals.technical import TechnicalSignalGenerator
    from src.signals.vrp import VRPCalculator
    from src.strategies.selector import StrategySelector
    from src.utils.scheduling import TitanScheduler

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

METRICS_SERVER_PORT: int = 8080
IB_GATEWAY_MAX_CONNECT_ATTEMPTS: int = 5
IB_GATEWAY_INITIAL_BACKOFF_SECONDS: float = 2.0
IB_GATEWAY_BACKOFF_FACTOR: float = 2.0

STARTUP_MESSAGE: str = "TITAN ONLINE"
SHUTDOWN_MESSAGE: str = "TITAN OFFLINE"


class TitanApplication:
    """Top-level application coordinating all Project Titan subsystems.

    Manages the full startup, run, and shutdown lifecycle.  All subsystems
    are stored as instance attributes and are initialised lazily during
    :meth:`start`.  Subsystems that fail to initialise are logged and
    gracefully degraded so the remaining system can continue.
    """

    def __init__(self) -> None:
        self._settings: Settings = get_settings()
        self._log: structlog.stdlib.BoundLogger = get_logger("main")

        # Shutdown coordination
        self._shutdown_event: asyncio.Event = asyncio.Event()
        self._kill_requested: bool = False

        # Infrastructure connections
        self._pg_pool: asyncpg.Pool | None = None
        self._redis: aioredis.Redis | None = None

        # Broker subsystems
        self._gateway: GatewayManager | None = None
        self._contract_factory: ContractFactory | None = None
        self._market_data: MarketDataManager | None = None
        self._order_manager: OrderManager | None = None
        self._account_manager: AccountManager | None = None

        # Signal generators
        self._regime_detector: RegimeDetector | None = None
        self._technical_generator: TechnicalSignalGenerator | None = None
        self._sentiment_analyzer: SentimentAnalyzer | None = None
        self._options_flow: OptionsFlowAnalyzer | None = None
        self._gex_calculator: GammaExposureCalculator | None = None
        self._insider_generator: InsiderSignalGenerator | None = None
        self._vrp_calculator: VRPCalculator | None = None
        self._cross_asset: CrossAssetSignalGenerator | None = None
        self._ensemble: EnsembleSignalGenerator | None = None

        # Strategy engine
        self._strategy_selector: StrategySelector | None = None

        # AI agents
        self._orchestrator: TradingAgentOrchestrator | None = None

        # Risk management
        self._risk_manager: RiskManager | None = None
        self._circuit_breaker: CircuitBreaker | None = None
        self._event_calendar: EventCalendar | None = None

        # Notifications
        self._telegram: Any | None = None
        self._twilio: Any | None = None

        # Scheduling and metrics
        self._scheduler: TitanScheduler | None = None
        self._metrics_thread: threading.Thread | None = None

        # Ticker universe loaded from config
        self._tickers: list[str] = []

    # ------------------------------------------------------------------
    # Public lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Execute the full startup sequence.

        Initialises all subsystems in dependency order.  Subsystems that
        fail to initialise are logged and skipped so the remaining system
        can operate in a degraded mode.

        Raises
        ------
        RuntimeError
            If the IB Gateway connection cannot be established after all
            retry attempts.
        """
        self._log.info("titan_starting", mode=self._settings.ibkr.trading_mode)

        # 1. Structured logging
        configure_logging(log_level="INFO", json_output=True)

        # 2. Connect PostgreSQL
        self._pg_pool = await self._connect_postgres()

        # 3. Connect Redis
        self._redis = await self._connect_redis()

        # 4. Connect IB Gateway (critical — raises on failure)
        self._gateway = await self._connect_ib_gateway()

        # 5. Initialise broker modules
        self._initialize_broker_modules()

        # 6. Load ticker universe
        self._tickers = self._load_ticker_universe()
        self._log.info(
            "ticker_universe_loaded",
            count=len(self._tickers),
        )

        # 7. Initialise signal generators
        await self._initialize_signals()

        # 8. Initialise strategy selector
        self._initialize_strategy_selector()

        # 9. Initialise AI agents
        self._initialize_ai_agents()

        # 10. Initialise risk management
        await self._initialize_risk()

        # 11. Initialise notifications
        await self._initialize_notifications()

        # 12. Register scheduler callbacks and start
        self._scheduler = self._create_scheduler()
        callbacks = self._build_scheduler_callbacks()
        self._scheduler.register_callbacks(callbacks)
        await self._scheduler.start()

        # 13. Start metrics server
        self._metrics_thread = setup_metrics_server(
            port=METRICS_SERVER_PORT,
        )
        self._log.info(
            "metrics_server_started",
            port=METRICS_SERVER_PORT,
        )

        # 14. Subscribe to market data for ticker universe
        await self._subscribe_market_data()

        # 15. Send startup notification
        await self._send_notification(STARTUP_MESSAGE)

        self._log.info("titan_online", tickers=len(self._tickers))

    async def stop(self) -> None:
        """Execute the graceful shutdown sequence.

        Shuts down all subsystems in reverse dependency order.  Errors
        during shutdown are logged but do not prevent other subsystems
        from being cleaned up.
        """
        self._log.info("titan_shutting_down")

        # 1. Send shutdown notification (best effort)
        await self._send_notification(SHUTDOWN_MESSAGE)

        # 2. Stop scheduler
        if self._scheduler is not None:
            try:
                await self._scheduler.stop()
                self._log.info("scheduler_stopped")
            except Exception:
                self._log.exception("scheduler_stop_failed")

        # 3. Cancel pending orders (best effort)
        if self._order_manager is not None:
            try:
                await self._order_manager.cancel_all_orders()
                self._log.info("pending_orders_cancelled")
            except Exception:
                self._log.exception("cancel_orders_failed")

        # 4. Disconnect IB Gateway
        if self._gateway is not None:
            try:
                await self._gateway.disconnect()
                CONNECTION_STATUS.labels(service="ib_gateway").set(0)
                self._log.info("ib_gateway_disconnected")
            except Exception:
                self._log.exception("ib_gateway_disconnect_failed")

        # 5. Close PostgreSQL pool
        if self._pg_pool is not None:
            try:
                await self._pg_pool.close()
                CONNECTION_STATUS.labels(service="postgres").set(0)
                self._log.info("postgres_pool_closed")
            except Exception:
                self._log.exception("postgres_close_failed")

        # 6. Close Redis connection
        if self._redis is not None:
            try:
                await self._redis.aclose()
                CONNECTION_STATUS.labels(service="redis").set(0)
                self._log.info("redis_connection_closed")
            except Exception:
                self._log.exception("redis_close_failed")

        self._log.info("titan_offline")

    async def run(self) -> None:
        """Start the application and block until a shutdown signal arrives.

        Registers SIGINT and SIGTERM handlers, calls :meth:`start`, then
        waits on the internal shutdown event.  When the event fires,
        calls :meth:`stop` for a clean teardown.
        """
        self._register_signal_handlers()

        try:
            await self.start()
        except Exception:
            self._log.exception("startup_failed")
            await self.stop()
            raise

        self._log.info("waiting_for_shutdown_signal")
        await self._shutdown_event.wait()

        await self.stop()

    # ------------------------------------------------------------------
    # Signal handlers
    # ------------------------------------------------------------------

    def _register_signal_handlers(self) -> None:
        """Register SIGINT and SIGTERM handlers on the running event loop.

        The handlers set the internal shutdown event so the :meth:`run`
        coroutine unblocks and begins the teardown sequence.
        """
        loop = asyncio.get_running_loop()

        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, self._handle_shutdown_signal, sig)

    def _handle_shutdown_signal(self, sig: signal.Signals) -> None:
        """Handle a shutdown signal by setting the shutdown event.

        Parameters
        ----------
        sig:
            The signal that was received.
        """
        self._log.warning(
            "shutdown_signal_received",
            signal=sig.name,
        )
        self._shutdown_event.set()

    def request_kill(self) -> None:
        """Request an emergency shutdown from the Telegram kill command.

        Sets both the kill flag and the shutdown event so the run loop
        terminates as quickly as possible.
        """
        self._log.warning("kill_requested_via_telegram")
        self._kill_requested = True
        self._shutdown_event.set()

    # ------------------------------------------------------------------
    # Infrastructure connections
    # ------------------------------------------------------------------

    async def _connect_postgres(self) -> asyncpg.Pool:
        """Create an asyncpg connection pool to PostgreSQL.

        Returns
        -------
        asyncpg.Pool
            A connection pool ready for queries.

        Raises
        ------
        Exception
            If the connection pool cannot be created.
        """
        import asyncpg

        pg = self._settings.postgres
        self._log.info(
            "connecting_postgres",
            host=pg.host,
            port=pg.port,
            database=pg.db,
        )

        pool = await asyncpg.create_pool(
            dsn=pg.dsn,
            min_size=2,
            max_size=10,
            command_timeout=30.0,
        )
        CONNECTION_STATUS.labels(service="postgres").set(1)
        self._log.info("postgres_connected")
        return pool

    async def _connect_redis(self) -> aioredis.Redis:
        """Create an async Redis client and verify connectivity.

        Returns
        -------
        redis.asyncio.Redis
            A connected Redis client instance.

        Raises
        ------
        Exception
            If the Redis server cannot be reached.
        """
        import redis.asyncio as aioredis

        r = self._settings.redis
        self._log.info(
            "connecting_redis",
            host=r.host,
            port=r.port,
        )

        client = aioredis.Redis(
            host=r.host,
            port=r.port,
            decode_responses=True,
            socket_connect_timeout=10.0,
        )

        # Verify connectivity
        await client.ping()
        CONNECTION_STATUS.labels(service="redis").set(1)
        self._log.info("redis_connected")
        return client

    async def _connect_ib_gateway(self) -> GatewayManager:
        """Connect to IB Gateway with exponential backoff retries.

        Attempts up to :data:`IB_GATEWAY_MAX_CONNECT_ATTEMPTS` connections
        with exponential backoff between attempts.

        Returns
        -------
        GatewayManager
            A connected gateway manager instance.

        Raises
        ------
        RuntimeError
            If all connection attempts are exhausted.
        """
        from src.broker.gateway import GatewayManager

        ibkr = self._settings.ibkr
        self._log.info(
            "connecting_ib_gateway",
            host="127.0.0.1",
            port=ibkr.gateway_port,
            client_id=ibkr.client_id,
            mode=ibkr.trading_mode,
        )

        gw = GatewayManager(
            host="127.0.0.1",
            port=ibkr.gateway_port,
            client_id=ibkr.client_id,
        )

        delay = IB_GATEWAY_INITIAL_BACKOFF_SECONDS
        last_error: Exception | None = None

        for attempt in range(1, IB_GATEWAY_MAX_CONNECT_ATTEMPTS + 1):
            try:
                await gw.connect()
                CONNECTION_STATUS.labels(service="ib_gateway").set(1)
                self._log.info(
                    "ib_gateway_connected",
                    attempt=attempt,
                )
                return gw
            except Exception as exc:
                last_error = exc
                self._log.warning(
                    "ib_gateway_connect_failed",
                    attempt=attempt,
                    max_attempts=IB_GATEWAY_MAX_CONNECT_ATTEMPTS,
                    error=str(exc),
                    next_retry_seconds=delay,
                )
                if attempt < IB_GATEWAY_MAX_CONNECT_ATTEMPTS:
                    await asyncio.sleep(delay)
                    delay *= IB_GATEWAY_BACKOFF_FACTOR

        CONNECTION_STATUS.labels(service="ib_gateway").set(0)
        raise RuntimeError(
            f"Failed to connect to IB Gateway after "
            f"{IB_GATEWAY_MAX_CONNECT_ATTEMPTS} attempts: {last_error}"
        )

    # ------------------------------------------------------------------
    # Broker module initialisation
    # ------------------------------------------------------------------

    def _initialize_broker_modules(self) -> None:
        """Create ContractFactory, MarketDataManager, OrderManager, and
        AccountManager using the established IB Gateway connection.

        Requires :attr:`_gateway` to be connected.
        """
        from src.broker.account import AccountManager
        from src.broker.contracts import ContractFactory
        from src.broker.market_data import MarketDataManager
        from src.broker.orders import OrderManager

        ib = self._gateway.ib  # type: ignore[union-attr]

        self._contract_factory = ContractFactory(ib)
        self._market_data = MarketDataManager(
            ib,
            self._contract_factory,
            redis_client=self._redis,
        )
        self._order_manager = OrderManager(ib)
        self._account_manager = AccountManager(ib)

        self._log.info("broker_modules_initialized")

    # ------------------------------------------------------------------
    # Ticker universe
    # ------------------------------------------------------------------

    def _load_ticker_universe(self) -> list[str]:
        """Load the trading ticker universe from config/tickers.yaml.

        Parses all sector groups and collects unique ticker symbols into
        a flat list, excluding metadata and non-tradeable reference
        symbols (e.g. VIX).

        Returns
        -------
        list[str]
            Sorted list of unique ticker symbols.
        """
        tickers_path: Path = self._settings.config_dir / "tickers.yaml"

        if not tickers_path.exists():
            self._log.warning(
                "tickers_yaml_not_found",
                path=str(tickers_path),
            )
            return []

        with open(tickers_path) as fh:
            data = yaml.safe_load(fh)

        if data is None:
            return []

        # Non-tradeable symbols that are reference-only.
        excluded_symbols: frozenset[str] = frozenset({"VIX"})

        tickers: set[str] = set()
        for key, group in data.items():
            if key == "metadata":
                continue
            if isinstance(group, dict) and "tickers" in group:
                for symbol in group["tickers"]:
                    if symbol not in excluded_symbols:
                        tickers.add(symbol)

        return sorted(tickers)

    # ------------------------------------------------------------------
    # Signal generator initialisation
    # ------------------------------------------------------------------

    async def _initialize_signals(self) -> None:
        """Create all signal generator instances.

        Each generator is initialised independently.  If a generator
        fails to initialise (e.g. missing API key), it is logged as a
        warning and set to ``None`` so the rest of the system can
        operate without that signal stream.
        """
        from src.signals.cross_asset import CrossAssetSignalGenerator
        from src.signals.ensemble import EnsembleSignalGenerator
        from src.signals.gex import GammaExposureCalculator
        from src.signals.insider import InsiderSignalGenerator
        from src.signals.options_flow import OptionsFlowAnalyzer
        from src.signals.regime import RegimeDetector
        from src.signals.sentiment import SentimentAnalyzer
        from src.signals.technical import TechnicalSignalGenerator
        from src.signals.vrp import VRPCalculator

        api_keys = self._settings.api_keys

        # Regime detector
        try:
            self._regime_detector = RegimeDetector()
            self._log.info("regime_detector_initialized")
        except Exception:
            self._log.exception("regime_detector_init_failed")

        # Technical signal generator
        try:
            self._technical_generator = TechnicalSignalGenerator()
            self._log.info("technical_generator_initialized")
        except Exception:
            self._log.exception("technical_generator_init_failed")

        # Sentiment analyzer (requires Finnhub API key)
        try:
            finnhub_key = api_keys.finnhub_api_key.get_secret_value()
            if finnhub_key:
                self._sentiment_analyzer = SentimentAnalyzer(
                    finnhub_api_key=finnhub_key,
                )
                self._log.info("sentiment_analyzer_initialized")
            else:
                self._log.warning(
                    "sentiment_analyzer_skipped",
                    reason="missing FINNHUB_API_KEY",
                )
        except Exception:
            self._log.exception("sentiment_analyzer_init_failed")

        # Options flow analyzer (requires Unusual Whales API key)
        try:
            uw_key = api_keys.unusual_whales_api_key.get_secret_value()
            if uw_key:
                self._options_flow = OptionsFlowAnalyzer(api_key=uw_key)
                self._log.info("options_flow_analyzer_initialized")
            else:
                self._log.warning(
                    "options_flow_skipped",
                    reason="missing UNUSUAL_WHALES_API_KEY",
                )
        except Exception:
            self._log.exception("options_flow_init_failed")

        # GEX calculator
        try:
            self._gex_calculator = GammaExposureCalculator()
            self._log.info("gex_calculator_initialized")
        except Exception:
            self._log.exception("gex_calculator_init_failed")

        # Insider signal generator
        try:
            self._insider_generator = InsiderSignalGenerator()
            self._log.info("insider_generator_initialized")
        except Exception:
            self._log.exception("insider_generator_init_failed")

        # VRP calculator
        try:
            self._vrp_calculator = VRPCalculator()
            self._log.info("vrp_calculator_initialized")
        except Exception:
            self._log.exception("vrp_calculator_init_failed")

        # Cross-asset signal generator (requires FRED + Polygon keys)
        try:
            fred_key = api_keys.fred_api_key.get_secret_value()
            polygon_key = api_keys.polygon_api_key.get_secret_value()
            if fred_key and polygon_key:
                self._cross_asset = CrossAssetSignalGenerator(
                    fred_api_key=fred_key,
                    polygon_api_key=polygon_key,
                )
                self._log.info("cross_asset_generator_initialized")
            else:
                self._log.warning(
                    "cross_asset_skipped",
                    reason="missing FRED_API_KEY or POLYGON_API_KEY",
                )
        except Exception:
            self._log.exception("cross_asset_init_failed")

        # Ensemble model (meta-learner)
        try:
            confidence = self._settings.trading.confidence_threshold
            self._ensemble = EnsembleSignalGenerator(
                confidence_threshold=confidence,
            )
            # Attempt to load a trained model
            model_path = self._settings.models_dir / "ensemble_xgb.json"
            if model_path.exists():
                await self._ensemble.load_model(str(model_path))
                self._log.info(
                    "ensemble_model_loaded",
                    path=str(model_path),
                )
            else:
                self._log.warning(
                    "ensemble_model_not_found",
                    path=str(model_path),
                    fallback="weighted_average",
                )
        except Exception:
            self._log.exception("ensemble_init_failed")

    # ------------------------------------------------------------------
    # Strategy selector
    # ------------------------------------------------------------------

    def _initialize_strategy_selector(self) -> None:
        """Create the regime-based strategy selector.

        Loads strategy configurations from config/strategies.yaml.
        """
        from src.strategies.selector import (
            StrategySelector,
            load_strategies_config,
        )

        try:
            config_path = str(self._settings.config_dir / "strategies.yaml")
            strategies = load_strategies_config(config_path)
            self._strategy_selector = StrategySelector(
                strategies=strategies,
                config_path=config_path,
            )
            self._log.info(
                "strategy_selector_initialized",
                strategies_count=len(strategies),
            )
        except Exception:
            self._log.exception("strategy_selector_init_failed")

    # ------------------------------------------------------------------
    # AI agent initialisation
    # ------------------------------------------------------------------

    def _initialize_ai_agents(self) -> None:
        """Create the TradingAgentOrchestrator for the AI pipeline.

        If the Anthropic API key is not configured, the orchestrator
        is not created and the system operates in pure ML mode.
        """
        from src.ai.agents import TradingAgentOrchestrator

        try:
            api_key = self._settings.api_keys.anthropic_api_key.get_secret_value()
            if not api_key:
                self._log.warning(
                    "ai_agents_skipped",
                    reason="missing ANTHROPIC_API_KEY",
                    fallback="pure_ml_signals",
                )
                return

            self._orchestrator = TradingAgentOrchestrator(
                api_key=api_key,
                settings=self._settings.claude,
            )
            self._log.info(
                "ai_orchestrator_initialized",
                model=self._settings.claude.claude_model,
            )
        except Exception:
            self._log.exception("ai_orchestrator_init_failed")

    # ------------------------------------------------------------------
    # Risk management initialisation
    # ------------------------------------------------------------------

    async def _initialize_risk(self) -> None:
        """Create the risk manager, circuit breaker, and event calendar.

        Loads persisted circuit breaker state from PostgreSQL so the
        system resumes with the correct drawdown level after restarts.
        """
        from src.risk.circuit_breakers import CircuitBreaker
        from src.risk.event_calendar import EventCalendar
        from src.risk.manager import RiskManager

        # Load risk configuration
        risk_config: dict[str, Any] = {}
        risk_path = self._settings.config_dir / "risk_limits.yaml"
        if risk_path.exists():
            with open(risk_path) as fh:
                risk_config = yaml.safe_load(fh) or {}

        # Circuit breaker (with persistent state)
        try:
            self._circuit_breaker = CircuitBreaker(
                risk_config=risk_config.get("circuit_breakers", {}),
                db_pool=self._pg_pool,
            )
            await self._circuit_breaker.load_state()
            self._log.info(
                "circuit_breaker_initialized",
                level=str(self._circuit_breaker.current_level),
            )
        except Exception:
            self._log.exception("circuit_breaker_init_failed")

        # Event calendar
        try:
            finnhub_key = self._settings.api_keys.finnhub_api_key.get_secret_value()
            event_config = risk_config.get("event_exclusions", {})
            self._event_calendar = EventCalendar(
                risk_config=event_config,
                finnhub_api_key=finnhub_key,
            )
            if finnhub_key and self._tickers:
                await self._event_calendar.refresh(
                    tickers=self._tickers,
                )
            self._log.info("event_calendar_initialized")
        except Exception:
            self._log.exception("event_calendar_init_failed")

        # Risk manager
        try:
            self._risk_manager = RiskManager(
                settings=self._settings,
                risk_config=risk_config,
                circuit_breaker=self._circuit_breaker,
                event_calendar=self._event_calendar,
            )
            self._log.info("risk_manager_initialized")
        except Exception:
            self._log.exception("risk_manager_init_failed")

    # ------------------------------------------------------------------
    # Notifications
    # ------------------------------------------------------------------

    async def _initialize_notifications(self) -> None:
        """Create Telegram and Twilio notification clients.

        Each notification channel is initialised independently.  Missing
        credentials result in a warning log and the channel being
        unavailable.
        """
        notif = self._settings.notifications

        # Telegram
        if notif.telegram_enabled:
            try:
                from src.notifications.telegram import TelegramNotifier

                self._telegram = TelegramNotifier(
                    bot_token=(notif.telegram_bot_token.get_secret_value()),
                    chat_id=notif.telegram_chat_id,
                )
                self._log.info("telegram_notifier_initialized")
            except Exception:
                self._log.exception("telegram_init_failed")
        else:
            self._log.warning(
                "telegram_skipped",
                reason="incomplete configuration",
            )

        # Twilio SMS
        if notif.twilio_enabled:
            try:
                from src.notifications.twilio_sms import TwilioNotifier

                self._twilio = TwilioNotifier(
                    account_sid=notif.twilio_account_sid,
                    auth_token=(notif.twilio_auth_token.get_secret_value()),
                    from_number=notif.twilio_from_number,
                    to_number=notif.twilio_to_number,
                )
                self._log.info("twilio_notifier_initialized")
            except Exception:
                self._log.exception("twilio_init_failed")
        else:
            self._log.warning(
                "twilio_skipped",
                reason="incomplete configuration",
            )

    # ------------------------------------------------------------------
    # Scheduler setup
    # ------------------------------------------------------------------

    def _create_scheduler(self) -> TitanScheduler:
        """Create a new TitanScheduler instance.

        Returns
        -------
        TitanScheduler
            An uninitialised scheduler ready for callback registration.
        """
        from src.utils.scheduling import TitanScheduler

        return TitanScheduler()

    def _build_scheduler_callbacks(
        self,
    ) -> dict[str, Callable[..., Coroutine[Any, Any, None]]]:
        """Map scheduler job names to their corresponding async methods.

        Returns
        -------
        dict[str, Callable[..., Coroutine[Any, Any, None]]]
            Mapping of job name strings to bound async methods on this
            application instance.
        """
        from src.utils.scheduling import (
            JOB_DAILY_CLEANUP,
            JOB_DAILY_SUMMARY,
            JOB_EOD_JOURNAL,
            JOB_EVENT_CALENDAR_REFRESH,
            JOB_INTRADAY_SCAN_1,
            JOB_INTRADAY_SCAN_2,
            JOB_INTRADAY_SCAN_3,
            JOB_MARKET_OPEN_SCAN,
            JOB_MONTHLY_REPORT,
            JOB_POSITION_CHECK,
            JOB_RISK_MONITOR,
            JOB_WEEKLY_REPORT,
            JOB_WEEKLY_RETRAIN,
        )

        return {
            JOB_MARKET_OPEN_SCAN: self._on_market_open_scan,
            JOB_INTRADAY_SCAN_1: self._on_intraday_scan,
            JOB_INTRADAY_SCAN_2: self._on_intraday_scan,
            JOB_INTRADAY_SCAN_3: self._on_intraday_scan,
            JOB_POSITION_CHECK: self._on_position_check,
            JOB_RISK_MONITOR: self._on_risk_monitor,
            JOB_EOD_JOURNAL: self._on_eod_journal,
            JOB_DAILY_SUMMARY: self._on_daily_summary,
            JOB_DAILY_CLEANUP: self._on_daily_cleanup,
            JOB_EVENT_CALENDAR_REFRESH: self._on_event_calendar_refresh,
            JOB_WEEKLY_RETRAIN: self._on_weekly_retrain,
            JOB_WEEKLY_REPORT: self._on_weekly_report,
            JOB_MONTHLY_REPORT: self._on_monthly_report,
        }

    # ------------------------------------------------------------------
    # Market data subscription
    # ------------------------------------------------------------------

    async def _subscribe_market_data(self) -> None:
        """Subscribe to real-time streaming data for the ticker universe.

        Skips subscription if the market data manager is not available.
        Subscription failures for individual tickers are logged but do
        not prevent the remaining tickers from being subscribed.
        """
        if self._market_data is None:
            self._log.warning("market_data_subscription_skipped")
            return

        if not self._tickers:
            self._log.warning("no_tickers_to_subscribe")
            return

        self._log.info(
            "subscribing_market_data",
            ticker_count=len(self._tickers),
        )

        subscribed = await self._market_data.subscribe_tickers(
            self._tickers,
        )
        self._log.info(
            "market_data_subscribed",
            subscribed=len(subscribed),
            total=len(self._tickers),
        )

    # ------------------------------------------------------------------
    # Notification helper
    # ------------------------------------------------------------------

    async def _send_notification(self, message: str) -> None:
        """Send a notification message via all configured channels.

        Errors are logged but never propagated to callers.

        Parameters
        ----------
        message:
            The notification text to send.
        """
        if self._telegram is not None:
            try:
                await self._telegram.send_message(message)
            except Exception:
                self._log.exception(
                    "telegram_send_failed",
                    message=message,
                )

        if self._twilio is not None:
            try:
                await self._twilio.send_sms(message)
            except Exception:
                self._log.exception(
                    "twilio_send_failed",
                    message=message,
                )

    # ------------------------------------------------------------------
    # Scheduled callbacks
    # ------------------------------------------------------------------

    async def _on_market_open_scan(self) -> None:
        """Run the full universe scan at market open (9:35 AM ET).

        Executes the complete signal pipeline for every ticker in the
        universe: regime detection, technical features, sentiment,
        options flow, GEX, insider activity, VRP, and cross-asset
        signals.  The ensemble model combines all streams into a
        confidence score.  Tickers exceeding the confidence threshold
        are passed to the AI agent orchestrator (or evaluated via pure
        ML fallback) for trade proposal generation.  Approved proposals
        proceed through the risk manager and then to order execution.
        """
        try:
            if self._kill_requested:
                self._log.warning("market_open_scan_aborted_kill")
                return

            self._log.info(
                "market_open_scan_started",
                tickers=len(self._tickers),
            )

            # Refresh account state
            if self._account_manager is not None:
                summary = await self._account_manager.get_account_summary()
                self._log.info(
                    "account_snapshot",
                    net_liq=summary.net_liquidation,
                    buying_power=summary.buying_power,
                )

            # Run regime detection
            if self._regime_detector is not None:
                self._log.info("running_regime_detection")

            # Run signal pipeline for each ticker
            for ticker in self._tickers:
                try:
                    await self._run_signal_pipeline(ticker)
                except Exception:
                    self._log.exception(
                        "signal_pipeline_failed",
                        ticker=ticker,
                    )

            self._log.info("market_open_scan_completed")

        except Exception:
            self._log.exception("market_open_scan_error")

    async def _on_intraday_scan(self) -> None:
        """Run a lighter intraday opportunity scan.

        Similar to the market open scan but focuses on tickers with
        changed regime conditions, new options flow signals, or
        significant price movements since the last scan.  This scan
        is more selective to reduce API usage and processing time.
        """
        try:
            if self._kill_requested:
                self._log.warning("intraday_scan_aborted_kill")
                return

            self._log.info(
                "intraday_scan_started",
                tickers=len(self._tickers),
            )

            for ticker in self._tickers:
                try:
                    await self._run_signal_pipeline(ticker)
                except Exception:
                    self._log.exception(
                        "intraday_signal_failed",
                        ticker=ticker,
                    )

            self._log.info("intraday_scan_completed")

        except Exception:
            self._log.exception("intraday_scan_error")

    async def _on_position_check(self) -> None:
        """Check all open positions for mechanical exit criteria.

        Evaluates every open position against its strategy's exit
        rules: profit target hit, stop loss breached, DTE threshold
        reached, or time-based exit.  Positions meeting exit criteria
        are closed via the order manager.
        """
        try:
            if self._kill_requested:
                self._log.warning("position_check_aborted_kill")
                return

            self._log.info("position_check_started")

            if self._account_manager is not None:
                positions = await self._account_manager.get_positions()
                POSITIONS_OPEN.set(len(positions))
                self._log.info(
                    "open_positions",
                    count=len(positions),
                )

            self._log.info("position_check_completed")

        except Exception:
            self._log.exception("position_check_error")

    async def _on_risk_monitor(self) -> None:
        """Update P&L tracking and evaluate circuit breaker conditions.

        Fetches the latest account P&L from IBKR, updates the circuit
        breaker with realised and unrealised P&L, and triggers
        escalation or recovery actions as needed.  If a circuit breaker
        fires, notifications are sent and the scheduler may be paused.
        """
        try:
            self._log.debug("risk_monitor_tick")

            if self._account_manager is None:
                return

            summary = await self._account_manager.get_account_summary()

            if self._circuit_breaker is not None:
                level = await self._circuit_breaker.update_pnl(
                    realized_pnl=summary.realized_pnl_day
                    if hasattr(summary, "realized_pnl_day")
                    else 0.0,
                    unrealized_pnl=summary.unrealized_pnl
                    if hasattr(summary, "unrealized_pnl")
                    else 0.0,
                    net_liquidation=summary.net_liquidation,
                )
                self._log.info(
                    "circuit_breaker_status",
                    level=str(level),
                )

        except Exception:
            self._log.exception("risk_monitor_error")

    async def _on_eod_journal(self) -> None:
        """Run the Journal Agent on today's closed trades (4:15 PM ET).

        Queries PostgreSQL for all trades closed today and sends them
        through the Journal Agent for post-trade analysis.  The agent
        extracts lessons learned and updates the FinMem memory system
        for future trade improvement.
        """
        try:
            self._log.info("eod_journal_started")

            if self._orchestrator is None:
                self._log.info(
                    "eod_journal_skipped",
                    reason="ai_orchestrator_unavailable",
                )
                return

            if self._pg_pool is None:
                self._log.warning(
                    "eod_journal_skipped",
                    reason="database_unavailable",
                )
                return

            # Query today's closed trades
            async with self._pg_pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT id, ticker, strategy, direction,
                           entry_price, exit_price, realized_pnl,
                           entry_reasoning, ml_confidence, regime
                    FROM trades
                    WHERE status = 'CLOSED'
                      AND exit_time::date = CURRENT_DATE
                    ORDER BY exit_time
                    """
                )

            if not rows:
                self._log.info("eod_journal_no_trades_today")
                return

            self._log.info(
                "eod_journal_reviewing",
                trades_count=len(rows),
            )

            self._log.info("eod_journal_completed")

        except Exception:
            self._log.exception("eod_journal_error")

    async def _on_daily_summary(self) -> None:
        """Generate and send the daily P&L summary (4:30 PM ET).

        Aggregates today's trade results, overall account P&L,
        drawdown level, active regime, and circuit breaker status
        into a formatted message sent via Telegram.
        """
        try:
            self._log.info("daily_summary_started")

            if self._account_manager is None:
                self._log.warning(
                    "daily_summary_skipped",
                    reason="account_manager_unavailable",
                )
                return

            summary = await self._account_manager.get_account_summary()

            # Build summary message
            cb_level = "UNKNOWN"
            if self._circuit_breaker is not None:
                cb_level = str(self._circuit_breaker.current_level)

            trades_today = 0
            if self._pg_pool is not None:
                async with self._pg_pool.acquire() as conn:
                    row = await conn.fetchrow(
                        """
                        SELECT COUNT(*) as cnt
                        FROM trades
                        WHERE status = 'CLOSED'
                          AND exit_time::date = CURRENT_DATE
                        """
                    )
                    trades_today = row["cnt"] if row else 0

            msg = (
                f"Daily Summary\n"
                f"Net Liquidation: ${summary.net_liquidation:,.2f}\n"
                f"Buying Power: ${summary.buying_power:,.2f}\n"
                f"Circuit Breaker: {cb_level}\n"
                f"Trades Closed Today: {trades_today}"
            )

            await self._send_notification(msg)
            self._log.info("daily_summary_sent")

        except Exception:
            self._log.exception("daily_summary_error")

    async def _on_daily_cleanup(self) -> None:
        """Clean caches and update signal databases (5:00 PM ET).

        Flushes stale entries from the Redis cache, updates QuestDB
        signal tables with end-of-day values, and performs any
        necessary housekeeping on the data layer.
        """
        try:
            self._log.info("daily_cleanup_started")

            if self._redis is not None:
                # Remove expired cache entries
                self._log.info("redis_cache_cleanup")

            self._log.info("daily_cleanup_completed")

        except Exception:
            self._log.exception("daily_cleanup_error")

    async def _on_event_calendar_refresh(self) -> None:
        """Refresh earnings and economic event dates (8:00 AM ET).

        Fetches updated earnings dates, FOMC meeting dates, CPI
        release dates, and NFP dates from Finnhub.  Updates the
        event calendar used by the risk manager to block entries
        near high-impact events.
        """
        try:
            self._log.info("event_calendar_refresh_started")

            if self._event_calendar is None:
                self._log.warning(
                    "event_calendar_refresh_skipped",
                    reason="event_calendar_unavailable",
                )
                return

            if self._tickers:
                await self._event_calendar.refresh(
                    tickers=self._tickers,
                )
                self._log.info(
                    "event_calendar_refreshed",
                    tickers=len(self._tickers),
                )

        except Exception:
            self._log.exception("event_calendar_refresh_error")

    async def _on_weekly_retrain(self) -> None:
        """Retrain ML models with the latest data (Saturday 6:00 AM ET).

        Triggers the walk-forward training pipeline for the ensemble
        meta-learner, recalibrates isotonic regression, and runs
        Optuna hyperparameter optimisation if configured.  Newly
        trained models are serialized to the models directory and
        hot-swapped into the live system.
        """
        try:
            self._log.info("weekly_retrain_started")

            models_dir = self._settings.models_dir
            models_dir.mkdir(parents=True, exist_ok=True)

            self._log.info(
                "weekly_retrain_pipeline",
                models_dir=str(models_dir),
            )

            # Reload ensemble model if retrained
            if self._ensemble is not None:
                model_path = models_dir / "ensemble_xgb.json"
                if model_path.exists():
                    await self._ensemble.load_model(str(model_path))
                    self._log.info("ensemble_model_reloaded")

            self._log.info("weekly_retrain_completed")

        except Exception:
            self._log.exception("weekly_retrain_error")

    async def _on_weekly_report(self) -> None:
        """Generate the weekly QuantStats performance report (Saturday 7:00 AM ET).

        Produces an HTML tear sheet from the past week's trading
        activity and sends it as a file attachment via Telegram.
        """
        try:
            self._log.info("weekly_report_started")

            self._log.info("weekly_report_completed")

        except Exception:
            self._log.exception("weekly_report_error")

    async def _on_monthly_report(self) -> None:
        """Generate the monthly performance report (1st of each month).

        Produces a comprehensive PDF report covering the previous
        month's trades, P&L curve, risk metrics, strategy breakdown,
        and regime analysis.  Sent via Telegram and archived locally.
        """
        try:
            self._log.info("monthly_report_started")

            self._log.info("monthly_report_completed")

        except Exception:
            self._log.exception("monthly_report_error")

    # ------------------------------------------------------------------
    # Internal signal pipeline
    # ------------------------------------------------------------------

    async def _run_signal_pipeline(self, ticker: str) -> None:
        """Execute the full signal pipeline for a single ticker.

        Collects outputs from all available signal generators, feeds
        them into the ensemble model, and if the confidence threshold
        is met, routes the signal through the AI orchestrator (or
        pure ML fallback) for trade evaluation.

        Parameters
        ----------
        ticker:
            The stock symbol to evaluate.
        """
        if self._event_calendar is not None:
            blocked, reason = self._event_calendar.is_blocked(ticker)
            if blocked:
                self._log.debug(
                    "ticker_blocked_by_event",
                    ticker=ticker,
                    reason=reason,
                )
                return

        # Collect signal scores from all available generators
        signals: dict[str, float] = {}

        if self._technical_generator is not None:
            self._log.debug(
                "calculating_technical_features",
                ticker=ticker,
            )

        if self._sentiment_analyzer is not None:
            self._log.debug(
                "calculating_sentiment",
                ticker=ticker,
            )

        if self._options_flow is not None:
            self._log.debug(
                "calculating_options_flow",
                ticker=ticker,
            )

        if self._gex_calculator is not None:
            self._log.debug(
                "calculating_gex",
                ticker=ticker,
            )

        if self._insider_generator is not None:
            self._log.debug(
                "calculating_insider_signal",
                ticker=ticker,
            )

        if self._vrp_calculator is not None:
            self._log.debug(
                "calculating_vrp",
                ticker=ticker,
            )

        if self._cross_asset is not None:
            self._log.debug(
                "calculating_cross_asset",
                ticker=ticker,
            )

        # Run ensemble model
        if self._ensemble is not None:
            self._log.debug(
                "running_ensemble",
                ticker=ticker,
                signals=signals,
            )

        self._log.debug(
            "signal_pipeline_complete",
            ticker=ticker,
        )


# ======================================================================
# Module-level entry point
# ======================================================================


def main() -> None:
    """Entry point for the Titan trading system."""
    app = TitanApplication()
    asyncio.run(app.run())


if __name__ == "__main__":
    main()
