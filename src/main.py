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
import math
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
IB_GATEWAY_MAX_CONNECT_ATTEMPTS: int = 10
IB_GATEWAY_INITIAL_BACKOFF_SECONDS: float = 5.0
IB_GATEWAY_BACKOFF_FACTOR: float = 2.0

STARTUP_MESSAGE: str = "TITAN ONLINE"
SHUTDOWN_MESSAGE: str = "TITAN OFFLINE"


def _is_nan(value: float) -> bool:
    """Return True if the value is NaN."""
    try:
        return math.isnan(value)
    except (TypeError, ValueError):
        return True


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

        # Trading pause flag — set by /stop, cleared by /start.
        # When True, scans skip trade proposal generation but risk
        # monitoring and position checks continue.
        self._trading_paused: bool = False

        # Trade submission lock — prevents overlapping scans from
        # evaluating and submitting duplicate trades concurrently.
        self._trade_lock: asyncio.Lock = asyncio.Lock()

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

        # Cached VIX bars (fetched once per scan cycle, not per ticker)
        self._cached_vix_bars: Any = None

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

        # 10b. Reconcile IBKR positions against DB on restart
        await self._reconcile_positions()

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

        # 15. Send startup notification with full account/portfolio summary
        await self._send_startup_notification()

        self._log.info("titan_online", tickers=len(self._tickers))

        # 16. Run a startup scan if we came up during market hours.
        # This ensures crash recovery doesn't miss the day's scans.
        await self._maybe_run_startup_scan()

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

        # 7. Stop Telegram bot
        if self._telegram is not None:
            try:
                await self._telegram.stop()
                self._log.info("telegram_bot_stopped")
            except Exception:
                self._log.exception("telegram_stop_failed")

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

    def pause_trading(self) -> None:
        """Pause new trade proposals while keeping the system running.

        Risk monitoring, position checks, and notifications continue.
        Existing positions are untouched.  Resume with :meth:`resume_trading`.
        """
        self._trading_paused = True
        self._log.warning("trading_paused")

    def resume_trading(self) -> None:
        """Resume trade proposal generation after a pause."""
        self._trading_paused = False
        self._log.info("trading_resumed")

    # ------------------------------------------------------------------
    # Infrastructure connections
    # ------------------------------------------------------------------

    async def _connect_postgres(self) -> asyncpg.Pool:
        """Create an asyncpg connection pool to PostgreSQL.

        Uses explicit connection parameters instead of DSN to avoid SSL
        negotiation issues inside Docker networks.  Retries with
        exponential back-off to handle container startup ordering.

        Returns
        -------
        asyncpg.Pool
            A connection pool ready for queries.

        Raises
        ------
        Exception
            If the connection pool cannot be created after all retries.
        """

        import asyncpg

        pg = self._settings.postgres
        self._log.info(
            "connecting_postgres",
            host=pg.host,
            port=pg.port,
            database=pg.db,
        )

        max_retries = 60
        retry_delay = 2.0

        for attempt in range(1, max_retries + 1):
            try:
                pool = await asyncpg.create_pool(
                    host=pg.host,
                    port=pg.port,
                    database=pg.db,
                    user=pg.user,
                    password=pg.password.get_secret_value(),
                    min_size=2,
                    max_size=10,
                    command_timeout=30.0,
                    ssl=False,
                )
                CONNECTION_STATUS.labels(service="postgres").set(1)
                self._log.info("postgres_connected", attempt=attempt)
                return pool
            except (OSError, asyncpg.PostgresError, asyncpg.InterfaceError) as exc:
                if attempt == max_retries:
                    self._log.error(
                        "postgres_connect_failed",
                        attempt=attempt,
                        error=str(exc),
                    )
                    raise
                self._log.warning(
                    "postgres_connect_retry",
                    attempt=attempt,
                    max_retries=max_retries,
                    delay=retry_delay,
                    error=str(exc),
                )
                await asyncio.sleep(retry_delay)
                retry_delay = min(retry_delay * 1.5, 30.0)

        raise RuntimeError("Failed to connect to PostgreSQL")

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
            host=ibkr.gateway_host,
            port=ibkr.gateway_port,
            client_id=ibkr.client_id,
            mode=ibkr.trading_mode,
        )

        gw = GatewayManager(
            host=ibkr.gateway_host,
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

        # Regime detector — fit once on SPY+VIX at startup (market-level regime)
        try:
            self._regime_detector = RegimeDetector()
            self._log.info("regime_detector_initialized")

            if self._market_data is not None:
                try:
                    spy_bars = await self._market_data.get_historical_bars(
                        "SPY",
                        duration="4 Y",
                        bar_size="1 day",
                    )
                    vix_bars = await self._market_data.get_historical_bars(
                        "VIX",
                        duration="4 Y",
                        bar_size="1 day",
                    )
                    if (
                        spy_bars is not None
                        and not spy_bars.empty
                        and vix_bars is not None
                        and not vix_bars.empty
                    ):
                        self._regime_detector.fit(spy_bars, vix_bars["close"])
                        self._log.info("regime_model_fitted_at_startup")
                    else:
                        self._log.warning("regime_startup_fit_skipped_no_data")
                except Exception:
                    self._log.exception("regime_startup_fit_failed")
            else:
                self._log.warning("regime_startup_fit_skipped_no_market_data")
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
                await self._sentiment_analyzer.load_model()
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
            load_strategies_from_config,
        )

        try:
            config_path = str(self._settings.config_dir / "strategies.yaml")
            strategies = load_strategies_from_config(config_path)
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
                order_manager=self._order_manager,
                db_pool=self._pg_pool,
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
    # Position reconciliation
    # ------------------------------------------------------------------

    async def _reconcile_positions(self) -> None:
        """Compare IBKR positions against PostgreSQL trades table.

        For each IBKR position, check if there's a matching open trade
        in the DB.  Log discrepancies — orphaned IBKR positions (no DB
        record) and stale DB records (no IBKR position).  Notifies the
        operator if any discrepancies are found.
        """
        if self._account_manager is None or self._pg_pool is None:
            self._log.warning(
                "reconciliation_skipped",
                reason="account_manager or pg_pool unavailable",
            )
            return

        try:
            ibkr_positions = await self._account_manager.get_positions()
            db_trades = await self._pg_pool.fetch(
                "SELECT id, ticker, strategy, status, entry_price, quantity, "
                "max_profit, max_loss, entry_time "
                "FROM trades WHERE status = 'open'",
            )

            # Build lookup sets
            ibkr_tickers = {
                getattr(p, "ticker", None) or getattr(p, "symbol", "")
                for p in ibkr_positions
            }
            ibkr_tickers.discard("")
            db_tickers = {r["ticker"] for r in db_trades}

            # Orphaned IBKR positions (in broker but not in DB)
            orphaned = ibkr_tickers - db_tickers
            for ticker in orphaned:
                self._log.warning(
                    "orphaned_ibkr_position",
                    ticker=ticker,
                    note="Position in IBKR with no matching open trade in DB",
                )

            # Stale DB records (in DB but not in broker)
            stale = db_tickers - ibkr_tickers
            for ticker in stale:
                self._log.warning(
                    "stale_db_trade",
                    ticker=ticker,
                    note="Open trade in DB with no matching IBKR position",
                )

            # Notify operator if there are discrepancies
            if orphaned or stale:
                await self._send_notification(
                    f"POSITION RECONCILIATION: {len(orphaned)} orphaned IBKR, "
                    f"{len(stale)} stale DB records — manual review needed",
                )

            self._log.info(
                "position_reconciliation_complete",
                ibkr_count=len(ibkr_positions),
                db_count=len(db_trades),
                orphaned=len(orphaned),
                stale=len(stale),
            )

        except Exception:
            self._log.exception("position_reconciliation_failed")

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
                # Wire callbacks for interactive commands.
                self._telegram.set_kill_callback(self.request_kill)
                self._telegram.set_pause_callback(self.pause_trading)
                self._telegram.set_resume_callback(self.resume_trading)
                self._telegram.set_system_state(self._get_system_state)
                await self._telegram.start()
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
                from src.notifications.twilio_sms import TwilioSMSNotifier

                self._twilio = TwilioSMSNotifier(
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
            JOB_DAILY_PNL_RESET,
            JOB_DAILY_SUMMARY,
            JOB_EOD_JOURNAL,
            JOB_EVENT_CALENDAR_REFRESH,
            JOB_INTRADAY_SCAN_1,
            JOB_INTRADAY_SCAN_2,
            JOB_INTRADAY_SCAN_3,
            JOB_MARKET_OPEN_SCAN,
            JOB_MONTHLY_PNL_RESET,
            JOB_MONTHLY_REPORT,
            JOB_POSITION_CHECK,
            JOB_RISK_MONITOR,
            JOB_WEEKLY_PNL_RESET,
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
            JOB_DAILY_PNL_RESET: self._on_daily_pnl_reset,
            JOB_WEEKLY_PNL_RESET: self._on_weekly_pnl_reset,
            JOB_MONTHLY_PNL_RESET: self._on_monthly_pnl_reset,
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

    async def _send_startup_notification(self) -> None:
        """Send a rich startup notification with account and portfolio details.

        Falls back to a simple "TITAN ONLINE" alert if the Telegram
        notifier does not support the rich startup summary or if the
        account data is unavailable.
        """
        if self._telegram is None:
            return

        try:
            state = self._get_system_state()
            await self._telegram.send_startup_summary(state)
        except Exception:
            self._log.exception("startup_summary_failed")
            # Fall back to simple notification
            try:
                await self._telegram.send_system_alert(STARTUP_MESSAGE)
            except Exception:
                self._log.exception("startup_fallback_notification_failed")

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
                await self._telegram.send_system_alert(message)
            except Exception:
                self._log.exception(
                    "telegram_send_failed",
                    message=message,
                )

        if self._twilio is not None:
            try:
                await self._twilio.send_alert("GENERAL", message)
            except Exception:
                self._log.exception(
                    "twilio_send_failed",
                    message=message,
                )

    # ------------------------------------------------------------------
    # System state for Telegram commands
    # ------------------------------------------------------------------

    def _get_system_state(self) -> dict[str, Any]:
        """Build a snapshot of current system state for Telegram commands.

        Returns a dict consumed by TelegramNotifier's command handlers
        (``/status``, ``/portfolio``, ``/positions``).  All values are
        synchronously derived from already-cached subsystem data so this
        callable is safe to invoke from any context.
        """
        state: dict[str, Any] = {
            "connected": False,
            "trading_mode": self._settings.ibkr.trading_mode,
            "trading_paused": self._trading_paused,
            "net_liquidation": 0.0,
            "buying_power": 0.0,
            "excess_liquidity": 0.0,
            "maint_margin": 0.0,
            "daily_pnl": 0.0,
            "unrealized_pnl": 0.0,
            "realized_pnl": 0.0,
            "open_positions": 0,
            "max_positions": self._settings.trading.max_concurrent_positions,
            "positions": [],
            "circuit_breaker_level": "NORMAL",
            "regime": "unknown",
            "tickers_count": len(self._tickers),
        }

        # Gateway connectivity
        if self._gateway is not None:
            import contextlib

            with contextlib.suppress(Exception):
                state["connected"] = self._gateway.is_connected()

        # Account data (ib_async caches these locally after subscription)
        if self._account_manager is not None:
            try:
                ib = self._account_manager._ib
                # Portfolio items are cached by ib_async after reqAccountUpdates
                raw_items = ib.portfolio()
                positions_list: list[dict[str, Any]] = []
                for pi in raw_items:
                    contract = pi.contract
                    pnl = pi.unrealizedPNL if not _is_nan(pi.unrealizedPNL) else 0.0
                    positions_list.append(
                        {
                            "ticker": contract.symbol or contract.localSymbol or "???",
                            "sec_type": contract.secType or "",
                            "strategy": contract.secType or "",
                            "quantity": pi.position,
                            "avg_cost": pi.averageCost,
                            "market_value": pi.marketValue
                            if not _is_nan(pi.marketValue)
                            else 0.0,
                            "unrealized_pnl": pnl,
                        }
                    )
                state["positions"] = positions_list
                state["open_positions"] = len(positions_list)

                # Account summary values from ib_async cache
                for av in ib.accountValues():
                    tag = av.tag
                    if av.currency not in ("USD", "BASE", ""):
                        continue
                    try:
                        val = float(av.value)
                    except (ValueError, TypeError):
                        continue
                    if tag == "NetLiquidation":
                        state["net_liquidation"] = val
                    elif tag == "BuyingPower":
                        state["buying_power"] = val
                    elif tag == "ExcessLiquidity":
                        state["excess_liquidity"] = val
                    elif tag == "FullMaintMarginReq":
                        state["maint_margin"] = val
            except Exception:
                self._log.debug("system_state_account_fetch_error", exc_info=True)

            # PnL from subscription (already live-cached)
            with contextlib.suppress(Exception):
                pnl_sub = self._account_manager._pnl_subscription
                if pnl_sub is not None:
                    if not _is_nan(pnl_sub.dailyPnL):
                        state["daily_pnl"] = pnl_sub.dailyPnL
                    if not _is_nan(pnl_sub.unrealizedPnL):
                        state["unrealized_pnl"] = pnl_sub.unrealizedPnL
                    if not _is_nan(pnl_sub.realizedPnL):
                        state["realized_pnl"] = pnl_sub.realizedPnL

        # Circuit breaker
        if self._circuit_breaker is not None:
            with contextlib.suppress(Exception):
                state["circuit_breaker_level"] = str(
                    self._circuit_breaker.current_level
                )

        # Regime
        if self._regime_detector is not None:
            try:
                regime = getattr(self._regime_detector, "_previous_regime", None)
                if regime is not None:
                    state["regime"] = str(regime)
            except Exception:
                pass

        return state

    # ------------------------------------------------------------------
    # Startup scan (crash recovery)
    # ------------------------------------------------------------------

    async def _maybe_run_startup_scan(self) -> None:
        """Run an immediate scan if the bot starts during market hours.

        After a crash or restart, scheduled cron scans that already
        passed for the day will not fire retroactively.  This method
        checks if the current time falls within US equity market hours
        (9:35 AM – 3:55 PM ET, Mon–Fri) and, if so, triggers a full
        universe scan identical to the market-open scan.
        """
        from datetime import datetime

        import pytz

        eastern = pytz.timezone("US/Eastern")
        now = datetime.now(eastern)

        # Only on weekdays
        if now.weekday() >= 5:
            self._log.info("startup_scan_skipped_weekend")
            return

        market_open = now.replace(hour=9, minute=35, second=0, microsecond=0)
        market_close = now.replace(hour=15, minute=55, second=0, microsecond=0)

        if not (market_open <= now <= market_close):
            self._log.info(
                "startup_scan_skipped_outside_hours",
                current_time=now.strftime("%H:%M:%S"),
            )
            return

        self._log.info(
            "startup_scan_triggered",
            current_time=now.strftime("%H:%M:%S"),
            reason="crash_recovery",
        )

        try:
            await self._on_market_open_scan()
        except Exception:
            self._log.exception("startup_scan_failed")

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

            if self._trading_paused:
                self._log.info("market_open_scan_skipped_paused")
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

            # Cache VIX bars once for the entire scan cycle so every
            # ticker's regime prediction reuses the same data.
            if self._market_data is not None:
                try:
                    self._cached_vix_bars = await self._market_data.get_historical_bars(
                        "VIX",
                        duration="100 D",
                        bar_size="1 day",
                    )
                except Exception:
                    self._log.warning("vix_cache_fetch_failed")

            # Run signal pipeline for each ticker under trade lock to
            # prevent overlapping scans from submitting duplicate orders.
            async with self._trade_lock:
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

            if self._trading_paused:
                self._log.info("intraday_scan_skipped_paused")
                return

            self._log.info(
                "intraday_scan_started",
                tickers=len(self._tickers),
            )

            # Refresh VIX cache for this scan cycle
            if self._market_data is not None:
                try:
                    self._cached_vix_bars = await self._market_data.get_historical_bars(
                        "VIX",
                        duration="100 D",
                        bar_size="1 day",
                    )
                except Exception:
                    self._log.warning("vix_cache_fetch_failed")

            async with self._trade_lock:
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

            if self._account_manager is None:
                self._log.warning("position_check_skipped", reason="no_account_manager")
                return

            positions = await self._account_manager.get_positions()
            POSITIONS_OPEN.set(len(positions))
            self._log.info("open_positions", count=len(positions))

            if not positions:
                self._log.info("position_check_completed", exits=0)
                return

            # Load open trades from DB for enrichment — PositionInfo from
            # IBKR only has con_id, ticker, quantity, avg_cost, market_value,
            # unrealized_pnl.  Strategy, max_profit, max_loss, entry_time
            # come from the trades table.
            db_trades: dict[str, Any] = {}
            if self._pg_pool is not None:
                try:
                    rows = await self._pg_pool.fetch(
                        "SELECT ticker, strategy, max_profit, max_loss, entry_time "
                        "FROM trades WHERE status = 'open'",
                    )
                    for row in rows:
                        db_trades[row["ticker"]] = dict(row)
                except Exception:
                    self._log.warning("position_check_db_fetch_failed")

            exits_triggered = 0
            for position in positions:
                try:
                    ticker = getattr(position, "ticker", None) or getattr(
                        position, "symbol", None
                    )
                    unrealized_pnl = getattr(position, "unrealized_pnl", 0.0) or 0.0

                    # Enrich with DB trade data
                    db_trade = db_trades.get(ticker, {})
                    strategy_name = db_trade.get("strategy", "unknown")
                    max_profit = db_trade.get("max_profit")
                    max_loss = db_trade.get("max_loss")

                    # Compute DTE — would need options contract expiry date
                    # which is not available from PositionInfo.  Set to None
                    # for now; full DTE check requires contract lookup.
                    dte: int | None = None

                    should_exit = False
                    exit_reason = ""

                    # Exit rule 1: DTE threshold — close at 21 DTE regardless of P&L
                    if dte is not None and dte <= 21:
                        should_exit = True
                        exit_reason = f"DTE threshold reached ({dte} DTE <= 21)"

                    # Exit rule 2: Profit target hit (50-65% of max profit)
                    if not should_exit and max_profit and max_profit > 0:
                        profit_pct = unrealized_pnl / max_profit
                        if profit_pct >= 0.50:
                            should_exit = True
                            exit_reason = f"Profit target hit ({profit_pct:.0%} of max)"

                    # Exit rule 3: Stop loss breached (100% of max loss)
                    if (
                        not should_exit
                        and max_loss
                        and max_loss > 0
                        and abs(unrealized_pnl) >= max_loss
                        and unrealized_pnl < 0
                    ):
                        should_exit = True
                        loss_amt = abs(unrealized_pnl)
                        exit_reason = (
                            f"Stop loss breached "
                            f"(loss ${loss_amt:.0f}"
                            f" >= max ${max_loss:.0f})"
                        )

                    if should_exit:
                        exits_triggered += 1
                        self._log.warning(
                            "exit_criteria_met",
                            ticker=ticker,
                            strategy=strategy_name,
                            reason=exit_reason,
                            unrealized_pnl=round(unrealized_pnl, 2),
                            dte=dte,
                        )

                        # Submit close order via order manager
                        if self._order_manager is not None:
                            try:
                                await self._order_manager.close_position(position)
                                self._log.info(
                                    "exit_order_submitted",
                                    ticker=ticker,
                                    strategy=strategy_name,
                                    reason=exit_reason,
                                )
                            except Exception:
                                self._log.exception(
                                    "exit_order_failed",
                                    ticker=ticker,
                                    strategy=strategy_name,
                                )
                                # Send alert on failed exit — operator must intervene
                                msg = (
                                    f"EXIT ORDER FAILED: "
                                    f"{ticker} {strategy_name}"
                                    f" — {exit_reason}"
                                )
                                await self._send_notification(
                                    msg,
                                )
                    else:
                        self._log.debug(
                            "position_holding",
                            ticker=ticker,
                            strategy=strategy_name,
                            unrealized_pnl=round(unrealized_pnl, 2),
                            dte=dte,
                        )

                except Exception:
                    self._log.exception(
                        "position_exit_eval_failed",
                        position=str(position),
                    )

            self._log.info(
                "position_check_completed",
                positions_checked=len(positions),
                exits_triggered=exits_triggered,
            )

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

            # Convert rows to trade dicts for the Journal Agent
            trades: list[dict[str, Any]] = [dict(row) for row in rows]
            try:
                journal_entry = await self._orchestrator.run_journal(trades)
                self._log.info(
                    "eod_journal_review_complete",
                    trades_reviewed=len(trades),
                    total_pnl=journal_entry.get("total_pnl", 0.0)
                    if journal_entry
                    else 0.0,
                )
            except Exception:
                self._log.exception("eod_journal_agent_failed")

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
            cleaned_keys: int = 0

            if self._redis is not None:
                try:
                    # Scan for expired or stale market-data keys
                    cursor: int = 0
                    stale_patterns = ["mkt:*", "chain:*", "snapshot:*"]
                    for pattern in stale_patterns:
                        cursor = 0
                        while True:
                            cursor, keys = await self._redis.scan(
                                cursor=cursor,
                                match=pattern,
                                count=200,
                            )
                            if keys:
                                # Check TTL — delete keys that are already
                                # persisting (TTL == -1) to prevent unbounded growth
                                for key in keys:
                                    ttl = await self._redis.ttl(key)
                                    if ttl == -1:  # No expiry set
                                        await self._redis.delete(key)
                                        cleaned_keys += 1
                            if cursor == 0:
                                break
                    self._log.info(
                        "redis_cache_cleanup",
                        keys_removed=cleaned_keys,
                    )
                except Exception:
                    self._log.exception("redis_cleanup_failed")

            # Flush QuestDB buffer if the writer is available
            if hasattr(self, "_questdb_writer") and self._questdb_writer is not None:
                try:
                    await self._questdb_writer.flush()
                    self._log.info("questdb_buffer_flushed")
                except Exception:
                    self._log.exception("questdb_flush_failed")

            self._log.info(
                "daily_cleanup_completed",
                redis_keys_cleaned=cleaned_keys,
            )

        except Exception:
            self._log.exception("daily_cleanup_error")

    async def _on_daily_pnl_reset(self) -> None:
        """Reset the daily P&L accumulator before market open (9:29 AM ET).

        Without this reset, yesterday's P&L carries over and the daily
        circuit breaker fires on stale data.
        """
        try:
            if self._circuit_breaker is not None:
                self._circuit_breaker.reset_daily_pnl()
                self._log.info("daily_pnl_reset_completed")
            else:
                self._log.warning(
                    "daily_pnl_reset_skipped", reason="no_circuit_breaker"
                )
        except Exception:
            self._log.exception("daily_pnl_reset_error")

    async def _on_weekly_pnl_reset(self) -> None:
        """Reset the weekly P&L accumulator on Monday before open (9:29 AM ET).

        Without this reset, prior week's P&L accumulates and triggers
        false weekly circuit breaker alerts.
        """
        try:
            if self._circuit_breaker is not None:
                self._circuit_breaker.reset_weekly_pnl()
                self._log.info("weekly_pnl_reset_completed")
            else:
                self._log.warning(
                    "weekly_pnl_reset_skipped", reason="no_circuit_breaker"
                )
        except Exception:
            self._log.exception("weekly_pnl_reset_error")

    async def _on_monthly_pnl_reset(self) -> None:
        """Reset the monthly P&L accumulator on 1st of month (9:29 AM ET).

        Without this reset, prior month's P&L accumulates and triggers
        false monthly circuit breaker alerts.
        """
        try:
            if self._circuit_breaker is not None:
                self._circuit_breaker.reset_monthly_pnl()
                self._log.info("monthly_pnl_reset_completed")
            else:
                self._log.warning(
                    "monthly_pnl_reset_skipped", reason="no_circuit_breaker"
                )
        except Exception:
            self._log.exception("monthly_pnl_reset_error")

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

            from src.ml.features import FeatureEngineer
            from src.ml.trainer import WalkForwardTrainer

            models_dir: Path = self._settings.models_dir
            models_dir.mkdir(parents=True, exist_ok=True)

            # Fetch training data from PostgreSQL
            if self._pg_pool is None:
                self._log.warning("weekly_retrain_skipped", reason="no_database")
                return

            async with self._pg_pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT ticker, strategy, direction, entry_price,
                           exit_price, realized_pnl, ml_confidence,
                           regime, entry_time, exit_time
                    FROM trades
                    WHERE status = 'CLOSED'
                      AND exit_time IS NOT NULL
                    ORDER BY exit_time
                    """
                )

            if len(rows) < 50:
                self._log.warning(
                    "weekly_retrain_insufficient_data",
                    trades=len(rows),
                    minimum=50,
                )
                return

            self._log.info(
                "weekly_retrain_pipeline",
                models_dir=str(models_dir),
                training_trades=len(rows),
            )

            # Build features and train in executor (CPU-bound)
            import asyncio

            loop = asyncio.get_running_loop()
            engineer = FeatureEngineer()
            trainer = WalkForwardTrainer(
                model_dir=str(models_dir),
            )

            def _train_sync() -> None:
                import shutil
                from pathlib import Path

                import pandas as pd

                trade_df = pd.DataFrame([dict(r) for r in rows])
                # Build simple feature matrix from trade metadata
                x_features = engineer.build_trade_features(trade_df)
                y = (trade_df["realized_pnl"] > 0).astype(int)
                has_features = (
                    x_features is not None
                    and not x_features.empty
                    and len(x_features) == len(y)
                )
                if has_features:
                    result = trainer.train(x_features, y)
                    self._log.info(
                        "weekly_retrain_result",
                        mean_auc=round(result.mean_auc, 4)
                        if hasattr(result, "mean_auc")
                        else "N/A",
                        best_fold=result.best_fold
                        if hasattr(result, "best_fold")
                        else "N/A",
                    )

                    # Copy versioned model to the fixed name the ensemble expects
                    mdir = Path(str(models_dir))
                    versioned = sorted(mdir.glob("titan_xgboost_ensemble_v*.json"))
                    if versioned:
                        latest = versioned[-1]
                        canonical = mdir / "ensemble_xgb.json"
                        shutil.copy2(str(latest), str(canonical))
                        self._log.info(
                            "model_copied_to_canonical",
                            source=latest.name,
                            dest=canonical.name,
                        )
                        # Also copy companion calibrator if it exists
                        cal_src = latest.with_suffix(".calibrator.pkl")
                        if cal_src.exists():
                            cal_dst = mdir / "ensemble_xgb.calibrator.pkl"
                            shutil.copy2(str(cal_src), str(cal_dst))
                            self._log.info(
                                "calibrator_copied_to_canonical",
                                source=cal_src.name,
                                dest=cal_dst.name,
                            )

            await loop.run_in_executor(None, _train_sync)

            # Hot-swap: reload ensemble model if retrained
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

            if self._pg_pool is None:
                self._log.warning("weekly_report_skipped", reason="no_database")
                return

            async with self._pg_pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT ticker, strategy, direction, realized_pnl,
                           entry_time, exit_time, ml_confidence
                    FROM trades
                    WHERE status = 'CLOSED'
                      AND exit_time >= NOW() - INTERVAL '7 days'
                    ORDER BY exit_time
                    """
                )

            if not rows:
                self._log.info("weekly_report_no_trades")
                await self._send_notification(
                    "Weekly Report: No trades closed this week."
                )
                return

            total_pnl = sum(float(r["realized_pnl"] or 0) for r in rows)
            winners = sum(1 for r in rows if float(r["realized_pnl"] or 0) > 0)
            losers = len(rows) - winners
            win_rate = winners / len(rows) * 100 if rows else 0

            msg = (
                f"Weekly Report\n"
                f"Trades: {len(rows)}\n"
                f"Win Rate: {win_rate:.1f}%\n"
                f"Winners: {winners} | Losers: {losers}\n"
                f"Total P&L: ${total_pnl:,.2f}"
            )
            await self._send_notification(msg)
            self._log.info(
                "weekly_report_completed",
                trades=len(rows),
                total_pnl=round(total_pnl, 2),
            )

        except Exception:
            self._log.exception("weekly_report_error")

    async def _on_monthly_report(self) -> None:
        """Generate the monthly performance report (1st of each month).

        Produces a summary covering the previous month's trades, P&L,
        win rate, strategy breakdown, and regime analysis.
        Sent via Telegram.
        """
        try:
            self._log.info("monthly_report_started")

            if self._pg_pool is None:
                self._log.warning("monthly_report_skipped", reason="no_database")
                return

            async with self._pg_pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT ticker, strategy, direction, realized_pnl,
                           entry_time, exit_time, ml_confidence, regime
                    FROM trades
                    WHERE status = 'CLOSED'
                      AND exit_time >= DATE_TRUNC('month', NOW()) - INTERVAL '1 month'
                      AND exit_time < DATE_TRUNC('month', NOW())
                    ORDER BY exit_time
                    """
                )

            if not rows:
                self._log.info("monthly_report_no_trades")
                await self._send_notification("Monthly Report: No trades last month.")
                return

            total_pnl = sum(float(r["realized_pnl"] or 0) for r in rows)
            winners = sum(1 for r in rows if float(r["realized_pnl"] or 0) > 0)
            win_rate = winners / len(rows) * 100 if rows else 0

            # Strategy breakdown
            by_strat: dict[str, float] = {}
            for r in rows:
                s = str(r["strategy"])
                by_strat[s] = by_strat.get(s, 0.0) + float(r["realized_pnl"] or 0)
            strat_lines = "\n".join(
                f"  {s}: ${p:,.2f}"
                for s, p in sorted(by_strat.items(), key=lambda x: x[1], reverse=True)
            )

            msg = (
                f"Monthly Report\n"
                f"Trades: {len(rows)}\n"
                f"Win Rate: {win_rate:.1f}%\n"
                f"Total P&L: ${total_pnl:,.2f}\n"
                f"Strategy Breakdown:\n{strat_lines}"
            )
            await self._send_notification(msg)
            self._log.info(
                "monthly_report_completed",
                trades=len(rows),
                total_pnl=round(total_pnl, 2),
            )

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
        from src.signals.ensemble import SignalInputs
        from src.utils.metrics import CONFIDENCE_SCORE, SIGNAL_SCORE

        # ----- 1. Event calendar gate -----
        if self._event_calendar is not None:
            blocked, reason = self._event_calendar.is_blocked(ticker)
            if blocked:
                self._log.debug(
                    "ticker_blocked_by_event",
                    ticker=ticker,
                    reason=reason,
                )
                return

        # ----- 2. Get spot price -----
        spot_price: float = 0.0
        if self._market_data is not None:
            try:
                snapshot = await self._market_data.get_snapshot(ticker)
                spot_price = (
                    snapshot.last if not _is_nan(snapshot.last) else snapshot.mid
                )
                if _is_nan(spot_price):
                    spot_price = 0.0
            except Exception:
                self._log.warning("spot_price_fetch_failed", ticker=ticker)

        if spot_price <= 0:
            self._log.debug("skipping_ticker_no_price", ticker=ticker)
            return

        # ----- 3. Gather signals in parallel -----
        # Each coroutine is wrapped so failures are isolated — a single
        # generator going down does not block the rest of the pipeline.
        signal_inputs_kwargs: dict[str, Any] = {}

        async def _safe_technical() -> None:
            if self._technical_generator is None or self._market_data is None:
                return
            try:
                # Fetch recent OHLCV bars from IBKR for technical features
                bars = await self._market_data.get_historical_bars(
                    ticker,
                    duration="2 Y",
                    bar_size="1 day",
                )
                if bars is not None and not bars.empty:
                    features_df = self._technical_generator.calculate_features(bars)
                    if not features_df.empty:
                        last_row = features_df.iloc[-1]
                        tech_features: dict[str, float] = {}
                        for col in features_df.columns:
                            if col not in ("open", "high", "low", "close", "volume"):
                                val = last_row.get(col, 0.0)
                                if not _is_nan(val):
                                    tech_features[col] = float(val)
                        signal_inputs_kwargs["technical_features"] = tech_features
                        # Use RSI as a quick composite proxy
                        rsi = tech_features.get("RSI_14", 50.0)
                        signal_inputs_kwargs["technical_score"] = max(
                            0.0, min(1.0, rsi / 100.0)
                        )
                        SIGNAL_SCORE.labels(
                            signal_type="technical",
                            ticker=ticker,
                        ).set(signal_inputs_kwargs["technical_score"])
            except Exception:
                self._log.warning("technical_failed", ticker=ticker)

        async def _safe_sentiment() -> None:
            if self._sentiment_analyzer is None:
                return
            try:
                result = await self._sentiment_analyzer.analyze_news(ticker)
                signal_inputs_kwargs["sentiment_score"] = result.score
                signal_inputs_kwargs["sentiment_articles"] = result.num_articles
                signal_inputs_kwargs["sentiment_confidence"] = result.avg_confidence
                SIGNAL_SCORE.labels(signal_type="sentiment", ticker=ticker).set(
                    result.score
                )
            except Exception:
                self._log.warning("sentiment_failed", ticker=ticker)

        async def _safe_options_flow() -> None:
            if self._options_flow is None:
                return
            try:
                transactions = await self._options_flow.fetch_flow_data(ticker)
                activities = self._options_flow.detect_unusual_activity(transactions)
                flow_signal = self._options_flow.calculate_flow_score(
                    activities, ticker
                )
                signal_inputs_kwargs["flow_score"] = flow_signal.score
                signal_inputs_kwargs["flow_consistency"] = flow_signal.consistency
                signal_inputs_kwargs["flow_net_premium"] = flow_signal.net_premium
                signal_inputs_kwargs["flow_num_unusual"] = flow_signal.num_unusual
                SIGNAL_SCORE.labels(signal_type="flow", ticker=ticker).set(
                    flow_signal.score
                )
            except Exception:
                self._log.warning("options_flow_failed", ticker=ticker)

        async def _safe_insider() -> None:
            if self._insider_generator is None:
                return
            try:
                filings = await self._insider_generator.fetch_form4_filings(ticker)
                insider_signal = self._insider_generator.calculate_insider_signal(
                    ticker, filings
                )
                signal_inputs_kwargs["insider_score"] = insider_signal.score
                signal_inputs_kwargs["insider_num_buys"] = insider_signal.num_buys
                signal_inputs_kwargs["insider_num_sells"] = insider_signal.num_sells
                signal_inputs_kwargs["insider_net_value"] = insider_signal.net_value
                SIGNAL_SCORE.labels(signal_type="insider", ticker=ticker).set(
                    insider_signal.score
                )
            except Exception:
                self._log.warning("insider_failed", ticker=ticker)

        async def _safe_cross_asset() -> None:
            if self._cross_asset is None:
                return
            try:
                macro = await self._cross_asset.fetch_macro_data()
                vix_ts = await self._cross_asset.fetch_vix_term_structure()
                cross_prices = await self._cross_asset.fetch_cross_asset_prices()
                ca_signal = self._cross_asset.calculate_cross_asset_signal(
                    macro,
                    vix_ts,
                    cross_prices,
                )
                signal_inputs_kwargs["cross_asset_score"] = ca_signal.score
                signal_inputs_kwargs["cross_asset_bias"] = ca_signal.bias
                signal_inputs_kwargs["cross_asset_yield_curve_score"] = (
                    ca_signal.yield_curve_score
                )
                signal_inputs_kwargs["cross_asset_credit_score"] = (
                    ca_signal.credit_score
                )
                signal_inputs_kwargs["cross_asset_vix_ts_score"] = (
                    ca_signal.vix_ts_score
                )
                SIGNAL_SCORE.labels(signal_type="cross_asset", ticker=ticker).set(
                    ca_signal.score
                )
            except Exception:
                self._log.warning("cross_asset_failed", ticker=ticker)

        async def _safe_gex() -> None:
            if self._gex_calculator is None or self._market_data is None:
                return
            try:
                chain = await self._market_data.get_options_chain(ticker)
                if not chain:
                    return

                # Enrich with Greeks + OI (subscribe, wait, extract)
                greeks_map = await self._market_data.get_option_greeks(chain)

                # Cancel option subscriptions to free streaming lines
                await self._market_data.cancel_option_subscriptions()

                if not greeks_map:
                    self._log.warning("gex_no_greeks", ticker=ticker)
                    return

                # Convert OptionGreeks to dict format for calculate_gex
                chain_dicts: list[dict[str, Any]] = []
                for greeks in greeks_map.values():
                    if greeks.gamma == 0.0:
                        continue  # Skip contracts with no gamma data
                    chain_dicts.append(
                        {
                            "ticker": ticker,
                            "symbol": ticker,
                            "strike": greeks.strike,
                            "right": greeks.right,
                            "open_interest": greeks.open_interest,
                            "oi": greeks.open_interest,
                            "gamma": greeks.gamma,
                            "delta": greeks.delta,
                        }
                    )

                if not chain_dicts:
                    self._log.warning("gex_no_valid_greeks", ticker=ticker)
                    return

                gex_profile = self._gex_calculator.calculate_gex(
                    chain_dicts, spot_price
                )
                gex_levels = self._gex_calculator.identify_levels(
                    {s.strike: s.net_gex for s in gex_profile.gex_by_strike},
                    spot_price,
                )
                gex_signal = self._gex_calculator.get_gex_signal(
                    gex_profile,
                    gex_levels,
                    spot_price,
                )
                signal_inputs_kwargs["gex_score"] = gex_signal.score
                signal_inputs_kwargs["gex_net_gex"] = gex_signal.net_gex
                signal_inputs_kwargs["gex_regime"] = gex_signal.regime
                SIGNAL_SCORE.labels(signal_type="gex", ticker=ticker).set(
                    gex_signal.score
                )
            except Exception:
                self._log.warning("gex_failed", ticker=ticker)

        # Fire all independent async generators concurrently
        await asyncio.gather(
            _safe_technical(),
            _safe_sentiment(),
            _safe_options_flow(),
            _safe_insider(),
            _safe_cross_asset(),
            _safe_gex(),
            return_exceptions=True,
        )

        # ----- 4. Regime detection (needed for strategy selection) -----
        # The HMM model is fitted once at startup on SPY+VIX.  Here we only
        # call predict() using recent price data and the scan-level VIX cache.
        regime: str = "unknown"
        regime_confidence: float = 0.0
        if (
            self._regime_detector is not None
            and self._regime_detector._model is not None
            and self._market_data is not None
        ):
            try:
                price_bars = await self._market_data.get_historical_bars(
                    ticker,
                    duration="100 D",
                    bar_size="1 day",
                )
                vix_bars = self._cached_vix_bars
                if (
                    price_bars is not None
                    and not price_bars.empty
                    and vix_bars is not None
                    and not vix_bars.empty
                ):
                    regime_result = self._regime_detector.predict(
                        price_bars,
                        vix_bars["close"],
                    )
                    regime = regime_result.regime
                    regime_confidence = regime_result.confidence
                    signal_inputs_kwargs["regime"] = regime
                    signal_inputs_kwargs["regime_confidence"] = regime_confidence
            except Exception:
                self._log.warning("regime_detection_failed", ticker=ticker)

        # ----- 5. VRP (uses historical IV data from IBKR) -----
        iv_rank: float = 0.0
        if self._vrp_calculator is not None and self._market_data is not None:
            try:
                # Fetch historical IV to compute IV Rank, IV Percentile
                iv_df = await self._market_data.get_historical_iv(ticker, days=252)
                if not iv_df.empty and "iv" in iv_df.columns:
                    iv_series = iv_df["iv"].dropna()
                    current_iv = (
                        float(iv_series.iloc[-1]) if len(iv_series) > 0 else 0.0
                    )

                    # Compute realized volatility from spot returns
                    price_bars = await self._market_data.get_historical_bars(
                        ticker,
                        duration="60 D",
                        bar_size="1 day",
                    )
                    rv_current: float = 0.0
                    if (
                        price_bars is not None
                        and "close" in price_bars.columns
                        and len(price_bars) > 20
                    ):
                        import numpy as np

                        log_returns = np.log(
                            price_bars["close"] / price_bars["close"].shift(1),
                        ).dropna()
                        rv_current = float(log_returns.std() * np.sqrt(252)) * 100.0

                    vrp_result = self._vrp_calculator.calculate_vrp(
                        current_iv * 100.0,
                        rv_current,
                    )
                    computed_iv_rank = self._vrp_calculator.calculate_iv_rank(
                        current_iv,
                        iv_series,
                    )
                    computed_iv_pct = self._vrp_calculator.calculate_iv_percentile(
                        current_iv,
                        iv_series,
                    )
                    hv_iv_ratio = (
                        rv_current / (current_iv * 100.0) if current_iv > 0 else 1.0
                    )

                    vrp_signal = self._vrp_calculator.get_vrp_signal(
                        iv_rank=computed_iv_rank,
                        iv_percentile=computed_iv_pct,
                        vrp=vrp_result.vrp,
                        hv_iv_ratio=hv_iv_ratio,
                    )
                    signal_inputs_kwargs["vrp_score"] = vrp_signal.score
                    signal_inputs_kwargs["vrp_iv_rank"] = vrp_signal.iv_rank
                    signal_inputs_kwargs["vrp_iv_percentile"] = vrp_signal.iv_percentile
                    signal_inputs_kwargs["vrp_hv_iv_ratio"] = vrp_signal.hv_iv_ratio
                    signal_inputs_kwargs["vrp_spread"] = vrp_result.vrp
                    iv_rank = vrp_signal.iv_rank
                    SIGNAL_SCORE.labels(signal_type="vrp", ticker=ticker).set(
                        vrp_signal.score
                    )
                else:
                    self._log.debug("vrp_no_iv_data", ticker=ticker)
            except Exception:
                self._log.warning("vrp_failed", ticker=ticker)

        # ----- 6. Build ensemble input and score -----
        signal_inputs = SignalInputs(**signal_inputs_kwargs)

        if self._ensemble is None:
            self._log.debug("ensemble_not_available", ticker=ticker)
            return

        try:
            ensemble_result = await self._ensemble.generate_signal(
                ticker, signal_inputs
            )
        except Exception:
            self._log.exception("ensemble_scoring_failed", ticker=ticker)
            return

        CONFIDENCE_SCORE.observe(ensemble_result.confidence)
        SIGNAL_SCORE.labels(signal_type="ensemble", ticker=ticker).set(
            ensemble_result.confidence,
        )

        self._log.info(
            "ensemble_scored",
            ticker=ticker,
            confidence=round(ensemble_result.confidence, 4),
            should_trade=ensemble_result.should_trade,
            direction=ensemble_result.direction_bias,
        )

        if not ensemble_result.should_trade:
            self._log.debug(
                "below_confidence_threshold",
                ticker=ticker,
                confidence=round(ensemble_result.confidence, 4),
            )
            return

        # ----- 7. Route to AI orchestrator (or ML fallback) -----
        if self._orchestrator is not None:
            try:
                # Build the pipeline state with all context
                state = {
                    "ticker": ticker,
                    "ml_scores": {
                        "confidence": ensemble_result.confidence,
                        "raw_score": ensemble_result.raw_score,
                        "direction_bias": ensemble_result.direction_bias,
                        "contributions": ensemble_result.signal_contributions,
                    },
                    "regime": regime,
                    "iv_rank": iv_rank,
                    "sentiment_score": signal_inputs.sentiment_score,
                    "gex_data": {
                        "score": signal_inputs.gex_score,
                        "net_gex": signal_inputs.gex_net_gex,
                        "regime": signal_inputs.gex_regime,
                    },
                    "options_chain": [],
                    "account_summary": {},
                    "current_positions": [],
                    "portfolio_greeks": {},
                    "circuit_breaker_state": {},
                    "event_calendar": {},
                    "correlation_data": {},
                    "proposals": [],
                    "risk_evaluations": [],
                    "execution_results": [],
                    "journal_entries": [],
                    "errors": [],
                    "step": "start",
                    "should_execute": True,
                    "fallback_mode": False,
                }

                # Populate account summary if available
                if self._account_manager is not None:
                    try:
                        summary = await self._account_manager.get_account_summary()
                        state["account_summary"] = {
                            "net_liquidation": summary.net_liquidation,
                            "buying_power": getattr(summary, "buying_power", 0.0),
                        }
                        positions = await self._account_manager.get_positions()
                        state["current_positions"] = [
                            {
                                "ticker": getattr(p, "ticker", ""),
                                "strategy": getattr(p, "strategy", ""),
                            }
                            for p in positions
                        ]
                    except Exception:
                        self._log.warning("account_data_fetch_failed", ticker=ticker)

                # Populate circuit breaker state
                if self._circuit_breaker is not None:
                    allowed, reason_msg, size_mult = (
                        self._circuit_breaker.is_trading_allowed()
                    )
                    state["circuit_breaker_state"] = {
                        "allowed": allowed,
                        "reason": reason_msg,
                        "size_multiplier": size_mult,
                        "level": getattr(
                            self._circuit_breaker, "current_level", "NORMAL"
                        ),
                    }
                    if not allowed:
                        self._log.info(
                            "circuit_breaker_blocking_trade",
                            ticker=ticker,
                            level=state["circuit_breaker_state"]["level"],
                        )
                        return

                # Run through AI pipeline with automatic fallback
                result_state = await self._orchestrator.run_with_fallback(state)

                # Log results
                exec_results = result_state.get("execution_results", [])
                errors = result_state.get("errors", [])
                if exec_results:
                    self._log.info(
                        "trade_executed",
                        ticker=ticker,
                        results=len(exec_results),
                    )
                    await self._send_notification(
                        f"TRADE EXECUTED: {ticker} — "
                        f"confidence {ensemble_result.confidence:.2f}, "
                        f"regime {regime}"
                    )
                elif errors:
                    self._log.warning(
                        "pipeline_errors",
                        ticker=ticker,
                        errors=errors,
                    )

            except Exception:
                self._log.exception("orchestrator_failed", ticker=ticker)
        else:
            # Pure ML fallback — log the opportunity for manual review
            self._log.warning(
                "high_confidence_signal_no_orchestrator",
                ticker=ticker,
                confidence=round(ensemble_result.confidence, 4),
                direction=ensemble_result.direction_bias,
                regime=regime,
            )
            await self._send_notification(
                f"SIGNAL (no AI): {ticker} — "
                f"confidence {ensemble_result.confidence:.2f}, "
                f"direction {ensemble_result.direction_bias}, "
                f"regime {regime}"
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
