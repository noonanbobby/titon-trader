"""Unit tests for src/broker/market_data.py — MarketDataManager.

Tests cover:
    - _INDEX_SYMBOLS constant correctness (VIX, SPX, NDX, etc.)
    - Index routing in get_snapshot() — calls create_index for VIX
    - Index routing in get_historical_bars() — calls create_index for VIX
    - Index routing in get_historical_iv() — calls create_index for VIX
    - Stock routing preserved for non-index tickers (AAPL, MSFT)
    - MarketSnapshot Pydantic model construction
    - OptionGreeks model defaults
    - IVSurface model construction
    - _safe_float helper function
    - _dte and _expiry_to_date helpers
    - subscription tracking and limits
"""

from __future__ import annotations

import math
from datetime import date
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest

from src.broker.market_data import (
    _INDEX_SYMBOLS,
    IVSurface,
    MarketDataManager,
    MarketSnapshot,
    OptionGreeks,
    _dte,
    _expiry_to_date,
    _safe_float,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_ib() -> MagicMock:
    """Return a mock ib_async.IB instance."""
    ib = MagicMock()
    ib.reqMktData = MagicMock()
    ib.cancelMktData = MagicMock()
    ib.reqHistoricalDataAsync = AsyncMock(return_value=[])
    ib.qualifyContractsAsync = AsyncMock()
    return ib


@pytest.fixture()
def mock_factory() -> MagicMock:
    """Return a mock ContractFactory."""
    factory = MagicMock()
    factory.create_stock = AsyncMock()
    factory.create_index = AsyncMock()
    factory.get_option_chain_params = AsyncMock(return_value=[])
    return factory


@pytest.fixture()
def mdm(mock_ib: MagicMock, mock_factory: MagicMock) -> MarketDataManager:
    """Return a MarketDataManager wired to mocks."""
    return MarketDataManager(mock_ib, mock_factory)


# ---------------------------------------------------------------------------
# _INDEX_SYMBOLS constant
# ---------------------------------------------------------------------------


class TestIndexSymbols:
    """Tests for the _INDEX_SYMBOLS module-level constant."""

    def test_vix_in_index_symbols(self) -> None:
        """VIX must be in _INDEX_SYMBOLS — this is the primary fix."""
        assert "VIX" in _INDEX_SYMBOLS

    def test_spx_in_index_symbols(self) -> None:
        """SPX is a CBOE index, not a stock."""
        assert "SPX" in _INDEX_SYMBOLS

    def test_ndx_in_index_symbols(self) -> None:
        """NDX (NASDAQ 100 index) should be routed as Index."""
        assert "NDX" in _INDEX_SYMBOLS

    def test_rut_in_index_symbols(self) -> None:
        """RUT (Russell 2000 index) should be routed as Index."""
        assert "RUT" in _INDEX_SYMBOLS

    def test_vvix_in_index_symbols(self) -> None:
        """VVIX (volatility of VIX) should be routed as Index."""
        assert "VVIX" in _INDEX_SYMBOLS

    def test_aapl_not_in_index_symbols(self) -> None:
        """AAPL is a stock, not an index."""
        assert "AAPL" not in _INDEX_SYMBOLS

    def test_msft_not_in_index_symbols(self) -> None:
        """MSFT is a stock, not an index."""
        assert "MSFT" not in _INDEX_SYMBOLS

    def test_is_frozenset(self) -> None:
        """_INDEX_SYMBOLS should be immutable."""
        assert isinstance(_INDEX_SYMBOLS, frozenset)

    def test_expected_count(self) -> None:
        """Should contain exactly 9 known CBOE indices."""
        assert len(_INDEX_SYMBOLS) == 9


# ---------------------------------------------------------------------------
# Index routing in get_snapshot
# ---------------------------------------------------------------------------


class TestGetSnapshotRouting:
    """Tests that get_snapshot() routes index tickers to create_index()."""

    @pytest.mark.asyncio()
    async def test_vix_uses_create_index(
        self, mdm: MarketDataManager, mock_factory: MagicMock, mock_ib: MagicMock
    ) -> None:
        """get_snapshot('VIX') should call create_index, not create_stock."""
        mock_contract = MagicMock()
        mock_factory.create_index.return_value = mock_contract

        mock_ticker = MagicMock()
        mock_ticker.bid = 18.5
        mock_ticker.ask = 19.0
        mock_ticker.last = 18.75
        mock_ticker.volume = 1000
        mock_ib.reqMktData.return_value = mock_ticker

        with patch("src.broker.market_data.asyncio.sleep", new_callable=AsyncMock):
            await mdm.get_snapshot("VIX")

        mock_factory.create_index.assert_called_once_with("VIX")
        mock_factory.create_stock.assert_not_called()

    @pytest.mark.asyncio()
    async def test_aapl_uses_create_stock(
        self, mdm: MarketDataManager, mock_factory: MagicMock, mock_ib: MagicMock
    ) -> None:
        """get_snapshot('AAPL') should call create_stock, not create_index."""
        mock_contract = MagicMock()
        mock_factory.create_stock.return_value = mock_contract

        mock_ticker = MagicMock()
        mock_ticker.bid = 175.0
        mock_ticker.ask = 175.5
        mock_ticker.last = 175.25
        mock_ticker.volume = 50000
        mock_ib.reqMktData.return_value = mock_ticker

        with patch("src.broker.market_data.asyncio.sleep", new_callable=AsyncMock):
            await mdm.get_snapshot("AAPL")

        mock_factory.create_stock.assert_called_once_with("AAPL")
        mock_factory.create_index.assert_not_called()


# ---------------------------------------------------------------------------
# Index routing in get_historical_bars
# ---------------------------------------------------------------------------


class TestGetHistoricalBarsRouting:
    """Tests that get_historical_bars() routes index tickers correctly."""

    @pytest.mark.asyncio()
    async def test_vix_uses_create_index(
        self, mdm: MarketDataManager, mock_factory: MagicMock, mock_ib: MagicMock
    ) -> None:
        """get_historical_bars('VIX') should call create_index."""
        mock_contract = MagicMock()
        mock_factory.create_index.return_value = mock_contract
        mock_ib.reqHistoricalDataAsync.return_value = []

        await mdm.get_historical_bars("VIX", duration="100 D")

        mock_factory.create_index.assert_called_once_with("VIX")
        mock_factory.create_stock.assert_not_called()

    @pytest.mark.asyncio()
    async def test_spx_uses_create_index(
        self, mdm: MarketDataManager, mock_factory: MagicMock, mock_ib: MagicMock
    ) -> None:
        """get_historical_bars('SPX') should call create_index."""
        mock_contract = MagicMock()
        mock_factory.create_index.return_value = mock_contract
        mock_ib.reqHistoricalDataAsync.return_value = []

        await mdm.get_historical_bars("SPX")

        mock_factory.create_index.assert_called_once_with("SPX")
        mock_factory.create_stock.assert_not_called()

    @pytest.mark.asyncio()
    async def test_msft_uses_create_stock(
        self, mdm: MarketDataManager, mock_factory: MagicMock, mock_ib: MagicMock
    ) -> None:
        """get_historical_bars('MSFT') should call create_stock."""
        mock_contract = MagicMock()
        mock_factory.create_stock.return_value = mock_contract
        mock_ib.reqHistoricalDataAsync.return_value = []

        await mdm.get_historical_bars("MSFT")

        mock_factory.create_stock.assert_called_once_with("MSFT")
        mock_factory.create_index.assert_not_called()

    @pytest.mark.asyncio()
    async def test_vix_qualification_failure_returns_none(
        self, mdm: MarketDataManager, mock_factory: MagicMock
    ) -> None:
        """If VIX qualification fails, get_historical_bars should return None."""
        mock_factory.create_index.side_effect = ValueError("Failed")

        result = await mdm.get_historical_bars("VIX")
        assert result is None

    @pytest.mark.asyncio()
    async def test_returns_dataframe_on_success(
        self, mdm: MarketDataManager, mock_factory: MagicMock, mock_ib: MagicMock
    ) -> None:
        """Successful historical bar fetch should return a DataFrame."""
        mock_contract = MagicMock()
        mock_factory.create_stock.return_value = mock_contract

        mock_bar = MagicMock()
        mock_bar.date = date(2026, 1, 15)
        mock_bar.open = 175.0
        mock_bar.high = 178.0
        mock_bar.low = 174.0
        mock_bar.close = 177.0
        mock_bar.volume = 50000
        mock_ib.reqHistoricalDataAsync.return_value = [mock_bar]

        result = await mdm.get_historical_bars("AAPL")
        assert result is not None
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1
        assert "close" in result.columns


# ---------------------------------------------------------------------------
# Index routing in get_historical_iv
# ---------------------------------------------------------------------------


class TestGetHistoricalIvRouting:
    """Tests that get_historical_iv() routes index tickers correctly."""

    @pytest.mark.asyncio()
    async def test_vix_uses_create_index(
        self, mdm: MarketDataManager, mock_factory: MagicMock, mock_ib: MagicMock
    ) -> None:
        """get_historical_iv('VIX') should call create_index."""
        mock_contract = MagicMock()
        mock_factory.create_index.return_value = mock_contract
        mock_ib.reqHistoricalDataAsync.return_value = []

        await mdm.get_historical_iv("VIX")

        mock_factory.create_index.assert_called_once_with("VIX")
        mock_factory.create_stock.assert_not_called()

    @pytest.mark.asyncio()
    async def test_aapl_uses_create_stock(
        self, mdm: MarketDataManager, mock_factory: MagicMock, mock_ib: MagicMock
    ) -> None:
        """get_historical_iv('AAPL') should call create_stock."""
        mock_contract = MagicMock()
        mock_factory.create_stock.return_value = mock_contract
        mock_ib.reqHistoricalDataAsync.return_value = []

        await mdm.get_historical_iv("AAPL")

        mock_factory.create_stock.assert_called_once_with("AAPL")
        mock_factory.create_index.assert_not_called()


# ---------------------------------------------------------------------------
# Pydantic model tests
# ---------------------------------------------------------------------------


class TestMarketSnapshot:
    """Tests for the MarketSnapshot Pydantic model."""

    def test_default_nan_values(self) -> None:
        """Unset price fields should default to NaN."""
        snap = MarketSnapshot(ticker="AAPL")
        assert math.isnan(snap.bid)
        assert math.isnan(snap.ask)
        assert math.isnan(snap.last)

    def test_with_values(self) -> None:
        """Snapshot with explicit values should store them correctly."""
        snap = MarketSnapshot(
            ticker="AAPL", bid=175.0, ask=175.5, last=175.25, volume=50000
        )
        assert snap.ticker == "AAPL"
        assert snap.bid == 175.0
        assert snap.ask == 175.5


class TestOptionGreeks:
    """Tests for the OptionGreeks Pydantic model."""

    def test_defaults_to_zero(self) -> None:
        """All Greeks should default to 0.0."""
        greeks = OptionGreeks(con_id=12345)
        assert greeks.delta == 0.0
        assert greeks.gamma == 0.0
        assert greeks.theta == 0.0
        assert greeks.vega == 0.0
        assert greeks.implied_vol == 0.0

    def test_with_values(self) -> None:
        """Greeks with explicit values should store them correctly."""
        greeks = OptionGreeks(
            con_id=12345, delta=0.55, gamma=0.03, theta=-0.12, vega=0.15
        )
        assert greeks.delta == 0.55
        assert greeks.gamma == 0.03


class TestIVSurface:
    """Tests for the IVSurface model."""

    def test_empty_surface(self) -> None:
        """Empty IV surface should have no data entries."""
        surface = IVSurface(ticker="AAPL")
        assert surface.ticker == "AAPL"
        assert len(surface.data) == 0


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------


class TestSafeFloat:
    """Tests for the _safe_float helper."""

    def test_normal_float(self) -> None:
        assert _safe_float(3.14) == 3.14

    def test_none_returns_default(self) -> None:
        assert _safe_float(None) == 0.0

    def test_none_custom_default(self) -> None:
        assert _safe_float(None, -1.0) == -1.0

    def test_nan_returns_default(self) -> None:
        assert _safe_float(float("nan")) == 0.0

    def test_inf_returns_default(self) -> None:
        assert _safe_float(float("inf")) == 0.0

    def test_string_returns_default(self) -> None:
        assert _safe_float("not_a_number") == 0.0

    def test_int_coercion(self) -> None:
        assert _safe_float(42) == 42.0


class TestExpiryHelpers:
    """Tests for _expiry_to_date and _dte helpers."""

    def test_expiry_to_date(self) -> None:
        """YYYYMMDD string should parse to date correctly."""
        result = _expiry_to_date("20260320")
        assert result == date(2026, 3, 20)

    def test_dte_future_date(self) -> None:
        """DTE for a future date should be positive."""
        ref = date(2026, 1, 1)
        result = _dte("20301231", ref)
        assert result > 0

    def test_dte_past_date(self) -> None:
        """DTE for a past date should be negative."""
        ref = date(2026, 6, 1)
        result = _dte("20260101", ref)
        assert result < 0


# ---------------------------------------------------------------------------
# Subscription tracking
# ---------------------------------------------------------------------------


class TestSubscriptionTracking:
    """Tests for MarketDataManager subscription management."""

    def test_initial_subscription_count_zero(self, mdm: MarketDataManager) -> None:
        """New manager should have zero subscriptions."""
        assert mdm.subscription_count == 0

    def test_subscribed_tickers_empty(self, mdm: MarketDataManager) -> None:
        """New manager should have no subscribed tickers."""
        assert mdm.subscribed_tickers == []
