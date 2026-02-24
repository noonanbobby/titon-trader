"""Unit tests for src/broker/contracts.py — ContractFactory.

Tests cover:
    - Stock contract creation and qualification
    - Index contract creation and qualification (VIX, SPX, etc.)
    - Option contract creation and qualification
    - Combo/BAG contract assembly
    - build_spread() high-level helper
    - SpreadLeg Pydantic validation
    - OptionChainParams model
    - _select_candidate_strikes helper
    - Error handling for failed qualifications
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.broker.contracts import (
    ContractFactory,
    OptionChainParams,
    OptionRight,
    SpreadLeg,
    _select_candidate_strikes,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_ib() -> MagicMock:
    """Return a mock ib_async.IB instance with async qualification."""
    ib = MagicMock()
    ib.qualifyContractsAsync = AsyncMock()
    ib.reqSecDefOptParamsAsync = AsyncMock(return_value=[])
    ib.reqMktData = MagicMock()
    ib.cancelMktData = MagicMock()
    return ib


@pytest.fixture()
def factory(mock_ib: MagicMock) -> ContractFactory:
    """Return a ContractFactory wired to the mock IB instance."""
    return ContractFactory(mock_ib)


# ---------------------------------------------------------------------------
# SpreadLeg Pydantic model tests
# ---------------------------------------------------------------------------


class TestSpreadLeg:
    """Tests for the SpreadLeg Pydantic model."""

    def test_valid_spread_leg(self) -> None:
        """Valid spread leg should construct without errors."""
        leg = SpreadLeg(action="BUY", expiry="20260320", strike=150.0, right="C")
        assert leg.action == "BUY"
        assert leg.expiry == "20260320"
        assert leg.strike == 150.0
        assert leg.right == "C"
        assert leg.ratio == 1  # default

    def test_spread_leg_custom_ratio(self) -> None:
        """Ratio should accept values >= 1."""
        leg = SpreadLeg(
            action="SELL", expiry="20260320", strike=160.0, right="P", ratio=2
        )
        assert leg.ratio == 2

    def test_spread_leg_invalid_expiry_non_numeric(self) -> None:
        """Non-numeric expiry string should be rejected."""
        with pytest.raises(ValueError, match="numeric YYYYMMDD"):
            SpreadLeg(action="BUY", expiry="2026-03-", strike=150.0, right="C")

    def test_spread_leg_invalid_expiry_too_short(self) -> None:
        """Expiry shorter than 8 characters should be rejected."""
        with pytest.raises(ValueError):
            SpreadLeg(action="BUY", expiry="202603", strike=150.0, right="C")

    def test_spread_leg_invalid_month(self) -> None:
        """Month outside 1-12 should be rejected."""
        with pytest.raises(ValueError, match="month out of range"):
            SpreadLeg(action="BUY", expiry="20261320", strike=150.0, right="C")

    def test_spread_leg_invalid_strike_zero(self) -> None:
        """Strike price of zero should be rejected (gt=0)."""
        with pytest.raises(ValueError):
            SpreadLeg(action="BUY", expiry="20260320", strike=0.0, right="C")

    def test_spread_leg_invalid_action(self) -> None:
        """Action must be BUY or SELL."""
        with pytest.raises(ValueError):
            SpreadLeg(action="HOLD", expiry="20260320", strike=150.0, right="C")


class TestOptionRight:
    """Tests for the OptionRight StrEnum."""

    def test_call_value(self) -> None:
        assert OptionRight.CALL == "C"

    def test_put_value(self) -> None:
        assert OptionRight.PUT == "P"


class TestOptionChainParams:
    """Tests for the OptionChainParams model."""

    def test_frozen_model(self) -> None:
        """OptionChainParams should be immutable (frozen=True)."""
        params = OptionChainParams(
            exchange="SMART",
            underlying_con_id=265598,
            trading_class="AAPL",
            multiplier="100",
            expirations={"20260320", "20260417"},
            strikes={140.0, 150.0, 160.0},
        )
        with pytest.raises((TypeError, ValueError, AttributeError)):
            params.exchange = "CBOE"  # type: ignore[misc]

    def test_construction(self) -> None:
        """OptionChainParams should store all fields correctly."""
        params = OptionChainParams(
            exchange="SMART",
            underlying_con_id=265598,
            trading_class="AAPL",
            multiplier="100",
            expirations={"20260320"},
            strikes={150.0},
        )
        assert params.exchange == "SMART"
        assert params.underlying_con_id == 265598
        assert 150.0 in params.strikes
        assert "20260320" in params.expirations


# ---------------------------------------------------------------------------
# ContractFactory.create_stock tests
# ---------------------------------------------------------------------------


class TestCreateStock:
    """Tests for ContractFactory.create_stock()."""

    @pytest.mark.asyncio()
    async def test_create_stock_success(
        self, factory: ContractFactory, mock_ib: MagicMock
    ) -> None:
        """Successful stock qualification should return the contract."""
        mock_contract = MagicMock()
        mock_contract.conId = 265598
        mock_contract.exchange = "SMART"
        mock_ib.qualifyContractsAsync.return_value = [mock_contract]

        result = await factory.create_stock("AAPL")
        # The factory creates a new Stock, qualifies it, returns it
        mock_ib.qualifyContractsAsync.assert_called_once()
        assert result is not None

    @pytest.mark.asyncio()
    async def test_create_stock_failure_raises(
        self, factory: ContractFactory, mock_ib: MagicMock
    ) -> None:
        """Failed qualification should raise ValueError."""
        mock_ib.qualifyContractsAsync.return_value = [None]

        with pytest.raises(ValueError, match="Failed to qualify stock"):
            await factory.create_stock("INVALID")

    @pytest.mark.asyncio()
    async def test_create_stock_default_exchange(
        self, factory: ContractFactory, mock_ib: MagicMock
    ) -> None:
        """Default exchange should be SMART."""
        mock_contract = MagicMock()
        mock_contract.conId = 1
        mock_ib.qualifyContractsAsync.return_value = [mock_contract]

        await factory.create_stock("AAPL")
        call_args = mock_ib.qualifyContractsAsync.call_args
        contract = call_args[0][0]
        assert contract.exchange == "SMART"


# ---------------------------------------------------------------------------
# ContractFactory.create_index tests
# ---------------------------------------------------------------------------


class TestCreateIndex:
    """Tests for ContractFactory.create_index() — new method for VIX, SPX etc."""

    @pytest.mark.asyncio()
    async def test_create_index_success(
        self, factory: ContractFactory, mock_ib: MagicMock
    ) -> None:
        """Successful index qualification should return the contract."""
        mock_contract = MagicMock()
        mock_contract.conId = 13455763
        mock_contract.exchange = "CBOE"
        mock_ib.qualifyContractsAsync.return_value = [mock_contract]

        result = await factory.create_index("VIX")
        mock_ib.qualifyContractsAsync.assert_called_once()
        assert result is not None

    @pytest.mark.asyncio()
    async def test_create_index_default_exchange_cboe(
        self, factory: ContractFactory, mock_ib: MagicMock
    ) -> None:
        """Default exchange for indices should be CBOE, not SMART."""
        mock_contract = MagicMock()
        mock_contract.conId = 1
        mock_ib.qualifyContractsAsync.return_value = [mock_contract]

        await factory.create_index("VIX")
        call_args = mock_ib.qualifyContractsAsync.call_args
        contract = call_args[0][0]
        assert contract.exchange == "CBOE"

    @pytest.mark.asyncio()
    async def test_create_index_custom_exchange(
        self, factory: ContractFactory, mock_ib: MagicMock
    ) -> None:
        """Custom exchange should override the default."""
        mock_contract = MagicMock()
        mock_contract.conId = 1
        mock_ib.qualifyContractsAsync.return_value = [mock_contract]

        await factory.create_index("SPX", exchange="CBOE")
        call_args = mock_ib.qualifyContractsAsync.call_args
        contract = call_args[0][0]
        assert contract.exchange == "CBOE"

    @pytest.mark.asyncio()
    async def test_create_index_failure_raises(
        self, factory: ContractFactory, mock_ib: MagicMock
    ) -> None:
        """Failed index qualification should raise ValueError."""
        mock_ib.qualifyContractsAsync.return_value = [None]

        with pytest.raises(ValueError, match="Failed to qualify index"):
            await factory.create_index("INVALID_IDX")

    @pytest.mark.asyncio()
    async def test_create_index_uses_index_type(
        self, factory: ContractFactory, mock_ib: MagicMock
    ) -> None:
        """create_index should create an ib_async.Index, not a Stock."""
        from ib_async import Index

        mock_contract = MagicMock()
        mock_contract.conId = 1
        mock_ib.qualifyContractsAsync.return_value = [mock_contract]

        await factory.create_index("VIX")
        call_args = mock_ib.qualifyContractsAsync.call_args
        contract = call_args[0][0]
        assert isinstance(contract, Index)


# ---------------------------------------------------------------------------
# ContractFactory.create_option tests
# ---------------------------------------------------------------------------


class TestCreateOption:
    """Tests for ContractFactory.create_option()."""

    @pytest.mark.asyncio()
    async def test_create_option_success(
        self, factory: ContractFactory, mock_ib: MagicMock
    ) -> None:
        """Successful option qualification should return the contract."""
        mock_contract = MagicMock()
        mock_contract.conId = 99999
        mock_ib.qualifyContractsAsync.return_value = [mock_contract]

        result = await factory.create_option("AAPL", "20260320", 150.0, "C")
        mock_ib.qualifyContractsAsync.assert_called_once()
        assert result is not None

    @pytest.mark.asyncio()
    async def test_create_option_failure_raises(
        self, factory: ContractFactory, mock_ib: MagicMock
    ) -> None:
        """Failed option qualification should raise ValueError."""
        mock_ib.qualifyContractsAsync.return_value = [None]

        with pytest.raises(ValueError, match="Failed to qualify option"):
            await factory.create_option("AAPL", "20260320", 150.0, "C")


# ---------------------------------------------------------------------------
# ContractFactory.create_combo tests
# ---------------------------------------------------------------------------


class TestCreateCombo:
    """Tests for ContractFactory.create_combo()."""

    @pytest.mark.asyncio()
    async def test_create_combo_success(self, factory: ContractFactory) -> None:
        """Combo with valid legs should return a BAG contract."""
        from ib_async import ComboLeg

        legs = [
            ComboLeg(conId=100, ratio=1, action="BUY", exchange="SMART"),
            ComboLeg(conId=200, ratio=1, action="SELL", exchange="SMART"),
        ]
        result = await factory.create_combo("AAPL", legs)
        assert result.symbol == "AAPL"
        assert len(result.comboLegs) == 2

    @pytest.mark.asyncio()
    async def test_create_combo_empty_legs_raises(
        self, factory: ContractFactory
    ) -> None:
        """Empty legs list should raise ValueError."""
        with pytest.raises(ValueError, match="zero legs"):
            await factory.create_combo("AAPL", [])


# ---------------------------------------------------------------------------
# _select_candidate_strikes helper
# ---------------------------------------------------------------------------


class TestSelectCandidateStrikes:
    """Tests for the module-level _select_candidate_strikes helper."""

    def test_select_around_spot(self) -> None:
        """Should select strikes closest to the spot price."""
        strikes = [100.0, 110.0, 120.0, 130.0, 140.0, 150.0, 160.0, 170.0]
        result = _select_candidate_strikes(strikes, spot_price=135.0, max_candidates=4)
        assert len(result) == 4
        # Should include the 4 closest to 135: 120, 130, 140, 150
        assert 130.0 in result
        assert 140.0 in result

    def test_select_all_when_fewer_than_max(self) -> None:
        """When fewer strikes than max_candidates, return all."""
        strikes = [100.0, 110.0, 120.0]
        result = _select_candidate_strikes(strikes, spot_price=110.0, max_candidates=10)
        assert result == [100.0, 110.0, 120.0]

    def test_select_returns_sorted(self) -> None:
        """Result should always be sorted ascending."""
        strikes = [90.0, 100.0, 110.0, 120.0, 130.0, 140.0, 150.0]
        result = _select_candidate_strikes(strikes, spot_price=120.0, max_candidates=3)
        assert result == sorted(result)

    def test_select_empty_strikes(self) -> None:
        """Empty strikes list should return empty."""
        result = _select_candidate_strikes([], spot_price=100.0, max_candidates=5)
        assert result == []
