"""Cross-asset signal generation for Project Titan.

Fetches macroeconomic data from FRED, VIX term structure, and
cross-asset prices (DXY, copper, gold) to produce a composite
risk-on / risk-off signal.  Each data source is scored individually
and combined into a single score ranging from -1.0 (maximum risk-off)
to +1.0 (maximum risk-on).

Usage::

    from src.signals.cross_asset import CrossAssetSignalGenerator

    generator = CrossAssetSignalGenerator(
        fred_api_key="your_key",
        polygon_api_key="your_key",
    )
    macro = await generator.fetch_macro_data()
    vix_ts = await generator.fetch_vix_term_structure()
    cross = await generator.fetch_cross_asset_prices()
    signal = generator.calculate_cross_asset_signal(macro, vix_ts, cross)
    print(signal.score, signal.bias)
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import TYPE_CHECKING, Any

import httpx
from pydantic import BaseModel, Field

from src.utils.logging import get_logger

if TYPE_CHECKING:
    import structlog

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FRED_BASE_URL: str = "https://api.stlouisfed.org/fred/series/observations"
POLYGON_BASE_URL: str = "https://api.polygon.io"

# FRED series IDs
FRED_SERIES_DGS2: str = "DGS2"
FRED_SERIES_DGS10: str = "DGS10"
FRED_SERIES_FED_FUNDS: str = "DFF"
FRED_SERIES_HY_OAS: str = "BAMLH0A0HYM2"

# Polygon ticker symbols
POLYGON_VIX_TICKER: str = "I:VIX"
POLYGON_VIX3M_TICKER: str = "I:VIX3M"
POLYGON_DXY_TICKER: str = "I:DXY"
POLYGON_COPPER_TICKER: str = "C:HGUSD"
POLYGON_GOLD_TICKER: str = "C:XAUUSD"

# HTTP client defaults
HTTP_TIMEOUT_SECONDS: float = 30.0
FRED_LOOKBACK_DAYS: int = 14

# Component weights for composite score
WEIGHT_YIELD_CURVE: float = 0.25
WEIGHT_CREDIT_SPREAD: float = 0.25
WEIGHT_VIX_TERM_STRUCTURE: float = 0.25
WEIGHT_DXY: float = 0.10
WEIGHT_COPPER_GOLD: float = 0.15

# DXY scoring thresholds
DXY_STRONG_DOLLAR: float = 105.0
DXY_WEAK_DOLLAR: float = 95.0

# Copper/Gold ratio scoring thresholds
COPPER_GOLD_RATIO_HIGH: float = 0.00015
COPPER_GOLD_RATIO_LOW: float = 0.00010


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class VIXStructure(StrEnum):
    """VIX term structure state."""

    CONTANGO = "contango"
    BACKWARDATION = "backwardation"


class MarketBias(StrEnum):
    """Overall market bias derived from cross-asset signals."""

    RISK_ON = "risk_on"
    NEUTRAL = "neutral"
    RISK_OFF = "risk_off"


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class MacroSnapshot(BaseModel):
    """Snapshot of macroeconomic indicators from FRED.

    Attributes:
        yield_2y: 2-Year Treasury yield (percent).
        yield_10y: 10-Year Treasury yield (percent).
        spread_2y10y: 10Y minus 2Y yield spread (percent).
        fed_funds_rate: Effective Federal Funds rate (percent).
        hy_oas: ICE BofA High Yield Option-Adjusted Spread (percent).
        timestamp: When the snapshot was taken.
    """

    yield_2y: float
    yield_10y: float
    spread_2y10y: float
    fed_funds_rate: float
    hy_oas: float
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class VIXTermStructure(BaseModel):
    """VIX term structure snapshot.

    Attributes:
        vix: Current VIX (30-day implied volatility) level.
        vix3m: Current VIX3M (3-month implied volatility) level.
        ratio: VIX / VIX3M ratio.  Values below 1.0 indicate contango
            (normal); values above 1.0 indicate backwardation (stress).
        structure: ``"contango"`` or ``"backwardation"``.
    """

    vix: float
    vix3m: float
    ratio: float
    structure: str  # VIXStructure value


class CrossAssetPrices(BaseModel):
    """Cross-asset price snapshot.

    Attributes:
        dxy: US Dollar Index level.
        copper: Copper price (USD per pound).
        gold: Gold price (USD per ounce).
        copper_gold_ratio: Copper price divided by gold price.
    """

    dxy: float
    copper: float
    gold: float
    copper_gold_ratio: float


class CrossAssetSignal(BaseModel):
    """Composite cross-asset signal with per-component breakdowns.

    The ``score`` ranges from -1.0 (maximum risk-off) to +1.0
    (maximum risk-on).

    Attributes:
        score: Composite cross-asset score (-1.0 to +1.0).
        yield_curve_score: Yield curve contribution (-1.0 to +1.0).
        credit_score: Credit spread contribution (-1.0 to +1.0).
        vix_ts_score: VIX term structure contribution (-1.0 to +1.0).
        dxy_score: DXY contribution (-1.0 to +1.0).
        copper_gold_score: Copper/gold ratio contribution (-1.0 to +1.0).
        macro: The underlying macro data snapshot.
        bias: Overall market bias classification.
        timestamp: When the signal was generated.
    """

    score: float = Field(ge=-1.0, le=1.0)
    yield_curve_score: float
    credit_score: float
    vix_ts_score: float
    dxy_score: float
    copper_gold_score: float
    macro: MacroSnapshot
    bias: str  # MarketBias value
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


# ---------------------------------------------------------------------------
# CrossAssetSignalGenerator
# ---------------------------------------------------------------------------


class CrossAssetSignalGenerator:
    """Generates cross-asset risk-on / risk-off signals.

    Fetches macroeconomic data from FRED, VIX term structure data, and
    cross-asset prices (DXY, copper, gold) to produce a composite
    signal score.  Each component is scored individually and combined
    using fixed weights.

    Args:
        fred_api_key: API key for the FRED (Federal Reserve Economic
            Data) service.
        polygon_api_key: API key for the Polygon.io market data
            service.
    """

    def __init__(self, fred_api_key: str, polygon_api_key: str) -> None:
        self._fred_api_key: str = fred_api_key
        self._polygon_api_key: str = polygon_api_key
        self._log: structlog.stdlib.BoundLogger = get_logger("signals.cross_asset")

        self._log.info(
            "cross_asset_generator_initialized",
            fred_key_set=bool(fred_api_key),
            polygon_key_set=bool(polygon_api_key),
        )

    # ------------------------------------------------------------------
    # FRED macro data
    # ------------------------------------------------------------------

    async def fetch_macro_data(self) -> MacroSnapshot:
        """Fetch macroeconomic indicators from the FRED API.

        Retrieves the most recent observations for:

        - 2-Year Treasury yield (DGS2)
        - 10-Year Treasury yield (DGS10)
        - Effective Federal Funds rate (DFF)
        - ICE BofA High Yield Option-Adjusted Spread (BAMLH0A0HYM2)

        The 2Y/10Y spread is calculated as ``DGS10 - DGS2``.

        Returns:
            A :class:`MacroSnapshot` with the latest available values.

        Raises:
            httpx.HTTPStatusError: If any FRED API call returns a
                non-2xx status code.
        """
        self._log.info("fetching_macro_data")

        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_SECONDS) as client:
            yield_2y = await self._fetch_fred_series(client, FRED_SERIES_DGS2)
            yield_10y = await self._fetch_fred_series(client, FRED_SERIES_DGS10)
            fed_funds = await self._fetch_fred_series(client, FRED_SERIES_FED_FUNDS)
            hy_oas = await self._fetch_fred_series(client, FRED_SERIES_HY_OAS)

        spread_2y10y = yield_10y - yield_2y

        snapshot = MacroSnapshot(
            yield_2y=round(yield_2y, 4),
            yield_10y=round(yield_10y, 4),
            spread_2y10y=round(spread_2y10y, 4),
            fed_funds_rate=round(fed_funds, 4),
            hy_oas=round(hy_oas, 4),
        )

        self._log.info(
            "macro_data_fetched",
            yield_2y=snapshot.yield_2y,
            yield_10y=snapshot.yield_10y,
            spread_2y10y=snapshot.spread_2y10y,
            fed_funds_rate=snapshot.fed_funds_rate,
            hy_oas=snapshot.hy_oas,
        )

        return snapshot

    async def _fetch_fred_series(
        self,
        client: httpx.AsyncClient,
        series_id: str,
    ) -> float:
        """Fetch the most recent numeric observation from a FRED series.

        Queries the FRED ``series/observations`` endpoint for the last
        ``FRED_LOOKBACK_DAYS`` days and returns the latest non-missing
        value.  FRED data may lag by one or more business days, so the
        lookback window ensures a value is found even on weekends and
        holidays.

        Args:
            client: An active :class:`httpx.AsyncClient`.
            series_id: The FRED series identifier (e.g. ``"DGS10"``).

        Returns:
            The most recent numeric observation as a float.

        Raises:
            httpx.HTTPStatusError: If the FRED API returns a non-2xx
                status code.
            ValueError: If no valid numeric observation is found
                within the lookback window.
        """
        end_date = datetime.now(UTC).strftime("%Y-%m-%d")
        start_date = (datetime.now(UTC) - timedelta(days=FRED_LOOKBACK_DAYS)).strftime(
            "%Y-%m-%d"
        )

        params: dict[str, str] = {
            "series_id": series_id,
            "api_key": self._fred_api_key,
            "file_type": "json",
            "observation_start": start_date,
            "observation_end": end_date,
            "sort_order": "desc",
            "limit": "5",
        }

        response = await client.get(FRED_BASE_URL, params=params)
        response.raise_for_status()

        data: dict[str, Any] = response.json()
        observations: list[dict[str, str]] = data.get("observations", [])

        for obs in observations:
            value_str = obs.get("value", ".")
            if value_str != "." and value_str:
                try:
                    value = float(value_str)
                    self._log.debug(
                        "fred_series_fetched",
                        series_id=series_id,
                        value=value,
                        date=obs.get("date"),
                    )
                    return value
                except ValueError:
                    continue

        msg = (
            f"No valid observation found for FRED series {series_id} "
            f"between {start_date} and {end_date}"
        )
        self._log.error("fred_series_no_data", series_id=series_id, message=msg)
        raise ValueError(msg)

    # ------------------------------------------------------------------
    # VIX term structure
    # ------------------------------------------------------------------

    async def fetch_vix_term_structure(self) -> VIXTermStructure:
        """Fetch VIX and VIX3M to determine the term structure state.

        Uses the Polygon.io API to retrieve the most recent VIX and
        VIX3M values.  The VIX/VIX3M ratio determines whether the
        term structure is in contango (ratio < 1.0, normal) or
        backwardation (ratio > 1.0, elevated stress).

        Returns:
            A :class:`VIXTermStructure` with VIX, VIX3M, their ratio,
            and the structure classification.

        Raises:
            httpx.HTTPStatusError: On non-2xx API responses.
            ValueError: If VIX or VIX3M data cannot be retrieved.
        """
        self._log.info("fetching_vix_term_structure")

        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_SECONDS) as client:
            vix = await self._fetch_polygon_snapshot(client, POLYGON_VIX_TICKER)
            vix3m = await self._fetch_polygon_snapshot(client, POLYGON_VIX3M_TICKER)

        if vix3m == 0.0:
            self._log.warning("vix3m_is_zero", vix=vix, vix3m=vix3m)
            ratio = 1.0
        else:
            ratio = vix / vix3m

        structure = VIXStructure.BACKWARDATION if ratio > 1.0 else VIXStructure.CONTANGO

        result = VIXTermStructure(
            vix=round(vix, 2),
            vix3m=round(vix3m, 2),
            ratio=round(ratio, 4),
            structure=structure.value,
        )

        self._log.info(
            "vix_term_structure_fetched",
            vix=result.vix,
            vix3m=result.vix3m,
            ratio=result.ratio,
            structure=result.structure,
        )

        return result

    # ------------------------------------------------------------------
    # Cross-asset prices
    # ------------------------------------------------------------------

    async def fetch_cross_asset_prices(self) -> CrossAssetPrices:
        """Fetch DXY, copper, and gold prices from Polygon.io.

        Calculates the copper/gold ratio as a risk appetite indicator.
        A rising ratio signals economic optimism (risk-on); a falling
        ratio signals caution (risk-off).

        Returns:
            A :class:`CrossAssetPrices` with DXY, copper, gold, and
            the copper/gold ratio.

        Raises:
            httpx.HTTPStatusError: On non-2xx API responses.
            ValueError: If any price cannot be retrieved.
        """
        self._log.info("fetching_cross_asset_prices")

        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_SECONDS) as client:
            dxy = await self._fetch_polygon_snapshot(client, POLYGON_DXY_TICKER)
            copper = await self._fetch_polygon_snapshot(client, POLYGON_COPPER_TICKER)
            gold = await self._fetch_polygon_snapshot(client, POLYGON_GOLD_TICKER)

        if gold == 0.0:
            self._log.warning("gold_price_is_zero", copper=copper, gold=gold)
            copper_gold_ratio = 0.0
        else:
            copper_gold_ratio = copper / gold

        result = CrossAssetPrices(
            dxy=round(dxy, 2),
            copper=round(copper, 4),
            gold=round(gold, 2),
            copper_gold_ratio=round(copper_gold_ratio, 8),
        )

        self._log.info(
            "cross_asset_prices_fetched",
            dxy=result.dxy,
            copper=result.copper,
            gold=result.gold,
            copper_gold_ratio=result.copper_gold_ratio,
        )

        return result

    async def _fetch_polygon_snapshot(
        self,
        client: httpx.AsyncClient,
        ticker: str,
    ) -> float:
        """Fetch the latest closing/value for a Polygon.io ticker.

        Uses the Polygon ``/v2/aggs/ticker/{ticker}/prev`` endpoint
        to get the previous day's close.  For index tickers (prefixed
        with ``I:``), this returns the index value.

        Args:
            client: An active :class:`httpx.AsyncClient`.
            ticker: Polygon ticker symbol (e.g. ``"I:VIX"``,
                ``"C:XAUUSD"``).

        Returns:
            The latest closing price or index value.

        Raises:
            httpx.HTTPStatusError: On non-2xx API responses.
            ValueError: If no results are returned by the API.
        """
        url = f"{POLYGON_BASE_URL}/v2/aggs/ticker/{ticker}/prev"
        params: dict[str, str] = {
            "adjusted": "true",
            "apiKey": self._polygon_api_key,
        }

        response = await client.get(url, params=params)
        response.raise_for_status()

        data: dict[str, Any] = response.json()
        results: list[dict[str, Any]] = data.get("results", [])

        if not results:
            msg = f"No results from Polygon for ticker {ticker}"
            self._log.error("polygon_no_data", ticker=ticker, message=msg)
            raise ValueError(msg)

        close_price = float(results[0].get("c", 0.0))

        self._log.debug(
            "polygon_snapshot_fetched",
            ticker=ticker,
            close=close_price,
        )

        return close_price

    # ------------------------------------------------------------------
    # Composite signal calculation
    # ------------------------------------------------------------------

    def calculate_cross_asset_signal(
        self,
        macro: MacroSnapshot,
        vix_ts: VIXTermStructure,
        cross: CrossAssetPrices,
    ) -> CrossAssetSignal:
        """Calculate the composite cross-asset signal.

        Scores each component on a scale from approximately -1.0
        (risk-off) to +1.0 (risk-on) and combines them using fixed
        weights to produce a composite score.  The ``bias`` field
        classifies the composite as ``risk_on``, ``neutral``, or
        ``risk_off``.

        Component weights:

        - Yield curve (2Y/10Y spread): 25%
        - Credit spreads (HY OAS): 25%
        - VIX term structure: 25%
        - DXY: 10%
        - Copper/Gold ratio: 15%

        Args:
            macro: Macroeconomic snapshot from :meth:`fetch_macro_data`.
            vix_ts: VIX term structure from
                :meth:`fetch_vix_term_structure`.
            cross: Cross-asset prices from
                :meth:`fetch_cross_asset_prices`.

        Returns:
            A :class:`CrossAssetSignal` with the composite score and
            per-component breakdowns.
        """
        yc_score = self._score_yield_curve(macro.spread_2y10y)
        credit = self._score_credit_spreads(macro.hy_oas)
        vix_score = self._score_vix_term_structure(vix_ts.ratio)
        dxy_val = self._score_dxy(cross.dxy)
        cg_score = self._score_copper_gold(cross.copper_gold_ratio)

        composite = (
            WEIGHT_YIELD_CURVE * yc_score
            + WEIGHT_CREDIT_SPREAD * credit
            + WEIGHT_VIX_TERM_STRUCTURE * vix_score
            + WEIGHT_DXY * dxy_val
            + WEIGHT_COPPER_GOLD * cg_score
        )

        # Clamp to [-1.0, 1.0]
        composite = max(-1.0, min(1.0, composite))

        # Classify bias
        if composite > 0.2:
            bias = MarketBias.RISK_ON
        elif composite < -0.2:
            bias = MarketBias.RISK_OFF
        else:
            bias = MarketBias.NEUTRAL

        signal = CrossAssetSignal(
            score=round(composite, 4),
            yield_curve_score=round(yc_score, 4),
            credit_score=round(credit, 4),
            vix_ts_score=round(vix_score, 4),
            dxy_score=round(dxy_val, 4),
            copper_gold_score=round(cg_score, 4),
            macro=macro,
            bias=bias.value,
        )

        self._log.info(
            "cross_asset_signal_calculated",
            composite_score=signal.score,
            bias=signal.bias,
            yield_curve_score=signal.yield_curve_score,
            credit_score=signal.credit_score,
            vix_ts_score=signal.vix_ts_score,
            dxy_score=signal.dxy_score,
            copper_gold_score=signal.copper_gold_score,
        )

        return signal

    # ------------------------------------------------------------------
    # Individual component scoring
    # ------------------------------------------------------------------

    def _score_yield_curve(self, spread_2y10y: float) -> float:
        """Score the 2Y/10Y yield curve spread.

        A positive spread indicates a normal yield curve and is
        generally bullish for equities.  An inverted curve (negative
        spread) is a recession indicator.

        Scoring:

        - Spread > 1.0%: +0.8 (healthy steepening)
        - Spread 0.0--1.0%: +0.3 (normal)
        - Spread -0.5--0.0%: -0.5 (mild inversion)
        - Spread < -0.5%: -1.0 (deep inversion, recession signal)

        The function uses linear interpolation within each band for
        smooth transitions.

        Args:
            spread_2y10y: The 10-Year minus 2-Year Treasury yield
                spread in percentage points.

        Returns:
            Score from approximately -1.0 to +0.8.
        """
        if spread_2y10y > 1.0:
            score = 0.8
        elif spread_2y10y > 0.0:
            # Linear interpolation: 0.0 -> +0.3, 1.0 -> +0.8
            score = 0.3 + (spread_2y10y / 1.0) * 0.5
        elif spread_2y10y > -0.5:
            # Linear interpolation: 0.0 -> -0.5, -0.5 -> -1.0
            score = -0.5 + (spread_2y10y / -0.5) * -0.5
        else:
            score = -1.0

        self._log.debug(
            "yield_curve_scored",
            spread_2y10y=spread_2y10y,
            score=round(score, 4),
        )

        return score

    def _score_credit_spreads(self, hy_oas: float) -> float:
        """Score high-yield credit spreads (ICE BofA HY OAS).

        Tight credit spreads indicate risk appetite; widening spreads
        indicate stress and risk aversion.

        Scoring:

        - OAS < 3.0%: +0.5 (tight, risk-on)
        - OAS 3.0--5.0%: 0.0 (normal)
        - OAS 5.0--7.0%: -0.5 (widening, stress)
        - OAS > 7.0%: -1.0 (crisis-level credit stress)

        Linear interpolation is used within each band.

        Args:
            hy_oas: ICE BofA High Yield Option-Adjusted Spread in
                percentage points.

        Returns:
            Score from -1.0 to +0.5.
        """
        if hy_oas < 3.0:
            # Linear interpolation: lower OAS -> higher score, max +0.5
            # At OAS=1.5: +0.5, at OAS=3.0: 0.0
            score = 0.5 * (1.0 - (hy_oas / 3.0))
        elif hy_oas <= 5.0:
            # Linear interpolation: 3.0 -> 0.0, 5.0 -> -0.5
            score = -0.5 * ((hy_oas - 3.0) / 2.0)
        elif hy_oas <= 7.0:
            # Linear interpolation: 5.0 -> -0.5, 7.0 -> -1.0
            score = -0.5 - 0.5 * ((hy_oas - 5.0) / 2.0)
        else:
            score = -1.0

        self._log.debug(
            "credit_spread_scored",
            hy_oas=hy_oas,
            score=round(score, 4),
        )

        return score

    def _score_vix_term_structure(self, ratio: float) -> float:
        """Score the VIX/VIX3M ratio (term structure).

        In normal markets, VIX < VIX3M (contango), indicating that
        near-term volatility expectations are lower than longer-term.
        Backwardation (VIX > VIX3M) signals immediate stress.

        Scoring:

        - Ratio < 0.85: +0.5 (steep contango, complacent markets)
        - Ratio 0.85--1.0: +0.2 (normal contango)
        - Ratio 1.0--1.1: -0.3 (mild backwardation, caution)
        - Ratio > 1.1: -0.8 (deep backwardation, significant stress)

        Linear interpolation is used within each band.

        Args:
            ratio: VIX divided by VIX3M.

        Returns:
            Score from approximately -0.8 to +0.5.
        """
        if ratio < 0.85:
            score = 0.5
        elif ratio < 1.0:
            # Linear interpolation: 0.85 -> +0.5, 1.0 -> +0.2
            score = 0.5 - ((ratio - 0.85) / 0.15) * 0.3
        elif ratio < 1.1:
            # Linear interpolation: 1.0 -> -0.3, 1.1 -> -0.8
            score = -0.3 - ((ratio - 1.0) / 0.1) * 0.5
        else:
            score = -0.8

        self._log.debug(
            "vix_term_structure_scored",
            ratio=ratio,
            score=round(score, 4),
        )

        return score

    def _score_dxy(self, dxy: float) -> float:
        """Score the US Dollar Index (DXY).

        A very strong dollar can pressure equities (especially
        multinationals and emerging markets).  A moderately weak
        dollar is generally supportive.

        Scoring:

        - DXY > 105: -0.5 (strong dollar, equity headwind)
        - DXY 100--105: -0.2 (modestly strong)
        - DXY 95--100: +0.2 (neutral to supportive)
        - DXY < 95: +0.4 (weak dollar, supportive)

        Args:
            dxy: Current US Dollar Index level.

        Returns:
            Score from approximately -0.5 to +0.4.
        """
        if dxy > DXY_STRONG_DOLLAR:
            score = -0.5
        elif dxy > 100.0:
            # Linear interpolation: 100 -> -0.2, 105 -> -0.5
            score = -0.2 - ((dxy - 100.0) / 5.0) * 0.3
        elif dxy > DXY_WEAK_DOLLAR:
            # Linear interpolation: 95 -> +0.2, 100 -> -0.2
            score = 0.2 - ((dxy - 95.0) / 5.0) * 0.4
        else:
            score = 0.4

        self._log.debug(
            "dxy_scored",
            dxy=dxy,
            score=round(score, 4),
        )

        return score

    def _score_copper_gold(self, copper_gold_ratio: float) -> float:
        """Score the copper/gold ratio as an economic health indicator.

        Copper is an industrial metal sensitive to economic activity.
        Gold is a safe-haven asset.  A rising copper/gold ratio signals
        economic optimism (risk-on); a falling ratio signals economic
        caution (risk-off).

        Scoring:

        - Ratio > 0.00015: +0.6 (strong risk-on)
        - Ratio 0.00012--0.00015: +0.2 (moderate risk-on)
        - Ratio 0.00010--0.00012: -0.2 (neutral to mildly risk-off)
        - Ratio < 0.00010: -0.6 (strong risk-off)

        Args:
            copper_gold_ratio: Copper price divided by gold price.

        Returns:
            Score from approximately -0.6 to +0.6.
        """
        if copper_gold_ratio > COPPER_GOLD_RATIO_HIGH:
            score = 0.6
        elif copper_gold_ratio > 0.00012:
            # Linear interpolation: 0.00012 -> +0.2, 0.00015 -> +0.6
            fraction = (copper_gold_ratio - 0.00012) / (
                COPPER_GOLD_RATIO_HIGH - 0.00012
            )
            score = 0.2 + fraction * 0.4
        elif copper_gold_ratio > COPPER_GOLD_RATIO_LOW:
            # Linear interpolation: 0.00010 -> -0.2, 0.00012 -> +0.2
            fraction = (copper_gold_ratio - COPPER_GOLD_RATIO_LOW) / (
                0.00012 - COPPER_GOLD_RATIO_LOW
            )
            score = -0.2 + fraction * 0.4
        else:
            score = -0.6

        self._log.debug(
            "copper_gold_scored",
            copper_gold_ratio=copper_gold_ratio,
            score=round(score, 4),
        )

        return score
