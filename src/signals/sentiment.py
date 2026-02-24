"""FinBERT-based sentiment analysis for Project Titan.

Fetches recent news headlines from Finnhub and scores them through the
ProsusAI/finbert pre-trained transformer model.  Each headline receives a
positive / negative / neutral probability distribution.  The per-ticker
sentiment score is a time-weighted rolling average (exponential decay over
24 hours) that ranges from -1.0 (extreme bearish) to +1.0 (extreme bullish).

A contrarian filter is available for retail-heavy sources where extreme
readings historically signal the opposite direction.

Usage::

    from src.signals.sentiment import SentimentAnalyzer

    analyzer = SentimentAnalyzer(finnhub_api_key="your_key")
    await analyzer.load_model()
    result = await analyzer.analyze_news("AAPL")
    print(result.score, result.bias)
"""

from __future__ import annotations

import asyncio
import math
import time
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Literal

import httpx
import torch
from pydantic import BaseModel, Field
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.utils.logging import get_logger

if TYPE_CHECKING:
    import structlog
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FINNHUB_BASE_URL: str = "https://finnhub.io/api/v1"
HTTP_TIMEOUT_SECONDS: float = 30.0

# Finnhub free tier: 30 API calls per minute.
FINNHUB_RATE_LIMIT_CALLS: int = 30
FINNHUB_RATE_LIMIT_WINDOW_SECONDS: float = 60.0
FINNHUB_DELAY_BETWEEN_CALLS: float = (
    FINNHUB_RATE_LIMIT_WINDOW_SECONDS / FINNHUB_RATE_LIMIT_CALLS
)

# FinBERT label mapping — the model outputs logits in this order.
_FINBERT_LABELS: tuple[str, ...] = ("positive", "negative", "neutral")

# Maximum number of headlines to process per ticker to keep inference fast.
_MAX_HEADLINES_PER_TICKER: int = 50

# Tokenizer maximum input length (FinBERT uses 512 tokens).
_MAX_TOKEN_LENGTH: int = 512

# Batch size for FinBERT inference.
_INFERENCE_BATCH_SIZE: int = 16

# Contrarian filter thresholds.
_CONTRARIAN_EXTREME_THRESHOLD: float = 0.8
_CONTRARIAN_DAMPING_FACTOR: float = 0.30  # 30 % reduction

# Sentiment bias classification boundaries.
_BULLISH_THRESHOLD: float = 0.15
_BEARISH_THRESHOLD: float = -0.15


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class SentimentScore(BaseModel):
    """Sentiment score for a single piece of text.

    Attributes:
        text: The original headline or text that was analysed.
        label: The dominant sentiment class (positive / negative / neutral).
        confidence: The probability of the dominant class.
        positive_prob: Softmax probability for the positive class.
        negative_prob: Softmax probability for the negative class.
        neutral_prob: Softmax probability for the neutral class.
        timestamp: When the original news article was published.
    """

    text: str
    label: Literal["positive", "negative", "neutral"]
    confidence: float = Field(ge=0.0, le=1.0)
    positive_prob: float = Field(ge=0.0, le=1.0)
    negative_prob: float = Field(ge=0.0, le=1.0)
    neutral_prob: float = Field(ge=0.0, le=1.0)
    timestamp: datetime


class SentimentResult(BaseModel):
    """Aggregated sentiment result for a single ticker.

    Attributes:
        ticker: The stock symbol.
        score: Time-weighted rolling sentiment score in [-1.0, 1.0].
        num_articles: Number of articles analysed.
        avg_confidence: Mean confidence across all scored headlines.
        bias: Human-readable directional label.
        scores: Per-headline sentiment scores.
        calculated_at: Timestamp when the result was computed.
    """

    ticker: str
    score: float = Field(ge=-1.0, le=1.0)
    num_articles: int = Field(ge=0)
    avg_confidence: float = Field(ge=0.0, le=1.0)
    bias: Literal["bullish", "bearish", "neutral"]
    scores: list[SentimentScore]
    calculated_at: datetime


# ---------------------------------------------------------------------------
# SentimentAnalyzer
# ---------------------------------------------------------------------------


class SentimentAnalyzer:
    """Analyse news sentiment for equity tickers using FinBERT.

    Fetches recent news from Finnhub, tokenises and scores headlines through
    the ``ProsusAI/finbert`` transformer, then produces a time-weighted
    rolling sentiment score per ticker.

    Args:
        finnhub_api_key: API key for Finnhub news requests.
        model_name: Hugging Face model identifier for FinBERT.  Defaults to
            ``"ProsusAI/finbert"``.
    """

    def __init__(
        self,
        finnhub_api_key: str,
        model_name: str = "ProsusAI/finbert",
    ) -> None:
        self._api_key: str = finnhub_api_key
        self._model_name: str = model_name

        self._log: structlog.stdlib.BoundLogger = get_logger("signals.sentiment")

        # Model and tokenizer are loaded lazily via load_model().
        self._model: PreTrainedModel | None = None
        self._tokenizer: PreTrainedTokenizerBase | None = None
        self._model_loaded: bool = False

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    async def load_model(self) -> None:
        """Load the FinBERT model and tokenizer from Hugging Face.

        Moves the model to CPU (no GPU assumed for production deployments).
        Logs the time taken to load the model.  Calling this method multiple
        times is safe — subsequent calls return immediately.
        """
        if self._model_loaded:
            self._log.debug("model_already_loaded", model=self._model_name)
            return

        self._log.info("loading_finbert_model", model=self._model_name)
        start = time.monotonic()

        # Run the blocking Hugging Face download/load in a thread so that
        # the event loop is not blocked.
        loop = asyncio.get_running_loop()
        self._tokenizer = await loop.run_in_executor(
            None,
            lambda: AutoTokenizer.from_pretrained(self._model_name, use_fast=False),
        )
        self._model = await loop.run_in_executor(
            None,
            lambda: AutoModelForSequenceClassification.from_pretrained(
                self._model_name
            ),
        )

        # Ensure model is on CPU and in evaluation mode.
        self._model = self._model.to("cpu")  # type: ignore[union-attr]
        self._model.eval()  # type: ignore[union-attr]

        elapsed = time.monotonic() - start
        self._model_loaded = True

        self._log.info(
            "finbert_model_loaded",
            model=self._model_name,
            elapsed_seconds=round(elapsed, 2),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def analyze_news(self, ticker: str) -> SentimentResult:
        """Fetch recent news for *ticker* and produce an aggregated sentiment score.

        Retrieves the last 24 hours of company news from Finnhub, scores each
        headline through FinBERT, and computes a time-weighted rolling
        sentiment score.

        Args:
            ticker: The stock symbol (e.g. ``"AAPL"``).

        Returns:
            A :class:`SentimentResult` with the aggregated score and
            per-headline breakdowns.

        Raises:
            RuntimeError: If :meth:`load_model` has not been called.
        """
        self._ensure_model_loaded()

        headlines = await self._fetch_news(ticker)

        if not headlines:
            self._log.info("no_news_found", ticker=ticker)
            return SentimentResult(
                ticker=ticker,
                score=0.0,
                num_articles=0,
                avg_confidence=0.0,
                bias="neutral",
                scores=[],
                calculated_at=datetime.now(UTC),
            )

        # Limit the number of headlines to avoid excessive inference time.
        headlines = headlines[:_MAX_HEADLINES_PER_TICKER]

        texts = [h["headline"] for h in headlines]
        timestamps = [h["timestamp"] for h in headlines]

        loop = asyncio.get_running_loop()
        sentiment_scores = await loop.run_in_executor(
            None,
            self._score_texts,
            texts,
            timestamps,
        )

        rolling = self._calculate_rolling_sentiment(sentiment_scores)

        avg_conf = (
            sum(s.confidence for s in sentiment_scores) / len(sentiment_scores)
            if sentiment_scores
            else 0.0
        )

        bias = self._classify_bias(rolling)

        result = SentimentResult(
            ticker=ticker,
            score=round(rolling, 4),
            num_articles=len(sentiment_scores),
            avg_confidence=round(avg_conf, 4),
            bias=bias,
            scores=sentiment_scores,
            calculated_at=datetime.now(UTC),
        )

        self._log.info(
            "sentiment_analyzed",
            ticker=ticker,
            score=result.score,
            bias=result.bias,
            num_articles=result.num_articles,
            avg_confidence=result.avg_confidence,
        )
        return result

    async def analyze_batch(self, tickers: list[str]) -> dict[str, SentimentResult]:
        """Analyse sentiment for multiple tickers with Finnhub rate limiting.

        Processes tickers sequentially with a delay between Finnhub API calls
        to respect the free-tier rate limit of 30 calls per minute.

        Args:
            tickers: List of stock symbols.

        Returns:
            A mapping of ticker symbol to :class:`SentimentResult`.
        """
        self._ensure_model_loaded()

        results: dict[str, SentimentResult] = {}

        self._log.info("batch_sentiment_start", ticker_count=len(tickers))

        for idx, ticker in enumerate(tickers):
            try:
                result = await self.analyze_news(ticker)
                results[ticker] = result
            except Exception:
                self._log.exception("batch_sentiment_error", ticker=ticker)
                # Return a neutral fallback so the batch can continue.
                results[ticker] = SentimentResult(
                    ticker=ticker,
                    score=0.0,
                    num_articles=0,
                    avg_confidence=0.0,
                    bias="neutral",
                    scores=[],
                    calculated_at=datetime.now(UTC),
                )

            # Rate-limit Finnhub calls (skip delay after last ticker).
            if idx < len(tickers) - 1:
                await asyncio.sleep(FINNHUB_DELAY_BETWEEN_CALLS)

        self._log.info(
            "batch_sentiment_complete",
            ticker_count=len(tickers),
            results_count=len(results),
        )
        return results

    # ------------------------------------------------------------------
    # Internal: News fetching
    # ------------------------------------------------------------------

    @retry(
        retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    async def _fetch_news(
        self,
        ticker: str,
    ) -> list[dict[str, str | datetime]]:
        """Fetch recent company news headlines from Finnhub.

        Retrieves the last 24 hours of news for the given ticker.

        Args:
            ticker: The stock symbol.

        Returns:
            A list of dicts with keys ``"headline"`` (str) and
            ``"timestamp"`` (datetime).
        """
        now = datetime.now(UTC)
        from_date = (now - timedelta(hours=24)).strftime("%Y-%m-%d")
        to_date = now.strftime("%Y-%m-%d")

        self._log.debug(
            "fetching_news",
            ticker=ticker,
            from_date=from_date,
            to_date=to_date,
        )

        try:
            async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_SECONDS) as client:
                response = await client.get(
                    f"{FINNHUB_BASE_URL}/company-news",
                    params={
                        "symbol": ticker.upper(),
                        "from": from_date,
                        "to": to_date,
                        "token": self._api_key,
                    },
                )
                response.raise_for_status()
                articles = response.json()

        except httpx.HTTPStatusError as exc:
            self._log.warning(
                "news_fetch_http_error",
                ticker=ticker,
                status_code=exc.response.status_code,
            )
            raise
        except Exception:
            self._log.exception("news_fetch_failed", ticker=ticker)
            raise

        if not isinstance(articles, list):
            self._log.warning(
                "unexpected_news_response",
                ticker=ticker,
                response_type=type(articles).__name__,
            )
            return []

        headlines: list[dict[str, str | datetime]] = []
        for article in articles:
            headline = article.get("headline", "").strip()
            epoch = article.get("datetime", 0)
            source = article.get("source", "")

            if not headline:
                continue

            try:
                ts = datetime.fromtimestamp(epoch, tz=UTC)
            except (ValueError, TypeError, OSError):
                ts = now

            # Only include articles from the last 24 hours.
            if (now - ts).total_seconds() > 86400:
                continue

            headlines.append(
                {
                    "headline": headline,
                    "timestamp": ts,
                    "source": source,
                }
            )

        # Sort by timestamp descending (most recent first).
        headlines.sort(key=lambda h: h["timestamp"], reverse=True)  # type: ignore[arg-type,return-value]

        self._log.debug(
            "news_fetched",
            ticker=ticker,
            total_articles=len(articles),
            valid_headlines=len(headlines),
        )
        return headlines  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Internal: FinBERT scoring
    # ------------------------------------------------------------------

    def _score_texts(
        self,
        texts: list[str],
        timestamps: list[datetime],
    ) -> list[SentimentScore]:
        """Tokenize and score a batch of texts through the FinBERT model.

        Processes texts in mini-batches of :data:`_INFERENCE_BATCH_SIZE` to
        keep memory usage bounded.

        Args:
            texts: Headline strings to score.
            timestamps: Corresponding publication timestamps.

        Returns:
            A list of :class:`SentimentScore` objects, one per input text.
        """
        assert self._model is not None  # noqa: S101
        assert self._tokenizer is not None  # noqa: S101

        all_scores: list[SentimentScore] = []

        for batch_start in range(0, len(texts), _INFERENCE_BATCH_SIZE):
            batch_end = min(batch_start + _INFERENCE_BATCH_SIZE, len(texts))
            batch_texts = texts[batch_start:batch_end]
            batch_timestamps = timestamps[batch_start:batch_end]

            # Tokenize the batch.
            encoded = self._tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=_MAX_TOKEN_LENGTH,
                return_tensors="pt",
            )

            # Run inference with no gradient tracking.
            with torch.no_grad():
                outputs = self._model(**encoded)
                logits = outputs.logits

            # Apply softmax to obtain class probabilities.
            probabilities = torch.nn.functional.softmax(logits, dim=-1)

            for idx in range(len(batch_texts)):
                probs = probabilities[idx].tolist()

                positive_prob = float(probs[0])
                negative_prob = float(probs[1])
                neutral_prob = float(probs[2])

                # Determine dominant label.
                max_idx = probs.index(max(probs))
                label = _FINBERT_LABELS[max_idx]
                confidence = float(probs[max_idx])

                all_scores.append(
                    SentimentScore(
                        text=batch_texts[idx],
                        label=label,  # type: ignore[arg-type]
                        confidence=round(confidence, 4),
                        positive_prob=round(positive_prob, 4),
                        negative_prob=round(negative_prob, 4),
                        neutral_prob=round(neutral_prob, 4),
                        timestamp=batch_timestamps[idx],
                    )
                )

        self._log.debug(
            "texts_scored",
            num_texts=len(texts),
            num_scores=len(all_scores),
        )
        return all_scores

    # ------------------------------------------------------------------
    # Internal: Rolling sentiment
    # ------------------------------------------------------------------

    def _calculate_rolling_sentiment(
        self,
        scores: list[SentimentScore],
        decay_hours: float = 24.0,
    ) -> float:
        """Compute a time-weighted rolling sentiment score.

        Recent headlines are weighted more heavily using exponential decay.
        The decay half-life equals *decay_hours*.  The resulting score is
        clamped to [-1.0, 1.0].

        Each headline contributes ``positive_prob - negative_prob`` to the
        directional signal, multiplied by its time-decay weight.

        Args:
            scores: Per-headline sentiment scores.
            decay_hours: The exponential decay half-life in hours.

        Returns:
            A sentiment score from -1.0 (extreme bearish) to +1.0 (extreme
            bullish).  Returns 0.0 if *scores* is empty.
        """
        if not scores:
            return 0.0

        now = datetime.now(UTC)

        # Decay constant: lambda = ln(2) / half_life.
        decay_lambda = math.log(2.0) / max(decay_hours, 0.01)

        weighted_sum: float = 0.0
        weight_total: float = 0.0

        for score in scores:
            # Compute age of the headline in hours.
            age_seconds = max((now - score.timestamp).total_seconds(), 0.0)
            age_hours = age_seconds / 3600.0

            # Exponential decay weight.
            weight = math.exp(-decay_lambda * age_hours)

            # Directional signal: positive_prob - negative_prob.
            signal = score.positive_prob - score.negative_prob

            weighted_sum += signal * weight
            weight_total += weight

        if weight_total == 0.0:
            return 0.0

        raw_score = weighted_sum / weight_total

        # Clamp to [-1.0, 1.0].
        return max(-1.0, min(1.0, raw_score))

    # ------------------------------------------------------------------
    # Internal: Contrarian filter
    # ------------------------------------------------------------------

    def _apply_contrarian_filter(self, score: float, source: str) -> float:
        """Apply contrarian damping for retail-heavy sentiment sources.

        When retail sentiment sources (e.g. WallStreetBets, StockTwits,
        Reddit) show extreme readings, the signal is dampened toward neutral
        because historically such extremes tend to be counter-indicators.

        * Extreme bullishness (> 0.8): score reduced by 30%.
        * Extreme bearishness (< -0.8): score increased by 30%.

        For non-retail sources the score passes through unchanged.

        Args:
            score: The raw sentiment score in [-1.0, 1.0].
            source: The name of the news source.

        Returns:
            The (possibly dampened) sentiment score in [-1.0, 1.0].
        """
        retail_keywords = (
            "wallstreetbets",
            "wsb",
            "stocktwits",
            "reddit",
            "r/wallstreetbets",
            "r/stocks",
            "r/options",
        )

        source_lower = source.lower()
        is_retail = any(kw in source_lower for kw in retail_keywords)

        if not is_retail:
            return score

        if score > _CONTRARIAN_EXTREME_THRESHOLD:
            adjusted = score * (1.0 - _CONTRARIAN_DAMPING_FACTOR)
            self._log.debug(
                "contrarian_damping_applied",
                source=source,
                original_score=round(score, 4),
                adjusted_score=round(adjusted, 4),
                direction="bullish",
            )
            return adjusted

        if score < -_CONTRARIAN_EXTREME_THRESHOLD:
            adjusted = score * (1.0 - _CONTRARIAN_DAMPING_FACTOR)
            self._log.debug(
                "contrarian_damping_applied",
                source=source,
                original_score=round(score, 4),
                adjusted_score=round(adjusted, 4),
                direction="bearish",
            )
            return adjusted

        return score

    # ------------------------------------------------------------------
    # Internal: Helpers
    # ------------------------------------------------------------------

    def _ensure_model_loaded(self) -> None:
        """Raise if :meth:`load_model` has not been called.

        Raises:
            RuntimeError: If the model and tokenizer are not loaded.
        """
        if not self._model_loaded or self._model is None or self._tokenizer is None:
            raise RuntimeError(
                "FinBERT model not loaded. Call `await analyzer.load_model()` "
                "before analysing sentiment."
            )

    @staticmethod
    def _classify_bias(score: float) -> Literal["bullish", "bearish", "neutral"]:
        """Classify a numeric sentiment score into a directional bias.

        Args:
            score: The sentiment score in [-1.0, 1.0].

        Returns:
            ``"bullish"`` if score > 0.15, ``"bearish"`` if score < -0.15,
            otherwise ``"neutral"``.
        """
        if score > _BULLISH_THRESHOLD:
            return "bullish"
        if score < _BEARISH_THRESHOLD:
            return "bearish"
        return "neutral"
