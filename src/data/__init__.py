"""Data access layer for Project Titan.

This package provides async API clients for all external data sources:

- :mod:`~src.data.cache` — Redis-based caching and rate limiting
- :mod:`~src.data.questdb` — QuestDB time-series writer/reader
- :mod:`~src.data.polygon` — Polygon.io market data
- :mod:`~src.data.finnhub` — Finnhub news and calendar data
- :mod:`~src.data.fred` — FRED macroeconomic data
- :mod:`~src.data.quiver` — Quiver Quantitative alternative data
- :mod:`~src.data.sec_edgar` — SEC EDGAR Form 4 insider filings
- :mod:`~src.data.unusual_whales` — Unusual Whales options flow (graceful degradation)
"""
