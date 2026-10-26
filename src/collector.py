"""Concurrent collection of articles from NewsAPI, Alpha Vantage and Finnhub.

If no API keys are set, the collector falls back to the bundled sample file
(data/sample_news.csv) so the whole pipeline can be run offline.
"""
import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path

import aiohttp
import pandas as pd

from .config import CONFIG, WATCHLIST, MARKET_KEYWORDS

log = logging.getLogger(__name__)
SAMPLE_FILE = Path(__file__).resolve().parent.parent / "data" / "sample_news.csv"


def has_api_keys():
    return any(CONFIG[k] for k in ("newsapi_key", "alphavantage_key", "finnhub_key"))


async def _fetch_json(session, url, params):
    try:
        async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=20)) as r:
            if r.status == 200:
                return await r.json()
            log.warning("HTTP %s from %s", r.status, url)
    except Exception as exc:  # a network error must never stop the pipeline
        log.warning("request failed: %s", exc)
    return None


async def poll_newsapi(session, query):
    if not CONFIG["newsapi_key"]:
        return []
    data = await _fetch_json(session, "https://newsapi.org/v2/everything",
                             {"q": query, "language": "en", "sortBy": "publishedAt",
                              "pageSize": 50, "apiKey": CONFIG["newsapi_key"]})
    return [{"source": "NewsAPI", "headline": a.get("title") or "",
             "body": a.get("content") or a.get("description") or "",
             "published": a.get("publishedAt"), "ticker": None}
            for a in (data or {}).get("articles", [])]


async def poll_alphavantage(session, ticker):
    if not CONFIG["alphavantage_key"]:
        return []
    data = await _fetch_json(session, "https://www.alphavantage.co/query",
                             {"function": "NEWS_SENTIMENT", "tickers": ticker,
                              "apikey": CONFIG["alphavantage_key"]})
    out = []
    for a in (data or {}).get("feed", []):
        ts = a.get("time_published")
        iso = (datetime.strptime(ts, "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc).isoformat()
               if ts else None)
        out.append({"source": "AlphaVantage", "headline": a.get("title", ""),
                    "body": a.get("summary", ""), "published": iso, "ticker": ticker})
    return out


async def poll_finnhub(session, ticker):
    if not CONFIG["finnhub_key"]:
        return []
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    data = await _fetch_json(session, "https://finnhub.io/api/v1/company-news",
                             {"symbol": ticker, "from": today, "to": today,
                              "token": CONFIG["finnhub_key"]})
    return [{"source": "Finnhub", "headline": a.get("headline", ""),
             "body": a.get("summary", ""),
             "published": datetime.fromtimestamp(a.get("datetime", 0), timezone.utc).isoformat(),
             "ticker": ticker}
            for a in (data or [])]


async def collect_round():
    """Run every API request of one polling round concurrently."""
    async with aiohttp.ClientSession() as session:
        tasks = [poll_newsapi(session, q) for q in MARKET_KEYWORDS + list(WATCHLIST)]
        tasks += [poll_alphavantage(session, t) for t in WATCHLIST]
        tasks += [poll_finnhub(session, t) for t in WATCHLIST]
        results = await asyncio.gather(*tasks)
    return [a for batch in results for a in batch if a["headline"]]


def load_sample():
    """Offline mode: read the bundled sample headlines, stamped relative to now."""
    df = pd.read_csv(SAMPLE_FILE)
    now = pd.Timestamp.now(tz="UTC")
    df["published"] = [(now - pd.Timedelta(minutes=int(m))).isoformat() for m in df["minutes_ago"]]
    df["ticker"] = df["ticker"].astype(object).where(df["ticker"].notna(), None)
    df["body"] = df["body"].fillna("")
    return df[["source", "headline", "body", "published", "ticker"]].to_dict("records")


def collect():
    if has_api_keys():
        return asyncio.run(collect_round())
    log.info("No API keys set - using offline sample data (data/sample_news.csv)")
    return load_sample()
