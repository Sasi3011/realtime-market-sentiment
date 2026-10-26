"""Scoring loop: de-duplicate -> clean -> resolve ticker -> score -> store."""
import logging
import time
from datetime import datetime, timezone

from . import storage
from .collector import collect
from .config import CONFIG, WATCHLIST
from .preprocess import Deduplicator, clean_text, resolve_ticker
from .sentiment import HybridSentimentEngine

log = logging.getLogger(__name__)


def process_batch(articles, engine, dedup, con):
    stored = 0
    for a in articles:
        a["headline"] = clean_text(a["headline"])
        if not a["headline"] or dedup.is_duplicate(a["headline"]):
            continue
        a["body"] = clean_text(a.get("body")) or a["headline"]
        a["published"] = a.get("published") or datetime.now(timezone.utc).isoformat()
        a["ticker"] = resolve_ticker(a)
        a["sector"] = WATCHLIST.get(a["ticker"], "Market-wide")
        a["p_t"], a["lex"], a["score"], a["label"] = engine.score(a["headline"] + ". " + a["body"])
        storage.insert(con, a)
        stored += 1
    con.commit()
    return stored


def run(once=False, use_finbert=True):
    con = storage.connect()
    engine, dedup = HybridSentimentEngine(use_finbert=use_finbert), Deduplicator()
    log.info("Sentiment backend for Pt: %s", engine.backend)
    while True:
        start = time.time()
        articles = collect()
        n = process_batch(articles, engine, dedup, con)
        log.info("fetched %d articles, stored %d new", len(articles), n)
        if once:
            return con
        time.sleep(max(0, CONFIG["poll_seconds"] - (time.time() - start)))
