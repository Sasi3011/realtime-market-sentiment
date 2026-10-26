"""All tunable parameters of the pipeline (Table 3.2 of the report) in one place."""
import os

CONFIG = {
    "newsapi_key": os.getenv("NEWSAPI_KEY", ""),
    "alphavantage_key": os.getenv("ALPHAVANTAGE_KEY", ""),
    "finnhub_key": os.getenv("FINNHUB_KEY", ""),
    "poll_seconds": 60,          # polling interval
    "dedup_window_min": 30,      # near-duplicate look-back window
    "dedup_threshold": 0.90,     # cosine similarity cut-off
    "lexicon_weight": 0.28,      # w in S = (1 - w) * Pt + w * L
    "pos_threshold": 0.15,       # S > 0.15  -> positive
    "neg_threshold": -0.15,      # S < -0.15 -> negative
    "half_life_hours": 1.0,      # time-decay half-life
    "db_path": "sentiment.db",
}

WATCHLIST = {
    "AAPL": "Technology", "MSFT": "Technology", "NVDA": "Technology",
    "JPM": "Financials", "GS": "Financials",
    "XOM": "Energy", "CVX": "Energy",
    "PFE": "Healthcare", "JNJ": "Healthcare",
}

COMPANY_TO_TICKER = {
    "apple": "AAPL", "microsoft": "MSFT", "nvidia": "NVDA", "jpmorgan": "JPM",
    "goldman sachs": "GS", "exxon": "XOM", "chevron": "CVX", "pfizer": "PFE",
    "johnson & johnson": "JNJ",
}

MARKET_KEYWORDS = ["stock market", "federal reserve", "inflation", "earnings"]
