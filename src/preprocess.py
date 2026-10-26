"""Text cleaning, near-duplicate detection and ticker resolution."""
import re
import time

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .config import CONFIG, COMPANY_TO_TICKER, WATCHLIST

STOPWORDS = {"the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with", "as", "at",
             "by", "is", "are", "was", "were", "be", "its", "it", "this", "that", "from"}


def clean_text(text):
    text = re.sub(r"<[^>]+>", " ", text or "")                    # HTML remnants
    text = re.sub(r"\[\+\d+ chars\]", " ", text)                   # NewsAPI truncation marker
    text = re.sub(r"(?i)(click here|subscribe now|all rights reserved).*", " ", text)
    text = re.sub(r"http\S+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokens(text):
    """Lower-cased word tokens for the lexicon layer (stop words removed)."""
    return [t for t in re.findall(r"[a-z][a-z'&-]*", text.lower()) if t not in STOPWORDS]


def normalise_headline(text):
    """Lower-case, drop punctuation and trailing source tags such as ' - Reuters'."""
    text = re.sub(r"\s+[-|:]\s+[\w .]{1,25}$", "", text.strip())
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def resolve_ticker(article):
    """Use the API ticker if present, otherwise look for a known company name."""
    if article.get("ticker") in WATCHLIST:
        return article["ticker"]
    text = (article.get("headline", "") + " " + article.get("body", "")).lower()
    for name, tick in COMPANY_TO_TICKER.items():
        if name in text:
            return tick
    return "MARKET"


class Deduplicator:
    """Rejects a headline that is near-identical (cosine >= threshold) to one
    seen within the last `dedup_window_min` minutes."""

    def __init__(self, threshold=None, window_min=None):
        self.threshold = threshold or CONFIG["dedup_threshold"]
        self.window = (window_min or CONFIG["dedup_window_min"]) * 60
        self.seen = []                                             # (timestamp, headline)

    def is_duplicate(self, headline, now=None):
        now = now or time.time()
        self.seen = [(t, h) for t, h in self.seen if now - t < self.window]
        norm = normalise_headline(headline)
        if self.seen:
            corpus = [h for _, h in self.seen] + [norm]
            vec = CountVectorizer(analyzer="char_wb", ngram_range=(3, 5)).fit_transform(corpus)
            if cosine_similarity(vec[-1], vec[:-1]).max() >= self.threshold:
                return True
        self.seen.append((now, norm))
        return False
