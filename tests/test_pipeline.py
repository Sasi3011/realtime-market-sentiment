import math

import pandas as pd
import pytest

from src.aggregate import aggregate
from src.preprocess import Deduplicator, clean_text, resolve_ticker
from src.sentiment import HybridSentimentEngine


@pytest.fixture(scope="module")
def engine():
    return HybridSentimentEngine(use_finbert=False)


def test_hybrid_formula(engine):
    p_t, lex, s, _ = engine.score("Company beats estimates")
    assert math.isclose(s, round(0.72 * p_t + 0.28 * lex, 4), abs_tol=1e-3)


def test_labels(engine):
    assert engine.score("Shares surge to record high after earnings beat")[3] == "positive"
    assert engine.score("Regulator opens probe as company misses estimates")[3] == "negative"
    assert engine.score("Company schedules annual meeting for May")[3] == "neutral"


def test_negation_flips_lexicon_term():
    assert HybridSentimentEngine.lexicon_score(["not", "beat"]) == -0.8


def test_adaptive_lexicon(engine):
    before = engine.lexicon_score(["moonshot"])
    engine.update_lexicon({"moonshot": 0.6})
    assert before == 0.0 and engine.lexicon_score(["moonshot"]) == 0.6


def test_deduplicator_catches_syndicated_copy():
    d = Deduplicator()
    assert not d.is_duplicate("Exxon shares fall as oil prices plunge")
    assert d.is_duplicate("Exxon shares fall as oil prices plunge - Reuters")
    assert not d.is_duplicate("Exxon shares rise as oil prices recover")


def test_clean_and_resolve():
    assert clean_text("<p>Hello   world</p> [+120 chars]") == "Hello world"
    assert resolve_ticker({"headline": "Pfizer raises outlook", "body": ""}) == "PFE"
    assert resolve_ticker({"headline": "Inflation cools", "body": ""}) == "MARKET"


def test_time_decay_prefers_recent_articles():
    now = pd.Timestamp("2026-10-26T12:00:00Z")
    df = pd.DataFrame({"ticker": ["X", "X"], "score": [1.0, -1.0],
                       "published": ["2026-10-26T12:00:00Z", "2026-10-26T09:00:00Z"]})
    a_t = aggregate(df, now=now, half_life_hours=1.0)["A_t"].iloc[0]
    assert math.isclose(a_t, (1 - 0.125) / (1 + 0.125), abs_tol=1e-3)   # weights 1 and 2^-3
