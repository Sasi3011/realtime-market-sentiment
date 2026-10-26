"""Hybrid sentiment engine:  S = (1 - w) * Pt + w * L   (Equation 3.1).

Pt comes from FinBERT (ProsusAI/finbert) when the optional `transformers`
package is installed; otherwise the engine falls back to VADER so the project
runs on any machine. L is the adaptive finance-lexicon polarity (Equation 3.2).
"""
import logging

import numpy as np

from .config import CONFIG
from .preprocess import tokens

log = logging.getLogger(__name__)

FIN_LEXICON = {
    "beat": 0.8, "beats": 0.8, "upgrade": 0.7, "upgraded": 0.7, "outperform": 0.7,
    "record": 0.5, "surge": 0.6, "surges": 0.6, "buyback": 0.5, "growth": 0.4,
    "profit": 0.4, "raises": 0.4, "rally": 0.5,
    "miss": -0.8, "misses": -0.8, "downgrade": -0.7, "downgraded": -0.7, "probe": -0.6,
    "layoffs": -0.6, "lawsuit": -0.6, "cut": -0.5, "cuts": -0.5, "loss": -0.5,
    "default": -0.9, "decline": -0.5, "falls": -0.4, "fall": -0.4, "plunge": -0.7,
    "recall": -0.5, "volatile": -0.2, "liability": 0.0,
}
NEGATIONS = {"not", "no", "never", "without", "less", "despite"}


class HybridSentimentEngine:
    def __init__(self, w=None, use_finbert=True):
        self.w = CONFIG["lexicon_weight"] if w is None else w
        self.backend = "vader"
        self._finbert = None
        if use_finbert:
            try:
                from transformers import pipeline
                self._finbert = pipeline("text-classification", model="ProsusAI/finbert",
                                         top_k=None, truncation=True)
                self.backend = "finbert"
            except Exception as exc:
                log.info("FinBERT unavailable (%s) - using VADER for Pt", exc.__class__.__name__)
        if self._finbert is None:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            self._vader = SentimentIntensityAnalyzer()

    def transformer_score(self, text):
        """Signed score Pt = P(positive) - P(negative), in [-1, 1]."""
        if self._finbert is not None:
            probs = {d["label"].lower(): d["score"] for d in self._finbert(text[:2000])[0]}
            return probs.get("positive", 0.0) - probs.get("negative", 0.0)
        return self._vader.polarity_scores(text)["compound"]

    @staticmethod
    def lexicon_score(toks):
        """L = (1/n) * sum(v_i), with the sign flipped after a negation word."""
        vals = []
        for i, tok in enumerate(toks):
            if tok in FIN_LEXICON:
                v = FIN_LEXICON[tok]
                if i > 0 and toks[i - 1] in NEGATIONS:
                    v = -v
                vals.append(v)
        return float(np.mean(vals)) if vals else 0.0

    @staticmethod
    def update_lexicon(new_terms):
        """Adaptive layer: add or re-weight terms without retraining the model."""
        FIN_LEXICON.update(new_terms)

    @staticmethod
    def label(s):
        if s > CONFIG["pos_threshold"]:
            return "positive"
        if s < CONFIG["neg_threshold"]:
            return "negative"
        return "neutral"

    def score(self, text):
        p_t = self.transformer_score(text)
        lex = self.lexicon_score(tokens(text))
        s = (1 - self.w) * p_t + self.w * lex
        return round(p_t, 4), round(lex, 4), round(s, 4), self.label(s)
