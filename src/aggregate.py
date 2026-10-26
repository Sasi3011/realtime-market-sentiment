"""Time-decay aggregation (Equation 3.3):
A_t = sum(S_i * exp(-lambda * dt_i)) / sum(exp(-lambda * dt_i)),  lambda = ln 2 / half-life.
"""
import math

import numpy as np
import pandas as pd

from .config import CONFIG


def aggregate(df, key="ticker", now=None, half_life_hours=None):
    if df.empty:
        return pd.DataFrame(columns=[key, "A_t", "articles"])
    now = now or pd.Timestamp.now(tz="UTC")
    lam = math.log(2) / (half_life_hours or CONFIG["half_life_hours"])
    d = df.copy()
    d["published"] = pd.to_datetime(d["published"], utc=True, format="ISO8601")
    d["dt_h"] = ((now - d["published"]).dt.total_seconds() / 3600).clip(lower=0)
    d["wt"] = np.exp(-lam * d["dt_h"])
    d["ws"] = d["score"] * d["wt"]
    g = d.groupby(key).agg(ws=("ws", "sum"), wt=("wt", "sum"), articles=("score", "size"))
    g["A_t"] = (g["ws"] / g["wt"]).round(4)
    return g[["A_t", "articles"]].reset_index().sort_values("A_t", ascending=False)
