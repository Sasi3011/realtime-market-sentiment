# Real-Time Algorithmic Market Sentiment Extraction via Live Scraping of Financial News APIs

**Author:** SASIKIRAN T T (722824104216) · B.E. Computer Science and Engineering
**Guide:** Dr. K. Suresh Kumar, M.Tech., Ph.D., Associate Professor, CSE
**Institution:** Sri Eshwar College of Engineering, Coimbatore · Data Science Project, October 2026

A simple, working pipeline that collects financial news from **NewsAPI**, **Alpha Vantage** and
**Finnhub**, removes syndicated duplicates, scores each article with a **hybrid sentiment engine**
(transformer + adaptive finance lexicon), aggregates the scores per ticker and sector with
**time-decay weighting**, and shows the result on a live **Plotly Dash** dashboard.

## How it works

```
News APIs ──► Collector (async, every 60 s) ──► Clean + de-duplicate + resolve ticker
          ──► Hybrid sentiment engine ──► SQLite ──► Time-decay index ──► Dashboard
```

| Step | Formula |
|---|---|
| Hybrid score | `S = (1 − w) · Pt + w · L`, with `w = 0.28` |
| Lexicon polarity | `L = (1/n) · Σ vᵢ` (sign flipped after a negation word) |
| Label | positive if `S > 0.15`, negative if `S < −0.15`, otherwise neutral |
| Time-decay index | `Aₜ = Σ Sᵢ·e^(−λΔtᵢ) / Σ e^(−λΔtᵢ)`, `λ = ln 2 / half-life` (1 hour) |

`Pt` comes from **FinBERT** (`ProsusAI/finbert`) when `transformers` is installed; otherwise the
project automatically falls back to **VADER**, so it runs on any laptop.

## Project structure

```
realtime-market-sentiment/
├── main.py               # command-line entry point
├── dashboard.py          # live Plotly Dash dashboard
├── src/
│   ├── config.py         # all parameters (polling, thresholds, w, half-life, watch list)
│   ├── collector.py      # concurrent API polling + offline sample mode
│   ├── preprocess.py     # cleaning, near-duplicate detection, ticker resolution
│   ├── sentiment.py      # hybrid FinBERT/VADER + finance lexicon engine
│   ├── aggregate.py      # time-decay aggregation per ticker / sector
│   ├── storage.py        # SQLite storage
│   └── pipeline.py       # scoring loop
├── data/sample_news.csv  # synthetic sample headlines for offline demo
├── tests/test_pipeline.py
└── requirements.txt
```

## Quick start

```bash
pip install -r requirements.txt

python main.py once                       # one round (offline sample data if no API keys)
python main.py report                     # sentiment index per ticker and sector
python main.py score "Company beats estimates and raises guidance"
python dashboard.py                       # open http://127.0.0.1:8050
pytest -q                                 # run the tests
```

### Live mode with real news

Get free API keys from [newsapi.org](https://newsapi.org), [alphavantage.co](https://www.alphavantage.co)
and [finnhub.io](https://finnhub.io), then:

```bash
export NEWSAPI_KEY=...  ALPHAVANTAGE_KEY=...  FINNHUB_KEY=...     # Windows: set NEWSAPI_KEY=...
python main.py run          # polls every 60 seconds; keep it running
python dashboard.py         # in a second terminal
```

### Optional: FinBERT

```bash
pip install transformers torch
```

## Example output (offline sample data)

```
Time-decay sentiment index per sector (A_t):
     sector     A_t  articles
 Technology  0.37         7
 Financials  0.29         4
Market-wide  0.22         4
 Healthcare -0.13         4
     Energy -0.74         3
```

> `data/sample_news.csv` contains **synthetic headlines written for demonstration only**; they are
> not real news. Use API keys for live data.
