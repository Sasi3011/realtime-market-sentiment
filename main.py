"""Command-line entry point.

    python main.py run          # live pipeline, polls every 60 s (needs API keys)
    python main.py once         # one polling round, then show results
    python main.py score "Company beats estimates and raises guidance"
    python main.py report       # time-decay sentiment index per ticker and sector

Without API keys, `run` and `once` use the bundled offline sample data.
"""
import argparse
import logging

from src import storage
from src.aggregate import aggregate
from src.pipeline import run
from src.sentiment import HybridSentimentEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def show_report(con):
    df = storage.load(con)
    if df.empty:
        print("No articles stored yet. Run:  python main.py once")
        return
    print(f"\nStored articles: {len(df)}")
    print(df["label"].value_counts().to_string())
    print("\nTime-decay sentiment index per ticker (A_t):")
    print(aggregate(df, "ticker").to_string(index=False))
    print("\nTime-decay sentiment index per sector (A_t):")
    print(aggregate(df, "sector").to_string(index=False))


def main():
    ap = argparse.ArgumentParser(description="Real-time financial news sentiment pipeline")
    ap.add_argument("mode", choices=["run", "once", "score", "report"])
    ap.add_argument("text", nargs="?", help="headline to score (mode 'score')")
    ap.add_argument("--no-finbert", action="store_true", help="use VADER instead of FinBERT for Pt")
    args = ap.parse_args()

    if args.mode == "score":
        engine = HybridSentimentEngine(use_finbert=not args.no_finbert)
        p_t, lex, s, label = engine.score(args.text or "")
        print(f"backend={engine.backend}  Pt={p_t}  L={lex}  S={s}  label={label}")
    elif args.mode == "once":
        show_report(run(once=True, use_finbert=not args.no_finbert))
    elif args.mode == "report":
        show_report(storage.connect())
    else:
        run(once=False, use_finbert=not args.no_finbert)


if __name__ == "__main__":
    main()
