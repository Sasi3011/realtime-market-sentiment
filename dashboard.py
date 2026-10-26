"""Live dashboard (Plotly Dash). Start the pipeline first, then run:

    python dashboard.py      ->  open http://127.0.0.1:8050
"""
import pandas as pd
import plotly.express as px
from dash import Dash, Input, Output, dcc, html

from src import storage
from src.aggregate import aggregate

app = Dash(__name__)
app.title = "Market Sentiment Dashboard"
app.layout = html.Div(style={"fontFamily": "Times New Roman", "margin": "24px"}, children=[
    html.H2("Real-Time Market Sentiment Dashboard"),
    dcc.Dropdown(id="ticker", placeholder="Select a ticker", style={"width": "300px"}),
    dcc.Graph(id="index"),
    dcc.Graph(id="trend"),
    html.H4("Latest headlines"),
    html.Ul(id="headlines"),
    dcc.Interval(id="tick", interval=60_000),
])


@app.callback(Output("ticker", "options"), Output("index", "figure"), Output("trend", "figure"),
              Output("headlines", "children"), Input("ticker", "value"), Input("tick", "n_intervals"))
def refresh(ticker, _):
    df = storage.load(storage.connect())
    if df.empty:
        empty = px.bar(title="No data yet - run: python main.py once")
        return [], empty, empty, []
    options = sorted(df["ticker"].unique())
    idx = aggregate(df, "ticker")
    bar = px.bar(idx, x="ticker", y="A_t", color="A_t", color_continuous_scale="RdYlGn",
                 range_color=[-1, 1], title="Time-decay sentiment index A_t by ticker")
    sub = df[df["ticker"] == (ticker or options[0])].copy()
    sub["published"] = pd.to_datetime(sub["published"], utc=True, format="ISO8601")
    sub = sub.sort_values("published")
    trend = px.line(sub, x="published", y="score", markers=True,
                    title=f"Article sentiment over time - {ticker or options[0]}")
    latest = df.sort_values("published", ascending=False).head(10)
    items = [html.Li(f"[{r.label.upper()} {r.score:+.2f}] {r.ticker}: {r.headline} ({r.source})")
             for r in latest.itertuples()]
    return options, bar, trend, items


if __name__ == "__main__":
    app.run(debug=False, port=8050)
