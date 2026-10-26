"""SQLite storage of scored articles."""
import sqlite3

import pandas as pd

from .config import CONFIG

SCHEMA = """CREATE TABLE IF NOT EXISTS articles(
    id INTEGER PRIMARY KEY AUTOINCREMENT, source TEXT, ticker TEXT, sector TEXT,
    headline TEXT, body TEXT, published TEXT, p_t REAL, lex REAL, score REAL, label TEXT)"""


def connect(path=None):
    con = sqlite3.connect(path or CONFIG["db_path"], check_same_thread=False)
    con.execute(SCHEMA)
    con.commit()
    return con


def insert(con, a):
    con.execute("""INSERT INTO articles(source,ticker,sector,headline,body,published,p_t,lex,score,label)
                   VALUES(?,?,?,?,?,?,?,?,?,?)""",
                (a["source"], a["ticker"], a["sector"], a["headline"], a["body"],
                 a["published"], a["p_t"], a["lex"], a["score"], a["label"]))


def load(con):
    return pd.read_sql("SELECT * FROM articles", con)
