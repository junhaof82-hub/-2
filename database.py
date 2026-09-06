from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd

from config import settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS prices (
    ticker TEXT NOT NULL, timestamp TEXT NOT NULL, interval TEXT NOT NULL,
    open REAL, high REAL, low REAL, close REAL, volume REAL,
    PRIMARY KEY (ticker, timestamp, interval)
);
CREATE TABLE IF NOT EXISTS news (
    id INTEGER PRIMARY KEY AUTOINCREMENT, ticker TEXT NOT NULL, title TEXT, summary TEXT,
    publisher TEXT, published_at TEXT, url TEXT, sentiment_score REAL, importance REAL,
    label TEXT, payload_json TEXT, created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT, ticker TEXT NOT NULL, prediction_date TEXT NOT NULL,
    horizon INTEGER NOT NULL, probability REAL NOT NULL, expected_low REAL, expected_high REAL,
    bullish_score REAL, risk_score REAL, model_probs_json TEXT, news_score REAL,
    technical_score REAL, actual_up INTEGER, actual_return REAL, evaluated_at TEXT,
    confidence REAL, data_quality REAL, component_probs_json TEXT, news_coverage REAL,
    model_version TEXT, payload_json TEXT, created_at TEXT NOT NULL,
    UNIQUE(ticker, prediction_date, horizon)
);
CREATE TABLE IF NOT EXISTS actual_results (
    ticker TEXT NOT NULL, prediction_date TEXT NOT NULL, horizon INTEGER NOT NULL,
    actual_date TEXT, actual_up INTEGER, actual_return REAL, created_at TEXT NOT NULL,
    PRIMARY KEY (ticker, prediction_date, horizon)
);
CREATE TABLE IF NOT EXISTS model_accuracy (
    id INTEGER PRIMARY KEY AUTOINCREMENT, ticker TEXT, horizon INTEGER NOT NULL, model_name TEXT NOT NULL,
    accuracy REAL, precision REAL, recall REAL, roc_auc REAL, sample_size INTEGER,
    balanced_accuracy REAL, brier_score REAL, log_loss REAL, mcc REAL, as_of TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS backtest_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT, ticker TEXT NOT NULL, horizon INTEGER NOT NULL,
    model_name TEXT NOT NULL, metrics_json TEXT NOT NULL, method TEXT, created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS daily_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT, ticker TEXT NOT NULL, snapshot_date TEXT NOT NULL,
    technical_json TEXT, news_json TEXT, macro_json TEXT, fundamentals_json TEXT,
    options_json TEXT, relative_strength_json TEXT, created_at TEXT NOT NULL,
    UNIQUE(ticker, snapshot_date)
);
"""


def _now() -> str: return datetime.now(timezone.utc).isoformat()


class Database:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path or settings.db_path); self.path.parent.mkdir(parents=True, exist_ok=True); self.init_db()

    @contextmanager
    def connect(self):
        conn = sqlite3.connect(self.path); conn.row_factory = sqlite3.Row
        try: yield conn; conn.commit()
        finally: conn.close()

    def init_db(self):
        with self.connect() as conn:
            conn.executescript(SCHEMA)
            # Safe migrations for databases created by the previous version.
            migrations = {
                'predictions': [('confidence','REAL'),('data_quality','REAL'),('component_probs_json','TEXT'),('news_coverage','REAL'),('model_version','TEXT'),('payload_json','TEXT')],
                'model_accuracy': [('balanced_accuracy','REAL'),('brier_score','REAL'),('log_loss','REAL'),('mcc','REAL')],
                'backtest_results': [('method','TEXT')],
            }
            for table, cols in migrations.items():
                existing = {r['name'] for r in conn.execute(f'PRAGMA table_info({table})').fetchall()}
                for name, typ in cols:
                    if name not in existing:
                        conn.execute(f'ALTER TABLE {table} ADD COLUMN {name} {typ}')

    def save_prices(self, ticker: str, df: pd.DataFrame, interval: str = '1d') -> None:
        rows = []
        for idx, r in df.iterrows():
            rows.append((ticker.upper(), pd.Timestamp(idx).isoformat(), interval,
                         float(r.get('open')) if pd.notna(r.get('open')) else None,
                         float(r.get('high')) if pd.notna(r.get('high')) else None,
                         float(r.get('low')) if pd.notna(r.get('low')) else None,
                         float(r.get('close')) if pd.notna(r.get('close')) else None,
                         float(r.get('volume')) if pd.notna(r.get('volume')) else None))
        with self.connect() as conn: conn.executemany('INSERT OR REPLACE INTO prices VALUES (?,?,?,?,?,?,?,?)', rows)

    def save_news(self, ticker: str, items: list[dict]) -> None:
        with self.connect() as conn:
            for x in items:
                conn.execute('''INSERT INTO news(ticker,title,summary,publisher,published_at,url,sentiment_score,importance,label,payload_json,created_at)
                                VALUES(?,?,?,?,?,?,?,?,?,?,?)''', (
                    ticker.upper(), x.get('title'), x.get('summary'), x.get('publisher'), str(x.get('published_at') or ''),
                    x.get('url'), x.get('sentiment_score'), x.get('importance'), x.get('label'), json.dumps(x, default=str), _now()))

    def save_prediction(self, result: dict) -> None:
        ticker = result['ticker'].upper(); date = result.get('prediction_date') or _now()
        compact = {k: result.get(k) for k in ('model_version','data_provider','last_price','risk_score','technical_score','data_quality','catalysts','risks')}
        with self.connect() as conn:
            for h, p in result['probabilities'].items():
                lo, hi = result['expected_ranges'].get(int(h), (None, None))
                component_probs = {
                    'ml': result.get('ml_probabilities', {}).get(int(h)),
                    'technical': result.get('technical_score', 50) / 100 if result.get('technical_score') is not None else None,
                    'news': (result.get('news', {}).get('sentiment_score', 0) + 100) / 200,
                }
                conn.execute('''INSERT OR REPLACE INTO predictions(
                    ticker,prediction_date,horizon,probability,expected_low,expected_high,bullish_score,risk_score,
                    model_probs_json,news_score,technical_score,actual_up,actual_return,evaluated_at,
                    confidence,data_quality,component_probs_json,news_coverage,model_version,payload_json,created_at)
                    VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''', (
                    ticker, date, int(h), float(p), lo, hi, result.get('bullish_score'), result.get('risk_score'),
                    json.dumps(result.get('model_probabilities', {}).get(int(h), {}), default=str),
                    result.get('news', {}).get('sentiment_score'), result.get('technical_score'), None, None, None,
                    result.get('confidence', {}).get(int(h)), result.get('data_quality'), json.dumps(component_probs, default=str),
                    result.get('news', {}).get('coverage_score'), result.get('model_version'), json.dumps(compact, default=str), _now()))

    def save_daily_snapshot(self, result: dict) -> None:
        day = pd.Timestamp(result.get('prediction_date') or _now()).date().isoformat(); ticker = result['ticker'].upper()
        with self.connect() as conn:
            conn.execute('''INSERT OR REPLACE INTO daily_snapshots(ticker,snapshot_date,technical_json,news_json,macro_json,fundamentals_json,options_json,relative_strength_json,created_at)
                            VALUES(?,?,?,?,?,?,?,?,?)''', (
                ticker, day, json.dumps(result.get('latest_technical', {}), default=str), json.dumps(result.get('news', {}), default=str),
                json.dumps(result.get('macro', {}), default=str), json.dumps(result.get('fundamentals', {}), default=str),
                json.dumps(result.get('options', {}), default=str), json.dumps(result.get('relative_strength', {}), default=str), _now()))

    def save_model_accuracy(self, ticker: str | None, horizon: int, model_name: str, metrics: dict, sample_size: int) -> None:
        with self.connect() as conn:
            conn.execute('''INSERT INTO model_accuracy(ticker,horizon,model_name,accuracy,precision,recall,roc_auc,sample_size,balanced_accuracy,brier_score,log_loss,mcc,as_of)
                            VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)''', (
                ticker, horizon, model_name, metrics.get('accuracy', metrics.get('prediction_accuracy')), metrics.get('precision'), metrics.get('recall'),
                metrics.get('roc_auc'), sample_size, metrics.get('balanced_accuracy'), metrics.get('brier_score'), metrics.get('log_loss'), metrics.get('mcc'), _now()))

    def save_backtest(self, ticker: str, horizon: int, model_name: str, metrics: dict, method: str | None = None) -> None:
        with self.connect() as conn:
            conn.execute('INSERT INTO backtest_results(ticker,horizon,model_name,metrics_json,method,created_at) VALUES(?,?,?,?,?,?)',
                         (ticker.upper(), horizon, model_name, json.dumps(metrics, default=str), method, _now()))

    def query_df(self, sql: str, params: tuple = ()) -> pd.DataFrame:
        with self.connect() as conn: return pd.read_sql_query(sql, conn, params=params)

    def latest_accuracy(self) -> pd.DataFrame:
        return self.query_df('SELECT * FROM model_accuracy ORDER BY id DESC LIMIT 1000')

    def prediction_history(self, ticker: str | None = None) -> pd.DataFrame:
        if ticker: return self.query_df('SELECT * FROM predictions WHERE ticker=? ORDER BY prediction_date DESC', (ticker.upper(),))
        return self.query_df('SELECT * FROM predictions ORDER BY prediction_date DESC LIMIT 2000')
