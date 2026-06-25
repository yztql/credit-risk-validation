"""
SQL-backed audit log for all validation runs.

Uses SQLite (zero-config) with four tables:
  validation_runs       — one row per pipeline execution
  dq_findings           — one row per data quality check result
  model_metrics         — segment-level performance metrics
  benchmark_comparisons — champion vs. challenger vs. naive
"""

import json
import sqlite3
from datetime import datetime
from typing import List

import pandas as pd

DB_PATH = "validation_log.db"

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS validation_runs (
    run_id        TEXT    PRIMARY KEY,
    timestamp     TEXT,
    dataset_name  TEXT,
    model_name    TEXT,
    status        TEXT,
    n_samples     INTEGER,
    n_findings    INTEGER
);

CREATE TABLE IF NOT EXISTS dq_findings (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id        TEXT,
    check_name    TEXT,
    status        TEXT,
    severity      TEXT,
    affected_rows INTEGER,
    affected_cols TEXT,
    detail        TEXT
);

CREATE TABLE IF NOT EXISTS model_metrics (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id        TEXT,
    segment       TEXT,
    macro_f1      REAL,
    precision     REAL,
    recall        REAL,
    roc_auc       REAL,
    n_samples     INTEGER,
    default_rate  REAL
);

CREATE TABLE IF NOT EXISTS benchmark_comparisons (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id        TEXT,
    model_name    TEXT,
    macro_f1      REAL,
    precision     REAL,
    recall        REAL
);
"""


class ValidationLogger:
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        with self._conn() as c:
            c.executescript(_SCHEMA_SQL)

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def log_run(
        self, run_id: str, dataset_name: str, model_name: str,
        status: str, n_samples: int, n_findings: int,
    ) -> None:
        with self._conn() as c:
            c.execute(
                "INSERT OR REPLACE INTO validation_runs VALUES (?,?,?,?,?,?,?)",
                (run_id, datetime.utcnow().isoformat(), dataset_name,
                 model_name, status, n_samples, n_findings),
            )

    def log_dq_findings(self, run_id: str, findings: List) -> None:
        rows = [
            (run_id, f.check, f.status, f.severity,
             f.affected_rows, json.dumps(f.affected_cols), f.detail)
            for f in findings
        ]
        with self._conn() as c:
            c.executemany(
                "INSERT INTO dq_findings "
                "(run_id,check_name,status,severity,affected_rows,affected_cols,detail) "
                "VALUES (?,?,?,?,?,?,?)",
                rows,
            )

    def log_segment_metrics(self, run_id: str, seg_df: pd.DataFrame) -> None:
        rows = [
            (run_id, row["segment"], row.get("macro_f1"), row.get("precision"),
             row.get("recall"), row.get("roc_auc"), int(row["n"]), row.get("default_rate"))
            for _, row in seg_df.iterrows()
        ]
        with self._conn() as c:
            c.executemany(
                "INSERT INTO model_metrics "
                "(run_id,segment,macro_f1,precision,recall,roc_auc,n_samples,default_rate) "
                "VALUES (?,?,?,?,?,?,?,?)",
                rows,
            )

    def log_benchmark(self, run_id: str, bench_df: pd.DataFrame) -> None:
        rows = [
            (run_id, row["model"], row["macro_f1"],
             row["precision_default"], row["recall_default"])
            for _, row in bench_df.iterrows()
        ]
        with self._conn() as c:
            c.executemany(
                "INSERT INTO benchmark_comparisons "
                "(run_id,model_name,macro_f1,precision,recall) VALUES (?,?,?,?,?)",
                rows,
            )

    # ── query helpers ─────────────────────────────────────────────────────────

    def run_history(self) -> pd.DataFrame:
        with self._conn() as c:
            return pd.read_sql(
                "SELECT * FROM validation_runs ORDER BY timestamp DESC", c
            )

    def findings_for_run(self, run_id: str) -> pd.DataFrame:
        with self._conn() as c:
            return pd.read_sql(
                "SELECT * FROM dq_findings WHERE run_id = ?", c, params=(run_id,)
            )

    def metrics_for_run(self, run_id: str) -> pd.DataFrame:
        with self._conn() as c:
            return pd.read_sql(
                "SELECT * FROM model_metrics WHERE run_id = ?", c, params=(run_id,)
            )
