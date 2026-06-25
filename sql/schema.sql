-- Credit Risk Validation Pipeline — Audit Log Schema
-- Mirrored in code by src/sql_logger.py (SQLite).
-- Can be ported to PostgreSQL by replacing AUTOINCREMENT with SERIAL.

CREATE TABLE IF NOT EXISTS validation_runs (
    run_id        TEXT    PRIMARY KEY,
    timestamp     TEXT,                   -- ISO-8601 UTC
    dataset_name  TEXT,
    model_name    TEXT,
    status        TEXT,                   -- PASS | FAIL | ABORTED
    n_samples     INTEGER,
    n_findings    INTEGER
);

CREATE TABLE IF NOT EXISTS dq_findings (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id        TEXT    REFERENCES validation_runs(run_id),
    check_name    TEXT,                   -- missing_values | duplicates | schema | ...
    status        TEXT,                   -- PASS | FAIL | WARN
    severity      TEXT,                   -- INFO | LOW | MEDIUM | HIGH | CRITICAL
    affected_rows INTEGER,
    affected_cols TEXT,                   -- JSON array of column names
    detail        TEXT
);

CREATE TABLE IF NOT EXISTS model_metrics (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id        TEXT    REFERENCES validation_runs(run_id),
    segment       TEXT,                   -- Retail | SME | Corporate | (overall)
    macro_f1      REAL,
    precision     REAL,
    recall        REAL,
    roc_auc       REAL,
    n_samples     INTEGER,
    default_rate  REAL
);

CREATE TABLE IF NOT EXISTS benchmark_comparisons (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id        TEXT    REFERENCES validation_runs(run_id),
    model_name    TEXT,                   -- Champion | Benchmark | Naive Baseline
    macro_f1      REAL,
    precision     REAL,
    recall        REAL
);

-- Useful audit queries
-- All runs that failed:
--   SELECT * FROM validation_runs WHERE status = 'FAIL';
--
-- DQ findings by severity across all runs:
--   SELECT severity, COUNT(*) FROM dq_findings GROUP BY severity;
--
-- Segment macro-F1 drift across runs:
--   SELECT run_id, segment, macro_f1 FROM model_metrics ORDER BY run_id, segment;
