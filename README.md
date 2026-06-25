# Credit Risk Model Evaluation and Validation Pipeline

A model validation workflow for credit risk, built to mirror the independent review process used in IFRS 9 / AIRB environments. Tests prediction quality, data quality, input consistency, and failure cases across repeated validation scenarios, and logs all findings to a SQL audit trail.

---

## Features

- **Data quality checks** — missing values, duplicate rows, schema mismatches, invalid input ranges, ID uniqueness, class imbalance
- **Schema gate** — aborts model evaluation automatically when required feature columns are absent or mis-typed
- **Model performance** — confusion matrix, macro-F1, ROC-AUC, precision, recall against a configurable pass threshold
- **Segment-level validation** — per-segment performance breakdown (Retail / SME / Corporate)
- **Benchmark comparison** — champion (Logistic Regression) vs. challenger (Random Forest) vs. naive stratified baseline
- **Error analysis** — mean feature profiles of False Negatives (missed defaults) and False Positives (false alarms)
- **SQL audit log** — every run, finding, metric, and benchmark comparison persisted to SQLite for independent review

---

## Project Structure

```
credit-risk-validation/
├── run_validation.py          # entry point — runs all five validation scenarios
├── requirements.txt
├── src/
│   ├── data_generator.py      # synthetic credit data + five scenario variants
│   ├── data_quality.py        # six named DQ checks, each returning a structured Finding
│   ├── model_trainer.py       # champion, benchmark, and naive baseline models
│   ├── metrics.py             # confusion matrix, macro-F1, segment metrics, error analysis
│   ├── sql_logger.py          # SQLite-backed audit logging across four tables
│   └── validator.py           # orchestration pipeline — runs all stages for one dataset
├── sql/
│   └── schema.sql             # DDL reference (mirrored in sql_logger.py)
└── tests/
    └── test_validation.py     # 15 unit tests covering DQ checks, metrics, and model outputs
```

---

## Quickstart

```bash
# 1. Clone and enter the project
git clone <repo-url>
cd credit-risk-validation

# 2. Create a virtual environment and install dependencies
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 3. Run the full validation suite
python run_validation.py

# 4. Run the test suite
pip install pytest
pytest tests/ -v
```

On first run, synthetic data and trained models are generated automatically under `data/` and `models/`.

---

## Validation Scenarios

The pipeline runs against five datasets in sequence:

| Scenario | Injected Issue | Expected Outcome |
|---|---|---|
| Clean validation set | None | PASS / FAIL on macro-F1 threshold |
| Missing values | ~8% nulls across four key features | FAIL — DQ finding (MEDIUM severity) |
| Duplicate rows | 5% duplicate records | FAIL — two DQ findings (duplicates + ID uniqueness) |
| Schema mismatch | Renamed columns, wrong dtypes | ABORTED — schema gate blocks model evaluation |
| Invalid input values | Negative income, credit score < 0, DTI > 1 | FAIL — DQ finding (HIGH severity) |

---

## Metrics

| Metric | Description |
|---|---|
| Macro F1 | Unweighted average F1 across both classes — preferred over accuracy under class imbalance |
| ROC-AUC | Ranking quality of predicted default probabilities |
| Confusion Matrix | TN / FP / FN / TP counts for threshold-based decisions |
| Segment macro-F1 | Per-segment (Retail, SME, Corporate) performance — detects sub-population degradation |
| Benchmark delta | Champion macro-F1 minus Random Forest and naive baseline |

Pass threshold for macro-F1 is set to **0.55** in `validator.py:MACRO_F1_THRESHOLD`. Adjust to match your organisation's model risk standards.

---

## Audit Log

All results are persisted to `validation_log.db` (SQLite). Four tables:

```
validation_runs       — one row per pipeline execution (run_id, status, timestamp)
dq_findings           — one row per data quality check result (severity, affected columns)
model_metrics         — segment-level macro-F1, precision, recall, ROC-AUC
benchmark_comparisons — champion vs. challenger vs. naive baseline
```

Query examples:

```sql
-- All runs that produced a FAIL or ABORT
SELECT * FROM validation_runs WHERE status != 'PASS';

-- DQ findings grouped by severity
SELECT severity, COUNT(*) FROM dq_findings GROUP BY severity;

-- Segment macro-F1 across all runs
SELECT run_id, segment, macro_f1 FROM model_metrics ORDER BY run_id, segment;
```

---

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.11+ |
| Modelling | Scikit-learn (LogisticRegression, RandomForest, DummyClassifier) |
| Data | Pandas, NumPy |
| Audit log | SQLite via Python `sqlite3` |
| Tests | Pytest |

---

## Financial Context

The pipeline is designed around credit risk model validation workflows relevant to IFRS 9 provisioning and AIRB frameworks:

- **Segment-level performance** corresponds to validating PD/LGD models across retail, SME, and non-retail business lines
- **Schema gate** enforces input consistency — a critical check when model inputs arrive from upstream data pipelines
- **Macro-F1 threshold** reflects the use of class-imbalance-aware metrics for low-default portfolios
- **Error analysis** (False Negative profiling) maps to identifying borrower segments where a PD model underestimates default risk
- **SQL audit log** supports independent review and provides documentation for internal audit and regulatory inquiries
