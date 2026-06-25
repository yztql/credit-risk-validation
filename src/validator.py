"""
Orchestrates a single validation run across all pipeline stages:
  1. Data quality checks
  2. Schema gate  — aborts model evaluation if required columns are missing
  3. Overall model metrics (confusion matrix, macro-F1, ROC-AUC)
  4. Segment-level performance
  5. Benchmark comparison (champion vs. RF vs. naive)
  6. Error analysis (feature profiles of FN / FP errors)
  7. SQL logging of all findings
"""

import uuid
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

from src.data_quality import DataQualityChecker
from src.metrics import (
    benchmark_comparison,
    compute_metrics,
    error_analysis,
    segment_metrics,
)
from src.model_trainer import FEATURE_COLS, TARGET_COL
from src.sql_logger import ValidationLogger

MACRO_F1_THRESHOLD = 0.52   # minimum acceptable macro-F1 for a PASS verdict


def _predict(model, df: pd.DataFrame):
    """Return (filtered_df, y_pred, y_prob) dropping rows with missing features."""
    valid = df.dropna(subset=[c for c in FEATURE_COLS if c in df.columns])
    X = valid[[c for c in FEATURE_COLS if c in valid.columns]]
    if X.shape[1] < len(FEATURE_COLS):
        return valid, None, None          # missing required feature columns
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else None
    return valid, y_pred, y_prob


def run_validation(
    df: pd.DataFrame,
    champion,
    benchmark,
    naive,
    dataset_name: str,
    logger: Optional[ValidationLogger] = None,
    verbose: bool = True,
) -> dict:
    run_id = f"RUN_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6].upper()}"
    separator = "=" * 64

    if verbose:
        print(f"\n{separator}")
        print(f"  Validation Run : {run_id}")
        print(f"  Dataset        : {dataset_name}  ({len(df):,} rows)")
        print(separator)

    # ── 1. Data quality ───────────────────────────────────────────────────────
    checker  = DataQualityChecker()
    findings = checker.run_all(df)
    dq_fails = [f for f in findings if f.status == "FAIL"]

    if verbose:
        print("\n[1] DATA QUALITY CHECKS")
        for f in findings:
            icon = "✓" if f.status == "PASS" else ("✗" if f.status == "FAIL" else "!")
            print(f"    {icon}  [{f.severity:<8}]  {f.check:<20}  {f.detail}")

    # ── 2. Schema gate ────────────────────────────────────────────────────────
    has_features = all(c in df.columns for c in FEATURE_COLS)
    has_target   = TARGET_COL in df.columns

    if not has_features:
        if verbose:
            missing = [c for c in FEATURE_COLS if c not in df.columns]
            print(f"\n  [!] Schema gate: required feature columns missing {missing}.")
            print(      "      Skipping model evaluation.")
        status = "ABORTED — SCHEMA ERROR"
        result = {"run_id": run_id, "status": status, "dq_findings": findings}
        if logger:
            logger.log_run(run_id, dataset_name, "champion_LR", status, len(df), len(dq_fails))
            logger.log_dq_findings(run_id, findings)
        return result

    # ── 3. Overall model metrics ──────────────────────────────────────────────
    valid_df, champ_pred, champ_prob = _predict(champion,  df)
    _,        bench_pred, _          = _predict(benchmark, df)
    _,        naive_pred, _          = _predict(naive,     df)

    if not has_target or champ_pred is None:
        if verbose:
            print("\n  [!] Target column missing — cannot compute performance metrics.")
        return {"run_id": run_id, "status": "ABORTED", "dq_findings": findings}

    y_true   = valid_df[TARGET_COL].astype(int)
    overall  = compute_metrics(y_true, champ_pred, champ_prob)
    f1_pass  = overall["macro_f1"] >= MACRO_F1_THRESHOLD

    if verbose:
        print("\n[2] OVERALL MODEL PERFORMANCE  (Champion — Logistic Regression)")
        print(f"    Macro F1       : {overall['macro_f1']:.4f}  "
              f"({'PASS' if f1_pass else 'FAIL — below threshold ' + str(MACRO_F1_THRESHOLD)})")
        print(f"    ROC-AUC        : {overall['roc_auc'] or 'N/A'}")
        print(f"    Precision      : {overall['precision_class1']:.4f}")
        print(f"    Recall         : {overall['recall_class1']:.4f}")
        print(f"    Confusion Matrix  TN={overall['TN']}  FP={overall['FP']}  "
              f"FN={overall['FN']}  TP={overall['TP']}")

    # ── 4. Segment-level performance ──────────────────────────────────────────
    seg_df = pd.DataFrame()
    if "segment" in valid_df.columns:
        seg_df = segment_metrics(valid_df, champ_pred)
        if verbose:
            print("\n[3] SEGMENT-LEVEL PERFORMANCE")
            print(seg_df.to_string(index=False))

    # ── 5. Benchmark comparison ───────────────────────────────────────────────
    bench_df = benchmark_comparison(y_true, champ_pred, bench_pred, naive_pred)
    if verbose:
        print("\n[4] BENCHMARK COMPARISON")
        print(bench_df.to_string(index=False))

    # ── 6. Error analysis ─────────────────────────────────────────────────────
    err_df = error_analysis(valid_df, champ_pred)
    if verbose and not err_df.empty:
        print("\n[5] ERROR ANALYSIS — avg feature values by misclassification type")
        print(err_df.to_string())

    # ── 7. Overall verdict ────────────────────────────────────────────────────
    status = "PASS" if (not dq_fails and f1_pass) else "FAIL"
    n_findings = len(dq_fails) + (0 if f1_pass else 1)

    if verbose:
        print(f"\n{'─'*64}")
        print(f"  RESULT: {status}   |   {len(dq_fails)} DQ finding(s)   |   "
              f"macro-F1 = {overall['macro_f1']:.4f}")
        print(f"{'─'*64}")

    # ── 8. SQL logging ────────────────────────────────────────────────────────
    if logger:
        logger.log_run(run_id, dataset_name, "champion_LR", status, len(df), n_findings)
        logger.log_dq_findings(run_id, findings)
        if not seg_df.empty:
            logger.log_segment_metrics(run_id, seg_df)
        logger.log_benchmark(run_id, bench_df)

    return {
        "run_id":          run_id,
        "status":          status,
        "dq_findings":     findings,
        "overall_metrics": overall,
        "segment_metrics": seg_df,
        "benchmark":       bench_df,
        "error_analysis":  err_df,
    }
