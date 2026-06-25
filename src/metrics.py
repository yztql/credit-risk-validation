"""
Quantitative metrics for credit risk model validation.

Produces confusion matrices, macro-F1, segment-level breakdowns,
benchmark comparisons, and error analysis — all structured for
SQL logging and audit-ready reporting.
"""

from typing import Dict, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

FEATURE_COLS = [
    "credit_score", "annual_income", "dti_ratio", "loan_amount",
    "employment_years", "num_derogatory_marks", "payment_history",
]
TARGET_COL = "default"


def compute_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
) -> Dict:
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn = fp = fn = tp = 0

    metrics = {
        "macro_f1":        round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "precision_class1": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall_class1":    round(recall_score(y_true, y_pred, zero_division=0), 4),
        "n_samples":        int(len(y_true)),
        "n_defaults":       int(y_true.sum()),
        "default_rate":     round(float(y_true.mean()), 4),
        "confusion_matrix": cm.tolist(),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
    }

    if y_prob is not None:
        try:
            metrics["roc_auc"] = round(roc_auc_score(y_true, y_prob), 4)
        except ValueError:
            metrics["roc_auc"] = None
    else:
        metrics["roc_auc"] = None

    return metrics


def segment_metrics(df: pd.DataFrame, y_pred: np.ndarray, segment_col: str = "segment") -> pd.DataFrame:
    df = df.copy()
    df["_pred"] = y_pred
    rows = []
    for seg, grp in df.groupby(segment_col):
        valid = grp.dropna(subset=[TARGET_COL])
        if len(valid) < 5:
            continue
        yt = valid[TARGET_COL].astype(int)
        yp = valid["_pred"]
        m = compute_metrics(yt, yp)
        rows.append({
            "segment":      seg,
            "n":            m["n_samples"],
            "default_rate": m["default_rate"],
            "macro_f1":     m["macro_f1"],
            "precision":    m["precision_class1"],
            "recall":       m["recall_class1"],
            "roc_auc":      m.get("roc_auc"),
        })
    return pd.DataFrame(rows)


def error_analysis(df: pd.DataFrame, y_pred: np.ndarray) -> pd.DataFrame:
    """Return mean feature values by error type (FN / FP) vs correct predictions."""
    df = df.copy()
    df["_pred"] = y_pred
    df = df.dropna(subset=[TARGET_COL])
    yt = df[TARGET_COL].astype(int)

    df["error_type"] = "Correct"
    df.loc[(yt == 1) & (df["_pred"] == 0), "error_type"] = "False Negative (missed default)"
    df.loc[(yt == 0) & (df["_pred"] == 1), "error_type"] = "False Positive (false alarm)"

    available = [c for c in FEATURE_COLS if c in df.columns]
    summary = df.groupby("error_type")[available].mean().round(3)
    summary.insert(0, "count", df.groupby("error_type").size())
    return summary


def benchmark_comparison(
    y_true: pd.Series,
    champion_pred: np.ndarray,
    benchmark_pred: np.ndarray,
    naive_pred: np.ndarray,
) -> pd.DataFrame:
    rows = []
    for name, pred in [
        ("Champion — Logistic Regression", champion_pred),
        ("Benchmark — Random Forest",      benchmark_pred),
        ("Naive Baseline — Stratified",    naive_pred),
    ]:
        rows.append({
            "model":             name,
            "macro_f1":          round(f1_score(y_true, pred, average="macro", zero_division=0), 4),
            "recall_default":    round(recall_score(y_true, pred, zero_division=0), 4),
            "precision_default": round(precision_score(y_true, pred, zero_division=0), 4),
        })
    return pd.DataFrame(rows)
