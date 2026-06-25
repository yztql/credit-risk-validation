"""
Data quality checks for credit risk model validation.

Each check returns a Finding dataclass with status (PASS/FAIL/WARN),
severity, and details — structured for SQL logging and audit reports.
"""

from dataclasses import dataclass, field
from typing import List
import pandas as pd
import numpy as np

EXPECTED_SCHEMA = {
    "customer_id":          "string",
    "credit_score":         "numeric",
    "annual_income":        "numeric",
    "dti_ratio":            "numeric",
    "loan_amount":          "numeric",
    "employment_years":     "numeric",
    "num_derogatory_marks": "numeric",
    "payment_history":      "numeric",
    "segment":              "string",
    "product_type":         "string",
    "default":              "numeric",
}

VALID_RANGES = {
    "credit_score":         (300,  850),
    "annual_income":        (0,    None),
    "dti_ratio":            (0.0,  1.0),
    "loan_amount":          (0,    None),
    "employment_years":     (0,    50),
    "payment_history":      (0.0,  1.0),
    "num_derogatory_marks": (0,    20),
}


@dataclass
class Finding:
    check:        str
    status:       str            # PASS / FAIL / WARN
    detail:       str
    severity:     str            # INFO / LOW / MEDIUM / HIGH / CRITICAL
    affected_rows: int           = 0
    affected_cols: List[str]     = field(default_factory=list)


class DataQualityChecker:
    def __init__(self, schema: dict = None, valid_ranges: dict = None):
        self.schema       = schema       or EXPECTED_SCHEMA
        self.valid_ranges = valid_ranges or VALID_RANGES

    # ── individual checks ────────────────────────────────────────────────────

    def check_schema(self, df: pd.DataFrame) -> Finding:
        missing_cols = [c for c in self.schema if c not in df.columns]
        type_errors  = []
        for col, expected in self.schema.items():
            if col not in df.columns:
                continue
            dtype = df[col].dtype
            if expected == "numeric":
                ok = pd.api.types.is_numeric_dtype(dtype)
            else:
                ok = pd.api.types.is_object_dtype(dtype) or pd.api.types.is_string_dtype(dtype)
            if not ok:
                type_errors.append(f"{col} (expected {expected}, got {dtype})")

        issues = []
        if missing_cols:
            issues.append(f"Missing columns: {missing_cols}")
        if type_errors:
            issues.append(f"Type mismatches: {type_errors}")
        if not issues:
            return Finding("schema", "PASS", "Schema matches expected structure.", "INFO")

        severity = "CRITICAL" if missing_cols else "HIGH"
        return Finding(
            "schema", "FAIL", "; ".join(issues), severity,
            affected_cols=missing_cols + [e.split()[0] for e in type_errors],
        )

    def check_missing_values(self, df: pd.DataFrame) -> Finding:
        missing = df.isnull().sum()
        missing = missing[missing > 0]
        if missing.empty:
            return Finding("missing_values", "PASS", "No missing values detected.", "INFO")
        details = ", ".join(f"{c}: {v} ({v/len(df):.1%})" for c, v in missing.items())
        severity = "HIGH" if (missing / len(df)).max() > 0.10 else "MEDIUM"
        return Finding(
            "missing_values", "FAIL",
            f"Missing values found — {details}",
            severity,
            affected_rows=int(df[missing.index].isnull().any(axis=1).sum()),
            affected_cols=list(missing.index),
        )

    def check_duplicates(self, df: pd.DataFrame) -> Finding:
        n_dups = int(df.duplicated().sum())
        if n_dups == 0:
            return Finding("duplicates", "PASS", "No duplicate rows detected.", "INFO")
        pct = n_dups / len(df)
        severity = "HIGH" if pct > 0.05 else "MEDIUM"
        return Finding(
            "duplicates", "FAIL",
            f"{n_dups} duplicate rows ({pct:.1%} of dataset).",
            severity,
            affected_rows=n_dups,
        )

    def check_id_uniqueness(self, df: pd.DataFrame, id_col: str = "customer_id") -> Finding:
        if id_col not in df.columns:
            return Finding("id_uniqueness", "WARN", f"ID column '{id_col}' not found.", "LOW")
        n_dups = int(df[id_col].duplicated().sum())
        if n_dups == 0:
            return Finding("id_uniqueness", "PASS", f"All {id_col} values are unique.", "INFO")
        return Finding(
            "id_uniqueness", "FAIL",
            f"{n_dups} duplicate {id_col} values detected.",
            "HIGH",
            affected_rows=n_dups,
            affected_cols=[id_col],
        )

    def check_invalid_ranges(self, df: pd.DataFrame) -> Finding:
        violations: dict = {}
        for col, (lo, hi) in self.valid_ranges.items():
            if col not in df.columns:
                continue
            series = pd.to_numeric(df[col], errors="coerce")
            mask = pd.Series(False, index=df.index)
            if lo is not None:
                mask |= series < lo
            if hi is not None:
                mask |= series > hi
            n = int(mask.sum())
            if n:
                violations[col] = n
        if not violations:
            return Finding("invalid_ranges", "PASS", "All values within expected ranges.", "INFO")
        details = ", ".join(f"{c}: {n} rows" for c, n in violations.items())
        return Finding(
            "invalid_ranges", "FAIL",
            f"Out-of-range values detected — {details}",
            "HIGH",
            affected_rows=sum(violations.values()),
            affected_cols=list(violations.keys()),
        )

    def check_class_imbalance(self, df: pd.DataFrame, target: str = "default") -> Finding:
        if target not in df.columns:
            return Finding("class_imbalance", "WARN", f"Target column '{target}' not found.", "LOW")
        if not pd.api.types.is_numeric_dtype(df[target]):
            return Finding("class_imbalance", "WARN", f"Target column '{target}' is non-numeric.", "LOW")
        rate = float(df[target].mean())
        if rate < 0.02 or rate > 0.50:
            return Finding(
                "class_imbalance", "WARN",
                f"Default rate {rate:.1%} — potential class imbalance; use macro-F1 over accuracy.",
                "MEDIUM",
            )
        return Finding("class_imbalance", "PASS", f"Default rate {rate:.1%} — acceptable.", "INFO")

    # ── run all ──────────────────────────────────────────────────────────────

    def run_all(self, df: pd.DataFrame) -> List[Finding]:
        return [
            self.check_schema(df),
            self.check_missing_values(df),
            self.check_duplicates(df),
            self.check_id_uniqueness(df),
            self.check_invalid_ranges(df),
            self.check_class_imbalance(df),
        ]
