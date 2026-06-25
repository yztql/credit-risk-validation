"""
Unit tests for the validation pipeline components.
Run with:  pytest tests/ -v
"""

import numpy as np
import pandas as pd
import pytest

from src.data_generator import (
    generate_clean_validation,
    generate_duplicates,
    generate_invalid_inputs,
    generate_missing_values,
    generate_schema_mismatch,
    generate_training_data,
)
from src.data_quality import DataQualityChecker
from src.metrics import benchmark_comparison, compute_metrics, error_analysis, segment_metrics
from src.model_trainer import train_champion, train_naive_baseline


# ── fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def train_df():
    return generate_training_data(n=500, seed=42)


@pytest.fixture(scope="module")
def champion(train_df):
    return train_champion(train_df)


@pytest.fixture(scope="module")
def naive(train_df):
    return train_naive_baseline(train_df)


# ── data quality ──────────────────────────────────────────────────────────────

class TestDataQuality:
    def setup_method(self):
        self.checker = DataQualityChecker()

    def test_clean_data_passes_all_checks(self):
        df = generate_clean_validation(n=200, seed=1)
        findings = self.checker.run_all(df)
        fails = [f for f in findings if f.status == "FAIL"]
        assert not fails, f"Expected no failures on clean data, got: {[f.check for f in fails]}"

    def test_missing_values_detected(self):
        df = generate_missing_values(n=200, seed=2)
        f = self.checker.check_missing_values(df)
        assert f.status == "FAIL"
        assert f.affected_rows > 0

    def test_duplicates_detected(self):
        df = generate_duplicates(n=200, seed=3)
        f = self.checker.check_duplicates(df)
        assert f.status == "FAIL"
        assert f.affected_rows > 0

    def test_schema_mismatch_detected(self):
        df = generate_schema_mismatch(n=200, seed=4)
        f = self.checker.check_schema(df)
        assert f.status == "FAIL"
        assert f.severity == "CRITICAL"

    def test_invalid_ranges_detected(self):
        df = generate_invalid_inputs(n=200, seed=5)
        f = self.checker.check_invalid_ranges(df)
        assert f.status == "FAIL"
        assert len(f.affected_cols) > 0

    def test_id_uniqueness_pass(self):
        df = generate_clean_validation(n=100, seed=6)
        f = self.checker.check_id_uniqueness(df)
        assert f.status == "PASS"

    def test_id_uniqueness_fail_on_duplicates(self):
        df = generate_duplicates(n=100, seed=7)
        f = self.checker.check_id_uniqueness(df)
        assert f.status == "FAIL"


# ── metrics ───────────────────────────────────────────────────────────────────

class TestMetrics:
    def test_compute_metrics_returns_required_keys(self):
        y_true = pd.Series([0, 1, 0, 1, 1, 0])
        y_pred = np.array([0, 1, 0, 0, 1, 1])
        m = compute_metrics(y_true, y_pred)
        for key in ("macro_f1", "TN", "FP", "FN", "TP", "n_samples", "confusion_matrix"):
            assert key in m, f"Missing key: {key}"

    def test_confusion_matrix_sums_to_n(self):
        y_true = pd.Series([0, 1, 0, 1, 1, 0])
        y_pred = np.array([0, 1, 1, 0, 1, 0])
        m = compute_metrics(y_true, y_pred)
        assert m["TN"] + m["FP"] + m["FN"] + m["TP"] == len(y_true)

    def test_segment_metrics_returns_one_row_per_segment(self):
        df = generate_clean_validation(n=300, seed=10)
        champion = train_champion(generate_training_data(n=400, seed=42))
        X = df[["credit_score","annual_income","dti_ratio","loan_amount",
                "employment_years","num_derogatory_marks","payment_history"]]
        y_pred = champion.predict(X)
        seg_df = segment_metrics(df, y_pred)
        assert set(seg_df["segment"]).issubset({"Retail", "SME", "Corporate"})
        assert len(seg_df) >= 1

    def test_error_analysis_has_fn_and_fp_columns(self):
        df = generate_clean_validation(n=300, seed=11)
        champion = train_champion(generate_training_data(n=400, seed=42))
        X = df[["credit_score","annual_income","dti_ratio","loan_amount",
                "employment_years","num_derogatory_marks","payment_history"]]
        y_pred = champion.predict(X)
        err = error_analysis(df, y_pred)
        assert "count" in err.columns

    def test_macro_f1_above_naive_baseline(self, champion, naive, train_df):
        val = generate_clean_validation(n=300, seed=20)
        X = val[["credit_score","annual_income","dti_ratio","loan_amount",
                 "employment_years","num_derogatory_marks","payment_history"]]
        y_true       = val["default"].astype(int)
        champ_pred   = champion.predict(X)
        naive_pred   = naive.predict(X)
        bench_df = benchmark_comparison(y_true, champ_pred, champ_pred, naive_pred)
        champ_f1 = bench_df.loc[bench_df["model"].str.contains("Champion"), "macro_f1"].values[0]
        naive_f1 = bench_df.loc[bench_df["model"].str.contains("Naive"),    "macro_f1"].values[0]
        assert champ_f1 >= naive_f1, "Champion must outperform naive baseline"


# ── model ────────────────────────────────────────────────────────────────────

class TestModel:
    def test_champion_predict_shape(self, champion):
        val = generate_clean_validation(n=50, seed=30)
        X = val[["credit_score","annual_income","dti_ratio","loan_amount",
                 "employment_years","num_derogatory_marks","payment_history"]]
        preds = champion.predict(X)
        assert len(preds) == len(val)

    def test_champion_outputs_binary(self, champion):
        val = generate_clean_validation(n=50, seed=31)
        X = val[["credit_score","annual_income","dti_ratio","loan_amount",
                 "employment_years","num_derogatory_marks","payment_history"]]
        preds = champion.predict(X)
        assert set(preds).issubset({0, 1})

    def test_predict_proba_between_zero_and_one(self, champion):
        val = generate_clean_validation(n=50, seed=32)
        X = val[["credit_score","annual_income","dti_ratio","loan_amount",
                 "employment_years","num_derogatory_marks","payment_history"]]
        probs = champion.predict_proba(X)[:, 1]
        assert (probs >= 0).all() and (probs <= 1).all()
