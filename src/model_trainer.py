"""
Train champion, benchmark, and naive baseline models on credit risk training data.

Champion  — Logistic Regression (interpretable, audit-friendly)
Benchmark — Random Forest       (challenger model for performance comparison)
Naive     — DummyClassifier     (floor; macro-F1 must exceed this to be meaningful)
"""

import pickle
from pathlib import Path

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURE_COLS = [
    "credit_score", "annual_income", "dti_ratio", "loan_amount",
    "employment_years", "num_derogatory_marks", "payment_history",
]
TARGET_COL = "default"


def _prep(df: pd.DataFrame):
    df = df.dropna(subset=FEATURE_COLS + [TARGET_COL])
    return df[FEATURE_COLS], df[TARGET_COL].astype(int)


def train_champion(df: pd.DataFrame) -> Pipeline:
    X, y = _prep(df)
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model",  LogisticRegression(class_weight="balanced", max_iter=500, random_state=42)),
    ])
    pipe.fit(X, y)
    return pipe


def train_benchmark(df: pd.DataFrame) -> Pipeline:
    X, y = _prep(df)
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model",  RandomForestClassifier(
            n_estimators=100, max_depth=5, class_weight="balanced", random_state=42,
        )),
    ])
    pipe.fit(X, y)
    return pipe


def train_naive_baseline(df: pd.DataFrame) -> DummyClassifier:
    X, y = _prep(df)
    clf = DummyClassifier(strategy="stratified", random_state=42)
    clf.fit(X, y)
    return clf


def save_models(champion: Pipeline, benchmark: Pipeline, output_dir: str = "models") -> None:
    Path(output_dir).mkdir(exist_ok=True)
    with open(f"{output_dir}/champion.pkl",  "wb") as f:
        pickle.dump(champion,  f)
    with open(f"{output_dir}/benchmark.pkl", "wb") as f:
        pickle.dump(benchmark, f)
    print(f"Models saved -> {output_dir}/")


def load_model(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)
