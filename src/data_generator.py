"""
Generate synthetic credit risk datasets for validation pipeline testing.

Produces one training set and five validation scenarios — clean, missing values,
duplicate rows, schema mismatch, and invalid input ranges.
"""

import numpy as np
import pandas as pd
from pathlib import Path

SEGMENTS = ["Retail", "SME", "Corporate"]
PRODUCTS = ["Mortgage", "Auto Loan", "Credit Card", "Term Loan"]

FEATURE_COLS = [
    "credit_score", "annual_income", "dti_ratio", "loan_amount",
    "employment_years", "num_derogatory_marks", "payment_history",
]


def _generate_base(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    credit_score         = rng.integers(450, 851, n)
    annual_income        = np.round(rng.lognormal(11.0, 0.5, n), 2)
    dti_ratio            = np.round(rng.beta(2, 5, n), 4)
    loan_amount          = np.round(rng.lognormal(10.5, 0.7, n), 2)
    employment_years     = np.round(rng.exponential(5, n).clip(0, 30), 1)
    num_derogatory_marks = rng.integers(0, 6, n)
    payment_history      = np.round(rng.beta(8, 2, n), 4)
    segment              = rng.choice(SEGMENTS, n, p=[0.60, 0.30, 0.10])
    product_type         = rng.choice(PRODUCTS, n)

    log_odds = (
        -4.0
        + (700 - credit_score) * 0.01
        + dti_ratio * 3.0
        - payment_history * 2.5
        + num_derogatory_marks * 0.5
        - employment_years * 0.05
    )
    prob_default = 1 / (1 + np.exp(-log_odds))
    default = rng.binomial(1, prob_default).astype("int64")

    return pd.DataFrame({
        "customer_id":          [f"C{i:06d}" for i in range(n)],
        "credit_score":         credit_score.astype("int64"),
        "annual_income":        annual_income,
        "dti_ratio":            dti_ratio,
        "loan_amount":          loan_amount,
        "employment_years":     employment_years,
        "num_derogatory_marks": num_derogatory_marks.astype("int64"),
        "payment_history":      payment_history,
        "segment":              segment,
        "product_type":         product_type,
        "default":              default,
    })


def generate_training_data(n: int = 4000, seed: int = 42) -> pd.DataFrame:
    return _generate_base(n, seed)


def generate_clean_validation(n: int = 1000, seed: int = 99) -> pd.DataFrame:
    return _generate_base(n, seed)


def generate_missing_values(n: int = 1000, seed: int = 100, missing_frac: float = 0.08) -> pd.DataFrame:
    df = _generate_base(n, seed)
    rng = np.random.default_rng(seed + 1)
    for col in ["credit_score", "annual_income", "dti_ratio", "payment_history"]:
        mask = rng.random(n) < missing_frac
        df.loc[mask, col] = np.nan
    return df


def generate_duplicates(n: int = 1000, seed: int = 101, dup_frac: float = 0.05) -> pd.DataFrame:
    df = _generate_base(n, seed)
    dups = df.sample(int(n * dup_frac), random_state=seed)
    return pd.concat([df, dups], ignore_index=True)


def generate_schema_mismatch(n: int = 1000, seed: int = 102) -> pd.DataFrame:
    """Renames two required columns and casts credit_score to string."""
    df = _generate_base(n, seed)
    df = df.rename(columns={
        "dti_ratio":            "debt_to_income",
        "num_derogatory_marks": "derog_marks",
    })
    df["credit_score"] = df["credit_score"].astype(str)
    return df


def generate_invalid_inputs(n: int = 1000, seed: int = 103, invalid_frac: float = 0.05) -> pd.DataFrame:
    """Injects impossible values: negative income, credit score < 0, DTI > 1."""
    df = _generate_base(n, seed)
    rng = np.random.default_rng(seed + 1)
    idx = rng.choice(n, int(n * invalid_frac), replace=False)
    third = len(idx) // 3
    df.loc[idx[:third],       "credit_score"]  = -999
    df.loc[idx[third:2*third], "annual_income"] = -50_000.0
    df.loc[idx[2*third:],      "dti_ratio"]     = 5.0
    return df


def save_all(output_dir: str = "data") -> None:
    Path(output_dir).mkdir(exist_ok=True)
    generate_training_data().to_csv(       f"{output_dir}/train.csv",               index=False)
    generate_clean_validation().to_csv(    f"{output_dir}/val_clean.csv",           index=False)
    generate_missing_values().to_csv(      f"{output_dir}/val_missing.csv",         index=False)
    generate_duplicates().to_csv(          f"{output_dir}/val_duplicates.csv",      index=False)
    generate_schema_mismatch().to_csv(     f"{output_dir}/val_schema_mismatch.csv", index=False)
    generate_invalid_inputs().to_csv(      f"{output_dir}/val_invalid_inputs.csv",  index=False)
    print(f"Generated 6 datasets -> {output_dir}/")


if __name__ == "__main__":
    save_all()
