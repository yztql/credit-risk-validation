"""
Credit Risk Model Evaluation and Validation Pipeline
-----------------------------------------------------
Usage:  python run_validation.py

Runs the full validation suite across five scenarios:
  1. Clean validation set          (expected: PASS)
  2. Missing values injected       (expected: FAIL — DQ)
  3. Duplicate rows injected       (expected: FAIL — DQ)
  4. Schema mismatch               (expected: ABORTED — schema gate)
  5. Invalid input values          (expected: FAIL — DQ)

All results are logged to validation_log.db (SQLite).
"""

from pathlib import Path

import pandas as pd

from src.data_generator import save_all
from src.model_trainer import (
    load_model,
    save_models,
    train_benchmark,
    train_champion,
    train_naive_baseline,
)
from src.sql_logger import ValidationLogger
from src.validator import run_validation

DATA_DIR  = Path("data")
MODEL_DIR = Path("models")

SCENARIOS = {
    "val_clean":           "Clean Validation Set",
    "val_missing":         "Missing Values",
    "val_duplicates":      "Duplicate Rows",
    "val_schema_mismatch": "Schema Mismatch",
    "val_invalid_inputs":  "Invalid Input Values",
}


def main() -> None:
    # 1. Generate synthetic data
    if not DATA_DIR.exists() or not (DATA_DIR / "train.csv").exists():
        print("Generating synthetic datasets...")
        save_all(str(DATA_DIR))

    # 2. Train or load models
    if (MODEL_DIR / "champion.pkl").exists():
        print("Loading existing models...")
        champion  = load_model(str(MODEL_DIR / "champion.pkl"))
        benchmark = load_model(str(MODEL_DIR / "benchmark.pkl"))
    else:
        print("Training models on synthetic credit data...")
        train_df  = pd.read_csv(DATA_DIR / "train.csv")
        champion  = train_champion(train_df)
        benchmark = train_benchmark(train_df)
        save_models(champion, benchmark, str(MODEL_DIR))

    train_df = pd.read_csv(DATA_DIR / "train.csv")
    naive    = train_naive_baseline(train_df)

    # 3. Initialise SQL audit logger
    logger = ValidationLogger()

    # 4. Run validation across all scenarios
    print(f"\nRunning {len(SCENARIOS)} validation scenarios...\n")
    results = {}
    for filename, label in SCENARIOS.items():
        df = pd.read_csv(DATA_DIR / f"{filename}.csv")
        results[label] = run_validation(
            df, champion, benchmark, naive,
            dataset_name=label,
            logger=logger,
            verbose=True,
        )

    # 5. Print suite summary
    print("\n" + "=" * 64)
    print("VALIDATION SUITE SUMMARY")
    print("=" * 64)
    print(f"  {'Scenario':<32} {'Status':<28} {'Macro F1':<10} {'DQ Fails'}")
    print("  " + "-" * 60)
    for label, r in results.items():
        f1     = r.get("overall_metrics", {}).get("macro_f1")
        f1_str = f"{f1:.4f}" if isinstance(f1, float) else "N/A"
        n_fail = sum(1 for f in r.get("dq_findings", []) if f.status == "FAIL")
        print(f"  {label:<32} {r['status']:<28} {f1_str:<10} {n_fail}")

    # 6. Show audit log from SQL
    print("\nAudit Log — validation_log.db:")
    history = logger.run_history()
    print(history[["run_id", "dataset_name", "status", "n_samples", "n_findings",
                   "timestamp"]].to_string(index=False))


if __name__ == "__main__":
    main()
