
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parent.parent

TRAIN_PATH = PROJECT_ROOT / "data" / "reference" / "train.csv"
OUTPUT_PATH = PROJECT_ROOT / "artifacts" / "model_threshold_comparison.csv"

TARGET_COLUMN = "Class"

FEATURE_COLUMNS = [
    "Time",
    "Amount",
    *[f"V{i}" for i in range(1, 29)],
]

THRESHOLDS = [
    0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95
]


def load_data():
    print("Loading training data...")

    data = pd.read_csv(TRAIN_PATH)

    required = FEATURE_COLUMNS + [TARGET_COLUMN]

    if data.empty or any(c not in data.columns for c in required):
        raise ValueError("Training data is empty or missing columns.")

    if data[required].isnull().any().any():
        raise ValueError("Training data contains missing values.")

    data = data.sort_values("Time", kind="mergesort").reset_index(drop=True)

    return data


def build_preprocessor():
    return ColumnTransformer(
        transformers=[
            ("time", StandardScaler(), ["Time"]),
            (
                "amount",
                Pipeline([
                    ("log", FunctionTransformer(np.log1p, validate=False)),
                    ("scale", StandardScaler()),
                ]),
                ["Amount"],
            ),
            (
                "pca",
                StandardScaler(),
                [f"V{i}" for i in range(1, 29)],
            ),
        ],
        remainder="drop",
    )


def build_models():
    return {
        "logistic_regression": LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            random_state=42,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=42,
        ),
    }


def evaluate_model(name, model, X_train, y_train, X_val, y_val):
    print(f"\nTraining {name}...")

    pipeline = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("model", model),
    ])

    pipeline.fit(X_train, y_train)

    probabilities = pipeline.predict_proba(X_val)[:, 1]

    pr_auc = average_precision_score(y_val, probabilities)

    print(f"Validation PR-AUC: {pr_auc:.5f}")

    rows = []

    for threshold in THRESHOLDS:
        predictions = (probabilities >= threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y_val,
            predictions,
            labels=[0, 1],
        ).ravel()

        rows.append({
            "model": name,
            "threshold": threshold,
            "pr_auc": pr_auc,
            "precision": precision_score(
                y_val, predictions, zero_division=0
            ),
            "recall": recall_score(
                y_val, predictions, zero_division=0
            ),
            "f1_score": f1_score(
                y_val, predictions, zero_division=0
            ),
            "true_positives": int(tp),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_negatives": int(tn),
            "flagged_transactions": int(tp + fp),
        })

    return rows


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    data = load_data()

    split_index = int(len(data) * 0.80)

    train_data = data.iloc[:split_index]
    validation_data = data.iloc[split_index:]

    X_train = train_data[FEATURE_COLUMNS]
    y_train = train_data[TARGET_COLUMN]

    X_val = validation_data[FEATURE_COLUMNS]
    y_val = validation_data[TARGET_COLUMN]

    print("\n========== DATA SPLIT ==========")
    print(f"Training rows: {len(X_train):,}")
    print(f"Validation rows: {len(X_val):,}")
    print(f"Validation fraud cases: {int(y_val.sum())}")

    all_results = []

    for name, model in build_models().items():
        rows = evaluate_model(
            name, model, X_train, y_train, X_val, y_val
        )
        all_results.extend(rows)

    results = pd.DataFrame(all_results)

    results.to_csv(OUTPUT_PATH, index=False)

    print("\n========== MODEL / THRESHOLD COMPARISON ==========")

    display_columns = [
        "model",
        "threshold",
        "precision",
        "recall",
        "f1_score",
        "false_positives",
        "false_negatives",
        "flagged_transactions",
    ]

    print(
        results[display_columns].to_string(
            index=False,
            formatters={
                "precision": "{:.4f}".format,
                "recall": "{:.4f}".format,
                "f1_score": "{:.4f}".format,
            },
        )
    )

    print("\nFull comparison saved to:")
    print(OUTPUT_PATH)

    print("\nModel threshold comparison completed.")


if __name__ == "__main__":
    main()
