
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parent.parent

TRAIN_PATH = PROJECT_ROOT / "data" / "reference" / "train.csv"
OUTPUT_PATH = PROJECT_ROOT / "artifacts" / "recall_target_comparison.csv"

TARGET_COLUMN = "Class"

FEATURE_COLUMNS = [
    "Time",
    "Amount",
    *[f"V{i}" for i in range(1, 29)],
]

RECALL_TARGETS = [0.70, 0.80, 0.90]


def load_data():
    data = pd.read_csv(TRAIN_PATH)

    required = FEATURE_COLUMNS + [TARGET_COLUMN]

    if data.empty or any(c not in data.columns for c in required):
        raise ValueError("Dataset is empty or missing required columns.")

    if data[required].isnull().any().any():
        raise ValueError("Dataset contains missing values.")

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


def find_threshold_for_recall(y_true, probabilities, target_recall):
    """
    Find the highest threshold that achieves at least
    the requested recall on the validation set.
    """

    candidate_thresholds = np.unique(probabilities)

    best_result = None

    for threshold in candidate_thresholds:
        predictions = (probabilities >= threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y_true,
            predictions,
            labels=[0, 1],
        ).ravel()

        recall = tp / (tp + fn) if (tp + fn) else 0

        if recall >= target_recall:
            precision = tp / (tp + fp) if (tp + fp) else 0

            if (
                best_result is None
                or threshold > best_result["threshold"]
            ):
                best_result = {
                    "threshold": float(threshold),
                    "precision": float(precision),
                    "recall": float(recall),
                    "true_positives": int(tp),
                    "false_positives": int(fp),
                    "false_negatives": int(fn),
                    "true_negatives": int(tn),
                    "flagged_transactions": int(tp + fp),
                }

    return best_result


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

    print("========== VALIDATION DATA ==========")
    print(f"Training rows: {len(X_train):,}")
    print(f"Validation rows: {len(X_val):,}")
    print(f"Validation fraud cases: {int(y_val.sum())}")

    all_results = []

    for model_name, model in build_models().items():

        print(f"\nTraining {model_name}...")

        pipeline = Pipeline([
            ("preprocessor", build_preprocessor()),
            ("model", model),
        ])

        pipeline.fit(X_train, y_train)

        probabilities = pipeline.predict_proba(X_val)[:, 1]

        pr_auc = average_precision_score(y_val, probabilities)

        print(f"Validation PR-AUC: {pr_auc:.5f}")

        for target in RECALL_TARGETS:

            result = find_threshold_for_recall(
                y_val,
                probabilities,
                target,
            )

            if result is None:
                print(
                    f"Could not achieve recall target {target:.0%}"
                )
                continue

            result["model"] = model_name
            result["target_recall"] = target
            result["pr_auc"] = float(pr_auc)

            all_results.append(result)

    results = pd.DataFrame(all_results)

    results = results[
        [
            "model",
            "target_recall",
            "threshold",
            "pr_auc",
            "precision",
            "recall",
            "true_positives",
            "false_positives",
            "false_negatives",
            "true_negatives",
            "flagged_transactions",
        ]
    ]

    results.to_csv(OUTPUT_PATH, index=False)

    print("\n========== RECALL-CONSTRAINED COMPARISON ==========")

    print(
        results.to_string(
            index=False,
            formatters={
                "target_recall": "{:.0%}".format,
                "threshold": "{:.5f}".format,
                "pr_auc": "{:.5f}".format,
                "precision": "{:.4f}".format,
                "recall": "{:.4f}".format,
            },
        )
    )

    print("\nResults saved to:")
    print(OUTPUT_PATH)

    print("\nRecall-constrained comparison completed.")


if __name__ == "__main__":
    main()
