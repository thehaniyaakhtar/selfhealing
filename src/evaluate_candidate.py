
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TRAIN_PATH = PROJECT_ROOT / "data" / "reference" / "train.csv"
EVALUATION_PATH = PROJECT_ROOT / "data" / "reference" / "evaluation.csv"

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
METRICS_PATH = ARTIFACTS_DIR / "candidate_evaluation.json"
MODEL_PATH = ARTIFACTS_DIR / "random_forest_candidate.joblib"

TARGET_COLUMN = "Class"

FEATURE_COLUMNS = [
    "Time",
    "Amount",
    *[f"V{i}" for i in range(1, 29)],
]

# Chosen using the validation set only.
DECISION_THRESHOLD = 0.395


# --------------------------------------------------
# LOAD AND VALIDATE DATA
# --------------------------------------------------

def load_dataset(path, dataset_name):

    if not path.exists():
        raise FileNotFoundError(
            f"{dataset_name} dataset not found: {path}"
        )

    data = pd.read_csv(path)

    required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]

    missing = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing:
        raise ValueError(
            f"{dataset_name} is missing columns: {missing}"
        )

    if data.empty:
        raise ValueError(f"{dataset_name} is empty.")

    if data[required_columns].isnull().any().any():
        raise ValueError(
            f"{dataset_name} contains missing values."
        )

    if not set(data[TARGET_COLUMN].unique()).issubset({0, 1}):
        raise ValueError(
            f"{dataset_name} has invalid target values."
        )

    if not np.isfinite(
        data[FEATURE_COLUMNS].to_numpy(dtype=float)
    ).all():
        raise ValueError(
            f"{dataset_name} contains non-finite features."
        )

    return data


# --------------------------------------------------
# PREPROCESSING
# --------------------------------------------------

def build_preprocessor():

    return ColumnTransformer(
        transformers=[
            (
                "time",
                StandardScaler(),
                ["Time"],
            ),
            (
                "amount",
                Pipeline([
                    (
                        "log_transform",
                        FunctionTransformer(
                            np.log1p,
                            validate=False,
                        ),
                    ),
                    (
                        "scaler",
                        StandardScaler(),
                    ),
                ]),
                ["Amount"],
            ),
            (
                "pca_features",
                StandardScaler(),
                [
                    f"V{i}"
                    for i in range(1, 29)
                ],
            ),
        ],
        remainder="drop",
    )


# --------------------------------------------------
# MODEL TRAINING
# --------------------------------------------------

def train_candidate(X_train, y_train):

    model = Pipeline([
        ("preprocessor", build_preprocessor()),
        (
            "model",
            RandomForestClassifier(
                n_estimators=200,
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=42,
            ),
        ),
    ])

    print("\nTraining Random Forest candidate...")

    model.fit(X_train, y_train)

    return model


# --------------------------------------------------
# EVALUATION
# --------------------------------------------------

def evaluate_candidate(model, X_evaluation, y_evaluation):

    print("\nGenerating evaluation predictions...")

    probabilities = model.predict_proba(
        X_evaluation
    )[:, 1]

    predictions = (
        probabilities >= DECISION_THRESHOLD
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_evaluation,
        predictions,
        labels=[0, 1],
    ).ravel()

    metrics = {
        "model": "random_forest",
        "decision_threshold": DECISION_THRESHOLD,
        "precision": float(
            precision_score(
                y_evaluation,
                predictions,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_evaluation,
                predictions,
                zero_division=0,
            )
        ),
        "f1_score": float(
            f1_score(
                y_evaluation,
                predictions,
                zero_division=0,
            )
        ),
        "pr_auc": float(
            average_precision_score(
                y_evaluation,
                probabilities,
            )
        ),
        "roc_auc": float(
            roc_auc_score(
                y_evaluation,
                probabilities,
            )
        ),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_negatives": int(tn),
        "flagged_transactions": int(tp + fp),
        "evaluation_rows": int(len(y_evaluation)),
        "evaluation_fraud_cases": int(y_evaluation.sum()),
    }

    return metrics


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    ARTIFACTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Loading training and evaluation datasets...")

    training_data = load_dataset(
        TRAIN_PATH,
        "Training",
    )

    evaluation_data = load_dataset(
        EVALUATION_PATH,
        "Evaluation",
    )

    # Reproduce the exact chronological internal split.
    split_index = int(len(training_data) * 0.80)

    internal_training = training_data.iloc[:split_index]

    X_train = internal_training[FEATURE_COLUMNS]
    y_train = internal_training[TARGET_COLUMN]

    X_evaluation = evaluation_data[FEATURE_COLUMNS]
    y_evaluation = evaluation_data[TARGET_COLUMN]

    if y_train.nunique() < 2 or y_evaluation.nunique() < 2:
        raise ValueError(
            "Training and evaluation must both contain "
            "legitimate and fraudulent transactions."
        )

    print("\n========== DATA SUMMARY ==========")
    print(f"Internal training rows: {len(X_train):,}")
    print(f"Evaluation rows: {len(X_evaluation):,}")
    print(
        f"Evaluation fraud cases: "
        f"{int(y_evaluation.sum())}"
    )
    print(f"Decision threshold: {DECISION_THRESHOLD}")

    model = train_candidate(X_train, y_train)

    metrics = evaluate_candidate(
        model,
        X_evaluation,
        y_evaluation,
    )

    joblib.dump(model, MODEL_PATH)

    import json

    with open(METRICS_PATH, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    print("\n========== UNSEEN EVALUATION RESULTS ==========")

    for name, value in metrics.items():
        if isinstance(value, float):
            print(f"{name}: {value:.6f}")
        else:
            print(f"{name}: {value}")

    print("\nCandidate model saved:")
    print(MODEL_PATH)

    print("\nEvaluation metrics saved:")
    print(METRICS_PATH)

    print("\nCandidate evaluation completed successfully.")


if __name__ == "__main__":
    main()
