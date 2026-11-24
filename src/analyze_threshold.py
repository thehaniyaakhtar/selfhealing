
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler


# --------------------------------------------------
# PROJECT PATHS AND CONFIGURATION
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TRAIN_PATH = PROJECT_ROOT / "data" / "reference" / "train.csv"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
OUTPUT_PATH = ARTIFACTS_DIR / "threshold_analysis.csv"

TARGET_COLUMN = "Class"

FEATURE_COLUMNS = [
    "Time",
    "Amount",
    *[f"V{i}" for i in range(1, 29)],
]

# Thresholds to compare.
THRESHOLDS = [
    0.05,
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
    0.95,
]


# --------------------------------------------------
# LOAD AND VALIDATE DATA
# --------------------------------------------------

def load_training_data():
    print("Loading training dataset...")

    if not TRAIN_PATH.exists():
        raise FileNotFoundError(
            f"Training dataset not found: {TRAIN_PATH}"
        )

    data = pd.read_csv(TRAIN_PATH)

    required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]

    missing_columns = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    if data.empty:
        raise ValueError("Training dataset is empty.")

    if data[required_columns].isnull().any().any():
        raise ValueError("Dataset contains missing values.")

    if not set(data[TARGET_COLUMN].unique()).issubset({0, 1}):
        raise ValueError("Target must contain only 0 and 1.")

    data = data.sort_values(
        "Time",
        kind="mergesort"
    ).reset_index(drop=True)

    print(f"Total rows: {len(data):,}")
    print(
        f"Fraud cases: "
        f"{int(data[TARGET_COLUMN].sum()):,}"
    )

    return data


# --------------------------------------------------
# BUILD PREPROCESSING PIPELINE
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
                Pipeline(
                    steps=[
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
                    ]
                ),
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
# CHRONOLOGICAL INTERNAL SPLIT
# --------------------------------------------------

def create_internal_split(data):

    split_index = int(len(data) * 0.80)

    train_data = data.iloc[:split_index].copy()
    validation_data = data.iloc[split_index:].copy()

    X_train = train_data[FEATURE_COLUMNS]
    y_train = train_data[TARGET_COLUMN]

    X_validation = validation_data[FEATURE_COLUMNS]
    y_validation = validation_data[TARGET_COLUMN]

    print("\n========== INTERNAL SPLIT ==========")
    print(f"Training rows: {len(X_train):,}")
    print(f"Validation rows: {len(X_validation):,}")
    print(f"Validation fraud cases: {int(y_validation.sum())}")

    if y_train.nunique() < 2 or y_validation.nunique() < 2:
        raise ValueError(
            "Both internal training and validation "
            "must contain both target classes."
        )

    return X_train, y_train, X_validation, y_validation


# --------------------------------------------------
# TRAIN LOGISTIC REGRESSION
# --------------------------------------------------

def train_model(X_train, y_train, X_validation):

    print("\nFitting preprocessor...")

    preprocessor = build_preprocessor()

    X_train_processed = preprocessor.fit_transform(X_train)
    X_validation_processed = preprocessor.transform(X_validation)

    print("Training Logistic Regression...")

    model = LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        random_state=42,
    )

    model.fit(X_train_processed, y_train)

    probabilities = model.predict_proba(
        X_validation_processed
    )[:, 1]

    return probabilities


# --------------------------------------------------
# EVALUATE DIFFERENT THRESHOLDS
# --------------------------------------------------

def evaluate_thresholds(y_true, probabilities):

    results = []

    for threshold in THRESHOLDS:

        predictions = (
            probabilities >= threshold
        ).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y_true,
            predictions,
            labels=[0, 1],
        ).ravel()

        results.append(
            {
                "threshold": threshold,
                "precision": precision_score(
                    y_true,
                    predictions,
                    zero_division=0,
                ),
                "recall": recall_score(
                    y_true,
                    predictions,
                    zero_division=0,
                ),
                "f1_score": f1_score(
                    y_true,
                    predictions,
                    zero_division=0,
                ),
                "true_positives": int(tp),
                "false_positives": int(fp),
                "true_negatives": int(tn),
                "false_negatives": int(fn),
                "flagged_transactions": int(tp + fp),
            }
        )

    return pd.DataFrame(results)


# --------------------------------------------------
# MAIN EXECUTION
# --------------------------------------------------

def main():

    ARTIFACTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    data = load_training_data()

    (
        X_train,
        y_train,
        X_validation,
        y_validation,
    ) = create_internal_split(data)

    probabilities = train_model(
        X_train,
        y_train,
        X_validation,
    )

    results = evaluate_thresholds(
        y_validation,
        probabilities,
    )

    results.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print("\n========== THRESHOLD COMPARISON ==========")

    display_columns = [
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

    print("\nResults saved to:")
    print(OUTPUT_PATH)

    print("\nThreshold analysis completed successfully.")


if __name__ == "__main__":
    main()
