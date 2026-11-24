
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)


# --------------------------------------------------
# 1. Project configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TRAIN_PATH = (
    PROJECT_ROOT / "data" / "reference" / "train.csv"
)

EVALUATION_PATH = (
    PROJECT_ROOT / "data" / "reference" / "evaluation.csv"
)

ARTIFACT_DIR = PROJECT_ROOT / "artifacts"

MODEL_PATH = ARTIFACT_DIR / "baseline_model.joblib"

METRICS_PATH = (
    PROJECT_ROOT / "artifacts" / "baseline_metrics.json"
)

RANDOM_STATE = 42

TARGET_COLUMN = "Class"

PCA_COLUMNS = [f"V{i}" for i in range(1, 29)]

FEATURE_COLUMNS = ["Time", "Amount"] + PCA_COLUMNS


# --------------------------------------------------
# 2. Load datasets
# --------------------------------------------------

def load_data():

    if not TRAIN_PATH.exists():
        raise FileNotFoundError(TRAIN_PATH)

    if not EVALUATION_PATH.exists():
        raise FileNotFoundError(EVALUATION_PATH)

    train_df = pd.read_csv(TRAIN_PATH)

    evaluation_df = pd.read_csv(EVALUATION_PATH)

    return train_df, evaluation_df


# --------------------------------------------------
# 3. Validate input data
# --------------------------------------------------

def validate_data(train_df, evaluation_df):

    for name, df in [
        ("Training", train_df),
        ("Evaluation", evaluation_df)
    ]:

        required = FEATURE_COLUMNS + [TARGET_COLUMN]

        if not set(required).issubset(df.columns):
            raise ValueError(
                f"{name} dataset has missing columns."
            )

        if df.empty:
            raise ValueError(
                f"{name} dataset is empty."
            )

        if not np.isfinite(
            df[FEATURE_COLUMNS].to_numpy()
        ).all():
            raise ValueError(
                f"{name} contains invalid numerical values."
            )

        if not df[TARGET_COLUMN].isin([0, 1]).all():
            raise ValueError(
                f"{name} has invalid target labels."
            )

    print("Input datasets validated.")


# --------------------------------------------------
# 4. Build preprocessing pipeline
# --------------------------------------------------

def build_preprocessor():

    amount_pipeline = Pipeline(
        steps=[
            (
                "log_transform",
                FunctionTransformer(
                    np.log1p,
                    feature_names_out="one-to-one"
                )
            ),
            (
                "scaler",
                StandardScaler()
            )
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("time", StandardScaler(), ["Time"]),
            ("amount", amount_pipeline, ["Amount"]),
            ("pca", StandardScaler(), PCA_COLUMNS)
        ],
        remainder="drop",
        verbose_feature_names_out=False
    )


# --------------------------------------------------
# 5. Create chronological training and validation sets
# --------------------------------------------------

def create_validation_split(train_df):

    train_df = train_df.sort_values(
        "Time",
        kind="stable"
    ).reset_index(drop=True)

    split_index = int(len(train_df) * 0.80)

    internal_train = train_df.iloc[:split_index].copy()

    validation = train_df.iloc[split_index:].copy()

    print("\n========== INTERNAL SPLIT ==========")

    print(f"Internal training rows: {len(internal_train):,}")
    print(f"Validation rows: {len(validation):,}")

    print(
        f"Internal training fraud cases: "
        f"{(internal_train[TARGET_COLUMN] == 1).sum():,}"
    )

    print(
        f"Validation fraud cases: "
        f"{(validation[TARGET_COLUMN] == 1).sum():,}"
    )

    if internal_train[TARGET_COLUMN].nunique() != 2:
        raise ValueError(
            "Internal training data must contain both classes."
        )

    if validation[TARGET_COLUMN].nunique() != 2:
        raise ValueError(
            "Validation data must contain both classes."
        )

    return internal_train, validation


# --------------------------------------------------
# 6. Define candidate algorithms
# --------------------------------------------------

def build_models():

    return {
        "logistic_regression": LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            random_state=RANDOM_STATE
        ),

        "random_forest": RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=RANDOM_STATE
        )
    }


# --------------------------------------------------
# 7. Train and compare candidates
# --------------------------------------------------

def train_and_compare(internal_train, validation):

    X_train = internal_train[FEATURE_COLUMNS]
    y_train = internal_train[TARGET_COLUMN]

    X_validation = validation[FEATURE_COLUMNS]
    y_validation = validation[TARGET_COLUMN]

    preprocessor = build_preprocessor()

    print("\nFitting validation preprocessor...")

    X_train_processed = preprocessor.fit_transform(X_train)

    X_validation_processed = preprocessor.transform(
        X_validation
    )

    models = build_models()

    results = {}

    print("\n========== MODEL COMPARISON ==========")

    for name, model in models.items():

        print(f"\nTraining {name}...")

        model.fit(
            X_train_processed,
            y_train
        )

        probabilities = model.predict_proba(
            X_validation_processed
        )[:, 1]

        predictions = (probabilities >= 0.5).astype(int)

        pr_auc = average_precision_score(
            y_validation,
            probabilities
        )

        results[name] = {
            "model": model,
            "pr_auc": float(pr_auc)
        }

        print(f"PR-AUC: {pr_auc:.5f}")

        print(
            f"Precision: "
            f"{precision_score(y_validation, predictions, zero_division=0):.5f}"
        )

        print(
            f"Recall: "
            f"{recall_score(y_validation, predictions, zero_division=0):.5f}"
        )

        print(
            f"F1-score: "
            f"{f1_score(y_validation, predictions, zero_division=0):.5f}"
        )

    best_name = max(
        results,
        key=lambda name: results[name]["pr_auc"]
    )

    print(f"\nSelected candidate: {best_name}")

    return best_name, results[best_name]["pr_auc"]


# --------------------------------------------------
# 8. Retrain selected model on full training data
# --------------------------------------------------

def train_final_model(best_name, train_df):

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[TARGET_COLUMN]

    preprocessor = build_preprocessor()

    print("\nFitting final preprocessor...")

    X_train_processed = preprocessor.fit_transform(
        X_train
    )

    model = build_models()[best_name]

    print(f"Training final {best_name}...")

    model.fit(
        X_train_processed,
        y_train
    )

    return preprocessor, model


# --------------------------------------------------
# 9. Evaluate final model
# --------------------------------------------------

def evaluate_model(preprocessor, model, evaluation_df):

    X_test = evaluation_df[FEATURE_COLUMNS]
    y_test = evaluation_df[TARGET_COLUMN]

    X_test_processed = preprocessor.transform(X_test)

    probabilities = model.predict_proba(
        X_test_processed
    )[:, 1]

    predictions = (probabilities >= 0.5).astype(int)

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1]
    )

    metrics = {
        "precision": float(
            precision_score(y_test, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y_test, predictions, zero_division=0)
        ),
        "f1_score": float(
            f1_score(y_test, predictions, zero_division=0)
        ),
        "pr_auc": float(
            average_precision_score(y_test, probabilities)
        ),
        "roc_auc": float(
            roc_auc_score(y_test, probabilities)
        ),
        "confusion_matrix": matrix.tolist()
    }

    print("\n========== FINAL EVALUATION ==========")

    for name, value in metrics.items():
        print(f"{name}: {value}")

    return metrics


# --------------------------------------------------
# 10. Save model and evaluation artifacts
# --------------------------------------------------

def save_artifacts(
    best_name,
    validation_pr_auc,
    preprocessor,
    model,
    metrics
):

    ARTIFACT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    model_bundle = {
        "model": model,
        "preprocessor": preprocessor,
        "model_name": best_name,
        "feature_columns": FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN
    }

    joblib.dump(
        model_bundle,
        MODEL_PATH
    )

    report = {
        "selected_model": best_name,
        "validation_pr_auc": validation_pr_auc,
        "evaluation_metrics": metrics
    }

    with open(METRICS_PATH, "w") as file:
        json.dump(report, file, indent=4)

    print(f"\nModel saved: {MODEL_PATH}")
    print(f"Metrics saved: {METRICS_PATH}")


# --------------------------------------------------
# 11. Execute baseline training workflow
# --------------------------------------------------

def main():

    print("Loading datasets...")

    train_df, evaluation_df = load_data()

    validate_data(train_df, evaluation_df)

    internal_train, validation = create_validation_split(
        train_df
    )

    best_name, validation_pr_auc = train_and_compare(
        internal_train,
        validation
    )

    preprocessor, model = train_final_model(
        best_name,
        train_df
    )

    metrics = evaluate_model(
        preprocessor,
        model,
        evaluation_df
    )

    save_artifacts(
        best_name,
        validation_pr_auc,
        preprocessor,
        model,
        metrics
    )

    print("\nBaseline training completed successfully.")


if __name__ == "__main__":
    main()
