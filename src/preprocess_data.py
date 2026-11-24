
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline


# 1. Project paths and configuration

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_ROOT = PROJECT_ROOT / "data"

INPUT_PATHS = {
    "train": DATA_ROOT / "reference" / "train.csv",
    "reference": DATA_ROOT / "reference" / "drift_reference.csv",
    "evaluation": DATA_ROOT / "reference" / "evaluation.csv",
    "production": DATA_ROOT / "production" / "production.csv"
}

OUTPUT_PATHS = {
    "train": DATA_ROOT / "reference" / "processed" / "train.csv",
    "reference": DATA_ROOT / "reference" / "processed" / "drift_reference.csv",
    "evaluation": DATA_ROOT / "reference" / "processed" / "evaluation.csv",
    "production": DATA_ROOT / "production" / "processed" / "production.csv"
}

ARTIFACT_PATH = (
    PROJECT_ROOT / "artifacts" / "preprocessor.joblib"
)

TARGET_COLUMN = "Class"
PCA_COLUMNS = [f"V{i}" for i in range(1, 29)]
FEATURE_COLUMNS = ["Time", "Amount"] + PCA_COLUMNS


# 2. Load prepared datasets

def load_datasets():

    datasets = {}

    for name, path in INPUT_PATHS.items():

        if not path.exists():
            raise FileNotFoundError(
                f"{name} dataset not found: {path}"
            )

        datasets[name] = pd.read_csv(path)

    return datasets

# 3. Validate datasets

def validate_datasets(datasets):
    for name, df in datasets.items():
        required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]

        missing = [
            column for column in required_columns
            if column not in df.columns
        ]

        if missing:
            raise ValueError(
                f"{name} dataset is missing columns: {missing}"
            )

        if df.empty:
            raise ValueError(
                f"{name} dataset is empty."
            )

        features = df[FEATURE_COLUMNS]

        if not np.isfinite(features.to_numpy()).all():
            raise ValueError(
                f"{name} contains missing or infinite feature values."
            )

        if (df["Amount"] < 0).any():
            raise ValueError(
                f"{name} contains negative transaction amounts."
            )

        if not df[TARGET_COLUMN].isin([0, 1]).all():
            raise ValueError(
                f"{name} contains invalid target labels."
            )

    print("All datasets passed validation.")


# 4. Build preprocessing pipeline

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

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "time",
                StandardScaler(),
                ["Time"]
            ),
            (
                "amount",
                amount_pipeline,
                ["Amount"]
            ),
            (
                "pca",
                StandardScaler(),
                PCA_COLUMNS
            )
        ],
        remainder="drop",
        verbose_feature_names_out=False
    )

    return preprocessor


# 5. Separate features and target

def separate_features_target(df):
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()
    return X, y


# 6. Fit on training data and transform all datasets

def preprocess_datasets(datasets):
    preprocessor = build_preprocessor()
    X_train, y_train = separate_features_target(
        datasets["train"]
    )

    print("\nFitting preprocessor on training data...")

    X_train_processed = preprocessor.fit_transform(
        X_train
    )

    processed_datasets = {}

    processed_datasets["train"] = (
        pd.DataFrame(
            X_train_processed,
            columns=preprocessor.get_feature_names_out()
        )
        .assign(Class=y_train.to_numpy())
    )

    for name in ["reference", "evaluation", "production"]:

        X, y = separate_features_target(
            datasets[name]
        )

        print(f"Transforming {name} data...")
        X_processed = preprocessor.transform(X)

        processed_datasets[name] = (
            pd.DataFrame(
                X_processed,
                columns=preprocessor.get_feature_names_out()
            )
            .assign(Class=y.to_numpy())
        )

    return preprocessor, processed_datasets


# 7. Save preprocessor and transformed datasets

def save_artifacts(preprocessor, processed_datasets):

    ARTIFACT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        preprocessor,
        ARTIFACT_PATH
    )

    print(f"\nSaved preprocessor: {ARTIFACT_PATH}")

    for name, df in processed_datasets.items():
        output_path = OUTPUT_PATHS[name]

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        df.to_csv(
            output_path,
            index=False
        )

        print(f"Saved processed {name}: {output_path}")


# 8. Display preprocessing summary

def display_summary(processed_datasets):

    print("\n========== PREPROCESSING SUMMARY ==========")
    for name, df in processed_datasets.items():

        print(f"\n{name.upper()}")

        print(f"Rows: {len(df):,}")

        print(f"Features: {len(df.columns) - 1}")

        print(
            f"Fraud cases: {(df['Class'] == 1).sum():,}"
        )

        print(
            f"Legitimate cases: {(df['Class'] == 0).sum():,}"
        )

    print("\nPreprocessing completed successfully.")


# 9. Execute preprocessing pipeline

def main():

    print("Loading prepared datasets...")

    datasets = load_datasets()

    validate_datasets(datasets)

    preprocessor, processed_datasets = (
        preprocess_datasets(datasets)
    )

    save_artifacts(
        preprocessor,
        processed_datasets
    )

    display_summary(processed_datasets)


if __name__ == "__main__":
    main()