import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline

# Project paths and configuration
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

# Load prepared datasets
def load_datasets():

    datasets = {}

    for name, path in INPUT_PATHS.items():

        if not path.exists():
            raise FileNotFoundError(
                f"{name} dataset not found: {path}"
            )

        datasets[name] = pd.read_csv(path)

    return datasets

# Validate Datasetss

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
                f"{name} dataset is empty"
            )
        
        features = df[FEATURE_COLUMNS]
        
        if not np.isinfinite(features.to_numpy()).all():
            raise ValueError(
                f"{name} contains missing or infinite feature values."
            )
            
        if (df["Amount"] < 0).any():
            raise ValueError(
                f"{name} contains negative transaction amounts"
            )
            
        
        if not df[TARGET_COLUMN].isin([0, 1]).all():
            raise ValueError(
                f"{name} contaons invalid target labels"
            )
            
    print("All datasets passed validation")
    
# Build preprocessing pipeline

def build_preprocessor():
    amount_pipeline = Pipeline(
        steps = [
            (
                "log_transform",
                FunctionTransformer(
                    np.loglp,,
                    feature_names_out = "one-to-one"
                )
            ),
            (
                "scaler"
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
                "amount",
                StandardScaler(),
                ["Amount"]
                
        ]
    )