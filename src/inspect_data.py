from pathlib import Path

import numpy as np
import pandas as pd

# locate and load the raw dataset
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "creditcard.csv"

def load_dataset():
    """Load the original credit card transaction dataset"""
    
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {DATA_PATH}"
        )
    
    df = pd.read_csv(DATA_PATH)
    return df

# Validate the dataset structure

def validate_dataset(df):
    """Check whether the dataset has expected structre."""
    
    required_columns = (
        ["Time", "Amount", "Class"]
        + [f"V{i}" for i in range(1, 29)]
    )
    
    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]
    
    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )
        
    if df.empty:
        raise ValueError("Dataset is empty")
    
    if not set(df["Class"].dropna().unique()).issubset({0, 1}):
        raise ValueError("Unexpected values in target column.")
    
    print("\nDataset structure: VALID")
    

# inspect dataset dimensions and columns

def inspect_structure(df):
    print("\n========== DATASET STRUCTURE ==========")

    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")

    print("\nColumn names:")
    print(df.columns.tolist())

    print("\nData types:")
    print(df.dtypes)

# inspect data quality
def inspect_quality(df):
    print("\n========== DATA QUALITY ==========")

    print("\nMissing values:")
    print(df.isnull().sum()[df.isnull().sum() > 0])

    print(f"\nTotal missing values: {df.isnull().sum().sum()}")

    print(f"\nDuplicate rows: {df.duplicated().sum()}")

    print("\nInfinite numerical values:")
    numeric_df = df.select_dtypes(include=np.number)

    print(
        np.isinf(numeric_df.to_numpy()).sum()
    )
    
# Inpect fraud distribution

def inspect_target(df):
    print("\n=====TARGET DISTRIBUTION=====")
    
    counts = df["Class"].value_counts().sort_index()
    
    
    percentages = (
        df["Class"]
        .value_counts(normalize=True)
        .sort_index() * 100
    )
    
    for label in counts.index:
        category = "Fraudulent" if label == 1 else "legitimate"
        
        print(
            f"{category}: {counts[label]:,} "
            f"({percentages[label]:.4f}%)"
        )
        
# Inspect numerical feature distributions

def inspect_features(df):
    print("\n========== NUMERICAL STATISTICS ==========")

    print(df.describe().T.to_string())

    print("\n========== TRANSACTION AMOUNT ==========")

    print(f"Minimum: {df['Amount'].min():.2f}")
    print(f"Maximum: {df['Amount'].max():.2f}")
    print(f"Mean: {df['Amount'].mean():.2f}")
    print(f"Median: {df['Amount'].median():.2f}")

    print("\n========== TRANSACTION TIME ==========")

    print(f"Minimum: {df['Time'].min()}")
    print(f"Maximum: {df['Time'].max()}")

    print(
        f"Time range: "
        f"{df['Time'].max() - df['Time'].min()} seconds"
    )

# Compare frauduelent and legit transactions

def inspect_class_features(df):
    print("\n====CLASS-WISE COMPARISON=====")
    
    comparison = df.groupby("Class")[
        ["Amount", "Time"]
    ].agg(["mean", "median", "std"])
    
    print(comparison)
    
# Execute the inspection pipeline

def main():
    print("Loading raw transaction dataset")
    
    df = load_dataset()
    
    validate_dataset(df)
    
    inspect_structure(df)
    inspect_quality(df)
    inspect_target(df)
    inspect_features(df)
    inspect_class_features(df)
    
    print("\nDataset inspection completed.")
    
    
if __name__ == "__main__":
    main()