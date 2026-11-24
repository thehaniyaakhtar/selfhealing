from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_PATH = PROJECT_ROOT / "data" / "raw" / "creditcard.csv"

TRAIN_PATH = PROJECT_ROOT / "data" / "reference" / "train.csv"
REFERENCE_PATH = PROJECT_ROOT / "data" / "reference" / "drift_reference.csv"
EVALUATION_PATH = PROJECT_ROOT / "data" / "reference" / "evaluation.csv"
PRODUCTION_PATH = PROJECT_ROOT / "data" / "production" / "production.csv"

# Load and validate the original data

def load_data():

    if not RAW_PATH.exists():
        raise FileNotFoundError(
            f"Raw dataset not found: {RAW_PATH}"
        )

    df = pd.read_csv(RAW_PATH)

    required_columns = (
        ["Time", "Amount", "Class"]
        + [f"V{i}" for i in range(1, 29)]
    )

    if not set(required_columns).issubset(df.columns):
        raise ValueError("Dataset does not contain all required columns.")

    if df["Class"].isnull().any():
        raise ValueError("Target column contains missing values.")

    if df["Time"].isnull().any():
        raise ValueError("Time column contains missing values.")

    if not df["Class"].isin([0, 1]).all():
        raise ValueError("Invalid target labels.")

    return df


# Soort observations chronologically
def sort_data(df):
    
    df = df.sort_values(
        by="Time",
        kind="stable"
    ).reset_index(drop = True)
    
    return df

# Split observations into seperate datasets

def split_data(df):
    total_rows = len(df)
    
    train_end = int(total_rows * 0.60)
    reference_end = int(total_rows * 0.70)
    evaluation_end = int(total_rows * 0.85)
    
    train = df.iloc[:train_end].copy()    
    
    reference = df.iloc[
        train_end:reference_end
    ].copy()
    
    evaluation = df.iloc[
        reference_end:evaluation_end
    ].copy()
    
    production = df.iloc[
        evaluation_end:
    ].copy()
    
    return train, reference, evaluation, production

# Validate the generated datasets

def validated_splits(datasets):
    
    names = [
        "Training",
        "Reference",
        "Evaluation"
        "Production"
    ]
    
    print("\n====SPLIT VALIDATION====")
    
    previous_max_time = None
    
    for name, dataset in zip(names, datasets):
        
        if dataset.empty:
            raise ValueError(f"{name} dataset is empty.")
        
        current_min_time = dataset["Time"].min()
        current_max_time = dataset["Time"].max()
        
        if previous_max_time is not None:
            if current_min_time > current_max_time:
                raise ValueError(
                    f"Chronological ordering failed at {name}."
                )
            
        previous_max_time = current_max_time
        
        
        print(f"\n{name} dataset")
        print(f"Rows: {len(dataset):,}")
        print(f"Time range: {current_min_time} - {current_max_time}")

        print("\nClass distribution:")
        print(dataset["Class"].value_counts().sort_index())

    print("\nAll dataset splits validated.")
    
# Save generated datasets
def save_datasets(datasets):

    paths = [
        TRAIN_PATH,
        REFERENCE_PATH,
        EVALUATION_PATH,
        PRODUCTION_PATH
    ]

    for dataset, path in zip(datasets, paths):

        path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        dataset.to_csv(
            path,
            index=False
        )

        print(f"Saved: {path}")

# Execute preparation pipeline

def main():
    print("Loading original dataset...")
    df = load_data()
    print(f"Original observations: {len(df):,}")
    df = sort_data(df)
    datasets = split_data(df)
    validated_splits(datasets)
    save_datasets(datasets)
    print("\nData preparation completed successfully")
    
if __name__ == "__main__":
    main()