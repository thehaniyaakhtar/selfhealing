# for each feature this calculates
# Mean and standard deviation
# Minimum and maximum
# Median
# 25th and 75th percentiles
# Number of observations
# Missing-value count

# save the profile as JSON artifact so future monitoring runs can use the sme fixed reference instead of recalculating the baseline from changing production data

# what does normal data look like for the model
# reference profile is a statistical fingerprint of the dataset the model was trained around
# new transactions arrive, the statistical fingerprint is compared against the original one

# when the avg trans amount remains unchanged, but the number of unusually large transactions increases
# the mean alone does not tell us enough
# thus percentiles and std dev are needed for additional info

# the number of fraud cases and fraud rate are recorded seperately, stored as metadata not as input feature stats

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.monitoring.data_validator import (
    FEATURE_COLUMNS,
    load_monitoring_dataset,
)


# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

REFERENCE_DATA_PATH = (
    PROJECT_ROOT / "data" / "reference" / "drift_reference.csv"
)

PROFILE_OUTPUT_PATH = (
    PROJECT_ROOT / "artifacts" / "monitoring" / "reference_profile.json"
)


# --------------------------------------------------
# PROFILE GENERATION
# --------------------------------------------------

def build_reference_profile(
    dataframe: pd.DataFrame,
    dataset_name: str,
) -> dict:

    profile = {
        "profile_version": "1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": dataset_name,
        "row_count": int(len(dataframe)),
        "feature_count": len(FEATURE_COLUMNS),
        "features": {},
    }

    for feature in FEATURE_COLUMNS:

        values = dataframe[feature]

        profile["features"][feature] = {
            "count": int(values.count()),
            "missing_count": int(values.isna().sum()),
            "mean": float(values.mean()),
            "std": float(values.std()),
            "min": float(values.min()),
            "max": float(values.max()),
            "median": float(values.median()),
            "q25": float(values.quantile(0.25)),
            "q75": float(values.quantile(0.75)),
        }

    return profile


# --------------------------------------------------
# LABEL METADATA
# --------------------------------------------------

def add_label_metadata(
    profile: dict,
    labels: pd.Series | None,
) -> dict:

    if labels is None:
        profile["label_metadata"] = {
            "available": False,
        }
        return profile

    label_counts = labels.value_counts().to_dict()

    fraud_count = int(label_counts.get(1, 0))
    legitimate_count = int(label_counts.get(0, 0))
    total_count = len(labels)

    profile["label_metadata"] = {
        "available": True,
        "fraud_count": fraud_count,
        "legitimate_count": legitimate_count,
        "fraud_rate": float(fraud_count / total_count),
    }

    return profile


# --------------------------------------------------
# SAVE PROFILE
# --------------------------------------------------

def save_reference_profile(
    profile: dict,
    output_path: Path,
) -> None:

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            profile,
            file,
            indent=4,
            allow_nan=False,
        )


# --------------------------------------------------
# MAIN EXECUTION
# --------------------------------------------------

def main():

    print("Loading reference dataset...")

    batch = load_monitoring_dataset(
        file_path=REFERENCE_DATA_PATH,
        dataset_name="reference",
    )

    print(f"Reference rows: {batch.row_count}")
    print(f"Reference features: {len(batch.features.columns)}")

    print("\nBuilding statistical profile...")

    profile = build_reference_profile(
        dataframe=batch.features,
        dataset_name=batch.source,
    )

    profile = add_label_metadata(
        profile=profile,
        labels=batch.labels,
    )

    print("\nSaving reference profile...")

    save_reference_profile(
        profile=profile,
        output_path=PROFILE_OUTPUT_PATH,
    )

    print("\n========== REFERENCE PROFILE ==========")
    print(f"Rows: {profile['row_count']}")
    print(f"Features: {profile['feature_count']}")
    print(f"Fraud cases: {profile['label_metadata']['fraud_count']}")
    print(f"Legitimate cases: {profile['label_metadata']['legitimate_count']}")
    print(
        f"Fraud rate: "
        f"{profile['label_metadata']['fraud_rate']:.6%}"
    )

    print("\nExample feature statistics: Amount")

    print(
        json.dumps(
            profile["features"]["Amount"],
            indent=4,
        )
    )

    print(f"\nProfile saved to: {PROFILE_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
    
# calc stats independently for every feature
# keep fraud labels seperate
# save a fixed snapshot

# observations
# reference dataset has fraud rate of 0.0843%
# original dataset has 0.172%
# fraud is rare, chronological splitting can produce diferent class proportions across time periods