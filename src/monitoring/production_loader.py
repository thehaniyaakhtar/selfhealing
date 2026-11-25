# load production data using existing vaalidator
# load the saved reference profile
# verify that the profile and incoming batch use the same feature schema
# produce a monitoring batch with source and timestamp metadata
# keep labels seperate from input features

# component that brings production data into monitoring pipeline and checks compatibility with reference baseline
# monitoring system must never compare incompatible datasets
# if across V1 - V28, V17 is missing it should not compare, instead discard the entire dataset


import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from src.monitoring.data_validator import (
    FEATURE_COLUMNS,
    MonitoringBatch,
    load_monitoring_dataset,
    DataValidationError,
)


# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PRODUCTION_DATA_PATH = (
    PROJECT_ROOT / "data" / "production" / "production.csv"
)

REFERENCE_PROFILE_PATH = (
    PROJECT_ROOT / "artifacts" / "monitoring" / "reference_profile.json"
)


# --------------------------------------------------
# MONITORING INPUT
# --------------------------------------------------

@dataclass
class ProductionMonitoringInput:
    reference_profile: dict
    production_batch: MonitoringBatch
    ingestion_timestamp: str


# --------------------------------------------------
# LOAD REFERENCE PROFILE
# --------------------------------------------------

def load_reference_profile(file_path: Path) -> dict:

    if not file_path.exists():
        raise FileNotFoundError(
            f"Reference profile not found: {file_path}"
        )

    try:
        with file_path.open("r", encoding="utf-8") as file:
            profile = json.load(file)

    except (OSError, json.JSONDecodeError) as error:
        raise DataValidationError(
            "Could not read the reference profile."
        ) from error

    # Verify the profile's basic structure.
    required_keys = {
        "profile_version",
        "created_at",
        "source",
        "row_count",
        "feature_count",
        "features",
    }

    missing_keys = required_keys - set(profile.keys())

    if missing_keys:
        raise DataValidationError(
            f"Reference profile is missing fields: "
            f"{sorted(missing_keys)}"
        )

    # Verify the feature count.
    if profile["feature_count"] != len(FEATURE_COLUMNS):
        raise DataValidationError(
            "Reference profile has an unexpected feature count."
        )

    # Verify feature names and their order.
    profile_features = list(profile["features"].keys())

    if profile_features != FEATURE_COLUMNS:
        raise DataValidationError(
            "Reference profile feature schema does not match "
            "the monitoring feature schema."
        )

    # Verify that each feature has the expected statistics.
    required_statistics = {
        "count",
        "missing_count",
        "mean",
        "std",
        "min",
        "max",
        "median",
        "q25",
        "q75",
    }

    for feature, statistics in profile["features"].items():

        missing_statistics = (
            required_statistics - set(statistics.keys())
        )

        if missing_statistics:
            raise DataValidationError(
                f"Reference profile for {feature} is missing "
                f"statistics: {sorted(missing_statistics)}"
            )

    return profile


# --------------------------------------------------
# LOAD PRODUCTION DATA
# --------------------------------------------------

def load_production_batch(
    file_path: Path,
) -> MonitoringBatch:

    return load_monitoring_dataset(
        file_path=file_path,
        dataset_name="production",
    )


# --------------------------------------------------
# BUILD MONITORING INPUT
# --------------------------------------------------

def prepare_monitoring_input(
    reference_profile_path: Path = REFERENCE_PROFILE_PATH,
    production_data_path: Path = PRODUCTION_DATA_PATH,
) -> ProductionMonitoringInput:

    print("Loading reference profile...")

    reference_profile = load_reference_profile(
        reference_profile_path
    )

    print("Reference profile loaded.")
    print(
        f"Reference observations: "
        f"{reference_profile['row_count']}"
    )

    print("\nLoading production data...")

    production_batch = load_production_batch(
        production_data_path
    )

    print("Production data loaded.")
    print(
        f"Production observations: "
        f"{production_batch.row_count}"
    )

    # Confirm that both datasets have identical feature schemas.
    reference_features = list(
        reference_profile["features"].keys()
    )

    production_features = list(
        production_batch.features.columns
    )

    if reference_features != production_features:
        raise DataValidationError(
            "Reference and production feature schemas differ."
        )

    print("\nFeature schema compatibility verified.")

    return ProductionMonitoringInput(
        reference_profile=reference_profile,
        production_batch=production_batch,
        ingestion_timestamp=datetime.now(
            timezone.utc
        ).isoformat(),
    )


# --------------------------------------------------
# MAIN EXECUTION
# --------------------------------------------------

def main():

    monitoring_input = prepare_monitoring_input()

    reference_profile = monitoring_input.reference_profile
    production_batch = monitoring_input.production_batch

    print("\n========== MONITORING INPUT ==========")

    print(
        f"Reference rows: "
        f"{reference_profile['row_count']}"
    )

    print(
        f"Production rows: "
        f"{production_batch.row_count}"
    )

    print(
        f"Features available: "
        f"{len(production_batch.features.columns)}"
    )

    print(
        f"Production labels available: "
        f"{production_batch.labels is not None}"
    )

    print(
        f"Ingestion timestamp: "
        f"{monitoring_input.ingestion_timestamp}"
    )

    print("\nMonitoring input prepared successfully.")


if __name__ == "__main__":
    main()

# it must return a structured object containing both datasets and ingestion metadata
# start with historiccal CSV since API prediction log ingestion will be seperate adapter since logs have diferent structure and may contain delayed/ unavailable labels


# 2 main functions
# load_reference_profile(): loads saved JSON baseline, checks all 30 expected features and stat. summaries
# prepare_monitoring_input(): coordinates the loading of both datasets and verifies compatibility

# we thus have 2 seperate inputs
# reference: histrical baseline that descibes what the models input data looks like NORMALLY
# Produciton: data currently being monitored, compared against the baseline
