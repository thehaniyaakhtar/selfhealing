# enforce the expected 30 input features
# Allow class as an optional label, nver treat as model input
# Reject missing cols
# Reject null, non numeric infinite values
# Validate amount and time are non negative
# Validate labels, if present are 0 or 1
# Return feature in a consistent column order


from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


# --------------------------------------------------
# MONITORING DATA CONTRACT
# --------------------------------------------------

# The model expects exactly these 30 input features.
FEATURE_COLUMNS = [
    "Time",
    "Amount",
    *[f"V{i}" for i in range(1, 29)],
]

# The actual fraud outcome is optional.
# It must never be passed to the model as an input.
LABEL_COLUMN = "Class"


# --------------------------------------------------
# VALIDATION RESULT
# --------------------------------------------------

@dataclass
class MonitoringBatch:
    features: pd.DataFrame
    labels: pd.Series | None
    source: str
    row_count: int


# --------------------------------------------------
# CUSTOM VALIDATION ERROR
# --------------------------------------------------

class DataValidationError(ValueError):
    """Raised when monitoring data violates its contract."""


# --------------------------------------------------
# DATAFRAME VALIDATION
# --------------------------------------------------

def validate_monitoring_dataframe(
    dataframe: pd.DataFrame,
    dataset_name: str,
) -> MonitoringBatch:

    if not isinstance(dataframe, pd.DataFrame):
        raise DataValidationError(
            f"{dataset_name}: Input must be a pandas DataFrame."
        )

    if dataframe.empty:
        raise DataValidationError(
            f"{dataset_name}: Dataset contains no rows."
        )

    if dataframe.columns.duplicated().any():
        duplicates = dataframe.columns[
            dataframe.columns.duplicated()
        ].tolist()

        raise DataValidationError(
            f"{dataset_name}: Duplicate columns found: {duplicates}"
        )

    actual_columns = set(dataframe.columns)

    expected_columns = set(FEATURE_COLUMNS)
    allowed_columns = expected_columns | {LABEL_COLUMN}

    missing_columns = expected_columns - actual_columns
    unexpected_columns = actual_columns - allowed_columns

    if missing_columns:
        raise DataValidationError(
            f"{dataset_name}: Missing required columns: "
            f"{sorted(missing_columns)}"
        )

    if unexpected_columns:
        raise DataValidationError(
            f"{dataset_name}: Unexpected columns: "
            f"{sorted(unexpected_columns)}"
        )

    # Select features in the exact order expected by the model.
    features = dataframe[FEATURE_COLUMNS].copy()

    # Convert feature columns to numeric values.
    try:
        features = features.apply(
            pd.to_numeric,
            errors="raise",
        )
    except (ValueError, TypeError) as error:
        raise DataValidationError(
            f"{dataset_name}: Non-numeric feature values found."
        ) from error

    # Reject missing values.
    if features.isnull().any().any():
        affected_columns = features.columns[
            features.isnull().any()
        ].tolist()

        raise DataValidationError(
            f"{dataset_name}: Missing values in "
            f"{affected_columns}"
        )

    # Reject infinite values.
    feature_values = features.to_numpy(dtype=np.float64)

    if not np.isfinite(feature_values).all():
        raise DataValidationError(
            f"{dataset_name}: Infinite or invalid numeric values found."
        )

    # Basic domain validation.
    if (features["Amount"] < 0).any():
        raise DataValidationError(
            f"{dataset_name}: Transaction amounts cannot be negative."
        )

    if (features["Time"] < 0).any():
        raise DataValidationError(
            f"{dataset_name}: Time values cannot be negative."
        )

    # Extract labels separately, if available.
    labels = None

    if LABEL_COLUMN in dataframe.columns:
        labels = dataframe[LABEL_COLUMN].copy()

        try:
            labels = pd.to_numeric(
                labels,
                errors="raise",
            )
        except (ValueError, TypeError) as error:
            raise DataValidationError(
                f"{dataset_name}: Labels must be numeric."
            ) from error

        if labels.isnull().any():
            raise DataValidationError(
                f"{dataset_name}: Missing target labels found."
            )

        if not labels.isin([0, 1]).all():
            raise DataValidationError(
                f"{dataset_name}: Labels must contain only 0 or 1."
            )

        labels = labels.astype("int8")

    return MonitoringBatch(
        features=features,
        labels=labels,
        source=dataset_name,
        row_count=len(features),
    )


# --------------------------------------------------
# CSV LOADING
# --------------------------------------------------

def load_monitoring_dataset(
    file_path: str | Path,
    dataset_name: str,
) -> MonitoringBatch:

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"{dataset_name}: File not found: {file_path}"
        )

    try:
        dataframe = pd.read_csv(file_path)
    except (OSError, pd.errors.ParserError) as error:
        raise DataValidationError(
            f"{dataset_name}: Could not read CSV file."
        ) from error

    return validate_monitoring_dataframe(
        dataframe=dataframe,
        dataset_name=dataset_name,
    )


# 2 objects are returned: features and labels
# frift detector should compare input features, not fraud outcome
# fraud lable s useful later when models actual performance has detoriorated
# It should not be accidentally leaked
# seperation also hels implement delayed-label monitoring 