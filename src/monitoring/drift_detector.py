# drft detector engine
# a system that compares each feature's reference distribution against its production dist using 
# Kolmogorov-Smirnov two sample test
# Calculates:
# KS statistic: how different 2 dist are
# P value: how much evidence is there against the dstribution being the same
# Drift status: wheter the feature crosses statitical significance threhols
# Multiple test correction: avoid getting into generating excessive false alarms across 30 features

# Data Drift: for a feature X:
# Preference(X) \ne Pproduction(X)
# aim: detect the evidence of that change

# Use KS test to compare cumulative distibution functions of 2 samples (the prob that they are less than or equ to)
# D = sup_x |Freference x - F production x|
# Take a value on the feature scale
# Calculate the proportion of ference observations at or below the value
# Calculate the equivalent proportion in production
# Measure he largest difference between those proportions

# KS stats D ranges from 0 to 1
# 0: the dist are v simialar
# 1: larger values: the dist are more substantial
# test also returns a p value, how surprisin the observed difference would be under the assumption that both samples come from the same dist
# small p value: evidence against the assumption

# Multiple testing correction:
# 30 features, test every feature at a significance level of 0.05
# testing many features can raise the chance of one of them being a false alarm

# Bonferroni correction:
# alpha adjusted = 0.05 / 29 = 0.001724
# feature will be flagged as stat. drifted when p value is below the threshold
# datasets are split chronologically, time is excluded 

# KS test needs underlying observatons from both datasets
# detector uses:
# original reference observations
# production batch loaded dataset
# json to verify that reference feature schema matches the profile


import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from scipy.stats import ks_2samp

from src.monitoring.data_validator import (
    FEATURE_COLUMNS,
    load_monitoring_dataset,
)
from src.monitoring.production_loader import (
    ProductionMonitoringInput,
    prepare_monitoring_input,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

REFERENCE_DATA_PATH = (
    PROJECT_ROOT / "data" / "reference" / "drift_reference.csv"
)

REPORT_PATH = (
    PROJECT_ROOT / "artifacts" / "monitoring" / "drift_report.json"
)

SIGNIFICANCE_LEVEL = 0.05

# Time is excluded because our datasets were split chronologically.
# It is still measured and reported as a temporal diagnostic.
EXCLUDED_FROM_ALERTS = {"Time"}

TESTED_FEATURES = [
    feature
    for feature in FEATURE_COLUMNS
    if feature not in EXCLUDED_FROM_ALERTS
]

ADJUSTED_SIGNIFICANCE_LEVEL = (
    SIGNIFICANCE_LEVEL / len(TESTED_FEATURES)
)


@dataclass
class FeatureDriftResult:
    feature: str
    monitoring_role: str
    reference_count: int
    production_count: int
    ks_statistic: float
    p_value: float
    significance_threshold: float | None
    drift_detected: bool | None


@dataclass
class DriftDetectionReport:
    generated_at: str
    reference_rows: int
    production_rows: int
    total_features: int
    tested_features: int
    significance_level: float
    correction_method: str
    adjusted_significance_level: float
    drifted_features: list[str]
    temporal_diagnostics: list[str]
    results: list[FeatureDriftResult]


def load_reference_observations():
    """
    Load the original reference observations.

    Summary statistics alone cannot be used for a
    two-sample KS test, so we need the raw observations.
    """

    if not REFERENCE_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Reference dataset not found: {REFERENCE_DATA_PATH}"
        )

    reference_batch = load_monitoring_dataset(
        file_path=REFERENCE_DATA_PATH,
        dataset_name="reference",
    )

    return reference_batch


def validate_monitoring_compatibility(
    reference_profile: dict,
    reference_features,
    production_features,
):
    """
    Ensure all monitoring inputs use the same feature schema.
    """

    profile_features = list(
        reference_profile["features"].keys()
    )

    reference_columns = list(reference_features.columns)
    production_columns = list(production_features.columns)

    if profile_features != FEATURE_COLUMNS:
        raise ValueError(
            "Reference profile feature schema is incompatible."
        )

    if reference_columns != FEATURE_COLUMNS:
        raise ValueError(
            "Reference observation feature schema is incompatible."
        )

    if production_columns != FEATURE_COLUMNS:
        raise ValueError(
            "Production feature schema is incompatible."
        )


def compare_feature_distributions(
    reference_values,
    production_values,
    feature_name: str,
) -> FeatureDriftResult:
    """
    Compare one feature using the two-sample KS test.
    """

    test_result = ks_2samp(
        reference_values,
        production_values,
        alternative="two-sided",
        method="auto",
    )

    is_temporal_diagnostic = (
        feature_name in EXCLUDED_FROM_ALERTS
    )

    if is_temporal_diagnostic:
        threshold = None
        drift_detected = None
        monitoring_role = "temporal_diagnostic"
    else:
        threshold = ADJUSTED_SIGNIFICANCE_LEVEL
        drift_detected = (
            test_result.pvalue < ADJUSTED_SIGNIFICANCE_LEVEL
        )
        monitoring_role = "drift_alert"

    return FeatureDriftResult(
        feature=feature_name,
        monitoring_role=monitoring_role,
        reference_count=len(reference_values),
        production_count=len(production_values),
        ks_statistic=float(test_result.statistic),
        p_value=float(test_result.pvalue),
        significance_threshold=threshold,
        drift_detected=drift_detected,
    )


def detect_drift(
    monitoring_input: ProductionMonitoringInput,
) -> DriftDetectionReport:
    """
    Compare reference and production feature distributions.
    """

    reference_batch = load_reference_observations()

    production_batch = monitoring_input.production_batch

    validate_monitoring_compatibility(
        reference_profile=monitoring_input.reference_profile,
        reference_features=reference_batch.features,
        production_features=production_batch.features,
    )

    results = []

    for feature in FEATURE_COLUMNS:
        reference_values = (
            reference_batch.features[feature].to_numpy()
        )

        production_values = (
            production_batch.features[feature].to_numpy()
        )

        result = compare_feature_distributions(
            reference_values=reference_values,
            production_values=production_values,
            feature_name=feature,
        )

        results.append(result)

    drifted_features = [
        result.feature
        for result in results
        if result.drift_detected is True
    ]

    temporal_diagnostics = [
        result.feature
        for result in results
        if result.monitoring_role == "temporal_diagnostic"
    ]

    return DriftDetectionReport(
        generated_at=datetime.now(timezone.utc).isoformat(),
        reference_rows=reference_batch.row_count,
        production_rows=production_batch.row_count,
        total_features=len(FEATURE_COLUMNS),
        tested_features=len(TESTED_FEATURES),
        significance_level=SIGNIFICANCE_LEVEL,
        correction_method="Bonferroni",
        adjusted_significance_level=ADJUSTED_SIGNIFICANCE_LEVEL,
        drifted_features=drifted_features,
        temporal_diagnostics=temporal_diagnostics,
        results=results,
    )


def save_drift_report(report: DriftDetectionReport):
    """
    Save the complete drift detection report as JSON.
    """

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_data = asdict(report)

    with open(REPORT_PATH, "w", encoding="utf-8") as file:
        json.dump(
            report_data,
            file,
            indent=4,
        )

    print(f"\nDrift report saved to: {REPORT_PATH}")


def print_drift_summary(report: DriftDetectionReport):
    """
    Print a readable summary of the drift detection results.
    """

    print("\n========== DRIFT DETECTION REPORT ==========")

    print(f"Reference observations: {report.reference_rows}")
    print(f"Production observations: {report.production_rows}")
    print(f"Total features: {report.total_features}")
    print(f"Features tested for alerts: {report.tested_features}")

    print(f"\nCorrection method: {report.correction_method}")
    print(f"Original significance level: {report.significance_level}")
    print(
        "Adjusted significance level: "
        f"{report.adjusted_significance_level:.6f}"
    )

    print("\n========== FEATURE RESULTS ==========")

    for result in report.results:
        if result.drift_detected is True:
            status = "DRIFT DETECTED"
        elif result.drift_detected is False:
            status = "NO DRIFT"
        else:
            status = "DIAGNOSTIC ONLY"

        print(
            f"{result.feature:>8} | "
            f"KS: {result.ks_statistic:.6f} | "
            f"p-value: {result.p_value:.6g} | "
            f"{status}"
        )

    print("\n========== SUMMARY ==========")

    print(
        f"Drifted features: {len(report.drifted_features)}"
    )

    if report.drifted_features:
        print(", ".join(report.drifted_features))
    else:
        print("No features crossed the drift threshold.")

    print(
        "Temporal diagnostics: "
        + ", ".join(report.temporal_diagnostics)
    )


def main():
    print("Preparing monitoring inputs...")

    monitoring_input = prepare_monitoring_input()

    print("\nLoading reference observations...")
    reference_batch = load_reference_observations()

    print(
        f"Reference observations loaded: "
        f"{reference_batch.row_count}"
    )

    print("\nRunning statistical drift detection...")

    report = detect_drift(monitoring_input)

    print_drift_summary(report)

    save_drift_report(report)

    print("\nDrift detection completed successfully.")


if __name__ == "__main__":
    main()
