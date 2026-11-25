# monitoring report and alerting layer
# platform can detect a stat. drift, but output is primarily a collection of test results
# results need to be interpretable

# n alert is not a retraining instruction.

# produces 2 outputs:
# reads existing drift_report.json
# monitoring_report.json: structured report containing run summary, affected features and stat. evidence and recommended action
# readable terminal summary


import json
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DRIFT_REPORT_PATH = (
    PROJECT_ROOT / "artifacts" / "monitoring" / "drift_report.json"
)

MONITORING_REPORT_PATH = (
    PROJECT_ROOT / "artifacts" / "monitoring" / "monitoring_report.json"
)

TOP_FEATURE_COUNT = 5


class MonitoringReportError(Exception):
    """Raised when monitoring report data is missing or invalid."""


def load_drift_report(file_path: Path) -> dict:
    """Load and validate the statistical drift report."""

    if not file_path.exists():
        raise FileNotFoundError(
            f"Drift report not found: {file_path}. "
            "Run the drift detector first."
        )

    try:
        with open(file_path, "r", encoding="utf-8") as file:
            report = json.load(file)
    except (OSError, json.JSONDecodeError) as error:
        raise MonitoringReportError(
            f"Could not read drift report: {error}"
        ) from error

    required_fields = {
        "generated_at",
        "reference_rows",
        "production_rows",
        "total_features",
        "tested_features",
        "significance_level",
        "correction_method",
        "adjusted_significance_level",
        "drifted_features",
        "temporal_diagnostics",
        "results",
    }

    missing_fields = required_fields - set(report.keys())

    if missing_fields:
        raise MonitoringReportError(
            f"Drift report is missing fields: {sorted(missing_fields)}"
        )

    if not isinstance(report["results"], list):
        raise MonitoringReportError(
            "Drift report results must be a list."
        )

    if not report["results"]:
        raise MonitoringReportError(
            "Drift report contains no feature results."
        )

    return report


def validate_feature_results(results: list) -> None:
    """Validate the individual feature test results."""

    required_fields = {
        "feature",
        "monitoring_role",
        "reference_count",
        "production_count",
        "ks_statistic",
        "p_value",
        "significance_threshold",
        "drift_detected",
    }

    for result in results:
        if not isinstance(result, dict):
            raise MonitoringReportError(
                "Each feature result must be a JSON object."
            )

        missing_fields = required_fields - set(result.keys())

        if missing_fields:
            raise MonitoringReportError(
                f"Feature result is missing fields: "
                f"{sorted(missing_fields)}"
            )

        if result["monitoring_role"] == "drift_alert":
            if not isinstance(result["drift_detected"], bool):
                raise MonitoringReportError(
                    f"Invalid drift status for {result['feature']}."
                )

        elif result["monitoring_role"] == "temporal_diagnostic":
            if result["drift_detected"] is not None:
                raise MonitoringReportError(
                    f"Temporal diagnostic {result['feature']} "
                    "must not have an alert decision."
                )

        else:
            raise MonitoringReportError(
                f"Unknown monitoring role for {result['feature']}."
            )


def build_monitoring_report(drift_report: dict) -> dict:
    """
    Convert statistical test results into an engineer-friendly
    monitoring report and alert.
    """

    results = drift_report["results"]

    validate_feature_results(results)

    alert_results = [
        result
        for result in results
        if result["monitoring_role"] == "drift_alert"
    ]

    temporal_results = [
        result
        for result in results
        if result["monitoring_role"] == "temporal_diagnostic"
    ]

    drifted_results = [
        result
        for result in alert_results
        if result["drift_detected"] is True
    ]

    unaffected_results = [
        result
        for result in alert_results
        if result["drift_detected"] is False
    ]

    # Rank features by KS statistic, not by p-value.
    # This prioritizes the largest observed distribution differences.
    ranked_drift = sorted(
        drifted_results,
        key=lambda result: result["ks_statistic"],
        reverse=True,
    )

    if drifted_results:
        monitoring_status = "DRIFT_DETECTED"

        recommended_action = (
            "Investigate the affected feature distributions and "
            "assess model performance when actual labels are available. "
            "Do not automatically retrain or promote a replacement model "
            "based on statistical drift alone."
        )
    else:
        monitoring_status = "NO_DRIFT"

        recommended_action = (
            "No alert-eligible features crossed the configured "
            "statistical significance threshold. Continue routine "
            "monitoring and evaluate model performance when labels "
            "are available."
        )

    # Feature drift is not equivalent to model performance degradation.
    performance_status = "NOT_ASSESSED"

    report = {
        "report_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),

        "monitoring_status": monitoring_status,

        "monitoring_window": {
            "reference_rows": drift_report["reference_rows"],
            "production_rows": drift_report["production_rows"],
            "reference_report_timestamp": drift_report["generated_at"],
        },

        "statistical_configuration": {
            "correction_method": drift_report["correction_method"],
            "significance_level": drift_report["significance_level"],
            "adjusted_significance_level": (
                drift_report["adjusted_significance_level"]
            ),
            "tested_features": drift_report["tested_features"],
            "total_features": drift_report["total_features"],
        },

        "summary": {
            "features_tested": len(alert_results),
            "features_drifted": len(drifted_results),
            "features_without_detected_drift": len(unaffected_results),
            "temporal_diagnostics": len(temporal_results),
            "drifted_feature_names": [
                result["feature"]
                for result in drifted_results
            ],
        },

        "highest_drift_features": [
            {
                "feature": result["feature"],
                "ks_statistic": result["ks_statistic"],
                "p_value": result["p_value"],
                "significance_threshold": (
                    result["significance_threshold"]
                ),
            }
            for result in ranked_drift[:TOP_FEATURE_COUNT]
        ],

        "temporal_diagnostics": [
            {
                "feature": result["feature"],
                "ks_statistic": result["ks_statistic"],
                "p_value": result["p_value"],
                "interpretation": (
                    "Reported for temporal analysis only; "
                    "not included in drift alert decisions."
                ),
            }
            for result in temporal_results
        ],

        "model_performance": {
            "status": performance_status,
            "reason": (
                "This report evaluates feature distribution drift. "
                "Model performance degradation has not been assessed "
                "by this component."
            ),
        },

        "alert": {
            "status": monitoring_status,
            "triggered": bool(drifted_results),
            "affected_feature_count": len(drifted_results),
            "recommended_action": recommended_action,
        },

        "feature_results": results,
    }

    return report


def save_monitoring_report(report: dict) -> None:
    """Persist the monitoring report as JSON."""

    MONITORING_REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        MONITORING_REPORT_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            indent=4,
        )

    print(
        f"\nMonitoring report saved to: "
        f"{MONITORING_REPORT_PATH}"
    )


def print_monitoring_summary(report: dict) -> None:
    """Print the monitoring status and important findings."""

    summary = report["summary"]
    alert = report["alert"]

    print("\n========== MONITORING SUMMARY ==========")

    print(f"Status: {report['monitoring_status']}")
    print(f"Reference rows: {report['monitoring_window']['reference_rows']}")
    print(f"Production rows: {report['monitoring_window']['production_rows']}")

    print("\n========== FEATURE SUMMARY ==========")

    print(f"Features tested: {summary['features_tested']}")
    print(f"Features drifted: {summary['features_drifted']}")
    print(
        "Features without detected drift: "
        f"{summary['features_without_detected_drift']}"
    )
    print(
        f"Temporal diagnostics: "
        f"{summary['temporal_diagnostics']}"
    )

    print("\n========== HIGHEST DRIFT FEATURES ==========")

    if report["highest_drift_features"]:
        for result in report["highest_drift_features"]:
            print(
                f"{result['feature']:>8} | "
                f"KS: {result['ks_statistic']:.6f} | "
                f"p-value: {result['p_value']:.6g}"
            )
    else:
        print("No drifted features to rank.")

    print("\n========== MODEL PERFORMANCE ==========")

    print(f"Status: {report['model_performance']['status']}")
    print(report["model_performance"]["reason"])

    print("\n========== ALERT ==========")

    print(f"Alert status: {alert['status']}")
    print(f"Alert triggered: {alert['triggered']}")
    print(
        f"Affected features: {alert['affected_feature_count']}"
    )
    print(f"Recommended action: {alert['recommended_action']}")


def main():
    print("Loading statistical drift report...")

    drift_report = load_drift_report(DRIFT_REPORT_PATH)

    print("Drift report loaded successfully.")

    print("\nBuilding monitoring report...")

    monitoring_report = build_monitoring_report(drift_report)

    print_monitoring_summary(monitoring_report)

    save_monitoring_report(monitoring_report)

    print("\nMonitoring report generation completed successfully.")


if __name__ == "__main__":
    main()
