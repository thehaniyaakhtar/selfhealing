# What is a promotion policy
# 2 models a candidates competing for production position
# incumbent is already serving predictions
# new candidate must prove that it is sufficiently good before allowed to proceed

# Policy uses 4 checks:
# Precision, Recall, PR AUC, ROC AUC regression

# require a minimum PR-AUC improvement of 0.005 over the incumbent


import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RETRAINING_DIR = (
    PROJECT_ROOT / "artifacts" / "retraining"
)

CANDIDATE_REPORT_PATH = (
    RETRAINING_DIR / "candidate_evaluation.json"
)

BOOTSTRAP_REPORT_PATH = (
    RETRAINING_DIR / "bootstrap_validation.json"
)

CANDIDATE_MODEL_PATH = (
    RETRAINING_DIR / "retrained_candidate.joblib"
)

INCUMBENT_MODEL_PATH = (
    PROJECT_ROOT / "artifacts" / "random_forest_candidate.joblib"
)

PROMOTION_DECISION_PATH = (
    RETRAINING_DIR / "promotion_decision.json"
)


# Minimum acceptable candidate performance.
MINIMUM_PRECISION = 0.90
MINIMUM_RECALL = 0.70
MINIMUM_PR_AUC = 0.80

# Required observed PR-AUC improvement.
MINIMUM_PR_AUC_IMPROVEMENT = 0.005

# Maximum acceptable decline in ROC-AUC.
MAXIMUM_ROC_AUC_REGRESSION = 0.01


REQUIRED_METRICS = {
    "precision",
    "recall",
    "pr_auc",
    "roc_auc",
}


def load_json_report(path: Path, report_name: str) -> dict:
    """Load a JSON report and validate its basic structure."""

    if not path.exists():
        raise FileNotFoundError(
            f"{report_name} not found: {path}"
        )

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:
        report = json.load(file)

    if not isinstance(report, dict):
        raise ValueError(
            f"{report_name} must contain a JSON object."
        )

    return report


def load_candidate_report() -> dict:
    """Load the model comparison report."""

    report = load_json_report(
        CANDIDATE_REPORT_PATH,
        "Candidate evaluation report",
    )

    required_sections = {
        "incumbent_metrics",
        "candidate_metrics",
        "candidate_model_path",
    }

    if not required_sections.issubset(report):
        raise ValueError(
            "Candidate report is missing required sections."
        )

    return report


def load_bootstrap_report() -> dict:
    """Load the paired bootstrap validation report."""

    report = load_json_report(
        BOOTSTRAP_REPORT_PATH,
        "Bootstrap validation report",
    )

    if "results" not in report:
        raise ValueError(
            "Bootstrap report is missing its results section."
        )

    results = report["results"]

    required_fields = {
        "confidence_interval",
        "observed_pr_auc_improvement",
        "confidence_level",
        "interpretation",
        "bootstrap_samples",
    }

    if not required_fields.issubset(results):
        raise ValueError(
            "Bootstrap results are incomplete."
        )

    return report


def validate_metrics(
    metrics: dict,
    model_name: str,
) -> None:
    """Validate model metric values."""

    missing_metrics = REQUIRED_METRICS - set(metrics)

    if missing_metrics:
        raise ValueError(
            f"{model_name} is missing metrics: "
            f"{sorted(missing_metrics)}"
        )

    for metric_name in REQUIRED_METRICS:
        value = metrics[metric_name]

        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not np.isfinite(value)
        ):
            raise ValueError(
                f"{model_name} has invalid {metric_name}: {value}"
            )

        if not 0.0 <= value <= 1.0:
            raise ValueError(
                f"{model_name} {metric_name} "
                "must be between 0 and 1."
            )


def validate_bootstrap_results(
    results: dict,
) -> None:
    """Validate bootstrap statistics before using them."""

    interval = results["confidence_interval"]

    if not isinstance(interval, dict):
        raise ValueError(
            "Bootstrap confidence interval must be an object."
        )

    if not {
        "lower_bound",
        "upper_bound",
    }.issubset(interval):
        raise ValueError(
            "Bootstrap confidence interval is incomplete."
        )

    lower = interval["lower_bound"]
    upper = interval["upper_bound"]

    observed = results["observed_pr_auc_improvement"]

    for name, value in [
        ("lower confidence bound", lower),
        ("upper confidence bound", upper),
        ("observed improvement", observed),
    ]:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not np.isfinite(value)
        ):
            raise ValueError(
                f"Invalid bootstrap value: {name}"
            )

    if lower > upper:
        raise ValueError(
            "Bootstrap confidence interval is invalid."
        )

    confidence_level = results["confidence_level"]

    if (
        isinstance(confidence_level, bool)
        or not isinstance(confidence_level, (int, float))
        or not 0 < confidence_level < 1
    ):
        raise ValueError(
            "Bootstrap confidence level is invalid."
        )

    if (
        isinstance(results["bootstrap_samples"], bool)
        or not isinstance(results["bootstrap_samples"], int)
        or results["bootstrap_samples"] < 1
    ):
        raise ValueError(
            "Bootstrap sample count is invalid."
        )


def check_artifacts(report: dict) -> None:
    """Verify that both model artifacts exist."""

    if not INCUMBENT_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Incumbent model not found: {INCUMBENT_MODEL_PATH}"
        )

    if not CANDIDATE_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Candidate model not found: {CANDIDATE_MODEL_PATH}"
        )

    reported_path = Path(report["candidate_model_path"])

    if not reported_path.is_absolute():
        reported_path = PROJECT_ROOT / reported_path

    if reported_path.resolve() != CANDIDATE_MODEL_PATH.resolve():
        raise ValueError(
            "Candidate report points to an unexpected model artifact."
        )


def evaluate_promotion(
    candidate_report: dict,
    bootstrap_report: dict,
) -> dict:
    """Apply operational and statistical promotion requirements."""

    incumbent = candidate_report["incumbent_metrics"]
    candidate = candidate_report["candidate_metrics"]

    bootstrap = bootstrap_report["results"]

    validate_metrics(
        incumbent,
        "Incumbent model",
    )

    validate_metrics(
        candidate,
        "Candidate model",
    )

    validate_bootstrap_results(bootstrap)

    check_artifacts(candidate_report)

    checks = []

    def add_check(
        name: str,
        passed: bool,
        observed_value,
        required_value,
        explanation: str,
    ):
        checks.append(
            {
                "name": name,
                "passed": bool(passed),
                "observed_value": observed_value,
                "required_value": required_value,
                "explanation": explanation,
            }
        )

    # 1. Minimum precision.
    add_check(
        "minimum_precision",
        candidate["precision"] >= MINIMUM_PRECISION,
        candidate["precision"],
        f">= {MINIMUM_PRECISION}",
        "Candidate must meet the minimum precision requirement.",
    )

    # 2. Minimum recall.
    add_check(
        "minimum_recall",
        candidate["recall"] >= MINIMUM_RECALL,
        candidate["recall"],
        f">= {MINIMUM_RECALL}",
        "Candidate must meet the minimum fraud recall requirement.",
    )

    # 3. Minimum PR-AUC.
    add_check(
        "minimum_pr_auc",
        candidate["pr_auc"] >= MINIMUM_PR_AUC,
        candidate["pr_auc"],
        f">= {MINIMUM_PR_AUC}",
        "Candidate must meet the minimum PR-AUC requirement.",
    )

    # 4. Minimum observed PR-AUC improvement.
    pr_auc_improvement = (
        candidate["pr_auc"] - incumbent["pr_auc"]
    )

    add_check(
        "minimum_pr_auc_improvement",
        pr_auc_improvement >= MINIMUM_PR_AUC_IMPROVEMENT,
        round(pr_auc_improvement, 6),
        f">= {MINIMUM_PR_AUC_IMPROVEMENT}",
        "Candidate must achieve the configured observed PR-AUC gain.",
    )

    # 5. Bootstrap confidence interval.
    confidence_interval = bootstrap["confidence_interval"]

    lower_bound = confidence_interval["lower_bound"]
    upper_bound = confidence_interval["upper_bound"]

    add_check(
        "positive_bootstrap_confidence_interval",
        lower_bound > 0,
        {
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
        },
        "Lower confidence bound must be greater than 0.",
        (
            "The bootstrap confidence interval must support "
            "a positive PR-AUC improvement."
        ),
    )

    # 6. Maximum ROC-AUC regression.
    roc_auc_change = (
        candidate["roc_auc"] - incumbent["roc_auc"]
    )

    add_check(
        "roc_auc_regression",
        roc_auc_change >= -MAXIMUM_ROC_AUC_REGRESSION,
        round(roc_auc_change, 6),
        f">= {-MAXIMUM_ROC_AUC_REGRESSION}",
        "Candidate must not exceed the permitted ROC-AUC decline.",
    )

    failed_checks = [
        check["name"]
        for check in checks
        if not check["passed"]
    ]

    if failed_checks:
        status = "REJECTED"

        explanation = (
            "The candidate failed one or more operational or "
            "statistical promotion requirements. The incumbent "
            "model must remain active."
        )

    else:
        status = "APPROVED"

        explanation = (
            "The candidate passed all configured operational and "
            "statistical promotion checks. It may proceed to a "
            "separate deployment review."
        )

    return {
        "generated_at": datetime.now(
            timezone.utc
        ).isoformat(),

        "promotion_status": status,

        "checks_passed": (
            len(checks) - len(failed_checks)
        ),

        "checks_total": len(checks),

        "failed_checks": failed_checks,

        "checks": checks,

        "policy": {
            "minimum_precision": MINIMUM_PRECISION,
            "minimum_recall": MINIMUM_RECALL,
            "minimum_pr_auc": MINIMUM_PR_AUC,
            "minimum_pr_auc_improvement": (
                MINIMUM_PR_AUC_IMPROVEMENT
            ),
            "maximum_roc_auc_regression": (
                MAXIMUM_ROC_AUC_REGRESSION
            ),
            "bootstrap_confidence_level": (
                bootstrap["confidence_level"]
            ),
            "bootstrap_lower_bound_must_exceed": 0,
        },

        "bootstrap_evidence": {
            "observed_improvement": (
                bootstrap["observed_pr_auc_improvement"]
            ),
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "interpretation": bootstrap["interpretation"],
            "bootstrap_samples": bootstrap["bootstrap_samples"],
        },

        "model_comparison": {
            "incumbent_pr_auc": incumbent["pr_auc"],
            "candidate_pr_auc": candidate["pr_auc"],
            "pr_auc_improvement": round(
                pr_auc_improvement,
                6,
            ),
            "incumbent_roc_auc": incumbent["roc_auc"],
            "candidate_roc_auc": candidate["roc_auc"],
            "roc_auc_change": round(
                roc_auc_change,
                6,
            ),
        },

        "explanation": explanation,

        "deployment_instruction": (
            "This report does not deploy or replace any model. "
            "The incumbent must remain active until a separate "
            "deployment process completes successfully."
        ),
    }


def save_promotion_decision(decision: dict) -> None:
    """Persist the promotion decision."""

    RETRAINING_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        PROMOTION_DECISION_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            decision,
            file,
            indent=4,
        )

    print(
        f"\nPromotion decision saved to: "
        f"{PROMOTION_DECISION_PATH}"
    )


def print_promotion_decision(decision: dict) -> None:
    """Display each promotion check."""

    print("\n========== PROMOTION POLICY ==========")

    for check in decision["checks"]:
        status = (
            "PASS" if check["passed"] else "FAIL"
        )

        print(f"\n{check['name']}: {status}")

        print(
            f"Observed: {check['observed_value']}"
        )

        print(
            f"Required: {check['required_value']}"
        )

    print("\n========== FINAL PROMOTION DECISION ==========")

    print(
        f"Status: {decision['promotion_status']}"
    )

    print(
        f"Checks passed: "
        f"{decision['checks_passed']}/"
        f"{decision['checks_total']}"
    )

    if decision["failed_checks"]:
        print("\nFailed checks:")

        for name in decision["failed_checks"]:
            print(f"- {name}")

    print("\nExplanation:")
    print(decision["explanation"])

    print("\nDeployment instruction:")
    print(decision["deployment_instruction"])


def main():
    print("Loading candidate evaluation report...")

    candidate_report = load_candidate_report()

    print("Candidate evaluation report loaded.")

    print("\nLoading bootstrap validation report...")

    bootstrap_report = load_bootstrap_report()

    print("Bootstrap validation report loaded.")

    print("\nApplying integrated promotion policy...")

    decision = evaluate_promotion(
        candidate_report=candidate_report,
        bootstrap_report=bootstrap_report,
    )

    print_promotion_decision(decision)

    save_promotion_decision(decision)

    print("\nIntegrated promotion policy completed successfully.")


if __name__ == "__main__":
    main()


# Candidte was rejected and incumbent mode remains protected
# ROC AUR failed

# 6th condition updatation: 
# lower bound of the 95% bootstrap confidence interval must be greater than zero
