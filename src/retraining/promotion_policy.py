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

# The candidate must improve PR-AUC by at least this amount.
MINIMUM_PR_AUC_IMPROVEMENT = 0.005

# Maximum acceptable decline in ROC-AUC relative to incumbent.
MAXIMUM_ROC_AUC_REGRESSION = 0.01


REQUIRED_METRICS = {
    "precision",
    "recall",
    "pr_auc",
    "roc_auc",
}


def load_candidate_report() -> dict:
    """Load the model comparison report."""

    if not CANDIDATE_REPORT_PATH.exists():
        raise FileNotFoundError(
            f"Candidate report not found: {CANDIDATE_REPORT_PATH}"
        )

    with open(
        CANDIDATE_REPORT_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        report = json.load(file)

    required_sections = {
        "incumbent_metrics",
        "candidate_metrics",
        "candidate_minus_incumbent",
        "candidate_model_path",
    }

    if not required_sections.issubset(report):
        raise ValueError(
            "Candidate report is missing required sections."
        )

    return report


def validate_metrics(
    metrics: dict,
    model_name: str,
) -> None:
    """Validate the metric values before making a decision."""

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
                f"{model_name} has an invalid value "
                f"for {metric_name}: {value}"
            )

        if not 0.0 <= value <= 1.0:
            raise ValueError(
                f"{model_name} metric {metric_name} "
                "must be between 0 and 1."
            )


def check_artifacts(report: dict) -> None:
    """Ensure both model artifacts exist before promotion review."""

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


def evaluate_promotion(report: dict) -> dict:
    """Apply the promotion rules to the candidate model."""

    incumbent = report["incumbent_metrics"]
    candidate = report["candidate_metrics"]

    validate_metrics(
        incumbent,
        "Incumbent model",
    )

    validate_metrics(
        candidate,
        "Candidate model",
    )

    check_artifacts(report)

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

    # Check 1: Minimum precision.
    add_check(
        name="minimum_precision",
        passed=(
            candidate["precision"] >= MINIMUM_PRECISION
        ),
        observed_value=candidate["precision"],
        required_value=f">= {MINIMUM_PRECISION}",
        explanation=(
            "Candidate precision must meet the minimum "
            "operational requirement."
        ),
    )

    # Check 2: Minimum recall.
    add_check(
        name="minimum_recall",
        passed=(
            candidate["recall"] >= MINIMUM_RECALL
        ),
        observed_value=candidate["recall"],
        required_value=f">= {MINIMUM_RECALL}",
        explanation=(
            "Candidate recall must meet the minimum "
            "fraud detection requirement."
        ),
    )

    # Check 3: Minimum PR-AUC.
    add_check(
        name="minimum_pr_auc",
        passed=(
            candidate["pr_auc"] >= MINIMUM_PR_AUC
        ),
        observed_value=candidate["pr_auc"],
        required_value=f">= {MINIMUM_PR_AUC}",
        explanation=(
            "Candidate PR-AUC must meet the minimum "
            "ranking performance requirement."
        ),
    )

    # Check 4: Meaningful PR-AUC improvement.
    pr_auc_improvement = (
        candidate["pr_auc"] - incumbent["pr_auc"]
    )

    add_check(
        name="minimum_pr_auc_improvement",
        passed=(
            pr_auc_improvement
            >= MINIMUM_PR_AUC_IMPROVEMENT
        ),
        observed_value=round(pr_auc_improvement, 6),
        required_value=(
            f">= {MINIMUM_PR_AUC_IMPROVEMENT}"
        ),
        explanation=(
            "Candidate must improve PR-AUC by at least "
            "the configured minimum amount."
        ),
    )

    # Check 5: Protect against excessive ROC-AUC regression.
    roc_auc_change = (
        candidate["roc_auc"] - incumbent["roc_auc"]
    )

    add_check(
        name="roc_auc_regression",
        passed=(
            roc_auc_change
            >= -MAXIMUM_ROC_AUC_REGRESSION
        ),
        observed_value=round(roc_auc_change, 6),
        required_value=(
            f">= {-MAXIMUM_ROC_AUC_REGRESSION}"
        ),
        explanation=(
            "Candidate ROC-AUC must not decline beyond "
            "the configured tolerance."
        ),
    )

    failed_checks = [
        check["name"]
        for check in checks
        if not check["passed"]
    ]

    if failed_checks:
        status = "REJECTED"

        explanation = (
            "The candidate failed one or more promotion checks. "
            "The incumbent model must remain active."
        )

    else:
        status = "APPROVED"

        explanation = (
            "The candidate passed all configured promotion checks. "
            "It is eligible to proceed to deployment review. "
            "This decision does not deploy the candidate."
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
            "Do not replace the incumbent model in this stage. "
            "A separate deployment process must handle model "
            "promotion, health verification and rollback."
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
    """Print every promotion check and its result."""

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

        for check_name in decision["failed_checks"]:
            print(f"- {check_name}")

    print("\nExplanation:")
    print(decision["explanation"])

    print("\nDeployment instruction:")
    print(decision["deployment_instruction"])


def main():
    print("Loading candidate evaluation report...")

    report = load_candidate_report()

    print("Candidate evaluation report loaded.")

    print("\nApplying promotion policy...")

    decision = evaluate_promotion(report)

    print_promotion_decision(decision)

    save_promotion_decision(decision)

    print("\nPromotion policy completed successfully.")


if __name__ == "__main__":
    main()
