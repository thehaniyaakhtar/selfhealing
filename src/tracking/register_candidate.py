
import json
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn

from mlflow import MlflowClient

from src.tracking.setup_mlflow import configure_mlflow


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RETRAINING_DIR = PROJECT_ROOT / "artifacts" / "retraining"

CANDIDATE_MODEL_PATH = (
    RETRAINING_DIR / "retrained_candidate.joblib"
)

CANDIDATE_REPORT_PATH = (
    RETRAINING_DIR / "candidate_evaluation.json"
)

BOOTSTRAP_REPORT_PATH = (
    RETRAINING_DIR / "bootstrap_validation.json"
)

PROMOTION_DECISION_PATH = (
    RETRAINING_DIR / "promotion_decision.json"
)

REGISTERED_MODEL_NAME = "self-healing-fraud-detector"

SOURCE_TRAINING_RUN_ID = (
    "8cfca97e4d98483b96369e6213ba58da"
)

# Explicitly trust only the scikit-learn tree structure
# used internally by our locally trained Random Forest.
TRUSTED_SKOPS_TYPES = [
    "sklearn.tree._tree.Tree"
]


def load_json(path: Path, description: str) -> dict:
    """Load a JSON report."""

    if not path.exists():
        raise FileNotFoundError(
            f"{description} not found: {path}"
        )

    with open(path, "r", encoding="utf-8") as file:
        report = json.load(file)

    if not isinstance(report, dict):
        raise ValueError(
            f"{description} must contain a JSON object."
        )

    return report


def validate_inputs():
    """Validate candidate artifacts and promotion information."""

    required_paths = [
        CANDIDATE_MODEL_PATH,
        CANDIDATE_REPORT_PATH,
        BOOTSTRAP_REPORT_PATH,
        PROMOTION_DECISION_PATH,
    ]

    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(
                f"Required file not found: {path}"
            )

    candidate_report = load_json(
        CANDIDATE_REPORT_PATH,
        "Candidate evaluation report",
    )

    bootstrap_report = load_json(
        BOOTSTRAP_REPORT_PATH,
        "Bootstrap validation report",
    )

    promotion_decision = load_json(
        PROMOTION_DECISION_PATH,
        "Promotion decision",
    )

    status = promotion_decision.get("promotion_status")

    if status not in {"APPROVED", "REJECTED"}:
        raise ValueError(
            f"Unexpected promotion status: {status}"
        )

    if "candidate_metrics" not in candidate_report:
        raise ValueError(
            "Candidate metrics are missing."
        )

    if "results" not in bootstrap_report:
        raise ValueError(
            "Bootstrap results are missing."
        )

    return (
        candidate_report,
        bootstrap_report,
        promotion_decision,
    )


def register_candidate(
    candidate_report: dict,
    bootstrap_report: dict,
    promotion_decision: dict,
):
    """Log and register the candidate without deploying it."""

    client = MlflowClient()

    # Verify the original experiment run exists.
    source_run = client.get_run(
        SOURCE_TRAINING_RUN_ID
    )

    print(
        f"Source training run verified: "
        f"{source_run.info.run_id}"
    )

    # Load the already-trained model.
    # No retraining or modification occurs here.
    candidate_model = joblib.load(
        CANDIDATE_MODEL_PATH
    )

    candidate_metrics = candidate_report[
        "candidate_metrics"
    ]

    bootstrap_results = bootstrap_report["results"]

    promotion_status = promotion_decision[
        "promotion_status"
    ]

    deployment_eligible = (
        promotion_status == "APPROVED"
    )

    print("\nStarting MLflow registry logging run...")

    with mlflow.start_run(
        run_name="registry-retrained-random-forest-candidate-v1"
    ) as run:

        mlflow.set_tags(
            {
                "model_name": REGISTERED_MODEL_NAME,
                "model_role": "candidate",
                "promotion_status": promotion_status,
                "deployment_eligible": str(
                    deployment_eligible
                ).lower(),
                "deployment_status": "NOT_DEPLOYED",
                "source_training_run_id": (
                    SOURCE_TRAINING_RUN_ID
                ),
            }
        )

        # Record evaluation and bootstrap metrics.
        mlflow.log_metrics(
            {
                "precision": candidate_metrics["precision"],
                "recall": candidate_metrics["recall"],
                "f1_score": candidate_metrics["f1_score"],
                "pr_auc": candidate_metrics["pr_auc"],
                "roc_auc": candidate_metrics["roc_auc"],
                "bootstrap_pr_auc_improvement": (
                    bootstrap_results[
                        "observed_pr_auc_improvement"
                    ]
                ),
                "bootstrap_ci_lower": (
                    bootstrap_results[
                        "confidence_interval"
                    ]["lower_bound"]
                ),
                "bootstrap_ci_upper": (
                    bootstrap_results[
                        "confidence_interval"
                    ]["upper_bound"]
                ),
            }
        )

        # Log the model using the scikit-learn flavor.
        # Only the reviewed tree type is explicitly trusted.
        model_info = mlflow.sklearn.log_model(
            sk_model=candidate_model,
            name="candidate_model",
            skops_trusted_types=TRUSTED_SKOPS_TYPES,
        )

        # Preserve the evaluation evidence.
        for report_path in [
            CANDIDATE_REPORT_PATH,
            BOOTSTRAP_REPORT_PATH,
            PROMOTION_DECISION_PATH,
        ]:
            mlflow.log_artifact(
                str(report_path),
                artifact_path="evaluation",
            )

        registry_run_id = run.info.run_id

    print("Model artifact logged successfully.")

    # Register the model as a new version.
    registered_version = mlflow.register_model(
        model_uri=model_info.model_uri,
        name=REGISTERED_MODEL_NAME,
    )

    version = registered_version.version

    # Attach lifecycle and audit tags.
    tags = {
        "promotion_status": promotion_status,
        "deployment_eligible": str(
            deployment_eligible
        ).lower(),
        "deployment_status": "NOT_DEPLOYED",
        "source_training_run_id": SOURCE_TRAINING_RUN_ID,
    }

    for key, value in tags.items():
        client.set_model_version_tag(
            REGISTERED_MODEL_NAME,
            version,
            key,
            value,
        )

    client.update_model_version(
        name=REGISTERED_MODEL_NAME,
        version=version,
        description=(
            f"Random Forest candidate. "
            f"Promotion status: {promotion_status}. "
            f"PR-AUC: {candidate_metrics['pr_auc']:.6f}. "
            f"ROC-AUC: {candidate_metrics['roc_auc']:.6f}. "
            "Not deployed."
        ),
    )

    return {
        "model_name": REGISTERED_MODEL_NAME,
        "model_version": version,
        "promotion_status": promotion_status,
        "deployment_eligible": deployment_eligible,
        "deployment_status": "NOT_DEPLOYED",
        "source_training_run_id": SOURCE_TRAINING_RUN_ID,
        "registry_run_id": registry_run_id,
        "model_uri": registered_version.source,
    }


def main():
    print("========== MODEL REGISTRY ==========")

    print("\nLoading candidate information...")

    (
        candidate_report,
        bootstrap_report,
        promotion_decision,
    ) = validate_inputs()

    print("Candidate information loaded.")

    print("\nConfiguring MLflow...")

    configure_mlflow()

    print("\nRegistering candidate model...")

    result = register_candidate(
        candidate_report,
        bootstrap_report,
        promotion_decision,
    )

    print("\n========== REGISTRATION SUMMARY ==========")

    for key, value in result.items():
        print(f"{key}: {value}")

    print("\nModel registration completed successfully.")


if __name__ == "__main__":
    main()
