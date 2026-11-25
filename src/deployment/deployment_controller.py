
import json

from datetime import datetime, timezone
from pathlib import Path

from mlflow import MlflowClient

from src.tracking.setup_mlflow import configure_mlflow


# --------------------------------------------------
# PROJECT CONFIGURATION
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEPLOYMENT_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "deployment"
)

DEPLOYMENT_AUDIT_PATH = (
    DEPLOYMENT_DIR
    / "deployment_attempt.json"
)

REGISTERED_MODEL_NAME = "self-healing-fraud-detector"

# Version 1 is our currently registered test candidate.
TARGET_MODEL_VERSION = "1"


# --------------------------------------------------
# AUDIT RECORD
# --------------------------------------------------

def save_deployment_audit(record):

    DEPLOYMENT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        DEPLOYMENT_AUDIT_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            record,
            file,
            indent=4,
        )


# --------------------------------------------------
# DEPLOYMENT ELIGIBILITY CHECK
# --------------------------------------------------

def evaluate_deployment_eligibility(model_version):

    tags = model_version.tags

    promotion_status = tags.get("promotion_status")
    deployment_eligible = tags.get("deployment_eligible")
    deployment_status = tags.get("deployment_status")
    lifecycle_status = tags.get("lifecycle_status")

    checks = {
        "promotion_approved": (
            promotion_status == "APPROVED"
        ),

        "deployment_eligible": (
            deployment_eligible == "true"
        ),

        "not_already_deployed": (
            deployment_status == "NOT_DEPLOYED"
        ),

        "approved_pending_deployment": (
            lifecycle_status == "APPROVED_PENDING_DEPLOYMENT"
        ),
    }

    return checks


# --------------------------------------------------
# DEPLOYMENT CONTROLLER
# --------------------------------------------------

def deploy_model():

    print("\n========== DEPLOYMENT CONTROLLER ==========")

    print("\nConfiguring MLflow...")

    configure_mlflow()

    client = MlflowClient()

    print("\nRetrieving registered model version...")

    model_version = client.get_model_version(
        name=REGISTERED_MODEL_NAME,
        version=TARGET_MODEL_VERSION,
    )

    print(f"Model: {model_version.name}")
    print(f"Version: {model_version.version}")

    # --------------------------------------------------
    # CHECK DEPLOYMENT ELIGIBILITY
    # --------------------------------------------------

    print("\nEvaluating deployment safety checks...")

    checks = evaluate_deployment_eligibility(
        model_version
    )

    for check_name, passed in checks.items():

        status = "PASS" if passed else "FAIL"

        print(
            f"{check_name}: {status}"
        )

    all_checks_passed = all(checks.values())

    # --------------------------------------------------
    # BUILD AUDIT RECORD
    # --------------------------------------------------

    tags = model_version.tags

    audit_record = {
        "timestamp": datetime.now(
            timezone.utc
        ).isoformat(),

        "model_name": model_version.name,

        "model_version": model_version.version,

        "source_training_run_id": tags.get(
            "source_training_run_id"
        ),

        "promotion_status": tags.get(
            "promotion_status"
        ),

        "deployment_eligible": tags.get(
            "deployment_eligible"
        ),

        "deployment_status_before_attempt": tags.get(
            "deployment_status"
        ),

        "deployment_checks": checks,

        "deployment_status": (
            "ELIGIBLE_FOR_DEPLOYMENT"
            if all_checks_passed
            else "BLOCKED"
        ),

        "deployment_executed": False,

        "incumbent_modified": False,
    }

    # --------------------------------------------------
    # BLOCK OR AUTHORIZE
    # --------------------------------------------------

    if not all_checks_passed:

        audit_record["reason"] = (
            "Model failed one or more deployment "
            "eligibility checks."
        )

        save_deployment_audit(audit_record)

        print("\nDEPLOYMENT BLOCKED")

        print(
            "The candidate is not authorized "
            "to replace the incumbent model."
        )

        print(
            f"\nAudit saved to: {DEPLOYMENT_AUDIT_PATH}"
        )

        return

    # --------------------------------------------------
    # SAFETY GATE PASSED
    # --------------------------------------------------

    audit_record["reason"] = (
        "All deployment eligibility checks passed. "
        "Actual deployment has not yet been implemented."
    )

    save_deployment_audit(audit_record)

    print("\nDEPLOYMENT GATE PASSED")

    print(
        "The candidate is eligible for the next "
        "deployment stage."
    )

    print(
        "Actual model replacement has not been executed."
    )

    print(
        f"\nAudit saved to: {DEPLOYMENT_AUDIT_PATH}"
    )


# --------------------------------------------------
# MAIN
# --------------------------------------------------

if __name__ == "__main__":

    deploy_model()
