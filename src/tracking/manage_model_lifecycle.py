
import json
from pathlib import Path

from mlflow import MlflowClient

from src.tracking.setup_mlflow import configure_mlflow


# --------------------------------------------------
# PROJECT CONFIGURATION
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROMOTION_DECISION_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "retraining"
    / "promotion_decision.json"
)

REGISTERED_MODEL_NAME = "self-healing-fraud-detector"

SOURCE_TRAINING_RUN_ID = (
    "8cfca97e4d98483b96369e6213ba58da"
)

LATEST_EVALUATED_ALIAS = "latest-evaluated"

CONTROLLER_VERSION = "1.1"


# --------------------------------------------------
# INPUT VALIDATION
# --------------------------------------------------

def load_promotion_decision():

    if not PROMOTION_DECISION_PATH.exists():
        raise FileNotFoundError(
            f"Promotion decision not found: "
            f"{PROMOTION_DECISION_PATH}"
        )

    with open(
        PROMOTION_DECISION_PATH,
        "r",
        encoding="utf-8"
    ) as file:
        decision = json.load(file)

    # Match the actual promotion policy schema.
    promotion_status = decision.get("promotion_status")

    if promotion_status not in {"APPROVED", "REJECTED"}:
        raise ValueError(
            "Invalid or missing promotion_status. "
            "Expected APPROVED or REJECTED."
        )

    # Validate that the report contains its policy evidence.
    if not isinstance(decision.get("checks"), list):
        raise ValueError(
            "Promotion report is missing its checks list."
        )

    if not isinstance(decision.get("checks_passed"), int):
        raise ValueError(
            "Promotion report has an invalid checks_passed value."
        )

    if not isinstance(decision.get("checks_total"), int):
        raise ValueError(
            "Promotion report has an invalid checks_total value."
        )

    if decision["checks_passed"] > decision["checks_total"]:
        raise ValueError(
            "Promotion report contains inconsistent check counts."
        )

    return decision


# --------------------------------------------------
# REGISTERED VERSION DISCOVERY
# --------------------------------------------------

def find_registered_version(client):

    versions = client.search_model_versions(
        filter_string=f"name='{REGISTERED_MODEL_NAME}'"
    )

    matching_versions = [
        version
        for version in versions
        if version.tags.get("source_training_run_id")
        == SOURCE_TRAINING_RUN_ID
    ]

    if len(matching_versions) == 0:
        raise RuntimeError(
            "No registered model version matches "
            "the source training run ID."
        )

    if len(matching_versions) > 1:
        raise RuntimeError(
            "Multiple registered versions match this "
            "training run. Refusing to choose one automatically."
        )

    return matching_versions[0]


# --------------------------------------------------
# LIFECYCLE POLICY
# --------------------------------------------------

def build_lifecycle_tags(
    promotion_status,
    existing_tags
):

    deployment_status = existing_tags.get(
        "deployment_status",
        "NOT_DEPLOYED"
    )

    if promotion_status == "REJECTED":

        if deployment_status in {
            "DEPLOYED",
            "DEPLOYMENT_VERIFIED"
        }:
            raise RuntimeError(
                "Safety violation: this version is already "
                "marked as deployed. Refusing to rewrite "
                "its lifecycle status."
            )

        return {
            "promotion_status": "REJECTED",
            "deployment_eligible": "false",
            "deployment_status": "NOT_DEPLOYED",
            "lifecycle_status": "REJECTED",
            "lifecycle_controller_version": CONTROLLER_VERSION,
        }

    # Approval is permission to enter deployment evaluation.
    # It does not mean that deployment has occurred.

    if deployment_status == "NOT_DEPLOYED":
        lifecycle_status = "APPROVED_PENDING_DEPLOYMENT"
    else:
        lifecycle_status = "APPROVED_DEPLOYMENT_IN_PROGRESS"

    return {
        "promotion_status": "APPROVED",
        "deployment_eligible": "true",
        "deployment_status": deployment_status,
        "lifecycle_status": lifecycle_status,
        "lifecycle_controller_version": CONTROLLER_VERSION,
    }


# --------------------------------------------------
# REGISTRY UPDATE
# --------------------------------------------------

def synchronize_model_lifecycle(
    client,
    model_version,
    decision
):

    version_number = model_version.version
    existing_tags = model_version.tags

    promotion_status = decision["promotion_status"]

    lifecycle_tags = build_lifecycle_tags(
        promotion_status,
        existing_tags
    )

    print("\nUpdating model version tags...")

    for key, value in lifecycle_tags.items():

        client.set_model_version_tag(
            name=REGISTERED_MODEL_NAME,
            version=version_number,
            key=key,
            value=value
        )

    # This alias identifies the latest evaluated version.
    # It does not imply approval or production deployment.

    client.set_registered_model_alias(
        name=REGISTERED_MODEL_NAME,
        alias=LATEST_EVALUATED_ALIAS,
        version=version_number
    )

    # Verify the saved tags.

    updated_version = client.get_model_version(
        name=REGISTERED_MODEL_NAME,
        version=version_number
    )

    for key, expected_value in lifecycle_tags.items():

        actual_value = updated_version.tags.get(key)

        if actual_value != expected_value:
            raise RuntimeError(
                f"Registry verification failed for {key}: "
                f"expected {expected_value}, got {actual_value}"
            )

    # Verify alias assignment.

    alias_version = client.get_model_version_by_alias(
        name=REGISTERED_MODEL_NAME,
        alias=LATEST_EVALUATED_ALIAS
    )

    if alias_version.version != version_number:
        raise RuntimeError(
            "Alias verification failed. "
            "The alias points to a different model version."
        )

    return updated_version


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def manage_model_lifecycle():

    print("\n========== MODEL LIFECYCLE CONTROLLER ==========")

    print("\nLoading promotion decision...")

    decision = load_promotion_decision()

    print(
        f"Promotion status: {decision['promotion_status']}"
    )

    print(
        f"Policy checks passed: "
        f"{decision['checks_passed']}/"
        f"{decision['checks_total']}"
    )

    print("\nConfiguring MLflow...")

    configure_mlflow()

    client = MlflowClient()

    print("\nFinding registered model version...")

    model_version = find_registered_version(client)

    print(f"Registered model: {model_version.name}")
    print(f"Model version: {model_version.version}")

    print(
        "Source training run: "
        f"{model_version.tags.get('source_training_run_id')}"
    )

    print("\nApplying lifecycle policy...")

    updated_version = synchronize_model_lifecycle(
        client=client,
        model_version=model_version,
        decision=decision
    )

    print("\n========== LIFECYCLE SUMMARY ==========")

    print(f"Model: {updated_version.name}")
    print(f"Version: {updated_version.version}")

    print(
        "Promotion status: "
        f"{updated_version.tags.get('promotion_status')}"
    )

    print(
        "Deployment eligible: "
        f"{updated_version.tags.get('deployment_eligible')}"
    )

    print(
        "Deployment status: "
        f"{updated_version.tags.get('deployment_status')}"
    )

    print(
        "Lifecycle status: "
        f"{updated_version.tags.get('lifecycle_status')}"
    )

    print(f"Alias: {LATEST_EVALUATED_ALIAS}")
    print("Champion alias: NOT MODIFIED")

    print(
        "\nLifecycle synchronization completed successfully."
    )


if __name__ == "__main__":
    manage_model_lifecycle()
