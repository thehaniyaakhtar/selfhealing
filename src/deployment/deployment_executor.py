
import json
import os
import tempfile

from datetime import datetime, timezone
from pathlib import Path

import joblib
import mlflow.sklearn
import numpy as np
import pandas as pd

from mlflow import MlflowClient

from src.tracking.setup_mlflow import configure_mlflow
from src.deployment.deployment_controller import (
    evaluate_deployment_eligibility,
)


# --------------------------------------------------
# PROJECT CONFIGURATION
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEPLOYMENT_DIR = PROJECT_ROOT / "artifacts" / "deployment"
MODEL_STORE_DIR = DEPLOYMENT_DIR / "models"

ACTIVE_MANIFEST_PATH = DEPLOYMENT_DIR / "active_model.json"
EXECUTION_AUDIT_PATH = DEPLOYMENT_DIR / "deployment_execution.json"

CANDIDATE_REPORT_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "retraining"
    / "candidate_evaluation.json"
)

EVALUATION_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "reference"
    / "evaluation.csv"
)

REGISTERED_MODEL_NAME = "self-healing-fraud-detector"
TARGET_MODEL_VERSION = "1"


# The API expects these 30 input features in this order.
FEATURE_COLUMNS = (
    ["Time", "Amount"]
    + [f"V{i}" for i in range(1, 29)]
)


# --------------------------------------------------
# FILE UTILITIES
# --------------------------------------------------

def load_json(path: Path, description: str) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"{description} not found: {path}"
        )

    with open(path, "r", encoding="utf-8") as file:
        result = json.load(file)

    if not isinstance(result, dict):
        raise ValueError(
            f"{description} must contain a JSON object."
        )

    return result


def atomic_write_json(path: Path, data: dict):
    """
    Write JSON to a temporary file and atomically
    replace the destination.
    """

    path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            suffix=".tmp",
            delete=False,
        ) as file:

            temporary_path = Path(file.name)

            json.dump(data, file, indent=4)
            file.flush()
            os.fsync(file.fileno())

        os.replace(temporary_path, path)

    finally:
        if (
            temporary_path is not None
            and temporary_path.exists()
        ):
            temporary_path.unlink()


def save_audit(record: dict):
    atomic_write_json(
        EXECUTION_AUDIT_PATH,
        record,
    )


# --------------------------------------------------
# CANDIDATE VALIDATION
# --------------------------------------------------

def validate_candidate_model(model):
    """Validate the loaded model's inference interface."""

    if not hasattr(model, "predict_proba"):
        raise ValueError(
            "Model does not support predict_proba."
        )

    if not hasattr(model, "classes_"):
        raise ValueError(
            "Model does not expose its trained classes."
        )

    if 1 not in list(model.classes_):
        raise ValueError(
            "Fraud class 1 is missing from the model."
        )


def validate_inference(model):
    """
    Run a real inference check using one observation
    from the held-out evaluation dataset.
    """

    evaluation_data = pd.read_csv(
        EVALUATION_DATA_PATH
    )

    missing_features = (
        set(FEATURE_COLUMNS)
        - set(evaluation_data.columns)
    )

    if missing_features:
        raise ValueError(
            f"Evaluation data is missing features: "
            f"{sorted(missing_features)}"
        )

    sample = evaluation_data[
        FEATURE_COLUMNS
    ].iloc[[0]]

    probabilities = model.predict_proba(sample)

    fraud_class_index = list(
        model.classes_
    ).index(1)

    fraud_probability = float(
        probabilities[0][fraud_class_index]
    )

    if not np.isfinite(fraud_probability):
        raise ValueError(
            "Model returned a non-finite probability."
        )

    if not 0.0 <= fraud_probability <= 1.0:
        raise ValueError(
            "Model returned an invalid probability."
        )

    return fraud_probability


# --------------------------------------------------
# DEPLOYMENT EXECUTOR
# --------------------------------------------------

def execute_deployment():

    print("\n========== DEPLOYMENT EXECUTOR ==========")

    DEPLOYMENT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    MODEL_STORE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("\nConfiguring MLflow...")

    configure_mlflow()

    client = MlflowClient()

    print("\nRetrieving exact registered model version...")

    model_version = client.get_model_version(
        name=REGISTERED_MODEL_NAME,
        version=TARGET_MODEL_VERSION,
    )

    tags = model_version.tags

    print(f"Model: {model_version.name}")
    print(f"Version: {model_version.version}")
    print(f"Registry status: {model_version.status}")

    # --------------------------------------------------
    # AUTHORIZATION GATE
    # --------------------------------------------------

    print("\nRechecking deployment authorization...")

    checks = evaluate_deployment_eligibility(
        model_version
    )

    checks["registry_version_ready"] = (
        model_version.status == "READY"
    )

    checks["source_training_run_present"] = bool(
        tags.get("source_training_run_id")
    )

    for name, passed in checks.items():
        print(
            f"{name}: {'PASS' if passed else 'FAIL'}"
        )

    if not all(checks.values()):

        audit = {
            "timestamp": datetime.now(
                timezone.utc
            ).isoformat(),
            "model_name": model_version.name,
            "model_version": model_version.version,
            "deployment_checks": checks,
            "deployment_status": "BLOCKED",
            "deployment_executed": False,
            "incumbent_modified": False,
            "reason": (
                "The registered model did not pass "
                "all deployment authorization checks."
            ),
        }

        save_audit(audit)

        print("\nDEPLOYMENT BLOCKED")
        print("The incumbent model remains unchanged.")
        print(
            f"Audit saved to: {EXECUTION_AUDIT_PATH}"
        )

        return

    # --------------------------------------------------
    # LOAD DEPLOYMENT CONFIGURATION
    # --------------------------------------------------

    candidate_report = load_json(
        CANDIDATE_REPORT_PATH,
        "Candidate evaluation report",
    )

    threshold = float(
        candidate_report["decision_threshold"]
    )

    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError(
            "Candidate decision threshold is invalid."
        )

    print(f"\nDecision threshold: {threshold}")

    # --------------------------------------------------
    # DOWNLOAD EXACT MLflow MODEL VERSION
    # --------------------------------------------------

    model_uri = (
        f"models:/{REGISTERED_MODEL_NAME}/"
        f"{TARGET_MODEL_VERSION}"
    )

    print(f"\nLoading registered model: {model_uri}")

    model = mlflow.sklearn.load_model(
        model_uri
    )

    validate_candidate_model(model)

    fraud_probability = validate_inference(model)

    print(
        "Model inference validation passed."
    )

    print(
        f"Test fraud probability: {fraud_probability:.6f}"
    )

    # --------------------------------------------------
    # SAVE VERSIONED DEPLOYMENT ARTIFACT
    # --------------------------------------------------

    version_directory = (
        MODEL_STORE_DIR
        / f"version_{TARGET_MODEL_VERSION}"
    )

    version_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_path = (
        version_directory
        / "model.joblib"
    )

    temporary_model_path = (
        version_directory
        / "model.tmp"
    )

    try:
        joblib.dump(
            model,
            temporary_model_path,
        )

        # Confirm the saved artifact can be loaded.
        verified_model = joblib.load(
            temporary_model_path
        )

        validate_candidate_model(verified_model)

        os.replace(
            temporary_model_path,
            model_path,
        )

    finally:
        if temporary_model_path.exists():
            temporary_model_path.unlink()

    print(
        f"\nVersioned model artifact saved: {model_path}"
    )

    # --------------------------------------------------
    # PREPARE ACTIVE MODEL MANIFEST
    # --------------------------------------------------

    relative_model_path = (
        model_path
        .relative_to(PROJECT_ROOT)
        .as_posix()
    )

    manifest = {
        "model_name": REGISTERED_MODEL_NAME,
        "model_version": str(TARGET_MODEL_VERSION),
        "model_path": relative_model_path,
        "decision_threshold": threshold,
        "mlflow_model_uri": model_uri,
        "activated_at": datetime.now(
            timezone.utc
        ).isoformat(),
    }

    # Preserve the previous manifest so it can be
    # restored if a later activation step fails.
    previous_manifest = None

    if ACTIVE_MANIFEST_PATH.exists():
        previous_manifest = (
            ACTIVE_MANIFEST_PATH.read_bytes()
        )

    previous_deployment_status = tags.get(
        "deployment_status"
    )

    previous_lifecycle_status = tags.get(
        "lifecycle_status"
    )

    # --------------------------------------------------
    # ATOMIC ACTIVATION
    # --------------------------------------------------

    print("\nActivating model...")

    try:

        atomic_write_json(
            ACTIVE_MANIFEST_PATH,
            manifest,
        )

        # Update registry metadata only after the
        # active manifest has been written successfully.
        client.set_model_version_tag(
            REGISTERED_MODEL_NAME,
            TARGET_MODEL_VERSION,
            "deployment_status",
            "DEPLOYED",
        )

        client.set_model_version_tag(
            REGISTERED_MODEL_NAME,
            TARGET_MODEL_VERSION,
            "lifecycle_status",
            "DEPLOYED",
        )

    except Exception:

        print(
            "\nActivation failed. Restoring previous manifest."
        )

        if previous_manifest is None:

            if ACTIVE_MANIFEST_PATH.exists():
                ACTIVE_MANIFEST_PATH.unlink()

        else:

            temporary_path = (
                ACTIVE_MANIFEST_PATH.with_suffix(
                    ".rollback.tmp"
                )
            )

            temporary_path.write_bytes(
                previous_manifest
            )

            os.replace(
                temporary_path,
                ACTIVE_MANIFEST_PATH,
            )

        # Restore registry tags if activation partially
        # updated them.
        if previous_deployment_status is not None:
            client.set_model_version_tag(
                REGISTERED_MODEL_NAME,
                TARGET_MODEL_VERSION,
                "deployment_status",
                previous_deployment_status,
            )

        if previous_lifecycle_status is not None:
            client.set_model_version_tag(
                REGISTERED_MODEL_NAME,
                TARGET_MODEL_VERSION,
                "lifecycle_status",
                previous_lifecycle_status,
            )

        raise

    # --------------------------------------------------
    # SUCCESS AUDIT
    # --------------------------------------------------

    audit = {
        "timestamp": datetime.now(
            timezone.utc
        ).isoformat(),
        "model_name": REGISTERED_MODEL_NAME,
        "model_version": TARGET_MODEL_VERSION,
        "model_uri": model_uri,
        "model_path": relative_model_path,
        "decision_threshold": threshold,
        "deployment_checks": checks,
        "deployment_status": "DEPLOYED",
        "deployment_executed": True,
        "incumbent_modified": True,
        "inference_validation": "PASSED",
    }

    save_audit(audit)

    print("\nDEPLOYMENT SUCCESSFUL")
    print(f"Active model version: {TARGET_MODEL_VERSION}")
    print(f"Model path: {relative_model_path}")
    print(f"Threshold: {threshold}")
    print(
        f"Audit saved to: {EXECUTION_AUDIT_PATH}"
    )


# --------------------------------------------------
# MAIN
# --------------------------------------------------

if __name__ == "__main__":
    execute_deployment()

# Recheck the registered model's authorization
# Download the exact MLflow version and validate its inference behaviour
# Save a versioned .joblib artifact and write the active manifest atomically
# Update MLflow deployment tags only after activation succeeds