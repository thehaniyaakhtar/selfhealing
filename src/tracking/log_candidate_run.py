
import json
from pathlib import Path

import mlflow

from src.tracking.setup_mlflow import configure_mlflow


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RETRAINING_DIR = PROJECT_ROOT / "artifacts" / "retraining"

CANDIDATE_REPORT_PATH = (
    RETRAINING_DIR / "candidate_evaluation.json"
)

BOOTSTRAP_REPORT_PATH = (
    RETRAINING_DIR / "bootstrap_validation.json"
)

PROMOTION_REPORT_PATH = (
    RETRAINING_DIR / "promotion_decision.json"
)

CANDIDATE_MODEL_PATH = (
    RETRAINING_DIR / "retrained_candidate.joblib"
)


def load_json(path: Path) -> dict:
    """Load a JSON report."""

    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found: {path}"
        )

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def log_candidate_experiment():
    """Record the existing candidate training run in MLflow."""

    print("Loading candidate experiment information...")

    candidate_report = load_json(CANDIDATE_REPORT_PATH)
    bootstrap_report = load_json(BOOTSTRAP_REPORT_PATH)
    promotion_report = load_json(PROMOTION_REPORT_PATH)

    if not CANDIDATE_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Candidate model not found: {CANDIDATE_MODEL_PATH}"
        )

    candidate_metrics = candidate_report["candidate_metrics"]
    incumbent_metrics = candidate_report["incumbent_metrics"]

    bootstrap_results = bootstrap_report["results"]

    # Configure the existing MLflow experiment.
    experiment = configure_mlflow()

    print("\nStarting MLflow experiment run...")

    with mlflow.start_run(
        run_name="retrained-random-forest-candidate-v1"
    ) as run:

        # -----------------------------------------
        # 1. Log model configuration
        # -----------------------------------------

        parameters = {
            "model_type": "RandomForestClassifier",
            "n_estimators": 200,
            "class_weight": "balanced_subsample",
            "random_state": 42,
            "decision_threshold": 0.395,
            "preprocessing": "embedded_sklearn_pipeline",
            "training_strategy": "train_plus_reference_data",
            "training_observations": 199364,
            "training_fraud_cases": 384,
            "evaluation_observations": 42721,
            "evaluation_fraud_cases": 56,
        }

        mlflow.log_params(parameters)

        # -----------------------------------------
        # 2. Log candidate metrics
        # -----------------------------------------

        for metric_name, value in candidate_metrics.items():

            if isinstance(value, (int, float)):
                mlflow.log_metric(
                    f"candidate_{metric_name}",
                    float(value),
                )

        # -----------------------------------------
        # 3. Log incumbent metrics
        # -----------------------------------------

        for metric_name, value in incumbent_metrics.items():

            if isinstance(value, (int, float)):
                mlflow.log_metric(
                    f"incumbent_{metric_name}",
                    float(value),
                )

        # -----------------------------------------
        # 4. Log model performance differences
        # -----------------------------------------

        for metric_name in [
            "precision",
            "recall",
            "f1_score",
            "pr_auc",
            "roc_auc",
        ]:

            difference = (
                candidate_metrics[metric_name]
                - incumbent_metrics[metric_name]
            )

            mlflow.log_metric(
                f"{metric_name}_improvement",
                float(difference),
            )

        # -----------------------------------------
        # 5. Log bootstrap evidence
        # -----------------------------------------

        confidence_interval = (
            bootstrap_results["confidence_interval"]
        )

        mlflow.log_metric(
            "bootstrap_pr_auc_improvement",
            float(
                bootstrap_results[
                    "observed_pr_auc_improvement"
                ]
            ),
        )

        mlflow.log_metric(
            "bootstrap_ci_lower",
            float(confidence_interval["lower_bound"]),
        )

        mlflow.log_metric(
            "bootstrap_ci_upper",
            float(confidence_interval["upper_bound"]),
        )

        mlflow.log_metric(
            "bootstrap_candidate_win_fraction",
            float(
                bootstrap_results[
                    "bootstrap_probability_candidate_wins"
                ]
            ),
        )

        # -----------------------------------------
        # 6. Log promotion outcome
        # -----------------------------------------

        promotion_status = promotion_report[
            "promotion_status"
        ]

        mlflow.set_tag(
            "promotion_status",
            promotion_status,
        )

        mlflow.set_tag(
            "candidate_type",
            "retrained_candidate",
        )

        mlflow.set_tag(
            "project",
            "self-healing-ml-platform",
        )

        mlflow.set_tag(
            "experiment_stage",
            "automated_retraining",
        )

        mlflow.set_tag(
            "bootstrap_interpretation",
            bootstrap_results["interpretation"],
        )

        # -----------------------------------------
        # 7. Log model and reports as artifacts
        # -----------------------------------------

        mlflow.log_artifact(
            str(CANDIDATE_MODEL_PATH),
            artifact_path="candidate_model",
        )

        mlflow.log_artifact(
            str(CANDIDATE_REPORT_PATH),
            artifact_path="evaluation",
        )

        mlflow.log_artifact(
            str(BOOTSTRAP_REPORT_PATH),
            artifact_path="evaluation",
        )

        mlflow.log_artifact(
            str(PROMOTION_REPORT_PATH),
            artifact_path="evaluation",
        )

        print("\n========== MLFLOW RUN SUMMARY ==========")

        print(f"Experiment ID: {experiment.experiment_id}")

        print(f"Run ID: {run.info.run_id}")

        print(f"Run name: {run.info.run_name}")

        print(f"Candidate PR-AUC: {candidate_metrics['pr_auc']:.6f}")

        print(f"Candidate ROC-AUC: {candidate_metrics['roc_auc']:.6f}")

        print(f"Promotion status: {promotion_status}")

        print("\nModel and reports logged successfully.")

    print("\nMLflow experiment run completed.")


if __name__ == "__main__":
    log_candidate_experiment()

# First run will record 