# MLFLOW experiment tracking and model registry stage
# system does not have a proper, queryable history of every training run
# if 20 candidates were run over the course of several weeks, one should be able to say:
# which training dataset was used by each, hyperparameters used, metrics did each run acheive. why was a candidate rejected, model version approved for deployement

# Thus MLFlow

from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient


# Resolve the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# SQLite database for experiment metadata.
MLFLOW_DATABASE_PATH = PROJECT_ROOT / "mlflow.db"

# Directory for experiment artifacts.
MLFLOW_ARTIFACT_DIR = PROJECT_ROOT / "mlartifacts"

EXPERIMENT_NAME = "self-healing-ml-platform"


def configure_mlflow():
    """Configure MLflow using SQLite and local artifact storage."""

    # Create the artifact directory.
    MLFLOW_ARTIFACT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Construct the absolute SQLite tracking URI.
    database_uri = (
        f"sqlite:///{MLFLOW_DATABASE_PATH.resolve().as_posix()}"
    )

    # Configure MLflow's tracking backend.
    mlflow.set_tracking_uri(database_uri)

    # Create a client connected to the SQLite database.
    client = MlflowClient(
        tracking_uri=database_uri
    )

    # Check whether our experiment already exists.
    experiment = client.get_experiment_by_name(
        EXPERIMENT_NAME
    )

    if experiment is None:

        # Create the experiment with its artifact location.
        artifact_location = (
            MLFLOW_ARTIFACT_DIR.resolve().as_uri()
        )

        experiment_id = client.create_experiment(
            name=EXPERIMENT_NAME,
            artifact_location=artifact_location,
        )

        print("New MLflow experiment created.")

    else:

        # Reuse the existing experiment.
        experiment_id = experiment.experiment_id

        print("Existing MLflow experiment found.")

    # Select the experiment for subsequent MLflow runs.
    mlflow.set_experiment(
        experiment_id=experiment_id
    )

    # Retrieve its final configuration.
    experiment = client.get_experiment(
        experiment_id
    )

    print("\n========== MLFLOW CONFIGURATION ==========")

    print(f"Tracking URI: {mlflow.get_tracking_uri()}")

    print(f"Database: {MLFLOW_DATABASE_PATH.resolve()}")

    print(f"Experiment name: {experiment.name}")

    print(f"Experiment ID: {experiment.experiment_id}")

    print(f"Artifact location: {experiment.artifact_location}")

    print("\nMLflow SQLite configuration successful.")

    return experiment


if __name__ == "__main__":
    configure_mlflow()


# Retraining pipeline -> Train candidate -> Evaluate candidate 

# -> Log experiment, parameters, metrics, dataset details, model artifact -> MLFLOW
# -> Save artifacts seperately

# MLFlow -> Exp tracing, Model Registry -> Promotion Policy -> Approved/ Rejected -> Deployement

