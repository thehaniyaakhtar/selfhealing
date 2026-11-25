# building a model candidate
# retraining is not replacing the existing model
# it means creating a second model test it and decide if its better
# it should also pass check, until then original model is deployed

# retraining pipeline:
# Build retraining dataset: combine original training dataset with newly availble labelled referenced data
# Train candidate model: Train new Random Forest without touching deployed model
# Evaluate both models: Test incumbent and candidate on the same held out evaluation dataset
# candidate passes: eligible for promotion review
# Candidate fails: Rejecct candidate
# Promotion is a separate step. Neither outcome automatically changes the deployed model.

# An important distinction about the data
# 4 relevant datasets
# train.csv: Original training observations
# drift_referece.csv: Additional labelled observations for retraining
# evaluation.csv: Independent evaluation of the new candidate
# production.csv: Historical production replay used by monitoring and recovery decisions

# for first retraining experiment, combine train.csv and drift_reference.csv
# original model was trained only using part of the original training data
# there are now additional labelled observations available
# retraining gives the model an opportunity to learn from more examples

# the evaluation dataset was used in earlier model evaluation
# independent of candidate training

# retrain same model family and config initially
# Random Forest
# 200 trees
# class_weight="balanced_subsample"
# Random state 42
# Decision threshold 0.395

# If we simultaneously change the algorithm, 
# hyperparameters, threshold and training data,  
# won't know what caused any improvement or deterioration.

# compare evaluaton metrics and save the candidate model



import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from src.monitoring.data_validator import load_monitoring_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAIN_DATA_PATH = (
    PROJECT_ROOT / "data" / "reference" / "train.csv"
)

REFERENCE_DATA_PATH = (
    PROJECT_ROOT / "data" / "reference" / "drift_reference.csv"
)

EVALUATION_DATA_PATH = (
    PROJECT_ROOT / "data" / "reference" / "evaluation.csv"
)

RECOVERY_DECISION_PATH = (
    PROJECT_ROOT / "artifacts" / "monitoring" / "recovery_decision.json"
)

INCUMBENT_MODEL_PATH = (
    PROJECT_ROOT / "artifacts" / "random_forest_candidate.joblib"
)

RETRAINING_ARTIFACT_DIR = (
    PROJECT_ROOT / "artifacts" / "retraining"
)

CANDIDATE_MODEL_PATH = (
    RETRAINING_ARTIFACT_DIR / "retrained_candidate.joblib"
)

CANDIDATE_REPORT_PATH = (
    RETRAINING_ARTIFACT_DIR / "candidate_evaluation.json"
)


# Keep the existing model's operating threshold
# so that we compare both models under the same policy.
DECISION_THRESHOLD = 0.395

RANDOM_STATE = 42
NUMBER_OF_TREES = 200

FEATURE_COLUMNS = (
    ["Time", "Amount"]
    + [f"V{i}" for i in range(1, 29)]
)

COMPONENT_COLUMNS = [
    f"V{i}" for i in range(1, 29)
]


def verify_retraining_trigger() -> dict:
    """Ensure the recovery engine has requested retraining."""

    if not RECOVERY_DECISION_PATH.exists():
        raise FileNotFoundError(
            f"Recovery decision not found: {RECOVERY_DECISION_PATH}"
        )

    with open(
        RECOVERY_DECISION_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        decision_report = json.load(file)

    decision = decision_report.get("decision")

    if decision != "TRIGGER_RETRAINING_EVALUATION":
        raise RuntimeError(
            "Retraining was not authorized by the recovery engine. "
            f"Current decision: {decision}"
        )

    print("Retraining trigger verified.")

    return decision_report


def load_dataset(path: Path, dataset_name: str):
    """Load a dataset through the existing monitoring validator."""

    if not path.exists():
        raise FileNotFoundError(
            f"{dataset_name} dataset not found: {path}"
        )

    batch = load_monitoring_dataset(
        file_path=path,
        dataset_name=dataset_name,
    )

    if batch.labels is None:
        raise ValueError(
            f"{dataset_name} dataset has no actual labels."
        )

    if list(batch.features.columns) != FEATURE_COLUMNS:
        raise ValueError(
            f"{dataset_name} dataset has an unexpected feature schema."
        )

    return batch


def build_retraining_dataset():
    """
    Combine original training data with the labelled
    reference observations.

    The evaluation dataset is deliberately excluded.
    """

    print("\nLoading original training data...")

    train_batch = load_dataset(
        TRAIN_DATA_PATH,
        "training",
    )

    print(
        f"Original training observations: "
        f"{train_batch.row_count}"
    )

    print("\nLoading additional reference data...")

    reference_batch = load_dataset(
        REFERENCE_DATA_PATH,
        "reference",
    )

    print(
        f"Additional reference observations: "
        f"{reference_batch.row_count}"
    )

    features = pd.concat(
        [
            train_batch.features,
            reference_batch.features,
        ],
        ignore_index=True,
    )

    labels = np.concatenate(
        [
            np.asarray(train_batch.labels, dtype=int),
            np.asarray(reference_batch.labels, dtype=int),
        ]
    )

    # Preserve chronological order.
    combined = features.copy()
    combined["Class"] = labels

    combined = combined.sort_values(
        by="Time",
        kind="stable",
    ).reset_index(drop=True)

    X_train = combined[FEATURE_COLUMNS]
    y_train = combined["Class"].astype(int)

    if set(y_train.unique()) != {0, 1}:
        raise ValueError(
            "Retraining data must contain both legitimate "
            "and fraud examples."
        )

    if not np.isfinite(X_train.to_numpy(dtype=float)).all():
        raise ValueError(
            "Retraining features contain non-finite values."
        )

    fraud_count = int((y_train == 1).sum())
    legitimate_count = int((y_train == 0).sum())

    print("\n========== RETRAINING DATASET ==========")
    print(f"Total observations: {len(X_train)}")
    print(f"Fraud cases: {fraud_count}")
    print(f"Legitimate cases: {legitimate_count}")
    print(
        f"Fraud rate: {fraud_count / len(y_train):.6%}"
    )

    return X_train, y_train


def build_candidate_pipeline() -> Pipeline:
    """Create the candidate's preprocessing and model pipeline."""

    amount_pipeline = Pipeline(
        steps=[
            (
                "log_transform",
                FunctionTransformer(
                    np.log1p,
                    feature_names_out="one-to-one",
                ),
            ),
            ("scaler", StandardScaler()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "time",
                StandardScaler(),
                ["Time"],
            ),
            (
                "amount",
                amount_pipeline,
                ["Amount"],
            ),
            (
                "components",
                StandardScaler(),
                COMPONENT_COLUMNS,
            ),
        ],
        remainder="drop",
    )

    classifier = RandomForestClassifier(
        n_estimators=NUMBER_OF_TREES,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


def evaluate_model(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    """Evaluate a model using the common decision threshold."""

    if 1 not in model.classes_:
        raise ValueError(
            "The model does not contain fraud class 1."
        )

    fraud_class_index = list(model.classes_).index(1)

    probabilities = model.predict_proba(
        X_test
    )[:, fraud_class_index]

    predictions = (
        probabilities >= DECISION_THRESHOLD
    ).astype(int)

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0,
    )

    pr_auc = average_precision_score(
        y_test,
        probabilities,
    )

    roc_auc = roc_auc_score(
        y_test,
        probabilities,
    )

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1],
    ).ravel()

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "pr_auc": float(pr_auc),
        "roc_auc": float(roc_auc),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_negatives": int(tn),
        "fraud_cases": int((y_test == 1).sum()),
        "legitimate_cases": int((y_test == 0).sum()),
        "flagged_transactions": int(predictions.sum()),
        "decision_threshold": DECISION_THRESHOLD,
    }


def compare_models(
    incumbent_metrics: dict,
    candidate_metrics: dict,
) -> dict:
    """Calculate candidate improvements against the incumbent."""

    metric_names = [
        "precision",
        "recall",
        "f1_score",
        "pr_auc",
        "roc_auc",
    ]

    differences = {}

    for metric in metric_names:
        differences[metric] = round(
            candidate_metrics[metric]
            - incumbent_metrics[metric],
            6,
        )

    return differences


def save_candidate_report(report: dict) -> None:
    """Save the evaluation report without changing the incumbent."""

    RETRAINING_ARTIFACT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        CANDIDATE_REPORT_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            indent=4,
        )

    print(
        f"\nCandidate evaluation report saved to: "
        f"{CANDIDATE_REPORT_PATH}"
    )


def print_metrics(name: str, metrics: dict) -> None:
    """Print a readable model evaluation summary."""

    print(f"\n========== {name} ==========")

    print(f"Precision: {metrics['precision']:.6f}")
    print(f"Recall: {metrics['recall']:.6f}")
    print(f"F1-score: {metrics['f1_score']:.6f}")
    print(f"PR-AUC: {metrics['pr_auc']:.6f}")
    print(f"ROC-AUC: {metrics['roc_auc']:.6f}")

    print(f"True positives: {metrics['true_positives']}")
    print(f"False positives: {metrics['false_positives']}")
    print(f"False negatives: {metrics['false_negatives']}")
    print(f"True negatives: {metrics['true_negatives']}")


def main():
    print("========== RETRAINING PIPELINE ==========")

    decision_report = verify_retraining_trigger()

    print(
        f"Recovery decision: {decision_report['decision']}"
    )

    # Build the new training dataset.
    X_train, y_train = build_retraining_dataset()

    # Load the held-out evaluation data.
    print("\nLoading candidate evaluation data...")

    evaluation_batch = load_dataset(
        EVALUATION_DATA_PATH,
        "evaluation",
    )

    X_test = evaluation_batch.features
    y_test = evaluation_batch.labels.astype(int)

    print(
        f"Evaluation observations: {evaluation_batch.row_count}"
    )
    print(
        f"Evaluation fraud cases: {int((y_test == 1).sum())}"
    )

    # Load the existing production model.
    print("\nLoading incumbent model...")

    if not INCUMBENT_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Incumbent model not found: {INCUMBENT_MODEL_PATH}"
        )

    incumbent_model = joblib.load(
        INCUMBENT_MODEL_PATH
    )

    print("Incumbent model loaded successfully.")

    # Evaluate the incumbent on the same evaluation dataset.
    print("\nEvaluating incumbent model...")

    incumbent_metrics = evaluate_model(
        incumbent_model,
        X_test,
        y_test,
    )

    print_metrics(
        "INCUMBENT MODEL",
        incumbent_metrics,
    )

    # Train a separate candidate.
    print("\nBuilding retrained candidate...")

    candidate_model = build_candidate_pipeline()

    print("Training candidate model...")

    candidate_model.fit(
        X_train,
        y_train,
    )

    print("Candidate training completed.")

    # Evaluate candidate on exactly the same observations.
    print("\nEvaluating retrained candidate...")

    candidate_metrics = evaluate_model(
        candidate_model,
        X_test,
        y_test,
    )

    print_metrics(
        "RETRAINED CANDIDATE",
        candidate_metrics,
    )

    # Compare both models.
    differences = compare_models(
        incumbent_metrics,
        candidate_metrics,
    )

    print("\n========== MODEL COMPARISON ==========")

    for metric, difference in differences.items():
        print(
            f"{metric}: {difference:+.6f}"
        )

    # Save candidate separately.
    RETRAINING_ARTIFACT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        candidate_model,
        CANDIDATE_MODEL_PATH,
    )

    print(
        f"\nCandidate model saved to: "
        f"{CANDIDATE_MODEL_PATH}"
    )

    report = {
        "generated_at": datetime.now(
            timezone.utc
        ).isoformat(),

        "recovery_decision": decision_report["decision"],

        "training_data": {
            "sources": [
                "data/reference/train.csv",
                "data/reference/drift_reference.csv",
            ],
            "observations": int(len(X_train)),
            "fraud_cases": int((y_train == 1).sum()),
            "legitimate_cases": int((y_train == 0).sum()),
        },

        "evaluation_data": {
            "source": "data/reference/evaluation.csv",
            "observations": int(len(X_test)),
            "fraud_cases": int((y_test == 1).sum()),
        },

        "decision_threshold": DECISION_THRESHOLD,

        "incumbent_metrics": incumbent_metrics,

        "candidate_metrics": candidate_metrics,

        "candidate_minus_incumbent": differences,

        "candidate_model_path": str(
            CANDIDATE_MODEL_PATH
        ),

        "promotion_status": "NOT_DECIDED",

        "deployment_instruction": (
            "The candidate has only been trained and evaluated. "
            "Do not replace the incumbent model until the separate "
            "promotion policy has been executed."
        ),
    }

    save_candidate_report(report)

    print("\n========== RETRAINING COMPLETE ==========")
    print("Candidate training and evaluation completed.")
    print("The incumbent model has not been modified.")
    print("Promotion has not yet been authorized.")


if __name__ == "__main__":
    main()

# the  given file:
# check the recovery engine for requested retraining
# load and combine the training datasets
# train new Random Forest
# evaluate both old and new models on same evaluaton data
# save the candidate and comparison report seperately

# RESULT:
# candidate has not demonstrated a clear operational improvement yet.
# Cadidate has higher PR AUC and lower ROC AUC
# At fixed threshold, both models make exactly same classifications


# Can AUC score change when predictions dont?
# 1. Model prodces fraud possibilities
# Apply threshold to them
# Below it: legitimate
# Above it: Fraud
# Two models can produce slightly different probabilities for every transaction but still puts every transacction on the same side of the threshold
# model artifacts accomplished:
# current incumbent model and newly trained candidate and a comparison report

