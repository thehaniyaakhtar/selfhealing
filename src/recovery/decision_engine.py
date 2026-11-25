# when data cchanges, how does the platform decide whether to leave the model, investigate or initiate recovery
# system detects drift n 29 features, doesnt know whether the model is actually performing worse

# 2 things to monitor
# Drift Drift
# if inputs eaching the model have changed
# transaction amounts in production may look differently from transaction amounts in refernce dataset

# model performance degradation
# is the model making worse predictions
# for feaud detection:
# precision: of all the transactions flagged, how many were actually fraud
# recall: of all the fraud cases, how many did the model catch
# F1 Score: A combined measure of precision and recall
# PR AUC: how well the model distinguishes fraud from legit transacions across different decision thresholds

# A model can experience input drift while maintaining good performance. 
# # It can also experience performance degradation without every feature showing significant drift.

# Why labels matter:
# model makes predictions before knowing the actual outcome     ss       
# if a transaction arrives, model predicts fraud, API records prediction, several days later the oucome bcomes available
# till then one cannot reliably calculate if the prediction was correct 

# Decisions taken by Engine:
# No drift and perf healthy     CONTINUE_MONITORING
# Drift detected, performance healthy       INVESTIGATE_DRIFT
# performace degraded and enough labels available       TRIGGER_RETRAINING_EVALUATION
# drift detected, insufficient labels       WAIT_FOR_LABELS

# TRIGGER_RETRAINING_EVALUATION, not DEPLOY_NEW_MODEL
# Retraining creates a candidate. 
# The candidate must still pass evaluation before it can replace the existing model.

# Establish performance policy
# need numerical rules for deciding when performance is unacceptable
# Initially use provisional operational guardrails
# If any metric falls below its guardrail, we'll classify performance as degraded and initiate a retraining evaluation.


import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np

from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.monitoring.data_validator import load_monitoring_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PRODUCTION_DATA_PATH = (
    PROJECT_ROOT / "data" / "production" / "production.csv"
)

MODEL_PATH = (
    PROJECT_ROOT / "artifacts" / "random_forest_candidate.joblib"
)

MONITORING_REPORT_PATH = (
    PROJECT_ROOT / "artifacts" / "monitoring" / "monitoring_report.json"
)

DECISION_REPORT_PATH = (
    PROJECT_ROOT / "artifacts" / "monitoring" / "recovery_decision.json"
)


# Existing model's operational prediction threshold.
DECISION_THRESHOLD = 0.395

# Provisional performance guardrails.
MINIMUM_PRECISION = 0.90
MINIMUM_RECALL = 0.70
MINIMUM_PR_AUC = 0.80

# Minimum number of actual fraud examples needed for
# a preliminary performance assessment.
MINIMUM_FRAUD_CASES = 30


def load_monitoring_report() -> dict:
    """Load the latest feature drift monitoring report."""

    if not MONITORING_REPORT_PATH.exists():
        raise FileNotFoundError(
            f"Monitoring report not found: {MONITORING_REPORT_PATH}. "
            "Run the monitoring report first."
        )

    with open(
        MONITORING_REPORT_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        report = json.load(file)

    required_fields = {"monitoring_status", "summary"}

    if not required_fields.issubset(report):
        raise ValueError(
            "Monitoring report is missing required fields."
        )

    if report["monitoring_status"] not in {
        "DRIFT_DETECTED",
        "NO_DRIFT",
    }:
        raise ValueError(
            "Monitoring report contains an unsupported status."
        )

    return report


def load_production_data():
    """Load production observations and their available labels."""

    if not PRODUCTION_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Production data not found: {PRODUCTION_DATA_PATH}"
        )

    return load_monitoring_dataset(
        file_path=PRODUCTION_DATA_PATH,
        dataset_name="production",
    )


def load_current_model():
    """Load the currently evaluated model artifact."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model artifact not found: {MODEL_PATH}"
        )

    return joblib.load(MODEL_PATH)


def evaluate_production_performance(
    model,
    production_batch,
) -> dict:
    """
    Evaluate the current model using production observations
    for which actual labels are available.
    """

    if production_batch.labels is None:
        return {
            "status": "NOT_ASSESSED",
            "reason": "Actual production labels are unavailable.",
            "metrics": None,
        }

    y_true = np.asarray(production_batch.labels, dtype=int)

    fraud_cases = int(np.sum(y_true == 1))
    legitimate_cases = int(np.sum(y_true == 0))

    if fraud_cases < MINIMUM_FRAUD_CASES:
        return {
            "status": "NOT_ASSESSED",
            "reason": (
                f"Only {fraud_cases} fraud examples are available. "
                f"At least {MINIMUM_FRAUD_CASES} are required."
            ),
            "fraud_cases": fraud_cases,
            "legitimate_cases": legitimate_cases,
            "metrics": None,
        }

    if 1 not in model.classes_:
        raise ValueError(
            "The loaded model does not contain fraud class 1."
        )

    fraud_class_index = list(model.classes_).index(1)

    probabilities = model.predict_proba(
        production_batch.features
    )[:, fraud_class_index]

    predictions = (
        probabilities >= DECISION_THRESHOLD
    ).astype(int)

    precision = precision_score(
        y_true,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        predictions,
        zero_division=0,
    )

    pr_auc = average_precision_score(
        y_true,
        probabilities,
    )

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()

    metrics = {
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "pr_auc": float(pr_auc),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_negatives": int(tn),
        "fraud_cases": fraud_cases,
        "legitimate_cases": legitimate_cases,
        "flagged_transactions": int(np.sum(predictions)),
        "decision_threshold": DECISION_THRESHOLD,
    }

    failed_guardrails = []

    if precision < MINIMUM_PRECISION:
        failed_guardrails.append("precision")

    if recall < MINIMUM_RECALL:
        failed_guardrails.append("recall")

    if pr_auc < MINIMUM_PR_AUC:
        failed_guardrails.append("pr_auc")

    if failed_guardrails:
        performance_status = "DEGRADED"
    else:
        performance_status = "HEALTHY"

    return {
        "status": performance_status,
        "reason": (
            "Performance evaluated using available production labels."
        ),
        "failed_guardrails": failed_guardrails,
        "metrics": metrics,
    }


def make_recovery_decision(
    monitoring_report: dict,
    performance_report: dict,
) -> dict:
    """
    Combine feature drift evidence and model performance
    to decide the appropriate next operational action.
    """

    drift_detected = (
        monitoring_report["monitoring_status"] == "DRIFT_DETECTED"
    )

    performance_status = performance_report["status"]

    if performance_status == "DEGRADED":
        decision = "TRIGGER_RETRAINING_EVALUATION"

        explanation = (
            "One or more model performance guardrails were not met. "
            "A retraining evaluation should be initiated. "
            "The existing model must remain active until a candidate "
            "passes independent evaluation and promotion checks."
        )

    elif performance_status == "NOT_ASSESSED":
        if drift_detected:
            decision = "WAIT_FOR_LABELS"

            explanation = (
                "Feature drift was detected, but model performance "
                "cannot currently be assessed. Obtain sufficient "
                "actual labels before deciding whether retraining "
                "is necessary."
            )
        else:
            decision = "CONTINUE_MONITORING"

            explanation = (
                "No feature drift was detected. Model performance "
                "is not currently assessable, so continue monitoring "
                "and collect actual labels."
            )

    elif drift_detected:
        decision = "INVESTIGATE_DRIFT"

        explanation = (
            "Feature drift was detected, but current model performance "
            "meets the configured guardrails. Investigate the changed "
            "feature distributions without automatically retraining."
        )

    else:
        decision = "CONTINUE_MONITORING"

        explanation = (
            "No feature drift was detected and current model "
            "performance meets the configured guardrails."
        )

    return {
        "decision": decision,
        "generated_at": datetime.now(timezone.utc).isoformat(),

        "signals": {
            "feature_drift_detected": drift_detected,
            "performance_status": performance_status,
            "failed_performance_guardrails": (
                performance_report.get("failed_guardrails", [])
            ),
        },

        "performance_policy": {
            "minimum_precision": MINIMUM_PRECISION,
            "minimum_recall": MINIMUM_RECALL,
            "minimum_pr_auc": MINIMUM_PR_AUC,
            "minimum_fraud_cases": MINIMUM_FRAUD_CASES,
            "decision_threshold": DECISION_THRESHOLD,
        },

        "performance_assessment": performance_report,

        "explanation": explanation,

        "deployment_instruction": (
            "Do not replace the currently deployed model based "
            "on this decision alone."
        ),
    }


def save_decision_report(decision_report: dict) -> None:
    """Persist the recovery decision."""

    DECISION_REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        DECISION_REPORT_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            decision_report,
            file,
            indent=4,
        )

    print(
        f"\nRecovery decision saved to: "
        f"{DECISION_REPORT_PATH}"
    )


def print_decision_summary(decision_report: dict) -> None:
    """Display the performance assessment and recovery decision."""

    performance = decision_report["performance_assessment"]

    print("\n========== PERFORMANCE ASSESSMENT ==========")
    print(f"Status: {performance['status']}")
    print(f"Reason: {performance['reason']}")

    if performance.get("metrics") is not None:
        metrics = performance["metrics"]

        print(f"\nPrecision: {metrics['precision']:.6f}")
        print(f"Recall: {metrics['recall']:.6f}")
        print(f"F1-score: {metrics['f1_score']:.6f}")
        print(f"PR-AUC: {metrics['pr_auc']:.6f}")

        print(f"Fraud cases: {metrics['fraud_cases']}")
        print(
            f"Legitimate cases: {metrics['legitimate_cases']}"
        )

        print(
            f"False positives: {metrics['false_positives']}"
        )
        print(
            f"False negatives: {metrics['false_negatives']}"
        )

    print("\n========== RECOVERY DECISION ==========")

    print(f"Decision: {decision_report['decision']}")

    print("\nExplanation:")
    print(decision_report["explanation"])

    print("\nDeployment instruction:")
    print(decision_report["deployment_instruction"])


def main():
    print("Loading monitoring report...")
    monitoring_report = load_monitoring_report()

    print("Monitoring report loaded successfully.")

    print("\nLoading production observations...")
    production_batch = load_production_data()

    print(
        f"Production observations: {production_batch.row_count}"
    )

    print(
        f"Actual labels available: "
        f"{production_batch.labels is not None}"
    )

    print("\nLoading current model...")
    model = load_current_model()

    print("Model loaded successfully.")

    print("\nEvaluating production performance...")

    performance_report = evaluate_production_performance(
        model=model,
        production_batch=production_batch,
    )

    print("\nCombining monitoring signals...")

    decision_report = make_recovery_decision(
        monitoring_report=monitoring_report,
        performance_report=performance_report,
    )

    print_decision_summary(decision_report)

    save_decision_report(decision_report)

    print("\nRecovery decision engine completed successfully.")


if __name__ == "__main__":
    main()

