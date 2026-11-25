# paired bootstrap validation
# what is bootstrap validation: statistical resampling technique
# instead of trusting one evaluation repeatedly create new samples from those transactions, allow observations to be selected more than once
# For each sample:
# delta PR AUC = PR AUC (candidate) - PR AUC (incumbent)

# Paired bootstrap:
# both models were tested on the same transaction
# preserve the pairing when resampling
# no independent sampling of two models predictions

# process followed:
# generate a random sampe of transaction indices
# select same indices for both models and the acctual labels
# Calculate both PR AUC scores
# subtract incumbent PR AUC from candidate ones
# Repeat 1000 times
# calculate 95% CI of the differences 

# 95% CI of the mean is a range with upper and lower number calculated from a sample
# True population mean is unknown, range describes the possible values the meean could be

# Entire interval above 0       Evidence that candidate PR AUC is higher
# Interval crosses 0        Improvement in inconclusive
# Entire interval is below 0        Evidence that candidate PR AUC is lower


import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np

from sklearn.metrics import average_precision_score

from src.monitoring.data_validator import load_monitoring_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]

EVALUATION_DATA_PATH = (
    PROJECT_ROOT / "data" / "reference" / "evaluation.csv"
)

INCUMBENT_MODEL_PATH = (
    PROJECT_ROOT / "artifacts" / "random_forest_candidate.joblib"
)

CANDIDATE_MODEL_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "retraining"
    / "retrained_candidate.joblib"
)

OUTPUT_DIR = (
    PROJECT_ROOT / "artifacts" / "retraining"
)

OUTPUT_PATH = (
    OUTPUT_DIR / "bootstrap_validation.json"
)

NUMBER_OF_BOOTSTRAPS = 1000
RANDOM_STATE = 42
CONFIDENCE_LEVEL = 0.95


def load_evaluation_data():
    """Load the same labelled evaluation observations used for comparison."""

    if not EVALUATION_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Evaluation data not found: {EVALUATION_DATA_PATH}"
        )

    batch = load_monitoring_dataset(
        file_path=EVALUATION_DATA_PATH,
        dataset_name="evaluation",
    )

    if batch.labels is None:
        raise ValueError(
            "Bootstrap validation requires actual labels."
        )

    X = batch.features
    y = np.asarray(batch.labels, dtype=int)

    if len(np.unique(y)) != 2:
        raise ValueError(
            "Evaluation data must contain both classes."
        )

    return X, y


def load_models():
    """Load the incumbent and retrained candidate separately."""

    if not INCUMBENT_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Incumbent model not found: {INCUMBENT_MODEL_PATH}"
        )

    if not CANDIDATE_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Candidate model not found: {CANDIDATE_MODEL_PATH}"
        )

    incumbent = joblib.load(INCUMBENT_MODEL_PATH)
    candidate = joblib.load(CANDIDATE_MODEL_PATH)

    return incumbent, candidate


def get_fraud_probabilities(model, X, model_name):
    """Extract the probability assigned to fraud class 1."""

    if 1 not in model.classes_:
        raise ValueError(
            f"{model_name} does not contain fraud class 1."
        )

    fraud_index = list(model.classes_).index(1)

    probabilities = model.predict_proba(X)[:, fraud_index]

    if not np.isfinite(probabilities).all():
        raise ValueError(
            f"{model_name} produced non-finite probabilities."
        )

    return probabilities


def run_paired_bootstrap(
    y_true,
    incumbent_probabilities,
    candidate_probabilities,
):
    """
    Resample the same transaction indices for both models.

    Each bootstrap iteration calculates the difference:
    candidate PR-AUC minus incumbent PR-AUC.
    """

    rng = np.random.default_rng(RANDOM_STATE)

    observation_count = len(y_true)

    differences = []

    for iteration in range(NUMBER_OF_BOOTSTRAPS):

        # Sample transaction indices with replacement.
        indices = rng.integers(
            low=0,
            high=observation_count,
            size=observation_count,
        )

        # Both models receive exactly the same sampled transactions.
        sampled_labels = y_true[indices]

        sampled_incumbent = (
            incumbent_probabilities[indices]
        )

        sampled_candidate = (
            candidate_probabilities[indices]
        )

        # Defensive check: both classes must be present.
        if len(np.unique(sampled_labels)) != 2:
            continue

        incumbent_pr_auc = average_precision_score(
            sampled_labels,
            sampled_incumbent,
        )

        candidate_pr_auc = average_precision_score(
            sampled_labels,
            sampled_candidate,
        )

        difference = (
            candidate_pr_auc - incumbent_pr_auc
        )

        differences.append(float(difference))

        if (iteration + 1) % 100 == 0:
            print(
                f"Completed {iteration + 1}/"
                f"{NUMBER_OF_BOOTSTRAPS} bootstrap samples."
            )

    if len(differences) < NUMBER_OF_BOOTSTRAPS * 0.95:
        raise RuntimeError(
            "Too many bootstrap samples were skipped. "
            "Check the evaluation dataset."
        )

    return np.asarray(differences, dtype=float)


def summarize_bootstrap(
    incumbent_pr_auc,
    candidate_pr_auc,
    differences,
):
    """Calculate the observed improvement and bootstrap interval."""

    alpha = 1.0 - CONFIDENCE_LEVEL

    lower_bound = np.percentile(
        differences,
        100 * (alpha / 2),
    )

    upper_bound = np.percentile(
        differences,
        100 * (1 - alpha / 2),
    )

    observed_difference = (
        candidate_pr_auc - incumbent_pr_auc
    )

    probability_candidate_wins = np.mean(
        differences > 0
    )

    if lower_bound > 0:
        interpretation = (
            "POSITIVE_IMPROVEMENT_SUPPORTED"
        )

    elif upper_bound < 0:
        interpretation = (
            "CANDIDATE_PERFORMANCE_LOWER"
        )

    else:
        interpretation = (
            "IMPROVEMENT_INCONCLUSIVE"
        )

    return {
        "incumbent_pr_auc": float(incumbent_pr_auc),
        "candidate_pr_auc": float(candidate_pr_auc),

        "observed_pr_auc_improvement": float(
            observed_difference
        ),

        "confidence_level": CONFIDENCE_LEVEL,

        "confidence_interval": {
            "lower_bound": float(lower_bound),
            "upper_bound": float(upper_bound),
        },

        "bootstrap_samples": int(len(differences)),

        "bootstrap_probability_candidate_wins": float(
            probability_candidate_wins
        ),

        "interpretation": interpretation,
    }


def save_report(report):
    """Save the bootstrap validation report."""

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            indent=4,
        )

    print(f"\nReport saved to: {OUTPUT_PATH}")


def main():
    print("========== PAIRED BOOTSTRAP VALIDATION ==========")

    print("\nLoading evaluation dataset...")

    X, y = load_evaluation_data()

    print(f"Evaluation observations: {len(X)}")
    print(f"Fraud cases: {int((y == 1).sum())}")
    print(f"Legitimate cases: {int((y == 0).sum())}")

    print("\nLoading models...")

    incumbent, candidate = load_models()

    print("Both models loaded successfully.")

    print("\nGenerating incumbent fraud probabilities...")

    incumbent_probabilities = get_fraud_probabilities(
        incumbent,
        X,
        "Incumbent",
    )

    print("\nGenerating candidate fraud probabilities...")

    candidate_probabilities = get_fraud_probabilities(
        candidate,
        X,
        "Candidate",
    )

    # Calculate original observed PR-AUC values.
    incumbent_pr_auc = average_precision_score(
        y,
        incumbent_probabilities,
    )

    candidate_pr_auc = average_precision_score(
        y,
        candidate_probabilities,
    )

    print("\n========== OBSERVED PERFORMANCE ==========")

    print(f"Incumbent PR-AUC: {incumbent_pr_auc:.6f}")
    print(f"Candidate PR-AUC: {candidate_pr_auc:.6f}")

    print(
        "Observed improvement: "
        f"{candidate_pr_auc - incumbent_pr_auc:+.6f}"
    )

    print("\nStarting paired bootstrap...")

    differences = run_paired_bootstrap(
        y_true=y,
        incumbent_probabilities=incumbent_probabilities,
        candidate_probabilities=candidate_probabilities,
    )

    results = summarize_bootstrap(
        incumbent_pr_auc=incumbent_pr_auc,
        candidate_pr_auc=candidate_pr_auc,
        differences=differences,
    )

    report = {
        "generated_at": datetime.now(
            timezone.utc
        ).isoformat(),

        "evaluation_dataset": (
            "data/reference/evaluation.csv"
        ),

        "evaluation_observations": int(len(y)),

        "evaluation_fraud_cases": int((y == 1).sum()),

        "method": "paired_nonparametric_bootstrap",

        "random_state": RANDOM_STATE,

        "results": results,

        "limitations": [
            (
                "The confidence interval describes resampling "
                "uncertainty conditional on this evaluation dataset."
            ),
            (
                "It does not guarantee future production performance."
            ),
            (
                "Repeated use of the same evaluation dataset can "
                "introduce selection bias."
            ),
            (
                "This interval is not a formal hypothesis-test p-value."
            ),
        ],
    }

    print("\n========== BOOTSTRAP RESULTS ==========")

    print(
        f"Observed improvement: "
        f"{results['observed_pr_auc_improvement']:+.6f}"
    )

    print(
        f"95% confidence interval: "
        f"[{results['confidence_interval']['lower_bound']:.6f}, "
        f"{results['confidence_interval']['upper_bound']:.6f}]"
    )

    print(
        "Bootstrap samples where candidate wins: "
        f"{results['bootstrap_probability_candidate_wins']:.2%}"
    )

    print(
        f"Interpretation: {results['interpretation']}"
    )

    save_report(report)

    print("\nBootstrap validation completed successfully.")


if __name__ == "__main__":
    main()

# Result is inconclusive 
# Lower Bound: -0.016173
# Upper Bound: +0.033475
# includes 0
# Thus, no difference between the incumbent and candidate
# nterval also includes negative values, meaning the incumbent could outperform the candidate on a resampled dataset.
# includes positive values too, meaning the candidate could outperform the incumbent
# don't have sufficiently clear evidence that the candidate's PR-AUC improvement is reliable

# two reasons not to promote the candidate