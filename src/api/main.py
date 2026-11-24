
import logging
from pathlib import Path

import joblib
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field


# --------------------------------------------------
# PROJECT CONFIGURATION
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = PROJECT_ROOT / "artifacts" / "random_forest_candidate.joblib"

MODEL_VERSION = "random-forest-candidate-v1"
DECISION_THRESHOLD = 0.395

FEATURE_COLUMNS = [
    "Time",
    "Amount",
    *[f"V{i}" for i in range(1, 29)]
]


# --------------------------------------------------
# LOGGING
# --------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


# --------------------------------------------------
# LOAD TRAINED MODEL
# --------------------------------------------------

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model artifact not found: {MODEL_PATH}"
    )

model = joblib.load(MODEL_PATH)

logger.info("Model loaded successfully.")
logger.info("Model version: %s", MODEL_VERSION)


# --------------------------------------------------
# API INITIALIZATION
# --------------------------------------------------

app = FastAPI(
    title="Self-Healing ML Platform",
    description="Fraud detection inference API",
    version="1.0.0"
)


# --------------------------------------------------
# INPUT VALIDATION
# --------------------------------------------------

class Transaction(BaseModel):

    model_config = ConfigDict(allow_inf_nan=False)

    Time: float
    Amount: float = Field(ge=0)

    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float


# --------------------------------------------------
# HEALTH CHECK
# --------------------------------------------------

@app.get("/health")
def health_check():

    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_version": MODEL_VERSION
    }


# --------------------------------------------------
# FRAUD PREDICTION
# --------------------------------------------------

@app.post("/predict")
def predict(transaction: Transaction):

    # Convert validated input into a DataFrame.
    transaction_data = transaction.model_dump()

    input_df = pd.DataFrame(
        [transaction_data],
        columns=FEATURE_COLUMNS
    )

    # Generate fraud probabilities using the full pipeline.
    probabilities = model.predict_proba(input_df)[0]

    # Find the probability associated with fraud class 1.
    fraud_class_index = list(model.classes_).index(1)

    fraud_probability = float(
        probabilities[fraud_class_index]
    )

    # Apply the operational decision threshold.
    predicted_class = int(
        fraud_probability >= DECISION_THRESHOLD
    )

    decision = (
        "FRAUD"
        if predicted_class == 1
        else "LEGITIMATE"
    )

    logger.info(
        "Prediction generated | model=%s | decision=%s",
        MODEL_VERSION,
        decision
    )

    return {
        "model_name": "Random Forest",
        "model_version": MODEL_VERSION,
        "fraud_probability": round(fraud_probability, 6),
        "decision_threshold": DECISION_THRESHOLD,
        "predicted_class": predicted_class,
        "decision": decision
    }
