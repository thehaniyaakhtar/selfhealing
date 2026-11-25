
import json
import logging
import uuid

from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field


# --------------------------------------------------
# PROJECT CONFIGURATION
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "random_forest_candidate.joblib"
)

LOG_DIRECTORY = PROJECT_ROOT / "logs"
PREDICTION_LOG_PATH = LOG_DIRECTORY / "predictions.jsonl"

MODEL_VERSION = "random-forest-candidate-v1"
DECISION_THRESHOLD = 0.395

FEATURE_COLUMNS = [
    "Time",
    "Amount",
    *[f"V{i}" for i in range(1, 29)]
]


# --------------------------------------------------
# LOGGING CONFIGURATION
# --------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)

LOG_DIRECTORY.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# LOAD MODEL
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
# PREDICTION PERSISTENCE
# --------------------------------------------------

def save_prediction(record: dict) -> bool:
    """
    Append one prediction record to the JSONL log.

    Each line is an independent JSON object.
    Returns True if the record was written successfully.
    """

    try:
        with open(
            PREDICTION_LOG_PATH,
            "a",
            encoding="utf-8"
        ) as log_file:

            log_file.write(
                json.dumps(record, allow_nan=False) + "\n"
            )

        return True

    except (OSError, ValueError, TypeError):
        logger.exception("Could not persist prediction.")
        return False


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

    transaction_data = transaction.model_dump()

    input_df = pd.DataFrame(
        [transaction_data],
        columns=FEATURE_COLUMNS
    )

    probabilities = model.predict_proba(input_df)[0]

    fraud_class_index = list(model.classes_).index(1)

    fraud_probability = float(
        probabilities[fraud_class_index]
    )

    predicted_class = int(
        fraud_probability >= DECISION_THRESHOLD
    )

    decision = (
        "FRAUD"
        if predicted_class == 1
        else "LEGITIMATE"
    )

    # Generate a unique identifier for this prediction.
    prediction_id = str(uuid.uuid4())

    # Capture the current UTC timestamp.
    prediction_timestamp = datetime.now(
        timezone.utc
    ).isoformat()

    # Construct the persistent prediction record.
    prediction_record = {
        "prediction_id": prediction_id,
        "timestamp": prediction_timestamp,
        "features": transaction_data,
        "fraud_probability": fraud_probability,
        "decision_threshold": DECISION_THRESHOLD,
        "predicted_class": predicted_class,
        "decision": decision,
        "model_name": "Random Forest",
        "model_version": MODEL_VERSION,
        "actual_label": None
    }

    # Persist the prediction.
    prediction_logged = save_prediction(
        prediction_record
    )

    logger.info(
        "Prediction generated | id=%s | decision=%s | logged=%s",
        prediction_id,
        decision,
        prediction_logged
    )

    return {
        "prediction_id": prediction_id,
        "model_name": "Random Forest",
        "model_version": MODEL_VERSION,
        "fraud_probability": round(fraud_probability, 6),
        "decision_threshold": DECISION_THRESHOLD,
        "predicted_class": predicted_class,
        "decision": decision,
        "prediction_logged": prediction_logged
    }
