
import json
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

import pandas as pd

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

EVALUATION_PATH = (
    PROJECT_ROOT / "data" / "reference" / "evaluation.csv"
)

API_URL = "http://127.0.0.1:8000/predict"

FEATURE_COLUMNS = [
    "Time",
    "Amount",
    *[f"V{i}" for i in range(1, 29)]
]

# --------------------------------------------------
# LOAD EVALUATION DATA
# --------------------------------------------------

if not EVALUATION_PATH.exists():
    raise FileNotFoundError(
        f"Evaluation dataset not found: {EVALUATION_PATH}"
    )

df = pd.read_csv(EVALUATION_PATH)

required_columns = FEATURE_COLUMNS + ["Class"]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing columns: {missing_columns}"
    )

# --------------------------------------------------
# TEST API WITH REAL TRANSACTIONS
# --------------------------------------------------

print("\n========== API INTEGRATION TEST ==========\n")

for actual_label in [0, 1]:

    matching_rows = df[df["Class"] == actual_label]

    if matching_rows.empty:
        print(f"No transaction found for class {actual_label}")
        continue

    row = matching_rows.iloc[0]

    payload = (
        row[FEATURE_COLUMNS]
        .astype(float)
        .to_dict()
    )

    request = Request(
        API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    try:
        with urlopen(request, timeout=15) as response:

            result = json.loads(
                response.read().decode("utf-8")
            )

            print(
                f"Actual label: "
                f"{'FRAUD' if actual_label == 1 else 'LEGITIMATE'}"
            )

            print(f"HTTP status: {response.status}")
            print(f"API decision: {result['decision']}")
            print(
                f"Fraud probability: "
                f"{result['fraud_probability']}"
            )
            print(
                f"Decision threshold: "
                f"{result['decision_threshold']}"
            )
            print(f"Model version: {result['model_version']}")
            print("-" * 45)

    except HTTPError as error:
        print(f"HTTP error: {error.code}")
        print(error.read().decode("utf-8"))

    except URLError as error:
        print(f"Could not connect to API: {error.reason}")

print("\nAPI integration test completed.")