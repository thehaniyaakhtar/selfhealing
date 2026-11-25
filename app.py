
import html
import json
import subprocess
import sys
import urllib.request
from pathlib import Path

import pandas as pd
import streamlit as st


# =========================================================
# CONFIGURATION
# =========================================================

ROOT = Path(__file__).resolve().parent
API_URL = "http://127.0.0.1:8000"

EVALUATION_DATA = ROOT / "data" / "reference" / "evaluation.csv"

ARTIFACTS = {
    "monitoring": ROOT / "artifacts" / "monitoring" / "monitoring_report.json",
    "drift": ROOT / "artifacts" / "monitoring" / "drift_report.json",
    "recovery": ROOT / "artifacts" / "monitoring" / "recovery_decision.json",
    "candidate": ROOT / "artifacts" / "retraining" / "candidate_evaluation.json",
    "promotion": ROOT / "artifacts" / "retraining" / "promotion_decision.json",
    "deployment": ROOT / "artifacts" / "deployment" / "deployment_execution.json",
}

MODULES = {
    "monitoring": [
        "src.monitoring.production_loader",
        "src.monitoring.drift_detector",
        "src.monitoring.monitoring_report",
        "src.recovery.decision_engine",
    ],
    "retraining": [
        "src.retraining.candidate_trainer",
        "src.retraining.bootstrap_validation",
        "src.retraining.promotion_policy",
    ],
    "deployment": [
        "src.deployment.deployment_controller",
        "src.deployment.deployment_executor",
    ],
}


# =========================================================
# PAGE SETUP
# =========================================================

st.set_page_config(
    page_title="SELFHEAL | ML Operations",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&display=swap');

    :root {
        --bg: #080d14;
        --panel: #101923;
        --panel2: #131f2b;
        --border: #233442;
        --cyan: #42e8e0;
        --blue: #5d9dff;
        --green: #4ade9a;
        --amber: #ffbd59;
        --red: #ff647c;
        --muted: #8295a7;
        --white: #eaf3fa;
    }

    html, body, [class*="css"] {
        font-family: 'Manrope', sans-serif;
    }

    .stApp {
        background:
            radial-gradient(ellipse at 75% 0%, rgba(27, 92, 113, .16), transparent 38%),
            radial-gradient(ellipse at 0% 40%, rgba(32, 68, 112, .12), transparent 35%),
            var(--bg);
        color: var(--white);
    }

    [data-testid="stHeader"] {
        background: rgba(8,13,20,.85);
    }

    [data-testid="stSidebar"] {
        background: #0b131d;
        border-right: 1px solid var(--border);
    }

    [data-testid="stSidebar"] > div {
        padding-top: 1.5rem;
    }

    .block-container {
        max-width: 1500px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3, h4 {
        color: var(--white) !important;
        letter-spacing: -0.6px;
    }

    p, label, li {
        color: #b7c6d3;
    }

    code, pre, .mono {
        font-family: 'DM Mono', monospace !important;
    }

    .brand {
        font-size: 25px;
        font-weight: 800;
        letter-spacing: 3px;
        color: var(--white);
    }

    .brand span {
        color: var(--cyan);
    }

    .brand-sub {
        color: var(--muted);
        font-family: 'DM Mono', monospace;
        font-size: 9px;
        letter-spacing: 2px;
        margin-top: 3px;
    }

    .eyebrow {
        color: var(--cyan);
        font-family: 'DM Mono', monospace;
        font-size: 10px;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 8px;
    }

    .hero {
        background: linear-gradient(115deg, #122332 0%, #101923 55%, #10212a 100%);
        border: 1px solid #294454;
        border-radius: 18px;
        padding: 28px 30px;
        position: relative;
        overflow: hidden;
        margin-bottom: 22px;
    }

    .hero:after {
        content: "◈";
        position: absolute;
        right: 40px;
        top: -38px;
        font-size: 190px;
        color: rgba(66,232,224,.055);
    }

    .hero-title {
        font-size: 32px;
        font-weight: 800;
        color: #f2f8fc;
        letter-spacing: -1.4px;
    }

    .hero-copy {
        color: #9db2c3;
        font-size: 13px;
        max-width: 650px;
        line-height: 1.8;
        margin-top: 8px;
    }

    .tag {
        display: inline-block;
        border: 1px solid #315363;
        background: rgba(66,232,224,.07);
        color: var(--cyan);
        border-radius: 5px;
        padding: 5px 9px;
        font-family: 'DM Mono', monospace;
        font-size: 9px;
        letter-spacing: 1px;
        margin-right: 6px;
        margin-top: 13px;
    }

    .metric-card {
        background: linear-gradient(145deg, #14212d, #101923);
        border: 1px solid var(--border);
        border-radius: 13px;
        padding: 17px 18px;
        min-height: 112px;
        margin-bottom: 10px;
    }

    .metric-label {
        color: #8299aa;
        font-family: 'DM Mono', monospace;
        font-size: 9px;
        letter-spacing: 1.5px;
        text-transform: uppercase;
    }

    .metric-value {
        color: #edf7fc;
        font-size: 25px;
        font-weight: 800;
        margin-top: 10px;
        line-height: 1.2;
        overflow-wrap: anywhere;
    }

    .metric-note {
        color: #8ba0b0;
        font-size: 10px;
        margin-top: 7px;
    }

    .section-label {
        color: var(--cyan);
        font-family: 'DM Mono', monospace;
        font-size: 10px;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-top: 23px;
        margin-bottom: 12px;
    }

    .panel {
        background: #101923;
        border: 1px solid var(--border);
        border-radius: 13px;
        padding: 19px;
        margin-bottom: 13px;
    }

    .panel-title {
        color: #e6f0f7;
        font-weight: 700;
        font-size: 14px;
        margin-bottom: 5px;
    }

    .panel-subtitle {
        color: #8499aa;
        font-size: 11px;
        line-height: 1.6;
    }

    .status {
        display: inline-block;
        border-radius: 4px;
        padding: 5px 9px;
        font-family: 'DM Mono', monospace;
        font-size: 9px;
        letter-spacing: 1px;
        font-weight: 500;
    }

    .status-green {
        background: rgba(74,222,154,.12);
        color: var(--green);
        border: 1px solid rgba(74,222,154,.25);
    }

    .status-red {
        background: rgba(255,100,124,.12);
        color: var(--red);
        border: 1px solid rgba(255,100,124,.25);
    }

    .status-amber {
        background: rgba(255,189,89,.12);
        color: var(--amber);
        border: 1px solid rgba(255,189,89,.25);
    }

    .status-blue {
        background: rgba(93,157,255,.12);
        color: var(--blue);
        border: 1px solid rgba(93,157,255,.25);
    }

    .pipeline-step {
        background: #101923;
        border: 1px solid var(--border);
        border-radius: 9px;
        padding: 13px 15px;
        margin-bottom: 8px;
    }

    .pipeline-number {
        color: var(--cyan);
        font-family: 'DM Mono', monospace;
        font-size: 10px;
    }

    .pipeline-name {
        color: #eaf3fa;
        font-size: 12px;
        font-weight: 700;
        margin-top: 4px;
    }

    .pipeline-desc {
        color: #8295a7;
        font-size: 10px;
        margin-top: 3px;
    }

    div.stButton > button {
        background: linear-gradient(100deg, #1cbbb9, #3b9edb);
        color: #06131b;
        border: 0;
        border-radius: 8px;
        font-weight: 800;
        min-height: 42px;
        transition: .2s ease;
    }

    div.stButton > button:hover {
        border: 0;
        color: #06131b;
        box-shadow: 0 0 22px rgba(66,232,224,.25);
        transform: translateY(-1px);
    }

    div[data-testid="stExpander"] {
        background: #101923;
        border: 1px solid var(--border);
        border-radius: 9px;
    }

    div[data-testid="stDataFrame"] {
        border: 1px solid var(--border);
        border-radius: 9px;
    }

    .stSelectbox div[data-baseweb="select"] > div {
        background: #111e29;
        border-color: #2a414f;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 1px solid var(--border);
    }

    .stTabs [data-baseweb="tab"] {
        color: #91a5b5;
        font-size: 12px;
    }

    .stTabs [aria-selected="true"] {
        color: var(--cyan) !important;
    }

    hr {
        border-color: var(--border);
    }

    [data-testid="stMetric"] {
        background: #101923;
        border: 1px solid var(--border);
        border-radius: 11px;
        padding: 13px;
    }

    [data-testid="stMetricLabel"] {
        color: #8295a7;
    }

    [data-testid="stMetricValue"] {
        color: #eaf3fa;
    }

    .footer {
        color: #607587;
        font-family: 'DM Mono', monospace;
        font-size: 9px;
        letter-spacing: 1px;
        text-align: center;
        padding-top: 25px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# HELPERS
# =========================================================

def read_json(path):
    try:
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except (OSError, json.JSONDecodeError) as error:
        return {"_error": str(error)}


def nested_value(data, keys):
    if isinstance(data, dict):
        for key in keys:
            if key in data and data[key] is not None:
                return data[key]

        for value in data.values():
            found = nested_value(value, keys)
            if found is not None:
                return found

    if isinstance(data, list):
        for item in data:
            found = nested_value(item, keys)
            if found is not None:
                return found

    return None


def metric_card(label, value, note="", accent="cyan"):
    colors = {
        "cyan": "#42e8e0",
        "green": "#4ade9a",
        "amber": "#ffbd59",
        "red": "#ff647c",
        "blue": "#5d9dff",
    }
    color = colors.get(accent, colors["cyan"])

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{html.escape(str(label))}</div>
            <div class="metric-value" style="color:{color}">
                {html.escape(str(value))}
            </div>
            <div class="metric-note">{html.escape(str(note))}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_header(eyebrow, title, description=None):
    st.markdown(
        f"""
        <div class="section-label">{html.escape(eyebrow)}</div>
        <h2 style="margin-top:0">{html.escape(title)}</h2>
        """,
        unsafe_allow_html=True,
    )
    if description:
        st.caption(description)


def run_module(module_name, timeout=1800):
    try:
        result = subprocess.run(
            [sys.executable, "-m", module_name],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        return {
            "module": module_name,
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "return_code": result.returncode,
        }
    except subprocess.TimeoutExpired:
        return {
            "module": module_name,
            "success": False,
            "stdout": "",
            "stderr": f"Timed out after {timeout} seconds.",
            "return_code": -1,
        }
    except Exception as error:
        return {
            "module": module_name,
            "success": False,
            "stdout": "",
            "stderr": str(error),
            "return_code": -1,
        }


def run_sequence(name):
    results = []

    for module in MODULES[name]:
        result = run_module(module)
        results.append(result)

        if not result["success"]:
            break

    st.session_state["execution_results"] = results
    st.session_state["execution_name"] = name


def show_execution_log():
    results = st.session_state.get("execution_results", [])

    if not results:
        return

    st.markdown('<div class="section-label">EXECUTION TRACE</div>',
                unsafe_allow_html=True)

    for result in results:
        icon = "✓" if result["success"] else "✕"

        with st.expander(f"{icon}  {result['module']}"):
            if result["stdout"]:
                st.code(result["stdout"], language="text")
            if result["stderr"]:
                st.error(result["stderr"])
            st.caption(f"Exit code: {result['return_code']}")


def api_request(endpoint, method="GET", payload=None):
    body = None
    headers = {"Accept": "application/json"}

    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    request = urllib.request.Request(
        f"{API_URL}{endpoint}",
        data=body,
        headers=headers,
        method=method,
    )

    with urllib.request.urlopen(request, timeout=20) as response:
        raw = response.read().decode("utf-8")
        return {
            "status_code": response.status,
            "data": json.loads(raw) if raw else {},
        }


def api_health():
    try:
        return api_request("/health")["data"]
    except Exception:
        return None


def predict_sample(label):
    if not EVALUATION_DATA.exists():
        raise FileNotFoundError(f"Missing dataset: {EVALUATION_DATA}")

    data = pd.read_csv(EVALUATION_DATA)

    selected = data[data["Class"] == label]
    if selected.empty:
        raise ValueError(f"No sample found for class {label}")

    row = selected.iloc[0]

    columns = ["Time", "Amount"] + [f"V{i}" for i in range(1, 29)]
    payload = {column: float(row[column]) for column in columns}

    return api_request("/predict", "POST", payload)


def collect_feature_drift(report):
    """Find the feature-level KS results in the drift report."""
    if not isinstance(report, dict):
        return pd.DataFrame()

    possible_lists = []

    def walk(value):
        if isinstance(value, dict):
            if (
                "feature" in value
                and any(key in value for key in ["ks_statistic", "ks_stat"])
            ):
                possible_lists.append(value)

            for child in value.values():
                walk(child)

        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(report)

    if not possible_lists:
        return pd.DataFrame()

    frame = pd.DataFrame(possible_lists)

    if "ks_statistic" not in frame.columns and "ks_stat" in frame.columns:
        frame["ks_statistic"] = frame["ks_stat"]

    if "drift_detected" not in frame.columns:
        frame["drift_detected"] = False

    frame["ks_statistic"] = pd.to_numeric(
        frame["ks_statistic"], errors="coerce"
    )

    return frame.dropna(subset=["ks_statistic"]).sort_values(
        "ks_statistic", ascending=False
    )


def file_exists(path):
    return path.exists()


def display_json_artifact(label, path):
    data = read_json(path)

    with st.expander(label):
        if data is None:
            st.info("No artifact found.")
        else:
            st.json(data)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:
    st.markdown(
        """
        <div class="brand">SELF<span>HEAL</span></div>
        <div class="brand-sub">AUTONOMOUS ML OPERATIONS</div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    page = st.radio(
        "NAVIGATION",
        [
            "Overview",
            "Inference Lab",
            "Monitoring Centre",
            "Recovery Pipeline",
            "Deployment Control",
            "Execution Logs",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")

    health = api_health()

    if health:
        st.markdown(
            '<span class="status status-green">● API ONLINE</span>',
            unsafe_allow_html=True,
        )
        st.caption(
            f"Model: {health.get('model_version', 'Unknown')}"
        )
    else:
        st.markdown(
            '<span class="status status-red">● API OFFLINE</span>',
            unsafe_allow_html=True,
        )
        st.caption("Start FastAPI on port 8000.")

    st.markdown("---")
    st.caption("ENVIRONMENT")
    st.markdown(
        '<span class="status status-blue">LOCAL DEMONSTRATION</span>',
        unsafe_allow_html=True,
    )
    st.caption("Build 1.0.0  /  Fraud Detection")


# =========================================================
# SHARED DATA
# =========================================================

monitoring = read_json(ARTIFACTS["monitoring"])
drift = read_json(ARTIFACTS["drift"])
recovery = read_json(ARTIFACTS["recovery"])
candidate = read_json(ARTIFACTS["candidate"])
promotion = read_json(ARTIFACTS["promotion"])
deployment = read_json(ARTIFACTS["deployment"])

drift_frame = collect_feature_drift(drift)

api = api_health()

active_model = (
    api.get("model_version", "Unavailable")
    if api else "API offline"
)

monitor_status = (
    nested_value(monitoring, ["status", "monitoring_status"])
    if monitoring else None
)

recovery_status = (
    nested_value(recovery, ["decision", "recovery_decision", "action"])
    if recovery else None
)

promotion_status = (
    nested_value(promotion, ["promotion_status", "status", "decision"])
    if promotion else None
)

deployment_status = (
    deployment.get("deployment_status")
    if deployment else None
)


# =========================================================
# TOP HEADER
# =========================================================

st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">INTELLIGENT MODEL OPERATIONS / CONTROL PLANE</div>
        <div class="hero-title">Self-Healing ML Platform</div>
        <div class="hero-copy">
            Monitor. Diagnose. Recover. Protect.
            An automated fraud detection system designed to identify
            data drift, evaluate model degradation, retrain candidates
            and prevent unsafe model deployment.
        </div>
        <span class="tag">LIVE INFERENCE</span>
        <span class="tag">DRIFT DETECTION</span>
        <span class="tag">AUTOMATED RETRAINING</span>
        <span class="tag">SAFE DEPLOYMENT</span>
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# PAGE: OVERVIEW
# =========================================================

if page == "Overview":

    section_header(
        "01 / SYSTEM OVERVIEW",
        "Operations at a glance",
        "Current state of the model serving and recovery lifecycle.",
    )

    cols = st.columns(4)

    with cols[0]:
        metric_card(
            "API STATUS",
            "ONLINE" if api else "OFFLINE",
            "FastAPI inference service",
            "green" if api else "red",
        )

    with cols[1]:
        metric_card(
            "ACTIVE MODEL",
            active_model,
            "Currently serving model version",
            "cyan",
        )

    with cols[2]:
        metric_card(
            "DRIFT STATUS",
            monitor_status or "NOT RUN",
            "Latest monitoring report",
            "amber" if monitor_status == "DRIFT_DETECTED" else "green",
        )

    with cols[3]:
        metric_card(
            "RECOVERY ACTION",
            recovery_status or "NOT RUN",
            "Decision engine output",
            "amber",
        )

    section_header(
        "02 / LIFECYCLE",
        "The recovery architecture",
        "A controlled sequence from data monitoring to deployment authorization.",
    )

    steps = [
        ("01", "DATA INGESTION", "Reference and production observations"),
        ("02", "DRIFT DETECTION", "Statistical distribution comparison"),
        ("03", "RECOVERY DECISION", "Evaluate drift and model performance"),
        ("04", "CANDIDATE TRAINING", "Train a replacement model candidate"),
        ("05", "PROMOTION POLICY", "Evaluate candidate quality and confidence"),
        ("06", "DEPLOYMENT GATE", "Authorize or block model replacement"),
    ]

    cols = st.columns(3)

    for i, (number, name, description) in enumerate(steps):
        with cols[i % 3]:
            st.markdown(
                f"""
                <div class="pipeline-step">
                    <div class="pipeline-number">STAGE / {number}</div>
                    <div class="pipeline-name">{name}</div>
                    <div class="pipeline-desc">{description}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    section_header("03 / CURRENT DECISIONS", "Latest system outcomes")

    left, right = st.columns(2)

    with left:
        st.markdown(
            '<div class="panel"><div class="panel-title">Promotion policy</div>',
            unsafe_allow_html=True,
        )
        if promotion_status:
            color = "green" if str(promotion_status).upper() == "APPROVED" else "amber"
            st.markdown(
                f'<span class="status status-{color}">{html.escape(str(promotion_status).upper())}</span>',
                unsafe_allow_html=True,
            )
        else:
            st.caption("No promotion decision available.")
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown(
            '<div class="panel"><div class="panel-title">Deployment outcome</div>',
            unsafe_allow_html=True,
        )
        if deployment_status:
            color = "green" if deployment_status == "DEPLOYED" else "amber"
            st.markdown(
                f'<span class="status status-{color}">{html.escape(str(deployment_status).upper())}</span>',
                unsafe_allow_html=True,
            )
            if deployment:
                st.caption(
                    "Incumbent modified: "
                    + str(deployment.get("incumbent_modified", "Unknown"))
                )
        else:
            st.caption("No deployment execution report available.")
        st.markdown("</div>", unsafe_allow_html=True)

    section_header("04 / ARTIFACT HEALTH", "Project components")

    checks = {
        "Baseline model": ROOT / "artifacts" / "random_forest_candidate.joblib",
        "Reference profile": ROOT / "artifacts" / "monitoring" / "reference_profile.json",
        "Drift report": ARTIFACTS["drift"],
        "Recovery decision": ARTIFACTS["recovery"],
        "Candidate model": ROOT / "artifacts" / "retraining" / "retrained_candidate.joblib",
        "Promotion report": ARTIFACTS["promotion"],
    }

    rows = []
    for name, path in checks.items():
        rows.append({
            "Component": name,
            "Status": "AVAILABLE" if path.exists() else "NOT GENERATED",
            "Path": str(path.relative_to(ROOT)),
        })

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# =========================================================
# PAGE: INFERENCE LAB
# =========================================================

elif page == "Inference Lab":

    section_header(
        "02 / INFERENCE LAB",
        "Transaction intelligence",
        "Send real evaluation samples through the live FastAPI model endpoint.",
    )

    if not api:
        st.error("API is offline. Start FastAPI before running predictions.")
    else:
        st.markdown(
            '<div class="panel"><div class="panel-title">Inference endpoint</div>'
            '<div class="panel-subtitle">POST /predict · JSON transaction payload · '
            f'{html.escape(str(active_model))}</div></div>',
            unsafe_allow_html=True,
        )

        sample_type = st.radio(
            "Choose a transaction sample",
            [0, 1],
            format_func=lambda x: (
                "Legitimate transaction" if x == 0 else "Fraudulent transaction"
            ),
            horizontal=True,
        )

        if st.button("▶  EXECUTE INFERENCE", use_container_width=True):
            try:
                with st.spinner("Executing model inference..."):
                    result = predict_sample(sample_type)["data"]
                st.session_state["last_prediction"] = result
                st.session_state["last_actual_label"] = sample_type
            except Exception as error:
                st.error(str(error))

        result = st.session_state.get("last_prediction")

        if result:
            st.markdown('<div class="section-label">INFERENCE RESULT</div>',
                        unsafe_allow_html=True)

            c1, c2, c3 = st.columns(3)

            with c1:
                metric_card(
                    "MODEL DECISION",
                    result.get("decision", "Unknown"),
                    "Predicted transaction class",
                    "red" if result.get("predicted_class") == 1 else "green",
                )

            with c2:
                probability = result.get("fraud_probability")
                metric_card(
                    "FRAUD PROBABILITY",
                    f"{float(probability):.4f}" if probability is not None else "N/A",
                    "Model predicted probability",
                    "amber",
                )

            with c3:
                metric_card(
                    "DECISION THRESHOLD",
                    result.get("decision_threshold", "N/A"),
                    "Configured classification boundary",
                    "cyan",
                )

            actual = st.session_state.get("last_actual_label")
            predicted = result.get("predicted_class")

            if actual is not None and predicted is not None:
                if int(actual) == int(predicted):
                    st.success("Prediction matches the sample's actual label.")
                else:
                    st.warning("Prediction does not match the sample's actual label.")

            with st.expander("Full inference response"):
                st.json(result)


# =========================================================
# PAGE: MONITORING CENTRE
# =========================================================

elif page == "Monitoring Centre":

    section_header(
        "03 / OBSERVABILITY",
        "Data drift intelligence",
        "Statistical comparison of reference and production feature distributions.",
    )

    if drift:
        total = nested_value(
            monitoring,
            ["features_tested", "total_features_tested"],
        )
        drifted = nested_value(
            monitoring,
            ["features_drifted", "drifted_features_count"],
        )

        c1, c2, c3 = st.columns(3)

        with c1:
            metric_card(
                "FEATURES TESTED",
                total if total is not None else len(drift_frame),
                "Alert-eligible features",
                "blue",
            )

        with c2:
            metric_card(
                "DRIFTED FEATURES",
                drifted if drifted is not None else (
                    int(drift_frame["drift_detected"].sum())
                    if not drift_frame.empty else "N/A"
                ),
                "Statistical drift signals",
                "amber",
            )

        with c3:
            metric_card(
                "MONITORING STATUS",
                monitor_status or "AVAILABLE",
                "Latest monitoring outcome",
                "amber" if monitor_status == "DRIFT_DETECTED" else "green",
            )

        if not drift_frame.empty:
            st.markdown('<div class="section-label">FEATURE DRIFT RANKING</div>',
                        unsafe_allow_html=True)

            chart_frame = drift_frame[["feature", "ks_statistic"]].copy()
            chart_frame = chart_frame.set_index("feature")

            st.bar_chart(chart_frame.head(10), height=300)

            display_columns = [
                col for col in [
                    "feature",
                    "monitoring_role",
                    "ks_statistic",
                    "p_value",
                    "significance_threshold",
                    "drift_detected",
                ]
                if col in drift_frame.columns
            ]

            st.dataframe(
                drift_frame[display_columns],
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("Feature-level results were not found in the drift report.")

    else:
        st.info("No drift report exists yet. Run monitoring from the Recovery Pipeline.")

    st.markdown('<div class="section-label">MONITORING EXECUTION</div>',
                unsafe_allow_html=True)

    st.write(
        "This executes the actual data loader, drift detector, report generator "
        "and recovery decision engine."
    )

    if st.button("▶  RUN MONITORING + RECOVERY", use_container_width=True):
        with st.spinner("Analysing production data..."):
            run_sequence("monitoring")

        results = st.session_state.get("execution_results", [])

        if results and all(item["success"] for item in results):
            st.success("Monitoring and recovery sequence completed.")
            st.rerun()
        else:
            st.error("A monitoring module failed. Inspect the execution log below.")

    display_json_artifact("Monitoring report JSON", ARTIFACTS["monitoring"])
    display_json_artifact("Drift report JSON", ARTIFACTS["drift"])
    display_json_artifact("Recovery decision JSON", ARTIFACTS["recovery"])


# =========================================================
# PAGE: RECOVERY PIPELINE
# =========================================================

elif page == "Recovery Pipeline":

    section_header(
        "04 / MODEL RECOVERY",
        "Candidate retraining",
        "Train, evaluate and assess a replacement model without modifying the incumbent.",
    )

    c1, c2, c3 = st.columns(3)

    candidate_metrics = candidate.get("candidate_metrics", {}) if candidate else {}

    with c1:
        metric_card(
            "CANDIDATE PR-AUC",
            f"{candidate_metrics.get('pr_auc', 0):.4f}"
            if isinstance(candidate_metrics, dict)
            and isinstance(candidate_metrics.get("pr_auc"), (int, float))
            else "N/A",
            "Precision-recall area",
            "cyan",
        )

    with c2:
        metric_card(
            "CANDIDATE RECALL",
            f"{candidate_metrics.get('recall', 0):.4f}"
            if isinstance(candidate_metrics, dict)
            and isinstance(candidate_metrics.get("recall"), (int, float))
            else "N/A",
            "Fraud detection rate",
            "blue",
        )

    with c3:
        metric_card(
            "PROMOTION OUTCOME",
            promotion_status or "NOT RUN",
            "Candidate eligibility",
            "green" if str(promotion_status).upper() == "APPROVED" else "amber",
        )

    st.markdown('<div class="section-label">RETRAINING WORKFLOW</div>',
                unsafe_allow_html=True)

    st.markdown(
        """
        <div class="pipeline-step">
            <div class="pipeline-number">01 / TRAIN</div>
            <div class="pipeline-name">Candidate trainer</div>
            <div class="pipeline-desc">Builds a new model from the retraining dataset.</div>
        </div>
        <div class="pipeline-step">
            <div class="pipeline-number">02 / VALIDATE</div>
            <div class="pipeline-name">Bootstrap validation</div>
            <div class="pipeline-desc">Measures uncertainty in candidate performance differences.</div>
        </div>
        <div class="pipeline-step">
            <div class="pipeline-number">03 / DECIDE</div>
            <div class="pipeline-name">Promotion policy</div>
            <div class="pipeline-desc">Applies configured performance and confidence guardrails.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.warning(
        "This runs actual model training and overwrites the latest candidate "
        "evaluation artifacts. Do not repeatedly execute if not required."
    )

    if st.button("▶  EXECUTE RETRAINING PIPELINE", use_container_width=True):
        with st.spinner("Training and validating candidate model..."):
            run_sequence("retraining")

        results = st.session_state.get("execution_results", [])

        if results and all(item["success"] for item in results):
            st.success("Retraining and promotion evaluation completed.")
            st.rerun()
        else:
            st.error("The sequence stopped because a module failed.")

    if candidate:
        with st.expander("Candidate evaluation details"):
            st.json(candidate)

    if promotion:
        with st.expander("Promotion decision and checks"):
            st.json(promotion)

    display_json_artifact("Bootstrap validation report",
                          ROOT / "artifacts" / "retraining" / "bootstrap_validation.json")


# =========================================================
# PAGE: DEPLOYMENT CONTROL
# =========================================================

elif page == "Deployment Control":

    section_header(
        "05 / DEPLOYMENT GOVERNANCE",
        "Model deployment control",
        "A guarded deployment process that refuses unauthorized model promotion.",
    )

    if deployment:
        c1, c2, c3 = st.columns(3)

        with c1:
            metric_card(
                "DEPLOYMENT STATUS",
                deployment.get("deployment_status", "UNKNOWN"),
                "Latest deployment execution",
                "green" if deployment.get("deployment_status") == "DEPLOYED" else "amber",
            )

        with c2:
            metric_card(
                "EXECUTED",
                deployment.get("deployment_executed", False),
                "Activation result",
                "green" if deployment.get("deployment_executed") else "amber",
            )

        with c3:
            modified = deployment.get("incumbent_modified", False)
            metric_card(
                "INCUMBENT MODIFIED",
                modified,
                "Production model protection",
                "red" if modified else "green",
            )

    else:
        st.info("No deployment execution report found.")

    st.markdown('<div class="section-label">AUTHORIZATION CHECKS</div>',
                unsafe_allow_html=True)

    if deployment and isinstance(deployment.get("deployment_checks"), dict):
        checks = deployment["deployment_checks"]

        check_rows = []
        for name, passed in checks.items():
            check_rows.append({
                "Authorization check": name.replace("_", " ").upper(),
                "Result": "PASS" if passed else "FAIL",
            })

        st.dataframe(
            pd.DataFrame(check_rows),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.caption("Run the deployment workflow to generate authorization checks.")

    st.warning(
        "The deployment workflow uses the existing MLflow registry and safety "
        "policy. It will not bypass a rejected candidate."
    )

    if st.button("▶  RUN DEPLOYMENT SAFETY GATE", use_container_width=True):
        with st.spinner("Verifying model authorization..."):
            run_sequence("deployment")

        results = st.session_state.get("execution_results", [])

        if results and all(item["success"] for item in results):
            st.success("Deployment safety sequence completed.")
            st.rerun()
        else:
            st.error("A deployment module failed. Inspect the execution log.")

    if deployment:
        with st.expander("Full deployment audit"):
            st.json(deployment)


# =========================================================
# PAGE: EXECUTION LOGS
# =========================================================

elif page == "Execution Logs":

    section_header(
        "06 / SYSTEM TRACE",
        "Execution console",
        "Inspect output from real Python module executions.",
    )

    results = st.session_state.get("execution_results", [])

    if results:
        for result in results:
            icon = "✓" if result["success"] else "✕"

            st.markdown(
                f"""
                <div class="panel">
                    <div class="panel-title">{icon} {html.escape(result["module"])}</div>
                    <div class="panel-subtitle">
                        Exit code: {result["return_code"]}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if result["stdout"]:
                st.code(result["stdout"], language="text")

            if result["stderr"]:
                st.error(result["stderr"])
    else:
        st.info("No module executions recorded in this dashboard session.")

    st.markdown('<div class="section-label">ARTIFACT INSPECTOR</div>',
                unsafe_allow_html=True)

    for label, path in [
        ("Monitoring report", ARTIFACTS["monitoring"]),
        ("Drift report", ARTIFACTS["drift"]),
        ("Recovery decision", ARTIFACTS["recovery"]),
        ("Candidate evaluation", ARTIFACTS["candidate"]),
        ("Promotion decision", ARTIFACTS["promotion"]),
        ("Deployment execution", ARTIFACTS["deployment"]),
    ]:
        display_json_artifact(label, path)


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
    <div class="footer">
        SELFHEAL / AUTONOMOUS ML OPERATIONS / FRAUD DETECTION
        <br><br>
        MONITORING · RECOVERY · GOVERNANCE · DEPLOYMENT
    </div>
    """,
    unsafe_allow_html=True,
)
