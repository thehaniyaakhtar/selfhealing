
import json
import logging
import threading

from pathlib import Path

import joblib


# --------------------------------------------------
# PROJECT CONFIGURATION
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

ACTIVE_MODEL_MANIFEST = (
    PROJECT_ROOT
    / "artifacts"
    / "deployment"
    / "active_model.json"
)

FALLBACK_MODEL_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "random_forest_candidate.joblib"
)

FALLBACK_MODEL_NAME = "Random Forest"
FALLBACK_MODEL_VERSION = "random-forest-candidate-v1"
FALLBACK_THRESHOLD = 0.395


logger = logging.getLogger(__name__)


# --------------------------------------------------
# MODEL MANAGER
# --------------------------------------------------

class ModelManager:

    def __init__(self):

        self._lock = threading.RLock()

        self._active_model = None
        self._model_name = None
        self._model_version = None
        self._decision_threshold = None

        self._manifest_mtime_ns = None

        self._initialize_model()

    # --------------------------------------------------
    # MODEL VALIDATION
    # --------------------------------------------------

    @staticmethod
    def validate_model(model):

        if not hasattr(model, "predict_proba"):
            raise ValueError(
                "Model does not support predict_proba."
            )

        if not hasattr(model, "classes_"):
            raise ValueError(
                "Model does not expose classes_."
            )

        if 1 not in list(model.classes_):
            raise ValueError(
                "Model does not contain the fraud class."
            )

    # --------------------------------------------------
    # INITIALIZATION
    # --------------------------------------------------

    def _initialize_model(self):

        if ACTIVE_MODEL_MANIFEST.exists():

            logger.info(
                "Active model manifest found."
            )

            self._reload_from_manifest()

        else:

            logger.info(
                "No active manifest found. "
                "Loading fallback incumbent model."
            )

            model = joblib.load(
                FALLBACK_MODEL_PATH
            )

            self.validate_model(model)

            self._active_model = model
            self._model_name = FALLBACK_MODEL_NAME
            self._model_version = FALLBACK_MODEL_VERSION
            self._decision_threshold = FALLBACK_THRESHOLD

            logger.info(
                "Fallback model loaded: %s",
                self._model_version
            )

    # --------------------------------------------------
    # MANIFEST LOADING
    # --------------------------------------------------

    def _reload_from_manifest(self):

        with self._lock:

            if not ACTIVE_MODEL_MANIFEST.exists():
                return

            manifest_stat = ACTIVE_MODEL_MANIFEST.stat()

            current_mtime = manifest_stat.st_mtime_ns

            if current_mtime == self._manifest_mtime_ns:
                return

            with open(
                ACTIVE_MODEL_MANIFEST,
                "r",
                encoding="utf-8"
            ) as file:

                manifest = json.load(file)

            required_fields = {
                "model_name",
                "model_version",
                "model_path",
                "decision_threshold"
            }

            if not required_fields.issubset(manifest):
                raise ValueError(
                    "Active model manifest is missing "
                    "required fields."
                )

            model_path = (
                PROJECT_ROOT
                / manifest["model_path"]
            ).resolve()

            try:
                model_path.relative_to(PROJECT_ROOT)
            except ValueError:
                raise ValueError(
                    "Model path must remain inside "
                    "the project directory."
                )

            if not model_path.is_file():
                raise FileNotFoundError(
                    f"Model artifact not found: {model_path}"
                )

            threshold = float(
                manifest["decision_threshold"]
            )

            if not 0 <= threshold <= 1:
                raise ValueError(
                    "Decision threshold must be between 0 and 1."
                )

            logger.info(
                "Loading model version %s...",
                manifest["model_version"]
            )

            # Load and validate before changing active state.
            candidate_model = joblib.load(
                model_path
            )

            self.validate_model(candidate_model)

            # Update all active model properties together.
            self._active_model = candidate_model

            self._model_name = manifest["model_name"]
            self._model_version = str(
                manifest["model_version"]
            )

            self._decision_threshold = threshold

            self._manifest_mtime_ns = current_mtime

            logger.info(
                "Active model successfully updated: %s",
                self._model_version
            )

    # --------------------------------------------------
    # ACTIVE MODEL ACCESS
    # --------------------------------------------------

    def get_active_model(self):

        with self._lock:

            if ACTIVE_MODEL_MANIFEST.exists():

                try:
                    self._reload_from_manifest()

                except Exception:

                    logger.exception(
                        "Could not load updated model. "
                        "Continuing with previous active model."
                    )

            return {
                "model": self._active_model,
                "model_name": self._model_name,
                "model_version": self._model_version,
                "decision_threshold": self._decision_threshold
            }
            
# Loading the original model if no deployment manifest exists.
# Detecting when the active manifest changes.
# Loading and validating the replacement before switching.
# Keeping the previous model active if the replacement cannot be loaded.
