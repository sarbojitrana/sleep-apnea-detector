"""Explicit offline training, versioned artifacts, and shared inference."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile

import joblib
import sklearn

from src.data import (DATA_PATH, FEATURES, INPUT_COLUMNS, ROOT, build_features,
                      dataset_audit, load_dataset, split_data, validate_record)
from src.evaluate import cross_validate_model, evaluate_model
from src.model import train_tuned_model
from src.utils import DISCLAIMER, reported_symptoms

MODEL_PATH = ROOT / "models/apnea_svm.joblib"
SCHEMA_VERSION = 2


def _atomic_save(path, writer):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as temporary:
        name = temporary.name
    try:
        writer(name)
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def train_pipeline(data_path=DATA_PATH, model_path=MODEL_PATH, save=True):
    X, y = load_dataset(data_path)
    X_train, X_test, y_train, y_test = split_data(X, y)
    # Neither tuning nor nested development CV sees the holdout.
    nested = cross_validate_model(X_train, y_train)
    estimator, best_params = train_tuned_model(X_train, y_train)
    metrics = evaluate_model(estimator, X_test, y_test)
    metrics.update({
        "best_params": best_params, "nested_development_cv": nested,
        "dataset": dataset_audit(X, y), "training": dataset_audit(X_train, y_train),
        "holdout": dataset_audit(X_test, y_test),
        "evaluation_method": "Fixed grouped holdout; grouped nested CV on development rows only; F2 tuning; zero-margin threshold.",
        "interval_note": "Row-level descriptive interval; duplicates reduce independence. Not clinical sensitivity.",
        "baseline_all_negative_accuracy": float((y_test == 0).mean()),
        "baseline_all_positive_recall": 1.0,
        "baseline_all_positive_precision": float(y_test.mean()),
    })
    metadata = {
        "schema_version": SCHEMA_VERSION, "features": FEATURES,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "sklearn_version": sklearn.__version__,
        "dataset_sha256": hashlib.sha256(Path(data_path).read_bytes()).hexdigest(),
        "decision_threshold": 0.0,
        "feature_bounds": {column: [float(X_train[column].min()), float(X_train[column].max())]
                           for column in FEATURES},
        "feature_values": {column: sorted(X_train[column].unique().tolist())
                           for column in ("Gender", "BMI Category")},
        "metrics": metrics,
    }
    metadata["model_id"] = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()[:16]
    bundle = {"estimator": estimator, "metadata": metadata}
    if save:
        _atomic_save(model_path, lambda name: joblib.dump(bundle, name))
        _atomic_save(Path(model_path).with_suffix(".report.json"),
                     lambda name: Path(name).write_text(json.dumps(metadata, indent=2) + "\n"))
    return bundle, metrics


def load_model(model_path=MODEL_PATH):
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(f"No trained model at {path}. Run `python app.py train` first.")
    # joblib is executable serialization: load only your own trusted artifact.
    bundle = joblib.load(path)
    if not isinstance(bundle, dict) or set(bundle) != {"estimator", "metadata"}:
        raise ValueError("Legacy or invalid model artifact. Run `python app.py train` to rebuild it.")
    metadata = bundle["metadata"]
    if metadata.get("schema_version") != SCHEMA_VERSION or metadata.get("features") != FEATURES:
        raise ValueError("Model input schema changed. Run `python app.py train`.")
    if metadata.get("sklearn_version") != sklearn.__version__:
        raise ValueError("Model was trained with a different scikit-learn version. Retrain in this environment.")
    if list(getattr(bundle["estimator"], "feature_names_in_", [])) != FEATURES:
        raise ValueError("Artifact estimator does not match the input schema. Retrain it.")
    return bundle


def predict(bundle, record):
    clean = validate_record(record)
    features = build_features(clean)
    metadata = bundle["metadata"]
    warnings = []
    for key, column in INPUT_COLUMNS.items():
        value = float(features.iloc[0][column])
        low, high = metadata["feature_bounds"][column]
        supported = (value in metadata["feature_values"][column]
                     if column in metadata["feature_values"] else low <= value <= high)
        if not supported:
            warnings.append(f"{key} is outside the training data support ({low:g}–{high:g}).")
    symptoms = reported_symptoms(clean)
    # Range checks are a minimum guard, not a complete distribution-shift detector.
    margin = None if warnings else float(bundle["estimator"].decision_function(features)[0])
    flagged = None if margin is None else bool(margin >= metadata["decision_threshold"])
    status = "insufficient_evidence" if flagged is None else "raised_signal" if flagged else "lower_signal"
    follow_up = bool(symptoms or flagged is True)
    return {
        "status": status, "screen_positive": flagged,
        "decision_margin": margin, "model_id": metadata["model_id"],
        "reported_symptoms": symptoms, "warnings": warnings,
        "follow_up_recommended": follow_up,
        "next_step": (
            "Discuss these findings and symptoms with a clinician; ask whether a sleep assessment is appropriate."
            if follow_up else
            "The model cannot assess these inputs. Check the measurements and seek clinical advice if concerned."
            if warnings else
            "A lower signal does not exclude apnea. Seek clinical advice if symptoms or concerns persist."
        ),
        "disclaimer": DISCLAIMER,
    }
