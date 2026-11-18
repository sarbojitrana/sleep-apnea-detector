"""Combined SVM retaining oxygen dynamics and an honestly separated context field."""
from datetime import datetime, timezone
import hashlib
import json
import math
from numbers import Real
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.model_selection import GridSearchCV, GroupShuffleSplit, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.combined import (FEATURES as CORE_FEATURES, LIMITS, average_sensor_window,
                          build_combined_features, prepare_combined_windows, DISCLAIMER)
from src.data import ROOT
from src.pipeline import _atomic_save
from src.sensor_training import (RAW_PATH, KAGGLE_REF, KAGGLE_VERSION, EXPECTED_SHA256,
                                 SOURCE_DOI, DATA_DOI, sample_training_rows,
                                 evaluate_scores, _coverage)

PREDICTOR_PATH = ROOT / "models/predictor_svm.joblib"
FEATURES = [*CORE_FEATURES, "spo2_min", "spo2_std"]
TARGET_RECALL = 0.85
EXAMPLE = {"spo2": 97.0, "heart_rate": 65.0, "respiratory_rate": 16.0,
           "perfusion_index": 3.5, "spo2_min": 95.0, "spo2_std": 0.8, "sleep_duration": 7.0}


class SensorWeights(BaseEstimator, TransformerMixin):
    """Learn the strength of ancillary sensors using patient-separated inner CV."""
    def __init__(self, features=tuple(FEATURES), sensor_weight=1.0):
        self.features = features
        self.sensor_weight = sensor_weight

    def fit(self, X, y=None):
        if np.asarray(X).shape[1] != len(self.features):
            raise ValueError("Feature count does not match sensor weights.")
        self.n_features_in_ = len(self.features)
        return self

    def transform(self, X):
        weights = np.array([1.0 if name.startswith("spo2") else self.sensor_weight
                            for name in self.features])
        return np.asarray(X) * weights


def build_features(record):
    if not isinstance(record, dict):
        raise ValueError("Provide an object with all six sensor parameters.")
    missing = set(FEATURES) - record.keys()
    extra = record.keys() - set(FEATURES) - {"sleep_duration"}
    if missing:
        raise ValueError("Enter all sensor parameters: " + ", ".join(sorted(missing)))
    if extra:
        raise ValueError("Unsupported parameters: " + ", ".join(sorted(extra)))
    core = build_combined_features({name: record[name] for name in CORE_FEATURES})
    row = core.iloc[0].to_dict()
    for name, low, high in (("spo2_min", 1, 100), ("spo2_std", 0, 50)):
        value = record[name]
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
            raise ValueError(f"{name} must be a measured number.")
        try:
            valid = math.isfinite(value) and low <= value <= high
        except OverflowError:
            valid = False
        if not valid:
            raise ValueError(f"{name} must be between {low} and {high}.")
        row[name] = float(value)
    if row["spo2_min"] > row["spo2"]:
        raise ValueError("Lowest SpO₂ cannot exceed average SpO₂.")
    return pd.DataFrame([row], columns=FEATURES), sleep_context(record.get("sleep_duration"))


def sleep_context(duration):
    context = {}
    if duration is not None:
        if isinstance(duration, (bool, np.bool_)) or not isinstance(duration, Real):
            raise ValueError("Sleep duration must be a number of hours.")
        try:
            valid = math.isfinite(duration) and 0.5 <= duration <= 16
        except OverflowError:
            valid = False
        if not valid:
            raise ValueError("Sleep duration must be between 0.5 and 16 hours.")
        context["sleep_duration"] = float(duration)
    return context


def summarize_sensor_window(signals):
    """Apply training-equivalent quality checks and six summaries to four 1 Hz channels."""
    if not isinstance(signals, dict) or set(signals) != set(CORE_FEATURES):
        raise ValueError("Provide all four synchronized sensor channels.")
    for values in signals.values():
        if not isinstance(values, (list, tuple, np.ndarray)) or len(values) != 60:
            raise ValueError("Each sensor channel must contain 60 one-second readings.")
        if any(value is not None and (isinstance(value, (bool, np.bool_))
                                     or not isinstance(value, Real)) for value in values):
            raise ValueError("Sensor readings must be measured numbers or null for missing samples.")
    core, quality = average_sensor_window(signals)
    if core is None:
        return None, quality
    oxygen = np.asarray(signals["spo2"], dtype=float)
    low, high = LIMITS["spo2"]
    valid = np.isfinite(oxygen) & (oxygen >= low) & (oxygen <= high)
    record = {**core.iloc[0].to_dict(), "spo2_min": float(oxygen[valid].min()),
              "spo2_std": float(oxygen[valid].std())}
    return record, quality


def predict_sensor_window(bundle, signals, sleep_duration=None):
    context = sleep_context(sleep_duration)
    record, quality = summarize_sensor_window(signals)
    window = {"window_seconds": 60, "sample_rate_hz": 1, "signal_quality": quality,
              "sensor_features": record}
    if record is None:
        return {**window, "model_id": bundle["metadata"]["model_id"],
                "context": context, "context_not_used_by_model": list(context),
                "disclaimer": DISCLAIMER, "model_status": "research_not_externally_validated",
                "status": "insufficient_evidence", "screen_positive": None,
                "label": "Unable to assess these measurements", "decision_margin": None,
                "warnings": ["A sensor channel failed the 90% valid-sample or five-second gap requirement."],
                "next_step": "Check sensor contact and collect a new synchronized interval. Poor-quality data is not a negative screening result."}
    return {**predict(bundle, {**record, **context}), **window}


def threshold_for_recall(labels, scores, target=TARGET_RECALL):
    if not 0 < target <= 1:
        raise ValueError("Target recall must be in (0,1].")
    positive = np.sort(np.asarray(scores)[np.asarray(labels) == 1])
    if not len(positive) or not np.isfinite(positive).all():
        raise ValueError("Threshold selection needs finite positive-class development scores.")
    allowed_misses = math.floor((1 - target) * len(positive) + 1e-9)
    return float(positive[allowed_misses])


def tune(frame, columns):
    sampled = sample_training_rows(frame)
    estimator = Pipeline([
        ("scaler", StandardScaler()),
        ("sensor_weights", SensorWeights(features=tuple(columns))),
        ("svc", SVC(class_weight="balanced", random_state=42)),
    ])
    # Chosen before evaluation, using only development-fold labels.
    grid = {"sensor_weights__sensor_weight": [0.05, 0.25, 1.0],
            "svc__kernel": ["rbf"], "svc__C": [0.1, 1, 10]}
    cv = StratifiedGroupKFold(3, shuffle=True, random_state=23)
    search = GridSearchCV(estimator, grid, cv=cv, scoring="balanced_accuracy",
                          n_jobs=1, error_score="raise")
    search.fit(sampled[columns], sampled["target"], groups=sampled["patient"])
    scores = np.empty(len(sampled))
    for train, valid in cv.split(sampled[columns], sampled["target"], sampled["patient"]):
        fitted = clone(search.best_estimator_)
        fitted.fit(sampled.iloc[train][columns], sampled.iloc[train]["target"])
        scores[valid] = fitted.decision_function(sampled.iloc[valid][columns])
    threshold = threshold_for_recall(sampled["target"], scores)
    return search.best_estimator_, threshold, search.best_params_, len(sampled)


def metrics_at_threshold(labels, scores, threshold):
    # The shared evaluator uses zero as its boundary; shift the score consistently.
    result = evaluate_scores(labels, np.asarray(scores) - threshold)
    result["decision_threshold"] = float(threshold)
    return result


def nested_evaluate(frame, columns, progress):
    shifted_scores = np.empty(len(frame))
    folds = []
    cv = StratifiedGroupKFold(3, shuffle=True, random_state=73)
    for number, (train, valid) in enumerate(cv.split(frame[columns], frame["target"], frame["patient"]), 1):
        training, validation = frame.iloc[train], frame.iloc[valid]
        assert not set(training["patient"]) & set(validation["patient"])
        progress(f"  Nested patient fold {number}/3")
        model, threshold, params, fit_windows = tune(training, columns)
        score = model.decision_function(validation[columns])
        shifted_scores[valid] = score - threshold
        folds.append({"patients": sorted(validation["patient"].unique().tolist()),
                      "threshold": threshold, "best_params": params, "fit_windows": fit_windows,
                      "metrics": metrics_at_threshold(validation["target"], score, threshold)})
    return {"pooled": evaluate_scores(frame["target"], shifted_scores), "folds": folds}


def train_predictor(path=RAW_PATH, model_path=PREDICTOR_PATH, progress=print):
    frame, audit = prepare_combined_windows(path, include_dynamics=True)
    train, test = next(GroupShuffleSplit(1, test_size=0.2, random_state=42).split(
        frame, frame["target"], frame["patient"]))
    development, evaluation = frame.iloc[train], frame.iloc[test]
    development_patients = sorted(development["patient"].unique().tolist())
    evaluation_patients = sorted(evaluation["patient"].unique().tolist())
    assert not set(development_patients) & set(evaluation_patients)
    experiments, selected_model, selected_threshold = {}, None, None
    for name, columns in (("four_average_retrained", CORE_FEATURES), ("combined_with_oxygen_dynamics", FEATURES)):
        progress(f"Evaluating {name} using development-only threshold selection")
        nested = nested_evaluate(development, columns, progress)
        model, threshold, params, fitted_rows = tune(development, columns)
        experiments[name] = {
            "features": list(columns), "nested_development": nested,
            "holdout": metrics_at_threshold(evaluation["target"], model.decision_function(evaluation[columns]), threshold),
            "threshold": threshold, "best_params": params, "fit_windows": fitted_rows,
        }
        if name == "combined_with_oxygen_dynamics":
            selected_model, selected_threshold = model, threshold
    report = {
        "dataset": {"kaggle_ref": KAGGLE_REF, "version": KAGGLE_VERSION, "sha256": EXPECTED_SHA256,
                    "publication": SOURCE_DOI, "original_data": DATA_DOI, "patients": len(audit)},
        "measurement_period": "Sensor summaries from the same completed 60-second interval.",
        "target": "At least one expert-annotated apnea/hypopnea second in that interval; not person-level diagnosis.",
        "split": {"seed": 42, "development_patients": development_patients,
                  "holdout_patients": evaluation_patients},
        "threshold_selection": {"method": "Positive-score quantile from patient-separated out-of-fold training predictions.",
                                "target_development_recall": TARGET_RECALL,
                                "note": "Target recall is a development objective, not a guaranteed clinical or evaluation sensitivity."},
        "model_selection": "Inner patient-group CV maximizes balanced accuracy; outer validation refits tuning and threshold.",
        "experiments": experiments,
        "quality": {"holdout": _coverage(audit, evaluation_patients),
                    "all_patients": _coverage(audit, audit["patient"].tolist())},
        "context_only": {"sleep_duration": "User-authorized context only. Not recorded in OSASUD; not an estimator input or learned risk effect."},
        "sensor_transfer_note": "Source HR/RR are clinical monitor/ECG-derived scalar quantities; transfer to optical/wearable estimators is unvalidated.",
        "evaluation_note": "These patients and the fixed evaluation split were previously examined. Results are exploratory; no new external validation.",
        "baselines": {"all_negative_accuracy": float((evaluation["target"] == 0).mean()),
                      "all_positive_recall": 1.0, "all_positive_precision": float(evaluation["target"].mean())},
        "limitations": [
            "Only 30 stroke-unit patients; samples within a patient are correlated.",
            "Actual sleep duration, age, BMI, and steps are not jointly available in this file.",
            "No synthetic clinical fields or cross-dataset row joins were used.",
            "Window-level scores cannot confirm or exclude whether a person has sleep apnea.",
        ],
    }
    metadata = {
        "predictor_schema_version": 1, "features": FEATURES, "context_fields": ["sleep_duration"],
        "decision_threshold": selected_threshold, "sklearn_version": sklearn.__version__,
        "trained_at": datetime.now(timezone.utc).isoformat(), "window_seconds": 60, "report": report,
        "feature_bounds": {name: [float(development[name].min()), float(development[name].max())] for name in FEATURES},
    }
    metadata["model_id"] = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()[:16]
    bundle = {"metadata": metadata, "estimator": selected_model}
    _atomic_save(model_path, lambda name: joblib.dump(bundle, name))
    _atomic_save(Path(model_path).with_suffix(".report.json"),
                 lambda name: Path(name).write_text(json.dumps(metadata, indent=2) + "\n"))
    return bundle


def load_predictor(path=PREDICTOR_PATH):
    bundle = joblib.load(path)
    if not isinstance(bundle, dict) or not {"metadata", "estimator"} <= bundle.keys():
        raise ValueError("Invalid predictor artifact. Run python predictor.py train.")
    meta = bundle["metadata"]
    if (meta.get("predictor_schema_version") != 1 or meta.get("features") != FEATURES
            or meta.get("sklearn_version") != sklearn.__version__
            or list(getattr(bundle["estimator"], "feature_names_in_", [])) != FEATURES):
        raise ValueError("Predictor artifact is incompatible. Run python predictor.py train.")
    return bundle


def predict(bundle, record):
    features, context = build_features(record)
    metadata = bundle["metadata"]
    warnings = []
    for name, value in features.iloc[0].items():
        low, high = metadata["feature_bounds"][name]
        if not low <= value <= high:
            warnings.append(f"{name} is outside the training range ({low:.2f}–{high:.2f}).")
    base = {"model_id": metadata["model_id"], "context": context,
            "context_not_used_by_model": list(context), "disclaimer": DISCLAIMER,
            "model_status": "research_not_externally_validated", "warnings": warnings}
    if warnings:
        return {**base, "status": "insufficient_evidence", "screen_positive": None,
                "label": "Unable to assess these measurements", "decision_margin": None,
                "next_step": "Check the measurements and seek clinical advice if concerned."}
    score = float(bundle["estimator"].decision_function(features)[0])
    positive = score >= metadata["decision_threshold"]
    return {**base, "status": "positive" if positive else "negative",
            "screen_positive": bool(positive), "decision_margin": score,
            "decision_threshold": float(metadata["decision_threshold"]),
            "label": "Positive screening result" if positive else "Negative screening result",
            "next_step": ("Discuss this screening result with a clinician." if positive else
                          "This screening did not flag a breathing-event pattern. Persistent symptoms still need assessment.")}
