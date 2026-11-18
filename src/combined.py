"""Single SVM over jointly recorded IoT-measurable quantities."""
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
from sklearn.model_selection import GroupShuffleSplit

from src.data import ROOT
from src.oximetry import WINDOW_SECONDS, MIN_VALID_FRACTION, MAX_MISSING_RUN
from src.pipeline import _atomic_save
from src.sensor_training import (RAW_PATH, EXPECTED_SHA256, KAGGLE_REF, KAGGLE_VERSION,
                                 SOURCE_DOI, DATA_DOI, EVENTS, tune, nested_evaluate,
                                 evaluate_scores, _coverage)

COMBINED_MODEL_PATH = ROOT / "models/combined_svm.joblib"
COLUMNS = {"spo2": "SpO2(%)", "heart_rate": "HR(bpm)",
           "respiratory_rate": "RR(rpm)", "perfusion_index": "PI(%)"}
FEATURES = list(COLUMNS)
LIMITS = {"spo2": (1, 100), "heart_rate": (30, 220),
          "respiratory_rate": (1, 80), "perfusion_index": (0.01, 30)}
EXAMPLE = {"spo2": 97.0, "heart_rate": 65.0, "respiratory_rate": 16.0, "perfusion_index": 3.5}
DISCLAIMER = "Screening prototype only. A positive result is not a diagnosis; a negative result does not rule out sleep apnea."
VALIDATION_WARNING = "The current combined model has poor measured sensitivity and is not reliable for health decisions."


def build_combined_features(record):
    if not isinstance(record, dict):
        raise ValueError("Provide one object containing all four measurements.")
    missing, extra = set(FEATURES) - record.keys(), record.keys() - set(FEATURES)
    if missing:
        raise ValueError("Enter all measurements: " + ", ".join(sorted(missing)))
    if extra:
        raise ValueError("Unsupported measurements: " + ", ".join(sorted(extra)))
    clean = {}
    for name, (low, high) in LIMITS.items():
        value = record[name]
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
            raise ValueError(f"{name} must be a measured number.")
        try:
            finite = math.isfinite(value)
        except OverflowError:
            finite = False
        if not finite or not low <= value <= high:
            raise ValueError(f"{name} must be a finite number between {low} and {high}.")
        clean[name] = float(value)
    return pd.DataFrame([clean], columns=FEATURES)


def average_sensor_window(signals):
    """Same 60-second interval for all channels; no made-up missing measurements."""
    if set(signals) != set(FEATURES):
        raise ValueError("All four sensor channels are required.")
    averages, quality = {}, {}
    for name, samples in signals.items():
        if len(samples) != WINDOW_SECONDS:
            raise ValueError("Each channel must contain 60 one-second samples.")
        if any(isinstance(value, (bool, np.bool_)) for value in samples):
            raise ValueError("Sensor samples must be measured numbers, not booleans.")
        values = np.asarray(samples, dtype=float)
        if values.ndim != 1:
            raise ValueError("Sensor channels must be one-dimensional.")
        low, high = LIMITS[name]
        valid = np.isfinite(values) & (values >= low) & (values <= high)
        longest = run = 0
        for missing in ~valid:
            run = run + 1 if missing else 0
            longest = max(longest, run)
        quality[name] = {"valid_fraction": float(valid.mean()), "longest_missing_run": longest}
        if valid.mean() < MIN_VALID_FRACTION or longest > MAX_MISSING_RUN:
            return None, quality
        averages[name] = float(values[valid].mean())
    return build_combined_features(averages), quality


def prepare_combined_windows(path=RAW_PATH, include_dynamics=False):
    if hashlib.sha256(Path(path).read_bytes()).hexdigest() != EXPECTED_SHA256:
        raise ValueError("Use the pinned Kaggle dataset; checksum mismatch.")
    data = pd.read_parquet(path, columns=["patient", "timestamp_datetime", "event", *COLUMNS.values()])
    if (data[["patient", "timestamp_datetime", "event"]].isna().any().any()
            or not data["event"].isin(EVENTS).all()
            or data.duplicated(["patient", "timestamp_datetime"]).any()):
        raise ValueError("Invalid patient identifiers, timestamps or event labels.")
    rows, audit = [], []
    for patient, frame in data.groupby("patient", sort=True):
        frame = frame.sort_values("timestamp_datetime")
        times = pd.to_datetime(frame["timestamp_datetime"])
        if not (times.diff().dropna() == pd.Timedelta(seconds=1)).all():
            raise ValueError("Patient recordings must have contiguous one-second samples.")
        signals = {name: frame[column].to_numpy(dtype=float) for name, column in COLUMNS.items()}
        target = (frame["event"] != "NONE").to_numpy(dtype=bool)
        count = {"patient": str(patient), "seconds": len(frame), "eligible_windows": 0,
                 "abstained_windows": 0, "all_positive_windows": 0,
                 "abstained_positive_windows": 0, "trailing_seconds": len(frame) % WINDOW_SECONDS}
        for start in range(0, len(frame) - WINDOW_SECONDS + 1, WINDOW_SECONDS):
            y = int(target[start:start + WINDOW_SECONDS].any())
            count["all_positive_windows"] += y
            features, _ = average_sensor_window(
                {name: values[start:start + WINDOW_SECONDS] for name, values in signals.items()})
            if features is None:
                count["abstained_windows"] += 1
                count["abstained_positive_windows"] += y
                continue
            row = {"patient": str(patient), "start_second": start, "target": y,
                   **features.iloc[0].to_dict()}
            if include_dynamics:
                oxygen = signals["spo2"][start:start + WINDOW_SECONDS]
                valid = np.isfinite(oxygen) & (oxygen >= LIMITS["spo2"][0]) & (oxygen <= LIMITS["spo2"][1])
                row["spo2_min"] = float(oxygen[valid].min())
                row["spo2_std"] = float(oxygen[valid].std())
            rows.append(row)
            count["eligible_windows"] += 1
        audit.append(count)
    return pd.DataFrame(rows), pd.DataFrame(audit)


def train_combined_model(path=RAW_PATH, model_path=COMBINED_MODEL_PATH, progress=print):
    frame, audit = prepare_combined_windows(path)
    train, test = next(GroupShuffleSplit(1, test_size=0.2, random_state=42).split(
        frame, frame["target"], frame["patient"]))
    development, holdout = frame.iloc[train], frame.iloc[test]
    dev_patients = sorted(development["patient"].unique().tolist())
    test_patients = sorted(holdout["patient"].unique().tolist())
    assert not set(dev_patients) & set(test_patients)
    experiments, selected_model = {}, None
    for name, columns in [("spo2_only", ["spo2"]), ("combined", FEATURES)]:
        progress(f"Evaluating {name} on matched sensor windows")
        nested = nested_evaluate(development, columns, progress)
        model, params, fitted_rows = tune(development, columns)
        experiments[name] = {
            "features": columns, "fit_windows": fitted_rows, "best_params": params,
            "nested_development": nested,
            "holdout": evaluate_scores(holdout["target"], model.decision_function(holdout[columns])),
        }
        if name == "combined":
            selected_model = model
    report = {
        "dataset": {"kaggle_ref": KAGGLE_REF, "version": KAGGLE_VERSION, "sha256": EXPECTED_SHA256,
                    "publication": SOURCE_DOI, "original_data": DATA_DOI, "patients": len(audit)},
        "feature_sources": COLUMNS,
        "measurement_period": "Averages from the same completed 60-second recording interval.",
        "target": "Any annotated apnea/hypopnea second in that interval; not a person-level OSA diagnosis.",
        "split": {"seed": 42, "development_patients": dev_patients, "holdout_patients": test_patients},
        "evaluation_note": "The OSASUD cohort and holdout were previously inspected in the oxygen experiment. This is an exploratory comparison, not new external validation.",
        "sensor_transfer_note": "Source HR and RR are clinical monitor/ECG-derived scalar measurements. No ECG waveform or EEG is an input. Substituting optical pulse and a wearable breathing-rate estimate requires device validation.",
        "quality": {"minimum_valid_fraction_per_channel": MIN_VALID_FRACTION,
                    "maximum_missing_run_seconds": MAX_MISSING_RUN, "imputation": "none",
                    "all_patients": _coverage(audit, audit["patient"].tolist()),
                    "holdout": _coverage(audit, test_patients)},
        "baselines": {"all_negative_accuracy": float((holdout["target"] == 0).mean()),
                      "all_positive_recall": 1.0,
                      "all_positive_precision": float(holdout["target"].mean())},
        "per_patient_recording_audit": audit.to_dict("records"),
        "experiments": experiments,
        "limitations": [
            "Only 30 stroke-unit patients; not validated in a general home-screening population.",
            "A single spot reading is not equivalent to the average over the training interval.",
            "These are interval-level labels; the model cannot confirm whether a person has OSA.",
            "No patient/profile/lifestyle rows from unrelated datasets were joined or fabricated.",
        ],
    }
    meta = {
        "combined_schema_version": 1, "features": FEATURES, "sklearn_version": sklearn.__version__,
        "trained_at": datetime.now(timezone.utc).isoformat(), "window_seconds": WINDOW_SECONDS,
        "feature_bounds": {name: [float(development[name].min()), float(development[name].max())] for name in FEATURES},
        "report": report,
    }
    meta["model_id"] = hashlib.sha256(json.dumps(meta, sort_keys=True).encode()).hexdigest()[:16]
    bundle = {"estimator": selected_model, "metadata": meta}
    _atomic_save(model_path, lambda name: joblib.dump(bundle, name))
    _atomic_save(Path(model_path).with_suffix(".report.json"),
                 lambda name: Path(name).write_text(json.dumps(meta, indent=2) + "\n"))
    return bundle


def load_combined_model(path=COMBINED_MODEL_PATH):
    bundle = joblib.load(path)
    if not isinstance(bundle, dict) or not {"metadata", "estimator"} <= bundle.keys():
        raise ValueError("Invalid combined model. Run python combined.py train.")
    meta = bundle["metadata"]
    if (meta.get("combined_schema_version") != 1 or meta.get("features") != FEATURES
            or meta.get("sklearn_version") != sklearn.__version__
            or list(getattr(bundle["estimator"], "feature_names_in_", [])) != FEATURES):
        raise ValueError("Combined model is incompatible. Run python combined.py train.")
    return bundle


def predict_combined(bundle, record):
    features = build_combined_features(record)
    warnings = []
    for name, value in features.iloc[0].items():
        low, high = bundle["metadata"]["feature_bounds"][name]
        if not low <= value <= high:
            warnings.append(f"{name} is outside the training range ({low:.2f}–{high:.2f}).")
    if warnings:
        return {"status": "insufficient_evidence", "screen_positive": None,
                "label": "Unable to assess these measurements", "decision_margin": None,
                "next_step": "Check these measurements. Seek clinical advice if symptoms or concerns persist.",
                "warnings": warnings, "model_id": bundle["metadata"]["model_id"], "disclaimer": DISCLAIMER,
                "model_status": "experimental_low_performance", "validation_warning": VALIDATION_WARNING}
    margin = float(bundle["estimator"].decision_function(features)[0])
    positive = margin >= 0
    return {
        "status": "positive" if positive else "negative", "screen_positive": bool(positive),
        "label": "Positive screening result" if positive else "Negative screening result",
        "decision_margin": margin, "warnings": [],
        "next_step": ("Possible sleep apnea: discuss this screening result with a clinician."
                      if positive else "This screening did not flag a breathing-event pattern. Persistent symptoms still need assessment."),
        "model_id": bundle["metadata"]["model_id"], "disclaimer": DISCLAIMER,
        "model_status": "experimental_low_performance", "validation_warning": VALIDATION_WARNING,
    }
