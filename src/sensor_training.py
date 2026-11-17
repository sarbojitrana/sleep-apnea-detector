"""Pinned Kaggle acquisition and patient-separated oximetry experiments."""
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
from urllib.request import Request, urlopen
import zipfile

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, GroupShuffleSplit, StratifiedGroupKFold

from src.data import ROOT
from src.evaluate import classification_metrics
from src.model import SCREENING_SCORER, build_model
from src.oximetry import (SENSOR_FEATURES, SENSOR_MODEL_PATH, WINDOW_SECONDS,
                          MIN_VALID_FRACTION, MAX_MISSING_RUN, InsufficientSignal,
                          extract_window_features)
from src.pipeline import _atomic_save
import joblib

KAGGLE_REF = "menes97/osasud-sleep-apnea-original-parquet"
KAGGLE_VERSION = 1
RAW_PATH = ROOT / "data/osasud/osasud_1hz.parquet"
EXPECTED_SHA256 = "6a685bac8a65bd52f986ab8d66547f49d73082d7555250f4e54d69c00a8a44b9"
SOURCE_DOI = "https://doi.org/10.1038/s41597-022-01272-y"
DATA_DOI = "https://doi.org/10.6084/m9.figshare.19214289.v1"
EVENTS = {"NONE", "HYPOPNEA", "APNEA-OBSTRUCTIVE", "APNEA-MIXED", "APNEA-CENTRAL"}
REQUIRED = ["patient", "timestamp_datetime", "SpO2(%)", "event"]
# Predeclared resource bound. Evaluation always uses every eligible held-out window.
MAX_TRAIN_WINDOWS_PER_PATIENT = 150
SENSOR_GRID = [
    {"svc__kernel": ["linear"], "svc__C": [0.1]},
    {"svc__kernel": ["rbf"], "svc__C": [1, 10], "svc__gamma": ["scale"]},
]


def download_dataset(path=RAW_PATH):
    path = Path(path)
    if path.exists():
        if hashlib.sha256(path.read_bytes()).hexdigest() != EXPECTED_SHA256:
            raise ValueError("Local dataset differs from pinned Kaggle version 1; use a new destination.")
        return path
    url = f"https://www.kaggle.com/api/v1/datasets/download/{KAGGLE_REF}/osasud_1hz.parquet?datasetVersionNumber={KAGGLE_VERSION}"
    with urlopen(Request(url, headers={"User-Agent": "SleepSense research dataset acquisition"}), timeout=60) as response:
        content = response.read(12_000_001)
    if len(content) > 12_000_000:
        raise ValueError("Unexpected download size; refusing this dataset response.")
    if content[:2] == b"PK":
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            info = archive.getinfo("osasud_1hz.parquet")
            if info.file_size > 12_000_000:
                raise ValueError("Unexpected decompressed dataset size.")
            content = archive.read(info)
    if hashlib.sha256(content).hexdigest() != EXPECTED_SHA256:
        raise ValueError("Kaggle file checksum mismatch; refusing changed data.")
    _atomic_save(path, lambda name: Path(name).write_bytes(content))
    return path


def prepare_windows(path=RAW_PATH):
    path = Path(path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != EXPECTED_SHA256:
        raise ValueError("Dataset checksum mismatch. Use the pinned Kaggle download.")
    # Do not even read ECG-derived HR/RR, PVCs, anomaly, or waveform channels as predictors.
    data = pd.read_parquet(path, columns=REQUIRED)
    if (data.empty or data[["patient", "timestamp_datetime", "event"]].isna().any().any()
            or not data["event"].isin(EVENTS).all()
            or data.duplicated(["patient", "timestamp_datetime"]).any()):
        raise ValueError("Dataset identifiers, timestamps or event annotations are invalid.")
    rows, audit_rows = [], []
    for patient, recording in data.groupby("patient", sort=True):
        recording = recording.sort_values("timestamp_datetime")
        times = pd.to_datetime(recording["timestamp_datetime"], errors="raise")
        if not (times.diff().dropna() == pd.Timedelta(seconds=1)).all():
            raise ValueError(f"Patient {patient} has a sampling gap; cannot form contiguous 1 Hz windows.")
        samples = recording["SpO2(%)"].to_numpy(dtype=float)
        labels = (recording["event"] != "NONE").to_numpy(dtype=bool)
        accepted = rejected = positives = rejected_positives = 0
        for start in range(0, len(recording) - WINDOW_SECONDS + 1, WINDOW_SECONDS):
            # Any annotated apnea/hypopnea second makes this window positive.
            target = int(labels[start:start + WINDOW_SECONDS].any())
            positives += target
            try:
                features, _ = extract_window_features(samples[start:start + WINDOW_SECONDS])
            except InsufficientSignal:
                rejected += 1
                rejected_positives += target
                continue
            rows.append({"patient": str(patient), "start_second": start, "target": target,
                         **features.iloc[0].to_dict()})
            accepted += 1
        audit_rows.append({
            "patient": str(patient), "seconds": len(recording), "eligible_windows": accepted,
            "abstained_windows": rejected, "all_positive_windows": positives,
            "abstained_positive_windows": rejected_positives,
            "trailing_seconds": len(recording) % WINDOW_SECONDS,
        })
    return pd.DataFrame(rows), pd.DataFrame(audit_rows)


def sample_training_rows(frame):
    positions = []
    for _, rows in frame.groupby("patient", sort=True):
        chosen = np.linspace(0, len(rows) - 1, min(len(rows), MAX_TRAIN_WINDOWS_PER_PATIENT), dtype=int)
        positions.extend(rows.iloc[chosen].index)
    return frame.loc[positions]


def tune(frame, columns):
    sampled = sample_training_rows(frame)
    search = GridSearchCV(
        build_model(), SENSOR_GRID, scoring=SCREENING_SCORER, n_jobs=1, error_score="raise",
        cv=StratifiedGroupKFold(3, shuffle=True, random_state=23),
    )
    search.fit(sampled[columns], sampled["target"], groups=sampled["patient"])
    return search.best_estimator_, search.best_params_, len(sampled)


def evaluate_scores(labels, scores):
    predictions = (scores >= 0).astype(int)
    result = classification_metrics(labels, predictions, scores)
    # Seconds/windows from the same patient are correlated; do not show a row-level CI.
    result.pop("sensitivity_interval_95")
    result["interval_note"] = "No independent-patient confidence interval estimated; window observations are correlated."
    result["report"] = classification_report(
        labels, predictions, labels=[0, 1],
        target_names=["No annotated breathing event", "Annotated apnea/hypopnea window"], zero_division=0)
    return result


def nested_evaluate(development, columns, progress=print):
    folds, oof_scores = [], np.empty(len(development))
    split = StratifiedGroupKFold(3, shuffle=True, random_state=73)
    for number, (train, valid) in enumerate(split.split(
            development[columns], development["target"], development["patient"]), start=1):
        progress(f"  Nested patient fold {number}/3")
        training, validation = development.iloc[train], development.iloc[valid]
        assert not set(training["patient"]) & set(validation["patient"])
        model, params, fitted_rows = tune(training, columns)
        scores = model.decision_function(validation[columns])
        oof_scores[valid] = scores
        folds.append({"patients": sorted(validation["patient"].unique().tolist()),
                      "fit_windows": fitted_rows, "best_params": params,
                      "metrics": evaluate_scores(validation["target"], scores)})
    return {"pooled": evaluate_scores(development["target"], oof_scores), "folds": folds}


def _coverage(audit, patients):
    selected = audit[audit["patient"].isin(patients)]
    accepted = int(selected["eligible_windows"].sum())
    rejected = int(selected["abstained_windows"].sum())
    positive = int(selected["all_positive_windows"].sum())
    rejected_positive = int(selected["abstained_positive_windows"].sum())
    return {
        "eligible_windows": accepted, "abstained_windows": rejected,
        "coverage": accepted / (accepted + rejected),
        "all_positive_windows": positive, "abstained_positive_windows": rejected_positive,
        "positive_window_coverage": (positive - rejected_positive) / positive if positive else None,
    }


def train_sensor_pipeline(path=RAW_PATH, model_path=SENSOR_MODEL_PATH, progress=print):
    frame, audit = prepare_windows(path)
    # Predeclared split seed; no search for a favourable set of patients.
    train, test = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42).split(
        frame, frame["target"], frame["patient"]))
    development, holdout = frame.iloc[train], frame.iloc[test]
    dev_patients, test_patients = sorted(development["patient"].unique()), sorted(holdout["patient"].unique())
    assert not set(dev_patients) & set(test_patients)
    experiments, main_model = {}, None
    for name, columns in [("mean_spo2_baseline", ["spo2_mean"]), ("oximetry_features", SENSOR_FEATURES)]:
        progress(f"Evaluating {name}: patient-separated nested CV")
        nested = nested_evaluate(development, columns, progress)
        model, params, fit_rows = tune(development, columns)
        experiments[name] = {
            "features": columns, "nested_development": nested,
            "holdout": evaluate_scores(holdout["target"], model.decision_function(holdout[columns])),
            "best_params": params, "fit_windows": fit_rows,
        }
        if name == "oximetry_features":
            main_model = model
    coverage = _coverage(audit, test_patients)
    report = {
        "dataset": {"kaggle_ref": KAGGLE_REF, "version": KAGGLE_VERSION,
                    "sha256": EXPECTED_SHA256, "publication": SOURCE_DOI, "original_data": DATA_DOI,
                    "patients": int(len(audit)), "seconds": int(audit["seconds"].sum())},
        "target_definition": "At least one annotated apnea/hypopnea second in a nonoverlapping 60-second window.",
        "task": "Retrospective window classification; not person-level OSA diagnosis or future-event prediction.",
        "split": {"seed": 42, "development_patients": dev_patients, "holdout_patients": test_patients},
        "quality": {"minimum_valid_fraction": MIN_VALID_FRACTION, "maximum_missing_run_seconds": MAX_MISSING_RUN,
                    "valid_spo2_percent": "(0, 100]", "imputation": "none",
                    "all_patients": _coverage(audit, audit["patient"].tolist()), "holdout": coverage},
        "per_patient_recording_audit": audit.to_dict("records"),
        "training_sampling": {"maximum_windows_per_patient": MAX_TRAIN_WINDOWS_PER_PATIENT,
                              "method": "Evenly spaced eligible windows, without using labels; evaluation uses all eligible windows."},
        "experiments": experiments,
        "limitations": [
            "Only 30 stroke-unit patients; limited generalization to other populations and IoT sensors.",
            "Hypopnea/central/mixed events are included; raised windows are not an OSA diagnosis.",
            "Quality exclusions reduce coverage; reported classification metrics apply only to eligible windows.",
            "No comparison to lifestyle-model accuracy is valid: different dataset, labels and unit of prediction.",
            "Mirror metadata checked against the source publication and source file MD5; conversion not independently re-created.",
        ],
    }
    meta = {
        "sensor_schema_version": 1, "features": SENSOR_FEATURES, "window_seconds": WINDOW_SECONDS,
        "sample_rate_hz": 1, "sklearn_version": sklearn.__version__,
        "trained_at": datetime.now(timezone.utc).isoformat(), "report": report,
    }
    meta["model_id"] = hashlib.sha256(json.dumps(meta, sort_keys=True).encode()).hexdigest()[:16]
    bundle = {"estimator": main_model, "metadata": meta}
    _atomic_save(model_path, lambda name: joblib.dump(bundle, name))
    _atomic_save(Path(model_path).with_suffix(".report.json"),
                 lambda name: Path(name).write_text(json.dumps(meta, indent=2) + "\n"))
    return bundle, report
