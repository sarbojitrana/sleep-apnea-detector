"""Research-only SpO2 window screening, independent of lifestyle inference."""
import joblib
import numpy as np
import pandas as pd
import sklearn

from src.data import ROOT

SENSOR_MODEL_PATH = ROOT / "models/oximetry_svm.joblib"
WINDOW_SECONDS = 60
SAMPLE_RATE_HZ = 1
MIN_VALID_FRACTION = 0.9
MAX_MISSING_RUN = 5
SENSOR_FEATURES = [
    "spo2_mean", "spo2_min", "spo2_std", "spo2_p10", "spo2_p90",
    "spo2_below_90_fraction", "spo2_range", "spo2_mean_absolute_step",
    "spo2_trend_per_minute",
]
SENSOR_DISCLAIMER = (
    "Experimental classification of 60-second SpO2 windows, not a diagnosis, "
    "apnea count, AHI, severity estimate, or real-time emergency alarm. "
    "Training data comes from a small stroke-unit population, not a validated home IoT device."
)


class InsufficientSignal(ValueError):
    pass


def signal_quality(samples):
    if len(samples) != WINDOW_SECONDS:
        raise ValueError("Each window must contain exactly 60 samples collected at 1 Hz.")
    if any(isinstance(value, (bool, np.bool_)) for value in samples):
        raise ValueError("SpO2 samples must be numbers or missing readings, not booleans.")
    try:
        values = np.asarray(samples, dtype=float)
    except (ValueError, TypeError, OverflowError) as error:
        raise ValueError("SpO2 samples must be numeric percentages or missing readings.") from error
    if values.ndim != 1:
        raise ValueError("SpO2 must be a one-dimensional sequence.")
    valid = np.isfinite(values) & (values > 0) & (values <= 100)
    longest = run = 0
    for bad in ~valid:
        run = run + 1 if bad else 0
        longest = max(longest, run)
    return values, valid, {"valid_fraction": float(valid.mean()), "longest_missing_run": int(longest)}


def extract_window_features(samples):
    values, valid, quality = signal_quality(samples)
    if quality["valid_fraction"] < MIN_VALID_FRACTION or quality["longest_missing_run"] > MAX_MISSING_RUN:
        raise InsufficientSignal("Need at least 90% valid samples and no missing run longer than 5 seconds.")
    clean = values[valid]
    paired = valid[1:] & valid[:-1]
    steps = np.abs(np.diff(values))[paired]
    time = np.arange(WINDOW_SECONDS, dtype=float)[valid] / 60
    row = {
        "spo2_mean": float(clean.mean()), "spo2_min": float(clean.min()),
        "spo2_std": float(clean.std()), "spo2_p10": float(np.quantile(clean, 0.1)),
        "spo2_p90": float(np.quantile(clean, 0.9)),
        "spo2_below_90_fraction": float((clean < 90).mean()),
        "spo2_range": float(clean.max() - clean.min()),
        "spo2_mean_absolute_step": float(steps.mean()) if len(steps) else 0.0,
        "spo2_trend_per_minute": float(np.polyfit(time, clean, 1)[0]),
    }
    return pd.DataFrame([row], columns=SENSOR_FEATURES), quality


def load_sensor_model(path=SENSOR_MODEL_PATH):
    bundle = joblib.load(path)  # Trusted local artifacts only; never accept uploaded pickle/joblib files.
    if not isinstance(bundle, dict) or not {"metadata", "estimator"} <= bundle.keys():
        raise ValueError("Invalid oximetry model artifact. Run python sensor.py train.")
    meta = bundle["metadata"]
    if (meta.get("sensor_schema_version") != 1 or meta.get("features") != SENSOR_FEATURES
            or meta.get("sklearn_version") != sklearn.__version__
            or meta.get("window_seconds") != WINDOW_SECONDS or meta.get("sample_rate_hz") != SAMPLE_RATE_HZ
            or list(getattr(bundle["estimator"], "feature_names_in_", [])) != SENSOR_FEATURES):
        raise ValueError("Oximetry artifact is incompatible. Retrain with python sensor.py train.")
    return bundle


def predict_window(bundle, samples):
    _, _, quality = signal_quality(samples)
    try:
        features, quality = extract_window_features(samples)
    except InsufficientSignal as error:
        return {"status": "insufficient_evidence", "screen_positive": None,
                "decision_margin": None, "quality": quality, "reason": str(error)}
    margin = float(bundle["estimator"].decision_function(features)[0])
    return {"status": "raised_signal" if margin >= 0 else "lower_signal",
            "screen_positive": bool(margin >= 0), "decision_margin": margin, "quality": quality}


def screen_recording(bundle, frame):
    if set(frame.columns) != {"elapsed_seconds", "spo2"}:
        raise ValueError("CSV must contain exactly elapsed_seconds and spo2 columns.")
    if not WINDOW_SECONDS <= len(frame) <= 43200:
        raise ValueError("Provide 60–43200 samples (one minute to 12 hours at 1 Hz).")
    for column in ("elapsed_seconds", "spo2"):
        if pd.api.types.is_bool_dtype(frame[column]):
            raise ValueError(f"{column} must be numeric, not Boolean.")
    try:
        seconds = pd.to_numeric(frame["elapsed_seconds"], errors="raise").to_numpy(dtype=float)
        samples = pd.to_numeric(frame["spo2"], errors="raise").to_numpy(dtype=float)
    except (ValueError, TypeError) as error:
        raise ValueError("Both columns must be numeric; leave absent SpO2 readings blank.") from error
    if (not np.isfinite(seconds).all() or (seconds < 0).any()
            or not np.equal(seconds, np.floor(seconds)).all() or not (np.diff(seconds) == 1).all()):
        raise ValueError("elapsed_seconds must contain consecutive nonnegative whole seconds at 1 Hz.")
    results = []
    for start in range(0, len(samples) - WINDOW_SECONDS + 1, WINDOW_SECONDS):
        result = predict_window(bundle, samples[start:start + WINDOW_SECONDS])
        results.append({"start_second": int(seconds[start]),
                        "end_second_exclusive": int(seconds[start] + WINDOW_SECONDS), **result})
    return {
        "model_id": bundle["metadata"]["model_id"], "window_seconds": WINDOW_SECONDS,
        "sample_rate_hz": SAMPLE_RATE_HZ, "windows": results,
        "raised_windows": sum(r["screen_positive"] is True for r in results),
        "abstained_windows": sum(r["screen_positive"] is None for r in results),
        "trailing_samples_not_scored": len(samples) % WINDOW_SECONDS,
        "disclaimer": SENSOR_DISCLAIMER,
    }
