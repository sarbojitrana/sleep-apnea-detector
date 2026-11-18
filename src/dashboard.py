"""Presentation helpers; demo measurements never become training data."""
from copy import deepcopy
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.predictor import EXAMPLE

HISTORY_LIMIT = 100
INPUT_LABELS = {
    "spo2": "Average SpO₂ (%)", "spo2_min": "Lowest SpO₂ (%)",
    "spo2_std": "SpO₂ standard deviation (pp)", "heart_rate": "Heart rate (bpm)",
    "respiratory_rate": "Breathing rate (breaths/min)", "perfusion_index": "Perfusion index (%)",
    "sleep_duration": "Sleep duration (h, context only)",
}
METRICS = {"Accuracy": "accuracy", "Sensitivity": "recall", "Specificity": "specificity",
           "Precision": "precision", "F1 score": "f1", "ROC AUC": "roc_auc"}


def random_demo(rng=None):
    """Generate one coherent synthetic oxygen interval, without selecting a result."""
    rng = np.random.default_rng() if rng is None else rng
    seconds = np.arange(60)
    baseline = rng.uniform(96, 99)
    dip = rng.uniform(0, 8) * np.exp(-0.5 * ((seconds - rng.uniform(20, 40)) / 6) ** 2)
    oxygen = np.clip(baseline - dip + rng.normal(0, .25, 60), 85, 100)
    return {"spo2": float(oxygen.mean()), "spo2_min": float(oxygen.min()),
            "spo2_std": float(oxygen.std()), "heart_rate": float(rng.integers(55, 101)),
            "respiratory_rate": float(rng.integers(10, 24)),
            "perfusion_index": round(float(rng.uniform(.5, 8)), 2),
            "sleep_duration": round(float(rng.uniform(4, 9.5)), 1)}


def fill_inputs(state, source):
    record = dict(EXAMPLE) if source == "Example demo" else random_demo()
    state.update(record)
    state["prefill"] = {"source": source, "inputs": dict(record)}
    state.pop("prediction", None)


def input_source(state, record):
    prefill = state.get("prefill")
    if prefill is None:
        return "Manual entry"
    if record == prefill["inputs"]:
        return prefill["source"]
    return "Edited " + prefill["source"].lower()


def remember_prediction(state, record, result):
    snapshot = {"time": datetime.now(timezone.utc).isoformat(timespec="microseconds"),
                "source": input_source(state, record), "inputs": deepcopy(record),
                "result": deepcopy(result)}
    state["prediction"] = snapshot
    state["prediction_history"] = [*state.get("prediction_history", []), snapshot][-HISTORY_LIMIT:]


def clear_history(state):
    state.pop("prediction_history", None)
    state.pop("prediction", None)


def history_frame(history):
    return pd.DataFrame([{
        "Time (UTC)": row["time"], "Source": row["source"],
        "Result": row["result"]["label"],
        **{label: row["inputs"].get(key) for key, label in INPUT_LABELS.items()},
        "Model": row["result"]["model_id"],
    } for row in reversed(history)])


def performance_frame(metadata):
    experiment = metadata["report"]["experiments"]["combined_with_oxygen_dynamics"]
    return pd.DataFrame({
        "Metric": list(METRICS),
        "Six-patient evaluation": [experiment["holdout"][key] for key in METRICS.values()],
        "Nested development validation": [experiment["nested_development"]["pooled"][key]
                                          for key in METRICS.values()],
    })
