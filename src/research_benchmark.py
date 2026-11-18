"""Development-only Random Forest comparison motivated by the reference papers.

Never writes a production model or evaluates the reserved six patients.
Run: python -m src.research_benchmark --report docs/research-benchmark.json
"""
import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold

from src.combined import prepare_combined_windows
from src.predictor import (FEATURES, PREDICTOR_PATH, TARGET_RECALL, load_predictor,
                           metrics_at_threshold, threshold_for_recall)
from src.sensor_training import RAW_PATH, evaluate_scores, sample_training_rows


def tune_forest(frame):
    sampled = sample_training_rows(frame)
    cv = StratifiedGroupKFold(3, shuffle=True, random_state=23)
    search = GridSearchCV(
        RandomForestClassifier(n_estimators=100, class_weight="balanced_subsample",
                               random_state=42, n_jobs=1),
        {"max_depth": [4, 8, None], "min_samples_leaf": [5, 15]},
        scoring="balanced_accuracy", cv=cv, n_jobs=1, error_score="raise")
    search.fit(sampled[FEATURES], sampled["target"], groups=sampled["patient"])
    scores = np.empty(len(sampled))
    for train, valid in cv.split(sampled[FEATURES], sampled["target"], sampled["patient"]):
        model = clone(search.best_estimator_)
        model.fit(sampled.iloc[train][FEATURES], sampled.iloc[train]["target"])
        scores[valid] = model.predict_proba(sampled.iloc[valid][FEATURES])[:, 1]
    return search.best_estimator_, threshold_for_recall(sampled["target"], scores), search.best_params_


def footprint(model, row, method):
    buffer = io.BytesIO()
    joblib.dump(model, buffer)
    predict = getattr(model, method)
    predict(row)
    elapsed = []
    for _ in range(50):
        start = perf_counter()
        predict(row)
        elapsed.append((perf_counter() - start) * 1000)
    return {"estimator_joblib_bytes": buffer.tell(),
            "single_interval_median_ms": float(np.median(elapsed)),
            "single_interval_p95_ms": float(np.quantile(elapsed, .95)),
            "note": "Development computer only, including pandas input handling; not an embedded-device benchmark."}


def benchmark(path=RAW_PATH):
    original_hash = hashlib.sha256(PREDICTOR_PATH.read_bytes()).hexdigest()
    bundle = load_predictor()
    previous = bundle["metadata"]["report"]
    frame, _ = prepare_combined_windows(path, include_dynamics=True)
    patients = set(previous["split"]["development_patients"])
    excluded = set(previous["split"]["holdout_patients"])
    assert not patients & excluded
    development = frame.loc[frame["patient"].isin(patients)].reset_index(drop=True)
    assert set(development["patient"]) == patients
    scores = np.empty(len(development))
    folds = []
    cv = StratifiedGroupKFold(3, shuffle=True, random_state=73)
    for number, (train, valid) in enumerate(cv.split(development[FEATURES], development["target"], development["patient"]), 1):
        training, validation = development.iloc[train], development.iloc[valid]
        assert not set(training["patient"]) & set(validation["patient"])
        print(f"Random Forest nested development fold {number}/3", flush=True)
        model, threshold, params = tune_forest(training)
        probability_score = model.predict_proba(validation[FEATURES])[:, 1]
        scores[valid] = probability_score - threshold
        folds.append({"training_patients": sorted(training["patient"].unique().tolist()),
                      "validation_patients": sorted(validation["patient"].unique().tolist()),
                      "best_params": params, "threshold": threshold,
                      "metrics": metrics_at_threshold(validation["target"], probability_score, threshold)})
    print("Fitting development-only forest for runtime and size measurements", flush=True)
    forest, _, _ = tune_forest(development)
    forest_metrics = evaluate_scores(development["target"], scores)
    svm_experiment = previous["experiments"]["combined_with_oxygen_dynamics"]
    # Reuse the recorded SVM nested folds only after confirming identical validation patients.
    for new, saved in zip(folds, svm_experiment["nested_development"]["folds"], strict=True):
        assert new["validation_patients"] == saved["patients"]
    result = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "reference": "https://doi.org/10.22214/ijraset.2023.53137",
        "question": "Does the paper's Random Forest advantage reproduce on our jointly recorded IoT summaries?",
        "protocol": {"features": FEATURES, "development_patients": sorted(patients),
                     "excluded_evaluation_patients": sorted(excluded),
                     "outer_group_folds": 3, "inner_group_folds": 3,
                     "outer_seed": 73, "inner_seed": 23,
                     "threshold_target_development_recall": TARGET_RECALL,
                     "sampling": "Up to 150 evenly spaced usable intervals per training patient; all eligible outer-validation intervals.",
                     "selection": "Inner patient-group balanced accuracy; threshold from inner out-of-fold positives only."},
        "svm_nested_development": svm_experiment["nested_development"],
        "random_forest_nested_development": {"pooled": forest_metrics, "folds": folds},
        "footprints": {
            "svm": footprint(bundle["estimator"], development.iloc[:1][FEATURES], "decision_function"),
            "random_forest": footprint(forest, development.iloc[:1][FEATURES], "predict_proba")},
        "production_model": {"id": bundle["metadata"]["model_id"], "sha256": original_hash,
                             "changed": False},
        "interpretation": "Development comparison only. Reported paper accuracy is not comparable across different cohorts and targets. No automatic promotion, new holdout tuning, or clinical reliability claim.",
    }
    assert hashlib.sha256(PREDICTOR_PATH.read_bytes()).hexdigest() == original_hash
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=RAW_PATH)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    result = benchmark(args.data)
    args.report.write_text(json.dumps(result, indent=2) + "\n")
    print("Report saved to", args.report)
