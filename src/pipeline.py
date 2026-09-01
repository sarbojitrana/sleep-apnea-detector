from pathlib import Path

import joblib

from src.data import DATA_PATH, build_features, load_dataset, split_data
from src.evaluate import cross_validate_model, evaluate_model
from src.model import train_tuned_model
from src.utils import calculate_risk, classify_severity

MODEL_PATH = Path("models/apnea_svm.joblib")
DECISION_THRESHOLD = 0.5


def train_pipeline(data_path=DATA_PATH, model_path=MODEL_PATH, save=True):
    X, y = load_dataset(data_path)
    X_train, X_test, y_train, y_test = split_data(X, y)

    model, best_params = train_tuned_model(X_train, y_train)

    metrics = evaluate_model(model, X_test, y_test)
    cv_mean, cv_std = cross_validate_model(model, X, y)
    metrics["cv_f1_mean"] = cv_mean
    metrics["cv_f1_std"] = cv_std
    metrics["best_params"] = best_params

    if save:
        model_path = Path(model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_path)

    return model, metrics


def load_model(model_path=MODEL_PATH):
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"no trained model at {model_path}. Run `python app.py train` first."
        )
    return joblib.load(model_path)


def load_or_train(model_path=MODEL_PATH):
    try:
        return load_model(model_path)
    except FileNotFoundError:
        model, _ = train_pipeline(model_path=model_path)
        return model


def predict(model, record, threshold=DECISION_THRESHOLD):
    features = build_features(record)
    probability = float(model.predict_proba(features)[0, 1])
    risk, reasons = calculate_risk(features.iloc[0])

    return {
        "apnea": bool(probability >= threshold),
        "probability": probability,
        "risk_score": risk,
        "severity": classify_severity(risk),
        "risk_factors": reasons,
        "disagreement": probability < threshold and risk >= 60,
    }
