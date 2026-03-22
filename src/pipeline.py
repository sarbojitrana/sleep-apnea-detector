import pandas as pd

from src.data import load_data, preprocess_data, encode_data, prepare_data
from src.model import train_tuned_model
from src.utils import calculate_risk, classify_severity


def train_pipeline():
    df = load_data("data/Sleep_health_and_lifestyle_dataset.csv")
    df = preprocess_data(df)
    df, encoders = encode_data(df)

    X_train, X_test, y_train, y_test, scaler = prepare_data(df)

    model, _ = train_tuned_model(X_train, y_train)

    return model, scaler


def predict(model, scaler, input_dict):
    df = pd.DataFrame([input_dict])

    # Feature engineering (same as training)
    df["Pulse Pressure"] = df["Systolic"] - df["Diastolic"]
    df["Sleep Efficiency"] = df["Quality of Sleep"] / df["Sleep Duration"]
    df["Activity Ratio"] = df["Daily Steps"] / (df["Physical Activity Level"] + 1)

    X = scaler.transform(df)
    pred = model.predict(X)[0]

    risk = calculate_risk(df.iloc[0])
    severity = classify_severity(risk)

    return pred, risk, severity