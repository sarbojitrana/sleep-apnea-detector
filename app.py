import argparse
import pandas as pd

from src.data import load_data, preprocess_data, encode_data, prepare_data
from src.model import train_tuned_model
from src.evalaute import evaluate_model
from src.utils import calculate_risk, classify_severity
from src.pipeline import train_pipeline, predict



def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--age", type=int, required=True)
    parser.add_argument("--sleep", type=float, required=True)
    parser.add_argument("--stress", type=int, required=True)
    parser.add_argument("--heart", type=int, required=True)
    parser.add_argument("--bmi", type=int, required=True)

    args = parser.parse_args()

    # Load + train model
    df = load_data("data/Sleep_health_and_lifestyle_dataset.csv")
    df = preprocess_data(df)
    df, encoders = encode_data(df)

    X_train, X_test, y_train, y_test, scaler = prepare_data(df)
    model, _ = train_tuned_model(X_train, y_train)

    # Create input row (simplified)
    sample = pd.DataFrame([{
        "Gender": 0,
        "Age": args.age,
        "Occupation": 0,
        "Sleep Duration": args.sleep,
        "Quality of Sleep": 5,
        "Physical Activity Level": 30,
        "Stress Level": args.stress,
        "BMI Category": args.bmi,
        "Heart Rate": args.heart,
        "Daily Steps": 5000,
        "Systolic": 120,
        "Diastolic": 80
    }])

    # Add engineered features (MUST MATCH TRAINING)
    sample["Pulse Pressure"] = sample["Systolic"] - sample["Diastolic"]
    sample["Sleep Efficiency"] = sample["Quality of Sleep"] / sample["Sleep Duration"]
    sample["Activity Ratio"] = sample["Daily Steps"] / (sample["Physical Activity Level"] + 1)
    sample_scaled = scaler.transform(sample)
    prediction = model.predict(sample_scaled)[0]

    risk = calculate_risk(sample.iloc[0])
    severity = classify_severity(risk)

    print("\n=== RESULT ===")

    if prediction == 1:
        print("Apnea: YES")
    else:
        print("Apnea: NO")

    print("Risk Score:", risk)
    print("Severity:", severity)

    # Add interpretation layer
    if prediction == 0 and risk >= 60:
        print("\n Warning: High risk indicators present despite negative prediction.")


if __name__ == "__main__":
    main()