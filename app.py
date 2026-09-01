import argparse

from src.data import DATA_PATH, DEFAULTS
from src.evaluate import format_metrics
from src.pipeline import MODEL_PATH, load_model, predict, train_pipeline


def add_predict_args(parser):
    parser.add_argument("--gender", choices=["Male", "Female"], default=DEFAULTS["gender"])
    parser.add_argument("--age", type=int, default=DEFAULTS["age"])
    parser.add_argument("--sleep", type=float, default=DEFAULTS["sleep_duration"])
    parser.add_argument("--quality", type=int, default=DEFAULTS["quality_of_sleep"])
    parser.add_argument("--activity", type=int, default=DEFAULTS["physical_activity"])
    parser.add_argument("--stress", type=int, default=DEFAULTS["stress"])
    parser.add_argument("--bmi", choices=["Normal", "Overweight", "Obese"], default=DEFAULTS["bmi"])
    parser.add_argument("--heart", type=int, default=DEFAULTS["heart_rate"])
    parser.add_argument("--steps", type=int, default=DEFAULTS["daily_steps"])
    parser.add_argument("--systolic", type=int, default=DEFAULTS["systolic"])
    parser.add_argument("--diastolic", type=int, default=DEFAULTS["diastolic"])


def run_train(args):
    print(f"Training on {args.data} ...")
    _, metrics = train_pipeline(data_path=args.data, model_path=args.model)

    print(f"\nBest parameters: {metrics['best_params']}")
    print(f"Cross-validated F1: {metrics['cv_f1_mean']:.3f} (+/- {metrics['cv_f1_std']:.3f})\n")
    print("Held-out test set")
    print("-" * 60)
    print(format_metrics(metrics))
    print(f"\nModel saved to {args.model}")


def run_predict(args):
    model = load_model(args.model)

    result = predict(model, {
        "gender": args.gender,
        "age": args.age,
        "sleep_duration": args.sleep,
        "quality_of_sleep": args.quality,
        "physical_activity": args.activity,
        "stress": args.stress,
        "bmi": args.bmi,
        "heart_rate": args.heart,
        "daily_steps": args.steps,
        "systolic": args.systolic,
        "diastolic": args.diastolic,
    })

    print("\n=== SLEEP APNEA SCREENING ===")
    print(f"Prediction  : {'LIKELY APNEA' if result['apnea'] else 'NO APNEA'}")
    print(f"Confidence  : {result['probability']:.1%}")
    print(f"Risk score  : {result['risk_score']}/100 ({result['severity']})")

    if result["risk_factors"]:
        print("Risk factors:")
        for factor in result["risk_factors"]:
            print(f"  - {factor}")

    if result["disagreement"]:
        print("\nWarning: high risk indicators present despite a negative prediction.")

    print("\nScreening aid only. Not a diagnosis; confirm with a sleep study.")


def main():
    parser = argparse.ArgumentParser(description="Sleep apnea screening from lifestyle and vitals data.")
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="train the SVM and save it to disk")
    train.add_argument("--data", default=DATA_PATH)
    train.add_argument("--model", default=MODEL_PATH)
    train.set_defaults(func=run_train)

    pred = sub.add_parser("predict", help="screen one person using the saved model")
    pred.add_argument("--model", default=MODEL_PATH)
    add_predict_args(pred)
    pred.set_defaults(func=run_predict)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
