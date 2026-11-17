"""Command-line entry point. Training is always an explicit operation."""
import argparse
import json
from pathlib import Path

from src.batch import parse_upload, screen_batch
from src.data import DATA_PATH, SYMPTOMS
from src.evaluate import format_metrics
from src.pipeline import MODEL_PATH, load_model, predict, train_pipeline


def add_predict_args(parser):
    parser.add_argument("--gender", choices=["Male", "Female"], required=True)
    parser.add_argument("--age", type=int, required=True)
    parser.add_argument("--sleep", dest="sleep_duration", type=float, required=True, help="hours per night")
    parser.add_argument("--bmi", choices=["Normal", "Overweight", "Obese"], required=True)
    parser.add_argument("--heart", dest="heart_rate", type=float, required=True, help="resting pulse, bpm")
    parser.add_argument("--steps", dest="daily_steps", type=int, required=True, help="steps per day")
    for symptom in SYMPTOMS:
        parser.add_argument("--" + symptom.replace("_", "-"), action="store_true")
    parser.add_argument("--json", action="store_true", help="emit a machine-readable result")


def run_train(args):
    _, metrics = train_pipeline(data_path=args.data, model_path=args.model)
    print("RESEARCH DATASET EVALUATION — not clinical performance")
    print(f"Dataset audit: {metrics['dataset']}")
    print(f"Best parameters: {metrics['best_params']}")
    print(f"Training / holdout rows: {metrics['training']['rows']} / {metrics['holdout']['rows']}")
    nested = metrics["nested_development_cv"]
    print(f"Nested development CV F2: {nested['f2_mean']:.3f} ± {nested['f2_std']:.3f}")
    print("\nFixed grouped holdout (raw classifier):")
    print(format_metrics(metrics))
    print(f"\nModel: {args.model}")
    print(f"Report: {Path(args.model).with_suffix('.report.json')}")


def run_predict(args):
    record = {key: getattr(args, key) for key in
              ("gender", "age", "sleep_duration", "bmi", "heart_rate", "daily_steps", *SYMPTOMS)}
    result = predict(load_model(args.model), record)
    if args.json:
        print(json.dumps(result, indent=2, allow_nan=False))
        return
    print(f"Screening result: {result['status'].replace('_', ' ')}")
    print(result["next_step"])
    for item in result["reported_symptoms"] + result["warnings"]:
        print(f"- {item}")
    print(f"\n{result['disclaimer']}")


def run_batch(args):
    path = Path(args.input)
    if path.stat().st_size > 1_000_000:
        raise ValueError("Upload must be at most 1 MB.")
    records = parse_upload(path.read_bytes(), path.suffix)
    results = screen_batch(load_model(args.model), records)
    text = json.dumps(results, indent=2, allow_nan=False) + "\n"
    if args.output:
        Path(args.output).write_text(text)
    else:
        print(text, end="")


def main():
    parser = argparse.ArgumentParser(description="IoT-friendly sleep apnea risk screening research prototype.")
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train", help="train, evaluate and save a versioned artifact")
    train.add_argument("--data", type=Path, default=DATA_PATH)
    train.add_argument("--model", type=Path, default=MODEL_PATH)
    train.set_defaults(func=run_train)
    pred = sub.add_parser("predict", help="screen a complete set of measured inputs")
    pred.add_argument("--model", type=Path, default=MODEL_PATH)
    add_predict_args(pred)
    pred.set_defaults(func=run_predict)
    batch = sub.add_parser("batch", help="screen up to 100 daily summaries from CSV or JSON")
    batch.add_argument("--model", type=Path, default=MODEL_PATH)
    batch.add_argument("--input", required=True)
    batch.add_argument("--output", help="optional JSON output path")
    batch.set_defaults(func=run_batch)
    args = parser.parse_args()
    try:
        args.func(args)
    except (ValueError, FileNotFoundError, OSError) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    main()
