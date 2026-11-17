"""Acquire, evaluate, and use the separate research oximetry model."""
import argparse
import json
from pathlib import Path

import pandas as pd

from src.evaluate import format_metrics
from src.oximetry import SENSOR_MODEL_PATH, load_sensor_model, screen_recording
from src.sensor_training import RAW_PATH, download_dataset, train_sensor_pipeline


def main():
    parser = argparse.ArgumentParser(description="Experimental 1 Hz SpO2 screening with patient-separated evaluation.")
    commands = parser.add_subparsers(dest="command", required=True)
    download = commands.add_parser("download", help="download the pinned 6.4 MB public Kaggle vitals file")
    download.add_argument("--data", type=Path, default=RAW_PATH)
    train = commands.add_parser("train", help="evaluate both SpO2 feature sets and save the sensor SVM")
    train.add_argument("--data", type=Path, default=RAW_PATH)
    train.add_argument("--model", type=Path, default=SENSOR_MODEL_PATH)
    train.add_argument("--report", type=Path, help="optional additional readable report destination")
    prediction = commands.add_parser("predict", help="screen a CSV of elapsed_seconds,spo2 at 1 Hz")
    prediction.add_argument("--input", type=Path, required=True)
    prediction.add_argument("--model", type=Path, default=SENSOR_MODEL_PATH)
    args = parser.parse_args()
    try:
        if args.command == "download":
            print(download_dataset(args.data))
        elif args.command == "train":
            bundle, report = train_sensor_pipeline(args.data, args.model)
            if args.report:
                args.report.parent.mkdir(parents=True, exist_ok=True)
                args.report.write_text(json.dumps(bundle["metadata"], indent=2) + "\n")
            for name, experiment in report["experiments"].items():
                print(f"\n{name} — held-out patients")
                print(format_metrics(experiment["holdout"]))
                print("Nested development:", experiment["nested_development"]["pooled"])
            print("\nHoldout quality coverage:", report["quality"]["holdout"])
            print(f"\nResearch model saved to {args.model}")
        else:
            if args.input.stat().st_size > 1_000_000:
                raise ValueError("CSV must be at most 1 MB.")
            result = screen_recording(load_sensor_model(args.model), pd.read_csv(args.input))
            print(json.dumps(result, indent=2, allow_nan=False))
    except (OSError, ValueError) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    main()
