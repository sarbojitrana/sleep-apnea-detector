"""Train or use the single combined-measurement predictor."""
import argparse
import json
from pathlib import Path

from src.combined import (COMBINED_MODEL_PATH, FEATURES, load_combined_model,
                          predict_combined, train_combined_model)
from src.sensor_training import RAW_PATH, download_dataset


def main():
    parser = argparse.ArgumentParser(description="One SVM combining SpO2, pulse, breathing rate and perfusion index.")
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train")
    train.add_argument("--data", type=Path, default=RAW_PATH)
    train.add_argument("--model", type=Path, default=COMBINED_MODEL_PATH)
    train.add_argument("--report", type=Path)
    predict = commands.add_parser("predict")
    predict.add_argument("--model", type=Path, default=COMBINED_MODEL_PATH)
    for field in FEATURES:
        predict.add_argument("--" + field.replace("_", "-"), dest=field, type=float, required=True)
    args = parser.parse_args()
    try:
        if args.command == "train":
            download_dataset(args.data)
            bundle = train_combined_model(args.data, args.model)
            if args.report:
                args.report.parent.mkdir(parents=True, exist_ok=True)
                args.report.write_text(json.dumps(bundle["metadata"], indent=2) + "\n")
            for name, experiment in bundle["metadata"]["report"]["experiments"].items():
                print(name, experiment["holdout"])
            print("Quality coverage:", bundle["metadata"]["report"]["quality"]["holdout"])
        else:
            print(json.dumps(predict_combined(load_combined_model(args.model),
                {field: getattr(args, field) for field in FEATURES}), indent=2, allow_nan=False))
    except (ValueError, OSError) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    main()
