"""Train and use the combined predictor with oxygen dynamics."""
import argparse
import json
from pathlib import Path

from src.predictor import FEATURES, PREDICTOR_PATH, load_predictor, predict, train_predictor
from src.sensor_training import RAW_PATH, download_dataset


def main():
    parser = argparse.ArgumentParser(description="Combined IoT screening predictor.")
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train")
    train.add_argument("--data", type=Path, default=RAW_PATH)
    train.add_argument("--model", type=Path, default=PREDICTOR_PATH)
    train.add_argument("--report", type=Path)
    prediction = commands.add_parser("predict")
    prediction.add_argument("--model", type=Path, default=PREDICTOR_PATH)
    for name in FEATURES:
        prediction.add_argument("--" + name.replace("_", "-"), dest=name, type=float, required=True)
    prediction.add_argument("--sleep-duration", type=float, help="Context only, not a trained feature.")
    args = parser.parse_args()
    try:
        if args.command == "train":
            download_dataset(args.data)
            bundle = train_predictor(args.data, args.model)
            if args.report:
                args.report.parent.mkdir(parents=True, exist_ok=True)
                args.report.write_text(json.dumps(bundle["metadata"], indent=2) + "\n")
            for name, result in bundle["metadata"]["report"]["experiments"].items():
                print(name, "evaluation", result["holdout"])
                print(name, "nested", result["nested_development"]["pooled"])
        else:
            record = {name: getattr(args, name) for name in FEATURES}
            if args.sleep_duration is not None:
                record["sleep_duration"] = args.sleep_duration
            print(json.dumps(predict(load_predictor(args.model), record), indent=2, allow_nan=False))
    except (OSError, ValueError) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    main()
