"""One explicit input contract for the dataset, UI, CLI, and device API."""
from pathlib import Path
from numbers import Real
import math

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data/Sleep_health_and_lifestyle_dataset.csv"
TARGET = "Sleep Disorder"
POSITIVE_LABEL = "Sleep Apnea"
BMI_MAP = {"normal": 0, "normal weight": 0, "overweight": 1, "obese": 2}
BMI_LABELS = {0: "Normal", 1: "Overweight", 2: "Obese"}
GENDER_MAP = {"female": 0, "male": 1}

INPUT_COLUMNS = {
    "gender": "Gender", "age": "Age", "sleep_duration": "Sleep Duration",
    "bmi": "BMI Category", "heart_rate": "Heart Rate", "daily_steps": "Daily Steps",
}
FEATURES = list(INPUT_COLUMNS.values())
NUMERIC_LIMITS = {
    "age": (18, 100), "sleep_duration": (0.5, 16),
    "heart_rate": (30, 220), "daily_steps": (0, 100000),
}
SYMPTOMS = ("loud_snoring", "witnessed_pauses", "daytime_sleepiness")
# For UI examples only: inference never fills missing readings.
EXAMPLE_RECORD = {
    "gender": "Female", "age": 40, "sleep_duration": 7.0,
    "bmi": "Normal", "heart_rate": 70, "daily_steps": 7000,
}


def _encode(value, mapping, name):
    if not isinstance(value, str) or value.strip().lower() not in mapping:
        raise ValueError(f"{name} must be one of: {', '.join(mapping)}")
    return mapping[value.strip().lower()]


def encode_bmi(value):
    return _encode(value, BMI_MAP, "bmi")


def encode_gender(value):
    return _encode(value, GENDER_MAP, "gender")


def validate_record(record):
    if not isinstance(record, dict):
        raise ValueError("Expected a JSON object containing the six required inputs.")
    missing = set(INPUT_COLUMNS) - record.keys()
    unknown = record.keys() - set(INPUT_COLUMNS) - set(SYMPTOMS)
    if missing:
        raise ValueError(f"Missing required inputs: {', '.join(sorted(missing))}")
    if unknown:
        raise ValueError(f"Unknown inputs: {', '.join(sorted(unknown))}")
    clean = dict(record)
    encode_gender(clean["gender"])
    clean["gender"] = clean["gender"].strip().title()
    clean["bmi"] = BMI_LABELS[encode_bmi(clean["bmi"])]
    for name, (low, high) in NUMERIC_LIMITS.items():
        value = clean[name]
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError(f"{name} must be a finite number, not a missing value or text.")
        try:
            finite = math.isfinite(value)
        except OverflowError:
            finite = False
        if not finite:
            raise ValueError(f"{name} must be a finite number.")
        if not low <= value <= high:
            raise ValueError(f"{name} must be between {low} and {high}.")
        if name in ("age", "daily_steps") and value != int(value):
            raise ValueError(f"{name} must be a whole number.")
        clean[name] = float(value)
    for name in SYMPTOMS:
        if name in clean and not isinstance(clean[name], bool):
            raise ValueError(f"{name} must be true or false.")
    return clean


def build_features(record):
    row = validate_record(record)
    row["gender"] = encode_gender(row["gender"])
    row["bmi"] = encode_bmi(row["bmi"])
    return pd.DataFrame([{column: row[key] for key, column in INPUT_COLUMNS.items()}],
                        columns=FEATURES, dtype=float)


def load_dataset(path=DATA_PATH):
    # Preserve literal 'None'. A genuinely blank/unknown target is an error.
    df = pd.read_csv(path, keep_default_na=False)
    required = set(FEATURES + [TARGET])
    if required - set(df.columns):
        raise ValueError(f"Dataset is missing columns: {sorted(required - set(df.columns))}")
    if df.empty or not df[TARGET].isin(["None", "Insomnia", POSITIVE_LABEL]).all():
        raise ValueError("Dataset needs labels None, Insomnia, or Sleep Apnea; blank targets are invalid.")
    records = df[FEATURES].rename(columns={v: k for k, v in INPUT_COLUMNS.items()}).to_dict("records")
    X = pd.concat([build_features(record) for record in records], ignore_index=True)
    y = (df[TARGET] == POSITIVE_LABEL).astype(int)
    if y.nunique() != 2:
        raise ValueError("Dataset needs both apnea-labelled and other-labelled rows.")
    return X, y


def profile_groups(X):
    """Group identical observed profiles, regardless of label or Person ID."""
    return pd.Series(pd.factorize(pd.MultiIndex.from_frame(X))[0], index=X.index)


def split_data(X, y, test_size=0.2, random_state=42):
    # Use one predeclared fold; never search for the split with the best result.
    folds = round(1 / test_size)
    if folds < 2:
        raise ValueError("test_size must produce at least two folds.")
    train, test = next(StratifiedGroupKFold(folds, shuffle=True, random_state=random_state)
                       .split(X, y, groups=profile_groups(X)))
    if y.iloc[train].nunique() < 2 or y.iloc[test].nunique() < 2:
        raise ValueError("Not enough independent profiles for a two-class held-out split.")
    return X.iloc[train], X.iloc[test], y.iloc[train], y.iloc[test]


def dataset_audit(X, y):
    grouped = pd.DataFrame({"group": profile_groups(X), "target": y}).groupby("group")["target"]
    return {
        "rows": len(X), "positive_rows": int(y.sum()), "unique_profiles": grouped.ngroups,
        "repeated_rows": len(X) - grouped.ngroups,
        "conflicting_binary_label_profiles": int((grouped.nunique() > 1).sum()),
    }
