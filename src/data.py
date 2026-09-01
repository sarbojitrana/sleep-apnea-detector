import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = "data/Sleep_health_and_lifestyle_dataset.csv"
TARGET = "Sleep Disorder"
POSITIVE_LABEL = "Sleep Apnea"

BMI_MAP = {"normal": 0, "normal weight": 0, "overweight": 1, "obese": 2}
BMI_LABELS = {0: "Normal", 1: "Overweight", 2: "Obese"}
GENDER_MAP = {"female": 0, "male": 1}

FEATURES = [
    "Gender",
    "Age",
    "Sleep Duration",
    "Quality of Sleep",
    "Physical Activity Level",
    "Stress Level",
    "BMI Category",
    "Heart Rate",
    "Daily Steps",
    "Systolic",
    "Diastolic",
    "Pulse Pressure",
    "Sleep Efficiency",
    "Activity Ratio",
]

DEFAULTS = {
    "gender": "Female",
    "age": 40,
    "sleep_duration": 7.0,
    "quality_of_sleep": 7,
    "physical_activity": 60,
    "stress": 5,
    "bmi": "Normal",
    "heart_rate": 70,
    "daily_steps": 7000,
    "systolic": 125,
    "diastolic": 82,
}


def encode_bmi(value):
    if isinstance(value, str):
        key = value.strip().lower()
        if key not in BMI_MAP:
            raise ValueError(f"unknown BMI category {value!r}, expected one of {sorted(set(BMI_MAP))}")
        return BMI_MAP[key]
    value = int(value)
    if value not in BMI_LABELS:
        raise ValueError(f"BMI code must be 0 (Normal), 1 (Overweight) or 2 (Obese), got {value}")
    return value


def encode_gender(value):
    if isinstance(value, str):
        key = value.strip().lower()
        if key not in GENDER_MAP:
            raise ValueError(f"unknown gender {value!r}, expected Male or Female")
        return GENDER_MAP[key]
    return int(value)


def build_features(record):
    row = dict(DEFAULTS)
    row.update({k: v for k, v in record.items() if v is not None})

    systolic = float(row["systolic"])
    diastolic = float(row["diastolic"])
    sleep_duration = float(row["sleep_duration"])
    quality = float(row["quality_of_sleep"])
    activity = float(row["physical_activity"])
    steps = float(row["daily_steps"])

    if sleep_duration <= 0:
        raise ValueError("sleep_duration must be greater than 0")

    return pd.DataFrame([{
        "Gender": encode_gender(row["gender"]),
        "Age": float(row["age"]),
        "Sleep Duration": sleep_duration,
        "Quality of Sleep": quality,
        "Physical Activity Level": activity,
        "Stress Level": float(row["stress"]),
        "BMI Category": encode_bmi(row["bmi"]),
        "Heart Rate": float(row["heart_rate"]),
        "Daily Steps": steps,
        "Systolic": systolic,
        "Diastolic": diastolic,
        "Pulse Pressure": systolic - diastolic,
        "Sleep Efficiency": quality / sleep_duration,
        "Activity Ratio": steps / (activity + 1.0),
    }])[FEATURES]


def load_dataset(path=DATA_PATH):
    df = pd.read_csv(path)

    y = (df[TARGET] == POSITIVE_LABEL).astype(int)

    bp = df["Blood Pressure"].str.split("/", expand=True)
    systolic = bp[0].astype(float)
    diastolic = bp[1].astype(float)

    X = pd.DataFrame({
        "Gender": df["Gender"].str.lower().map(GENDER_MAP),
        "Age": df["Age"].astype(float),
        "Sleep Duration": df["Sleep Duration"].astype(float),
        "Quality of Sleep": df["Quality of Sleep"].astype(float),
        "Physical Activity Level": df["Physical Activity Level"].astype(float),
        "Stress Level": df["Stress Level"].astype(float),
        "BMI Category": df["BMI Category"].str.lower().map(BMI_MAP),
        "Heart Rate": df["Heart Rate"].astype(float),
        "Daily Steps": df["Daily Steps"].astype(float),
        "Systolic": systolic,
        "Diastolic": diastolic,
        "Pulse Pressure": systolic - diastolic,
        "Sleep Efficiency": df["Quality of Sleep"] / df["Sleep Duration"],
        "Activity Ratio": df["Daily Steps"] / (df["Physical Activity Level"] + 1.0),
    })[FEATURES]

    if X.isna().any().any():
        missing = X.columns[X.isna().any()].tolist()
        raise ValueError(f"unmapped categorical values in columns: {missing}")

    return X.astype(float), y


def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
