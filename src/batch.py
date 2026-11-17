"""Small, bounded daily-summary imports; raw waveforms are not model inputs."""
import csv
import io
import json

from src.data import NUMERIC_LIMITS, SYMPTOMS
from src.pipeline import predict

MAX_UPLOAD_BYTES = 1_000_000
MAX_RECORDS = 100


def parse_upload(content, suffix):
    if len(content) > MAX_UPLOAD_BYTES:
        raise ValueError("Upload must be at most 1 MB.")
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError("Upload must use UTF-8 text.") from error
    if suffix.lower() == ".json":
        try:
            records = json.loads(text, parse_constant=lambda v: _invalid_constant(v))
        except (json.JSONDecodeError, RecursionError) as error:
            raise ValueError("Upload must contain valid JSON.") from error
        if isinstance(records, dict):
            records = [records]
    elif suffix.lower() == ".csv":
        reader = csv.DictReader(io.StringIO(text), strict=True)
        try:
            if not reader.fieldnames or len(reader.fieldnames) != len(set(reader.fieldnames)):
                raise ValueError("CSV needs unique named column headers.")
            records = []
            for row in reader:
                if None in row or any(value is None for value in row.values()):
                    raise ValueError("CSV rows must have the same number of fields as the header.")
                if len(records) >= MAX_RECORDS:
                    raise ValueError(f"Import at most {MAX_RECORDS} records at once.")
                for name in NUMERIC_LIMITS:
                    if name in row:
                        try:
                            row[name] = float(row[name])
                        except ValueError:
                            pass  # Shared validation reports the bad value for this row.
                for name in SYMPTOMS:
                    if name in row:
                        value = row[name].strip().lower()
                        if not value:
                            del row[name]
                        elif value in ("true", "false"):
                            row[name] = value == "true"
                records.append(row)
        except csv.Error as error:
            raise ValueError("Malformed CSV file.") from error
    else:
        raise ValueError("Use a .csv or .json file.")
    if not isinstance(records, list) or not 1 <= len(records) <= MAX_RECORDS:
        raise ValueError(f"Upload must contain 1–{MAX_RECORDS} records.")
    return records


def _invalid_constant(value):
    raise ValueError(f"JSON cannot contain {value}.")


def screen_batch(bundle, records):
    if not isinstance(records, list) or not 1 <= len(records) <= MAX_RECORDS:
        raise ValueError(f"Provide 1–{MAX_RECORDS} records.")
    results = []
    for row, record in enumerate(records, start=1):
        try:
            results.append({"row": row, **predict(bundle, record)})
        except ValueError as error:
            results.append({"row": row, "status": "invalid_input", "error": str(error)})
    return results
