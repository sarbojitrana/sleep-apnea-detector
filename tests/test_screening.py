import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.svm import SVC

from src.batch import parse_upload, screen_batch
from src.data import (DATA_PATH, EXAMPLE_RECORD, FEATURES, INPUT_COLUMNS, SYMPTOMS,
                      build_features, dataset_audit, load_dataset, profile_groups,
                      split_data, validate_record)
from src.evaluate import classification_metrics
from src.pipeline import load_model, predict, train_pipeline


class InputTests(unittest.TestCase):
    def test_explicit_feature_contract(self):
        self.assertEqual(list(build_features(EXAMPLE_RECORD).columns), FEATURES)
        self.assertEqual(len(FEATURES), 6)
        self.assertEqual(set(FEATURES), {"Gender", "Age", "Sleep Duration", "BMI Category", "Heart Rate", "Daily Steps"})

    def test_missing_inputs_are_never_defaulted(self):
        for name in INPUT_COLUMNS:
            with self.subTest(name=name):
                record = dict(EXAMPLE_RECORD)
                del record[name]
                with self.assertRaisesRegex(ValueError, "Missing required"):
                    build_features(record)

    def test_invalid_numbers(self):
        for key in ("age", "sleep_duration", "heart_rate", "daily_steps"):
            for value in (None, True, float("nan"), float("inf"), -1, "70", 10**500):
                with self.subTest(key=key, value_type=type(value).__name__):
                    with self.assertRaises(ValueError):
                        build_features({**EXAMPLE_RECORD, key: value})

    def test_integer_fields(self):
        for name in ("age", "daily_steps"):
            with self.assertRaises(ValueError):
                build_features({**EXAMPLE_RECORD, name: 40.5})

    def test_unknown_categories_and_numeric_codes(self):
        for key in ("gender", "bmi"):
            for value in ("unknown", "", 0, 1.5, None):
                with self.subTest(key=key, value=value):
                    with self.assertRaises(ValueError):
                        build_features({**EXAMPLE_RECORD, key: value})

    def test_category_normalization(self):
        record = {**EXAMPLE_RECORD, "bmi": " Normal Weight ", "gender": " female "}
        pd.testing.assert_frame_equal(build_features(record), build_features(EXAMPLE_RECORD))

    def test_symptoms_are_strict_booleans(self):
        for key in SYMPTOMS:
            with self.assertRaises(ValueError):
                validate_record({**EXAMPLE_RECORD, key: "false"})

    def test_extraneous_sensor_is_not_silently_ignored(self):
        with self.assertRaisesRegex(ValueError, "Unknown inputs"):
            build_features({**EXAMPLE_RECORD, "spo2": 95})

    def test_non_object(self):
        for value in (None, [], "text"):
            with self.assertRaises(ValueError):
                build_features(value)

    def test_bundled_target_mapping(self):
        X, y = load_dataset()
        self.assertEqual(len(X), 374)
        self.assertEqual(int(y.sum()), 78)
        self.assertEqual(dataset_audit(X, y)["unique_profiles"], 106)
        self.assertEqual(dataset_audit(X, y)["conflicting_binary_label_profiles"], 16)

    def test_unknown_or_empty_labels_are_not_assumed_healthy(self):
        source = pd.read_csv(DATA_PATH, keep_default_na=False).head(10)
        for invalid in ("", "Mystery"):
            with self.subTest(invalid=invalid), tempfile.TemporaryDirectory() as directory:
                source.loc[0, "Sleep Disorder"] = invalid
                path = Path(directory) / "bad.csv"
                source.to_csv(path, index=False)
                with self.assertRaisesRegex(ValueError, "labels"):
                    load_dataset(path)

    def test_dataset_and_inference_features_match(self):
        X, _ = load_dataset()
        df = pd.read_csv(DATA_PATH, keep_default_na=False)
        for index in (0, 3, 100, 373):
            record = {key: df.iloc[index][column] for key, column in INPUT_COLUMNS.items()}
            pd.testing.assert_frame_equal(build_features(record).reset_index(drop=True),
                                          X.iloc[[index]].reset_index(drop=True))


class GroupingTests(unittest.TestCase):
    def setUp(self):
        self.X, self.y = load_dataset()

    def test_heldout_profiles_are_disjoint(self):
        train, test, _, _ = split_data(self.X, self.y)
        self.assertFalse(set(map(tuple, train.to_numpy())) & set(map(tuple, test.to_numpy())))

    def test_cross_validation_profiles_are_disjoint(self):
        groups = profile_groups(self.X)
        for train, test in StratifiedGroupKFold(4, shuffle=True, random_state=73).split(
                self.X, self.y, groups):
            self.assertFalse(set(groups.iloc[train]) & set(groups.iloc[test]))

    def test_conflicting_labels_stay_together(self):
        X = pd.DataFrame([[1, 2], [1, 2], [3, 4]])
        groups = profile_groups(X)
        self.assertEqual(groups.iloc[0], groups.iloc[1])
        self.assertNotEqual(groups.iloc[0], groups.iloc[2])


class BatchTests(unittest.TestCase):
    def test_csv_json_parity(self):
        csv = ("gender,age,sleep_duration,bmi,heart_rate,daily_steps,witnessed_pauses\n"
               "Female,40,7,Normal,70,7000,true\n").encode()
        csv_record = parse_upload(csv, ".csv")[0]
        json_record = parse_upload(json.dumps({**EXAMPLE_RECORD, "witnessed_pauses": True}).encode(), ".json")[0]
        self.assertEqual(validate_record(csv_record), validate_record(json_record))

    def test_bad_uploads(self):
        cases = [(b"not json", ".json"), (b"[]", ".json"), (b"42", ".json"),
                 (b"age,age\n40,40", ".csv"), (b"age,bmi\n40", ".csv"),
                 (b"age\n40,50", ".csv"), (b"age\n", ".csv"), (b"\xff", ".csv"),
                 (b'{"age": NaN}', ".json"), (b"x", ".txt")]
        for content, suffix in cases:
            with self.subTest(content=content):
                with self.assertRaises(ValueError):
                    parse_upload(content, suffix)

    def test_upload_limits(self):
        with self.assertRaises(ValueError):
            parse_upload(b"x" * 1_000_001, ".json")
        with self.assertRaises(ValueError):
            parse_upload(json.dumps([EXAMPLE_RECORD] * 101).encode(), ".json")
        with self.assertRaises(ValueError):
            parse_upload(("age\n" + "40\n" * 101).encode(), ".csv")


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.path = Path(cls.directory.name) / "model.joblib"
        cls.bundle, cls.metrics = train_pipeline(model_path=cls.path)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def test_saved_svm_and_metadata(self):
        self.assertIsInstance(self.bundle["estimator"]["svc"], SVC)
        self.assertTrue(self.path.with_suffix(".report.json").exists())
        self.assertEqual(self.metrics["training"]["rows"] + self.metrics["holdout"]["rows"], 374)
        self.assertEqual(predict(self.bundle, EXAMPLE_RECORD), predict(load_model(self.path), EXAMPLE_RECORD))

    def test_reject_legacy_artifact(self):
        path = Path(self.directory.name) / "legacy.joblib"
        joblib.dump(self.bundle["estimator"], path)
        with self.assertRaisesRegex(ValueError, "Legacy"):
            load_model(path)

    def test_reject_incompatible_artifact(self):
        for key, value in (("schema_version", -1), ("sklearn_version", "0.0"), ("features", [])):
            bundle = copy.deepcopy(self.bundle)
            bundle["metadata"][key] = value
            path = Path(self.directory.name) / "incompatible.joblib"
            joblib.dump(bundle, path)
            with self.subTest(key=key), self.assertRaises(ValueError):
                load_model(path)

    def test_model_does_not_train_on_load(self):
        with self.assertRaises(FileNotFoundError):
            load_model(Path(self.directory.name) / "missing.joblib")

    def test_outside_support_abstains(self):
        bundle = {"metadata": self.bundle["metadata"], "estimator": Mock()}
        result = predict(bundle, {**EXAMPLE_RECORD, "age": 70})
        self.assertEqual(result["status"], "insufficient_evidence")
        self.assertIsNone(result["screen_positive"])
        self.assertIsNone(result["decision_margin"])
        bundle["estimator"].decision_function.assert_not_called()

    def test_symptoms_prompt_follow_up_on_negative_margin(self):
        estimator = Mock()
        estimator.decision_function.return_value = np.array([-2.0])
        bundle = {"metadata": self.bundle["metadata"], "estimator": estimator}
        result = predict(bundle, {**EXAMPLE_RECORD, "witnessed_pauses": True})
        self.assertEqual(result["status"], "lower_signal")
        self.assertTrue(result["follow_up_recommended"])
        self.assertEqual(len(result["reported_symptoms"]), 1)

    def test_symptoms_prompt_follow_up_even_when_abstaining(self):
        result = predict(self.bundle, {**EXAMPLE_RECORD, "age": 70, "witnessed_pauses": True})
        self.assertEqual(result["status"], "insufficient_evidence")
        self.assertTrue(result["follow_up_recommended"])

    def test_zero_margin_is_positive_everywhere(self):
        estimator = Mock()
        estimator.decision_function.return_value = np.array([0.0])
        result = predict({"metadata": self.bundle["metadata"], "estimator": estimator}, EXAMPLE_RECORD)
        self.assertTrue(result["screen_positive"])
        self.assertEqual(result["status"], "raised_signal")

    def test_results_are_json_safe_and_have_no_diagnostic_severity(self):
        for record in (EXAMPLE_RECORD, {**EXAMPLE_RECORD, "age": 70}):
            result = predict(self.bundle, record)
            json.dumps(result, allow_nan=False)
            for key in ("apnea", "probability", "severity", "risk_score"):
                self.assertNotIn(key, result)

    def test_holdout_evaluation_matches_served_decisions(self):
        X, y = load_dataset()
        _, test, _, truth = split_data(X, y)
        source = pd.read_csv(DATA_PATH, keep_default_na=False)
        predictions = []
        for index in test.index:
            record = {key: source.iloc[index][column] for key, column in INPUT_COLUMNS.items()}
            result = predict(self.bundle, record)
            self.assertIsNotNone(result["screen_positive"])
            predictions.append(int(result["screen_positive"]))
        self.assertEqual(int(sum((truth.to_numpy() == 1) & (np.array(predictions) == 0))),
                         self.metrics["false_negatives"])

    def test_bad_batch_row_does_not_discard_valid_rows(self):
        results = screen_batch(self.bundle, [EXAMPLE_RECORD, {"age": 40}, {**EXAMPLE_RECORD, "age": 70}])
        self.assertNotEqual(results[0]["status"], "invalid_input")
        self.assertEqual(results[1]["status"], "invalid_input")
        self.assertEqual(results[2]["status"], "insufficient_evidence")

    def test_metric_confusion_counts(self):
        metrics = classification_metrics([0, 0, 1, 1], [0, 1, 0, 1], [-2, 2, -1, 1])
        self.assertEqual(metrics["confusion_matrix"], [[1, 1], [1, 1]])
        self.assertEqual(metrics["recall"], 0.5)
        self.assertEqual(metrics["specificity"], 0.5)
        self.assertEqual(metrics["false_negatives"], 1)


class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from fastapi.testclient import TestClient
        from api import create_app
        cls.directory = tempfile.TemporaryDirectory()
        cls.path = Path(cls.directory.name) / "api-model.joblib"
        cls.bundle, _ = train_pipeline(model_path=cls.path)
        cls.token = "local-test-secret-" * 3
        cls.client = TestClient(create_app(cls.path, cls.token))
        cls.client.__enter__()
        cls.headers = {"Authorization": f"Bearer {cls.token}"}

    @classmethod
    def tearDownClass(cls):
        cls.client.__exit__(None, None, None)
        cls.directory.cleanup()

    def test_health_and_authentication(self):
        self.assertEqual(self.client.get("/health").status_code, 200)
        self.assertEqual(self.client.get("/v1/model").status_code, 401)
        self.assertEqual(self.client.post("/v1/screen", json=EXAMPLE_RECORD).status_code, 401)
        self.assertEqual(self.client.get("/v1/model", headers=self.headers).status_code, 200)
        self.assertEqual(self.client.get("/v1/model", headers={"Authorization": "Bearer wrong"}).status_code, 401)

    def test_api_matches_shared_prediction(self):
        response = self.client.post("/v1/screen", json=EXAMPLE_RECORD, headers=self.headers)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), predict(self.bundle, EXAMPLE_RECORD))

    def test_invalid_payloads(self):
        for record in ({}, [], {**EXAMPLE_RECORD, "age": "40"},
                       {**EXAMPLE_RECORD, "age": True},
                       {**EXAMPLE_RECORD, "gender": "unknown"},
                       {**EXAMPLE_RECORD, "spo2": 95},
                       {**EXAMPLE_RECORD, "witnessed_pauses": "false"}):
            with self.subTest(record=record):
                self.assertEqual(self.client.post("/v1/screen", json=record,
                                                 headers=self.headers).status_code, 422)

    def test_body_limit_and_content_type(self):
        self.assertEqual(self.client.post("/v1/screen", content="x" * 20000,
                         headers={**self.headers, "Content-Type": "application/json"}).status_code, 413)
        self.assertEqual(self.client.post("/v1/screen", content="text",
                         headers={**self.headers, "Content-Type": "text/plain"}).status_code, 415)

    def test_nonfinite_and_malformed_json(self):
        for body in ('{"age": NaN}', "{", '{"age": Infinity}'):
            response = self.client.post("/v1/screen", content=body,
                         headers={**self.headers, "Content-Type": "application/json"})
            self.assertEqual(response.status_code, 422)

    def test_api_returns_abstention(self):
        response = self.client.post("/v1/screen", json={**EXAMPLE_RECORD, "age": 70}, headers=self.headers)
        self.assertEqual(response.status_code, 200)
        self.assertIsNone(response.json()["screen_positive"])
        self.assertEqual(response.json()["status"], "insufficient_evidence")

    def test_api_refuses_short_key_at_startup(self):
        from fastapi.testclient import TestClient
        from api import create_app
        with self.assertRaisesRegex(RuntimeError, "APNEA_API_KEY"):
            with TestClient(create_app(self.path, "short")):
                pass


if __name__ == "__main__":
    unittest.main()
