import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from fastapi.testclient import TestClient

from api import create_app
from src.combined import (COMBINED_MODEL_PATH, EXAMPLE, FEATURES, average_sensor_window,
                          build_combined_features, load_combined_model, predict_combined)


class CombinedInputTests(unittest.TestCase):
    def test_every_field_is_required(self):
        for field in FEATURES:
            record = dict(EXAMPLE)
            del record[field]
            with self.subTest(field=field), self.assertRaises(ValueError):
                build_combined_features(record)

    def test_invalid_measurements_are_not_defaulted(self):
        for field in FEATURES:
            for value in (None, True, "97", np.nan, np.inf, -1, 10**500):
                with self.subTest(field=field, kind=type(value).__name__), self.assertRaises(ValueError):
                    build_combined_features({**EXAMPLE, field: value})

    def test_exact_four_numeric_features(self):
        frame = build_combined_features(EXAMPLE)
        self.assertEqual(list(frame.columns), ["spo2", "heart_rate", "respiratory_rate", "perfusion_index"])
        self.assertTrue(np.isfinite(frame.to_numpy()).all())
        with self.assertRaises(ValueError):
            build_combined_features({**EXAMPLE, "ahi": 30})

    def test_same_interval_averages_are_preserved(self):
        signals = {field: [value] * 60 for field, value in EXAMPLE.items()}
        features, quality = average_sensor_window(signals)
        np.testing.assert_allclose(features.iloc[0].to_numpy(), list(EXAMPLE.values()))
        self.assertTrue(all(value["valid_fraction"] == 1 for value in quality.values()))

    def test_one_missing_channel_prevents_a_joint_prediction(self):
        signals = {field: [value] * 60 for field, value in EXAMPLE.items()}
        signals["respiratory_rate"] = [np.nan] * 60
        frame, _ = average_sensor_window(signals)
        self.assertIsNone(frame)

    def test_missing_channel_and_boolean_waveforms_are_rejected(self):
        signals = {field: [value] * 60 for field, value in EXAMPLE.items()}
        with self.assertRaises(ValueError):
            average_sensor_window({field: values for field, values in signals.items() if field != "spo2"})
        with self.assertRaises(ValueError):
            average_sensor_window({**signals, "spo2": [True] * 60})


@unittest.skipUnless(COMBINED_MODEL_PATH.exists(), "Train the combined model first.")
class CombinedArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = load_combined_model()

    def test_one_estimator_receives_all_four_inputs(self):
        self.assertEqual(list(self.bundle["estimator"].feature_names_in_), FEATURES)
        self.assertTrue((self.bundle["estimator"]["scaler"].var_ > 0).all())
        base = predict_combined(self.bundle, EXAMPLE)["decision_margin"]
        for field, change in {"spo2": -1, "heart_rate": 2, "respiratory_rate": 1, "perfusion_index": 0.2}.items():
            result = predict_combined(self.bundle, {**EXAMPLE, field: EXAMPLE[field] + change})
            with self.subTest(field=field):
                self.assertNotEqual(base, result["decision_margin"])

    def test_supported_values_produce_one_binary_screening_result(self):
        result = predict_combined(self.bundle, EXAMPLE)
        self.assertIn(result["status"], ("positive", "negative"))
        self.assertIsInstance(result["screen_positive"], bool)
        self.assertNotIn("probability", result)
        self.assertNotIn("diagnosis", result)
        json.dumps(result, allow_nan=False)

    def test_unsupported_reading_is_not_silently_a_negative(self):
        result = predict_combined(self.bundle, {**EXAMPLE, "spo2": 50.0})
        self.assertEqual(result["status"], "insufficient_evidence")
        self.assertIsNone(result["screen_positive"])

    def test_report_discloses_patient_split_and_measurement_transfer(self):
        report = self.bundle["metadata"]["report"]
        self.assertFalse(set(report["split"]["development_patients"]) & set(report["split"]["holdout_patients"]))
        self.assertIn("ECG-derived", report["sensor_transfer_note"])
        self.assertIn("previously inspected", report["evaluation_note"])


@unittest.skipUnless(COMBINED_MODEL_PATH.exists(), "Train the combined model first.")
class CombinedApiTests(unittest.TestCase):
    def setUp(self):
        self.token = "combined-local-test-token-" * 3
        self.headers = {"Authorization": f"Bearer {self.token}"}
        self.client = TestClient(create_app(api_key=self.token))
        self.client.__enter__()

    def tearDown(self):
        self.client.__exit__(None, None, None)

    def test_authentication_and_shared_prediction(self):
        self.assertEqual(self.client.get("/v2/model").status_code, 401)
        response = self.client.post("/v2/screen", json=EXAMPLE, headers=self.headers)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), predict_combined(load_combined_model(), EXAMPLE))

    def test_missing_measurements_and_wrong_types(self):
        for body in ({}, {**EXAMPLE, "spo2": True}, {**EXAMPLE, "spo2": "97"}, {**EXAMPLE, "daily_steps": 7000}):
            with self.subTest(body=body):
                self.assertEqual(self.client.post("/v2/screen", json=body, headers=self.headers).status_code, 422)

    def test_combined_model_can_run_without_old_models(self):
        with tempfile.TemporaryDirectory() as directory:
            absent = Path(directory) / "absent.joblib"
            with TestClient(create_app(model_path=absent, sensor_model_path=absent, api_key=self.token)) as client:
                self.assertEqual(client.get("/health").status_code, 200)
                self.assertEqual(client.post("/v2/screen", json=EXAMPLE, headers=self.headers).status_code, 200)
                self.assertEqual(client.get("/v1/model", headers=self.headers).status_code, 503)

    def test_unsupported_measurements_return_explicit_uncertainty(self):
        response = self.client.post("/v2/screen", json={**EXAMPLE, "spo2": 50}, headers=self.headers)
        self.assertEqual(response.status_code, 200)
        self.assertIsNone(response.json()["screen_positive"])


if __name__ == "__main__":
    unittest.main()
