import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from src.data import ROOT
from src.pipeline import MODEL_PATH
from src.oximetry import (SENSOR_FEATURES, SENSOR_MODEL_PATH, InsufficientSignal,
                          extract_window_features, load_sensor_model, predict_window,
                          screen_recording)
from src.sensor_training import download_dataset, prepare_windows, sample_training_rows


class OximetryFeatureTests(unittest.TestCase):
    def test_known_oxygen_summary(self):
        values = [98.0] * 30 + [88.0] * 30
        features, quality = extract_window_features(values)
        row = features.iloc[0]
        self.assertEqual(list(features.columns), SENSOR_FEATURES)
        self.assertEqual(row["spo2_mean"], 93)
        self.assertEqual(row["spo2_min"], 88)
        self.assertEqual(row["spo2_range"], 10)
        self.assertEqual(row["spo2_below_90_fraction"], 0.5)
        self.assertEqual(quality["valid_fraction"], 1)
        self.assertLess(row["spo2_trend_per_minute"], 0)

    def test_constant_sensor_signal_has_finite_features(self):
        features, _ = extract_window_features([97.0] * 60)
        self.assertTrue(np.isfinite(features.to_numpy()).all())
        self.assertEqual(features.iloc[0]["spo2_std"], 0)
        self.assertEqual(features.iloc[0]["spo2_mean_absolute_step"], 0)

    def test_short_or_multidimensional_windows_rejected(self):
        for values in ([97.0] * 59, [[97.0]] * 60):
            with self.assertRaises(ValueError):
                extract_window_features(values)

    def test_booleans_are_not_readings(self):
        with self.assertRaises(ValueError):
            extract_window_features([True] * 60)

    def test_dropout_causes_abstention(self):
        for bad in (float("nan"), 0.0, 101.0, float("inf"), -1.0):
            values = [97.0] * 60
            values[20:26] = [bad] * 6
            with self.subTest(bad=bad), self.assertRaises(InsufficientSignal):
                extract_window_features(values)

    def test_missing_samples_are_not_filled_with_normal_oxygen(self):
        values = [96.0] * 60
        values[20:25] = [np.nan] * 5
        features, quality = extract_window_features(values)
        self.assertAlmostEqual(quality["valid_fraction"], 55 / 60)
        self.assertEqual(features.iloc[0]["spo2_mean"], 96.0)

    def test_low_quality_avoids_model_execution(self):
        result = predict_window({}, [np.nan] * 60)
        self.assertEqual(result["status"], "insufficient_evidence")
        self.assertIsNone(result["screen_positive"])

    def test_only_optical_features_are_predictors(self):
        forbidden = ("patient", "target", "event", "anomaly", "ecg", "hr", "rr", "pvc", "ahi")
        self.assertFalse(set(SENSOR_FEATURES) & set(forbidden))
        self.assertTrue(all(name.startswith("spo2_") for name in SENSOR_FEATURES))


class SensorTrainingContractTests(unittest.TestCase):
    def test_training_sampling_keeps_subjects_and_ignores_labels(self):
        frame = pd.DataFrame({"patient": ["a"] * 250 + ["b"] * 250,
                              "target": [0, 1] * 250, "value": np.arange(500)})
        first = sample_training_rows(frame)
        flipped = frame.copy()
        flipped["target"] = 1 - flipped["target"]
        self.assertEqual(first.index.tolist(), sample_training_rows(flipped).index.tolist())
        self.assertEqual(first.groupby("patient").size().to_dict(), {"a": 150, "b": 150})

    def test_unverified_data_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "incorrect.parquet"
            path.write_bytes(b"unverified file")
            with self.assertRaisesRegex(ValueError, "checksum"):
                prepare_windows(path)
            with self.assertRaisesRegex(ValueError, "pinned"):
                download_dataset(path)

    @unittest.skipUnless((ROOT / "docs/osasud-evaluation.json").exists(), "No saved sensor experiment report.")
    def test_report_has_patient_disjoint_validation_and_coverage(self):
        report = json.loads((ROOT / "docs/osasud-evaluation.json").read_text())["report"]
        development = set(report["split"]["development_patients"])
        holdout = set(report["split"]["holdout_patients"])
        self.assertFalse(development & holdout)
        self.assertEqual(len(development), 24)
        self.assertEqual(len(holdout), 6)
        for experiment in report["experiments"].values():
            seen = set()
            for fold in experiment["nested_development"]["folds"]:
                patients = set(fold["patients"])
                self.assertTrue(patients <= development)
                self.assertFalse(patients & holdout)
                self.assertFalse(patients & seen)
                seen.update(patients)
            self.assertEqual(seen, development)
        coverage = report["quality"]["holdout"]
        self.assertEqual(coverage["eligible_windows"], 2569)
        self.assertEqual(coverage["abstained_windows"], 753)
        self.assertGreater(coverage["abstained_positive_windows"], 0)


@unittest.skipUnless(SENSOR_MODEL_PATH.exists(), "Run python sensor.py download and train for sensor inference checks.")
class SensorArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = load_sensor_model()

    def test_recording_schema_and_sample_rate(self):
        for frame in (
            pd.DataFrame({"spo2": [97] * 60}),
            pd.DataFrame({"elapsed_seconds": np.arange(60) * 2, "spo2": [97] * 60}),
            pd.DataFrame({"elapsed_seconds": [0] * 60, "spo2": [97] * 60}),
            pd.DataFrame({"elapsed_seconds": np.arange(60), "spo2": [True] * 60}),
        ):
            with self.assertRaises(ValueError):
                screen_recording(self.bundle, frame)

    def test_recording_results_have_no_diagnosis_or_ahi(self):
        frame = pd.DataFrame({"elapsed_seconds": np.arange(125),
                              "spo2": [97.0] * 60 + [np.nan] * 60 + [97.0] * 5})
        result = screen_recording(self.bundle, frame)
        self.assertEqual(len(result["windows"]), 2)
        self.assertEqual(result["abstained_windows"], 1)
        self.assertEqual(result["trailing_samples_not_scored"], 5)
        self.assertNotIn("ahi", result)
        self.assertNotIn("apnea_count", result)
        json.dumps(result, allow_nan=False)

    def test_saved_model_matches_shared_window_features(self):
        values = [98.0] * 30 + [88.0] * 30
        features, _ = extract_window_features(values)
        margin = float(self.bundle["estimator"].decision_function(features)[0])
        self.assertAlmostEqual(predict_window(self.bundle, values)["decision_margin"], margin)


@unittest.skipUnless(SENSOR_MODEL_PATH.exists() and MODEL_PATH.exists(), "Train both models before sensor API checks.")
class SensorApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from fastapi.testclient import TestClient
        from api import create_app
        cls.token = "sensor-api-local-test-" * 3
        cls.headers = {"Authorization": f"Bearer {cls.token}"}
        cls.client = TestClient(create_app(api_key=cls.token))
        cls.client.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.client.__exit__(None, None, None)

    def test_sensor_route_authentication(self):
        self.assertEqual(self.client.get("/v1/oximetry/model").status_code, 401)
        self.assertEqual(self.client.get("/v1/oximetry/model", headers=self.headers).status_code, 200)

    def test_sensor_route_matches_shared_inference(self):
        samples = [98.0] * 30 + [88.0] * 30
        response = self.client.post("/v1/oximetry/screen", headers=self.headers,
                                    json={"spo2_samples": samples})
        self.assertEqual(response.status_code, 200)
        expected = predict_window(load_sensor_model(), samples)
        self.assertEqual(response.json()["decision_margin"], expected["decision_margin"])
        self.assertEqual(response.json()["screen_positive"], expected["screen_positive"])

    def test_null_readings_cause_explicit_abstention(self):
        response = self.client.post("/v1/oximetry/screen", headers=self.headers,
                                    json={"spo2_samples": [None] * 60})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "insufficient_evidence")
        self.assertIsNone(response.json()["screen_positive"])

    def test_bad_sensor_contract_is_rejected(self):
        for record in ({"spo2_samples": [97] * 59}, {"spo2_samples": [True] * 60},
                       {"spo2_samples": ["97"] * 60}, {"spo2_samples": [97] * 60, "ahi": 20}):
            with self.subTest(record=record):
                self.assertEqual(self.client.post("/v1/oximetry/screen", headers=self.headers,
                                                  json=record).status_code, 422)

    def test_missing_sensor_artifact_does_not_disable_lifestyle_api(self):
        from fastapi.testclient import TestClient
        from api import create_app
        with tempfile.TemporaryDirectory() as directory:
            with TestClient(create_app(api_key=self.token, sensor_model_path=Path(directory) / "missing.joblib")) as client:
                self.assertEqual(client.get("/health").status_code, 200)
                self.assertEqual(client.get("/v1/oximetry/model", headers=self.headers).status_code, 503)


if __name__ == "__main__":
    unittest.main()
