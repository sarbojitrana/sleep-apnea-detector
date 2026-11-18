import copy
import json
import unittest

import numpy as np
from fastapi.testclient import TestClient

from api import create_app
from src.predictor import (EXAMPLE, FEATURES, PREDICTOR_PATH, build_features,
                           load_predictor, metrics_at_threshold, predict,
                           threshold_for_recall)


class PredictorInputTests(unittest.TestCase):
    def test_six_sensor_features_and_separate_context(self):
        frame, context = build_features(EXAMPLE)
        self.assertEqual(list(frame.columns), FEATURES)
        self.assertEqual(context, {"sleep_duration": 7.0})
        self.assertNotIn("sleep_duration", frame.columns)
        self.assertEqual(len(FEATURES), 6)

    def test_sleep_context_is_optional(self):
        record = {key: value for key, value in EXAMPLE.items() if key != "sleep_duration"}
        _, context = build_features(record)
        self.assertEqual(context, {})

    def test_missing_required_sensors_are_not_imputed(self):
        for name in FEATURES:
            record = dict(EXAMPLE)
            del record[name]
            with self.subTest(name=name), self.assertRaises(ValueError):
                build_features(record)

    def test_invalid_additional_values(self):
        for name in ("spo2_min", "spo2_std", "sleep_duration"):
            for value in (True, "7", float("nan"), float("inf"), -1, 10**500):
                with self.subTest(name=name, value_type=type(value).__name__), self.assertRaises(ValueError):
                    build_features({**EXAMPLE, name: value})

    def test_inconsistent_oxygen_summary_is_invalid(self):
        with self.assertRaisesRegex(ValueError, "Lowest"):
            build_features({**EXAMPLE, "spo2_min": 99.0})

    def test_diagnostic_indices_are_not_inputs(self):
        with self.assertRaises(ValueError):
            build_features({**EXAMPLE, "ahi": 30})

    def test_threshold_uses_development_positive_scores(self):
        labels = np.array([1, 1, 1, 1, 0, 0])
        scores = np.array([0, 1, 2, 3, -1, 4])
        self.assertEqual(threshold_for_recall(labels, scores, target=0.75), 1)
        with self.assertRaises(ValueError):
            threshold_for_recall([0, 0], [0.1, 0.2])

    def test_evaluation_applies_selected_threshold(self):
        metrics = metrics_at_threshold([0, 0, 1, 1], [0.1, 0.2, 0.3, 0.4], 0.3)
        self.assertEqual(metrics["recall"], 1)
        self.assertEqual(metrics["specificity"], 1)
        self.assertEqual(metrics["confusion_matrix"], [[2, 0], [0, 2]])


@unittest.skipUnless(PREDICTOR_PATH.exists(), "Train the current predictor first.")
class PredictorArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = load_predictor()

    def test_original_four_and_dynamics_enter_one_estimator(self):
        self.assertEqual(list(self.bundle["estimator"].feature_names_in_), FEATURES)
        self.assertGreater(self.bundle["estimator"]["sensor_weights"].sensor_weight, 0)
        base = predict(self.bundle, EXAMPLE)["decision_margin"]
        for name, delta in {"spo2": -1, "heart_rate": 2, "respiratory_rate": 1,
                            "perfusion_index": 0.2, "spo2_min": -1, "spo2_std": 0.1}.items():
            with self.subTest(name=name):
                result = predict(self.bundle, {**EXAMPLE, name: EXAMPLE[name] + delta})
                self.assertNotEqual(result["decision_margin"], base)

    def test_sleep_duration_does_not_fabricate_a_learned_effect(self):
        first = predict(self.bundle, {**EXAMPLE, "sleep_duration": 5.0})
        second = predict(self.bundle, {**EXAMPLE, "sleep_duration": 9.0})
        self.assertEqual(first["decision_margin"], second["decision_margin"])
        self.assertEqual(first["screen_positive"], second["screen_positive"])
        self.assertEqual(first["context_not_used_by_model"], ["sleep_duration"])
        self.assertEqual(second["context"], {"sleep_duration": 9.0})

    def test_prediction_matches_trained_operating_point(self):
        features, _ = build_features(EXAMPLE)
        score = float(self.bundle["estimator"].decision_function(features)[0])
        threshold = self.bundle["metadata"]["decision_threshold"]
        self.assertEqual(predict(self.bundle, EXAMPLE)["screen_positive"], bool(score >= threshold))

    def test_unsupported_measurements_abstain(self):
        result = predict(self.bundle, {**EXAMPLE, "spo2": 50, "spo2_min": 45})
        self.assertIsNone(result["screen_positive"])
        self.assertEqual(result["status"], "insufficient_evidence")
        json.dumps(result, allow_nan=False)

    def test_patient_disjoint_nested_validation(self):
        report = self.bundle["metadata"]["report"]
        development = set(report["split"]["development_patients"])
        evaluation = set(report["split"]["holdout_patients"])
        self.assertFalse(development & evaluation)
        for experiment in report["experiments"].values():
            seen = set()
            for fold in experiment["nested_development"]["folds"]:
                patients = set(fold["patients"])
                self.assertTrue(patients <= development)
                self.assertFalse(patients & evaluation)
                self.assertFalse(patients & seen)
                seen.update(patients)
            self.assertEqual(seen, development)
        self.assertIn("previously", report["evaluation_note"])


@unittest.skipUnless(PREDICTOR_PATH.exists(), "Train the current predictor first.")
class PredictorApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.key = "predictor-local-test-key-" * 3
        cls.headers = {"Authorization": "Bearer " + cls.key}
        cls.client = TestClient(create_app(api_key=cls.key))
        cls.client.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.client.__exit__(None, None, None)

    def test_auth_and_current_model(self):
        self.assertEqual(self.client.get("/v3/model").status_code, 401)
        response = self.client.get("/v3/model", headers=self.headers)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["features"], FEATURES)

    def test_request_matches_shared_prediction_and_keeps_context(self):
        response = self.client.post("/v3/screen", headers=self.headers, json=EXAMPLE)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), predict(load_predictor(), EXAMPLE))
        self.assertEqual(response.json()["context_not_used_by_model"], ["sleep_duration"])

    def test_context_can_be_omitted_but_not_sensors(self):
        body = {key: value for key, value in EXAMPLE.items() if key != "sleep_duration"}
        self.assertEqual(self.client.post("/v3/screen", headers=self.headers, json=body).status_code, 200)
        del body["spo2_std"]
        self.assertEqual(self.client.post("/v3/screen", headers=self.headers, json=body).status_code, 422)

    def test_bad_types_and_impossible_summary(self):
        for body in ({**EXAMPLE, "sleep_duration": True}, {**EXAMPLE, "spo2_min": "95"}):
            self.assertEqual(self.client.post("/v3/screen", headers=self.headers, json=body).status_code, 422)
        # Cross-field consistency is checked by shared feature validation; route returns 422.
        response = self.client.post("/v3/screen", headers=self.headers, json={**EXAMPLE, "spo2_min": 99})
        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
