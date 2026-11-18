import unittest

from fastapi.testclient import TestClient

from api import create_app
from src.predictor import (PREDICTOR_PATH, load_predictor, predict, predict_sensor_window,
                           summarize_sensor_window)


def signals():
    return {"spo2": [96.] * 30 + [98.] * 30, "heart_rate": [65.] * 60,
            "respiratory_rate": [16.] * 60, "perfusion_index": [3.5] * 60}


class SensorWindowTests(unittest.TestCase):
    def test_computes_six_training_equivalent_summaries(self):
        record, quality = summarize_sensor_window(signals())
        self.assertEqual(record, {"spo2": 97., "spo2_min": 96., "spo2_std": 1.,
                                  "heart_rate": 65., "respiratory_rate": 16., "perfusion_index": 3.5})
        self.assertEqual(len(quality), 4)
        self.assertTrue(all(row["valid_fraction"] == 1 for row in quality.values()))

    def test_bad_quality_in_any_channel_abstains(self):
        for name in signals():
            with self.subTest(channel=name):
                values = signals()
                values[name][:6] = [None] * 6  # 90% valid but gap too long.
                record, quality = summarize_sensor_window(values)
                self.assertIsNone(record)
                self.assertEqual(quality[name]["longest_missing_run"], 6)
                values = signals()
                for index in range(0, 21, 3):
                    values[name][index] = None  # 53/60 valid, short gaps.
                self.assertIsNone(summarize_sensor_window(values)[0])

    def test_small_gaps_are_excluded_not_imputed(self):
        values = signals()
        values["spo2"][0] = None
        record, quality = summarize_sensor_window(values)
        self.assertAlmostEqual(record["spo2"], (96 * 29 + 98 * 30) / 59)
        self.assertAlmostEqual(quality["spo2"]["valid_fraction"], 59 / 60)

    def test_missing_wrong_length_and_nonnumeric_channels_are_rejected(self):
        for bad in ({}, {**signals(), "spo2": [97.] * 59},
                    {**signals(), "spo2": [True] * 60}, {**signals(), "spo2": ["97"] * 60}):
            with self.subTest(bad_keys=list(bad)), self.assertRaises(ValueError):
                summarize_sensor_window(bad)


@unittest.skipUnless(PREDICTOR_PATH.exists(), "Train the predictor first.")
class SensorWindowApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = load_predictor()
        cls.headers = {"Authorization": "Bearer " + "local-window-test-key-" * 3}
        cls.client = TestClient(create_app(api_key="local-window-test-key-" * 3))
        cls.client.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.client.__exit__(None, None, None)

    def test_window_route_matches_manual_summary_prediction(self):
        response = self.client.post("/v3/window", headers=self.headers,
                                    json={"channels": signals(), "sleep_duration": 7})
        self.assertEqual(response.status_code, 200)
        result = response.json()
        expected = predict(self.bundle, {**result["sensor_features"], "sleep_duration": 7})
        self.assertEqual(result["decision_margin"], expected["decision_margin"])
        self.assertEqual(result["screen_positive"], expected["screen_positive"])
        self.assertEqual(result["context_not_used_by_model"], ["sleep_duration"])
        self.assertEqual(result["window_seconds"], 60)
        self.assertEqual(len(result["signal_quality"]), 4)

    def test_quality_failure_is_not_a_negative_screen(self):
        values = signals()
        values["perfusion_index"] = [None] * 60
        result = predict_sensor_window(self.bundle, values)
        self.assertEqual(result["status"], "insufficient_evidence")
        self.assertIsNone(result["screen_positive"])
        response = self.client.post("/v3/window", headers=self.headers, json={"channels": values})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), result)

    def test_authentication_and_strict_window_schema(self):
        self.assertEqual(self.client.post("/v3/window", json={"channels": signals()}).status_code, 401)
        for body in ({"channels": {**signals(), "spo2": [True] * 60}},
                     {"channels": {**signals(), "spo2": [97.] * 61}},
                     {"channels": signals(), "ahi": 20}):
            self.assertEqual(self.client.post("/v3/window", headers=self.headers, json=body).status_code, 422)
