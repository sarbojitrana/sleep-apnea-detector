import csv
import io
import unittest

import numpy as np

from src.dashboard import HISTORY_LIMIT, history_frame, random_demo, remember_prediction
from src.predictor import EXAMPLE, build_features


class DashboardHistoryTests(unittest.TestCase):
    def test_generated_intervals_have_consistent_summaries(self):
        rng = np.random.default_rng(42)
        records = [random_demo(rng) for _ in range(100)]
        for record in records:
            features, _ = build_features(record)
            self.assertTrue(np.isfinite(features.to_numpy()).all())
            self.assertLess(record["spo2_min"], record["spo2"])
            self.assertGreater(record["spo2_std"], 0)
        self.assertGreater(len({row["spo2"] for row in records}), 90)

    def test_history_snapshots_bound_memory_and_export_all_measurements(self):
        state = {}
        record = dict(EXAMPLE)
        result = {"label": "Positive screening result", "model_id": "test-model"}
        for number in range(HISTORY_LIMIT + 2):
            record["sleep_duration"] = 5 + number / 100
            remember_prediction(state, record, result)
        record["sleep_duration"] = 12
        result["label"] = "changed"
        history = state["prediction_history"]
        self.assertEqual(len(history), HISTORY_LIMIT)
        self.assertEqual(history[0]["inputs"]["sleep_duration"], 5.02)
        self.assertEqual(history[-1]["result"]["label"], "Positive screening result")
        rows = list(csv.DictReader(io.StringIO(history_frame(history).to_csv(index=False))))
        self.assertEqual(len(rows), HISTORY_LIMIT)
        self.assertEqual(float(rows[0]["Sleep duration (h, context only)"]), 6.01)
        self.assertEqual(float(rows[0]["Average SpO₂ (%)"]), EXAMPLE["spo2"])
        self.assertEqual(rows[0]["Source"], "Manual entry")
        self.assertEqual(rows[0]["Model"], "test-model")
