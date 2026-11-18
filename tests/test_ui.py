"""Verify the single-form combined predictor as a user."""
import unittest

from streamlit.testing.v1 import AppTest

from src.combined import COMBINED_MODEL_PATH
from src.data import ROOT


@unittest.skipUnless(COMBINED_MODEL_PATH.exists(), "Train the combined model before UI tests.")
class DashboardTests(unittest.TestCase):
    def test_single_form_contains_all_joint_inputs(self):
        app = AppTest.from_file(str(ROOT / "ui.py"), default_timeout=20).run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.tabs), 0)
        self.assertEqual([item.label for item in app.number_input],
                         ["SpO₂ (%)", "Heart rate (bpm)", "Breathing rate (breaths/min)", "Perfusion index (%)"])
        self.assertTrue(app.warning)

    def test_required_inputs_and_example_prediction(self):
        app = AppTest.from_file(str(ROOT / "ui.py"), default_timeout=20).run()
        app.button(key="FormSubmitter:combined_screening-Predict").click().run()
        self.assertTrue(app.error)
        app.toggle[0].set_value(True).run()
        app.button(key="FormSubmitter:combined_screening-Predict").click().run()
        self.assertFalse(app.exception)
        self.assertTrue(app.session_state["combined_prediction"]["example"])
        self.assertIn(app.session_state["combined_prediction"]["result"]["status"],
                      ("positive", "negative", "insufficient_evidence"))

    def test_unsupported_measurement_is_not_a_negative_prediction(self):
        app = AppTest.from_file(str(ROOT / "ui.py"), default_timeout=20).run()
        app.toggle[0].set_value(True).run()
        app.number_input[0].set_value(50.0)
        app.button(key="FormSubmitter:combined_screening-Predict").click().run()
        self.assertFalse(app.exception)
        result = app.session_state["combined_prediction"]["result"]
        self.assertEqual(result["status"], "insufficient_evidence")
        self.assertIsNone(result["screen_positive"])
