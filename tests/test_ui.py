import unittest

from streamlit.testing.v1 import AppTest

from src.data import ROOT
from src.predictor import EXAMPLE, PREDICTOR_PATH


@unittest.skipUnless(PREDICTOR_PATH.exists(), "Train the predictor before UI checks.")
class DashboardTests(unittest.TestCase):
    def test_interface_contains_only_parameters_and_predict(self):
        app = AppTest.from_file(str(ROOT / "ui.py"), default_timeout=20).run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.number_input), 7)
        self.assertEqual(len(app.tabs), 0)
        self.assertEqual(len(app.toggle), 0)
        self.assertEqual(len(app.expander), 0)
        self.assertEqual([button.label for button in app.button], ["Predict"])

    def test_required_parameters_and_context(self):
        app = AppTest.from_file(str(ROOT / "ui.py"), default_timeout=20).run()
        app.button(key="FormSubmitter:screening-Predict").click().run()
        self.assertTrue(app.error)
        for field, value in EXAMPLE.items():
            app.number_input(key=field).set_value(value)
        app.button(key="FormSubmitter:screening-Predict").click().run()
        self.assertFalse(app.exception)
        self.assertIn(app.session_state["prediction"]["result"]["status"],
                      ("positive", "negative", "insufficient_evidence"))
        self.assertEqual(app.session_state["prediction"]["result"]["context_not_used_by_model"],
                         ["sleep_duration"])

    def test_impossible_oxygen_summary_is_rejected(self):
        app = AppTest.from_file(str(ROOT / "ui.py"), default_timeout=20).run()
        for field, value in EXAMPLE.items():
            app.number_input(key=field).set_value(value)
        app.number_input(key="spo2_min").set_value(99.0)
        app.button(key="FormSubmitter:screening-Predict").click().run()
        self.assertFalse(app.exception)
        self.assertTrue(app.error)
