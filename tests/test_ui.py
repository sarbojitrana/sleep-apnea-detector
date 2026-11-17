"""Exercise the dashboard as a user, including its failure and follow-up paths."""
import unittest

from streamlit.testing.v1 import AppTest

from src.data import ROOT
from src.pipeline import MODEL_PATH


@unittest.skipUnless(MODEL_PATH.exists(), "Train a model with python app.py train before UI tests.")
class DashboardTests(unittest.TestCase):
    def test_oxygen_research_tab_loads(self):
        app = AppTest.from_file(str(ROOT / "ui.py"), default_timeout=20).run()
        self.assertFalse(app.exception)
        self.assertIn("Overnight SpO₂ · research", [tab.label for tab in app.tabs])

    def test_required_inputs_and_example_screening(self):
        app = AppTest.from_file(str(ROOT / "ui.py"), default_timeout=20).run()
        self.assertFalse(app.exception)
        app.button(key="FormSubmitter:screening-Run screening").click().run()
        self.assertTrue(app.error)
        app.toggle[0].set_value(True).run()
        app.button(key="FormSubmitter:screening-Run screening").click().run()
        self.assertFalse(app.exception)
        self.assertIn("single_result", app.session_state)
        self.assertTrue(app.session_state["single_result"]["example"])

    def test_outside_training_support_and_symptom_follow_up(self):
        app = AppTest.from_file(str(ROOT / "ui.py"), default_timeout=20).run()
        app.toggle[0].set_value(True).run()
        app.number_input[0].set_value(70)
        app.checkbox[1].check()
        app.button(key="FormSubmitter:screening-Run screening").click().run()
        self.assertFalse(app.exception)
        result = app.session_state["single_result"]["result"]
        self.assertEqual(result["status"], "insufficient_evidence")
        self.assertTrue(result["follow_up_recommended"])
        self.assertTrue(app.warning)
