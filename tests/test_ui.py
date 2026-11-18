import unittest

from streamlit.testing.v1 import AppTest

from src.data import ROOT
from src.predictor import EXAMPLE, PREDICTOR_PATH, load_predictor


@unittest.skipUnless(PREDICTOR_PATH.exists(), "Train the predictor before UI checks.")
class DashboardTests(unittest.TestCase):
    def test_interface_exposes_parameters_metrics_and_session_tools(self):
        app = AppTest.from_file(str(ROOT / "ui.py"), default_timeout=20).run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.number_input), 7)
        self.assertEqual([tab.label for tab in app.tabs], ["Predict", "Performance", "History"])
        self.assertEqual(len(app.toggle), 0)
        self.assertEqual(len(app.expander), 0)
        self.assertEqual([button.label for button in app.button],
                         ["Enter example data", "Enter random data", "Predict", "Clear history"])
        measured = load_predictor()["metadata"]["report"]["experiments"]["combined_with_oxygen_dynamics"]["holdout"]
        self.assertEqual([item.value for item in app.metric],
                         [f"{measured[key]:.1%}" for key in ("accuracy", "recall", "specificity")])

    def test_example_random_history_and_clear(self):
        app = AppTest.from_file(str(ROOT / "ui.py"), default_timeout=20).run()
        app.button(key="example_data").click().run()
        for name, value in EXAMPLE.items():
            self.assertEqual(app.number_input(key=name).value, value)
        self.assertFalse(app.exception)
        app.button(key="FormSubmitter:screening-Predict").click().run()
        self.assertEqual(app.session_state["prediction_history"][0]["source"], "Example demo")
        first = dict(app.session_state["prediction_history"][0]["inputs"])
        app.button(key="random_data").click().run()
        self.assertNotIn("prediction", app.session_state)
        self.assertEqual(app.session_state["prediction_history"][0]["inputs"], first)
        app.button(key="FormSubmitter:screening-Predict").click().run()
        self.assertFalse(app.exception)
        history = app.session_state["prediction_history"]
        self.assertEqual(len(history), 2)
        self.assertEqual(history[-1]["source"], "Random demo")
        self.assertLessEqual(history[-1]["inputs"]["spo2_min"], history[-1]["inputs"]["spo2"])
        app.run()
        self.assertEqual(len(app.session_state["prediction_history"]), 2)
        other = AppTest.from_file(str(ROOT / "ui.py"), default_timeout=20).run()
        self.assertNotIn("prediction_history", other.session_state)
        app.button(key="clear_history").click().run()
        self.assertFalse(app.exception)
        self.assertNotIn("prediction_history", app.session_state)
        self.assertNotIn("prediction", app.session_state)

    def test_editing_a_demo_keeps_its_origin_in_history(self):
        app = AppTest.from_file(str(ROOT / "ui.py"), default_timeout=20).run()
        app.button(key="example_data").click().run()
        app.number_input(key="sleep_duration").set_value(8.0)
        app.button(key="FormSubmitter:screening-Predict").click().run()
        self.assertEqual(app.session_state["prediction_history"][-1]["source"], "Edited example demo")

    def test_required_parameters_and_context(self):
        app = AppTest.from_file(str(ROOT / "ui.py"), default_timeout=20).run()
        app.button(key="FormSubmitter:screening-Predict").click().run()
        self.assertTrue(app.error)
        self.assertNotIn("prediction_history", app.session_state)
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
