"""Screening dashboard with model evidence and private session history."""
import json

import pandas as pd
import streamlit as st

from src.dashboard import (HISTORY_LIMIT, clear_history, fill_inputs, history_frame,
                           performance_frame, remember_prediction)
from src.predictor import PREDICTOR_PATH, FEATURES, load_predictor, predict

st.set_page_config(page_title="SleepSense", page_icon="🌙", layout="centered")
st.title("SleepSense")
st.caption("Research screening only; a result cannot confirm or exclude sleep apnea.")


@st.cache_resource
def get_predictor(path, modified_ns):
    return load_predictor(path)


try:
    bundle = get_predictor(str(PREDICTOR_PATH), PREDICTOR_PATH.stat().st_mtime_ns)
except (OSError, ValueError):
    st.error("The screening model is unavailable.")
    st.stop()

metadata = bundle["metadata"]
report = metadata["report"]
metrics = report["experiments"]["combined_with_oxygen_dynamics"]["holdout"]
for column, (label, key) in zip(st.columns(3),
                              [("Accuracy", "accuracy"), ("Sensitivity", "recall"), ("Specificity", "specificity")]):
    column.metric(label, f"{metrics[key]:.1%}")
st.caption("Measured on usable 60-second intervals from six evaluation patients. These are model-study metrics, not confidence in your result.")

screen_tab, performance_tab, history_tab = st.tabs(["Predict", "Performance", "History"])
with screen_tab:
    st.write("Enter summaries from the same completed 60-second recording interval.")
    example_col, random_col = st.columns(2)
    example_col.button("Enter example data", key="example_data", width="stretch",
                       on_click=fill_inputs, args=(st.session_state, "Example demo"))
    random_col.button("Enter random data", key="random_data", width="stretch",
                      on_click=fill_inputs, args=(st.session_state, "Random demo"))
    if st.session_state.get("prefill"):
        st.info("Demo values were loaded. Generated data is for exploring the interface; it is not a patient's measurements or validation evidence.")

    with st.form("screening"):
        left, right = st.columns(2)
        with left:
            spo2 = st.number_input("Average SpO₂ (%)", min_value=1.0, max_value=100.0,
                                   value=None, step=0.1, key="spo2",
                                   help="Average over the same completed 60-second interval as the other sensor values.")
            lowest = st.number_input("Lowest SpO₂ (%)", min_value=1.0, max_value=100.0,
                                      value=None, step=0.1, key="spo2_min",
                                      help="Minimum valid oxygen saturation in that same interval.")
            variation = st.number_input("SpO₂ standard deviation (percentage points)", min_value=0.0, max_value=50.0,
                                         value=None, step=0.1, key="spo2_std",
                                         help="Population standard deviation of valid one-second oxygen readings in the interval.")
            duration = st.number_input("Sleep duration (hours, context only)", min_value=0.5, max_value=16.0,
                                        value=None, step=0.1, key="sleep_duration",
                                        help="Not yet trained: saved with the result but not used by this model because matched sleep-duration data is unavailable.")
        with right:
            heart = st.number_input("Heart rate (bpm)", min_value=30.0, max_value=220.0,
                                     value=None, step=1.0, key="heart_rate", help="Average over the same interval.")
            breathing = st.number_input("Breathing rate (breaths/min)", min_value=1.0, max_value=80.0,
                                          value=None, step=0.1, key="respiratory_rate", help="Average over the same interval.")
            perfusion = st.number_input("Perfusion index (%)", min_value=0.01, max_value=30.0,
                                         value=None, step=0.1, key="perfusion_index", help="Average optical perfusion index over the same interval.")
        submitted = st.form_submit_button("Predict", type="primary", width="stretch")

    if submitted:
        record = {"spo2": spo2, "heart_rate": heart, "respiratory_rate": breathing,
                  "perfusion_index": perfusion, "spo2_min": lowest, "spo2_std": variation,
                  "sleep_duration": duration}
        try:
            if any(record[name] is None for name in FEATURES):
                raise ValueError("Enter all six sensor parameters before predicting.")
            result = predict(bundle, record)
            remember_prediction(st.session_state, record, result)
        except ValueError as error:
            st.session_state.pop("prediction", None)
            st.error(str(error))

    if "prediction" in st.session_state:
        snapshot = st.session_state["prediction"]
        result = snapshot["result"]
        st.divider()
        st.caption("Latest submitted result · " + snapshot["source"])
        if result["screen_positive"] is False:
            st.info(result["label"])
        else:
            st.warning(result["label"])
        st.write(result["next_step"])
        for warning in result["warnings"]:
            st.warning(warning)
        if "sleep_duration" in result["context"]:
            st.caption(f"Sleep duration: {result['context']['sleep_duration']:g} hours, context only; not used in this prediction.")

with performance_tab:
    st.subheader("Measured model performance")
    st.dataframe(performance_frame(metadata).style.format({
        "Six-patient evaluation": "{:.3f}", "Nested development validation": "{:.3f}"}),
        hide_index=True, width="stretch")
    st.caption("All scores range from 0 to 1. Sensitivity is the fraction of eligible positive intervals flagged; specificity is the fraction of eligible negative intervals correctly rejected.")
    st.write("Evaluation confusion matrix")
    st.dataframe(pd.DataFrame(metrics["confusion_matrix"],
                             index=["Actual negative", "Actual positive"],
                             columns=["Predicted negative", "Predicted positive"]), width="stretch")
    quality = report["quality"]["holdout"]
    st.write(f"Signal-quality coverage: **{quality['coverage']:.1%}** "
             f"({quality['eligible_windows']:,} usable intervals; {quality['abstained_windows']:,} excluded). "
             f"False alarms: **{metrics['false_positives']:,}**. Missed eligible positive intervals: **{metrics['false_negatives']:,}**.")
    st.warning("Exploratory results from a small, previously examined stroke-unit cohort. "
               "Intervals from one person are correlated. External validation and testing with the actual IoT sensors are still needed.")
    st.caption("The source heart-rate and breathing-rate measurements came from clinical monitors. "
               "Sleep duration is context only. Poor-quality recordings are insufficient evidence, never a negative result.")
    st.caption(f"Loaded model: {metadata['model_id']} · trained {metadata['trained_at']} · six sensor inputs")
    st.download_button("Download model report", json.dumps(metadata, indent=2),
                       file_name="sleepsense-model-report.json", mime="application/json")

with history_tab:
    st.subheader("Prediction history")
    st.caption(f"Latest {HISTORY_LIMIT} submissions in this browser session only. "
               "History is not shared with other sessions or saved to a database; it resets when the session ends. "
               "Download it to keep a copy. Demo entries are labelled.")
    history = st.session_state.get("prediction_history", [])
    if history:
        table = history_frame(history)
        st.dataframe(table, hide_index=True, width="stretch")
        st.download_button("Download history CSV", table.to_csv(index=False),
                           file_name="sleepsense-history.csv", mime="text/csv")
    else:
        st.write("Your submitted measurements and results will appear here.")
    st.button("Clear history", key="clear_history", disabled=not history,
              on_click=clear_history, args=(st.session_state,))
