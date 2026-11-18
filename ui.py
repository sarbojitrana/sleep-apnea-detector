"""Parameter fields and one screening action."""
import streamlit as st

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

with st.form("screening"):
    left, right = st.columns(2)
    with left:
        spo2 = st.number_input("Average SpO₂ (%)", min_value=1.0, max_value=100.0,
                               value=None, step=0.1, key="spo2",
                               help="Average over the same completed 60-second interval as the other sensor values.")
        lowest = st.number_input("Lowest SpO₂ (%)", min_value=1.0, max_value=100.0,
                                  value=None, step=0.1, key="spo2_min",
                                  help="Minimum measured oxygen saturation in that same 60-second interval.")
        variation = st.number_input("SpO₂ standard deviation (percentage points)", min_value=0.0, max_value=50.0,
                                     value=None, step=0.1, key="spo2_std",
                                     help="Population standard deviation of valid one-second oxygen readings in the interval.")
        duration = st.number_input("Sleep duration (hours, context only)", min_value=0.5, max_value=16.0,
                                    value=None, step=0.1, key="sleep_duration",
                                    help="Not yet trained: saved with the result but not used by this model because matched sleep-duration data is unavailable.")
    with right:
        heart = st.number_input("Heart rate (bpm)", min_value=30.0, max_value=220.0,
                                 value=None, step=1.0, key="heart_rate",
                                 help="Average over the same 60-second interval.")
        breathing = st.number_input("Breathing rate (breaths/min)", min_value=1.0, max_value=80.0,
                                      value=None, step=0.1, key="respiratory_rate",
                                      help="Average over the same 60-second interval.")
        perfusion = st.number_input("Perfusion index (%)", min_value=0.01, max_value=30.0,
                                     value=None, step=0.1, key="perfusion_index",
                                     help="Average optical perfusion index over the same interval.")
    submitted = st.form_submit_button("Predict", type="primary", width="stretch")

if submitted:
    record = {"spo2": spo2, "heart_rate": heart, "respiratory_rate": breathing,
              "perfusion_index": perfusion, "spo2_min": lowest, "spo2_std": variation,
              "sleep_duration": duration}
    try:
        if any(record[name] is None for name in FEATURES):
            raise ValueError("Enter all six sensor parameters before predicting.")
        result = predict(bundle, record)
        st.session_state["prediction"] = {"inputs": record, "result": result}
    except ValueError as error:
        st.session_state.pop("prediction", None)
        st.error(str(error))

if "prediction" in st.session_state:
    result = st.session_state["prediction"]["result"]
    if result["screen_positive"] is True:
        st.warning(result["label"])
    elif result["screen_positive"] is False:
        st.info(result["label"])
    else:
        st.warning(result["label"])
    st.write(result["next_step"])
    for warning in result["warnings"]:
        st.warning(warning)
    if "sleep_duration" in result["context"]:
        st.caption(f"Sleep duration: {result['context']['sleep_duration']:g} hours, context only; not used in this prediction.")
