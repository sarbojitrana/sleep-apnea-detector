"""Streamlit dashboard over the same screening contract used by the device API."""
from datetime import datetime, timezone
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src.batch import MAX_UPLOAD_BYTES, parse_upload, screen_batch
from src.data import EXAMPLE_RECORD, INPUT_COLUMNS, ROOT
from src.pipeline import MODEL_PATH, load_model, predict
from src.utils import DISCLAIMER

st.set_page_config(page_title="SleepSense · Sleep apnea screening", page_icon="🌙", layout="wide")
st.title("SleepSense")
st.caption("Accessible sleep apnea screening · SVM research prototype")
st.info("A first check using everyday measurements. This prototype estimates a dataset pattern; "
        "it cannot diagnose apnea, measure its severity, or detect breathing emergencies.")


@st.cache_resource
def get_model(path, modified_ns):
    # File timestamp invalidates the cache after explicit retraining.
    return load_model(path)


try:
    bundle = get_model(str(MODEL_PATH), MODEL_PATH.stat().st_mtime_ns)
except (FileNotFoundError, ValueError) as error:
    st.error(str(error))
    st.code("python app.py train\nstreamlit run ui.py", language="bash")
    st.stop()

metadata = bundle["metadata"]
with st.sidebar:
    st.header("Your screening workspace")
    st.write("**Inputs:** age, dataset gender category, BMI category, sleep duration, resting pulse, daily steps.")
    st.caption("Profile details are entered once. Pulse can come from an optical PPG sensor; "
               "steps and sleep estimates can come from a wearable or be entered manually.")
    st.divider()
    st.write("**Model loaded**")
    st.code(metadata["model_id"], language=None)
    st.caption(f"Trained {metadata['trained_at'][:10]} · {len(metadata['features'])} inputs")
    st.caption("Inputs and screening history are kept in this session only by this application. "
               "Downloads are saved only when you request them.")
    if st.button("Clear session results", width="stretch"):
        for key in ("single_result", "batch_results", "history"):
            st.session_state.pop(key, None)
        st.rerun()

screen_tab, import_tab, history_tab, model_tab = st.tabs(
    ["Screening", "Device import", "Session history", "Model & evidence"])

STATUS_LABELS = {
    "raised_signal": "Raised screening signal",
    "lower_signal": "Lower screening signal",
    "insufficient_evidence": "Insufficient evidence",
    "invalid_input": "Invalid input",
}


def show_result(snapshot):
    result = snapshot["result"]
    st.subheader(STATUS_LABELS[result["status"]])
    st.caption(f"Screened at {snapshot['time']} · Model {result['model_id']}")
    if result["follow_up_recommended"]:
        st.warning(result["next_step"])
    else:
        st.info(result["next_step"])
    for warning in result["warnings"]:
        st.warning(warning)
    if result["reported_symptoms"]:
        st.write("**Reported symptoms requiring attention**")
        for symptom in result["reported_symptoms"]:
            st.write(f"• {symptom}")
        st.caption("Symptoms are follow-up prompts, separate from the SVM. They are not a validated questionnaire score.")
    with st.expander("Inputs and technical result"):
        st.json(snapshot)
        st.caption("The decision margin is a signed SVM score. It is not a probability, confidence, or severity.")
    st.download_button("Download this screening report", json.dumps(snapshot, indent=2),
                       file_name="sleep-screening.json", mime="application/json", key="single_download")


with screen_tab:
    st.subheader("Start with a daily summary")
    st.write("Use the main sleep period, a resting pulse measurement, and the full day's step total. "
             "A single pulse reading during activity is not a resting pulse.")
    use_example = st.toggle("Use example values", value=False,
                            help="Examples demonstrate the app. Replace every value with your own measurements.")
    values = EXAMPLE_RECORD if use_example else {}
    if use_example:
        st.warning("Example data is active. Any result describes the example inputs.")
    with st.form("screening"):
        left, right = st.columns(2)
        with left:
            st.markdown("#### Profile")
            gender = st.selectbox("Gender category used by the dataset", ["Female", "Male"],
                                  index=0 if use_example else None,
                                  help="The source has only these two categories; generalization beyond them is unvalidated.")
            age = st.number_input("Age (years)", 18, 100, value=values.get("age"), step=1)
            bmi = st.selectbox("BMI category", ["Normal", "Overweight", "Obese"],
                               index=0 if use_example else None,
                               help="Use your measured height/weight and an appropriate clinical classification; "
                                    "the source's category cutoffs are not documented.")
        with right:
            st.markdown("#### Everyday measurements")
            sleep = st.number_input("Sleep duration (hours/night)", 0.5, 16.0,
                                    value=values.get("sleep_duration"), step=0.1)
            heart = st.number_input("Resting pulse (bpm)", 30.0, 220.0,
                                    value=float(values["heart_rate"]) if use_example else None, step=1.0)
            steps = st.number_input("Daily steps (steps/day)", 0, 100000,
                                    value=values.get("daily_steps"), step=100)
        st.markdown("#### Symptoms you have noticed")
        st.caption("Optional. Unchecked boxes are not evidence that symptoms are absent.")
        snoring = st.checkbox("Persistent loud snoring")
        pauses = st.checkbox("Someone has noticed breathing pauses, gasping, or choking during sleep")
        sleepy = st.checkbox("Persistent daytime sleepiness")
        submitted = st.form_submit_button("Run screening", type="primary", width="stretch")

    if submitted:
        record = {
            "gender": gender, "age": age, "sleep_duration": sleep, "bmi": bmi,
            "heart_rate": heart, "daily_steps": steps, "loud_snoring": snoring,
            "witnessed_pauses": pauses, "daytime_sleepiness": sleepy,
        }
        try:
            result = predict(bundle, record)
            snapshot = {"time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                        "example": use_example, "inputs": record, "result": result}
            st.session_state["single_result"] = snapshot
            st.session_state.setdefault("history", []).append(snapshot)
            st.session_state["history"] = st.session_state["history"][-20:]
        except ValueError as error:
            st.session_state.pop("single_result", None)
            st.error(str(error))
    if "single_result" in st.session_state:
        st.divider()
        show_result(st.session_state["single_result"])

with import_tab:
    st.subheader("Import daily device summaries")
    st.write("Upload CSV or JSON containing up to 100 daily summaries (1 MB maximum). "
             "Each row must contain all six required inputs. Raw PPG, ECG, EEG, oxygen samples, "
             "and audio are not supported by this trained model.")
    st.caption("Heart rate must already be a valid resting-pulse estimate. Sleep duration must already "
               "be estimated by your device or entered manually; this app does not derive sleep from motion.")
    col1, col2 = st.columns(2)
    col1.download_button("Download CSV example", (ROOT / "examples/daily_summaries.csv").read_bytes(),
                         file_name="daily_summaries.csv", mime="text/csv")
    col2.download_button("Download JSON example", (ROOT / "examples/daily_summary.json").read_bytes(),
                         file_name="daily_summary.json", mime="application/json")
    uploaded = st.file_uploader("Daily summaries", type=["csv", "json"])
    if st.button("Screen imported summaries", disabled=uploaded is None):
        st.session_state.pop("batch_results", None)
        try:
            if uploaded.size > MAX_UPLOAD_BYTES:
                raise ValueError("Upload must be at most 1 MB.")
            records = parse_upload(uploaded.getvalue(), Path(uploaded.name).suffix)
            st.session_state["batch_results"] = screen_batch(bundle, records)
        except ValueError as error:
            st.error(str(error))
    if "batch_results" in st.session_state:
        results = st.session_state["batch_results"]
        st.dataframe(pd.DataFrame([{
            "Row": r["row"], "Result": STATUS_LABELS[r["status"]],
            "Follow-up prompted": r.get("follow_up_recommended"),
            "Details": r.get("error") or "; ".join(r.get("warnings", [])) or r.get("next_step"),
        } for r in results]), hide_index=True, width="stretch")
        st.download_button("Download all results", json.dumps(results, indent=2),
                           file_name="screening-results.json", mime="application/json")

with history_tab:
    st.subheader("Recent screenings in this session")
    history = st.session_state.get("history", [])
    if not history:
        st.write("Completed manual screenings will appear here. Device imports have a separate download.")
    else:
        st.dataframe(pd.DataFrame([{
            "Time (UTC)": item["time"], "Example": item["example"],
            "Result": STATUS_LABELS[item["result"]["status"]],
            "Follow-up prompted": item["result"]["follow_up_recommended"],
        } for item in reversed(history)]), hide_index=True, width="stretch")
        st.download_button("Download session reports", json.dumps(history, indent=2),
                           file_name="sleep-screening-session.json", mime="application/json")
    st.caption("Up to 20 manual screenings are retained for this browser session. This is not longitudinal clinical monitoring.")

with model_tab:
    st.subheader("What the evidence supports")
    st.write("This is a model of labels in the Kaggle Sleep Health and Lifestyle dataset. "
             "No independently verified sleep-study ground truth or external device validation is bundled.")
    metrics = metadata["metrics"]
    a, b, c = st.columns(3)
    a.metric("Dataset rows", metrics["dataset"]["rows"])
    b.metric("Distinct input profiles", metrics["dataset"]["unique_profiles"])
    c.metric("Conflicting-label profiles", metrics["dataset"]["conflicting_binary_label_profiles"])
    st.write("Repeated profiles stay in the same fold, including profiles with conflicting labels. "
             "Nested development validation retunes the SVM in each outer fold. "
             "The saved classifier is trained on development rows only; holdout rows remain unseen.")
    st.markdown("#### Dataset evaluation · raw SVM")
    left, right = st.columns(2)
    left.metric("Grouped holdout sensitivity", f"{metrics['recall']:.1%}")
    left.metric("Grouped holdout specificity", f"{metrics['specificity']:.1%}")
    nested = metrics["nested_development_cv"]["pooled"]
    right.metric("Nested development sensitivity", f"{nested['recall']:.1%}")
    right.metric("Nested development specificity", f"{nested['specificity']:.1%}")
    st.dataframe(pd.DataFrame(metrics["confusion_matrix"],
                 index=["Actual other label", "Actual apnea label"],
                 columns=["Predicted other label", "Predicted apnea label"]), width="stretch")
    interval = metrics["sensitivity_interval_95"]
    st.caption(f"Holdout: {metrics['holdout']['rows']} rows, {metrics['holdout']['unique_profiles']} profiles. "
               f"Row-level sensitivity interval: {interval[0]:.1%}–{interval[1]:.1%}. "
               "These are descriptive dataset metrics, not clinical accuracy. Repeated rows reduce independence.")
    st.markdown("#### Training support")
    rows = [{"Input": key, "Training minimum": metadata["feature_bounds"][column][0],
             "Training maximum": metadata["feature_bounds"][column][1]}
            for key, column in INPUT_COLUMNS.items() if key not in ("gender", "bmi")]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.write("Inputs outside these ranges receive “Insufficient evidence”. Being within the ranges "
             "does not establish that a person or device resembles the training population.")
    st.write("Optical pulse sensing and wearable sleep estimates need validation on real recordings. "
             "The current model does not use SpO₂, detect apnea events, or compute an apnea–hypopnea index.")
    st.markdown("[Dataset source](https://www.kaggle.com/datasets/uom190346a/sleep-health-and-lifestyle-dataset) · "
                "[AASM diagnostic guideline](https://aasm.org/resources/clinicalguidelines/diagnostic-testing-osa.pdf) · "
                "[FDA pulse oximeter limitations](https://www.fda.gov/consumers/consumer-updates/pulse-oximeter-basics)")
    with st.expander("Full reproducible training report"):
        st.json(metadata)
    st.download_button("Download model report", json.dumps(metadata, indent=2),
                       file_name="model-report.json", mime="application/json")

st.divider()
st.caption(DISCLAIMER)
