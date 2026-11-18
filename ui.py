"""One form, one jointly trained sensor predictor, one screening result."""
from datetime import datetime, timezone
import json

import pandas as pd
import streamlit as st

from src.combined import (COMBINED_MODEL_PATH, DISCLAIMER, EXAMPLE, load_combined_model,
                          predict_combined)

st.set_page_config(page_title="SleepSense · Sleep apnea screening", page_icon="🌙", layout="centered")
st.title("SleepSense")
st.write("Enter your measurements to screen for a sleep-apnea-related breathing pattern.")
st.caption("Use averages from the same completed 60-second measurement period, ideally recorded during sleep.")


@st.cache_resource
def get_model(path, modified_ns):
    return load_combined_model(path)


try:
    bundle = get_model(str(COMBINED_MODEL_PATH), COMBINED_MODEL_PATH.stat().st_mtime_ns)
except (OSError, ValueError) as error:
    st.error(str(error))
    st.code("python combined.py train\nstreamlit run ui.py", language="bash")
    st.stop()

metadata = bundle["metadata"]
study = metadata["report"]
metrics = study["experiments"]["combined"]["holdout"]
st.warning("Experimental prototype: the current combined model has poor validation performance "
           "and is not reliable for health decisions.")
st.caption(DISCLAIMER)

use_example = st.toggle("Use example measurements", value=False)
values = EXAMPLE if use_example else {}
if use_example:
    st.info("Example values are active. Replace all values to screen your own measurements.")

with st.form("combined_screening"):
    left, right = st.columns(2)
    with left:
        spo2 = st.number_input("SpO₂ (%)", min_value=1.0, max_value=100.0,
                               value=values.get("spo2"), step=0.1,
                               help="Average oxygen saturation percentage over the measurement period.")
        pulse = st.number_input("Heart rate (bpm)", min_value=30.0, max_value=220.0,
                                value=values.get("heart_rate"), step=1.0,
                                help="Average pulse/heart rate over the same period.")
    with right:
        breathing = st.number_input("Breathing rate (breaths/min)", min_value=1.0, max_value=80.0,
                                    value=values.get("respiratory_rate"), step=0.1,
                                    help="Average respiratory rate over the same period.")
        perfusion = st.number_input("Perfusion index (%)", min_value=0.01, max_value=30.0,
                                    value=values.get("perfusion_index"), step=0.1,
                                    help="Average PI reported by the optical pulse oximeter.")
    submitted = st.form_submit_button("Predict", type="primary", width="stretch")

if submitted:
    record = {"spo2": spo2, "heart_rate": pulse, "respiratory_rate": breathing,
              "perfusion_index": perfusion}
    try:
        if any(value is None for value in record.values()):
            raise ValueError("Enter all four measurements before predicting.")
        result = predict_combined(bundle, record)
        snapshot = {"time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "example": use_example, "inputs": record, "result": result}
        st.session_state["combined_prediction"] = snapshot
        st.session_state.setdefault("combined_history", []).append(snapshot)
        st.session_state["combined_history"] = st.session_state["combined_history"][-20:]
    except ValueError as error:
        st.session_state.pop("combined_prediction", None)
        st.error(str(error))

if "combined_prediction" in st.session_state:
    snapshot = st.session_state["combined_prediction"]
    result = snapshot["result"]
    st.divider()
    if result["screen_positive"] is True:
        st.error(result["label"])
    elif result["screen_positive"] is False:
        st.info(result["label"])
    else:
        st.warning(result["label"])
    st.write(result["next_step"])
    for warning in result["warnings"]:
        st.warning(warning)
    st.caption("This is an experimental screening output, not confirmation or exclusion of sleep apnea.")
    st.download_button("Download result", json.dumps(snapshot, indent=2),
                       file_name="sleepsense-result.json", mime="application/json")

with st.expander("Model details and measured performance"):
    st.write("All four measurements enter the same scaled SVM. There is no separate oxygen-only "
             "prediction in this form and no blending of independently trained models.")
    nested = study["experiments"]["combined"]["nested_development"]["pooled"]
    st.dataframe(pd.DataFrame({
        "Metric": ["Accuracy", "Sensitivity", "Specificity", "Precision", "F1", "ROC AUC"],
        "Six-patient evaluation": [metrics[k] for k in ("accuracy", "recall", "specificity", "precision", "f1", "roc_auc")],
        "Nested development validation": [nested[k] for k in ("accuracy", "recall", "specificity", "precision", "f1", "roc_auc")],
    }), hide_index=True, width="stretch")
    coverage = study["quality"]["holdout"]
    st.write(f"Usable-window coverage: {coverage['coverage']:.1%}. Results apply to measurement "
             "intervals from a small stroke-unit dataset, rather than independently diagnosed new patients.")
    st.write("The source heart-rate and breathing-rate values came from clinical monitors. "
             "Using optical pulse and wearable breathing estimates needs device validation.")
    st.write("Normal measurements while awake cannot establish that breathing remains normal during sleep.")
    st.markdown("[Kaggle source](https://www.kaggle.com/datasets/menes97/osasud-sleep-apnea-original-parquet) · "
                "[Original publication](https://doi.org/10.1038/s41597-022-01272-y)")
    st.download_button("Download full model report", json.dumps(metadata, indent=2),
                       file_name="combined-model-report.json", mime="application/json")

with st.expander("Recent results"):
    history = st.session_state.get("combined_history", [])
    if history:
        st.dataframe(pd.DataFrame([{
            "Time (UTC)": row["time"], "Example": row["example"],
            "Result": row["result"]["label"],
        } for row in reversed(history)]), hide_index=True, width="stretch")
    else:
        st.write("Predictions from this browser session appear here.")
    if st.button("Clear results"):
        st.session_state.pop("combined_history", None)
        st.session_state.pop("combined_prediction", None)
        st.rerun()
