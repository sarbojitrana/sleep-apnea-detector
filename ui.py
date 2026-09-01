import streamlit as st

from src.pipeline import load_or_train, predict

st.set_page_config(page_title="Sleep Apnea Screening", page_icon="😴")
st.title("Sleep Apnea Screening")
st.caption("Screening aid only. Not a diagnosis; confirm with a sleep study.")


@st.cache_resource
def get_model():
    return load_or_train()


model = get_model()

col1, col2 = st.columns(2)

with col1:
    st.subheader("Profile")
    gender = st.radio("Gender", ["Female", "Male"], horizontal=True)
    age = st.slider("Age", 18, 90, 40)
    bmi = st.select_slider("BMI category", ["Normal", "Overweight", "Obese"])

    st.subheader("Blood pressure")
    systolic = st.slider("Systolic", 90, 180, 125)
    diastolic = st.slider("Diastolic", 50, 120, 82)

with col2:
    st.subheader("Sleep")
    sleep = st.slider("Sleep duration (hours)", 3.0, 10.0, 7.0, step=0.1)
    quality = st.slider("Sleep quality (1-10)", 1, 10, 7)

    st.subheader("Daily activity and vitals")
    heart = st.slider("Resting heart rate (bpm)", 50, 120, 70)
    steps = st.slider("Daily steps", 1000, 20000, 7000, step=500)
    activity = st.slider("Physical activity (min/day)", 0, 120, 60)
    stress = st.slider("Stress level (1-10)", 1, 10, 5)

if st.button("Run screening", type="primary", use_container_width=True):
    result = predict(model, {
        "gender": gender,
        "age": age,
        "sleep_duration": sleep,
        "quality_of_sleep": quality,
        "physical_activity": activity,
        "stress": stress,
        "bmi": bmi,
        "heart_rate": heart,
        "daily_steps": steps,
        "systolic": systolic,
        "diastolic": diastolic,
    })

    st.divider()
    left, right = st.columns(2)
    left.metric("Prediction", "Likely apnea" if result["apnea"] else "No apnea")
    left.metric("Model confidence", f"{result['probability']:.1%}")
    right.metric("Rule-based risk", f"{result['risk_score']}/100", result["severity"])

    if result["risk_factors"]:
        st.write("**Risk factors detected**")
        for factor in result["risk_factors"]:
            st.write(f"- {factor}")
    else:
        st.write("No rule-based risk factors detected.")

    if result["disagreement"]:
        st.warning("High risk indicators present despite a negative prediction.")
