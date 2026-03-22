import streamlit as st

from src.pipeline import train_pipeline, predict

st.title("Sleep Apnea Detection System")

# Train once
model, scaler = train_pipeline()

st.header("Enter Patient Details")

age = st.slider("Age", 18, 80, 30)
sleep = st.slider("Sleep Duration", 3.0, 10.0, 6.0)
stress = st.slider("Stress Level", 1, 10, 5)
heart = st.slider("Heart Rate", 50, 120, 70)
bmi = st.selectbox("BMI Category", [0, 1, 2])

if st.button("Predict"):
    input_data = {
        "Gender": 0,
        "Age": age,
        "Occupation": 0,
        "Sleep Duration": sleep,
        "Quality of Sleep": 5,
        "Physical Activity Level": 30,
        "Stress Level": stress,
        "BMI Category": bmi,
        "Heart Rate": heart,
        "Daily Steps": 5000,
        "Systolic": 120,
        "Diastolic": 80
    }

    pred, risk, severity = predict(model, scaler, input_data)

    st.subheader("Result")
    st.write("Apnea:", "YES" if pred == 1 else "NO")
    st.write("Risk Score:", risk)
    st.write("Severity:", severity)

    if pred == 0 and risk >= 60:
        st.warning("High risk indicators despite negative prediction")