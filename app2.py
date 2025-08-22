import joblib
import numpy as np
import streamlit as st

model = joblib.load("Student_performance.pkl")
scaler = joblib.load("scaler.pkl")


st.title("🎓 Student Performance Prediction")


no_hr_study = st.selectbox("Enter the number of hours you studied:", [1,2,3,4,5,6,7,8,9,10])
previous_score = st.number_input("Enter your previous score:")
sleep_hr = st.number_input("Enter the number of hours you sleep:")
sample_paper = st.number_input("Number of sample papers you solved:", 1, 15, 1)
activity = st.radio("Are you involved in extra curricular activities?", ["Yes", "No"])


activity_val = 1 if activity == "Yes" else 0
if st.button("Predict"):
    input_data = np.array([[no_hr_study, previous_score, sleep_hr, sample_paper, activity_val]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    st.success(f"Predicted Score: {prediction:.2f}")

