import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load model and encoders
model = joblib.load("models/xgboost_model.pkl")
encoders = joblib.load("models/encoders.pkl")

st.set_page_config(page_title="Salary Predictor", layout="centered")
st.title("💼 Employee Salary Prediction App")

# Input form
with st.form("salary_form"):
    st.subheader("📋 Enter Employee Details")

    age = st.number_input("Age", min_value=18, max_value=70, value=30)
    gender = st.selectbox("Gender", encoders['Gender'].classes_)
    dept = st.selectbox("Department", encoders['Department'].classes_)
    job = st.selectbox("Job Title", encoders['Job_Title'].classes_)
    experience = st.slider("Years of Experience", 0, 40, 5)
    education = st.selectbox("Education Level", encoders['Education_Level'].classes_)
    location = st.selectbox("Location", encoders['Location'].classes_)

    submitted = st.form_submit_button("Predict Salary")

# Predict
if submitted:
    input_dict = {
        "Age": age,
        "Gender": encoders['Gender'].transform([gender])[0],
        "Department": encoders['Department'].transform([dept])[0],
        "Job_Title": encoders['Job_Title'].transform([job])[0],
        "Experience_Years": experience,
        "Education_Level": encoders['Education_Level'].transform([education])[0],
        "Location": encoders['Location'].transform([location])[0]
    }

    input_df = pd.DataFrame([input_dict])
    salary = model.predict(input_df)[0]

    st.success(f"💰 Predicted Salary: ${salary:,.2f}")
