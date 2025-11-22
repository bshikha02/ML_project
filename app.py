import streamlit as st
import numpy as np
import pandas as pd
import joblib

# ------------------------------
# Load Model + Scaler
# ------------------------------
model_bundle = joblib.load("salary_prediction_model.pkl")
model = model_bundle["best_model"]
scaler = model_bundle["scaler"]

st.set_page_config(page_title="Salary Prediction App", layout="centered")

# ------------------------------
# App Title
# ------------------------------
st.title("💼 Salary Prediction Web App")
st.write("Enter details below to predict the estimated salary.")

# ------------------------------
# User Inputs (Customise based on your dataset)
# ------------------------------
# Example fields — replace with fields actually used in your model dataset
job_title = st.selectbox(
    "Job Title", 
    ["Software Engineer", "Data Scientist", "ML Engineer", "Manager", "Analyst"]
)

experience = st.number_input("Years of Experience", min_value=0.0, max_value=40.0, step=0.5)
age = st.number_input("Age", min_value=18, max_value=70)
education_level = st.selectbox("Education Level", ["Bachelors", "Masters", "PhD"])

# Convert categorical → numeric (customize according to your dataset)
mapping_job = {
    "Software Engineer": 0,
    "Data Scientist": 1,
    "ML Engineer": 2,
    "Manager": 3,
    "Analyst": 4
}

mapping_edu = {
    "Bachelors": 0,
    "Masters": 1,
    "PhD": 2
}

# ------------------------------
# Prediction Button
# ------------------------------
if st.button("Predict Salary"):
    # Arrange input in array
    input_data = np.array([[ 
        mapping_job[job_title],
        experience,
        age,
        mapping_edu[education_level]
    ]])

    # Scale the input
    scaled_data = scaler.transform(input_data)

    # Predict
    prediction = model.predict(scaled_data)[0]

    st.success(f"💰 Estimated Salary: **₹{prediction:,.2f}**")


# ------------------------------
# Footer
# ------------------------------
st.markdown("---")
st.write("Created by Shikha — Salary Prediction Machine Learning Project")


