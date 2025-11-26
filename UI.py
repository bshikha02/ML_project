import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Salary Predictor", layout="wide")

st.title("💼 Salary Prediction App")
st.write("Enter candidate/job details to predict salary.")

# ------------------------------
# Load Saved Model Bundle
# ------------------------------
@st.cache_resource
def load_bundle():
    bundle = joblib.load("model_bundle.pkl")   # Change name if different
    return bundle

try:
    bundle = load_bundle()
    model = bundle["best_model"]
    scaler = bundle["scaler"]
    feature_columns = bundle["feature_columns"]
except:
    st.error("❌ Could not load model bundle. Ensure `model_bundle.pkl` is in the same folder.")
    st.stop()

# ------------------------------
# INPUT FIELDS
# ------------------------------

col1, col2, col3 = st.columns(3)

with col1:
    work_year = st.number_input("Work Year", min_value=1990, max_value=2030, value=2024)

    experience_level = st.selectbox(
        "Experience Level",
        ["EN", "MI", "SE", "EX"],
        help="EN = Entry, MI = Mid, SE = Senior, EX = Expert"
    )

    employment_type = st.selectbox(
        "Employment Type",
        ["FT", "PT", "CT", "FL"],
        help="FT = Full-Time, PT = Part-Time, CT = Contract, FL = Freelance"
    )

with col2:
    job_title = st.text_input("Job Title (exact match from dataset)", value="Data Scientist")

    salary_currency = st.text_input("Salary Currency", value="USD")

    employee_residence = st.text_input("Employee Residence", value="US")

with col3:
    remote_ratio = st.slider("Remote Work (%)", 0, 100, 100)

    company_location = st.text_input("Company Location", value="US")

    company_size = st.selectbox("Company Size", ["S", "M", "L"], help="S=Small, M=Medium, L=Large")


# ------------------------------
# Convert Inputs to DataFrame
# ------------------------------

input_data = pd.DataFrame([{
    "work_year": work_year,
    "experience_level": experience_level,
    "employment_type": employment_type,
    "job_title": job_title,
    "salary_currency": salary_currency,
    "employee_residence": employee_residence,
    "remote_ratio": remote_ratio,
    "company_location": company_location,
    "company_size": company_size
}])

# Ensure all required feature cols exist
missing_cols = [c for c in feature_columns if c not in input_data.columns]
for col in missing_cols:
    input_data[col] = 0   # Default filler

# Reorder
input_data = input_data[feature_columns]


# ------------------------------
# Predict Button
# ------------------------------
if st.button("Predict Salary"):
    try:
        # Scale numeric features
        X_scaled = scaler.transform(input_data)

        # Prediction
        salary_pred = model.predict(X_scaled)[0]

        st.success(f"💰 **Predicted Salary: {salary_pred:,.2f} USD**")
    except Exception as e:
        st.error(f"Prediction error: {e}")