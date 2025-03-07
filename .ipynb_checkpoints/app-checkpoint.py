import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load("loan_approval_model.pkl")

# Title of the web app
st.title("🏦 Loan Approval Prediction System")

# User input fields
income_annum = st.number_input("Annual Income (in INR)", min_value=0, step=10000)
loan_amount = st.number_input("Loan Amount Requested (in INR)", min_value=0, step=10000)
loan_term = st.number_input("Loan Term (in months)", min_value=1, step=6)
cibil_score = st.number_input("CIBIL Score (300-900)", min_value=300, max_value=900, step=10)
residential_assets_value = st.number_input("Residential Assets Value (in INR)", min_value=0, step=10000)
commercial_assets_value = st.number_input("Commercial Assets Value (in INR)", min_value=0, step=10000)
luxury_assets_value = st.number_input("Luxury Assets Value (in INR)", min_value=0, step=10000)
bank_assets_value = st.number_input("Bank Asset Value (in INR)", min_value=0, step=10000)
no_of_dependents = st.number_input("Number of Dependents", min_value=0, step=1)
education = st.selectbox("education_ Not Graduate", ["Graduate", "Not Graduate"])
self_employed_Yes = st.selectbox("self_employed_ Yes", ["No", "Yes"])

# Convert categorical inputs to numeric
education = 1 if education == "Not Graduate" else 0
self_employed_Yes = 1 if self_employed_Yes == "Yes" else 0

# Prediction button
if st.button("Check Loan Approval"):
    # Prepare input for model
    input_data = np.array([[no_of_dependents,income_annum, loan_amount, loan_term, cibil_score,
                            residential_assets_value, commercial_assets_value, luxury_assets_value,
                            bank_assets_value, education, self_employed_Yes]])
    
    # Make prediction
    prediction = model.predict(input_data)[0]

    # Display result
    if prediction == 1:
        st.success("✅ Congratulations! Your Loan is Approved.")
    else:
        st.error("❌ Sorry, Your Loan is Rejected.")
