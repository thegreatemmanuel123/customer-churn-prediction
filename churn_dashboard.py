import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Customer Churn Predictor", layout="wide")
st.title("Telco Customer Churn Prediction Tool")
st.markdown("Enter customer details to predict churn risk. Built to help businesses retain customers.")

# ────────────────────────────────────────────────
# Load and prepare data (we'll train a quick model here for demo)
@st.cache_data
def load_and_prepare():
    df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
    df = df.drop('customerID', axis=1)
    df['Churn'] = df['Churn'].map({'No': 0, 'Yes': 1})
    
    # Simple encoding for demo (match what we did earlier)
    yes_no_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling',
                   'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                   'TechSupport', 'StreamingTV', 'StreamingMovies']
    for col in yes_no_cols:
        df[col] = df[col].map({'Yes': 1, 'No': 0, 'No internet service': 0, 'No phone service': 0})
    
    df = pd.get_dummies(df, columns=['gender', 'MultipleLines', 'InternetService',
                                     'Contract', 'PaymentMethod'], drop_first=True)
    return df

df = load_and_prepare()

# Train a quick model on full data (for demo; in production use train/test split + save model)
X = df.drop('Churn', axis=1)
y = df['Churn']
model = LogisticRegression(max_iter=2000)
model.fit(X, y)

# ────────────────────────────────────────────────
# User inputs (key drivers from EDA)
st.sidebar.header("Customer Details")

tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)
monthly_charges = st.sidebar.slider("Monthly Charges ($)", 10.0, 120.0, 70.0)
total_charges = tenure * monthly_charges  # approximate

contract = st.sidebar.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
internet_service = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
payment_method = st.sidebar.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
paperless_billing = st.sidebar.radio("Paperless Billing", ["Yes", "No"])
senior_citizen = st.sidebar.radio("Senior Citizen", ["Yes", "No"])

# Prepare input as DataFrame (must match training columns)
input_data = {
    'tenure': tenure,
    'MonthlyCharges': monthly_charges,
    'TotalCharges': total_charges,
    'SeniorCitizen': 1 if senior_citizen == "Yes" else 0,
    'Partner': 0,  # simplified; add more if you want full form
    'Dependents': 0,
    'PhoneService': 1,
    'PaperlessBilling': 1 if paperless_billing == "Yes" else 0,
    # Dummies - set to 0, then override selected ones
}

# Contract dummies
input_data['Contract_One year'] = 1 if contract == "One year" else 0
input_data['Contract_Two year'] = 1 if contract == "Two year" else 0

# Internet dummies
input_data['InternetService_Fiber optic'] = 1 if internet_service == "Fiber optic" else 0
input_data['InternetService_No'] = 1 if internet_service == "No" else 0

# Payment dummies
input_data['PaymentMethod_Credit card (automatic)'] = 1 if payment_method == "Credit card (automatic)" else 0
input_data['PaymentMethod_Electronic check'] = 1 if payment_method == "Electronic check" else 0
input_data['PaymentMethod_Mailed check'] = 1 if payment_method == "Mailed check" else 0

# Fill missing columns with 0 (for all other dummies)
for col in X.columns:
    if col not in input_data:
        input_data[col] = 0

input_df = pd.DataFrame([input_data])

# Predict
if st.sidebar.button("Predict Churn Risk"):
    prob = model.predict_proba(input_df)[0][1]
    st.subheader(f"Churn Probability: **{prob:.1%}**")
    
    if prob > 0.5:
        st.error("🔴 HIGH RISK - Recommend immediate retention action (discount, call, upgrade offer)")
    elif prob > 0.3:
        st.warning("🟡 MEDIUM RISK - Monitor closely, consider proactive outreach")
    else:
        st.success("🟢 LOW RISK - Customer likely to stay")

# Some visuals
st.header("Key Business Insights from Data")
st.markdown("- Month-to-month contracts have the highest churn")
st.markdown("- New customers (low tenure) are most at risk")
st.markdown("- Higher monthly charges correlate with more churn")

st.markdown("Expand this dashboard with more inputs or better model for production use!")
