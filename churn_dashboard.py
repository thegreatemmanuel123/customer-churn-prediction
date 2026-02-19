import streamlit as st

st.set_page_config(page_title="Customer Churn Predictor", layout="wide")
st.title("Telco Customer Churn Prediction Tool")
st.markdown("Enter customer details to see churn risk. (Demo version - rule-based on key insights)")

# Sidebar inputs
st.sidebar.header("Customer Details")

tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)
monthly_charges = st.sidebar.slider("Monthly Charges ($)", 10.0, 120.0, 70.0)
contract = st.sidebar.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
internet_service = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

# Simple rule-based churn probability (from our EDA insights)
prob = 0.0

# High risk factors
if contract == "Month-to-month":
    prob += 0.50
if tenure < 12:
    prob += 0.30
if monthly_charges > 80:
    prob += 0.20
if internet_service == "Fiber optic":
    prob += 0.15

# Cap at 95%
prob = min(prob, 0.95)

# Predict button
if st.sidebar.button("Predict Churn Risk"):
    st.subheader(f"Churn Probability: **{prob:.1%}**")
    
    if prob > 0.50:
        st.error("🔴 HIGH RISK - Recommend retention offer (discount, support call)")
    elif prob > 0.30:
        st.warning("🟡 MEDIUM RISK - Monitor and consider outreach")
    else:
        st.success("🟢 LOW RISK - Likely to stay")

# Insights section
st.header("Key Insights from Analysis")
st.markdown("- Month-to-month contracts cause the most churn")
st.markdown("- Customers with low tenure (<12 months) are at high risk")
st.markdown("- Higher monthly charges increase churn likelihood")
st.markdown("- Full ML model and code in repo for advanced version")

st.markdown("Repo: https://github.com/thegreatemmanuel123/customer-churn-prediction")
