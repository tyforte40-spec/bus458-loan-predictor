import streamlit as st
import pickle
import pandas as pd
import numpy as np

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Loan Approval Predictor",
    page_icon="🏦",
    layout="centered"
)

# ── Styling ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}
h1, h2, h3 {
    font-family: 'DM Serif Display', serif;
}

.stApp {
    background: #f7f5f0;
}

.result-approved {
    background: #1a3c2e;
    color: #a8e6cf;
    border-radius: 12px;
    padding: 2rem;
    text-align: center;
    font-family: 'DM Serif Display', serif;
    font-size: 2rem;
    margin-top: 1.5rem;
}

.result-denied {
    background: #3c1a1a;
    color: #e6a8a8;
    border-radius: 12px;
    padding: 2rem;
    text-align: center;
    font-family: 'DM Serif Display', serif;
    font-size: 2rem;
    margin-top: 1.5rem;
}

.section-label {
    font-size: 0.75rem;
    font-weight: 500;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #888;
    margin-bottom: 0.25rem;
}

div[data-testid="stForm"] {
    background: #ffffff;
    border-radius: 16px;
    padding: 2rem;
    box-shadow: 0 2px 16px rgba(0,0,0,0.06);
}
</style>
""", unsafe_allow_html=True)

# ── Load model ─────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open("loan_approval_model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("Loan Approval Predictor")
st.markdown("Enter applicant details below to predict loan approval likelihood.")
st.divider()

# ── Input form ─────────────────────────────────────────────────────────────────
with st.form("prediction_form"):

    st.markdown("#### Applicant Profile")
    col1, col2 = st.columns(2)

    with col1:
        fico = st.slider("FICO Score", min_value=300, max_value=850, value=650, step=1)
        income = st.number_input("Monthly Gross Income ($)", min_value=0, max_value=50000,
                                  value=5000, step=100)
        housing = st.number_input("Monthly Housing Payment ($)", min_value=0, max_value=10000,
                                   value=1500, step=50)

    with col2:
        loan_amount = st.number_input("Loan Amount Requested ($)", min_value=1000,
                                       max_value=200000, value=30000, step=1000)
        bankrupt = st.selectbox("Ever Bankrupt or Foreclosed?", options=[0, 1],
                                 format_func=lambda x: "No" if x == 0 else "Yes")
        lender = st.selectbox("Lender", options=["A", "B", "C"])

    st.markdown("#### Loan & Employment Details")
    col3, col4 = st.columns(2)

    with col3:
        reason = st.selectbox("Loan Reason", options=[
            "cover_an_unexpected_cost",
            "credit_card_refinancing",
            "debt_conslidation",
            "home_improvement",
            "major_purchase",
            "other"
        ])
        emp_status = st.selectbox("Employment Status", options=[
            "full_time", "part_time", "unemployed"
        ])

    with col4:
        emp_sector = st.selectbox("Employment Sector", options=[
            "communication_services",
            "consumer_discretionary",
            "consumer_staples",
            "energy",
            "financials",
            "health_care",
            "industrials",
            "information_technology",
            "materials",
            "real_estate",
            "utilities"
        ])

    submitted = st.form_submit_button("Predict Approval", use_container_width=True)

# ── Prediction logic ────────────────────────────────────────────────────────────
if submitted:
    # Build a single-row dataframe matching EXACT training column order
    input_data = {
        'Granted_Loan_Amount':                          loan_amount,
        'FICO_score':                                   fico,
        'Monthly_Gross_Income':                         income,
        'Monthly_Housing_Payment':                      housing,
        'Ever_Bankrupt_or_Foreclose':                   bankrupt,
        'Reason_credit_card_refinancing':               int(reason == 'credit_card_refinancing'),
        'Reason_debt_conslidation':                     int(reason == 'debt_conslidation'),
        'Reason_home_improvement':                      int(reason == 'home_improvement'),
        'Reason_major_purchase':                        int(reason == 'major_purchase'),
        'Reason_other':                                 int(reason == 'other'),
        'Employment_Status_part_time':                  int(emp_status == 'part_time'),
        'Employment_Status_unemployed':                 int(emp_status == 'unemployed'),
        'Employment_Sector_consumer_discretionary':     int(emp_sector == 'consumer_discretionary'),
        'Employment_Sector_consumer_staples':           int(emp_sector == 'consumer_staples'),
        'Employment_Sector_energy':                     int(emp_sector == 'energy'),
        'Employment_Sector_financials':                 int(emp_sector == 'financials'),
        'Employment_Sector_health_care':                int(emp_sector == 'health_care'),
        'Employment_Sector_industrials':                int(emp_sector == 'industrials'),
        'Employment_Sector_information_technology':     int(emp_sector == 'information_technology'),
        'Employment_Sector_materials':                  int(emp_sector == 'materials'),
        'Employment_Sector_real_estate':                int(emp_sector == 'real_estate'),
        'Employment_Sector_utilities':                  int(emp_sector == 'utilities'),
        'Lender_B':                                     int(lender == 'B'),
        'Lender_C':                                     int(lender == 'C'),
    }

    input_df = pd.DataFrame([input_data])

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.divider()

    if prediction == 1:
        st.markdown(f"""
        <div class="result-approved">
            ✓ Approved<br>
            <span style="font-size:1rem; font-family:'DM Sans',sans-serif; opacity:0.8;">
                Approval probability: {probability*100:.1f}%
            </span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-denied">
            ✗ Not Approved<br>
            <span style="font-size:1rem; font-family:'DM Sans',sans-serif; opacity:0.8;">
                Approval probability: {probability*100:.1f}%
            </span>
        </div>
        """, unsafe_allow_html=True)

    # Key drivers
    st.markdown("#### Key factors in this decision")
    factors = {
        "FICO Score": fico,
        "Monthly Income": f"${income:,}",
        "Employment Status": emp_status.replace("_", " ").title(),
        "Lender": f"Lender {lender}",
        "Bankruptcy History": "Yes" if bankrupt else "No"
    }
    cols = st.columns(len(factors))
    for col, (label, value) in zip(cols, factors.items()):
        col.metric(label, value)
