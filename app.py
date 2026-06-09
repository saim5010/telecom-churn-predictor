
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import plotly.graph_objects as go
import os

# ── Absolute paths ─────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(
    page_title="Telecom Churn Predictor",
    page_icon="📡",
    layout="wide"
)

@st.cache_resource
def load_artifacts():
    model         = joblib.load(os.path.join(BASE_DIR, "model_xgb.pkl"))
    feature_names = pd.read_csv(os.path.join(BASE_DIR, "feature_names.csv"))["feature"].tolist()
    return model, feature_names

model, feature_names = load_artifacts()

# ── Header ─────────────────────────────────────────────────────────────
st.title("📡 IBM Telecom Customer Churn Predictor")
st.markdown("Enter customer profile → get real-time churn probability, SHAP explanation, and retention ROI.")
st.divider()

# ── Input form ─────────────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Account")
    tenure          = st.slider("Tenure (months)", 0, 72, 12)
    contract        = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    payment         = st.selectbox("Payment Method", [
                        "Electronic check", "Mailed check",
                        "Bank transfer (automatic)", "Credit card (automatic)"])
    paperless       = st.selectbox("Paperless Billing", ["Yes", "No"])
    monthly_charges = st.slider("Monthly Charges ($)", 20.0, 120.0, 65.0, step=0.5)
    total_charges   = st.number_input("Total Charges ($)",
                        value=float(round(monthly_charges * max(tenure, 1), 2)),
                        min_value=0.0)

with col2:
    st.subheader("Services")
    internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    if internet != "No":
        online_sec    = st.selectbox("Online Security",   ["Yes", "No"])
        online_bkp    = st.selectbox("Online Backup",     ["Yes", "No"])
        device_prot   = st.selectbox("Device Protection", ["Yes", "No"])
        tech_sup      = st.selectbox("Tech Support",      ["Yes", "No"])
        streaming_tv  = st.selectbox("Streaming TV",      ["Yes", "No"])
        streaming_mov = st.selectbox("Streaming Movies",  ["Yes", "No"])
    else:
        online_sec = online_bkp = device_prot = tech_sup = "No internet service"
        streaming_tv = streaming_mov = "No internet service"
    phone       = st.selectbox("Phone Service", ["Yes", "No"])
    multi_lines = st.selectbox("Multiple Lines", ["Yes", "No"]) if phone == "Yes" else "No phone service"

with col3:
    st.subheader("Demographics")
    gender     = st.selectbox("Gender",         ["Male", "Female"])
    senior     = st.selectbox("Senior Citizen", ["Yes", "No"])
    partner    = st.selectbox("Partner",        ["Yes", "No"])
    dependents = st.selectbox("Dependents",     ["Yes", "No"])

# ── Predict ────────────────────────────────────────────────────────────
st.divider()
predict_btn = st.button("🔮 Predict Churn Probability", type="primary", use_container_width=True)

if predict_btn:
    raw = {
        "TenureMonths":    tenure,
        "MonthlyCharges":  monthly_charges,
        "TotalCharges":    total_charges,
        "Gender":          gender,
        "SeniorCitizen":   senior,
        "Partner":         partner,
        "Dependents":      dependents,
        "PhoneService":    phone,
        "MultipleLines":   multi_lines,
        "InternetService": internet,
        "OnlineSecurity":  online_sec,
        "OnlineBackup":    online_bkp,
        "DeviceProtection":device_prot,
        "TechSupport":     tech_sup,
        "StreamingTV":     streaming_tv,
        "StreamingMovies": streaming_mov,
        "Contract":        contract,
        "PaperlessBilling":paperless,
        "PaymentMethod":   payment
    }
    df = pd.DataFrame([raw])

    # Feature engineering
    df["ChargesPerMonth"] = df["TotalCharges"] / df["TenureMonths"].replace(0, 1)
    service_cols = ["PhoneService","MultipleLines","InternetService",
                    "OnlineSecurity","OnlineBackup","DeviceProtection",
                    "TechSupport","StreamingTV","StreamingMovies"]
    df["ServicesCount"] = df[service_cols].apply(
        lambda row: sum(v not in ["No","No internet service","No phone service"]
                        for v in row), axis=1)
    df["SeniorNoSupport"] = (
        (df["SeniorCitizen"] == "Yes") & (df["TechSupport"] == "No")
    ).astype(int)

    # Label encode
    binary_cols = ["Partner","Dependents","MultipleLines","OnlineSecurity",
                   "OnlineBackup","DeviceProtection","TechSupport",
                   "StreamingTV","StreamingMovies","PaperlessBilling",
                   "PhoneService","Gender"]
    binary_map  = {"Yes":1,"No":0,"Male":1,"Female":0,
                   "No internet service":0,"No phone service":0}
    for col in binary_cols:
        df[col] = df[col].map(binary_map).fillna(0).astype(int)

    # One-hot encode
    df = pd.get_dummies(df, columns=["Contract","InternetService","PaymentMethod"])
    df.drop(columns=["SeniorCitizen"], errors="ignore", inplace=True)

    # Align to training features
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_names]

    # Predict
    proba      = float(model.predict_proba(df)[0][1])
    color      = "#EF4444" if proba >= 0.6 else "#F59E0B" if proba >= 0.3 else "#10B981"
    risk_label = "🔴 High Risk" if proba >= 0.6 else "🟡 Medium Risk" if proba >= 0.3 else "🟢 Low Risk"

    # Results
    r1, r2, r3 = st.columns(3)
    with r1:
        st.metric("Churn Probability",      f"{proba:.1%}")
    with r2:
        st.metric("Risk Band",              risk_label)
    with r3:
        st.metric("Monthly Revenue at Risk",f"${monthly_charges:,.2f}")

    # Gauge
    fig_gauge = go.Figure(go.Indicator(
        mode  = "gauge+number",
        value = proba * 100,
        number= {"suffix":"%","font":{"size":32}},
        gauge = {
            "axis":  {"range":[0,100]},
            "bar":   {"color":color},
            "steps": [
                {"range":[0, 30],"color":"#D1FAE5"},
                {"range":[30,60],"color":"#FEF3C7"},
                {"range":[60,100],"color":"#FEE2E2"}
            ],
            "threshold":{
                "line":{"color":"#1E3A5F","width":3},
                "thickness":0.75,"value":42
            }
        },
        title={"text":"Churn Probability Score"}
    ))
    fig_gauge.update_layout(height=300, margin=dict(t=50,b=0,l=30,r=30))
    st.plotly_chart(fig_gauge, use_container_width=True)

    # SHAP
    st.subheader("🔍 Why is this customer predicted to churn?")
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(df)
    shap_df = pd.DataFrame({
        "feature":    feature_names,
        "shap_value": shap_values[0]
    }).sort_values("shap_value", key=abs, ascending=False).head(10)

    fig_shap = go.Figure(go.Bar(
        x=shap_df["shap_value"],
        y=shap_df["feature"],
        orientation="h",
        marker_color=["#EF4444" if v > 0 else "#3B82F6"
                      for v in shap_df["shap_value"]]
    ))
    fig_shap.update_layout(
        title ="Top 10 Feature Contributions — Red increases churn risk · Blue decreases it",
        xaxis_title="SHAP Value",
        height=420,
        margin=dict(l=180,r=20,t=50,b=40)
    )
    st.plotly_chart(fig_shap, use_container_width=True)

    # ROI Calculator
    st.subheader("💰 Retention ROI Calculator")
    retention_cost = st.slider("Retention outreach cost per customer ($)", 10, 500, 50)
    annual_value   = monthly_charges * 12
    net_roi        = annual_value - retention_cost

    roi1, roi2, roi3 = st.columns(3)
    with roi1:
        st.metric("Annual Revenue at Risk",  f"${annual_value:,.2f}")
    with roi2:
        st.metric("Retention Outreach Cost", f"${retention_cost:,.2f}")
    with roi3:
        st.metric("Net ROI if Retained",
                  f"${net_roi:,.2f}",
                  delta="Worth retaining ✅" if net_roi > 0 else "Evaluate carefully ⚠️")

    # Recommendation
    st.divider()
    if proba >= 0.6:
        st.error(f"**High churn risk ({proba:.1%})** — Immediate retention action recommended. "
                 f"Prioritise contract upgrade offer and tech support outreach.")
    elif proba >= 0.3:
        st.warning(f"**Medium churn risk ({proba:.1%})** — Monitor this customer. "
                   f"Consider a loyalty discount or service upgrade.")
    else:
        st.success(f"**Low churn risk ({proba:.1%})** — Customer is likely to stay. "
                   f"Standard engagement is sufficient.")
