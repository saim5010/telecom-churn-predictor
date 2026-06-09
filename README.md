# 📡 IBM Telecom Customer Churn Prediction
> End-to-end data science project — SQL · Python · XGBoost · Power BI · Streamlit

## 🔗 Live Links
- **Streamlit App:** https://telecom-churn-predictor-ejckmwhbkxzibmq52w9vqm.streamlit.app
- **Power BI Dashboard:** https://app.powerbi.com/links/Xr1ZcmiNZT?ctid=6304c17f-8eec-4b7d-a042-5ff6e024881f&pbi_source=linkShare

---

## 📌 Problem Statement
A telecom company loses **26.54% of customers** monthly — translating to **$139K/month** in revenue at risk.
This project builds a complete churn prediction and retention decision system from raw data to live deployment.

---

## 🏗️ Project Architecture

    IBM Telco Dataset (7,043 customers)
            ↓
      Python — Data cleaning + EDA
            ↓
      SQLite — 9 business analysis queries
            ↓
      Python — Feature engineering + ML pipeline
            ↓
      XGBoost — Churn probability scores
            ↓
      Power BI — 4-page executive dashboard
            ↓
      Streamlit — Live churn predictor app

---

## 📊 Key Results

| Metric | Value |
|--------|-------|
| Overall churn rate | 26.54% |
| Monthly revenue at risk | $139,130 |
| Best model AUC-ROC | 0.843 |
| Production model | XGBoost @ threshold 0.42 |
| Recall on churners | 67% |
| F1 score (churn class) | 0.63 |
| High-risk segment churn rate | 80.6% |

---

## 🔍 Key Business Findings

- **Contract type** is the strongest predictor — Month-to-month churn at **42.7%** vs 2.8% for two-year
- **Tenure** is the best retention signal — customers past 24 months churn at only 9.5%
- **Electronic check** payers churn at 45.3% — double the rate of auto-pay customers
- **Fiber optic** customers churn at 41.9% despite paying the highest charges ($91.50 avg)
- Top churn reason: **Attitude of support person (10.3%)**
- At **30% retention**, estimated **$41.7K/month** in revenue can be saved

---

## 🛠️ Tech Stack

| Layer | Tools |
|-------|-------|
| Data ingestion | Python, Pandas, SQLite |
| SQL analysis | SQLite, CTEs, Window functions |
| EDA | Pandas, Matplotlib, Seaborn |
| Feature engineering | 3 custom features |
| ML models | Logistic Regression, Random Forest, XGBoost |
| Explainability | SHAP TreeExplainer |
| Dashboard | Power BI Desktop, DAX, What-If parameter |
| Deployment | Streamlit Community Cloud |

---

## 📄 Power BI Dashboard — 4 Pages

| Page | Description |
|------|-------------|
| P1 — Executive Overview | Churn rate KPIs, revenue at risk, contract split |
| P2 — Customer Segments | Churn by tenure, internet service, payment method |
| P3 — Risk Scoring | ML probability distribution, high-risk customer profiles |
| P4 — Retention ROI | What-If parameter — revenue saved vs retention rate |

---

## 🤖 Streamlit App Features

- Real-time churn probability prediction
- Gauge chart with risk band (Low / Medium / High)
- SHAP waterfall chart — explains why a customer is predicted to churn
- Retention ROI calculator — net value of retaining the customer

---

## 🎓 Certification
**Professional Certification in Data Science & AI Engineering**
IIT Guwahati x AlmaBetter (2025)

---

## 👤 Author
**Saim Mulani** · [LinkedIn](https://linkedin.com/in/saimmulani-data) · [GitHub](https://github.com/saim5010)
