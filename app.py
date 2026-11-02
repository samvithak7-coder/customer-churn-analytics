import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier

# Page configuration
st.set_page_config(
    page_title="Dynamic Churn & Retention Analytics",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Dynamic Customer Churn & Retention Analytics")
st.markdown("Upload any customer dataset (CSV) to automatically run exploratory analytics, visual dashboards, and churn risk predictions.")

# Sidebar File Uploader
st.sidebar.header("📁 Data Input")
uploaded_file = st.sidebar.file_uploader("Upload your CSV dataset", type=["csv"])

# Synthetic Data Generator (Fallback if no file uploaded)
@st.cache_data
def load_default_data():
    np.random.seed(42)
    n = 1000
    tenure = np.random.randint(1, 72, size=n)
    monthly_charges = np.round(np.random.uniform(20.0, 120.0, size=n), 2)
    contract = np.random.choice(["Month-to-month", "One year", "Two year"], size=n, p=[0.55, 0.25, 0.20])
    tech_support = np.random.choice(["Yes", "No"], size=n, p=[0.35, 0.65])
    
    churn_prob = (
        (contract == "Month-to-month") * 0.35 +
        (tech_support == "No") * 0.25 +
        (tenure < 12) * 0.20 +
        (monthly_charges > 70) * 0.15
    )
    churn_prob = np.clip(churn_prob, 0.05, 0.90)
    churn = np.random.binomial(1, churn_prob)
    churn_label = np.where(churn == 1, "Churned", "Retained")
    
    return pd.DataFrame({
        "CustomerID": [f"CUST-{1000+i}" for i in range(n)],
        "Tenure_Months": tenure,
        "Monthly_Charges": monthly_charges,
        "Contract": contract,
        "Tech_Support": tech_support,
        "Churn": churn,
        "Status": churn_label
    })

# Load user dataset or default
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success("Custom dataset loaded successfully!")
    
    # Auto-detect or map standard column names if user uploads custom data
    # (Checking if standard names exist, otherwise fallback mapping)
    if "Status" not in df.columns and "Churn" in df.columns:
        df["Status"] = np.where(df["Churn"] == 1, "Churned", "Retained")
    elif "Status" in df.columns and "Churn" not in df.columns:
        df["Churn"] = np.where(df["Status"].astype(str).str.lower().isin(["1", "true", "yes", "churned"]), 1, 0)
        
    if "Tenure_Months" not in df.columns:
        # Look for tenure-like column
        tenure_cols = [c for c in df.columns if 'tenure' in c.lower() or 'month' in c.lower()]
        if tenure_cols:
            df["Tenure_Months"] = df[tenure_cols[0]]
        else:
            df["Tenure_Months"] = np.random.randint(1, 24, size=len(df))
            
    if "Monthly_Charges" not in df.columns:
        charge_cols = [c for c in df.columns if 'charge' in c.lower() or 'fee' in c.lower() or 'amount' in c.lower()]
        if charge_cols:
            df["Monthly_Charges"] = df[charge_cols[0]]
        else:
            df["Monthly_Charges"] = 50.0
            
    if "Contract" not in df.columns:
        df["Contract"] = "Month-to-month"
        
    if "Tech_Support" not in df.columns:
        df["Tech_Support"] = "No"
else:
    df = load_default_data()
    st.sidebar.info("Tip: Upload your own CSV file above to analyze custom data. Using demo Telco dataset by default.")

# Sidebar Filters dynamically populated based on columns
st.sidebar.markdown("---")
st.sidebar.header("Filter Analytics")

contract_options = df["Contract"].unique() if "Contract" in df.columns else []
contract_filter = st.sidebar.multiselect("Contract Type", options=contract_options, default=contract_options)

tech_options = df["Tech_Support"].unique() if "Tech_Support" in df.columns else []
tech_filter = st.sidebar.multiselect("Tech Support Offered", options=tech_options, default=tech_options)

# Apply filters safely
filtered_df = df.copy()
if "Contract" in df.columns and contract_filter:
    filtered_df = filtered_df[filtered_df["Contract"].isin(contract_filter)]
if "Tech_Support" in df.columns and tech_filter:
    filtered_df = filtered_df[filtered_df["Tech_Support"].isin(tech_filter)]

# Top-level KPI Metrics
col1, col2, col3, col4 = st.columns(4)
total_cust = len(filtered_df)
churn_count = filtered_df["Churn"].sum() if "Churn" in filtered_df.columns else 0
churn_rate = (churn_count / total_cust * 100) if total_cust > 0 else 0
avg_rev = filtered_df["Monthly_Charges"].mean() if "Monthly_Charges" in filtered_df.columns else 0

col1.metric("Total Records Evaluated", f"{total_cust:,}")
col2.metric("Total Churned / Drop-offs", f"{churn_count:,}")
col3.metric("Drop-off / Churn Rate", f"{churn_rate:.1f}%")
col4.metric("Avg Monthly Value / User", f"${avg_rev:.2f}")

st.markdown("---")

# Visual Analytics Section
tab1, tab2, tab3 = st.tabs(["📈 Behavioral EDA", "🎯 Risk Segmentation", "🔮 Live Risk Predictor"])

with tab1:
    col_a, col_b = st.columns(2)
    
    with col_a:
        if "Contract" in filtered_df.columns and "Status" in filtered_df.columns:
            fig_contract = px.histogram(
                filtered_df, x="Contract", color="Status", barmode="group",
                title="Distribution by Contract Type",
                color_discrete_map={"Churned": "#EF553B", "Retained": "#636EFA"}
            )
            st.plotly_chart(fig_contract, use_container_width=True)
        else:
            st.write("Column 'Contract' or 'Status' not found in dataset.")
        
    with col_b:
        if "Tenure_Months" in filtered_df.columns and "Status" in filtered_df.columns:
            fig_tenure = px.box(
                filtered_df, x="Status", y="Tenure_Months", color="Status",
                title="Tenure Distribution vs Status",
                color_discrete_map={"Churned": "#EF553B", "Retained": "#636EFA"}
            )
            st.plotly_chart(fig_tenure, use_container_width=True)
        else:
            st.write("Column 'Tenure_Months' or 'Status' not found.")

with tab2:
    col_c, col_d = st.columns(2)
    
    with col_c:
        if "Tenure_Months" in filtered_df.columns and "Monthly_Charges" in filtered_df.columns:
            fig_charges = px.density_contour(
                filtered_df, x="Tenure_Months", y="Monthly_Charges", color="Status" if "Status" in filtered_df.columns else None,
                title="Charges vs Tenure Density"
            )
            st.plotly_chart(fig_charges, use_container_width=True)
        else:
            st.write("Required numeric columns missing for density plot.")
            
    with col_d:
        if "Tech_Support" in filtered_df.columns and "Status" in filtered_df.columns:
            tech_summary = filtered_df.groupby(["Tech_Support", "Status"]).size().reset_index(name="Count")
            fig_tech = px.bar(
                tech_summary, x="Tech_Support", y="Count", color="Status",
                title="Impact of Support/Features on Retention",
                color_discrete_map={"Churned": "#EF553B", "Retained": "#636EFA"}
            )
            st.plotly_chart(fig_tech, use_container_width=True)

with tab3:
    st.subheader("Predict Risk Probability on Custom Data")
    try:
        X = df[["Tenure_Months", "Monthly_Charges"]].copy()
        X["Contract_Month"] = (df["Contract"] == "Month-to-month").astype(int) if "Contract" in df.columns else 0
        X["No_Tech_Support"] = (df["Tech_Support"] == "No").astype(int) if "Tech_Support" in df.columns else 0
        y = df["Churn"]
        
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        input_col1, input_col2 = st.columns(2)
        with input_col1:
            in_tenure = st.slider("Tenure (Months)", 1, int(df["Tenure_Months"].max()) if not df["Tenure_Months"].empty else 12, 6)
            in_monthly = st.slider("Monthly Charges ($)", float(df["Monthly_Charges"].min()) if not df["Monthly_Charges"].empty else 20.0, float(df["Monthly_Charges"].max()) if not df["Monthly_Charges"].empty else 120.0, 85.0)
        with input_col2:
            in_contract = st.selectbox("Contract Type", options=df["Contract"].unique() if "Contract" in df.columns else ["Standard"])
            in_tech = st.selectbox("Support Included?", options=df["Tech_Support"].unique() if "Tech_Support" in df.columns else ["Yes", "No"])
        
        is_m2m = 1 if in_contract == "Month-to-month" else 0
        is_no_tech = 1 if in_tech == "No" else 0
        
        pred_risk = model.predict_proba([[in_tenure, in_monthly, is_m2m, is_no_tech]])[0][1]
        
        st.markdown(f"### Predicted Risk Probability: **{pred_risk*100:.1f}%**")
        if pred_risk > 0.5:
            st.error("⚠️ High Risk Detected!")
        else:
            st.success("✅ Low Risk Detected.")
    except Exception as e:
        st.info("Upload a dataset with standard columns (Tenure_Months, Monthly_Charges, Churn) to enable live machine learning predictions.")