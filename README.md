# Dynamic Customer Churn & Retention Analytics

An interactive Streamlit dashboard for exploring customer churn. Upload any customer CSV, or use the built-in synthetic dataset, to get instant KPIs, behavioral charts, risk segmentation and a live churn-risk predictor powered by a Random Forest model.

**Live demo:** https://customer-churn-analytics-n3cnuumezarigqiseeyphh.streamlit.app/

## Features

- **CSV upload:** analyze your own dataset, with automatic mapping of common column names (tenure, charges, churn status)
- **KPI cards:** total records, churned customers, churn rate and average monthly value per user
- **Sidebar filters:** slice the data by contract type and tech support
- **Behavioral EDA:** churn distribution by contract type and tenure distribution by status
- **Risk segmentation:** charges vs tenure density plot and the impact of tech support on retention
- **Live risk predictor:** set tenure, monthly charges, contract type and support, and get a churn probability from a Random Forest classifier
- **Model evaluation:** accuracy and ROC-AUC reported on a held-out 20% test set, plus a feature importance chart

## Tech Stack

Python, Streamlit, pandas, NumPy, scikit-learn, Plotly

## How It Works

1. Load the uploaded CSV, or generate a 1,000-customer synthetic dataset (fixed random seed).
2. Standardize columns (`Tenure_Months`, `Monthly_Charges`, `Contract`, `Tech_Support`, `Churn`).
3. Apply sidebar filters and compute KPIs and charts.
4. Split the data 80/20, train a `RandomForestClassifier` (100 trees) on tenure, monthly charges, contract type and tech support, and evaluate it on the held-out set.
5. Predict churn probability for the values chosen with the sliders.

## About the Demo Data

The default dataset is **synthetic**. Churn is generated from simple business rules (month-to-month contracts, no tech support, short tenure and high charges raise churn probability), so the dashboard works without any upload. Because of this, the model metrics on the demo data show that the model can learn those rules and say nothing about real customers. Upload a real CSV to analyze actual customer behavior.

## CSV Format

For the predictor to work, the CSV should include:

| Column | Description |
|--------|-------------|
| `Tenure_Months` | Months the customer has been active |
| `Monthly_Charges` | Monthly amount billed |
| `Churn` | 1 if churned, 0 if retained |
| `Contract` *(optional)* | e.g. Month-to-month, One year, Two year |
| `Tech_Support` *(optional)* | Yes / No |

## Run Locally

```bash
git clone https://github.com/samvithak7-coder/customer-churn-analytics.git
cd customer-churn-analytics
pip install -r requirements.txt
streamlit run app.py
```

## Deployment

Deployed on Streamlit Community Cloud. Every push to `main` triggers an automatic redeploy.

## Future Improvements

- Additional model comparison (logistic regression, gradient boosting)
- Support for more input features and categorical encoding
- Downloadable churn-risk report for uploaded datasets

## Author

**Samvitha K**
[GitHub](https://github.com/samvithak7-coder) · [LinkedIn](https://www.linkedin.com/in/kesapragada-samvitha-977714376/)
