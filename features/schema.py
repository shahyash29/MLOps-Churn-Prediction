import pandas as pd

EXPECTED = [
    "customerID","gender","SeniorCitizen","Partner","Dependents",
    "tenure","PhoneService","InternetService","Contract",
    "MonthlyCharges","TotalCharges","Churn"
]

def validate_columns(df: pd.DataFrame):
    missing = set(EXPECTED) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    if not df["Churn"].isin(["Yes","No"]).all():
        raise ValueError("Column 'Churn' must be Yes/No only.")