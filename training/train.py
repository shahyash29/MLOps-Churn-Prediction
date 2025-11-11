# training/train.py
import os
from pathlib import Path

import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer, make_column_selector as selector
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import joblib

ARTIFACTS_DIR = Path("artifacts")
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


def _coerce_known_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """
    Coerce columns that SHOULD be numeric (but might be 'object' due to blanks) into numeric.
    Specifically fixes 'TotalCharges' in the Telco dataset, and guards others.
    """
    numeric_should_be = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
    for col in numeric_should_be:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
   
    for col in numeric_should_be:
        if col in df.columns and df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())
    return df

def train_baseline(
    train_path: str = "data/processed/train.csv",
    test_path: str = "data/processed/test.csv",
    max_iter: int = 1000,
):

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    assert "churn" in train.columns, "Expected 'churn' column in training data."
    assert "churn" in test.columns, "Expected 'churn' column in test data."
    X_train, y_train = train.drop(columns=["churn"]), train["churn"].astype(int)
    X_test, y_test = test.drop(columns=["churn"]), test["churn"].astype(int)

    X_train = _coerce_known_numeric(X_train)
    X_test = _coerce_known_numeric(X_test)

    for c in X_train.columns:
        if X_train[c].dtype == "object":
            X_train[c] = X_train[c].fillna("NA")
            if c in X_test.columns:
                X_test[c] = X_test[c].fillna("NA")
        else:
            X_train[c] = X_train[c].fillna(0)
            if c in X_test.columns:
                X_test[c] = X_test[c].fillna(0)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), selector(dtype_exclude="object")),
            ("cat", OneHotEncoder(handle_unknown="ignore"), selector(dtype_include="object")),
        ]
    )

    pipe = Pipeline(
        steps=[
            ("pre", preprocessor),
            ("model", LogisticRegression(max_iter=max_iter, class_weight="balanced")),
        ]
    )

    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    report = classification_report(y_test, y_pred, digits=3)

    print("\nModel trained successfully!")
    print(f"Accuracy : {acc:.3f}")
    print(f"ROC-AUC  : {auc:.3f}")
    print("\nClassification report:\n", report)

    out_path = ARTIFACTS_DIR / "baseline_logreg_pipeline.pkl"
    joblib.dump(pipe, out_path)
    print(f"\nSaved model pipeline to: {out_path.resolve()}")

    num_cols = X_train.select_dtypes(exclude="object").columns.tolist()
    cat_cols = X_train.select_dtypes(include="object").columns.tolist()
    print("\nColumns summary:")
    print("  Numeric:     ", num_cols)
    print("  Categorical: ", cat_cols)

if __name__ == "__main__":
    train_baseline()
