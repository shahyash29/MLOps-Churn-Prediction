# training/train_mlflow.py
from pathlib import Path
import os, json
import numpy as np
import pandas as pd

from dotenv import load_dotenv
load_dotenv()

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

from sklearn.compose import ColumnTransformer, make_column_selector as selector
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, roc_auc_score, classification_report, confusion_matrix,
    roc_curve, auc, precision_recall_curve, average_precision_score
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False

ARTIFACTS_DIR = Path("artifacts"); ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# --- MLflow config from env ---
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "churn_exp"))

def _coerce_known_numeric(df: pd.DataFrame) -> pd.DataFrame:
    numeric_should_be = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
    for col in numeric_should_be:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            if df[col].isna().any():
                df[col] = df[col].fillna(df[col].median())
    return df

def _load_xy(train_path="data/processed/train.csv", test_path="data/processed/test.csv"):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    X_train, y_train = train.drop(columns=["churn"]), train["churn"].astype(int)
    X_test, y_test   = test.drop(columns=["churn"]),  test["churn"].astype(int)

    X_train = _coerce_known_numeric(X_train)
    X_test  = _coerce_known_numeric(X_test)

    for df in (X_train, X_test):
        for c in df.columns:
            if df[c].dtype == "object":
                df[c] = df[c].fillna("NA")
            else:
                df[c] = df[c].fillna(0)
    return X_train, X_test, y_train, y_test

def _preprocessor():
    return ColumnTransformer([
        ("num", StandardScaler(), selector(dtype_exclude="object")),
        ("cat", OneHotEncoder(handle_unknown="ignore"), selector(dtype_include="object")),
    ])

def _plot_and_log_curves(y_true, y_prob, run_dir: Path):
    run_dir.mkdir(parents=True, exist_ok=True)
    # ROC
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    fig1, ax1 = plt.subplots()
    ax1.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
    ax1.plot([0,1],[0,1],"--",linewidth=1)
    ax1.set_xlabel("FPR"); ax1.set_ylabel("TPR"); ax1.set_title("ROC")
    ax1.legend(loc="lower right")
    roc_path = run_dir / "roc_curve.png"
    fig1.savefig(roc_path, bbox_inches="tight", dpi=120); plt.close(fig1)

    # PR
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    fig2, ax2 = plt.subplots()
    ax2.plot(recall, precision, label=f"AP={ap:.3f}")
    ax2.set_xlabel("Recall"); ax2.set_ylabel("Precision"); ax2.set_title("PR")
    ax2.legend(loc="lower left")
    pr_path = run_dir / "pr_curve.png"
    fig2.savefig(pr_path, bbox_inches="tight", dpi=120); plt.close(fig2)

    if roc_path.exists(): mlflow.log_artifact(str(roc_path))
    if pr_path.exists():  mlflow.log_artifact(str(pr_path))

def _evaluate_and_log(name, pipe, X_test, y_test):
    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    aucv = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred).tolist()
    report = classification_report(y_test, y_pred, digits=3)

    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("roc_auc", aucv)
    mlflow.log_text(json.dumps({"confusion_matrix": cm}), "confusion_matrix.json")
    mlflow.log_text(report, "classification_report.txt")
    _plot_and_log_curves(y_test, y_prob, Path("artifacts") / f"{name}_plots")

    print(f"\n{name} trained")
    print(f"Accuracy: {acc:.3f} | ROC-AUC: {aucv:.3f}")
    print(report)

def run_logreg(max_iter=1000):
    X_train, X_test, y_train, y_test = _load_xy()
    pipe = Pipeline([
        ("pre", _preprocessor()),
        ("model", LogisticRegression(max_iter=max_iter, class_weight="balanced")),
    ])

    with mlflow.start_run(run_name="logreg_baseline"):
        mlflow.log_param("model_type", "logreg")
        mlflow.log_param("max_iter", max_iter)
        pipe.fit(X_train, y_train)

        X_sample = X_test.iloc[:5]
        signature = infer_signature(X_sample, pipe.predict(X_sample))

        _evaluate_and_log("logreg", pipe, X_test, y_test)

        # critical: artifact_path creates Artifacts/model
        mlflow.sklearn.log_model(
            sk_model=pipe,
            artifact_path="model",
            signature=signature,
            input_example=X_sample
        )

def run_xgb():
    if not HAS_XGB:
        print("XGBoost not installed; skipping."); return
    X_train, X_test, y_train, y_test = _load_xy()
    xgb = XGBClassifier(
        n_estimators=400, max_depth=5, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
        eval_metric="logloss", tree_method="hist", random_state=42
    )
    pipe = Pipeline([("pre", _preprocessor()), ("model", xgb)])

    with mlflow.start_run(run_name="xgb_baseline"):
        mlflow.log_param("model_type", "xgb")
        mlflow.log_params({
            "n_estimators": 400, "max_depth": 5, "learning_rate": 0.05,
            "subsample": 0.9, "colsample_bytree": 0.9, "reg_lambda": 1.0
        })
        pipe.fit(X_train, y_train)

        X_sample = X_test.iloc[:5]
        signature = infer_signature(X_sample, pipe.predict(X_sample))

        _evaluate_and_log("xgb", pipe, X_test, y_test)

        mlflow.sklearn.log_model(
            sk_model=pipe,
            artifact_path="model",
            signature=signature,
            input_example=X_sample
        )

if __name__ == "__main__":
    run_logreg(max_iter=1000)
    run_xgb()  