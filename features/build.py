from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from .schema import validate_columns

RAW = Path("data/raw/telco.csv")
OUT_DIR = Path("data/processed")

def clean_and_split(test_size: float = 0.2, seed: int = 42):
    OUT_DIR.mkdir(parents=True,exist_ok=True)

    df = pd.read_csv(RAW)
    validate_columns(df)

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["churn"] = df["Churn"].map({"Yes":1, "No":0})

    drop_cols = ["customerID","Churn"]
    X = df.drop(columns=drop_cols + ["churn"])
    y = df["churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )

    X_train.assign(churn=y_train.values).to_csv(OUT_DIR / "train.csv", index=False)
    X_test.assign(churn=y_test.values).to_csv(OUT_DIR / "test.csv", index=False)

    print("Saved:")
    print(" - data/processed/train.csv")
    print(" - data/processed/test.csv")

if __name__ == "__main__":
    clean_and_split()