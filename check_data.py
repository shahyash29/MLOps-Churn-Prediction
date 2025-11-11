import pandas as pd 
def check_telco_csv(path='data/raw/telco.csv'):
    df = pd.read_csv(path)
    print("Dataset loaded successfully!")
    print(f"Shape of data: {df.shape}")
    print(f"Columns in data: {list(df.columns)}\n")
    print(df.dtypes)
    print("Missing values in each column:")
    print(df.isnull().sum(), "\n")

    if "Churn" in df.columns:
        print("Target variable distribution (Churn):")
        print(df["Churn"].value_counts())
        print("\nChurn rate (%):")
        print(df["Churn"].value_counts(normalize=True) * 100)
    else:
        print("'Churn' column not found in dataset!")

if __name__ == "__main__":
    check_telco_csv()