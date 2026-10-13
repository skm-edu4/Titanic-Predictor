import pandas as pd

df = pd.read_csv("train.csv")

print("=== Missing values BEFORE cleaning ===")
print(df.isnull().sum())

median_age = df["age"].median()
print(f"\nMedian age: {median_age}")

df["age"] = df["age"].fillna(median_age)

df["embark_town"] = df["embark_town"].fillna(df["embark_town"].mode()[0])

print("\n=== Missing values AFTER cleaning ===")
print(df.isnull().sum())

df.to_csv("train_cleaned.csv", index=False)
print("\nSaved cleaned data to train_cleaned.csv")
