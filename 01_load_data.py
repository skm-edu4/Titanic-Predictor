import seaborn as sns
import pandas as pd

df = sns.load_dataset("titanic")
df.to_csv("train.csv", index=False)

print("=== First 5 rows (head) ===")
print(df.head())

print("\n=== Statistical summary (describe) ===")
print(df.describe(include="all"))
