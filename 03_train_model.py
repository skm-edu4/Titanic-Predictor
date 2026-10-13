import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv("train_cleaned.csv")

features = ["pclass", "sex", "age", "fare", "sibsp", "parch"]
X = df[features]
y = df["survived"]

X = pd.get_dummies(X, columns=["sex"], drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print(f"Features used: {features}")
print(f"Training rows: {len(X_train)} | Testing rows: {len(X_test)}")
print(f"Accuracy: {accuracy:.2%}")
