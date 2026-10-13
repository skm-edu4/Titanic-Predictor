import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv("train_cleaned.csv")

features = ["pclass", "sex", "age", "fare", "sibsp", "parch"]
X = df[features]
y = df["survived"]
X = pd.get_dummies(X, columns=["sex"], drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

test_passengers = X_test.copy()
test_passengers["survived_actual"] = y_test
test_passengers.to_csv("test.csv")

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("=== Evaluation ===")
print(f"Accuracy: {accuracy_score(y_test, predictions):.2%}\n")
print("Confusion matrix (rows=actual, cols=predicted):")
print(confusion_matrix(y_test, predictions), "\n")
print("Classification report:")
print(classification_report(y_test, predictions, target_names=["Died", "Survived"]))

submission = pd.DataFrame({
    "PassengerIndex": X_test.index,
    "Survived": predictions,
})
submission.to_csv("my_predictions.csv", index=False)

print("Saved test.csv (179 unseen passengers)")
print("Saved my_predictions.csv (model predictions)")
print("\n=== First 10 predictions ===")
print(submission.head(10).to_string(index=False))
