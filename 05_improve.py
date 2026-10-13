import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

df = pd.read_csv("train_cleaned.csv")

features = ["pclass", "sex", "age", "fare", "sibsp", "parch", "embark_town", "alone"]
X = df[features]
y = df["survived"]
X = pd.get_dummies(X, columns=["sex", "embark_town"], drop_first=True)
X["alone"] = X["alone"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

models = {
    "Logistic Regression": make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=1000)
    ),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest (100)": RandomForestClassifier(n_estimators=100, random_state=42),
    "Random Forest (500)": RandomForestClassifier(n_estimators=500, random_state=42),
}

print("=== Model Comparison ===")
print(f"{'Model':<22} {'Test Accuracy':>14} {'5-Fold CV':>10}")
print("-" * 48)

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    test_acc = accuracy_score(y_test, model.predict(X_test))
    cv_acc = cross_val_score(model, X, y, cv=5).mean()
    results[name] = (test_acc, cv_acc, model)
    print(f"{name:<22} {test_acc:>13.2%} {cv_acc:>9.2%}")

best_name = max(results, key=lambda n: results[n][1])
best_model = results[best_name][2]
print(f"\nBest model (by cross-validation): {best_name}")

if hasattr(best_model, "feature_importances_"):
    importances = pd.Series(
        best_model.feature_importances_, index=X.columns
    ).sort_values(ascending=False)
    print("\n=== Feature Importances (what the model learned) ===")
    print(importances.round(3).to_string())

old_features = ["pclass", "sex", "age", "fare", "sibsp", "parch"]
print(f"\nBaseline (Phase 3, {len(old_features)} features): 80.45%")
print(f"Improved ({len(features)} features, tuned models):  {results[best_name][0]:.2%}")
