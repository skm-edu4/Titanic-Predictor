"""07_advanced_model.py — Round 2: production-grade ML practices.

Improvements over scripts 03-05:

  1. stratify=y in the split          -> keeps the 38% survival ratio in BOTH sets
  2. Encoding INSIDE an sklearn Pipeline (ColumnTransformer +
     OneHotEncoder(handle_unknown="ignore")) -> no get_dummies-before-split issue
  3. Feature engineering: family_size, is_child
  4. GridSearchCV hyperparameter tuning (5-fold, train set only — test stays untouched)
  5. Shuffled+seeded StratifiedKFold for order-independent CV
  6. Permutation importance (less biased than impurity feature_importances_)
  7. Baseline-vs-tuned comparison: CV scores, ROC-AUC, and McNemar's
     exact test to check whether a test-set gap is even statistically real

Run from the project root:   venv/bin/python 07_advanced_model.py
"""

import os

import matplotlib
matplotlib.use("Agg")  # headless backend — no windows, works over SSH/CI
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import binomtest
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score, RocCurveDisplay)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     cross_val_score, train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

os.makedirs("plots", exist_ok=True)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(f"plots/{name}", dpi=150)
    plt.close(fig)
    print(f"  saved plots/{name}")


# ---------------------------------------------------------------- load data
df = pd.read_csv("train_cleaned.csv")

# -------------------------------------------------------- feature engineering
df["family_size"] = df["sibsp"] + df["parch"] + 1
df["is_child"] = (df["age"] < 12).astype(int)  # under-12s had vastly higher survival

BASE_NUM = ["pclass", "age", "fare", "sibsp", "parch"]   # script-03 baseline
BASE_CAT = ["sex"]
BASE_COLS = BASE_NUM + BASE_CAT
ENG_NUM = BASE_NUM + ["family_size", "is_child"]          # engineered set
ENG_CAT = BASE_CAT + ["embark_town"]

X = df[ENG_NUM + ENG_CAT]
y = df["survived"]

# ------------------------------------------------------- ONE stratified split
# Same rows for baseline AND tuned model -> apples-to-apples comparison.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("=== Split check (stratify=y keeps class balance) ===")
print(f"  survival rate — full: {y.mean():.1%} | train: {y_train.mean():.1%} | test: {y_test.mean():.1%}")
print("  (scripts 03-05 used an un-stratified split — compare WITHIN this table)\n")

# Same seeded, shuffled, stratified folds for baseline AND grid search,
# so their CV numbers are directly comparable.
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


def build_pipeline(num_cols, cat_cols, clf):
    """Preprocessing + model as ONE object: encoding can never leak or desync."""
    pre = ColumnTransformer(
        transformers=[
            ("num", "passthrough", num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )
    return Pipeline([("prep", pre), ("clf", clf)])


# ------------------------------------------------------------- 1. baseline
baseline = build_pipeline(
    BASE_NUM, BASE_CAT,
    RandomForestClassifier(n_estimators=100, random_state=42),
)
baseline.fit(X_train[BASE_COLS], y_train)
base_pred = baseline.predict(X_test[BASE_COLS])
base_prob = baseline.predict_proba(X_test[BASE_COLS])[:, 1]
base_acc = accuracy_score(y_test, base_pred)
base_auc = roc_auc_score(y_test, base_prob)
base_cv = cross_val_score(baseline, X_train[BASE_COLS], y_train, cv=CV)
print("=== Baseline (6 features, default Random Forest) ===")
print(f"  CV accuracy (train, 5-fold): {base_cv.mean():.2%} ± {base_cv.std():.2%}")
print(f"  test accuracy: {base_acc:.2%} | test ROC-AUC: {base_auc:.3f}\n")

# ------------------------------------------------- 2. GridSearchCV tuning
pipe = build_pipeline(
    ENG_NUM, ENG_CAT,
    RandomForestClassifier(random_state=42),  # n_jobs at 1: GridSearchCV does the parallelism
)
param_grid = {
    "clf__n_estimators": [100, 300, 500],
    "clf__max_depth": [None, 8, 16],
    "clf__min_samples_leaf": [1, 2, 4],
    "clf__max_features": ["sqrt", 0.5],
}  # 3 x 3 x 3 x 2 = 54 candidates x 5 folds = 270 fits

print("=== GridSearchCV (5-fold CV on the TRAINING set only) ===")
grid = GridSearchCV(pipe, param_grid, cv=CV, scoring="accuracy", n_jobs=-1)
grid.fit(X_train, y_train)
print(f"  best CV accuracy : {grid.best_score_:.2%}")
print(f"  best params      : {grid.best_params_}\n")

# ------------------------------------------- 3. final eval on untouched test
best = grid.best_estimator_
tuned_pred = best.predict(X_test)
tuned_prob = best.predict_proba(X_test)[:, 1]
tuned_acc = accuracy_score(y_test, tuned_pred)
tuned_auc = roc_auc_score(y_test, tuned_prob)

print("=== Tuned model on the untouched test set ===")
print(f"  test accuracy: {tuned_acc:.2%} | test ROC-AUC: {tuned_auc:.3f}\n")
print(classification_report(y_test, tuned_pred, target_names=["Died", "Survived"]))

# ------------------------- 4. comparison: CV, AUC, and significance
print("=== Baseline vs tuned ===")
print(f"  {'Model':<34} {'CV acc (5-fold)':>17} {'Test acc':>10} {'Test AUC':>10}")
print("  " + "-" * 74)
print(f"  {'Baseline (6 feat, default RF)':<34} {base_cv.mean():>16.2%} {base_acc:>9.2%} {base_auc:>10.3f}")
print(f"  {'Tuned (8 feat, GridSearchCV)':<34} {grid.best_score_:>16.2%} {tuned_acc:>9.2%} {tuned_auc:>10.3f}")

# McNemar's exact test on PAIRED predictions: is the test-set gap real,
# or just noise from which 179 rows happened to land in the test set?
b = int(((base_pred == y_test.values) & (tuned_pred != y_test.values)).sum())  # baseline right, tuned wrong
c = int(((base_pred != y_test.values) & (tuned_pred == y_test.values)).sum())  # baseline wrong, tuned right
p_value = binomtest(b, b + c, 0.5).pvalue if (b + c) > 0 else 1.0
print(f"\n  McNemar's exact test on paired test predictions: b={b}, c={c}, p={p_value:.3f}")
if p_value >= 0.05:
    print("  -> p >= 0.05: the test-set difference is STATISTICAL NOISE (models are tied).")
    print("     Model selection therefore stays CV-based, never test-based.")
else:
    print("  -> p < 0.05: the test-set difference is statistically significant.")
print()

# --------------------------------------------- 5. plots: ROC + confusion
fig, ax = plt.subplots(figsize=(6, 5))
RocCurveDisplay.from_estimator(baseline, X_test[BASE_COLS], y_test,
                               ax=ax, name="Baseline (6 feat)")
RocCurveDisplay.from_estimator(best, X_test, y_test,
                               ax=ax, name="Tuned (8 feat, GridSearch)")
ax.set_title("ROC curve — baseline vs tuned")
save(fig, "roc_curve.png")

fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(confusion_matrix(y_test, tuned_pred), annot=True, fmt="d", cmap="Blues",
            xticklabels=["Died", "Survived"], yticklabels=["Died", "Survived"], ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title(f"Confusion matrix — tuned Random Forest ({tuned_acc:.1%})")
save(fig, "confusion_matrix_advanced.png")

# --------------------------------------------- 6. two views of feature importance
# (a) impurity-based, at the one-hot encoded column level
names = [n.split("__", 1)[1] for n in best.named_steps["prep"].get_feature_names_out()]
impurity = pd.Series(best.named_steps["clf"].feature_importances_, index=names)
print("=== Impurity importance (biased toward continuous/high-cardinality features) ===")
print(impurity.sort_values(ascending=False).head(8).round(3).to_string())

# (b) permutation: shuffle ONE test column, measure accuracy drop — works on the
#     original input columns and is not biased by cardinality.
#     NOTE: computed on the test set for INTERPRETATION only — never for model
#     selection (that decision was made by CV inside GridSearchCV).
perm = permutation_importance(best, X_test, y_test, n_repeats=10,
                              random_state=42, scoring="accuracy")
perm_imp = pd.Series(perm.importances_mean, index=X_test.columns).sort_values(ascending=False)
print("\n=== Permutation importance (mean accuracy drop when column is shuffled) ===")
print(perm_imp.round(4).to_string())

fig, ax = plt.subplots(figsize=(7, 5))
perm_imp.sort_values(ascending=True).plot.barh(ax=ax, color="#55a868")
ax.set_title("Permutation importance (test set, 10 repeats)")
ax.set_xlabel("mean accuracy drop when the feature is shuffled")
save(fig, "permutation_importance.png")

print("\nDone.")
