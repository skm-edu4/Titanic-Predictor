"""08_final_model.py — Round 3: final evaluation protocol.

  1. Global median imputation ignores groups
      -> age imputed with median(age | sex, pclass), statistics FIT ON TRAIN ONLY
  2. Rich raw features unused
      -> `deck` letter added (explicit "Unknown" category doubles as the
         missingness indicator — no separate has_deck column needed)
  3. Test set reused for many reads
      -> three-way split: train (60%) / validation (20%) / final holdout (20%)
         * GridSearchCV on TRAIN only (5-fold CV)
         * decision-threshold tuning on VALIDATION
         * holdout evaluated EXACTLY ONCE, at the very end

Run from the project root:   venv/bin/python 08_final_model.py
"""

import os

import matplotlib
matplotlib.use("Agg")  # headless backend — no windows, works over SSH/CI
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import binomtest
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, RocCurveDisplay)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

os.makedirs("plots", exist_ok=True)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(f"plots/{name}", dpi=150)
    plt.close(fig)
    print(f"  saved plots/{name}")


def mcnemar(y_true, pred_a, pred_b):
    """McNemar's exact test: are two paired classifiers different?"""
    y_true = np.asarray(y_true)
    b = int(((pred_a == y_true) & (pred_b != y_true)).sum())  # A right, B wrong
    c = int(((pred_a != y_true) & (pred_b == y_true)).sum())  # A wrong, B right
    p = binomtest(b, b + c, 0.5).pvalue if (b + c) > 0 else 1.0
    return b, c, p


# =================================================== 1. raw data + 3-way split
raw = pd.read_csv("train.csv")   # RAW data (script 01) — cleaning stats fit on train below

X_raw = raw.drop(columns=["survived"])
y = raw["survived"]

X_train, X_temp, y_train, y_temp = train_test_split(
    X_raw, y, test_size=0.4, random_state=42, stratify=y)
X_val, X_hold, y_val, y_hold = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

print("=== 3-way stratified split (60 / 20 / 20) ===")
for name, ys in [("train", y_train), ("validation", y_val), ("holdout", y_hold)]:
    print(f"  {name:<11} n={len(ys):>3}  survival={ys.mean():.1%}")

# ============================== 2. cleaning statistics FIT ON TRAIN, then applied
age_lookup = X_train.groupby(["sex", "pclass"])["age"].median().to_dict()
global_age = X_train["age"].median()
embark_mode = X_train["embark_town"].mode()[0]

n_imputed = {name: int(df["age"].isna().sum())
             for name, df in [("train", X_train), ("validation", X_val), ("holdout", X_hold)]}
print(f"\n=== Group-wise age imputation (fit on train only) ===")
print(f"  missing ages filled — train: {n_imputed['train']}, "
      f"validation: {n_imputed['validation']}, holdout: {n_imputed['holdout']}")
print("  fitted group medians (train):")
print(X_train.groupby(["sex", "pclass"])["age"].median().round(1).to_string())

# edge case: if a sex x pclass group was entirely NaN in train, its median is NaN
# -> the second .fillna(global_age) catches exactly that (documented fallback).


def clean(df):
    df = df.copy()
    keys = pd.Series(list(zip(df["sex"], df["pclass"])), index=df.index)
    df["age"] = df["age"].fillna(keys.map(age_lookup)).fillna(global_age)
    df["embark_town"] = df["embark_town"].fillna(embark_mode)
    return df


def engineer(df):
    df = df.copy()
    df["family_size"] = df["sibsp"] + df["parch"] + 1
    df["is_child"] = (df["age"] < 12).astype(int)
    # "Unknown" is itself the missingness indicator — a separate has_deck column
    # would be a duplicate (one-hot of Unknown == has_deck, just inverted).
    df["deck"] = df["deck"].fillna("Unknown")
    return df


X_train, X_val, X_hold = (engineer(clean(d)) for d in (X_train, X_val, X_hold))

NUM = ["pclass", "age", "fare", "sibsp", "parch", "family_size", "is_child"]
CAT = ["sex", "embark_town", "deck"]
FEATURES = NUM + CAT

# ==================================================== 3. GridSearchCV on TRAIN
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

pre = ColumnTransformer([
    ("num", "passthrough", NUM),
    ("cat", OneHotEncoder(handle_unknown="ignore"), CAT),
])
pipe = Pipeline([("prep", pre),
                 ("clf", RandomForestClassifier(random_state=42))])
param_grid = {
    "clf__n_estimators": [100, 300, 500],
    "clf__max_depth": [None, 8, 16],
    "clf__min_samples_leaf": [1, 2, 4],
    "clf__max_features": ["sqrt", 0.5],
}  # same 54-candidate grid as Round 2, for continuity

print("\n=== GridSearchCV (5-fold CV on TRAIN only) ===")
grid = GridSearchCV(pipe, param_grid, cv=CV, scoring="accuracy", n_jobs=-1)
grid.fit(X_train[FEATURES], y_train)
print(f"  best CV accuracy: {grid.best_score_:.2%}")
print(f"  best params     : {grid.best_params_}")

best = grid.best_estimator_

# ============================ 4. threshold tuning on VALIDATION (never holdout)
val_prob = best.predict_proba(X_val[FEATURES])[:, 1]
thresholds = np.arange(0.30, 0.71, 0.02)
rows = []
for t in thresholds:
    pred = (val_prob >= t).astype(int)
    rows.append((t,
                 accuracy_score(y_val, pred),
                 precision_score(y_val, pred, zero_division=0),
                 recall_score(y_val, pred, zero_division=0),
                 f1_score(y_val, pred, zero_division=0)))
sweep = pd.DataFrame(rows, columns=["threshold", "accuracy", "precision", "recall", "f1"])
best_t = float(sweep.loc[sweep["accuracy"].idxmax(), "threshold"])
val_acc_05 = float(sweep.loc[sweep["threshold"].round(2) == 0.5, "accuracy"].iloc[0])
print(f"\n=== Decision-threshold tuning on VALIDATION (n={len(y_val)}) ===")
print(f"  best threshold = {best_t:.2f}  (val accuracy {sweep['accuracy'].max():.2%} "
      f"vs {val_acc_05:.2%} at the default 0.5)")

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(sweep["threshold"], sweep["accuracy"], lw=2.5, label="accuracy")
ax.plot(sweep["threshold"], sweep["precision"], "--", label="precision")
ax.plot(sweep["threshold"], sweep["recall"], "--", label="recall")
ax.axvline(best_t, color="red", lw=1, ls=":",
           label=f"selected threshold = {best_t:.2f}")
ax.set_xlabel("decision threshold")
ax.set_ylabel("score (validation set)")
ax.set_title("Threshold sweep on the validation set")
ax.legend()
save(fig, "threshold_tuning.png")

# ============ 5. interpretation on VALIDATION (holdout still unread at this point)
perm = permutation_importance(best, X_val[FEATURES], y_val, n_repeats=10,
                              random_state=42, scoring="accuracy")
perm_imp = pd.Series(perm.importances_mean, index=FEATURES).sort_values(ascending=False)
print("\n=== Permutation importance (VALIDATION set, 10 repeats) ===")
print(perm_imp.round(4).to_string())

# ======================================= 6. HOLDOUT — evaluated EXACTLY ONCE
print("\n" + "=" * 64)
print("FINAL HOLDOUT EVALUATION — first and only read of these 179 rows")
print("=" * 64)
hold_prob = best.predict_proba(X_hold[FEATURES])[:, 1]
hold_pred_05 = (hold_prob >= 0.5).astype(int)
hold_pred_t = (hold_prob >= best_t).astype(int)

hold_acc_05 = accuracy_score(y_hold, hold_pred_05)
hold_acc_t = accuracy_score(y_hold, hold_pred_t)
hold_auc = roc_auc_score(y_hold, hold_prob)

b, c, p = mcnemar(y_hold, hold_pred_05, hold_pred_t)
print(f"  accuracy @ threshold {best_t:.2f}: {hold_acc_t:.2%}")
print(f"  accuracy @ 0.5 (default): {hold_acc_05:.2%}")
print(f"  ROC-AUC: {hold_auc:.3f}")
print(f"  McNemar (tuned threshold vs 0.5): b={b}, c={c}, p={p:.3f}")
if p >= 0.05:
    print("  -> p >= 0.05: threshold tuning did NOT significantly change holdout accuracy.")
print()
print(classification_report(y_hold, hold_pred_t, target_names=["Died", "Survived"]))

fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(confusion_matrix(y_hold, hold_pred_t), annot=True, fmt="d", cmap="Blues",
            xticklabels=["Died", "Survived"], yticklabels=["Died", "Survived"], ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title(f"Final holdout — tuned RF (acc {hold_acc_t:.1%}, AUC {hold_auc:.3f})")
save(fig, "holdout_confusion.png")

fig, ax = plt.subplots(figsize=(6, 5))
RocCurveDisplay.from_estimator(best, X_hold[FEATURES], y_hold, ax=ax)
ax.set_title(f"Final holdout ROC (AUC = {hold_auc:.3f})")
save(fig, "holdout_roc.png")

print("=== Comparison across rounds (different splits/protocols) ===")
print("  Round 1/2 test accuracy : 80.45% - 82.68% (single 80/20 split)")
print(f"  Round 3 CV (train)      : {grid.best_score_:.2%} (5-fold)")
print(f"  Round 3 final holdout   : {hold_acc_t:.2%} @ threshold {best_t:.2f} | AUC {hold_auc:.3f}")
print("\nDone — holdout will not be read again.")
