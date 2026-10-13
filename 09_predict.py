"""09_predict.py — deployable inference CLI.

Run from the project root:

    venv/bin/python 09_predict.py                       # demo passengers
    venv/bin/python 09_predict.py '{"pclass":1,"sex":"female","age":29,\
"fare":100.0,"sibsp":1,"parch":1,"embark_town":"Cherbourg","deck":"B"}'

After the final evaluation (script 08), the production model is refit on
ALL 891 labeled rows — standard practice: evaluation is finished, so use all
available labels. No metric is computed here.

The SAME sklearn Pipeline does preprocessing + prediction -> train/serve skew
is impossible by construction.
"""

import json
import sys

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# ---- frozen decisions from Round 3 (see 08_final_model.py) ----
HYPERPARAMS = {"n_estimators": 100, "max_depth": None, "max_features": "sqrt",
               "min_samples_leaf": 2, "random_state": 42}
THRESHOLD = 0.56  # tuned on the validation set in script 08

NUM = ["pclass", "age", "fare", "sibsp", "parch", "family_size", "is_child"]
CAT = ["sex", "embark_town", "deck"]
REQUIRED = ["pclass", "sex", "age", "fare", "sibsp", "parch", "embark_town", "deck"]

DEMO_PASSENGERS = [
    {"pclass": 1, "sex": "female", "age": 29, "fare": 100.0,
     "sibsp": 1, "parch": 1, "embark_town": "Cherbourg", "deck": "B"},
    {"pclass": 3, "sex": "male", "age": 8, "fare": 20.0,
     "sibsp": 1, "parch": 2, "embark_town": "Southampton", "deck": "Unknown"},
    {"pclass": 3, "sex": "male", "age": 30, "fare": 8.0,
     "sibsp": 0, "parch": 0, "embark_town": "Southampton", "deck": "Unknown"},
]


def fit_production_model():
    """Refit on all 891 labeled rows — evaluation is complete."""
    raw = pd.read_csv("train.csv")

    # Same cleaning statistics as script 08, refit on all labeled data.
    age_lookup = raw.groupby(["sex", "pclass"])["age"].median().to_dict()
    global_age = raw["age"].median()
    embark_mode = raw["embark_town"].mode()[0]

    df = raw.copy()
    keys = pd.Series(list(zip(df["sex"], df["pclass"])), index=df.index)
    df["age"] = df["age"].fillna(keys.map(age_lookup)).fillna(global_age)
    df["embark_town"] = df["embark_town"].fillna(embark_mode)
    df["family_size"] = df["sibsp"] + df["parch"] + 1
    df["is_child"] = (df["age"] < 12).astype(int)
    df["deck"] = df["deck"].fillna("Unknown")

    pre = ColumnTransformer([
        ("num", "passthrough", NUM),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT),
    ])
    pipe = Pipeline([
        ("prep", pre),
        ("clf", RandomForestClassifier(**HYPERPARAMS)),
    ])
    pipe.fit(df[NUM + CAT], df["survived"])

    imputers = {"age_lookup": age_lookup, "global_age": global_age,
                "embark_mode": embark_mode}
    return pipe, imputers


def engineer_one(passenger, imputers):
    """Exactly the same feature steps as training — one shared code path."""
    missing = [k for k in REQUIRED if k not in passenger]
    if missing:
        raise ValueError(f"missing fields {missing}; required: {REQUIRED}")

    row = dict(passenger)
    if row.get("age") is None:  # edge case: age omitted -> group median fallback
        row["age"] = imputers["age_lookup"].get(
            (row["sex"], row["pclass"]), imputers["global_age"])
    if row.get("embark_town") is None:
        row["embark_town"] = imputers["embark_mode"]
    row["deck"] = row.get("deck") or "Unknown"

    row["family_size"] = row["sibsp"] + row["parch"] + 1
    row["is_child"] = int(row["age"] < 12)
    return pd.DataFrame([row])[NUM + CAT]


def main():
    if len(sys.argv) > 1:
        try:
            passengers = [json.loads(sys.argv[1])]
        except json.JSONDecodeError as exc:
            sys.exit(f"Invalid JSON: {exc}\n"
                     f"Pass ONE object with fields: {REQUIRED}")
    else:
        passengers = DEMO_PASSENGERS

    pipe, imputers = fit_production_model()
    print(f"Production model refit on 891 rows | frozen threshold = {THRESHOLD}\n")

    for i, passenger in enumerate(passengers, 1):
        X = engineer_one(passenger, imputers)
        p_survive = float(pipe.predict_proba(X)[0, 1])
        label = "SURVIVED" if p_survive >= THRESHOLD else "DIED"
        desc = (f"{passenger['pclass']}-class {passenger['sex']}, "
                f"age {passenger['age']}, fare {passenger['fare']}, "
                f"{passenger['embark_town']}, deck {passenger['deck']}")
        print(f"{i}. {desc:<75} -> {label:<9} (p = {p_survive:.2f})")

    print("\nNote: tree models never extrapolate — a 200-year-old receives the "
          "same prediction as the oldest training passenger.")


if __name__ == "__main__":
    main()
