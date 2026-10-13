"""06_visualize.py — Exploratory Data Analysis (EDA) plots.

Run from the project root:   venv/bin/python 06_visualize.py
Writes PNGs into plots/ (tracked in git — they render in the README).
"""

import os

import matplotlib
matplotlib.use("Agg")  # non-interactive backend: saves files without opening windows
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")

df = pd.read_csv("train_cleaned.csv")
df["survived_label"] = df["survived"].map({0: "Died", 1: "Survived"})
df["family_size"] = df["sibsp"] + df["parch"] + 1

os.makedirs("plots", exist_ok=True)


def save(fig, name):
    """Save a figure to plots/ and close it (frees memory)."""
    fig.tight_layout()
    fig.savefig(f"plots/{name}", dpi=150)
    plt.close(fig)
    print(f"  saved plots/{name}")


print("=== Generating EDA plots ===")

# 1. Survival rate by sex and ticket class
fig, ax = plt.subplots(figsize=(7, 5))
sns.barplot(
    data=df, x="sex", y="survived", hue="class",
    hue_order=["First", "Second", "Third"], errorbar=None, ax=ax,
)
ax.set_title("Survival rate by sex and ticket class")
ax.set_ylabel("Survival rate")
ax.set_xlabel("")
save(fig, "survival_by_sex_class.png")

# 2. Age distribution: survivors vs victims
fig, ax = plt.subplots(figsize=(7, 5))
sns.histplot(data=df, x="age", hue="survived_label", kde=True, bins=30, ax=ax)
ax.set_title("Age distribution: survivors vs victims")
save(fig, "age_distribution.png")

# 3. Fare vs age
fig, ax = plt.subplots(figsize=(7, 5))
sns.scatterplot(data=df, x="age", y="fare", hue="survived_label", alpha=0.6, ax=ax)
ax.set_ylim(0, 300)  # cap: a few $500+ outlier fares would squash everything else
ax.set_title("Fare vs age (fares above $300 clipped for readability)")
save(fig, "fare_vs_age.png")

# 4. Correlation heatmap (numeric features only)
fig, ax = plt.subplots(figsize=(7, 5))
numeric = ["survived", "pclass", "age", "sibsp", "parch", "fare"]
sns.heatmap(df[numeric].corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
ax.set_title("Feature correlation matrix")
save(fig, "correlation_heatmap.png")

# 5. Survival by family size (motivates the family_size feature in script 07)
fig, ax = plt.subplots(figsize=(7, 5))
sns.barplot(data=df, x="family_size", y="survived", errorbar=None,
            color="#4c72b0", ax=ax)
ax.set_title("Survival rate by family size")
ax.set_xlabel("Family size (siblings + spouses + parents + children + 1)")
save(fig, "family_size_survival.png")

print("Done — 5 plots written to plots/")
