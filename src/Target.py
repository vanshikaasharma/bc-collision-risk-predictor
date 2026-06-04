# Build high_risk target at street + time + crash-type level (Lower Mainland)
import pandas as pd
from pathlib import Path

df = pd.read_csv("data/processed/clean.csv")

# Score each crash (injury crashes count more than property-only)
is_casualty = df["severity"].astype(str).str.upper().str.contains("CASUALTY")
df["weight"] = 1
df.loc[is_casualty, "weight"] = 2
df.loc[is_casualty & (df["total_victims"] >= 2), "weight"] = 3

# Streets missing from ICBC export — keep as UNKNOWN for grouping
df["street"] = df["street"].fillna("UNKNOWN")

# Group by location + time + crash context (no region — all Lower Mainland)
group_cols = [
    "municipality",
    "street",
    "year",
    "month",
    "hour",
    "collision_type",
    "intersection_crash",
]

grouped = df.groupby(group_cols).agg(
    crash_count=("weight", "count"),
    risk_score=("weight", "sum"),
).reset_index()

# Top 25% of severity-weighted activity in each group = high_risk
cutoff = grouped["risk_score"].quantile(0.75)
grouped["high_risk"] = (grouped["risk_score"] >= cutoff).astype(int)

out = Path("data/processed/training_data.csv")
grouped.to_csv(out, index=False)

print("Saved:", out)
print("Rows (groups):", len(grouped))
print("Unique streets:", grouped["street"].nunique())
print("High risk %:", round(grouped["high_risk"].mean() * 100, 1))
print("\nSample:")
print(grouped.head(5))
