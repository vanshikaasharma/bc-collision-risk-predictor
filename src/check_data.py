import pandas as pd

path = "data/raw/Lower Mainland_Full Data_data.csv"

# Reading the data
df = pd.read_csv(path, encoding="utf-16", sep=",")

# Filtering the columns
keep_cols = [
    "Date Of Loss Year",
    "Month Of Year",
    "Municipality Name",
    "Crash Severity",
    "Derived Crash Configuration",
    "Time Category",
    "Region",
    "Total Victims",
    "Intersection Crash",
]

df = df[keep_cols].copy()

# Renaming the columns
df = df.rename(columns={
    "Date Of Loss Year": "year",
    "Month Of Year": "month",
    "Municipality Name": "municipality",
    "Crash Severity": "severity",
    "Derived Crash Configuration": "collision_type",
    "Time Category": "time_category",
    "Region": "region",
    "Total Victims": "total_victims",
    "Intersection Crash": "intersection_crash",
})

print(df.head(5))
print("\nColumns:", df.columns.tolist())
print("\nShape:", df.shape)
