import pandas as pd
from pathlib import Path

raw_path = "data/raw/Lower Mainland_Full Data_data.csv"
clean_path = Path("data/processed/clean.csv")

# Reading the data
df = pd.read_csv(raw_path, encoding="utf-16", sep=",")

# Filtering the columns
keep_cols = [
    "Date Of Loss Year",
    "Month Of Year",
    "Municipality Name",
    "Crash Severity",
    "Derived Crash Configuration",
    "Time Category",
    "Total Victims",
    "Intersection Crash",
    "Latitude",
    "Longitude",
    "Street Full Name",
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
    "Total Victims": "total_victims",
    "Intersection Crash": "intersection_crash",
    "Latitude": "latitude",
    "Longitude": "longitude",
    "Street Full Name": "street",
})


#Numbering the months
month_map = {
    "JANUARY": 1, "FEBRUARY": 2, "MARCH": 3, "APRIL": 4,
    "MAY": 5, "JUNE": 6, "JULY": 7, "AUGUST": 8,
    "SEPTEMBER": 9, "OCTOBER": 10, "NOVEMBER": 11, "DECEMBER": 12,
}
df["month"] = df["month"].astype(str).str.strip().str.upper().map(month_map)

# Numbering the hours
time_map = {
    "00:00-02:59": 1,
    "03:00-05:59": 4,
    "06:00-08:59": 7,
    "09:00-11:59": 10,
    "12:00-14:59": 13,
    "15:00-17:59": 16,
    "18:00-20:59": 19,
    "21:00-23:59": 22,
}
df["hour"] = df["time_category"].astype(str).str.strip().map(time_map)

# Cleaning the labels
df["municipality"] = df["municipality"].astype(str).str.title()
df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")

# Street names — normalize; drop useless values
df["street"] = df["street"].astype(str).str.strip().str.upper()
df.loc[df["street"].isin(["", "UNKNOWN", "NAN", "NONE"]), "street"] = pd.NA

# Saving the cleaned data (faster to reload next time)
clean_path.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(clean_path, index=False)

print("Saved:", clean_path)
print("Shape:", df.shape)
print(df[["year", "month", "hour", "municipality", "severity"]].head(3))

# Reloading the cleaned data
df2 = pd.read_csv(clean_path)
print("\nReloaded shape:", df2.shape)
