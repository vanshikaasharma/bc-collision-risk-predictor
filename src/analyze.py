# Exploratory Data Analysis
import pandas as pd

df = pd.read_csv("data/processed/clean.csv")

print("Total crashes:", len(df))
print("Years:", df["year"].min(), "to", df["year"].max())
print()

# Crashes per city (top 10)
by_city = df["municipality"].value_counts().head(10)
print("Top 10 cities by crash count:")
print(by_city)
print()

# Crashes by hour (when do most happen?)
by_hour = df["hour"].value_counts().sort_index()
print("Crashes by hour:")
print(by_hour)
