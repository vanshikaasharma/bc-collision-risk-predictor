# Shared helpers for classifier/regressor training
import pandas as pd

TOP_STREETS = 150

# Walk-forward: train on all years <= train_max_year, evaluate on test_year only
TEMPORAL_FOLDS = [
    {"train_max_year": 2023, "test_year": 2024, "train_label": "2021–2023"},
    {"train_max_year": 2024, "test_year": 2025, "train_label": "2022–2024"},
]

FEATURE_COLS = [
    "municipality", "street", "year", "month", "hour",
    "collision_type", "intersection_crash",
]
# No year — fair check that patterns generalize to a future year
FEATURE_COLS_TEMPORAL = [
    "municipality", "street", "month", "hour",
    "collision_type", "intersection_crash",
]

CAT_COLS = ["municipality", "street", "collision_type", "intersection_crash"]
NUM_COLS = ["year", "month", "hour"]
NUM_COLS_TEMPORAL = ["month", "hour"]


def load_training_df(path="data/processed/training_data.csv"):
    df = pd.read_csv(path)
    top = df["street"].value_counts().head(TOP_STREETS).index.tolist()
    df["street"] = df["street"].where(df["street"].isin(top), "OTHER")
    return df, top


def temporal_split(df, train_max_year, test_year):
    train = df[df["year"] <= train_max_year]
    test = df[df["year"] == test_year]
    return train, test
