# Compare regressors with temporal holdout, then save XGBoost on all years
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

from train_utils import (
    CAT_COLS,
    FEATURE_COLS,
    FEATURE_COLS_TEMPORAL,
    NUM_COLS,
    TEMPORAL_FOLDS,
    load_training_df,
    temporal_split,
)


def make_encoder(feature_cols):
    cat = [c for c in CAT_COLS if c in feature_cols]
    num = [c for c in NUM_COLS if c in feature_cols]
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
        ("num", "passthrough", num),
    ])


def compare_regressors(X_train, y_train, X_test, y_test, encoder):
    models = {
        "Dummy (mean)": DummyRegressor(strategy="mean"),
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=100, max_depth=8, random_state=42, n_jobs=-1
        ),
        "XGBoost": XGBRegressor(
            n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42
        ),
    }
    X_train_enc = encoder.fit_transform(X_train)
    X_test_enc = encoder.transform(X_test)

    comparison = []
    best_name = None
    best_model = None
    best_r2 = -999

    for name, model in models.items():
        model.fit(X_train_enc, y_train)
        pred = model.predict(X_test_enc)
        mae = mean_absolute_error(y_test, pred)
        rmse = np.sqrt(mean_squared_error(y_test, pred))
        r2 = r2_score(y_test, pred)
        comparison.append({
            "model": name,
            "mae": round(mae, 3),
            "rmse": round(rmse, 3),
            "r2": round(r2, 3),
        })
        if name != "Dummy (mean)" and r2 > best_r2:
            best_r2 = r2
            best_name = name
            best_model = model

    return comparison, best_name, best_model, best_r2, encoder


df, top_streets = load_training_df()
models_dir = Path("models")
models_dir.mkdir(exist_ok=True)

temporal_results = []
print("Regressor temporal holdout (no year feature):\n")

for fold in TEMPORAL_FOLDS:
    train_df, test_df = temporal_split(df, fold["train_max_year"], fold["test_year"])
    feat = FEATURE_COLS_TEMPORAL
    X_train, y_train = train_df[feat], train_df["risk_score"]
    X_test, y_test = test_df[feat], test_df["risk_score"]

    encoder = make_encoder(feat)
    comparison, best_name, _, best_r2, _ = compare_regressors(
        X_train, y_train, X_test, y_test, encoder
    )

    print(f"Train {fold['train_label']} → test {fold['test_year']}")
    print(f"{'Model':<22} {'MAE':>8} {'RMSE':>8} {'R²':>8}")
    print("-" * 50)
    for row in comparison:
        print(f"{row['model']:<22} {row['mae']:>8.3f} {row['rmse']:>8.3f} {row['r2']:>8.3f}")
    print(f"Best: {best_name} (R² {best_r2:.3f})\n")

    temporal_results.append({
        "train_years": fold["train_label"],
        "test_year": fold["test_year"],
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "comparison": comparison,
        "best_model": best_name,
        "best_r2": round(best_r2, 3),
    })

primary = temporal_results[0]
(models_dir / "regressor_metrics.json").write_text(json.dumps({
    "target": "risk_score",
    "eval_method": "temporal_holdout",
    "features_temporal": FEATURE_COLS_TEMPORAL,
    "temporal_folds": temporal_results,
    "comparison": primary["comparison"],
    "best_model": primary["best_model"],
    "best_r2": primary["best_r2"],
    "primary_test_year": primary["test_year"],
}, indent=2))

X_all = df[FEATURE_COLS]
y_all = df["risk_score"]
prod_encoder = make_encoder(FEATURE_COLS)
X_enc = prod_encoder.fit_transform(X_all)

prod_model = XGBRegressor(
    n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42
)
prod_model.fit(X_enc, y_all)

joblib.dump(
    {
        "model": prod_model,
        "encoder": prod_encoder,
        "feature_cols": FEATURE_COLS,
        "model_name": "XGBoost",
        "top_streets": top_streets,
        "target": "risk_score",
        "trained_on": "all years 2021–2025",
    },
    models_dir / "risk_regressor.joblib",
)

print("Saved regressor_metrics.json and risk_regressor.joblib")
