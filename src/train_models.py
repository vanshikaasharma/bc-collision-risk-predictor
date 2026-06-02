# Compare models on test set, then save whichever gives best results
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

df = pd.read_csv("data/processed/training_data.csv")

feature_cols = [
    "municipality", "year", "month", "hour",
    "collision_type", "intersection_crash", "region",
]
cat_cols = ["municipality", "collision_type", "intersection_crash", "region"]
num_cols = ["year", "month", "hour"]

X = df[feature_cols]
y = df["high_risk"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

encoder = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ("num", "passthrough", num_cols),
])
X_train_enc = encoder.fit_transform(X_train)
X_test_enc = encoder.transform(X_test)

#comparing models
models = {
    "Dummy": DummyClassifier(strategy="most_frequent"),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1),
    "XGBoost": XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42),
}

print("Model comparison:\n")
print(f"{'Model':<28} {'Accuracy':>10} {'ROC-AUC':>10}")
print("-" * 50)

comparison = []
best_name = None
best_model = None
best_auc = -1

for name, model in models.items():
    model.fit(X_train_enc, y_train)
    pred = model.predict(X_test_enc)
    proba = model.predict_proba(X_test_enc)[:, 1]
    acc = accuracy_score(y_test, pred)
    auc = roc_auc_score(y_test, proba)
    print(f"{name:<28} {acc:>10.3f} {auc:>10.3f}")
    comparison.append({"model": name, "accuracy": round(acc, 3), "roc_auc": round(auc, 3)})

    # Dummy is baseline only, so we never pick it as the final model
    if name != "Dummy" and auc > best_auc:
        best_auc = auc
        best_name = name
        best_model = model

models_dir = Path("models")
models_dir.mkdir(exist_ok=True)

(models_dir / "metrics.json").write_text(json.dumps({
    "comparison": comparison,
    "best_model": best_name,
    "best_roc_auc": round(best_auc, 3),
}, indent=2))

joblib.dump(
    {"model": best_model, "encoder": encoder, "feature_cols": feature_cols, "model_name": best_name},
    models_dir / "risk_model.joblib",
)

print(f"\nBest model (by ROC-AUC): {best_name} ({best_auc:.3f})")
print("Saved to models/risk_model.joblib")
