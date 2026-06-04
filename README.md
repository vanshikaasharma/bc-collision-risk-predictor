# BC Road Risk Intelligence Platform

End-to-end analytics on **ICBC-reported motor vehicle crashes** in British Columbia’s **Lower Mainland (2021–2025)**. The project cleans real open data, groups crashes into street/time contexts, trains two XGBoost models, evaluates them with **walk-forward (next-year) holdout**, and explores results in a **Streamlit** dashboard.

Built with Python, Pandas, scikit-learn, XGBoost, Plotly, and Streamlit.

---

## What this project does

1. **Describes what happened** — totals, cities, hours, maps (historical ICBC data).
2. **Scores where/when risk clusters** — machine learning on grouped contexts (city + street + month + hour + collision type + intersection).
3. **Checks generalization** — train on past years, test only the next year (2024, then 2025).

It does **not** predict individual driver crash probability, weather, or causation. It is **not** province-wide BC (export is Lower Mainland only).

---

## Project in two parts

### Part 1 — Data & exploration (what happened)

| What | Scripts / UI |
|------|----------------|
| Load & clean raw export | `src/check_data.py` → `data/processed/clean.csv` |
| Quick EDA from terminal | `src/analyze.py` |
| Dashboard | **Executive Summary**, **Collision Hotspot Intelligence Map**, **Municipality Analytics** |

### Part 2 — Machine learning (patterns & scoring)

Crashes are **grouped** into contexts; each group gets:

| Field | Meaning |
|-------|--------|
| `risk_score` | Sum of severity weights in that group (PDO = 1, casualty = 2, 2+ victims = 3) |
| `high_risk` | 1 if `risk_score` is in the top ~25% of all groups (75th percentile cutoff) |

Two models on the same groups (`src/Target.py` → `data/processed/training_data.csv`):

| Model | Script | Target | Dashboard output |
|-------|--------|--------|------------------|
| **Classifier** | `src/train_models.py` | `high_risk` (0/1) | **%** chance context is top-tier (`predict_proba`) |
| **Regressor** | `src/train_regressor.py` | `risk_score` | **Activity score** — estimated weighted sum (`predict`) |

### Two ways each model is used

| Use | Purpose |
|-----|--------|
| **Temporal holdout** (`models/metrics.json`) | Train on past years → test **only** the next year. No `year` in features — honest next-year check. |
| **Dashboard models** (`.joblib` files) | Trained on **all years 2021–2025** with `year` included for interactive **Street risk lookup**. |

---

## Key findings (from this dataset)

- **~953k** crash records (2021–2025); **~739k** grouped contexts for ML.
- **Vancouver, Surrey, Burnaby** dominate volume — risk is geographically concentrated.
- **~19%** casualty-related; severity weighting makes injury corridors stand out beyond raw counts.
- Reported crashes **increase year over year** in the export (2021 → 2025).
- Peak activity appears in **afternoon/evening hours** (see dashboard hourly chart).
- **Classifier** next-year holdout: XGBoost **ROC-AUC ~0.78** on 2024 and 2025 tests.
- **Regressor** next-year holdout: **R² ~0.39**, **MAE ~0.63** on `risk_score`.
- Patterns are **stable across holdout folds** — not a single-year artifact.

---

## Results (temporal holdout)

| Train years | Test year | Classifier (ROC-AUC) | Regressor (R²) |
|-------------|-----------|----------------------|----------------|
| 2021–2023 | 2024 | XGBoost **0.775** | XGBoost **0.39** |
| 2022–2024 | 2025 | XGBoost **0.776** | XGBoost **0.39** |

Metrics come from `python src/train_models.py` and `python src/train_regressor.py`. Re-run after data or target changes.

---

## Pipeline

```bash
python src/check_data.py      # → data/processed/clean.csv
python src/Target.py          # → data/processed/training_data.csv
python src/train_models.py    # → models/risk_model.joblib, metrics.json
python src/train_regressor.py # → models/risk_regressor.joblib, regressor_metrics.json
streamlit run app/dashboard.py
```

Open [http://localhost:8501](http://localhost:8501).

---

## Dashboard pages

| Page | What you see |
|------|----------------|
| **Executive Summary** | KPIs, **key findings**, charts (cities, hours, years, collision types) |
| **Collision Hotspot Intelligence Map** | Filtered crash map, clusters, top corridors |
| **Risk Prediction Center** | High-risk table, model metrics, **Street risk lookup** (pick city/street/time → % + score) |
| **Municipality Analytics** | Per-city stats and comparison |

---

## Data

- **Source:** [ICBC Statistics & open data](https://www.icbc.com/about-icbc/newsroom/Statistics) — *ICBC Reported Crashes* (Tableau export)
- **Scope:** Lower Mainland, 2021–2025
- **Local path:** `data/raw/Lower Mainland_Full Data_data.csv`

Data files are **not** in this repo (size + licence). Download from ICBC and place locally.

**Attribution:** *Contains information licensed under ICBC's Open Data Licence.*

---

## Setup

```bash
cd bc-collision-risk-predictor
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## Project structure

```
├── app/dashboard.py
├── src/
│   ├── check_data.py
│   ├── analyze.py
│   ├── Target.py
│   ├── train_utils.py
│   ├── train_models.py
│   └── train_regressor.py
├── data/raw/
├── data/processed/
└── models/
```

---

## Limitations

- Lower Mainland only; not all of BC.
- Context-level ML, not per-driver risk.
- `high_risk` uses a global top-quartile rule across all years.
- 2026+ would be forecast-only until ICBC publishes new years.
- No weather or engineering features in this ICBC export.

---

## Tech stack

Python · Pandas · scikit-learn · XGBoost · Plotly · Streamlit · joblib
