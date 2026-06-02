# BC Collision Risk Predictor

Machine learning project that analyzes **ICBC-reported motor vehicle crashes** across British Columbia’s Lower Mainland (2021–2025). It identifies high-risk driving contexts by time, location, and crash type, and presents results in an interactive **Streamlit** dashboard.

Built with Python, Pandas, scikit-learn, XGBoost, and Plotly.

---

## What it does

| Stage | Script | Output |
|-------|--------|--------|
| Load & clean | `src/check_data.py` | `data/processed/clean.csv` |
| Build target | `src/Target.py` | `data/processed/training_data.csv` |
| Train & compare models | `src/train_models.py` | `models/risk_model.joblib`, `models/metrics.json` |
| Explore & predict | `app/dashboard.py` | Interactive web app |

**Pipeline flow:**

```
ICBC CSV  →  cleaning  →  feature engineering  →  grouped targets
         →  model comparison (Dummy, LR, RF, XGBoost)
         →  best model saved  →  dashboard
```

---

## Results (held-out 20% test set)

| Model | Accuracy | ROC-AUC |
|-------|----------|---------|
| Dummy (baseline) | 0.735 | 0.500 |
| Logistic Regression | 0.827 | 0.893 |
| Random Forest | 0.781 | 0.906 |
| **XGBoost** (selected) | **0.892** | **0.956** |

The final model is chosen automatically by **highest test ROC-AUC** (Dummy excluded).

**Note:** The model predicts **high-risk contexts** (city + time + crash type groups), not individual driver behavior. Train/test metrics are on grouped rows derived from ~950k crash records.

---

## Data

- **Source:** [ICBC Statistics & open data](https://www.icbc.com/about-icbc/newsroom/Statistics) — *ICBC Reported Crashes* (Tableau export)
- **Scope:** Lower Mainland, 2021–2025 (~953k rows)
- **Local path:** `data/raw/Lower Mainland_Full Data_data.csv`

Data files are **not** in this repo (size + licence). Download from ICBC and place the file locally.

**Attribution (required when using ICBC data):**  
*Contains information licensed under ICBC's Open Data Licence.*

---

## Setup

```bash
cd bc-collision-risk-predictor
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run the pipeline

```bash
python src/check_data.py
python src/Target.py
python src/train_models.py
streamlit run app/dashboard.py
```

Open [http://localhost:8501](http://localhost:8501).

---

## Project structure

```
├── app/dashboard.py          # Streamlit UI
├── src/
│   ├── check_data.py         # Load ICBC CSV, clean, feature engineering
│   ├── analyze.py            # Quick EDA from terminal
│   ├── Target.py             # Build high_risk target
│   └── train_models.py       # Compare models, save best to joblib
├── data/raw/                 # Place ICBC export here
├── data/processed/           # clean.csv, training_data.csv (generated)
└── models/                   # risk_model.joblib, metrics.json (generated)
```

---

## Dashboard pages

1. **Overview** — crash totals, top cities, hourly patterns  
2. **Model results** — comparison table and ROC-AUC chart  
3. **Predict risk** — what-if scenario with risk probability gauge  

---

## Tech stack

Python · Pandas · scikit-learn · XGBoost · Plotly · Streamlit · joblib
