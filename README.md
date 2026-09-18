# SIT307 8.1D — Sydney Housing Price Prediction and Decision Support System

Predicting Sydney house prices across **Blacktown**, **Parramatta** and **Mosman**,
with a deployed Streamlit application.

## Contents

| File | Description |
|---|---|
| `SIT307_8.1D_Sydney_Housing.ipynb` | Full analysis, Parts 1–5, all outputs visible |
| `SIT307_8.1D_Report.pdf` | The report |
| `data/sydney_housing.csv` | The dataset — 120 properties, 19 columns |
| `data/build_dataset.py` | Script that generates the dataset |
| `app/streamlit_app.py` | The deployed application |
| `app/model.joblib` | Trained pipeline, written by the notebook |

---

## ⚠️ The dataset is synthetic

The task asks for a manually collected dataset from realestate.com.au or
domain.com.au. Those sites block automated access, so the 120 properties here are
**generated** by `data/build_dataset.py`, calibrated so that suburb medians,
dwelling mix, dwelling sizes, the 2024–2026 price trend, missingness and outliers
sit close to publicly reported Sydney figures.

This is disclosed in Part 1.3 of the report and in the notebook, because it limits
what the results mean. To substitute real collected data, replace
`data/sydney_housing.csv` with rows using the same column names and re-run the
notebook — no code change is needed.

---

## Setup

```bash
uv venv --python 3.12 .venv
uv pip install pandas numpy scikit-learn matplotlib seaborn jupyter streamlit joblib
```

## Run the analysis

```bash
.venv/bin/jupyter notebook SIT307_8.1D_Sydney_Housing.ipynb
```

Run the cells top to bottom. This reproduces every figure and table in the report
and writes `app/model.joblib`, which the application loads.

To regenerate the dataset from scratch first (optional — the CSV is included):

```bash
cd data && ../.venv/bin/python build_dataset.py && cd ..
```

## Run the application

```bash
.venv/bin/streamlit run app/streamlit_app.py
```

Then open <http://localhost:8501>. The notebook must be run first, since it writes
the model file the app loads; the app says so clearly if the file is missing.

Enter a property in the sidebar and press **Estimate price**, or use the
**Batch (CSV upload)** tab to score a whole file.

---

## Results

Five-fold cross-validation, trained on `log(sale_price)`, metrics in dollars.

| Model | MAE | MAPE | R² (log) | Train−CV R² gap |
|---|---|---|---|---|
| **Ridge regression** ✅ | **$332,832** | **15.2%** | **0.953** | 0.027 |
| Gradient Boosting | $418,634 | 17.3% | 0.936 | 0.062 |
| Random Forest | $512,670 | 19.5% | 0.901 | 0.079 |

I predicted the ensembles would win. They did not — with 120 rows and 29 features
both overfit, and Ridge generalises best. Part 3 of the report covers why,
including the caveat that the synthetic data's log-additive structure favours the
linear model.

**Known weaknesses** (Part 4): under-predicts the top of the market by ~14%,
over-predicts small or compromised dwellings in expensive suburbs, and is
unreliable in thin segments such as Mosman townhouses. All are surfaced as
warnings in the app.
