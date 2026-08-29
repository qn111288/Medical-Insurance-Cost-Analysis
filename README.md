# Health Insurance Portfolio Analysis

Final project for the Fundamentals of Data Science course (USTH) — a machine learning pipeline that predicts insurance **premium** from customer demographic, behavioral, and policy data.

**Note:** The included report (`Medical_insurance_analysis_report.pdf`) describes an analysis on the classic Kaggle "Medical Cost" dataset (1,338 records, target `charges`). The current notebook (`Health_insurance_.ipynb`) instead runs on the Mendeley "Dataset of health insurance portfolio" (228,711 records, target `premium`). These should be reconciled before publishing.

## Dataset

- Source: Mendeley "Dataset of health insurance portfolio" (via Kaggle)
- Size: 228,711 rows, 42 original columns (100+ after feature engineering)
- Target: `premium` (continuous, heavily right-skewed)
- Features: numeric (age, seniority, claims cost, medical services used), categorical (gender, distribution channel, product type, region), and date fields (policy/insured effect and lapse dates)
- Some columns have high missing rates (e.g. lapse dates missing ~76%)

## Pipeline

1. **Data loading** — auto-detects environment (Kaggle / Colab / local) and target column
2. **EDA** — descriptive stats, missing values, correlations, temporal/segment/geographic breakdowns
3. **Feature engineering** — temporal features, insurance-specific ratios, interaction and aggregation features
4. **Modeling** — XGBoost, LightGBM, CatBoost, tuned with Optuna (early stopping, GPU support)
5. **Ensembling** — weighted/simple averaging of tuned models
6. **Interpretation** — feature importance (SHAP where available)

## Results

| Model | R² | MAE | RMSE |
|---|---|---|---|
| XGBoost (Optuna-tuned) | 0.797 | 112.9 | 246.8 |
| XGBoost (baseline) | 0.626 | 176.8 | 334.6 |
| LightGBM (baseline) | 0.581 | 191.0 | 354.1 |
| CatBoost (baseline) | 0.562 | 197.9 | 362.1 |

80/20 train-test split, `random_state=42`. Strongest correlates with `premium`: age, number of medical services, insured seniority, annual claims cost.

## Project Structure

```
├── Health_insurance_.ipynb
├── Medical_insurance_analysis_report.pdf
├── data/       (local dataset folder, not tracked)
├── plots/      (generated output, not tracked)
└── README.md
```

## Setup & Usage

**Kaggle:** add the dataset, enable GPU (optional), Run All. Plots are saved to `/kaggle/working/plots/`.

**Colab:** upload the notebook, place the data under `/content/data/`, run cells in order.

**Local:**
```bash
git clone <repo-url>
cd <repo-folder>
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```
Place the data file at `data/Dataset of health insurance portfolio/Dataset of health insurance portfolio.xlsx`, then run the notebook.

**requirements.txt**
```
pandas==2.2.3
numpy==1.26.4
scikit-learn
matplotlib
seaborn
xgboost
lightgbm
catboost
optuna
shap
openpyxl
imbalanced-learn
```

## Tech Stack

pandas, numpy, scipy, matplotlib, seaborn, scikit-learn, XGBoost, LightGBM, CatBoost, Optuna, SHAP.

## Limitations

- High missingness in several columns; current imputation is simple median fill
- The ensemble step has a feature-count mismatch bug across base models, which hurts ensemble performance versus the best single model
- SHAP did not run to completion in the last execution and fell back to default feature importance
- Notebook and PDF report currently reference different datasets and results — needs reconciliation

## References

- Chen & Guestrin, *XGBoost: A Scalable Tree Boosting System*, KDD 2016
- Ke et al., *LightGBM: A Highly Efficient Gradient Boosting Decision Tree*, NIPS 2017
- Prokhorenkova et al., *CatBoost: Unbiased Boosting with Categorical Features*, NeurIPS 2018
- Lundberg & Lee, *A Unified Approach to Interpreting Model Predictions (SHAP)*, NeurIPS 2017
- Akiba et al., *Optuna: A Next-generation Hyperparameter Optimization Framework*, KDD 2019
