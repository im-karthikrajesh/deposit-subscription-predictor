# deposit-subscription-predictor

> Machine learning models to predict customer subscription to bank term deposits from telemarketing data.


## Overview
This repository contains an end-to-end classification pipeline that predicts whether a contacted customer will **subscribe to a bank term deposit**. The work is developed in Google Colab and includes data preparation, feature engineering, model selection with hyperparameter tuning, and evaluation, plus persisted artifacts for reuse.

## Dataset
- **Shape:** 4000 rows × 16 columns (as loaded in Colab)
- **Target:** `subscription_status` (renamed from `y`)
- **Class balance:** `no` = 3139 (78.5%), `yes` = 861 (21.5%)
- **Note:** The dataset file path in the notebook is set to `/content/cwk_data_20706811.csv`. Update the path to place the file elsewhere.

## Feature Engineering
Selected predictors used for modeling (excluding the call duration field to avoid leakage):
- `job`
- `education`
- `marital_housing_interaction`
- `housing_loan_interaction`
- `contact_poutcome_interaction`
- `previous`
- `pdays`
- `balance`
- `age`

**Interaction features created:**
- **marital_housing_interaction** = marital + housing
- **housing_loan_interaction** = housing + loan
- **contact_poutcome_interaction** = contact + poutcome


## Modeling Approach
- **Train/test split:** 80/20 with stratification (`random_state=42`)
- **Imbalance handling:** `SMOTE` applied **only on the training set**
- **Preprocessing:**
  - Categorical variables: `OneHotEncoder(drop='first')`
  - Numerical variables: `StandardScaler` for linear/KNN pipelines
  - Tree-based pipelines pass numerical features through without scaling
- **Algorithms evaluated:**
- Logistic Regression
- Gaussian Naive Bayes
- Decision Tree
- Random Forest (final model)
- K-Nearest Neighbors
- **Hyperparameter tuning:** `RandomizedSearchCV` (`cv=5`, `n_iter=10`, scoring = precision/recall/F1/ROC-AUC, `refit='precision'`)

## Results
**Cross-validation (10-fold on the resampled training set, final model = Random Forest):**
- Precision = **0.9215**
- Recall = **0.7971**
- F1-score = **0.8267**
- ROC AUC = **0.9325**

**Hold-out test set:**
- Precision = **0.6211**
- Recall = **0.3430**
- F1-score = **0.4419**
- ROC AUC = **0.7249**

> The gap between cross-validation and test scores likely reflects distribution shift and the effect of training-time oversampling (SMOTE). Threshold tuning and calibrated probabilities could further align precision/recall to operational targets.

## Artifacts
The notebook persists reusable assets:
- **Model:** `final_model_rf.pkl`
- **Encoder:** `one_hot_encoder.pkl`

## Quickstart
1. **Clone** the repository and place the dataset at a known path (or update the path in the notebook).
2. **Install dependencies:**
   ```bash
   pip install pandas numpy scikit-learn imbalanced-learn seaborn matplotlib joblib

   ```
3. **Open the notebook** in Colab/Jupyter and run all cells.
4. **(Optional) Inference:** load the artifacts and score new data:
   ```python
   import joblib, pandas as pd

   model = joblib.load("final_model_rf.pkl")
   encoder = joblib.load("one_hot_encoder.pkl")

   new_data = pd.DataFrame({
       "job": ["management"],
       "education": ["tertiary"],
       "marital_housing_interaction": ["married_yes"],
       "housing_loan_interaction": ["yes_no"],
       "contact_poutcome_interaction": ["cellular_success"],
       "previous": [1],
       "pdays": [30],
       "balance": [1500],
       "age": [45],
   })
   # Apply the same one-hot encoding columns as in training before predict/predict_proba
   # (see notebook for the exact preprocessing steps).
   ```

## Repository Structure
```
.
├─ Predictive_Analysis_for_Platinum_Deposit_Subscription.ipynb
├─ models/
│  ├─ final_model_rf.pkl
│  └─ one_hot_encoder.pkl
└─ README.md
```

## Requirements
Create a `requirements.txt` with the following minimal stack:

```txt
pandas
numpy
scikit-learn
imbalanced-learn
seaborn
matplotlib
joblib
```

## Notes & Next Steps
- Consider **calibration** (e.g., `CalibratedClassifierCV`) and **threshold selection** to meet business precision/recall targets.
- Track feature drift and re-train periodically, oversampling should be applied **inside** the training pipeline only.
- For fair comparison, ensure identical preprocessing across models and during inference.

---

**Author:** Karthik Rajesh  
**Environment:** Google Colab (Python, scikit-learn, imbalanced-learn, seaborn, matplotlib)
