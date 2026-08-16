# Predicting Smartphone Addiction

This repository contains my code and analysis for the [Predicting Smartphone Addiction](https://www.kaggle.com/competitions/playground-series-s6e8/overview) Kaggle competition.

## Dataset

To comply with Kaggle's data security rules, the competition dataset is not hosted in this repository. To run the code, you will need to download the data directly from Kaggle.

## Experiments & Model Iterations

### Attempt 1: Baseline LightGBM Classifier

Notebook: `Notebooks/LightGBM.ipynb`

#### Overview
Attempt 1 establishes a clean baseline model using a LightGBM Classifier evaluated with Stratified 5-Fold Cross-Validation.

#### Data Summary
- Train Dimensions: 691,369 rows, 14 columns
- Test Dimensions: 296,302 rows, 13 columns
- Target Column: `addicted_label` (Binary Classification)
- Target Distribution:
  - Class 1 (Addicted): 490,474 samples (70.94%)
  - Class 0 (Not Addicted): 200,895 samples (29.06%)
- Missing Values:
  - Train Missing Values: 870,360 total null entries across features
  - Test Missing Values: 377,157 total null entries across features

#### Preprocessing & Categorical Handling
- Categorical Features: `gender`, `stress_level`, `academic_work_impact`
- Encoding Strategy: Converted to pandas `category` data types for native LightGBM histogram-based split searching.
- Dropped Features: `id` (non-predictive row identifier).

#### Training Configuration
- Model: `lgb.LGBMClassifier`
- Objective: Binary Classification
- Evaluation Metric: ROC-AUC
- Hyperparameters: `n_estimators=1500`, `learning_rate=0.05`, `num_leaves=31`, `random_state=42`
- Early Stopping: 50 rounds on validation set

### Attempt 2: Advanced Feature Engineering LightGBM Classifier

Notebook: `Notebooks/improvedLGBM.ipynb`

#### Overview
Attempt 2 enhances the baseline model by introducing domain-specific feature engineering (ratios, rhythm metrics, interaction densities, and group aggregations) alongside hyperparameter tuning, evaluated with the exact same 5-Fold Stratified Cross-Validation (seed=42).

#### Data & Feature Summary
- Total Features: 33 features (12 base features + 21 engineered features)
- Engineered Feature Groups:
  - **Usage & Time Ratios:** `social_media_ratio`, `gaming_ratio`, `work_study_ratio`, `leisure_hours`, `leisure_ratio`
  - **Rhythm & Wellbeing Metrics:** `screen_sleep_ratio`, `weekend_daily_diff`, `weekend_daily_ratio`, `active_non_screen_hours`
  - **Phone Interaction Densities:** `notifications_per_screen_hour`, `app_opens_per_screen_hour`, `notifications_per_app_open`
  - **Group Aggregations & Categoricals:** `age_group` (binned quantile buckets), plus group-level mean and std aggregations of `daily_screen_time_hours` and `notifications_per_day` grouped by `stress_level` and `age_group`.

#### Preprocessing & Categorical Handling
- Categorical Features: `gender`, `stress_level`, `academic_work_impact`, `age_group`
- Encoding Strategy: Converted to pandas `category` data types for native LightGBM histogram-based split searching.
- Inf Handling: Zero-division `inf` values replaced with `np.nan` (handled natively by LightGBM).

#### Training Configuration
- Model: `lgb.LGBMClassifier`
- Objective: Binary Classification (`binary`)
- Evaluation Metric: ROC-AUC (`auc`)
- Hyperparameters: `n_estimators=2000`, `learning_rate=0.035`, `num_leaves=45`, `subsample=0.8`, `colsample_bytree=0.8`, `random_state=42`
- Early Stopping: 50 rounds on validation set

#### Validation Results
- Fold 1 ROC-AUC: 0.962712
- Fold 2 ROC-AUC: 0.963557
- Fold 3 ROC-AUC: 0.963897
- Fold 4 ROC-AUC: 0.964475
- Fold 5 ROC-AUC: 0.963424
- **Overall Out-Of-Fold (OOF) ROC-AUC Score: 0.963613**


