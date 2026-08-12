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
