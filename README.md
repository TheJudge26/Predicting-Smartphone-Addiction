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

### Attempt 3: Histogram XGBoost Classifier

Notebook: `Notebooks/XGBoostClassifier.ipynb`

#### Overview
Attempt 3 introduces model diversity by training an XGBoost Classifier (`XGBClassifier`) using the histogram-based tree building method (`tree_method='hist'`) and native categorical feature support on the exact 33-feature set engineered in Attempt 2 with the identical 5-Fold Stratified Cross-Validation split (seed=42).

#### Data & Feature Summary
- Total Features: 33 features (12 base features + 21 engineered features matching Attempt 2)
- Feature Set Included: Time ratios, sleep/screen rhythm metrics, phone interaction densities, quantile age binning (`age_group`), and group mean/std aggregations by `stress_level` and `age_group`.

#### Preprocessing & Categorical Handling
- Categorical Features: `gender`, `stress_level`, `academic_work_impact`, `age_group`
- Encoding Strategy: Converted to pandas `category` data types for native XGBoost categorical split searching (`enable_categorical=True`).
- Inf Handling: Zero-division `inf` values replaced with `np.nan` (handled natively by XGBoost).

#### Training Configuration
- Model: `xgb.XGBClassifier`
- Tree Method: `hist` (Histogram-based tree algorithm)
- Evaluation Metric: `auc`
- Hyperparameters: `n_estimators=2000`, `learning_rate=0.03`, `max_depth=6`, `subsample=0.8`, `colsample_bytree=0.8`, `enable_categorical=True`, `random_state=42`
- Early Stopping: 50 rounds on validation set

#### Validation Results
- Fold 1 ROC-AUC: 0.963264
- Fold 2 ROC-AUC: 0.963802
- Fold 3 ROC-AUC: 0.964093
- Fold 4 ROC-AUC: 0.964710
- Fold 5 ROC-AUC: 0.963668
- **Overall Out-Of-Fold (OOF) ROC-AUC Score: 0.963906**

#### Generated Artifacts
- Out-Of-Fold Predictions: `Data/Processed/oof_xgb_attempt3.npy`
- Test Set Predictions: `Data/Processed/test_xgb_attempt3.npy`
- Submission File: `Data/Processed/submission_attempt3_xgb.csv`

### Attempt 4: Model Blending & Ensembling (LightGBM + XGBoost)

Notebook: `Notebooks/Blending.ipynb`

#### Overview
Attempt 4 combines the Out-Of-Fold (OOF) validation predictions and test predictions from Attempt 2 (LightGBM Classifier) and Attempt 3 (Histogram XGBoost Classifier) using Weighted Probability Blending and Percentile Rank Averaging to optimize local CV ROC-AUC and minimize prediction variance.

#### Methodology & Blending Techniques
- **Input Models**: Attempt 2 (LightGBM, OOF ROC-AUC: 0.963613) and Attempt 3 (XGBoost, OOF ROC-AUC: 0.963906).
- **Prediction Correlation Analysis**: Evaluated Pearson correlation coefficient ($r$) between model prediction vectors to measure ensembling diversity.
- **Weighted Probability Blending**:
  - Optimized blending weight $w^* \in [0.0, 1.0]$ via SciPy Nelder-Mead optimization (`scipy.optimize.minimize`) on the objective function:
    $$\text{Blend OOF} = w \cdot \text{OOF}_{\text{LGB}} + (1 - w) \cdot \text{OOF}_{\text{XGB}}$$
  - Cross-verified with a 101-step grid search over $w \in [0.0, 1.0]$.
- **Percentile Rank Averaging**:
  - Normalized predictions using uniform percentile ranks (`scipy.stats.rankdata`):
    $$\text{Rank Blend} = w^* \cdot \text{Rank}(\text{LGB}) + (1 - w^*) \cdot \text{Rank}(\text{XGB})$$
  - Handled probability calibration differences between LightGBM and XGBoost tree outputs.

#### Generated Artifacts & Submission
- **Final Submission File**: `Data/Processed/submission_attempt4_blend.csv`
- **Submission Dimensions**: 296,302 rows, 2 columns (`id`, `addicted_label`)
- **Integrity Verification**: 0 missing values, probability values strictly bounded within $[0.0, 1.0]$.

### Attempt 5: Master Blend (Supercharged Features + 3-Way Tri-Model Ensemble)

Notebook: `Notebooks/Blending2.ipynb`

#### Overview
Attempt 5 supercharges our feature engineering pipeline with frequency encodings, stress/work group deviations, and log transformations, retrains LightGBM and XGBoost, introduces Model Family #3 (CatBoost Classifier), and performs a 3-Way SLSQP Optimal Weight Optimization to maximize out-of-fold ROC-AUC.

#### Data & Feature Upgrades Summary
- **Total Features:** 36 features (12 base features + 24 engineered features)
- **New Feature Additions:**
  - **Frequency (Count) Encoding:** `count_age`, `count_notifications`, `count_app_opens`, `count_screen_time`
  - **Group Aggregations & Stress Deviations:** Mean/std of `daily_screen_time_hours` grouped by `(stress_level, academic_work_impact)`, individual stress deviation (`daily_screen_time_hours - mean_screen_time_by_stress`), and mean/std of `notifications_per_day` grouped by `age_group`.
  - **Log Transformations:** `log_notifications` ($\log(1 + \text{notifications\_per\_day})$), `log_app_opens` ($\log(1 + \text{app\_opens\_per\_day})$).

#### Preprocessing & Categorical Handling
- Categorical Features: `gender`, `stress_level`, `academic_work_impact`, `age_group`
- Encoding Strategy: Converted to string-backed pandas `category` data types (`.astype(str).astype('category')`) for 100% cross-framework compatibility across LightGBM, XGBoost, and CatBoost.
- Inf Handling: Infinite values from ratio zero-divisions replaced with `np.nan`.

#### Single Model Validation Results (5-Fold Stratified CV, SEED=42)
- **LightGBM Classifier:** 5-Fold OOF ROC-AUC = **0.964637** (Up from 0.963613 in Attempt 2!)
- **Histogram XGBoost Classifier:** 5-Fold OOF ROC-AUC = **0.965059** (Up from 0.963906 in Attempt 3!)
- **CatBoost Classifier:** 5-Fold OOF ROC-AUC = **0.963231**

#### 3-Way Ensembling & Weight Optimization (SciPy SLSQP)
- **Pearson Correlation Analysis:** High diversity confirmed across prediction vectors ($r_{\text{LGB,XGB}} = 0.99641$, $r_{\text{LGB,CAT}} = 0.99339$, $r_{\text{XGB,CAT}} = 0.99440$).
- **Optimal Weight Solution:** $[w_{\text{LGB}} = 0.3329, w_{\text{XGB}} = 0.3342, w_{\text{CAT}} = 0.3329]$
- **3-Way Weighted Probability Blend OOF ROC-AUC:** **0.964840**
- **3-Way Percentile Rank Averaging OOF ROC-AUC:** **0.964829**

#### Generated Artifacts & Submission
- **Out-Of-Fold Predictions:** `Data/Processed/oof_lgb_attempt5.npy`, `Data/Processed/oof_xgb_attempt5.npy`, `Data/Processed/oof_cat_attempt5.npy`
- **Test Set Predictions:** `Data/Processed/test_lgb_attempt5.npy`, `Data/Processed/test_xgb_attempt5.npy`, `Data/Processed/test_cat_attempt5.npy`
- **Final Submission File:** `submissions/submission_attempt5_master_blend.csv` (and `Data/Processed/submission_attempt5_master_blend.csv`)
- **Submission Dimensions:** 296,302 rows, 2 columns (`id`, `addicted_label`)
- **Integrity Verification:** 0 missing values, probability values strictly bounded within $[0.0, 1.0]$.
