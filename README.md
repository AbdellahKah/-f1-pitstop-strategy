# Kaggle F1 Pit Stops Prediction (Playground Series S6E5)

This repository contains a high-performance machine learning pipeline designed to predict F1 driver pit stops (`PitNextLap`) using tabular strategy data. The solution leverages advanced domain-specific feature engineering, robust out-of-fold validation, data augmentation, and optimized ensembling of Gradient Boosted Decision Trees (GBDTs) as well as Tabular Neural Networks.

---

## 📂 Project Structure

```bash
├── f1_strategy_dataset_v4.csv              # Original external dataset used for augmentation
├── train.csv                               # Competition training dataset
├── test.csv                                # Competition test dataset
├── sample_submission.csv                   # Sample submission template
├── train_ensemble.py                       # GBDT ensemble pipeline (LightGBM, XGBoost, CatBoost)
├── ps-s6-e5-realmlp-pytabkit_example.ipynb  # Deep learning pipeline using PyTabKit (RealMLP)
├── oof_predictions.csv                     # Out-of-fold predictions generated from the GBDT ensemble
├── submission.csv                          # Final optimized predictions for Kaggle submission
└── README.md                               # Project documentation
```

---

## 📁 Data

Competition data (`train.csv`, `test.csv`, `sample_submission.csv`) is not included in this repository per Kaggle's terms of use. Download it directly from the competition page:

👉 [Kaggle Playground Series S6E5](https://www.kaggle.com/competitions/playground-series-s6e5/data)

The external augmentation dataset (`f1_strategy_dataset_v4.csv`) can be found on Kaggle datasets.

---


## 🛠️ Feature Engineering Pipeline

A major differentiator in predicting F1 pit strategy is constructing domain-specific features that model tire wear, track progression, and team strategies. The following features were engineered and processed:

### 1. Tire Age & Stint Normalization
* **Tire Fitting Lap (`_TyreStartLap`)**: Computed as `LapNumber - TyreLife` to pinpoint exactly which lap the current tire set was fitted.
* **Stint Efficiency (`_TyreLife_Stint_Ratio`)**: Computed as `TyreLife / (Stint + 1e-6)` to relate tire age to the number of stints run.
* **Compound-Specific Tyre Life (`_TyreLife_Compound_Diff` & `_TyreLife_Compound_Norm`)**: Normalizes the tire age against the mean and standard deviation of `TyreLife` grouped by tire compound, capturing how fast specific compounds wear out.

### 2. Degradation & Pace Ratios
* **Degradation Rate (`_Degradation_Per_Lap`)**: Computed as `Cumulative_Degradation / (TyreLife + 1e-6)` to measure the velocity of tire wear.
* **Degradation to Pace Ratio (`_Degradation_LapTime_Ratio`)**: Tracks how pace relates to tire degradation.
* **Pace-Degradation Interactions**:
  * `_LapTime (s)_*_Cumulative_Degradation`
  * `_LapTime (s)_*_Cumulative_Degradation_abs`
  * `_LapTime (s)_/_Cumulative_Degradation_abs`

### 3. Race Context & Estimation
* **Circuit Laps Estimator (`_TotalLaps_est`)**: Calculated as `LapNumber / (RaceProgress + 1e-8)` to dynamically estimate the total length (laps) of a race.
* **Stint Progression Ratios**:
  * `_LapNumber_/_RaceProgress`
  * `_TyreLife_/_LapNumber`

### 4. Categorical & Count Encodings
* **Strategy Frequency Mapping**: `_Driver_count`, `_Race_count`, and `_Compound_count` capture how often specific drivers, races, or tire compounds appear in the combined dataset, representing track-specific or driver-specific defaults.
* **High-Cardinality Interaction Features**:
  * `Race_Compound` (captures tire choice under track-specific conditions)
  * `Race_Year` (captures track-specific rules/characteristics across years)

---

## 🧪 Validation & Augmentation Strategy

To ensure model generalizability and prevent overfitting:
1. **5-Fold Stratified Cross-Validation**: Folds are split based on the target `PitNextLap` to preserve target distribution across training and validation folds.
2. **Leakage-Proof Data Augmentation**: Folds of the original external dataset (`f1_strategy_dataset_v4.csv`) are split and merged with corresponding folds of the competition dataset (`train.csv`) *inside* the cross-validation loop. This ensures validation folds remain strictly untouched by the augmented data.
3. **Out-of-Fold Target Encoding**: TargetEncoder is fit on each training split for high-cardinality interaction features (`Race_Compound`, `Race_Year`) and transformed onto the validation and test splits.

---

## 📈 Models & Results

The project implements two distinct modeling methodologies:

### 1. GBDT Ensemble (`train_ensemble.py`)
Three gradient boosting libraries are trained with early stopping on the 5-fold splits. SciPy's Nelder-Mead optimization is then used to find the mathematically optimal blending weights on the Out-of-Fold (OOF) predictions.

| Model / Blend | OOF ROC-AUC Score | Blending Weight |
|---|---|---|
| **LightGBM** | 0.947793 | -0.0128 |
| **XGBoost** | 0.951681 | 0.6829 |
| **CatBoost** | 0.949583 | 0.3299 |
| **Optimized Ensemble** | **0.952266** | **1.0000** |

*The ensembling process yielded a **0.952266 OOF ROC-AUC**, outperforming the single best model (XGBoost).*

### 2. Tabular Neural Networks (`ps-s6-e5-realmlp-pytabkit_example.ipynb`)
An alternative pipeline utilizing **PyTabKit's `RealMLP_TD_Classifier`** (an advanced tabular Multilayer Perceptron) is provided. It implements robust feature preprocessors (median centering, robust scaling, smooth clipping, one-hot/embedding layers) and reaches individual fold AUC scores between **0.9530 and 0.9551**.

---

## 🚀 How to Run

### Dependencies
Ensure you have the required packages installed:
```bash
pip install pandas numpy lightgbm xgboost catboost scikit-learn scipy pytorch pytabkit
```

### Running the GBDT Ensemble
To run the complete data loading, feature engineering, training, optimization, and submission pipeline:
```bash
python train_ensemble.py
```
This will produce:
- `oof_predictions.csv`: Out-of-fold predictions for validation.
- `submission.csv`: Final test predictions formatted for Kaggle (shape `(188165, 2)` with columns `id` and `PitNextLap`).

### Running the Tabular Neural Network
Open and run the Jupyter notebook `ps-s6-e5-realmlp-pytabkit_example.ipynb` in your environment (with GPU support recommended for PyTabKit training).
