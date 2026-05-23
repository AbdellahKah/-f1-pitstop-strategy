import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import TargetEncoder
from scipy.optimize import minimize
import warnings
import os

warnings.filterwarnings('ignore')

def main():
    print("==================================================")
    print("Starting Kaggle F1 Pit Stops Ensemble Pipeline")
    print("==================================================")
    
    print("Loading data...")
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')
    orig = pd.read_csv('f1_strategy_dataset_v4.csv')

    print(f"Train shape: {train.shape}")
    print(f"Test shape : {test.shape}")
    print(f"Orig shape : {orig.shape}")

    # Fill missing values in Compound column for the original dataset
    orig['Compound'] = orig['Compound'].fillna('UNKNOWN')

    # Drop Normalized_TyreLife from orig as it is not present in train/test
    if 'Normalized_TyreLife' in orig.columns:
        orig = orig.drop('Normalized_TyreLife', axis=1)

    ID = 'id'
    TARGET = 'PitNextLap'

    y = train[TARGET]
    X = train.drop([ID, TARGET], axis=1)
    train_ids = train[ID]

    y_orig = orig[TARGET]
    X_orig = orig.drop([TARGET], axis=1)

    X_test = test.drop([ID], axis=1)
    test_ids = test[ID]

    # Pre-calculate count maps on entire train + test + orig
    combined = pd.concat([X, X_test, X_orig], axis=0).reset_index(drop=True)
    
    driver_counts = combined['Driver'].value_counts()
    race_counts = combined['Race'].value_counts()
    compound_counts = combined['Compound'].value_counts()

    # Compound stats for TyreLife proxy
    compound_tyrelife_stats = X.groupby('Compound')['TyreLife'].agg(['mean', 'std']).to_dict()

    # Actual total laps per race & year based on observed max lap in entire dataset
    total_laps_map = combined.groupby(['Race', 'Year'])['LapNumber'].max().to_dict()

    def engineer_features(df):
        df = df.copy()
        
        # Count features
        df['_Driver_count'] = df['Driver'].map(driver_counts).fillna(0).astype('float32')
        df['_Race_count'] = df['Race'].map(race_counts).fillna(0).astype('float32')
        df['_Compound_count'] = df['Compound'].map(compound_counts).fillna(0).astype('float32')

        # Basic interactions
        df['_LapNumber_/_RaceProgress'] = (df['LapNumber'] / (df['RaceProgress'] + 1e-6)).astype('float32')
        df['_TyreLife_/_LapNumber'] = (df['TyreLife'] / df['LapNumber'].clip(lower=1)).astype('float32')
        df['_LapTime (s)_*_Cumulative_Degradation'] = (df['LapTime (s)'] * df['Cumulative_Degradation']).astype('float32')
        df['_LapTime (s)_*_Cumulative_Degradation_abs'] = (df['LapTime (s)'] * df['Cumulative_Degradation'].abs()).astype('float32')
        df['_LapTime (s)_/_Cumulative_Degradation_abs'] = (df['LapTime (s)'] / (df['Cumulative_Degradation'].abs() + 1e-6)).astype('float32')

        # Actual total laps based on observed max lap per race/year
        df['_TotalLaps_actual'] = pd.MultiIndex.from_frame(df[['Race', 'Year']]).map(total_laps_map).astype('float32')

        # Lap when tires were fitted
        df['_TyreStartLap'] = (df['LapNumber'] - df['TyreLife']).astype('float32')

        # Stint efficiency
        df['_TyreLife_Stint_Ratio'] = (df['TyreLife'] / (df['Stint'] + 1e-6)).astype('float32')

        # TyreLife proxy normalized by Compound
        mean_map = compound_tyrelife_stats['mean']
        std_map = compound_tyrelife_stats['std']
        df['_TyreLife_Compound_Mean'] = df['Compound'].map(mean_map).astype('float32')
        df['_TyreLife_Compound_Std'] = df['Compound'].map(std_map).astype('float32')
        df['_TyreLife_Compound_Diff'] = (df['TyreLife'] - df['_TyreLife_Compound_Mean']).astype('float32')
        df['_TyreLife_Compound_Norm'] = (df['_TyreLife_Compound_Diff'] / (df['_TyreLife_Compound_Std'] + 1e-6)).astype('float32')
        df = df.drop(['_TyreLife_Compound_Mean', '_TyreLife_Compound_Std'], axis=1)

        # Degradation indicators
        df['_Degradation_Per_Lap'] = (df['Cumulative_Degradation'] / (df['TyreLife'] + 1e-6)).astype('float32')
        df['_Degradation_LapTime_Ratio'] = (df['Cumulative_Degradation'] / (df['LapTime (s)'] + 1e-6)).astype('float32')

        # Year interactions
        df['_Year_diff'] = (df['Year'] - 2022).astype('float32')

        # Position-based pit strategy signal
        df['_Position_x_RaceProgress'] = (df['Position'] * df['RaceProgress']).astype('float32')
        df['_Position_x_TyreLife'] = (df['Position'] * df['TyreLife']).astype('float32')

        # Laps remaining estimate based on actual race length
        df['_LapsRemaining_est'] = (df['_TotalLaps_actual'] - df['LapNumber']).astype('float32')

        # TyreLife relative to estimated race end
        df['_TyreLife_vs_LapsRemaining'] = (df['TyreLife'] / (df['_LapsRemaining_est'] + 1e-6)).astype('float32')

        # Gap to leader interaction (Note: GapToLeader column is not present in the dataset)
        # df['_GapToLeader_x_TyreLife'] = (df['GapToLeader'] * df['TyreLife']).astype('float32')
        # df['_GapToLeader_x_RaceProgress'] = (df['GapToLeader'] * df['RaceProgress']).astype('float32')


        # Stint number squared (non-linear stint effect)
        df['_Stint_squared'] = (df['Stint'] ** 2).astype('float32')


        # Convert categories to category type
        for col in ['Driver', 'Compound', 'Race']:
            df[col] = df[col].astype('category')

        # Interaction string categories
        df['Race_Compound'] = (df['Race'].astype(str) + '_' + df['Compound'].astype(str)).astype('category')
        df['Race_Year'] = (df['Race'].astype(str) + '_' + df['Year'].astype(str)).astype('category')

        return df

    print("Initial columns in train dataset:", X.columns.tolist())
    print("Checking if 'Position' in dataset:", 'Position' in X.columns)
    print("Checking if 'GapToLeader' in dataset:", 'GapToLeader' in X.columns)
    print("Engineering features...")
    X = engineer_features(X)
    X_orig = engineer_features(X_orig)
    X_test = engineer_features(X_test)

    print(f"Dataset columns: {X.columns.tolist()}")

    # Set up StratifiedKFold
    folds = 10
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)

    # OOF Prediction Arrays
    oof_preds_lgb = np.zeros(len(X))
    oof_preds_xgb = np.zeros(len(X))
    oof_preds_cat = np.zeros(len(X))

    # Test Prediction Arrays
    test_preds_lgb = np.zeros(len(X_test))
    test_preds_xgb = np.zeros(len(X_test))
    test_preds_cat = np.zeros(len(X_test))

    # Categorical columns
    cat_cols_to_use = ['Driver', 'Compound', 'Race', 'Race_Compound', 'Race_Year']
    te_cols = ['Race_Compound', 'Race_Year']

    # Model Hyperparameters
    lgb_params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'num_leaves': 63,
        'max_depth': 8,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 1,
        'min_child_samples': 50,
        'random_state': 42,
        'n_estimators': 1500,
        'n_jobs': -1,
        'verbose': -1
    }

    xgb_params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'tree_method': 'hist',
        'device': 'cuda',
        'learning_rate': 0.05,
        'max_depth': 8,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 50,
        'random_state': 42,
        'n_estimators': 1500,
        'n_jobs': -1,
        'enable_categorical': True,
        'early_stopping_rounds': 50
    }

    cat_params = {
        'loss_function': 'Logloss',
        'eval_metric': 'AUC',
        'learning_rate': 0.05,
        'depth': 6,
        'iterations': 1500,
        'random_seed': 42,
        'thread_count': -1,
        'verbose': 0
    }

    print(f"Starting {folds}-Fold Cross-Validation training of LGBM, XGBoost, and CatBoost...")

    for fold, ((tr_idx, val_idx), (or_tr_idx, or_val_idx)) in enumerate(
            zip(skf.split(X, y), skf.split(X_orig, y_orig)), 1):
        
        print(f"\n========================================")
        print(f"               FOLD {fold}/{folds}               ")
        print(f"========================================")
        
        X_tr = X.iloc[tr_idx].copy()
        y_tr = y.iloc[tr_idx].copy()
        
        X_val = X.iloc[val_idx].copy()
        y_val = y.iloc[val_idx].copy()
        
        # Append corresponding fold of original dataset to train
        X_tr_orig = X_orig.iloc[or_tr_idx].copy()
        y_tr_orig = y_orig.iloc[or_tr_idx].copy()
        
        X_tr = pd.concat([X_tr, X_tr_orig], axis=0).reset_index(drop=True)
        y_tr = pd.concat([y_tr, y_tr_orig], axis=0).reset_index(drop=True)

        # Target Encoding
        TE = TargetEncoder(cv=5, smooth='auto', shuffle=True, random_state=42)
        X_tr_te = X_tr[te_cols].astype(str)
        X_val_te = X_val[te_cols].astype(str)
        X_test_te = X_test[te_cols].astype(str)

        tr_enc = TE.fit_transform(X_tr_te, y_tr)
        val_enc = TE.transform(X_val_te)
        tst_enc = TE.transform(X_test_te)

        te_names = [f"_{col}_TE" for col in te_cols]
        X_tr[te_names] = tr_enc.astype('float32')
        X_val[te_names] = val_enc.astype('float32')
        
        X_tst_fold = X_test.copy()
        X_tst_fold[te_names] = tst_enc.astype('float32')

        # Ensure correct pandas categories
        for col in cat_cols_to_use:
            X_tr[col] = X_tr[col].astype('category')
            X_val[col] = X_val[col].astype('category')
            X_tst_fold[col] = X_tst_fold[col].astype('category')

        # --------------------
        # 1. LIGHTGBM
        # --------------------
        print("Training LightGBM...")
        model_lgb = lgb.LGBMClassifier(**lgb_params)
        model_lgb.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
        )
        val_preds_lgb = model_lgb.predict_proba(X_val)[:, 1]
        oof_preds_lgb[val_idx] = val_preds_lgb
        test_preds_lgb += model_lgb.predict_proba(X_tst_fold)[:, 1] / folds
        score_lgb = roc_auc_score(y_val, val_preds_lgb)
        print(f"LightGBM AUC: {score_lgb:.6f}")

        # --------------------
        # 2. XGBOOST
        # --------------------
        print("Training XGBoost...")
        model_xgb = xgb.XGBClassifier(**xgb_params)
        model_xgb.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        val_preds_xgb = model_xgb.predict_proba(X_val)[:, 1]
        oof_preds_xgb[val_idx] = val_preds_xgb
        test_preds_xgb += model_xgb.predict_proba(X_tst_fold)[:, 1] / folds
        score_xgb = roc_auc_score(y_val, val_preds_xgb)
        print(f"XGBoost AUC: {score_xgb:.6f}")

        # --------------------
        # 3. CATBOOST
        # --------------------
        print("Training CatBoost...")
        # Convert categories to strings for CatBoost Pool (excluding target encoded high-cardinality interaction cols)
        X_tr_cat = X_tr.drop(['Race_Compound', 'Race_Year'], axis=1)
        X_val_cat = X_val.drop(['Race_Compound', 'Race_Year'], axis=1)
        X_tst_cat = X_tst_fold.drop(['Race_Compound', 'Race_Year'], axis=1)
        cat_features_cat = ['Driver', 'Compound', 'Race']
        for col in cat_features_cat:
            X_tr_cat[col] = X_tr_cat[col].astype(str)
            X_val_cat[col] = X_val_cat[col].astype(str)
            X_tst_cat[col] = X_tst_cat[col].astype(str)

        train_pool = Pool(X_tr_cat, y_tr, cat_features=cat_features_cat)
        val_pool = Pool(X_val_cat, y_val, cat_features=cat_features_cat)
        
        model_cat = CatBoostClassifier(**cat_params)
        model_cat.fit(
            train_pool,
            eval_set=val_pool,
            early_stopping_rounds=50,
            verbose=False
        )
        val_preds_cat = model_cat.predict_proba(X_val_cat)[:, 1]
        oof_preds_cat[val_idx] = val_preds_cat
        test_preds_cat += model_cat.predict_proba(X_tst_cat)[:, 1] / folds
        score_cat = roc_auc_score(y_val, val_preds_cat)
        print(f"CatBoost AUC: {score_cat:.6f}")

    print("\n======================================")
    print("Cross-Validation Training Completed!")
    print("======================================")
    
    score_lgb_all = roc_auc_score(y, oof_preds_lgb)
    score_xgb_all = roc_auc_score(y, oof_preds_xgb)
    score_cat_all = roc_auc_score(y, oof_preds_cat)
    
    print(f"Overall OOF LightGBM AUC: {score_lgb_all:.6f}")
    print(f"Overall OOF XGBoost  AUC: {score_xgb_all:.6f}")
    print(f"Overall OOF CatBoost  AUC: {score_cat_all:.6f}")

    # --------------------
    # ENSEMBLE OPTIMIZATION
    # --------------------
    print("\nOptimizing ensembling weights...")
    def objective(weights):
        w1, w2, w3 = weights / np.sum(weights)
        blend = w1 * oof_preds_lgb + w2 * oof_preds_xgb + w3 * oof_preds_cat
        return -roc_auc_score(y, blend)

    # Grid search / Nelder-Mead optimization
    best_score = np.inf
    best_weights = None

    for w0 in [[1/3,1/3,1/3], [0.1,0.7,0.2], [0.2,0.6,0.2], [0.0,1.0,0.0], [0.1,0.8,0.1]]:
        res = minimize(objective, w0, method='Nelder-Mead')
        if res.fun < best_score:
            best_score = res.fun
            best_weights = res.x / np.sum(res.x)
    
    print(f"Optimal Blending Weights: LightGBM={best_weights[0]:.4f}, XGBoost={best_weights[1]:.4f}, CatBoost={best_weights[2]:.4f}")
    
    oof_blend = best_weights[0] * oof_preds_lgb + best_weights[1] * oof_preds_xgb + best_weights[2] * oof_preds_cat
    score_blend = roc_auc_score(y, oof_blend)
    print(f"Overall Blended OOF AUC: {score_blend:.6f}")

    # Save OOF predictions
    oof_df = pd.DataFrame({
        'id': train_ids,
        'oof_lgb': oof_preds_lgb,
        'oof_xgb': oof_preds_xgb,
        'oof_cat': oof_preds_cat,
        'oof_blend': oof_blend,
        'target': y
    })
    oof_df.to_csv('oof_predictions.csv', index=False)
    print("OOF predictions saved to oof_predictions.csv")

    # Generate final test predictions
    test_blend = best_weights[0] * test_preds_lgb + best_weights[1] * test_preds_xgb + best_weights[2] * test_preds_cat
    sub = pd.DataFrame({
        'id': test_ids,
        'PitNextLap': test_blend
    })
    sub.to_csv('submission.csv', index=False)
    print("Final submission file saved to submission.csv")
    
    print("\nSubmission Head:")
    print(sub.head(10))
    print("\nSubmission Summary:")
    print(sub.describe().T)

if __name__ == '__main__':
    main()
