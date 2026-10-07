"""
Heart Attack Risk Prediction - XGBoost Training Module
Configures, tunes, validates, and serializes the XGBoost Classifier.
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score


def train_xgboost(
    X_train, y_train,
    feature_names=None,
    tune_hyperparameters: bool = True,
    random_state: int = 42
) -> dict:
    """
    Trains and tunes an XGBoost Classifier for Heart Attack Risk Prediction.
    """
    print("\n" + "="*50, flush=True)
    print("[+] TRAINING XGBOOST CLASSIFIER", flush=True)
    print("="*50, flush=True)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    # Calculate class balance ratio for scale_pos_weight
    neg_count = np.sum(y_train == 0)
    pos_count = np.sum(y_train == 1)
    scale_pos_weight = neg_count / max(1, pos_count)

    if tune_hyperparameters:
        print("Performing Grid Search Cross-Validation for XGBoost...", flush=True)
        param_grid = {
            'n_estimators': [100, 150],
            'max_depth': [3, 4, 6],
            'learning_rate': [0.05, 0.1],
            'subsample': [0.85, 1.0],
            'colsample_bytree': [0.85, 1.0],
            'reg_alpha': [0.0, 0.5]
        }

        base_xgb = XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            eval_metric='logloss',
            random_state=random_state,
            n_jobs=-1
        )

        grid_search = GridSearchCV(
            estimator=base_xgb,
            param_grid=param_grid,
            cv=cv,
            scoring='roc_auc',
            n_jobs=-1,
            verbose=0
        )
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        print(f"Best XGBoost Parameters: {best_params}", flush=True)
        print(f"Best Cross-Val ROC-AUC: {grid_search.best_score_:.4f}", flush=True)
    else:
        best_model = XGBClassifier(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            scale_pos_weight=scale_pos_weight,
            eval_metric='logloss',
            random_state=random_state,
            n_jobs=-1
        )
        best_model.fit(X_train, y_train)
        best_params = best_model.get_params()

    # Cross-validation performance check
    cv_auc_scores = cross_val_score(best_model, X_train, y_train, cv=cv, scoring='roc_auc')
    cv_acc_scores = cross_val_score(best_model, X_train, y_train, cv=cv, scoring='accuracy')

    # Feature Importance
    importances = best_model.feature_importances_
    feat_df = None
    if feature_names is not None and len(feature_names) == len(importances):
        feat_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values(by='importance', ascending=False).reset_index(drop=True)
        print("\nTop 5 Most Important Features (XGBoost):", flush=True)
        for i, row in feat_df.head(5).iterrows():
            print(f"  {i+1}. {row['feature']:<25}: {row['importance']:.4f}", flush=True)

    return {
        'model': best_model,
        'params': best_params,
        'cv_auc_mean': cv_auc_scores.mean(),
        'cv_auc_std': cv_auc_scores.std(),
        'cv_acc_mean': cv_acc_scores.mean(),
        'feature_importance_df': feat_df
    }


def save_xgb_model(model_obj, filepath: str):
    """Saves XGBoost model to disk."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model_obj, filepath)
    print(f"XGBoost model saved to: {filepath}", flush=True)
