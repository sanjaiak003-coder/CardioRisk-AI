"""
Heart Attack Risk Prediction - Random Forest Training Module
Configures, tunes, validates, and serializes the Random Forest Classifier.
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score


def train_random_forest(
    X_train, y_train,
    feature_names=None,
    tune_hyperparameters: bool = True,
    random_state: int = 42
) -> dict:
    """
    Trains and tunes a Random Forest Classifier for Heart Attack Risk Prediction.
    """
    print("\n" + "="*50, flush=True)
    print("[+] TRAINING RANDOM FOREST CLASSIFIER", flush=True)
    print("="*50, flush=True)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    if tune_hyperparameters:
        print("Performing Grid Search Cross-Validation for Random Forest...", flush=True)
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [5, 8, 12, None],
            'min_samples_split': [2, 4],
            'min_samples_leaf': [1, 2],
            'class_weight': ['balanced', None]
        }

        base_rf = RandomForestClassifier(random_state=random_state, oob_score=True)
        grid_search = GridSearchCV(
            estimator=base_rf,
            param_grid=param_grid,
            cv=cv,
            scoring='roc_auc',
            n_jobs=-1,
            verbose=0
        )
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        print(f"Best RF Parameters: {best_params}", flush=True)
        print(f"Best Cross-Val ROC-AUC: {grid_search.best_score_:.4f}", flush=True)
    else:
        best_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_split=4,
            min_samples_leaf=2,
            max_features='sqrt',
            class_weight='balanced',
            oob_score=True,
            random_state=random_state
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
        print("\nTop 5 Most Important Features (Random Forest):", flush=True)
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


def save_rf_model(model_obj, filepath: str):
    """Saves Random Forest model to disk."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model_obj, filepath)
    print(f"Random Forest model saved to: {filepath}", flush=True)
