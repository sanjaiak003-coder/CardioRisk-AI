"""
Heart Attack Risk Prediction - Master Machine Learning Pipeline
Unified execution for data preparation, feature engineering, model training, tuning, evaluation & export.
"""

import os
import sys
import json
import numpy as np
import pandas as pd

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import load_heart_data, split_data, FEATURE_NAMES
from src.preprocess import create_preprocessing_pipeline, save_preprocessor
from src.train_rf import train_random_forest, save_rf_model
from src.train_xgb import train_xgboost, save_xgb_model
from src.evaluate import evaluate_single_model, plot_and_save_visualizations


def run_training_pipeline(
    csv_path: str = None,
    output_dir: str = None,
    tune_hyperparameters: bool = True,
    random_state: int = 42
):
    """
    Executes end-to-end ML pipeline.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if output_dir is None:
        output_dir = os.path.join(base_dir, 'models')
    eval_dir = os.path.join(base_dir, 'static', 'eval_plots')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)

    print("="*60)
    print(">>> STARTING CLINICAL HEART ATTACK PREDICTION ML PIPELINE")
    print("="*60)

    # 1. Load Data
    print("\n[Step 1/6] Loading and validating clinical dataset...")
    df = load_heart_data(csv_path)
    print(f"Total clinical records: {len(df)}")
    print(f"Total features: {len(FEATURE_NAMES)}")

    # 2. Train / Test Split
    print("\n[Step 2/6] Performing Stratified Train/Test Partitioning...")
    X_train_raw, X_test_raw, y_train, y_test = split_data(df, test_size=0.20, random_state=random_state)

    # 3. Preprocessing & Feature Engineering
    print("\n[Step 3/6] Fitting Clinical Preprocessor & Feature Engineering...")
    preprocessor = create_preprocessing_pipeline(include_engineered=True)
    X_train = preprocessor.fit_transform(X_train_raw)
    X_test = preprocessor.transform(X_test_raw)

    # Extract engineered feature names
    fe_step = preprocessor.named_steps['feature_engineer']
    transformed_feature_names = fe_step.get_feature_names_out()
    print(f"Engineered features ({len(transformed_feature_names)}): {list(transformed_feature_names)}")

    # Save preprocessor
    save_preprocessor(preprocessor, os.path.join(output_dir, 'preprocessor.joblib'))

    # 4. Train Random Forest
    print("\n[Step 4/6] Training & Tuning Random Forest Classifier...")
    rf_results = train_random_forest(
        X_train, y_train,
        feature_names=transformed_feature_names,
        tune_hyperparameters=tune_hyperparameters,
        random_state=random_state
    )
    save_rf_model(rf_results['model'], os.path.join(output_dir, 'random_forest_model.joblib'))

    # 5. Train XGBoost
    print("\n[Step 5/6] Training & Tuning XGBoost Classifier...")
    xgb_results = train_xgboost(
        X_train, y_train,
        feature_names=transformed_feature_names,
        tune_hyperparameters=tune_hyperparameters,
        random_state=random_state
    )
    save_xgb_model(xgb_results['model'], os.path.join(output_dir, 'xgboost_model.joblib'))

    # 6. Comprehensive Evaluation
    print("\n[Step 6/6] Evaluating Models & Generating Clinical Diagnostics...")
    rf_eval = evaluate_single_model(rf_results['model'], X_test, y_test, model_name="Random Forest")
    xgb_eval = evaluate_single_model(xgb_results['model'], X_test, y_test, model_name="XGBoost")

    # Ensemble evaluation (Soft Voting)
    class EnsembleVotingClassifier:
        def __init__(self, m1, m2):
            self.m1 = m1
            self.m2 = m2
        def predict_proba(self, X):
            p1 = self.m1.predict_proba(X)
            p2 = self.m2.predict_proba(X)
            return 0.50 * p1 + 0.50 * p2
        def predict(self, X):
            probs = self.predict_proba(X)[:, 1]
            return (probs >= 0.50).astype(int)

    ensemble_model = EnsembleVotingClassifier(rf_results['model'], xgb_results['model'])
    ensemble_eval = evaluate_single_model(ensemble_model, X_test, y_test, model_name="Ensemble (RF + XGB)")

    # Save plots
    plot_and_save_visualizations(
        rf_eval=rf_eval,
        xgb_eval=xgb_eval,
        rf_feat_df=rf_results['feature_importance_df'],
        xgb_feat_df=xgb_results['feature_importance_df'],
        output_dir=eval_dir
    )

    # Save metadata
    metadata = {
        'timestamp': pd.Timestamp.now().isoformat(),
        'dataset_size': len(df),
        'train_samples': len(X_train_raw),
        'test_samples': len(X_test_raw),
        'features_original': FEATURE_NAMES,
        'features_engineered': list(transformed_feature_names),
        'metrics': {
            'random_forest': rf_eval,
            'xgboost': xgb_eval,
            'ensemble': ensemble_eval
        },
        'rf_top_features': rf_results['feature_importance_df'].to_dict(orient='records') if rf_results['feature_importance_df'] is not None else [],
        'xgb_top_features': xgb_results['feature_importance_df'].to_dict(orient='records') if xgb_results['feature_importance_df'] is not None else []
    }

    with open(os.path.join(output_dir, 'model_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    # Print summary table
    print("\n" + "="*70)
    print("CLINICAL PERFORMANCE BENCHMARK SUMMARY")
    print("="*70)
    summary_df = pd.DataFrame([
        {
            'Model': 'Random Forest',
            'Accuracy': f"{rf_eval['accuracy']:.2%}",
            'Sensitivity (Recall)': f"{rf_eval['sensitivity']:.2%}",
            'Specificity': f"{rf_eval['specificity']:.2%}",
            'Precision': f"{rf_eval['precision']:.2%}",
            'F1-Score': f"{rf_eval['f1_score']:.2%}",
            'ROC-AUC': f"{rf_eval['roc_auc']:.4f}",
            'Brier Score': f"{rf_eval['brier_score']:.4f}"
        },
        {
            'Model': 'XGBoost',
            'Accuracy': f"{xgb_eval['accuracy']:.2%}",
            'Sensitivity (Recall)': f"{xgb_eval['sensitivity']:.2%}",
            'Specificity': f"{xgb_eval['specificity']:.2%}",
            'Precision': f"{xgb_eval['precision']:.2%}",
            'F1-Score': f"{xgb_eval['f1_score']:.2%}",
            'ROC-AUC': f"{xgb_eval['roc_auc']:.4f}",
            'Brier Score': f"{xgb_eval['brier_score']:.4f}"
        },
        {
            'Model': 'Ensemble (RF + XGB)',
            'Accuracy': f"{ensemble_eval['accuracy']:.2%}",
            'Sensitivity (Recall)': f"{ensemble_eval['sensitivity']:.2%}",
            'Specificity': f"{ensemble_eval['specificity']:.2%}",
            'Precision': f"{ensemble_eval['precision']:.2%}",
            'F1-Score': f"{ensemble_eval['f1_score']:.2%}",
            'ROC-AUC': f"{ensemble_eval['roc_auc']:.4f}",
            'Brier Score': f"{ensemble_eval['brier_score']:.4f}"
        }
    ])
    print(summary_df.to_string(index=False))
    print("="*70)
    print(f"Pipeline successfully completed! Artifacts written to: {output_dir}")

    return metadata


if __name__ == '__main__':
    run_training_pipeline(tune_hyperparameters=True)
