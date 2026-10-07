"""
Heart Attack Risk Prediction - Model Evaluation & Visual Analytics Module
Computes clinical performance metrics, generates ROC/PR curves, and saves comparison artifacts.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg') # Headless rendering for backend / web
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve, auc,
    confusion_matrix, classification_report, brier_score_loss,
    balanced_accuracy_score
)


def evaluate_single_model(model, X_test, y_test, model_name="Model") -> dict:
    """
    Computes comprehensive clinical metrics for a binary classifier.
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred

    # Confusion matrix elements
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0 # Recall

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_prob)
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    brier = brier_score_loss(y_test, y_prob)

    fpr, tpr, roc_thresh = roc_curve(y_test, y_prob)
    precision_curve, recall_curve, pr_thresh = precision_recall_curve(y_test, y_prob)
    pr_auc = auc(recall_curve, precision_curve)

    metrics = {
        'model_name': model_name,
        'accuracy': float(acc),
        'balanced_accuracy': float(balanced_acc),
        'precision': float(prec),
        'recall': float(rec),
        'sensitivity': float(sensitivity),
        'specificity': float(specificity),
        'f1_score': float(f1),
        'roc_auc': float(roc_auc),
        'pr_auc': float(pr_auc),
        'brier_score': float(brier),
        'confusion_matrix': {
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn),
            'tp': int(tp)
        },
        'roc_curve': {
            'fpr': [float(x) for x in fpr],
            'tpr': [float(x) for x in tpr],
            'thresholds': [float(x) for x in roc_thresh]
        },
        'pr_curve': {
            'precision': [float(x) for x in precision_curve],
            'recall': [float(x) for x in recall_curve]
        }
    }

    return metrics


def plot_and_save_visualizations(
    rf_eval: dict,
    xgb_eval: dict,
    rf_feat_df: pd.DataFrame,
    xgb_feat_df: pd.DataFrame,
    output_dir: str
):
    """
    Generates high-resolution publication-quality clinical comparison charts.
    """
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams.update({'font.sans-serif': 'DejaVu Sans', 'font.size': 11})

    # 1. ROC Curves Comparison
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    ax.plot(
        rf_eval['roc_curve']['fpr'], rf_eval['roc_curve']['tpr'],
        color='#10B981', lw=2.5,
        label=f"Random Forest (AUC = {rf_eval['roc_auc']:.3f})"
    )
    ax.plot(
        xgb_eval['roc_curve']['fpr'], xgb_eval['roc_curve']['tpr'],
        color='#6366F1', lw=2.5, linestyle='-',
        label=f"XGBoost (AUC = {xgb_eval['roc_auc']:.3f})"
    )
    ax.plot([0, 1], [0, 1], color='#9CA3AF', lw=1.5, linestyle='--', label='Random Chance (AUC = 0.500)')
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.05])
    ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Positive Rate (Sensitivity / Recall)', fontsize=12, fontweight='bold')
    ax.set_title('Receiver Operating Characteristic (ROC) Comparison', fontsize=14, fontweight='bold', pad=15)
    ax.legend(loc='lower right', frameon=True, facecolor='white', framealpha=0.9)
    plt.tight_layout()
    roc_path = os.path.join(output_dir, 'roc_comparison.png')
    plt.savefig(roc_path)
    plt.close()

    # 2. Confusion Matrices Comparison
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=300)
    
    rf_cm = np.array([
        [rf_eval['confusion_matrix']['tn'], rf_eval['confusion_matrix']['fp']],
        [rf_eval['confusion_matrix']['fn'], rf_eval['confusion_matrix']['tp']]
    ])
    xgb_cm = np.array([
        [xgb_eval['confusion_matrix']['tn'], xgb_eval['confusion_matrix']['fp']],
        [xgb_eval['confusion_matrix']['fn'], xgb_eval['confusion_matrix']['tp']]
    ])

    sns.heatmap(rf_cm, annot=True, fmt='d', cmap='Greens', cbar=False, ax=axes[0],
                xticklabels=['Low Risk (0)', 'High Risk (1)'],
                yticklabels=['Low Risk (0)', 'High Risk (1)'],
                annot_kws={'size': 14, 'weight': 'bold'})
    axes[0].set_title(f"Random Forest Confusion Matrix\n(Accuracy: {rf_eval['accuracy']:.1%})", fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Predicted Label', fontweight='bold')
    axes[0].set_ylabel('True Clinical Label', fontweight='bold')

    sns.heatmap(xgb_cm, annot=True, fmt='d', cmap='Purples', cbar=False, ax=axes[1],
                xticklabels=['Low Risk (0)', 'High Risk (1)'],
                yticklabels=['Low Risk (0)', 'High Risk (1)'],
                annot_kws={'size': 14, 'weight': 'bold'})
    axes[1].set_title(f"XGBoost Confusion Matrix\n(Accuracy: {xgb_eval['accuracy']:.1%})", fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Predicted Label', fontweight='bold')
    axes[1].set_ylabel('True Clinical Label', fontweight='bold')

    plt.tight_layout()
    cm_path = os.path.join(output_dir, 'confusion_matrices.png')
    plt.savefig(cm_path)
    plt.close()

    # 3. Feature Importance Comparison
    if rf_feat_df is not None and xgb_feat_df is not None:
        fig, axes = plt.subplots(1, 2, figsize=(15, 6), dpi=300)
        
        top_rf = rf_feat_df.head(10).sort_values(by='importance', ascending=True)
        top_xgb = xgb_feat_df.head(10).sort_values(by='importance', ascending=True)

        axes[0].barh(top_rf['feature'], top_rf['importance'], color='#10B981', edgecolor='#047857')
        axes[0].set_title('Top 10 Clinical Features (Random Forest)', fontsize=13, fontweight='bold')
        axes[0].set_xlabel('Gini Impurity Importance', fontweight='bold')

        axes[1].barh(top_xgb['feature'], top_xgb['importance'], color='#6366F1', edgecolor='#4338CA')
        axes[1].set_title('Top 10 Clinical Features (XGBoost)', fontsize=13, fontweight='bold')
        axes[1].set_xlabel('Feature Weight / Gain Importance', fontweight='bold')

        plt.tight_layout()
        feat_path = os.path.join(output_dir, 'feature_importance.png')
        plt.savefig(feat_path)
        plt.close()

    # 4. Metrics Grouped Bar Chart
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)
    metric_keys = ['accuracy', 'precision', 'recall', 'specificity', 'f1_score', 'roc_auc']
    metric_labels = ['Accuracy', 'Precision', 'Sensitivity\n(Recall)', 'Specificity', 'F1-Score', 'ROC-AUC']
    
    rf_scores = [rf_eval[k] for k in metric_keys]
    xgb_scores = [xgb_eval[k] for k in metric_keys]

    x = np.arange(len(metric_keys))
    width = 0.35

    rects1 = ax.bar(x - width/2, rf_scores, width, label='Random Forest', color='#10B981', alpha=0.9)
    rects2 = ax.bar(x + width/2, xgb_scores, width, label='XGBoost', color='#6366F1', alpha=0.9)

    ax.set_ylabel('Score (0.0 - 1.0)', fontsize=12, fontweight='bold')
    ax.set_title('Clinical Model Benchmark Comparison (RF vs XGBoost)', fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontweight='bold')
    ax.set_ylim([0.0, 1.15])
    ax.legend(frameon=True, facecolor='white')

    for rects in [rects1, rects2]:
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    metrics_path = os.path.join(output_dir, 'metrics_comparison.png')
    plt.savefig(metrics_path)
    plt.close()

    print(f"Visualizations saved to: {output_dir}")
