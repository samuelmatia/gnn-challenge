"""
scoring_script.py

Comprehensive evaluation script for GNN predictions.
Computes 3 challenging metrics: Macro F1, MCC, and Balanced Accuracy.
"""

import os
import pandas as pd
import numpy as np
from sklearn.metrics import (
    f1_score,
    matthews_corrcoef,
    balanced_accuracy_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    cohen_kappa_score,
    accuracy_score,
    precision_score,
    recall_score,
    classification_report
)

def score_predictions(y_true, y_pred, y_proba=None):
    """
    Evaluate predictions with multiple metrics.
    
    Args:
        y_true: Ground truth labels (numpy array or list)
        y_pred: Hard predictions (0 or 1)
        y_proba: Soft predictions/probabilities (optional, for ROC-AUC, AP)
    
    Returns:
        Dictionary of metrics
    """
    
    metrics = {}
    
    # ===== BASIC METRICS =====
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    
    # ===== PRECISION & RECALL =====
    metrics['precision'] = precision_score(y_true, y_pred, average='binary', zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, average='binary', zero_division=0)
    
    # ===== DIFFICULT METRIC #1: MACRO F1-SCORE =====
    # Equal weight per class (penalizes minority class errors)
    metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['f1_binary'] = f1_score(y_true, y_pred, average='binary', zero_division=0)
    
    # ===== DIFFICULT METRIC #2: MATTHEW'S CORRELATION COEFFICIENT =====
    # Strictest metric: only positive if beats random chance
    metrics['mcc'] = matthews_corrcoef(y_true, y_pred)
    
    # ===== DIFFICULT METRIC #3: BALANCED ACCURACY =====
    # Average of per-class recalls (accounts for class imbalance)
    metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)
    
    # ===== CONFUSION MATRIX =====
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics['confusion_matrix'] = {'TP': int(tp), 'FP': int(fp), 'FN': int(fn), 'TN': int(tn)}
    
    # ===== SENSITIVITY & SPECIFICITY =====
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    metrics['sensitivity'] = sensitivity
    metrics['specificity'] = specificity
    metrics['youden_index'] = sensitivity + specificity - 1
    
    # ===== COHEN'S KAPPA =====
    # Agreement beyond chance
    metrics['cohen_kappa'] = cohen_kappa_score(y_true, y_pred)
    
    # ===== PROBABILISTIC METRICS (if y_proba provided) =====
    if y_proba is not None:
        y_proba_pos = y_proba[:, 1] if y_proba.shape[1] == 2 else y_proba.flatten()
        metrics['roc_auc'] = roc_auc_score(y_true, y_proba_pos)
        metrics['average_precision'] = average_precision_score(y_true, y_proba_pos)
        metrics['brier_score'] = np.mean((y_proba_pos - y_true) ** 2)
    
    # ===== CLASS DISTRIBUTION =====
    unique, counts = np.unique(y_true, return_counts=True)
    metrics['class_distribution'] = {int(k): int(v) for k, v in zip(unique, counts)}
    
    return metrics

def print_metrics(metrics, name="Evaluation Results"):
    """Pretty-print metrics."""
    print("\n" + "="*60)
    print(f"  {name}")
    print("="*60)
    
    print(f"\n📊 Basic Metrics:")
    print(f"  Accuracy:           {metrics['accuracy']:.4f}")
    print(f"  Precision:          {metrics['precision']:.4f}")
    print(f"  Recall:             {metrics['recall']:.4f}")
    
    print(f"\n🎯 Difficult Metrics (Recommended):")
    print(f"  Macro F1-Score:     {metrics['f1_macro']:.4f}    ← (equal weight per class)")
    print(f"  Balanced Accuracy:  {metrics['balanced_accuracy']:.4f}  ← (avg per-class recalls)")
    print(f"  MCC:                {metrics['mcc']:.4f}          ← (strictest metric)")
    
    print(f"\n📈 Other F1 Variants:")
    print(f"  F1 Binary:          {metrics['f1_binary']:.4f}")
    print(f"  F1 Weighted:        {metrics['f1_weighted']:.4f}")
    
    print(f"\n🔍 Clinical Metrics:")
    print(f"  Sensitivity (TPR):  {metrics['sensitivity']:.4f}")
    print(f"  Specificity (TNR):  {metrics['specificity']:.4f}")
    print(f"  Youden's Index:     {metrics['youden_index']:.4f}")
    print(f"  Cohen's Kappa:      {metrics['cohen_kappa']:.4f}")
    
    if 'roc_auc' in metrics:
        print(f"\n📉 Probabilistic Metrics:")
        print(f"  ROC-AUC:            {metrics['roc_auc']:.4f}")
        print(f"  Average Precision:  {metrics['average_precision']:.4f}")
        print(f"  Brier Score:        {metrics['brier_score']:.4f}")
    
    print(f"\n🔗 Confusion Matrix:")
    cm = metrics['confusion_matrix']
    print(f"  TN={cm['TN']:3d}  FP={cm['FP']:3d}")
    print(f"  FN={cm['FN']:3d}  TP={cm['TP']:3d}")
    
    print(f"\n📊 Class Distribution (y_true):")
    for cls, count in metrics['class_distribution'].items():
        print(f"  Class {cls}: {count} samples")
    
    print("="*60 + "\n")

def evaluate_submission(submission_path, ground_truth_path=None):
    """
    Evaluate a submission CSV against ground truth.
    
    Args:
        submission_path: Path to predictions CSV
        ground_truth_path: Path to ground truth CSV (if available)
    """
    # Load submission
    submission = pd.read_csv(submission_path)

    # Normalize expected columns
    if "id" in submission.columns and "y_pred" in submission.columns:
        submission = submission.rename(columns={"id": "node_id", "y_pred": "target"})
    elif "node_id" in submission.columns and "target" in submission.columns:
        pass
    else:
        print("❌ Submission must have either ['id','y_pred'] or ['node_id','target'] columns")
        return None
    
    # Check if ground truth available
    if ground_truth_path is None:
        print("⚠️  No ground truth provided. Cannot evaluate.")
        print(f"   Submission has {len(submission)} predictions")
        print(f"   Classes: {submission['target'].unique()}")
        return None
    
    # Load ground truth
    ground_truth = pd.read_csv(ground_truth_path)

    # Normalize ground truth columns/index
    if "node_id" not in ground_truth.columns:
        if "id" in ground_truth.columns:
            ground_truth = ground_truth.rename(columns={"id": "node_id"})
        elif ground_truth.index.name in {"node_id", "id"}:
            ground_truth = ground_truth.reset_index()
            if "id" in ground_truth.columns and "node_id" not in ground_truth.columns:
                ground_truth = ground_truth.rename(columns={"id": "node_id"})
        else:
            first_col = ground_truth.columns[0] if len(ground_truth.columns) > 0 else None
            if first_col is not None and str(first_col).startswith("Unnamed"):
                ground_truth = ground_truth.rename(columns={first_col: "node_id"})
            elif first_col is not None and isinstance(first_col, (int, np.integer)):
                # Likely headerless CSV; assume first column is node_id
                ground_truth = ground_truth.rename(columns={first_col: "node_id"})

    if "target" not in ground_truth.columns:
        if "y_true" in ground_truth.columns:
            ground_truth = ground_truth.rename(columns={"y_true": "target"})
        elif "label" in ground_truth.columns:
            ground_truth = ground_truth.rename(columns={"label": "target"})
        elif "disease_labels" in ground_truth.columns:
            ground_truth = ground_truth.rename(columns={"disease_labels": "target"})
        elif len(ground_truth.columns) >= 2 and "node_id" in ground_truth.columns:
            # If only node_id was named, assume the next column is target
            remaining = [c for c in ground_truth.columns if c != "node_id"]
            if remaining:
                ground_truth = ground_truth.rename(columns={remaining[0]: "target"})

    if "node_id" not in ground_truth.columns or "target" not in ground_truth.columns:
        # Final fallback: treat index as node_id and first column as target
        if "target" not in ground_truth.columns and len(ground_truth.columns) == 1:
            ground_truth = ground_truth.reset_index().rename(columns={"index": "node_id", ground_truth.columns[0]: "target"})
        if "node_id" in ground_truth.columns and "target" in ground_truth.columns:
            pass
        else:
            print("❌ Ground truth must have a node id column and a target/label column")
            print(f"   Columns found: {list(ground_truth.columns)}")
            return None
    
    # Ensure node_id types align
    ground_truth["node_id"] = ground_truth["node_id"].astype(str)
    submission["node_id"] = submission["node_id"].astype(str)

    # Merge on node_id
    merged = pd.merge(ground_truth, submission, on='node_id', suffixes=('_true', '_pred'))
    
    if len(merged) == 0:
        print("❌ No matching node_ids between submission and ground truth")
        return None
    
    y_true = merged['target_true'].values
    y_pred_raw = merged['target_pred'].values

    # If probabilities, threshold at 0.5
    if y_pred_raw.dtype.kind in {"f", "c"}:
        y_pred = (y_pred_raw >= 0.5).astype(int)
    else:
        y_pred = y_pred_raw.astype(int)
    
    # Evaluate
    metrics = score_predictions(y_true, y_pred)
    print_metrics(metrics, name="Submission Evaluation")
    
    return metrics

if __name__ == "__main__":
    import sys
    
    print("\n🎯 GNN Challenge Scoring Script")
    print("="*60)
    
    # Example usage
    submission_file = "submissions/inbox/example_team/example_run/predictions.csv"
    ground_truth_file = "data/test_labels.csv"  # True labels for test set
    
    if len(sys.argv) > 1:
        submission_file = sys.argv[1]
    
    if len(sys.argv) > 2:
        ground_truth_file = sys.argv[2]
    
    if os.path.exists(submission_file):
        print(f"\n📂 Submission: {submission_file}")
        print(f"   Ground truth: {ground_truth_file}")
        metrics = evaluate_submission(submission_file, ground_truth_file)
    else:
        print(f"❌ Submission file not found: {submission_file}")
        print(f"\nUsage: python scoring_script.py <submission_file.csv> [ground_truth_file.csv]")
        print(f"\nExample:")
        print(f"  python scoring_script.py submissions/inbox/my_team/run_001/predictions.csv")
