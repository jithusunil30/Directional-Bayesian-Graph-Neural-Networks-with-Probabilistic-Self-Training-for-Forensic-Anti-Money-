import os
import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_recall_fscore_support, roc_auc_score,
    precision_recall_curve, auc, accuracy_score,
    confusion_matrix, log_loss, brier_score_loss,
    roc_curve
)

def load_elliptic_data():
    """
    Loads PyTorch Geometric graph data for the Elliptic Bitcoin dataset.
    Extracts features, labels, time steps, and temporal splits.
    """
    possible_paths = [
        'elliptic_pyg_data.pt',
        os.path.join('..', 'dataset', 'elliptic_pyg_data.pt'),
        os.path.join('dataset', 'elliptic_pyg_data.pt'),
        os.path.join('..', 'elliptic_pyg_data.pt'),
        os.path.join('..', 'dataset', 'updated', 'elliptic_pyg_data.pt')
    ]
    
    data = None
    loaded_path = None
    for path in possible_paths:
        if os.path.exists(path):
            print(f"[Data] Loading graph dataset from: {path}")
            data = torch.load(path, weights_only=False)
            loaded_path = path
            break
            
    if data is None:
        raise FileNotFoundError(f"Could not find elliptic_pyg_data.pt in: {possible_paths}")

    x = data.x.numpy()
    y = data.y.numpy()
    time_steps = data.time_step.numpy()

    # Temporal split masks:
    # Train: Steps 1 to 30 (labeled nodes only)
    # Val/Calibration: Steps 31 to 34 (labeled nodes only)
    # Test: Steps 35 to 49 (labeled nodes only)
    train_mask_np = (time_steps <= 30) & (y != -1)
    val_mask_np = (time_steps >= 31) & (time_steps <= 34) & (y != -1)
    test_mask_np = (time_steps >= 35) & (time_steps <= 49) & (y != -1)

    data.train_mask = torch.tensor(train_mask_np, dtype=torch.bool)
    data.val_mask = torch.tensor(val_mask_np, dtype=torch.bool)
    data.test_mask = torch.tensor(test_mask_np, dtype=torch.bool)

    X_train, y_train = x[train_mask_np], y[train_mask_np]
    X_val, y_val = x[val_mask_np], y[val_mask_np]
    X_test, y_test = x[test_mask_np], y[test_mask_np]

    num_licit_tr = int(np.sum(y_train == 0))
    num_illicit_tr = int(np.sum(y_train == 1))
    pos_weight = float(num_licit_tr / max(num_illicit_tr, 1))

    print(f"=== Dataset Loaded Successfully ===")
    print(f"Total Nodes: {x.shape[0]:,}, Edges: {data.edge_index.shape[1]:,}, Features: {x.shape[1]}")
    print(f"Train Set (Steps 1-30): {X_train.shape[0]:,} nodes (Licit: {num_licit_tr:,}, Illicit: {num_illicit_tr:,})")
    print(f"Val Set   (Steps 31-34): {X_val.shape[0]:,} nodes (Licit: {np.sum(y_val==0):,}, Illicit: {np.sum(y_val==1):,})")
    print(f"Test Set  (Steps 35-49): {X_test.shape[0]:,} nodes (Licit: {np.sum(y_test==0):,}, Illicit: {np.sum(y_test==1):,})")
    print(f"Computed Class Imbalance Ratio (pos_weight): {pos_weight:.4f}\n")

    return data, X_train, y_train, X_val, y_val, X_test, y_test, pos_weight

def calculate_ece(probs, labels, n_bins=10):
    """Calculates Expected Calibration Error (ECE) for binary classification."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        if i == n_bins - 1:
            in_bin = (probs >= bin_lower) & (probs <= bin_upper)
        else:
            in_bin = (probs >= bin_lower) & (probs < bin_upper)
            
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(labels[in_bin])
            avg_confidence_in_bin = np.mean(probs[in_bin])
            ece += prop_in_bin * np.abs(avg_confidence_in_bin - accuracy_in_bin)
    return float(ece)

def evaluate_predictions(y_true, y_pred, y_prob):
    """Computes full evaluation metrics on test predictions."""
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[1], average='binary', zero_division=0
    )
    
    try:
        auc_roc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc_roc = 0.0
        
    try:
        pr_precision, pr_recall, _ = precision_recall_curve(y_true, y_prob)
        auc_pr = auc(pr_recall, pr_precision)
    except Exception:
        auc_pr = 0.0
        
    ece = calculate_ece(y_prob, y_true, n_bins=10)
    
    eps = 1e-15
    y_prob_clipped = np.clip(y_prob, eps, 1.0 - eps)
    ll = log_loss(y_true, y_prob_clipped)
    brier = brier_score_loss(y_true, y_prob)
    
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    
    return {
        'Accuracy': float(acc),
        'Precision': float(precision),
        'Recall': float(recall),
        'F1-Score': float(f1),
        'AUC-ROC': float(auc_roc),
        'AUC-PR': float(auc_pr),
        'ECE': float(ece),
        'Log-Loss': float(ll),
        'Brier-Score': float(brier),
        'TN': int(tn),
        'FP': int(fp),
        'FN': int(fn),
        'TP': int(tp)
    }

def plot_calibration_curve(probs, labels, model_name, save_path, n_bins=10):
    """Generates and saves a reliability diagram for model probability calibration."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_centers = 0.5 * (bin_boundaries[:-1] + bin_boundaries[1:])
    accuracies = []
    confidences = []
    counts = []
    
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        if i == n_bins - 1:
            in_bin = (probs >= bin_lower) & (probs <= bin_upper)
        else:
            in_bin = (probs >= bin_lower) & (probs < bin_upper)
            
        count = np.sum(in_bin)
        counts.append(count)
        if count > 0:
            accuracies.append(np.mean(labels[in_bin]))
            confidences.append(np.mean(probs[in_bin]))
        else:
            accuracies.append(0.0)
            confidences.append(bin_centers[i])
            
    ece = calculate_ece(probs, labels, n_bins=n_bins)
    
    plt.figure(figsize=(7, 6), dpi=300)
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect Calibration')
    plt.plot(confidences, accuracies, marker='o', linewidth=2.2, color='#1F77B4', label=f'{model_name} (ECE={ece:.4f})')
    plt.bar(bin_centers, accuracies, width=1.0/n_bins, alpha=0.18, color='#1F77B4', edgecolor='#1F77B4')
    
    plt.title(f'Reliability Diagram: {model_name}', fontsize=13, fontweight='bold', pad=12)
    plt.xlabel('Mean Predicted Probability (Confidence)', fontsize=11)
    plt.ylabel('Fraction of Positives (Empirical Accuracy)', fontsize=11)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='upper left', frameon=True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
