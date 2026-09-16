import os
import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

def load_data(pt_path='elliptic_pyg_data.pt', use_updated=False):
    """
    Loads PyTorch Geometric graph data and extracts features, targets, and masks.
    Checks multiple possible paths for the file.
    If use_updated is True, loads 'elliptic_pyg_data_updated.pt' and updates masks.
    """
    if use_updated and pt_path == 'elliptic_pyg_data.pt':
        pt_path = 'elliptic_pyg_data_updated.pt'
        
    possible_paths = [
        pt_path,
        os.path.join('updated', pt_path),
        os.path.join('..', pt_path),
        os.path.join('dataset', pt_path),
        os.path.join('dataset', 'updated', pt_path),
        os.path.join('src', pt_path)
    ]
    
    data = None
    for path in possible_paths:
        if os.path.exists(path):
            print(f"Loading graph data from: {path}...")
            data = torch.load(path, weights_only=False)
            break
            
    if data is None:
        raise FileNotFoundError(f"Could not find {pt_path} in any of the expected locations: {possible_paths}")

    # Node features, labels, and time steps
    x = data.x.numpy()
    y = data.y.numpy()
    time_steps = data.time_step.numpy()

    # If updated dataset or requested, adjust masks to include pseudo-labeled nodes
    if use_updated or 'updated' in pt_path:
        train_mask_np = (time_steps <= 30) & (y != -1)
        val_mask_np = (time_steps >= 31) & (time_steps <= 35) & (y != -1)
        test_mask_np = data.test_mask.numpy() # Keep ground-truth test nodes for benchmark comparison
        
        data.train_mask = torch.tensor(train_mask_np, dtype=torch.bool)
        data.val_mask = torch.tensor(val_mask_np, dtype=torch.bool)
    else:
        train_mask_np = data.train_mask.numpy()
        val_mask_np = data.val_mask.numpy()
        test_mask_np = data.test_mask.numpy()

    # Split node features and labels into tabular matrices
    X_train, y_train = x[train_mask_np], y[train_mask_np]
    X_val, y_val = x[val_mask_np], y[val_mask_np]
    X_test, y_test = x[test_mask_np], y[test_mask_np]

    print(f"Graph Statistics: {x.shape[0]} nodes, {data.edge_index.shape[1]} edges, {x.shape[1]} features.")
    print(f"Train nodes: {X_train.shape[0]} (Licit: {np.sum(y_train==0)}, Illicit: {np.sum(y_train==1)})")
    print(f"Val/Cal nodes: {X_val.shape[0]} (Licit: {np.sum(y_val==0)}, Illicit: {np.sum(y_val==1)})")
    print(f"Test nodes (Ground Truth): {X_test.shape[0]} (Licit: {np.sum(y_test==0)}, Illicit: {np.sum(y_test==1)})")

    return data, X_train, y_train, X_val, y_val, X_test, y_test

def calculate_ece(probs, labels, n_bins=10):
    """
    Calculates the Expected Calibration Error (ECE) for binary classification.
    probs: Array of predicted probabilities for the positive class (class 1)
    labels: Binary true labels (0 or 1)
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n_samples = len(probs)
    
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        # Select samples in the current bin
        if i == n_bins - 1:
            in_bin = (probs >= bin_lower) & (probs <= bin_upper)
        else:
            in_bin = (probs >= bin_lower) & (probs < bin_upper)
            
        prop_in_bin = np.mean(in_bin)
        
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(labels[in_bin])
            avg_confidence_in_bin = np.mean(probs[in_bin])
            ece += prop_in_bin * np.abs(avg_confidence_in_bin - accuracy_in_bin)
            
    return ece

from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, precision_recall_curve, auc, accuracy_score, confusion_matrix, log_loss, brier_score_loss

def evaluate_model(y_true, y_pred, y_prob):
    """
    Computes Accuracy, Precision, Recall, F1 for the illicit class (1), AUC-ROC, PR-AUC, ECE, Log-Loss, Brier Score, and Confusion Matrix.
    """
    acc = accuracy_score(y_true, y_pred)
    
    # compute precision, recall, f1 for class 1 (illicit)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, beta=1.0, labels=[1], average='binary', zero_division=0
    )
    
    try:
        auc_roc = roc_auc_score(y_true, y_prob)
    except Exception as e:
        auc_roc = 0.0
        print(f"Warning: ROC AUC computation failed: {e}")
        
    try:
        pr_precision, pr_recall, _ = precision_recall_curve(y_true, y_prob)
        auc_pr = auc(pr_recall, pr_precision)
    except Exception as e:
        auc_pr = 0.0
        
    ece = calculate_ece(y_prob, y_true, n_bins=10)
    
    # Clipped probabilities for log loss stability
    eps = 1e-15
    y_prob_clipped = np.clip(y_prob, eps, 1 - eps)
    ll = log_loss(y_true, y_prob_clipped)
    brier = brier_score_loss(y_true, y_prob)
    
    # Confusion matrix
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

def plot_reliability_diagram(probs, labels, model_name, save_path, n_bins=10):
    """
    Plots a Reliability Diagram (calibration curve) for predictions.
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_centers = 0.5 * (bin_boundaries[:-1] + bin_boundaries[1:])
    
    bin_accuracies = []
    bin_confidences = []
    bin_counts = []
    
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        if i == n_bins - 1:
            in_bin = (probs >= bin_lower) & (probs <= bin_upper)
        else:
            in_bin = (probs >= bin_lower) & (probs < bin_upper)
            
        if np.sum(in_bin) > 0:
            bin_accuracies.append(np.mean(labels[in_bin]))
            bin_confidences.append(np.mean(probs[in_bin]))
            bin_counts.append(np.sum(in_bin))
        else:
            bin_accuracies.append(0.0)
            bin_confidences.append(bin_centers[i])
            bin_counts.append(0)
            
    ece = calculate_ece(probs, labels, n_bins=n_bins)
    
    fig, ax1 = plt.subplots(figsize=(7, 6))
    
    # Plot perfect calibration line
    ax1.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect Calibration")
    
    # Plot model calibration
    ax1.plot(bin_confidences, bin_accuracies, marker="s", color="darkblue", 
             label=f"{model_name} (ECE: {ece:.4f})")
    
    ax1.set_xlabel("Mean Predicted Probability", fontsize=11)
    ax1.set_ylabel("Empirical Fraud Ratio (Accuracy)", fontsize=11)
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])
    ax1.set_title(f"Reliability Diagram: {model_name}", fontsize=13, fontweight='bold')
    ax1.legend(loc="upper left")
    ax1.grid(True, linestyle=":", alpha=0.6)
    
    # Inset histogram for sample distribution per bin
    ax2 = fig.add_axes([0.62, 0.18, 0.25, 0.18])
    ax2.bar(bin_centers, bin_counts, width=1.0/n_bins, color="lightblue", edgecolor="black", alpha=0.7)
    ax2.set_title("Bin Counts", fontsize=9)
    ax2.set_xticks([0.0, 0.5, 1.0])
    ax2.tick_params(axis='both', which='major', labelsize=8)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Reliability diagram saved at: {save_path}")
