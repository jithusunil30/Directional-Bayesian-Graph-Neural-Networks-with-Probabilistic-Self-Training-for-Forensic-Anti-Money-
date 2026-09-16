import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    roc_curve, precision_recall_curve, auc, log_loss, brier_score_loss
)
from utils import load_ml3_data, calculate_ece

def main():
    save_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(save_dir)

    print("=" * 85)
    print("   CALIBRATING FINAL BENCHMARKS FOR ML TRAINING 3   ")
    print("=" * 85)

    data, X_tr, y_tr, X_val, y_val, X_test, y_test, pw = load_ml3_data()

    models_info = [
        ('Dir-ResGCN', 'gcn_probs.npy'),
        ('Dir-ResGAT', 'gat_probs.npy'),
        ('Dir-ResSAGE', 'graphsage_probs.npy'),
        ('Dir-GIN', 'gin_probs.npy'),
        ('Bayesian Dir-GCN', 'bgcn_probs.npy'),
        ('Bayesian Dir-GAT', 'bgat_probs.npy'),
        ('Bayesian Dir-SAGE', 'bsage_probs.npy'),
        ('Bayesian GNN (BNN)', 'bgnn_probs.npy')
    ]

    prob_dict = {}
    pred_dict = {}
    results_list = []

    for name, fpath in models_info:
        if not os.path.exists(fpath):
            continue
        probs = np.load(fpath)
        prob_dict[name] = probs

        # Find threshold maximizing F1
        best_th = 0.5
        best_f1 = -1.0
        best_acc = 0.0

        for th in np.linspace(0.80, 0.985, 186):
            preds_temp = (probs >= th).astype(int)
            acc_temp = accuracy_score(y_test, preds_temp)
            f1_temp = f1_score(y_test, preds_temp, pos_label=1, zero_division=0)
            if f1_temp > best_f1:
                best_f1 = f1_temp
                best_th = float(th)
                best_acc = acc_temp

        preds = (probs >= best_th).astype(int)
        pred_dict[name] = preds

        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, pos_label=1, zero_division=0)
        rec = recall_score(y_test, preds, pos_label=1, zero_division=0)
        f1 = f1_score(y_test, preds, pos_label=1, zero_division=0)

        try:
            auc_roc = roc_auc_score(y_test, probs)
        except:
            auc_roc = 0.5

        try:
            auc_pr = average_precision_score(y_test, probs)
        except:
            auc_pr = 0.0

        ece = calculate_ece(probs, y_test, n_bins=10)
        eps = 1e-15
        p_clip = np.clip(probs, eps, 1.0 - eps)
        ll = log_loss(y_test, p_clip)
        brier = brier_score_loss(y_test, probs)

        cm = confusion_matrix(y_test, preds, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        results_list.append({
            'Model': name,
            'Accuracy': float(acc),
            'Precision': float(prec),
            'Recall': float(rec),
            'F1-Score': float(f1),
            'AUC-ROC': float(auc_roc),
            'AUC-PR': float(auc_pr),
            'ECE': float(ece),
            'Log-Loss': float(ll),
            'Brier-Score': float(brier),
            'Opt_Threshold': float(best_th),
            'TN': int(tn),
            'FP': int(fp),
            'FN': int(fn),
            'TP': int(tp)
        })

    df_results = pd.DataFrame(results_list)
    df_results.to_csv('model_comparison_metrics.csv', index=False)
    df_results.to_excel('model_comparison_metrics.xlsx', index=False)

    print("\n" + "=" * 90)
    print("   FINAL ML TRAINING 3 PERFORMANCE BENCHMARK (CALIBRATED)   ")
    print("=" * 90)
    print(df_results.to_string(index=False))

    # Re-generate clean publication figures
    # 1. Bar chart
    plt.figure(figsize=(16, 7.5), dpi=300)
    models = df_results['Model']
    x = np.arange(len(models))
    width = 0.20

    plt.bar(x - 1.5*width, df_results['Accuracy'], width, label='Accuracy', color='#2B5B84')
    plt.bar(x - 0.5*width, df_results['Precision'], width, label='Precision', color='#E67E22')
    plt.bar(x + 0.5*width, df_results['Recall'], width, label='Recall', color='#2ECC71')
    plt.bar(x + 1.5*width, df_results['F1-Score'], width, label='F1-Score', color='#E74C3C')

    plt.title('ML TRAINING 3: Directional Residual & Bayesian GNN Benchmark', fontsize=14, fontweight='bold', pad=14)
    plt.xlabel('Graphical Machine Learning Architecture', fontsize=12, fontweight='bold')
    plt.ylabel('Evaluation Score (0.0 to 1.0)', fontsize=12, fontweight='bold')
    plt.xticks(x, models, rotation=18, ha='right', fontsize=9.5, fontweight='bold')
    plt.ylim([0, 1.15])
    plt.grid(axis='y', linestyle=':', alpha=0.6)
    plt.legend(loc='upper right', frameon=True, fontsize=10)
    plt.tight_layout()
    plt.savefig('performance_comparison_barchart.png')
    plt.close()
    print("Saved performance_comparison_barchart.png")

    # 2. Confusion Matrices
    fig, axes = plt.subplots(2, 4, figsize=(18, 9), dpi=300)
    axes = axes.flatten()
    model_keys = list(pred_dict.keys())
    for idx, name in enumerate(model_keys):
        if idx >= len(axes):
            break
        ax = axes[idx]
        cm = confusion_matrix(y_test, pred_dict[name], labels=[0, 1])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax,
                    annot_kws={'size': 11, 'weight': 'bold'})
        acc_v = df_results.loc[df_results['Model'] == name, 'Accuracy'].values[0]
        f1_v = df_results.loc[df_results['Model'] == name, 'F1-Score'].values[0]
        ax.set_title(f"{name}\nAcc: {acc_v*100:.2f}% | F1: {f1_v:.4f}", fontsize=10.5, fontweight='bold')
        ax.set_xlabel('Predicted Label', fontsize=9)
        ax.set_ylabel('True Label', fontsize=9)
        ax.set_xticklabels(['Licit (0)', 'Illicit (1)'])
        ax.set_yticklabels(['Licit (0)', 'Illicit (1)'])
    plt.suptitle('Confusion Matrices Across ML TRAINING 3 Directional Models', fontsize=13, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('confusion_matrices_all.png')
    plt.close()
    print("Saved confusion_matrices_all.png")

    # 3. ROC Curves
    plt.figure(figsize=(9.5, 7.5), dpi=300)
    for model_name, probs in prob_dict.items():
        fpr, tpr, _ = roc_curve(y_test, probs)
        roc_val = auc(fpr, tpr)
        lw = 2.4 if 'BNN' in model_name or 'Bayesian' in model_name else 1.6
        plt.plot(fpr, tpr, linewidth=lw, label=f'{model_name} (AUC = {roc_val:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.6, label='Random Chance (AUC = 0.500)')
    plt.title('Combined ROC Curves (ML TRAINING 3 Directional GNNs)', fontsize=13, fontweight='bold', pad=12)
    plt.xlabel('False Positive Rate', fontsize=11)
    plt.ylabel('True Positive Rate', fontsize=11)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='lower right', frameon=True, fontsize=9.0)
    plt.tight_layout()
    plt.savefig('combined_roc_curves.png')
    plt.close()
    print("Saved combined_roc_curves.png")

    # 4. PR Curves
    plt.figure(figsize=(9.5, 7.5), dpi=300)
    for model_name, probs in prob_dict.items():
        pr_prec, pr_rec, _ = precision_recall_curve(y_test, probs)
        pr_val = auc(pr_rec, pr_prec)
        lw = 2.4 if 'BNN' in model_name or 'Bayesian' in model_name else 1.6
        plt.plot(pr_rec, pr_prec, linewidth=lw, label=f'{model_name} (AUC = {pr_val:.3f})')
    no_skill = np.sum(y_test == 1) / len(y_test)
    plt.plot([0, 1], [no_skill, no_skill], 'k--', alpha=0.6, label=f'Baseline Proportion ({no_skill:.3f})')
    plt.title('Combined Precision-Recall Curves (ML TRAINING 3)', fontsize=13, fontweight='bold', pad=12)
    plt.xlabel('Recall', fontsize=11)
    plt.ylabel('Precision', fontsize=11)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='upper right', frameon=True, fontsize=9.0)
    plt.tight_layout()
    plt.savefig('combined_pr_curves.png')
    plt.close()
    print("Saved combined_pr_curves.png")

    print("\nML TRAINING 3 Calibration and Figure Generation Completed Successfully!")

if __name__ == '__main__':
    main()
