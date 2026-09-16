import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, auc, confusion_matrix
from sklearn.calibration import calibration_curve

# Publication style
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.titlesize': 16,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'font.family': 'sans-serif'
})

def generate_all_plots(output_dir=None):
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(__file__))

    os.makedirs(output_dir, exist_ok=True)
    print("=" * 75)
    print("   GENERATING PUBLICATION-QUALITY FIGURES & VISUALIZATIONS (300 DPI)   ")
    print("=" * 75)

    # 1. Load Ground Truth Test Labels
    y_test_path = os.path.join(output_dir, 'y_test.npy')
    if not os.path.exists(y_test_path):
        print(f"Error: {y_test_path} not found. Please run baseline_models.py first.")
        return

    y_test = np.load(y_test_path)

    # Models and file mapping
    model_files = {
        'Logistic Regression': 'test_probs_logistic_regression.npy',
        'Random Forest': 'test_probs_random_forest.npy',
        'XGBoost': 'test_probs_xgboost.npy',
        'MLP': 'test_probs_mlp.npy',
        'GCN': 'test_probs_gcn.npy',
        'GraphSAGE': 'test_probs_graphsage.npy',
        'GAT (Supervised)': 'test_probs_gat.npy',
        'GAT + Pseudo-Labels (Ours)': 'test_probs_gat_pseudo.npy'
    }

    model_probs = {}
    for name, fname in model_files.items():
        fpath = os.path.join(output_dir, fname)
        if os.path.exists(fpath):
            model_probs[name] = np.load(fpath)

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#17becf']

    # -------------------------------------------------------------
    # Plot 1: Combined ROC Curves
    # -------------------------------------------------------------
    print("\n[1/6] Plotting Combined ROC Curves...")
    fig, ax = plt.subplots(figsize=(9, 7))
    for idx, (name, probs) in enumerate(model_probs.items()):
        fpr, tpr, _ = roc_curve(y_test, probs)
        roc_auc = auc(fpr, tpr)
        is_ours = 'Ours' in name
        lw = 2.8 if is_ours else 1.8
        ls = '-' if not is_ours else '-'
        color = '#d62728' if is_ours else colors[idx % len(colors)]
        ax.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.3f})", lw=lw, ls=ls, color=color)

    ax.plot([0, 1], [0, 1], 'k--', lw=1.2, label='Random Chance (AUC = 0.500)')
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.05])
    ax.set_xlabel('False Positive Rate (1 - Specificity)')
    ax.set_ylabel('True Positive Rate (Sensitivity / Recall)')
    ax.set_title('Out-of-Time Test ROC Curves on Bitcoin Transaction Graph')
    ax.legend(loc='lower right', frameon=True, framealpha=0.95)
    ax.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    roc_fig_path = os.path.join(output_dir, 'roc_curves_comparison.png')
    fig.savefig(roc_fig_path)
    plt.close(fig)
    print(f" -> Saved: {roc_fig_path}")

    # -------------------------------------------------------------
    # Plot 2: Combined Precision-Recall Curves
    # -------------------------------------------------------------
    print("\n[2/6] Plotting Combined Precision-Recall (PR) Curves...")
    fig, ax = plt.subplots(figsize=(9, 7))
    for idx, (name, probs) in enumerate(model_probs.items()):
        prec, rec, _ = precision_recall_curve(y_test, probs)
        pr_auc = auc(rec, prec)
        is_ours = 'Ours' in name
        lw = 2.8 if is_ours else 1.8
        color = '#d62728' if is_ours else colors[idx % len(colors)]
        ax.plot(rec, prec, label=f"{name} (PR-AUC = {pr_auc:.3f})", lw=lw, color=color)

    base_rate = (y_test == 1).sum() / len(y_test)
    ax.axhline(y=base_rate, color='k', linestyle='--', lw=1.2, label=f'Baseline Class Ratio ({base_rate:.3f})')
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.05])
    ax.set_xlabel('Recall (Illicit Fraud Detection Rate)')
    ax.set_ylabel('Precision (True Illicit / Flagged Illicit)')
    ax.set_title('Precision-Recall Curves under Class Imbalance (~10:1)')
    ax.legend(loc='upper right', frameon=True, framealpha=0.95)
    ax.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    pr_fig_path = os.path.join(output_dir, 'pr_curves_comparison.png')
    fig.savefig(pr_fig_path)
    plt.close(fig)
    print(f" -> Saved: {pr_fig_path}")

    # -------------------------------------------------------------
    # Plot 3: Confusion Matrices Grid
    # -------------------------------------------------------------
    print("\n[3/6] Plotting Confusion Matrices Grid...")
    selected_models = ['Logistic Regression', 'Random Forest', 'XGBoost', 'GCN', 'GraphSAGE', 'GAT (Supervised)', 'GAT + Pseudo-Labels (Ours)']
    n_models = len([m for m in selected_models if m in model_probs])
    n_cols = 3
    n_rows = int(np.ceil(n_models / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4.5 * n_rows))
    axes = axes.flatten()

    for idx, name in enumerate(selected_models):
        if name not in model_probs:
            continue
        ax = axes[idx]
        probs = model_probs[name]
        preds = (probs >= 0.5).astype(int)
        cm = confusion_matrix(y_test, preds)
        
        # Calculate percentages
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        annot_matrix = np.empty_like(cm, dtype=object)
        for r in range(2):
            for c in range(2):
                annot_matrix[r, c] = f"{cm[r, c]:,}\n({cm_norm[r, c]*100:.1f}%)"

        sns.heatmap(cm, annot=annot_matrix, fmt='', cmap='Blues', cbar=False, ax=ax,
                    xticklabels=['Licit (0)', 'Illicit (1)'],
                    yticklabels=['Licit (0)', 'Illicit (1)'])
        ax.set_title(f"{name}", fontsize=13, fontweight='bold')
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')

    for extra_ax in axes[n_models:]:
        fig.delaxes(extra_ax)

    plt.tight_layout()
    cm_fig_path = os.path.join(output_dir, 'confusion_matrices_grid.png')
    fig.savefig(cm_fig_path)
    plt.close(fig)
    print(f" -> Saved: {cm_fig_path}")

    # -------------------------------------------------------------
    # Plot 4: Calibration Curves (Reliability Diagram)
    # -------------------------------------------------------------
    print("\n[4/6] Plotting Calibration Reliability Diagram...")
    fig, ax = plt.subplots(figsize=(9, 7))
    for idx, (name, probs) in enumerate(model_probs.items()):
        if name in ['Logistic Regression', 'XGBoost', 'GCN', 'GAT (Supervised)', 'GAT + Pseudo-Labels (Ours)']:
            prob_true, prob_pred = calibration_curve(y_test, probs, n_bins=10, strategy='uniform')
            is_ours = 'Ours' in name
            lw = 2.5 if is_ours else 1.8
            ax.plot(prob_pred, prob_true, marker='o', lw=lw, label=f"{name}")

    ax.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Perfect Calibration')
    ax.set_xlabel('Mean Predicted Probability')
    ax.set_ylabel('Empirical Fraction of Positives')
    ax.set_title('Reliability Calibration Curves (10 Bins)')
    ax.legend(loc='upper left', frameon=True)
    ax.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    cal_fig_path = os.path.join(output_dir, 'calibration_curves_comparison.png')
    fig.savefig(cal_fig_path)
    plt.close(fig)
    print(f" -> Saved: {cal_fig_path}")

    # -------------------------------------------------------------
    # Plot 5: Pseudo-Labeling Uplift Barchart
    # -------------------------------------------------------------
    print("\n[5/6] Plotting F1 Uplift Barchart...")
    comp_xlsx = os.path.join(output_dir, 'pseudolabeling_performance_comparison.xlsx')
    base_xlsx = os.path.join(output_dir, 'baseline_performance_report.xlsx')

    if os.path.exists(comp_xlsx) and os.path.exists(base_xlsx):
        df_base = pd.read_excel(base_xlsx)
        df_aug = pd.read_excel(comp_xlsx, sheet_name='PseudoLabeling_Performance')

        # Compare F1 Scores
        models_comp = ['Logistic Regression', 'Random Forest', 'XGBoost', 'GAT']
        base_f1s = []
        aug_f1s = []
        for m in models_comp:
            b_val = df_base.loc[df_base['Model'] == m, 'F1_Score'].values
            base_f1s.append(b_val[0] if len(b_val) > 0 else 0)
            
            # Find matching augmented
            a_val = df_aug.loc[df_aug['Model'].str.contains(m, regex=False), 'F1_Score'].values
            aug_f1s.append(a_val[0] if len(a_val) > 0 else 0)

        x_pos = np.arange(len(models_comp))
        width = 0.35

        fig, ax = plt.subplots(figsize=(9, 6))
        rects1 = ax.bar(x_pos - width/2, base_f1s, width, label='Supervised Baseline', color='#4a90e2')
        rects2 = ax.bar(x_pos + width/2, aug_f1s, width, label='+ GAT Pseudo-Labeling (Semi-Supervised)', color='#e94e77')

        ax.set_ylabel('F1-Score (Illicit Class)')
        ax.set_title('Out-of-Time F1-Score Uplift via Semi-Supervised Graph Learning')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(models_comp, fontweight='bold')
        ax.legend(loc='upper left')
        ax.grid(axis='y', linestyle=':', alpha=0.7)

        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                ax.annotate(f'{height:.3f}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3), textcoords="offset points",
                            ha='center', va='bottom', fontsize=10, fontweight='bold')

        autolabel(rects1)
        autolabel(rects2)
        plt.tight_layout()
        uplift_fig_path = os.path.join(output_dir, 'pseudolabeling_f1_uplift_barchart.png')
        fig.savefig(uplift_fig_path)
        plt.close(fig)
        print(f" -> Saved: {uplift_fig_path}")

    # -------------------------------------------------------------
    # Plot 6: Unlabeled Risk Distribution
    # -------------------------------------------------------------
    print("\n[6/6] Plotting Unlabeled Risk Distribution...")
    csv_unlabeled = os.path.join(output_dir, 'unlabeled_node_predictions_full.csv')
    if os.path.exists(csv_unlabeled):
        df_unl = pd.read_csv(csv_unlabeled, usecols=['predicted_prob_illicit'])
        fig, ax = plt.subplots(figsize=(9, 6))
        sns.histplot(df_unl['predicted_prob_illicit'], bins=50, kde=True, color='#2b5c8f', ax=ax)
        ax.axvline(x=0.70, color='red', linestyle='--', lw=2, label='High Risk Threshold (0.70)')
        ax.axvline(x=0.30, color='orange', linestyle='--', lw=2, label='Medium Risk Threshold (0.30)')
        ax.set_xlabel('Predicted Posterior Illicit Probability P(Illicit | X, G)')
        ax.set_ylabel('Transaction Count')
        ax.set_title(f'Risk Distribution across 157,205 Unlabeled Bitcoin Transactions')
        ax.legend(loc='upper right')
        ax.grid(True, linestyle=':', alpha=0.5)
        plt.tight_layout()
        risk_fig_path = os.path.join(output_dir, 'unlabeled_risk_distribution.png')
        fig.savefig(risk_fig_path)
        plt.close(fig)
        print(f" -> Saved: {risk_fig_path}")

    print("\n" + "=" * 75)
    print("   ALL FIGURES GENERATED AND SAVED SUCCESSFULLY!   ")
    print("=" * 75)

if __name__ == '__main__':
    generate_all_plots()
