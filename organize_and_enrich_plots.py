import os
import shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def enrich_and_organize_ml1():
    base = "ML TRAINING 1"
    plots_dir = os.path.join(base, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Copy all existing pngs into plots/
    for f in os.listdir(base):
        if f.endswith('.png') and f != 'plots':
            shutil.copyfile(os.path.join(base, f), os.path.join(plots_dir, f))
            
    # Generate threshold sensitivity curve for BNN
    bgnn_path = os.path.join(base, "bgnn_probs.npy")
    y_test_path = os.path.join("paper_anti", "y_test.npy")
    if os.path.exists(bgnn_path) and os.path.exists(y_test_path):
        probs = np.load(bgnn_path)
        y_test = np.load(y_test_path)
        if len(probs) == len(y_test):
            ths = np.linspace(0.5, 0.99, 50)
            accs, f1s, recs, precs = [], [], [], []
            for th in ths:
                preds = (probs >= th).astype(int)
                accs.append(accuracy_score(y_test, preds))
                f1s.append(f1_score(y_test, preds, zero_division=0))
                recs.append(recall_score(y_test, preds, zero_division=0))
                precs.append(precision_score(y_test, preds, zero_division=0))
                
            fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
            ax.plot(ths, [a*100 for a in accs], label='Accuracy (%)', color='#27AE60', lw=2.5)
            ax.plot(ths, [f*100 for f in f1s], label='F1-Score (%)', color='#2980B9', lw=2.5)
            ax.plot(ths, [r*100 for r in recs], label='Recall (%)', color='#E67E22', lw=2)
            ax.plot(ths, [p*100 for p in precs], label='Precision (%)', color='#8E44AD', lw=2)
            ax.axhline(90, color='red', linestyle='--', label='90% Target Accuracy Threshold')
            ax.axvline(0.932, color='gray', linestyle=':', label='Optimal Calibrated Threshold (0.932)')
            ax.set_title("ML TRAINING 1: Bayesian GNN Threshold Sensitivity & 90%+ Accuracy Frontier", fontsize=12, weight='bold', color='#1B365D')
            ax.set_xlabel("Decision Threshold (θ)", fontsize=10)
            ax.set_ylabel("Metric Score (%)", fontsize=10)
            ax.set_ylim(0, 102)
            ax.grid(True, alpha=0.3)
            ax.legend(loc='lower left', fontsize=9)
            plt.tight_layout()
            out_path = os.path.join(plots_dir, "threshold_sensitivity_curves.png")
            plt.savefig(out_path, dpi=300)
            shutil.copyfile(out_path, os.path.join(base, "threshold_sensitivity_curves.png"))
            plt.close()

    print(f"ML TRAINING 1 plots organized: {len(os.listdir(plots_dir))} plots in {plots_dir}")

def enrich_and_organize_ml2():
    base = "ML TRAINING 2"
    plots_dir = os.path.join(base, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Copy all existing pngs into plots/
    for f in os.listdir(base):
        if f.endswith('.png') and f != 'plots':
            shutil.copyfile(os.path.join(base, f), os.path.join(plots_dir, f))
            
    # Generate metric trade-off scatter plot
    metrics_path = os.path.join(base, "model_comparison_metrics.csv")
    if os.path.exists(metrics_path):
        df = pd.read_csv(metrics_path)
        fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)
        sns.scatterplot(data=df, x='AUC-ROC', y='F1-Score', size='Recall', sizes=(80, 300), hue='Model', palette='tab10', ax=ax)
        for _, row in df.iterrows():
            ax.annotate(row['Model'], (row['AUC-ROC'] + 0.0015, row['F1-Score'] + 0.003), fontsize=8.5, weight='bold')
        ax.set_title("ML TRAINING 2: Discrimination vs. Detection Trade-Off (100% Dense Graph)", fontsize=12, weight='bold', color='#1B365D')
        ax.set_xlabel("AUC-ROC Score", fontsize=10)
        ax.set_ylabel("F1-Score", fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        plt.tight_layout()
        out_path = os.path.join(plots_dir, "metric_tradeoff_scatter.png")
        plt.savefig(out_path, dpi=300)
        shutil.copyfile(out_path, os.path.join(base, "metric_tradeoff_scatter.png"))
        plt.close()

    print(f"ML TRAINING 2 plots organized: {len(os.listdir(plots_dir))} plots in {plots_dir}")

def enrich_and_organize_ml3():
    base = "ML TRAINING 3"
    plots_dir = os.path.join(base, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Copy all existing pngs into plots/
    for f in os.listdir(base):
        if f.endswith('.png') and f != 'plots':
            shutil.copyfile(os.path.join(base, f), os.path.join(plots_dir, f))
            
    # Generate threshold sensitivity curve for ML 3 BNN
    bgnn_path = os.path.join(base, "bgnn_probs.npy")
    csv_path = os.path.join("dataset", "elliptic_dataset_fully_labeled.csv")
    if os.path.exists(bgnn_path) and os.path.exists(csv_path):
        probs = np.load(bgnn_path)
        df = pd.read_csv(csv_path, usecols=['time_step', 'final_label'])
        df_test = df[df['time_step'] >= 35]
        y_test = df_test['final_label'].values
        
        if len(probs) == len(y_test):
            ths = np.linspace(0.5, 0.99, 50)
            accs, f1s, recs, precs = [], [], [], []
            for th in ths:
                preds = (probs >= th).astype(int)
                accs.append(accuracy_score(y_test, preds))
                f1s.append(f1_score(y_test, preds, zero_division=0))
                recs.append(recall_score(y_test, preds, zero_division=0))
                precs.append(precision_score(y_test, preds, zero_division=0))
                
            fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
            ax.plot(ths, [a*100 for a in accs], label='Accuracy (%)', color='#27AE60', lw=2.5)
            ax.plot(ths, [f*100 for f in f1s], label='F1-Score (%)', color='#2980B9', lw=2.5)
            ax.plot(ths, [r*100 for r in recs], label='Recall (%)', color='#E67E22', lw=2)
            ax.plot(ths, [p*100 for p in precs], label='Precision (%)', color='#8E44AD', lw=2)
            ax.axvline(0.932, color='red', linestyle='--', label='Calibrated Threshold (0.932)')
            ax.set_title("ML TRAINING 3: Bayesian Directional GNN Threshold Frontier (67,504 Test Nodes)", fontsize=12, weight='bold', color='#1B365D')
            ax.set_xlabel("Decision Threshold (θ)", fontsize=10)
            ax.set_ylabel("Metric Score (%)", fontsize=10)
            ax.set_ylim(0, 102)
            ax.grid(True, alpha=0.3)
            ax.legend(loc='lower left', fontsize=9)
            plt.tight_layout()
            out_path = os.path.join(plots_dir, "threshold_sensitivity_curves.png")
            plt.savefig(out_path, dpi=300)
            shutil.copyfile(out_path, os.path.join(base, "threshold_sensitivity_curves.png"))
            plt.close()

    # Generate metric trade-off scatter plot for ML 3
    metrics_path = os.path.join(base, "model_comparison_metrics.csv")
    if os.path.exists(metrics_path):
        df = pd.read_csv(metrics_path)
        fig, ax = plt.subplots(figsize=(8.5, 5.5), dpi=300)
        sns.scatterplot(data=df, x='AUC-ROC', y='F1-Score', size='Recall', sizes=(80, 300), hue='Model', palette='Set2', ax=ax)
        for _, row in df.iterrows():
            ax.annotate(row['Model'], (row['AUC-ROC'] + 0.001, row['F1-Score'] + 0.003), fontsize=8.5, weight='bold')
        ax.set_title("ML TRAINING 3: Directional Residual GNNs Performance Frontier", fontsize=12, weight='bold', color='#1B365D')
        ax.set_xlabel("AUC-ROC Score", fontsize=10)
        ax.set_ylabel("F1-Score", fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        plt.tight_layout()
        out_path = os.path.join(plots_dir, "metric_tradeoff_scatter.png")
        plt.savefig(out_path, dpi=300)
        shutil.copyfile(out_path, os.path.join(base, "metric_tradeoff_scatter.png"))
        plt.close()

    print(f"ML TRAINING 3 plots organized: {len(os.listdir(plots_dir))} plots in {plots_dir}")

if __name__ == '__main__':
    enrich_and_organize_ml1()
    enrich_and_organize_ml2()
    enrich_and_organize_ml3()
