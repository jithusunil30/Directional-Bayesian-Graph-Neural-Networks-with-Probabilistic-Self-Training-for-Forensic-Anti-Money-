import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set publication style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.family'] = 'sans-serif'

plot_dirs = [
    r"C:\Users\niran\.gemini\antigravity-ide\brain\b422ee4b-f2cf-45f8-8301-5ffb87f5b8f2\plots",
    r"C:\Users\niran\.gemini\antigravity-ide\brain\e88b8197-7eb1-46c0-be23-e31e84dd96b3\plots"
]
for d in plot_dirs:
    os.makedirs(d, exist_ok=True)

# Data from inference evaluation
models = ['Random Forest (Baseline)', 'Proposed Bayesian GNN', 'Proposed Bayesian GAT']
tp = [740, 759, 757]
fn = [343, 324, 326]
fp = [7, 24, 25]
tn = [15580, 15563, 15562]
precision = [0.9906, 0.9693, 0.9680]
recall = [0.6833, 0.7008, 0.6990]
specificity = [0.9996, 0.9985, 0.9984]
f1_score = [0.8087, 0.8135, 0.8118]

# Create Figure 19: Comprehensive Confusion Metrics Comparison
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
colors = ['#34495e', '#1abc9c', '#3498db']

# Subplot 1: True Positives (Caught Illicit Transactions)
bars1 = axes[0, 0].bar(models, tp, color=colors, width=0.55, edgecolor='black', linewidth=1.2)
axes[0, 0].set_title("A. True Positives (TP): Illicit Transactions Correctly Identified", fontsize=12, fontweight='bold', pad=12)
axes[0, 0].set_ylabel("Transaction Count", fontsize=11)
axes[0, 0].set_ylim(0, 850)
for bar in bars1:
    yval = bar.get_height()
    axes[0, 0].text(bar.get_x() + bar.get_width()/2.0, yval + 12, f"{int(yval)}", ha='center', va='bottom', fontsize=11, fontweight='bold')

# Subplot 2: False Negatives (Missed Illicit Transactions)
bars2 = axes[0, 1].bar(models, fn, color=colors, width=0.55, edgecolor='black', linewidth=1.2)
axes[0, 1].set_title("B. False Negatives (FN): Missed Illicit Transactions (Lower is Better)", fontsize=12, fontweight='bold', pad=12)
axes[0, 1].set_ylabel("Transaction Count", fontsize=11)
axes[0, 1].set_ylim(0, 400)
for bar in bars2:
    yval = bar.get_height()
    axes[0, 1].text(bar.get_x() + bar.get_width()/2.0, yval + 8, f"{int(yval)}", ha='center', va='bottom', fontsize=11, fontweight='bold')

# Subplot 3: Recall (Sensitivity) vs Specificity vs Precision vs F1
x = np.arange(len(models))
width = 0.20

rects1 = axes[1, 0].bar(x - 1.5*width, [r*100 for r in recall], width, label='Recall (Sensitivity)', color='#e74c3c', edgecolor='black')
rects2 = axes[1, 0].bar(x - 0.5*width, [p*100 for p in precision], width, label='Precision', color='#2ecc71', edgecolor='black')
rects3 = axes[1, 0].bar(x + 0.5*width, [s*100 for s in specificity], width, label='Specificity', color='#9b59b6', edgecolor='black')
rects4 = axes[1, 0].bar(x + 1.5*width, [f*100 for f in f1_score], width, label='Illicit F1-Score', color='#f39c12', edgecolor='black')

axes[1, 0].set_title("C. Detailed Rate Metric Breakdown (%)", fontsize=12, fontweight='bold', pad=12)
axes[1, 0].set_ylabel("Percentage (%)", fontsize=11)
axes[1, 0].set_xticks(x)
axes[1, 0].set_xticklabels(models, fontsize=9.5, fontweight='bold')
axes[1, 0].set_ylim(50, 105)
axes[1, 0].legend(loc='lower right', frameon=True)

for rect in rects1 + rects2 + rects3 + rects4:
    h = rect.get_height()
    axes[1, 0].text(rect.get_x() + rect.get_width()/2.0, h + 0.8, f"{h:.1f}%", ha='center', va='bottom', fontsize=8, rotation=90)

# Subplot 4: Illicit F1-Score Improvement Over Random Forest
f1_diff = [(f - f1_score[0])*100 for f in f1_score]
bars4 = axes[1, 1].bar(models, f1_score, color=colors, width=0.55, edgecolor='black', linewidth=1.2)
axes[1, 1].set_title("D. Final Illicit F1-Score Benchmark Comparison", fontsize=12, fontweight='bold', pad=12)
axes[1, 1].set_ylabel("Illicit Class F1-Score", fontsize=11)
axes[1, 1].set_ylim(0.75, 0.83)
for bar, diff in zip(bars4, f1_diff):
    yval = bar.get_height()
    diff_str = f" ({diff:+.2f}%)" if diff != 0 else " (Baseline)"
    axes[1, 1].text(bar.get_x() + bar.get_width()/2.0, yval + 0.001, f"{yval:.4f}{diff_str}", ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.suptitle("Figure 19: Quantitative Confusion Metrics & Performance Improvement Comparison", fontsize=15, fontweight='bold', y=0.99)
plt.tight_layout()

for d in plot_dirs:
    plt.savefig(os.path.join(d, "fig19_confusion_metrics_comparison.png"), dpi=300)
plt.close()

print("[+] Figure 19 successfully generated and saved to plot directories!")
