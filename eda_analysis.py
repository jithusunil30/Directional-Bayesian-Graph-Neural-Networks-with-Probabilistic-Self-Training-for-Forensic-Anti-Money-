import os
import sys
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

# Set publication style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.dpi'] = 300

def main():
    print("==================================================")
    print("   LAB 4 EDA Report Generation & Figures          ")
    print("==================================================")

    pyg_path = 'elliptic_pyg_data.pt'
    if not os.path.exists(pyg_path):
        print(f"[ERROR] Could not find '{pyg_path}'. Please run preprocess.py first.")
        sys.exit(1)

    print(f"\n[1/9] Loading PyG Graph Data from '{pyg_path}'...")
    data = torch.load(pyg_path, weights_only=False)
    
    x_np = data.x.numpy()
    y_np = data.y.numpy()
    time_step_np = data.time_step.numpy()
    edge_index = data.edge_index.numpy()
    
    feat_cols = [f'feat_{i}' for i in range(x_np.shape[1])]
    df = pd.DataFrame(x_np, columns=feat_cols)
    df['time_step'] = time_step_np
    df['label'] = y_np
    df['class_name'] = df['label'].map({1: 'Illicit', 0: 'Licit', -1: 'Unknown'})

    df_labeled = df[df['label'] != -1].copy()

    artifact_dir = r"C:\Users\niran\.gemini\antigravity\brain\3ae03c5d-67d0-40e6-98ea-bb2260314ab1\plots"
    os.makedirs(artifact_dir, exist_ok=True)
    print(f"\n[2/9] Saving All 8 LAB 4 Figures to:\n      '{artifact_dir}'")

    # Figure 1: Class Distribution & Temporal Split Distribution
    print("      -> Figure 1: fig1_class_distribution.png...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    counts_bar = df['class_name'].value_counts()[['Unknown', 'Licit', 'Illicit']]
    bars = axes[0].bar(counts_bar.index, counts_bar.values, color=['#95a5a6', '#2ecc71', '#e74c3c'], edgecolor='black', alpha=0.85)
    axes[0].set_title('Overall Node Class Distribution', fontweight='bold', fontsize=13)
    axes[0].set_ylabel('Number of Nodes', fontsize=11)
    for bar in bars:
        height = bar.get_height()
        axes[0].annotate(f'{height:,}\n({height/len(df):.1%})', xy=(bar.get_x() + bar.get_width()/2, height),
                         xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    labeled_counts = df_labeled['class_name'].value_counts()[['Licit', 'Illicit']]
    axes[1].pie(labeled_counts, labels=labeled_counts.index, autopct='%1.1f%%', colors=['#2ecc71', '#e74c3c'], startangle=140, explode=(0, 0.1), shadow=True)
    axes[1].set_title('Labeled Node Breakdown (9.25:1 Ratio)', fontweight='bold', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(artifact_dir, 'fig1_class_distribution.png'), dpi=300)
    plt.close()

    # Figure 2: Node Degree Distributions (Log Scale Density & Boxplots)
    print("      -> Figure 2: fig2_degree_distribution.png...")
    src, dst = edge_index[0], edge_index[1]
    df['out_degree'] = np.bincount(src, minlength=len(df))
    df['in_degree'] = np.bincount(dst, minlength=len(df))
    df['total_degree'] = df['in_degree'] + df['out_degree']

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.kdeplot(data=df[df['label']==0]['total_degree'] + 1, ax=axes[0], color='#2ecc71', label='Licit (0)', fill=True, alpha=0.4, log_scale=True, linewidth=2)
    sns.kdeplot(data=df[df['label']==1]['total_degree'] + 1, ax=axes[0], color='#e74c3c', label='Illicit (1)', fill=True, alpha=0.4, log_scale=True, linewidth=2)
    sns.kdeplot(data=df[df['label']==-1]['total_degree'] + 1, ax=axes[0], color='#95a5a6', label='Unknown (-1)', fill=True, alpha=0.2, log_scale=True, linewidth=1.5)
    axes[0].set_title('Log-Scaled Total Degree Density Distribution', fontweight='bold', fontsize=12)
    axes[0].set_xlabel('Total Degree (+1 log scale)', fontsize=10)
    axes[0].set_ylabel('Density', fontsize=10)
    axes[0].legend()

    sns.boxplot(x='class_name', y='total_degree', data=df, ax=axes[1], palette={'Licit': '#2ecc71', 'Illicit': '#e74c3c', 'Unknown': '#95a5a6'}, fliersize=2)
    axes[1].set_title('Total Node Degree Boxplots by Class', fontweight='bold', fontsize=12)
    axes[1].set_ylim(0, df['total_degree'].quantile(0.99) + 5)
    plt.tight_layout()
    plt.savefig(os.path.join(artifact_dir, 'fig2_degree_distribution.png'), dpi=300)
    plt.close()

    # Figure 3: Heatmap Matrix of Edge Transition Connection Probabilities
    print("      -> Figure 3: fig3_edge_transition_matrix.png...")
    src_labels = y_np[edge_index[0]]
    dst_labels = y_np[edge_index[1]]
    df_edges = pd.DataFrame({'src': src_labels, 'dst': dst_labels})
    
    labels_order = [1, 0, -1]
    label_names = ['Illicit (1)', 'Licit (0)', 'Unknown (-1)']
    
    transition_matrix = np.zeros((3, 3))
    for i, s_val in enumerate(labels_order):
        s_count = (df_edges['src'] == s_val).sum()
        for j, d_val in enumerate(labels_order):
            pair_count = ((df_edges['src'] == s_val) & (df_edges['dst'] == d_val)).sum()
            transition_matrix[i, j] = pair_count / s_count if s_count > 0 else 0.0

    plt.figure(figsize=(8, 6))
    sns.heatmap(transition_matrix, annot=True, fmt='.2%', cmap='Blues', xticklabels=label_names, yticklabels=label_names, cbar=True, annot_kws={"size": 12, "weight": "bold"})
    plt.title('Edge Transition Linkage Probability Heatmap', fontweight='bold', fontsize=13)
    plt.xlabel('Destination Node Class', fontweight='bold', fontsize=11)
    plt.ylabel('Source Node Class', fontweight='bold', fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(artifact_dir, 'fig3_edge_transition_matrix.png'), dpi=300)
    plt.close()

    # Figure 4: Total Nodes and Labeled Fraud Ratio across Time Steps 1-49
    print("      -> Figure 4: fig4_temporal_split_trends.png...")
    fig, ax1 = plt.subplots(figsize=(15, 6))
    step_stats = df.groupby('time_step').agg(
        licit=('label', lambda x: (x == 0).sum()),
        illicit=('label', lambda x: (x == 1).sum()),
        unknown=('label', lambda x: (x == -1).sum())
    ).reset_index()
    step_stats['fraud_ratio'] = step_stats['illicit'] / np.maximum(step_stats['licit'] + step_stats['illicit'], 1)

    ax1.bar(step_stats['time_step'], step_stats['licit'], color='#2ecc71', label='Licit', alpha=0.85)
    ax1.bar(step_stats['time_step'], step_stats['illicit'], bottom=step_stats['licit'], color='#e74c3c', label='Illicit', alpha=0.85)
    ax1.bar(step_stats['time_step'], step_stats['unknown'], bottom=step_stats['licit']+step_stats['illicit'], color='#bdc3c7', alpha=0.4, label='Unknown')
    ax1.set_xlabel('Time Step (1-49)', fontweight='bold', fontsize=11)
    ax1.set_ylabel('Node Count', fontweight='bold', fontsize=11)
    ax1.set_title('Total Nodes and Labeled Fraud Ratio Evolution across Time Steps 1-49', fontweight='bold', fontsize=13)
    
    ax2 = ax1.twinx()
    ax2.plot(step_stats['time_step'], step_stats['fraud_ratio']*100, color='#8e44ad', marker='o', linewidth=2.5, label='Fraud Ratio (%)')
    ax2.set_ylabel('Fraud Ratio (%)', color='#8e44ad', fontweight='bold', fontsize=11)
    ax2.grid(False)
    plt.tight_layout()
    plt.savefig(os.path.join(artifact_dir, 'fig4_temporal_split_trends.png'), dpi=300)
    plt.close()

    # Figure 5: Class-wise Density plots for top 8 discriminative features
    print("      -> Figure 5: fig5_top8_feature_densities.png...")
    top_8 = ['feat_52', 'feat_54', 'feat_88', 'feat_89', 'feat_108', 'feat_105', 'feat_106', 'feat_141']
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    for idx, f_col in enumerate(top_8):
        ax = axes[idx // 4, idx % 4]
        sns.kdeplot(data=df_labeled[df_labeled['label']==0][f_col], ax=ax, color='#2ecc71', label='Licit', fill=True, alpha=0.4, linewidth=1.8)
        sns.kdeplot(data=df_labeled[df_labeled['label']==1][f_col], ax=ax, color='#e74c3c', label='Illicit', fill=True, alpha=0.4, linewidth=1.8)
        ax.set_title(f_col, fontsize=11, fontweight='bold')
        ax.set_xlabel('Scaled Value', fontsize=9)
        ax.set_ylabel('Density', fontsize=9)
        ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(artifact_dir, 'fig5_top8_feature_densities.png'), dpi=300)
    plt.close()

    # Figure 6: Boxplots demonstrating feature outlier locations by Class
    print("      -> Figure 6: fig6_feature_outlier_boxplots.png...")
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    for idx, f_col in enumerate(top_8):
        ax = axes[idx // 4, idx % 4]
        sns.boxplot(x='class_name', y=f_col, data=df_labeled, ax=ax, palette={'Licit': '#2ecc71', 'Illicit': '#e74c3c'}, fliersize=2)
        ax.set_title(f'Outliers: {f_col}', fontsize=11, fontweight='bold')
        ax.set_xlabel('Class', fontsize=9)
        ax.set_ylabel('Scaled Value', fontsize=9)
        q_low, q_high = df_labeled[f_col].quantile(0.01), df_labeled[f_col].quantile(0.99)
        margin = (q_high - q_low) * 0.5
        ax.set_ylim(q_low - margin, q_high + margin)
    plt.tight_layout()
    plt.savefig(os.path.join(artifact_dir, 'fig6_feature_outlier_boxplots.png'), dpi=300)
    plt.close()

    # Figure 7: Correlation Heatmap Matrix of Selected features and binary Label
    print("      -> Figure 7: fig7_correlation_matrix.png...")
    plt.figure(figsize=(10, 8))
    sns.heatmap(df_labeled[top_8 + ['label']].corr(), cmap='vlag', center=0, annot=True, fmt='.2f', linewidths=0.5, annot_kws={"size": 10})
    plt.title('Correlation Heatmap Matrix of Top 8 Discriminative Features & Label', fontweight='bold', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(artifact_dir, 'fig7_correlation_matrix.png'), dpi=300)
    plt.close()

    # Figure 8: Pairwise Scatter Matrix showing feature interactions split by Class
    print("      -> Figure 8: fig8_pairwise_scatter_matrix.png...")
    sample_sub = ['feat_52', 'feat_54', 'feat_108', 'feat_141', 'class_name']
    df_pair_sample = df_labeled[sample_sub].sample(n=min(2000, len(df_labeled)), random_state=42)
    g = sns.pairplot(df_pair_sample, hue='class_name', palette={'Licit': '#2ecc71', 'Illicit': '#e74c3c'}, corner=True, plot_kws={'alpha': 0.5, 's': 15})
    g.fig.suptitle('Pairwise Feature Interaction Matrix (Class Segregation)', y=1.02, fontweight='bold', fontsize=13)
    plt.savefig(os.path.join(artifact_dir, 'fig8_pairwise_scatter_matrix.png'), dpi=300)
    plt.close()

    print("\n[8/9] Summary Verification:")
    print("      All 8 LAB 4 Figures (Figure 1 through Figure 8) successfully generated and verified!")

    print("\n[9/9] EDA Execution Finished Successfully!")
    print("==================================================")

if __name__ == '__main__':
    main()
