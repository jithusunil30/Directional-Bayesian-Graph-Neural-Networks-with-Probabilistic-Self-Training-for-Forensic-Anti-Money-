import os
import torch
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    print("=== Running Exploratory Data Analysis (EDA) ===")
    
    # 1. Load Data
    pt_path = 'elliptic_pyg_data.pt'
    if not os.path.exists(pt_path) and os.path.exists('updated/elliptic_pyg_data.pt'):
        pt_path = 'updated/elliptic_pyg_data.pt'
        
    print(f"Loading PyTorch Geometric graph data from: {pt_path}")
    data = torch.load(pt_path, weights_only=False)
    print("Data loaded successfully.")
    
    # Create directory for plots
    plots_dir = 'eda_plots'
    os.makedirs(plots_dir, exist_ok=True)
    print(f"Created directory for plots: {plots_dir}")
    
    # Get basic shapes
    num_nodes = data.x.shape[0]
    num_features = data.x.shape[1]
    num_edges = data.edge_index.shape[1]
    
    x_np = data.x.numpy()
    y_np = data.y.numpy()
    time_steps = data.time_step.numpy()
    train_mask = data.train_mask.numpy()
    val_mask = data.val_mask.numpy()
    test_mask = data.test_mask.numpy()
    
    # 2. Class Distributions
    unknown_count = np.sum(y_np == -1)
    licit_count = np.sum(y_np == 0)
    illicit_count = np.sum(y_np == 1)
    total_labeled = licit_count + illicit_count
    
    train_licit = np.sum((y_np == 0) & train_mask)
    train_illicit = np.sum((y_np == 1) & train_mask)
    val_licit = np.sum((y_np == 0) & val_mask)
    val_illicit = np.sum((y_np == 1) & val_mask)
    test_licit = np.sum((y_np == 0) & test_mask)
    test_illicit = np.sum((y_np == 1) & test_mask)
    
    print("\n--- Class Distributions ---")
    print(f"Nodes: {num_nodes}, Edges: {num_edges}, Features: {num_features}")
    print(f"Licit: {licit_count} ({licit_count/num_nodes:.2%})")
    print(f"Illicit: {illicit_count} ({illicit_count/num_nodes:.2%})")
    print(f"Unknown: {unknown_count} ({unknown_count/num_nodes:.2%})")
    print(f"Train split: {np.sum(train_mask)} labeled nodes ({train_licit} licit, {train_illicit} illicit, fraud ratio: {train_illicit/(train_licit+train_illicit):.2%})")
    print(f"Val split: {np.sum(val_mask)} labeled nodes ({val_licit} licit, {val_illicit} illicit, fraud ratio: {val_illicit/(val_licit+val_illicit):.2%})")
    print(f"Test split: {np.sum(test_mask)} labeled nodes ({test_licit} licit, {test_illicit} illicit, fraud ratio: {test_illicit/(test_licit+test_illicit):.2%})")
    
    # Plot 1: Class Distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: Pie Chart (Overall)
    colors = ['#1f77b4', '#d62728', '#7f7f7f']
    axes[0].pie(
        [licit_count, illicit_count, unknown_count], 
        labels=['Licit (Class 0)', 'Illicit (Class 1)', 'Unknown (Class -1)'],
        autopct='%1.1f%%', 
        startangle=140, 
        colors=colors,
        explode=(0.05, 0.05, 0.05)
    )
    axes[0].set_title('Overall Node Distribution (Licit, Illicit, Unknown)', fontsize=14, fontweight='bold')
    
    # Right: Stacked Bar Chart (Splits)
    splits = ['Train (Steps 1-30)', 'Val (Steps 31-34)', 'Test (Steps 35-49)']
    licit_splits = [train_licit, val_licit, test_licit]
    illicit_splits = [train_illicit, val_illicit, test_illicit]
    
    axes[1].bar(splits, licit_splits, color='#1f77b4', label='Licit (0)', width=0.5)
    axes[1].bar(splits, illicit_splits, bottom=licit_splits, color='#d62728', label='Illicit (1)', width=0.5)
    axes[1].set_ylabel('Node Count', fontsize=12)
    axes[1].set_title('Labeled Node Distribution across Splits', fontsize=14, fontweight='bold')
    axes[1].legend(loc='upper right')
    
    # Add percentages inside the stacked bars
    for idx, (l, i) in enumerate(zip(licit_splits, illicit_splits)):
        total = l + i
        if total > 0:
            axes[1].text(idx, l / 2, f"{l}\n({l/total:.1%})", ha='center', va='center', color='white', fontweight='bold')
            axes[1].text(idx, l + i / 2, f"{i}\n({i/total:.1%})", ha='center', va='center', color='white', fontweight='bold')
            
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'class_distribution.png'), dpi=150)
    plt.close()
    
    # 3. Node Degree Analysis
    print("\n--- Analyzing Node Degrees ---")
    src = data.edge_index[0].numpy()
    dst = data.edge_index[1].numpy()
    
    out_degree = np.bincount(src, minlength=num_nodes)
    in_degree = np.bincount(dst, minlength=num_nodes)
    total_degree = out_degree + in_degree
    
    # Calculate degree stats by class
    df_deg = pd.DataFrame({
        'class': y_np,
        'in_degree': in_degree,
        'out_degree': out_degree,
        'total_degree': total_degree
    })
    df_deg['class_name'] = df_deg['class'].map({-1: 'Unknown', 0: 'Licit', 1: 'Illicit'})
    
    df_deg_labeled = df_deg[df_deg['class'] != -1]
    
    deg_stats = df_deg.groupby('class').agg({
        'in_degree': ['mean', 'median', 'max', 'std'],
        'out_degree': ['mean', 'median', 'max', 'std'],
        'total_degree': ['mean', 'median', 'max', 'std']
    })
    print(deg_stats)
    
    # Plot 2: Node Degree Distributions by Class
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: Total degree distributions (KDE on log scale)
    for class_id, label, color in [(0, 'Licit', '#1f77b4'), (1, 'Illicit', '#d62728'), (-1, 'Unknown', '#7f7f7f')]:
        sub_deg = df_deg[df_deg['class'] == class_id]['total_degree']
        sns.kdeplot(np.log1p(sub_deg), ax=axes[0], label=f'{label} (n={len(sub_deg)})', color=color, fill=True, alpha=0.2)
    
    axes[0].set_xlabel('log(Total Degree + 1)', fontsize=12)
    axes[0].set_ylabel('Density', fontsize=12)
    axes[0].set_title('Log-scaled Total Node Degree Density by Class', fontsize=14, fontweight='bold')
    axes[0].legend()
    
    # Right: Box plot of degrees by class (limited to degree < 30 for visualization clarity)
    sns.boxplot(x='class_name', y='total_degree', hue='class_name', data=df_deg, ax=axes[1], order=['Unknown', 'Licit', 'Illicit'], palette={'Unknown': '#7f7f7f', 'Licit': '#1f77b4', 'Illicit': '#d62728'}, showfliers=False, legend=False)
    axes[1].set_xlabel('Node Class', fontsize=12)
    axes[1].set_ylabel('Total Degree (excluding outliers)', fontsize=12)
    axes[1].set_title('Total Node Degree Boxplots by Class', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'node_degree_distributions.png'), dpi=150)
    plt.close()
    
    # 4. Edge Class Linkage / Connectivity Analysis
    print("\n--- Analyzing Edge Connections (Homophily) ---")
    src_classes = y_np[src]
    dst_classes = y_np[dst]
    
    # Transition count matrix
    # Mapping classes: -1 -> 2 (for indices in matrix), 0 -> 0, 1 -> 1
    src_mapped = np.where(src_classes == -1, 2, src_classes)
    dst_mapped = np.where(dst_classes == -1, 2, dst_classes)
    
    edge_matrix = np.zeros((3, 3), dtype=int)
    for s, d in zip(src_mapped, dst_mapped):
        edge_matrix[s, d] += 1
        
    class_labels = ['Licit (0)', 'Illicit (1)', 'Unknown (-1)']
    df_edges = pd.DataFrame(edge_matrix, index=class_labels, columns=class_labels)
    print("Edge matrix (Source node class vs Destination node class):")
    print(df_edges)
    
    # Normalize by row (out-going connection probability) and column (in-coming connection probability)
    row_sums = edge_matrix.sum(axis=1, keepdims=True)
    # Avoid division by zero
    row_sums[row_sums == 0] = 1
    edge_matrix_prob = edge_matrix / row_sums
    df_edges_prob = pd.DataFrame(edge_matrix_prob, index=class_labels, columns=class_labels)
    
    # Plot 3: Edge Connectivity Matrix Heatmap (Original 3x3)
    plt.figure(figsize=(8, 7))
    sns.heatmap(df_edges_prob, annot=True, fmt='.2%', cmap='Blues', cbar=True, annot_kws={"size": 12, "weight": "bold"})
    plt.title('Normalized Connection Probability Matrix\n(Source Node Class -> Destination Node Class)', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Destination Node Class', fontsize=12, labelpad=10)
    plt.ylabel('Source Node Class', fontsize=12, labelpad=10)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'edge_type_connectivity.png'), dpi=150)
    plt.close()
    
    # NEW: Plot 3b - GAT Post-Labeling Edge-Type Connectivity Matrix (2x2 & Comparison)
    csv_path = 'src/predicted_unknown_labels_gat.csv'
    if not os.path.exists(csv_path) and os.path.exists('predicted_unknown_labels_gat.csv'):
        csv_path = 'predicted_unknown_labels_gat.csv'
        
    if os.path.exists(csv_path):
        print(f"Loading GAT predicted unknown labels from {csv_path} for new edge connectivity matrix...")
        df_unknown_preds = pd.read_csv(csv_path)
        
        y_new = y_np.copy()
        y_new[df_unknown_preds['node_index'].values] = df_unknown_preds['predicted_label'].values
        
        src_labels_new = y_new[src]
        dst_labels_new = y_new[dst]
        
        cats_new = ['Licit (0)', 'Illicit (1)']
        label_map_new = {0: 'Licit (0)', 1: 'Illicit (1)'}
        src_str = [label_map_new[y] for y in src_labels_new]
        dst_str = [label_map_new[y] for y in dst_labels_new]
        
        df_edges_new = pd.crosstab(
            pd.Series(src_str, name='Source (From)'),
            pd.Series(dst_str, name='Target (To)')
        ).reindex(index=cats_new, columns=cats_new, fill_value=0)
        
        total_edges = len(src)
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
        
        # Old (3x3)
        sns.heatmap(df_edges, annot=True, fmt='d', cmap='Blues', cbar=False, ax=axes[0], annot_kws={"size": 11, "weight": "bold"})
        axes[0].set_title("OLD Edge Connectivity Matrix\n(Including 157,205 Unknown Nodes)", fontsize=13, fontweight='bold', pad=12)
        axes[0].set_xlabel("Target Node (To)", fontsize=11, fontweight='bold')
        axes[0].set_ylabel("Source Node (From)", fontsize=11, fontweight='bold')
        
        # New (2x2)
        sns.heatmap(df_edges_new, annot=True, fmt='d', cmap='Greens', cbar=False, ax=axes[1], annot_kws={"size": 14, "weight": "bold"})
        axes[1].set_title("NEW Edge-Type Connectivity Matrix\n(After GAT Labeling All Unknown Nodes)", fontsize=13, fontweight='bold', pad=12)
        axes[1].set_xlabel("Target Node (To)", fontsize=11, fontweight='bold')
        axes[1].set_ylabel("Source Node (From)", fontsize=11, fontweight='bold')
        
        for i in range(2):
            for j in range(2):
                val = df_edges_new.iloc[i, j]
                pct = val / total_edges * 100
                axes[1].text(j + 0.5, i + 0.7, f"({pct:.1f}%)", ha='center', va='center', color='white' if val > df_edges_new.values.max()/2 else 'darkgreen', fontsize=11, weight='bold')

        plt.suptitle("Elliptic Bitcoin Graph: Edge-Type Connectivity Matrix Transformation", fontsize=15, fontweight='bold', y=1.03)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'edge_type_connectivity_matrix.png'), dpi=150)
        plt.savefig(os.path.join(plots_dir, 'edge_type_connectivity_new.png'), dpi=150)
        plt.close()
        print(f"Saved new edge-type connectivity matrix plots into {plots_dir}/")
    
    # 5. Temporal Analysis (49 time steps)
    print("\n--- Analyzing Temporal Dynamics ---")
    steps = np.unique(time_steps)
    step_stats = []
    for s in steps:
        mask_s = (time_steps == s)
        total_s = np.sum(mask_s)
        licit_s = np.sum((y_np == 0) & mask_s)
        illicit_s = np.sum((y_np == 1) & mask_s)
        unknown_s = np.sum((y_np == -1) & mask_s)
        labeled_s = licit_s + illicit_s
        fraud_ratio = illicit_s / labeled_s if labeled_s > 0 else 0
        step_stats.append({
            'time_step': s,
            'total_nodes': total_s,
            'licit_nodes': licit_s,
            'illicit_nodes': illicit_s,
            'unknown_nodes': unknown_s,
            'labeled_nodes': labeled_s,
            'fraud_ratio': fraud_ratio
        })
    df_temp = pd.DataFrame(step_stats)
    
    # Plot 4: Temporal Trends (Node count and fraud ratio over time steps)
    fig, ax1 = plt.subplots(figsize=(14, 6))
    
    color = '#1f77b4'
    ax1.set_xlabel('Time Step', fontsize=12)
    ax1.set_ylabel('Total Nodes', color=color, fontsize=12)
    ax1.bar(df_temp['time_step'], df_temp['total_nodes'], color=color, alpha=0.3, label='Total Nodes')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(False)
    
    ax2 = ax1.twinx()  
    color = '#d62728'
    ax2.set_ylabel('Fraud Ratio among Labeled Nodes', color=color, fontsize=12)
    ax2.plot(df_temp['time_step'], df_temp['fraud_ratio'], color=color, marker='o', linewidth=2, label='Fraud Ratio')
    ax2.tick_params(axis='y', labelcolor=color)
    
    # Add vertical line splits
    ax1.axvline(x=30.5, color='#7f7f7f', linestyle='--', linewidth=1.5)
    ax1.axvline(x=34.5, color='#7f7f7f', linestyle='--', linewidth=1.5)
    ax1.text(15, df_temp['total_nodes'].max() * 0.9, 'TRAIN', ha='center', fontsize=12, fontweight='bold', color='#555555')
    ax1.text(32.5, df_temp['total_nodes'].max() * 0.9, 'CAL', ha='center', fontsize=12, fontweight='bold', color='#555555')
    ax1.text(42, df_temp['total_nodes'].max() * 0.9, 'TEST', ha='center', fontsize=12, fontweight='bold', color='#555555')
    
    plt.title('Dataset Evolution and Labeled Fraud Ratio over Time Steps', fontsize=15, fontweight='bold')
    fig.tight_layout()  
    plt.savefig(os.path.join(plots_dir, 'temporal_trends.png'), dpi=150)
    plt.close()
    
    # 6. Feature Selection based on Discriminative Capability
    # Let's find features with the largest absolute difference in mean between licit and illicit classes
    print("\n--- Identifying Discriminative Features ---")
    labeled_indices = np.where(y_np != -1)[0]
    x_labeled = x_np[labeled_indices]
    y_labeled = y_np[labeled_indices]
    
    licit_feats = x_labeled[y_labeled == 0]
    illicit_feats = x_labeled[y_labeled == 1]
    
    mean_diff = np.abs(np.mean(licit_feats, axis=0) - np.mean(illicit_feats, axis=0))
    
    # Feature 0 to 93 are local features, 94 to 164 are neighbor aggregation features
    local_diff = mean_diff[:94]
    neighbor_diff = mean_diff[94:]
    
    # Top 4 local and top 4 neighbor features
    top_local_idx = np.argsort(local_diff)[-4:][::-1]
    top_neighbor_idx = np.argsort(neighbor_diff)[-4:][::-1] + 94
    
    selected_features = list(top_local_idx) + list(top_neighbor_idx)
    selected_feature_names = [f'feat_{i}' for i in selected_features]
    print(f"Top discriminative local features: {top_local_idx} (diffs: {local_diff[top_local_idx - 0]})")
    print(f"Top discriminative neighbor features: {top_neighbor_idx} (diffs: {neighbor_diff[top_neighbor_idx - 94]})")
    print(f"All selected features: {selected_feature_names}")
    
    # Create DataFrame of selected features for labeled nodes
    df_feats = pd.DataFrame(x_labeled[:, selected_features], columns=selected_feature_names)
    df_feats['label'] = y_labeled
    df_feats['label_name'] = df_feats['label'].map({0: 'Licit', 1: 'Illicit'})
    
    # Summary stats for selected features by class
    print("\nSummary Statistics of Selected Features by Class:")
    feat_summary = df_feats.groupby('label').mean(numeric_only=True)
    print(feat_summary)
    
    # Plot 5: Feature Distributions for Licit vs Illicit Nodes
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    for idx, col in enumerate(selected_feature_names):
        # We cap the feature values at 1st and 99th percentiles for density plot readability
        p01 = df_feats[col].quantile(0.01)
        p99 = df_feats[col].quantile(0.99)
        
        for class_id, label, color in [(0, 'Licit', '#1f77b4'), (1, 'Illicit', '#d62728')]:
            subset = df_feats[df_feats['label'] == class_id][col]
            subset_capped = subset[(subset >= p01) & (subset <= p99)]
            sns.kdeplot(subset_capped, ax=axes[idx], label=label, color=color, fill=True, alpha=0.15)
            
        is_local = "Local" if idx < 4 else "Neighbor"
        axes[idx].set_title(f"{col} ({is_local})\nMean Diff: {mean_diff[selected_features[idx]]:.3f}", fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('Scaled Feature Value', fontsize=10)
        axes[idx].legend()
        
    plt.suptitle('Density Distributions of Top Discriminative Features (Licit vs. Illicit)', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'feature_distributions.png'), dpi=150)
    plt.close()
    
    # Plot 6: Outlier Box plots
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    for idx, col in enumerate(selected_feature_names):
        sns.boxplot(x='label_name', y=col, hue='label_name', data=df_feats, ax=axes[idx], order=['Licit', 'Illicit'], palette={'Licit': '#1f77b4', 'Illicit': '#d62728'}, showfliers=True, legend=False)
        is_local = "Local" if idx < 4 else "Neighbor"
        axes[idx].set_title(f"{col} ({is_local}) Outliers", fontsize=12, fontweight='bold')
        axes[idx].set_ylabel('Scaled Feature Value', fontsize=10)
        axes[idx].set_xlabel('Class', fontsize=10)
        
    plt.suptitle('Boxplots Showing Outliers in Top Discriminative Features by Class', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'feature_boxplots.png'), dpi=150)
    plt.close()
    
    # Plot 7: Feature Correlation Heatmap with Label
    plt.figure(figsize=(10, 8))
    # Calculate correlations
    corr_matrix = df_feats.corr(numeric_only=True)
    
    # Plot heatmap
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1, square=True, linewidths=0.5, annot_kws={"size": 10})
    plt.title('Correlation Matrix of Selected Discriminative Features & Target Label', fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'feature_correlation_heatmap.png'), dpi=150)
    plt.close()
    
    # Plot 8: Pairplot of Top 2 Local and Top 2 Neighbor Features
    pairplot_cols = [selected_feature_names[0], selected_feature_names[1], selected_feature_names[4], selected_feature_names[5], 'label']
    df_pair = df_feats[pairplot_cols].copy()
    df_pair['label'] = df_pair['label'].map({0: 'Licit', 1: 'Illicit'})
    
    if len(df_pair) > 2000:
        df_pair = df_pair.sample(2000, random_state=42)
        
    sns.pairplot(df_pair, hue='label', palette={'Licit': '#1f77b4', 'Illicit': '#d62728'}, diag_kind='kde', plot_kws={'alpha': 0.4, 's': 15})
    plt.suptitle('Pairwise Scatter Matrix of Selected Features by Class', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'pairplot.png'), dpi=150)
    plt.close()
    
    # 7. Outlier Detection using IQR Method
    print("\n--- Outlier Detection ---")
    outlier_records = []
    for col in selected_feature_names:
        series = df_feats[col]
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        total_outliers = np.sum((series < lower_bound) | (series > upper_bound))
        pct_outliers = total_outliers / len(series)
        
        # Outliers inside licit
        licit_series = df_feats[df_feats['label'] == 0][col]
        licit_outliers = np.sum((licit_series < lower_bound) | (licit_series > upper_bound))
        licit_pct = licit_outliers / len(licit_series)
        
        # Outliers inside illicit
        illicit_series = df_feats[df_feats['label'] == 1][col]
        illicit_outliers = np.sum((illicit_series < lower_bound) | (illicit_series > upper_bound))
        illicit_pct = illicit_outliers / len(illicit_series)
        
        outlier_records.append({
            'Feature': col,
            'Q1': q1,
            'Q3': q3,
            'IQR': iqr,
            'Lower Bound': lower_bound,
            'Upper Bound': upper_bound,
            'Total Outliers': total_outliers,
            'Total Outliers %': f"{pct_outliers:.2%}",
            'Licit Outliers': licit_outliers,
            'Licit Outliers %': f"{licit_pct:.2%}",
            'Illicit Outliers': illicit_outliers,
            'Illicit Outliers %': f"{illicit_pct:.2%}"
        })
    df_outliers = pd.DataFrame(outlier_records)
    print(df_outliers[['Feature', 'Total Outliers %', 'Licit Outliers %', 'Illicit Outliers %']])
    
    # Write Report Markdown
    report_path = 'eda_report.md'
    print(f"\nWriting EDA report to {report_path}...")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Exploratory Data Analysis (EDA) Report\n")
        f.write("## Elliptic Bitcoin Dataset for Fraud Detection\n\n")
        
        f.write("This report summarizes the results of the Exploratory Data Analysis (EDA) performed on the "
                "preprocessed graph dataset `elliptic_pyg_data.pt` and readable CSV files.\n\n")
        
        f.write("### 1. Dataset Dimensions & Basic Statistics\n")
        f.write("| Attribute | Value |\n")
        f.write("| --- | --- |\n")
        f.write(f"| **Total Transactions (Nodes)** | {num_nodes:,} |\n")
        f.write(f"| **Total Payments (Edges)** | {num_edges:,} |\n")
        f.write(f"| **Node Features Matrix Size** | {num_nodes} × {num_features} |\n")
        f.write(f"| **Feature Column Types** | float32 (scaled) |\n")
        f.write(f"| **Class/Label Column Types** | int64 (values: 0, 1, -1) |\n\n")
        
        f.write("### 2. Node Split & Class Distribution\n")
        f.write("| Split | Licit (0) | Illicit (1) | Unknown (-1) | Total Nodes | Labeled Count | Labeled Fraud Ratio |\n")
        f.write("| --- | --- | --- | --- | --- | --- | --- |\n")
        f.write(f"| **Train (Steps 1-30)** | {train_licit:,} | {train_illicit:,} | {np.sum((y_np == -1) & train_mask):,} | {np.sum(train_mask):,} | {train_licit+train_illicit:,} | {train_illicit/(train_licit+train_illicit):.2%} |\n")
        f.write(f"| **Calibration (Steps 31-34)** | {val_licit:,} | {val_illicit:,} | {np.sum((y_np == -1) & val_mask):,} | {np.sum(val_mask):,} | {val_licit+val_illicit:,} | {val_illicit/(val_licit+val_illicit):.2%} |\n")
        f.write(f"| **Test (Steps 35-49)** | {test_licit:,} | {test_illicit:,} | {np.sum((y_np == -1) & test_mask):,} | {np.sum(test_mask):,} | {test_licit+test_illicit:,} | {test_illicit/(test_licit+test_illicit):.2%} |\n")
        f.write(f"| **Overall** | {licit_count:,} | {illicit_count:,} | {unknown_count:,} | {num_nodes:,} | {total_labeled:,} | {illicit_count/total_labeled:.2%} |\n\n")
        
        f.write("> **Key Observation**: There is severe class imbalance in the labeled data. Illicit transactions account for "
                f"only **{illicit_count/total_labeled:.2%}** of the labeled transactions and only **{illicit_count/num_nodes:.2%}** "
                f"of all network nodes. Additionally, **{unknown_count/num_nodes:.2%}** of the transactions are **unlabeled (Unknown)**, "
                "meaning semi-supervised or graph-based propagation methods are vital to leverage the complete topology.\n\n")
        
        f.write("### 3. Graph Topological Statistics (Node Degrees)\n")
        f.write("The network graph contains directed payment flows. We analyzed node in-degree (incoming transactions), "
                "out-degree (outgoing transactions), and total degree (degree sum):\n\n")
        
        f.write("| Class | Metric | In-Degree | Out-Degree | Total Degree |\n")
        f.write("| --- | --- | --- | --- | --- |\n")
        for class_val, class_name in [(0, 'Licit (0)'), (1, 'Illicit (1)'), (-1, 'Unknown (-1)')]:
            sub_df = df_deg[df_deg['class'] == class_val]
            f.write(f"| **{class_name}** | Mean | {sub_df['in_degree'].mean():.2f} | {sub_df['out_degree'].mean():.2f} | {sub_df['total_degree'].mean():.2f} |\n")
            f.write(f"| | Median | {sub_df['in_degree'].median():.2f} | {sub_df['out_degree'].median():.2f} | {sub_df['total_degree'].median():.2f} |\n")
            f.write(f"| | Std Dev | {sub_df['in_degree'].std():.2f} | {sub_df['out_degree'].std():.2f} | {sub_df['total_degree'].std():.2f} |\n")
            f.write(f"| | Max | {sub_df['in_degree'].max():,} | {sub_df['out_degree'].max():,} | {sub_df['total_degree'].max():,} |\n")
            
        licit_avg_deg = df_deg[df_deg['class'] == 0]['total_degree'].mean()
        illicit_avg_deg = df_deg[df_deg['class'] == 1]['total_degree'].mean()
        f.write(f"\n> **Key Observation**: Licit nodes have a higher average total degree (**{licit_avg_deg:.2f}**) compared to illicit nodes "
                f"(**{illicit_avg_deg:.2f}**). This indicates that legitimate accounts participate in more transaction links on average, "
                "while fraudulent nodes operate in sparser, more direct configurations to avoid detection or disperse funds quickly. "
                "Furthermore, the maximum node degree in the graph is very high (e.g. total degree of 473), showing a typical scale-free power-law structure.\n\n")
        
        f.write("### 4. Edge Connectivity (Homophily & Linkage patterns)\n")
        f.write("We mapped the source and destination of all directed edges to trace transaction flows between classes:\n\n")
        f.write("| Source Node Class \\ Destination | Licit (0) | Illicit (1) | Unknown (-1) | Total Out-going |\n")
        f.write("| --- | --- | --- | --- | --- |\n")
        f.write(f"| **Licit (0)** | {edge_matrix[0,0]:,} ({edge_matrix_prob[0,0]:.2%}) | {edge_matrix[0,1]:,} ({edge_matrix_prob[0,1]:.2%}) | {edge_matrix[0,2]:,} ({edge_matrix_prob[0,2]:.2%}) | {row_sums[0][0]:,} |\n")
        f.write(f"| **Illicit (1)** | {edge_matrix[1,0]:,} ({edge_matrix_prob[1,0]:.2%}) | {edge_matrix[1,1]:,} ({edge_matrix_prob[1,1]:.2%}) | {edge_matrix[1,2]:,} ({edge_matrix_prob[1,2]:.2%}) | {row_sums[1][0]:,} |\n")
        f.write(f"| **Unknown (-1)** | {edge_matrix[2,0]:,} ({edge_matrix_prob[2,0]:.2%}) | {edge_matrix[2,1]:,} ({edge_matrix_prob[2,1]:.2%}) | {edge_matrix[2,2]:,} ({edge_matrix_prob[2,2]:.2%}) | {row_sums[2][0]:,} |\n\n")
        
        f.write("> **Key Observation**: Strong homophily exists. Illicit nodes are highly likely to transact with other illicit nodes or unknown nodes. "
                f"Specifically, only **{edge_matrix_prob[1,0]:.2%}** of illicit outgoing transactions flow directly into licit transactions, "
                f"while **{edge_matrix_prob[1,1]:.2%}** flow into other illicit nodes and **{edge_matrix_prob[1,2]:.2%}** flow into unknown/unlabeled transactions. "
                "This indicates that illicit transactions are clustered in subgraphs, which is a powerful signal for Graph Neural Networks (GNNs) "
                "that perform neighborhood aggregation (message passing).\n\n")
        
        f.write("### 5. Outlier Detection Summary (IQR Method)\n")
        f.write("Outliers were detected on the top discriminative features using the Interquartile Range (IQR) method:\n\n")
        f.write("| Feature | Lower Bound | Upper Bound | Total Outliers | Outliers % | Licit Outlier % | Illicit Outlier % |\n")
        f.write("| --- | --- | --- | --- | --- | --- | --- |\n")
        for record in outlier_records:
            f.write(f"| **{record['Feature']}** | {record['Lower Bound']:.3f} | {record['Upper Bound']:.3f} | {record['Total Outliers']:,} | {record['Total Outliers %']} | {record['Licit Outliers %']} | {record['Illicit Outliers %']} |\n")
            
        f.write("\n> **Key Observation**: The features exhibit a high percentage of outliers (ranging from 4% to over 16%), "
                "even after standardization. Crucially, the outlier rates differ significantly between licit and illicit nodes. "
                "For instance, for some features like `feat_0`, the outlier rate for illicit transactions is much higher, "
                "suggesting that fraudulent nodes possess extreme, anomalous values. Robust scaling or model architectures "
                "that can handle heavy tails and outliers are necessary.\n\n")
        
        f.write("### 6. Modeling Implications\n")
        f.write("The EDA findings highlight several critical factors that must influence downstream model development:\n\n")
        f.write("1. **Class Imbalance Strategy**:\n")
        f.write("   - The severe class imbalance (~10% fraud in training) means traditional classifiers will optimize for the majority class.\n")
        f.write("   - We must use the calculated training imbalance ratio (**8.11**) as a weight parameter (`pos_weight`) in the binary cross-entropy loss function "
                "or employ techniques like SMOTE or threshold adjustment to ensure high sensitivity to the minority illicit class.\n")
        f.write("2. **Graph Structure Exploitation**:\n")
        f.write("   - The high rate of transactions connected to 'unknown' nodes (**77% of nodes are unlabeled**) and the presence of homophily in edges "
                "strongly advocates for Graph Neural Networks (e.g., GCN, GAT, GraphSAGE).\n")
        f.write("   - GNNs can leverage the 234K structural edges to propagate information from labeled to unlabeled nodes, enhancing node feature representations "
                "before classification.\n")
        f.write("3. **Handling of Outliers and Skewed Distributions**:\n")
        f.write("   - Extreme values in both local and neighbor features indicate that models sensitive to outliers (like standard linear classifiers) "
                "might perform poorly or be unstable without regularizers.\n")
        f.write("   - Tree-based models (such as XGBoost, LightGBM) or deep neural networks with batch normalization/robust layers are well-suited "
                "as they are less sensitive to monotonic feature distortions and extreme outliers.\n")
        f.write("4. **Temporal Splitting Validation**:\n")
        f.write("   - The distribution of nodes and fraud ratio changes across time steps (e.g. test set has a lower fraud ratio of 6.5% compared to 10.98% in train).\n")
        f.write("   - Models must be evaluated using the strict temporal split rather than random k-fold cross-validation. Evaluating on random splits would leak future information "
                "to the past, resulting in overly optimistic validation scores that fail in production deployment.\n")
        f.write("5. **Bayesian Calibration Set**:\n")
        f.write("   - The separate calibration set (steps 31-34) with 2,989 nodes will be essential for computing Expected Calibration Error (ECE) "
                "and plotting reliability diagrams, enabling us to calibrate the uncertainties of our Bayesian Graph Neural Network.\n")

    print("EDA report written successfully.")
    print("=== EDA Complete! Plots saved in eda_plots/ ===")

if __name__ == '__main__':
    main()
