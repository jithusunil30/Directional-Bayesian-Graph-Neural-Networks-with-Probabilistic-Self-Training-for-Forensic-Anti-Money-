import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils import load_data

def main():
    print("=== Calculating Edge-Type Connectivity Matrices (Old vs New) ===", flush=True)
    
    # 1. Load Data
    data, _, _, _, _, _, _ = load_data()
    edge_index = data.edge_index.numpy()
    src, dst = edge_index[0], edge_index[1]
    total_edges = edge_index.shape[1]
    
    y_old = data.y.numpy() # Original labels: 0 (licit), 1 (illicit), -1 (unknown)
    
    # ----------------------------------------------------
    # 1. OLD EDGE CONNECTIVITY MATRIX (3x3: Licit, Illicit, Unknown)
    # ----------------------------------------------------
    # Map -1 to index 2 for matrix representation
    label_map_old = {0: 'Licit (0)', 1: 'Illicit (1)', -1: 'Unknown (-1)'}
    src_labels_old = [label_map_old[y] for y in y_old[src]]
    dst_labels_old = [label_map_old[y] for y in y_old[dst]]
    
    cats_old = ['Licit (0)', 'Illicit (1)', 'Unknown (-1)']
    df_edges_old = pd.crosstab(
        pd.Series(src_labels_old, name='Source (From)'),
        pd.Series(dst_labels_old, name='Target (To)')
    ).reindex(index=cats_old, columns=cats_old, fill_value=0)
    
    print("\n--- OLD EDGE CONNECTIVITY MATRIX (Original Data with Unknowns) ---", flush=True)
    print(df_edges_old, flush=True)
    print("\nPercentage Matrix (% of total 234,355 edges):", flush=True)
    print((df_edges_old / total_edges * 100).round(2), flush=True)

    # ----------------------------------------------------
    # 2. LOAD PREDICTED UNKNOWN LABELS & CONSTRUCT NEW LABELS
    # ----------------------------------------------------
    csv_paths = [
        'predicted_unknown_labels_gat.csv',
        'src/predicted_unknown_labels_gat.csv',
        os.path.join(os.path.dirname(__file__), 'predicted_unknown_labels_gat.csv'),
        os.path.join('dataset', 'src', 'predicted_unknown_labels_gat.csv')
    ]
    csv_path = None
    for p in csv_paths:
        if os.path.exists(p):
            csv_path = p
            break
            
    if csv_path is None:
        raise FileNotFoundError(f"Could not find predicted_unknown_labels_gat.csv in any of: {csv_paths}")
        
    print(f"\nLoading GAT predicted labels for unknown nodes from {csv_path}...", flush=True)
    df_unknown_preds = pd.read_csv(csv_path)
    
    # Create complete new label array y_new (all 203,769 nodes now have labels 0 or 1)
    y_new = y_old.copy()
    unknown_indices = df_unknown_preds['node_index'].values
    unknown_predicted_labels = df_unknown_preds['predicted_label'].values
    
    y_new[unknown_indices] = unknown_predicted_labels
    
    # Check that no node has label -1
    print(f"Total nodes with -1 label remaining: {(y_new == -1).sum()}", flush=True)
    print(f"New Total Licit Nodes (0): {(y_new == 0).sum()} ({(y_new == 0).mean():.2%})", flush=True)
    print(f"New Total Illicit Nodes (1): {(y_new == 1).sum()} ({(y_new == 1).mean():.2%})", flush=True)

    # ----------------------------------------------------
    # 3. NEW EDGE-TYPE CONNECTIVITY MATRIX (2x2: Licit, Illicit)
    # ----------------------------------------------------
    label_map_new = {0: 'Licit (0)', 1: 'Illicit (1)'}
    src_labels_new = [label_map_new[y] for y in y_new[src]]
    dst_labels_new = [label_map_new[y] for y in y_new[dst]]
    
    cats_new = ['Licit (0)', 'Illicit (1)']
    df_edges_new = pd.crosstab(
        pd.Series(src_labels_new, name='Source (From)'),
        pd.Series(dst_labels_new, name='Target (To)')
    ).reindex(index=cats_new, columns=cats_new, fill_value=0)

    df_edges_new_pct = (df_edges_new / total_edges * 100).round(2)

    print("\n--- NEW EDGE-TYPE CONNECTIVITY MATRIX (After GAT Labeling) ---", flush=True)
    print("Edge Counts:")
    print(df_edges_new, flush=True)
    print("\nPercentage Breakdown (% of Total Edges):")
    print(df_edges_new_pct, flush=True)

    # ----------------------------------------------------
    # 4. PLOT SIDE-BY-SIDE CONNECTIVITY HEATMAPS
    # ----------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
    
    # Plot 1: Old Edge Connectivity (3x3)
    ax1 = axes[0]
    sns.heatmap(df_edges_old, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax1,
                annot_kws={"size": 11, "weight": "bold"})
    ax1.set_title("OLD Edge Connectivity Matrix\n(Including 157,205 Unknown Nodes)", fontsize=13, fontweight='bold', pad=12)
    ax1.set_xlabel("Target Node (To)", fontsize=11, fontweight='bold')
    ax1.set_ylabel("Source Node (From)", fontsize=11, fontweight='bold')
    
    # Annotate percentage inside cells
    for i in range(3):
        for j in range(3):
            val = df_edges_old.iloc[i, j]
            pct = val / total_edges * 100
            ax1.text(j + 0.5, i + 0.7, f"({pct:.1f}%)", ha='center', va='center', color='gray' if val > df_edges_old.values.max()/2 else 'blue', fontsize=9)

    # Plot 2: New Edge Connectivity (2x2)
    ax2 = axes[1]
    sns.heatmap(df_edges_new, annot=True, fmt='d', cmap='Greens', cbar=False, ax=ax2,
                annot_kws={"size": 14, "weight": "bold"})
    ax2.set_title("NEW Edge-Type Connectivity Matrix\n(After GAT Labeling All Unknown Nodes)", fontsize=13, fontweight='bold', pad=12)
    ax2.set_xlabel("Target Node (To)", fontsize=11, fontweight='bold')
    ax2.set_ylabel("Source Node (From)", fontsize=11, fontweight='bold')
    
    for i in range(2):
        for j in range(2):
            val = df_edges_new.iloc[i, j]
            pct = val / total_edges * 100
            ax2.text(j + 0.5, i + 0.7, f"({pct:.1f}%)", ha='center', va='center', color='white' if val > df_edges_new.values.max()/2 else 'darkgreen', fontsize=11, weight='bold')

    plt.suptitle("Elliptic Bitcoin Graph: Edge-Type Connectivity Matrix Transformation", fontsize=15, fontweight='bold', y=1.03)
    plt.tight_layout()
    
    save_path = os.path.join('results', 'edge_type_connectivity_matrix.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"\nSaved Edge Connectivity Matrix plot to: {os.path.abspath(save_path)}", flush=True)

if __name__ == '__main__':
    main()
