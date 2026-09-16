import os
import sys
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler

def run_preprocessing(raw_data_dir=None, output_dir=None):
    if raw_data_dir is None:
        # Default relative to this script
        base_dir = os.path.dirname(os.path.abspath(__file__))
        raw_data_dir = os.path.join(base_dir, '..', 'dataset', 'elliptic_bitcoin_dataset')
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(__file__))

    os.makedirs(output_dir, exist_ok=True)
    print("=" * 70)
    print("   ELLIPTIC BITCOIN GRAPH DATASET: PREPROCESSING & GRAPH CONSTRUCTION   ")
    print("=" * 70)

    pyg_path = os.path.join(output_dir, 'elliptic_pyg_data.pt')
    if os.path.exists(pyg_path):
        print(f"\n[Info] Preprocessed PyG data file already exists at: {pyg_path}")
        print("Loading existing preprocessed graph object directly...")
        data = torch.load(pyg_path, weights_only=False)
        return data, None

    classes_path = os.path.join(raw_data_dir, 'elliptic_txs_classes.csv')
    edgelist_path = os.path.join(raw_data_dir, 'elliptic_txs_edgelist.csv')
    features_path = os.path.join(raw_data_dir, 'elliptic_txs_features.csv')

    print(f"\n[1/5] Loading Raw Files from: {raw_data_dir}")
    print(f" -> Loading classes: {classes_path}")
    df_classes = pd.read_csv(classes_path)
    print(f"    Classes loaded: {len(df_classes):,} rows.")

    print(f" -> Loading edgelist: {edgelist_path}")
    df_edgelist = pd.read_csv(edgelist_path)
    print(f"    Edgelist loaded: {len(df_edgelist):,} edges.")

    print(f" -> Loading features (headerless): {features_path}")
    df_features = pd.read_csv(features_path, header=None)
    print(f"    Features loaded: {df_features.shape[0]:,} rows x {df_features.shape[1]} columns.")

    # Assign column names: 0 -> txId, 1 -> time_step, 2..166 -> feat_0..feat_164
    col_names = {0: 'txId', 1: 'time_step'}
    for i in range(2, df_features.shape[1]):
        col_names[i] = f'feat_{i-2}'
    df_features = df_features.rename(columns=col_names)

    df_features['txId'] = df_features['txId'].astype(np.int64)
    df_classes['txId'] = df_classes['txId'].astype(np.int64)

    print("\n[2/5] Merging Features with Ground Truth Classes...")
    df_merged = pd.merge(df_features, df_classes, on='txId', how='inner')
    print(f"    Merged Dataset Shape: {df_merged.shape[0]:,} nodes x {df_merged.shape[1]} columns.")

    # Relabeling: '1' -> 1 (illicit), '2' -> 0 (licit), 'unknown' -> -1 (unlabeled)
    df_merged['label'] = df_merged['class'].map({'1': 1, '2': 0, 'unknown': -1})
    
    total_nodes = len(df_merged)
    licit_count = int((df_merged['label'] == 0).sum())
    illicit_count = int((df_merged['label'] == 1).sum())
    unknown_count = int((df_merged['label'] == -1).sum())
    labeled_count = licit_count + illicit_count

    print(f"\n[3/5] Class Distribution Breakdown:")
    print(f" -> Total Nodes:     {total_nodes:,} (100.0%)")
    print(f" -> Licit (0):       {licit_count:,} ({licit_count/total_nodes*100:.2f}%)")
    print(f" -> Illicit (1):     {illicit_count:,} ({illicit_count/total_nodes*100:.2f}%)")
    print(f" -> Unknown (-1):    {unknown_count:,} ({unknown_count/total_nodes*100:.2f}%)")
    print(f" -> Labeled Total:   {labeled_count:,} ({labeled_count/total_nodes*100:.2f}%)")
    print(f" -> Class Imbalance: ~{licit_count/illicit_count:.1f}:1 (Licit:Illicit)")

    # Temporal Splitting
    # Train: timesteps 1-30, Val/Cal: 31-34, Test: 35-49
    train_mask_np = (df_merged['time_step'] <= 30) & (df_merged['label'] != -1)
    val_mask_np = (df_merged['time_step'] >= 31) & (df_merged['time_step'] <= 34) & (df_merged['label'] != -1)
    test_mask_np = (df_merged['time_step'] >= 35) & (df_merged['label'] != -1)
    unlabeled_mask_np = (df_merged['label'] == -1)

    print(f"\n[4/5] Temporal Partitioning & Split Counts:")
    print(f" -> Train Set (t=1..30):       {train_mask_np.sum():,} labeled nodes "
          f"(Illicit: {(df_merged.loc[train_mask_np, 'label'] == 1).sum():,}, "
          f"Licit: {(df_merged.loc[train_mask_np, 'label'] == 0).sum():,})")
    print(f" -> Val/Cal Set (t=31..34):    {val_mask_np.sum():,} labeled nodes "
          f"(Illicit: {(df_merged.loc[val_mask_np, 'label'] == 1).sum():,}, "
          f"Licit: {(df_merged.loc[val_mask_np, 'label'] == 0).sum():,})")
    print(f" -> Test Set (t=35..49):       {test_mask_np.sum():,} labeled nodes "
          f"(Illicit: {(df_merged.loc[test_mask_np, 'label'] == 1).sum():,}, "
          f"Licit: {(df_merged.loc[test_mask_np, 'label'] == 0).sum():,})")
    print(f" -> Unlabeled Nodes (t=1..49): {unlabeled_mask_np.sum():,} nodes")

    # Feature Scaling (Fit on Train, transform all)
    feat_cols = [f'feat_{i}' for i in range(165)]
    # All 166 features = local (0-93) + neighbor aggregated (94-164) + time_step or raw 165 features
    scaler = StandardScaler()
    print("\n[5/5] Standardizing Features and Assembling PyG Graph Object...")
    # Standardize on all rows or train rows
    train_features = df_merged.loc[df_merged['time_step'] <= 30, feat_cols]
    scaler.fit(train_features)
    scaled_feats = scaler.transform(df_merged[feat_cols])

    # Node ID to 0-indexed integer mapping
    df_merged = df_merged.reset_index(drop=True)
    tx_to_idx = {tx_id: idx for idx, tx_id in enumerate(df_merged['txId'].values)}

    # Map edge list
    valid_edges = df_edgelist[df_edgelist['txId1'].isin(tx_to_idx) & df_edgelist['txId2'].isin(tx_to_idx)]
    src = [tx_to_idx[t1] for t1 in valid_edges['txId1'].values]
    dst = [tx_to_idx[t2] for t2 in valid_edges['txId2'].values]
    edge_index = torch.tensor([src, dst], dtype=torch.long)

    # PyG Data object
    x_tensor = torch.tensor(scaled_feats, dtype=torch.float32)
    y_tensor = torch.tensor(df_merged['label'].values, dtype=torch.float32)
    time_tensor = torch.tensor(df_merged['time_step'].values, dtype=torch.long)
    txid_tensor = torch.tensor(df_merged['txId'].values, dtype=torch.int64)

    data = Data(
        x=x_tensor,
        edge_index=edge_index,
        y=y_tensor,
        time_step=time_tensor,
        txId=txid_tensor,
        train_mask=torch.tensor(train_mask_np.values, dtype=torch.bool),
        val_mask=torch.tensor(val_mask_np.values, dtype=torch.bool),
        test_mask=torch.tensor(test_mask_np.values, dtype=torch.bool),
        unlabeled_mask=torch.tensor(unlabeled_mask_np.values, dtype=torch.bool)
    )

    pyg_path = os.path.join(output_dir, 'elliptic_pyg_data.pt')
    torch.save(data, pyg_path)
    print(f" -> PyG Data saved to: {pyg_path}")
    print(f"    Graph summary: {data.num_nodes:,} nodes, {data.num_edges:,} directed edges, {data.num_node_features} features.")

    # Save summary stats to Excel
    summary_path = os.path.join(output_dir, 'preprocessing_summary.xlsx')
    with pd.ExcelWriter(summary_path, engine='openpyxl') as writer:
        df_summary = pd.DataFrame({
            'Metric': [
                'Total Transaction Nodes', 'Total Directed Edges', 'Features per Node',
                'Licit Transactions (0)', 'Illicit Transactions (1)', 'Unlabeled Transactions (-1)',
                'Train Labeled (t=1..30)', 'Val Labeled (t=31..34)', 'Test Labeled (t=35..49)',
                'Class Imbalance Ratio (Licit:Illicit)'
            ],
            'Value': [
                total_nodes, len(src), len(feat_cols),
                licit_count, illicit_count, unknown_count,
                int(train_mask_np.sum()), int(val_mask_np.sum()), int(test_mask_np.sum()),
                f"{licit_count/illicit_count:.2f}:1"
            ]
        })
        df_summary.to_excel(writer, sheet_name='Dataset_Overview', index=False)

    print(f" -> Preprocessing summary saved to: {summary_path}")
    print("=" * 70)
    print("   PREPROCESSING COMPLETED SUCCESSFULLY!   ")
    print("=" * 70)
    return data, df_merged

if __name__ == '__main__':
    run_preprocessing()
