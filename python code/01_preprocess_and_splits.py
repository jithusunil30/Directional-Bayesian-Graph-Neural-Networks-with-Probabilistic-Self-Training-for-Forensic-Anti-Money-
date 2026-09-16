import os
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler

def main():
    print("=== Step 1: Load and Merge Data ===")
    classes_path = 'elliptic_bitcoin_dataset/elliptic_txs_classes.csv'
    edgelist_path = 'elliptic_bitcoin_dataset/elliptic_txs_edgelist.csv'
    features_path = 'elliptic_bitcoin_dataset/elliptic_txs_features.csv'

    print(f"Loading classes from {classes_path}...")
    df_classes = pd.read_csv(classes_path)
    print(f"Loaded classes: {df_classes.shape[0]} rows.")

    print(f"Loading edgelist from {edgelist_path}...")
    df_edgelist = pd.read_csv(edgelist_path)
    print(f"Loaded edgelist: {df_edgelist.shape[0]} rows.")

    print(f"Loading features from {features_path}...")
    # Features has no header. The first column is txId, second is time_step, next 165 are features
    df_features = pd.read_csv(features_path, header=None)
    print(f"Loaded features: {df_features.shape[0]} rows, {df_features.shape[1]} columns.")

    # Name the first two columns of features
    col_names = {0: 'txId', 1: 'time_step'}
    for i in range(2, df_features.shape[1]):
        col_names[i] = f'feat_{i-2}'
    df_features = df_features.rename(columns=col_names)

    # Ensure txId is integer type for merging
    df_features['txId'] = df_features['txId'].astype(np.int64)
    df_classes['txId'] = df_classes['txId'].astype(np.int64)

    print("Merging features and classes...")
    df_merged = pd.merge(df_features, df_classes, on='txId', how='inner')
    print(f"Merged dataframe shape: {df_merged.shape}")

    # Merge stats
    df_merge_stats = pd.DataFrame({
        'Metric': ['Features Row Count', 'Classes Row Count', 'Merged Row Count', 'Edgelist Row Count'],
        'Value': [df_features.shape[0], df_classes.shape[0], df_merged.shape[0], df_edgelist.shape[0]]
    })

    print("\n=== Step 2 & 3: Handle Unlabeled Nodes and Relabel Classes ===")
    # Class values in raw data: '1' (illicit), '2' (licit), 'unknown'
    # Convert '1' -> 1, '2' -> 0, 'unknown' -> -1
    df_merged['label'] = df_merged['class'].map({'1': 1, '2': 0, 'unknown': -1})
    
    unknown_count = (df_merged['label'] == -1).sum()
    licit_count = (df_merged['label'] == 0).sum()
    illicit_count = (df_merged['label'] == 1).sum()
    total_nodes = len(df_merged)

    print(f"Total nodes: {total_nodes}")
    print(f"Licit nodes (class 2 -> 0): {licit_count} ({licit_count/total_nodes:.2%})")
    print(f"Illicit nodes (class 1 -> 1): {illicit_count} ({illicit_count/total_nodes:.2%})")
    print(f"Unknown nodes (class unknown -> -1): {unknown_count} ({unknown_count/total_nodes:.2%})")

    df_node_handling = pd.DataFrame({
        'Metric': ['Total Nodes', 'Licit Nodes (0)', 'Illicit Nodes (1)', 'Unknown Nodes (-1)', 'Labeled Nodes'],
        'Count': [total_nodes, licit_count, illicit_count, unknown_count, licit_count + illicit_count],
        'Percentage': [1.0, licit_count/total_nodes, illicit_count/total_nodes, unknown_count/total_nodes, (licit_count + illicit_count)/total_nodes]
    })

    # Class counts before & after
    df_class_mapping = pd.DataFrame({
        'Original Class': ['1', '2', 'unknown'],
        'Mapped Label': [1, 0, -1],
        'Description': ['illicit/fraud', 'licit/legit', 'unlabeled/masked'],
        'Count': [illicit_count, licit_count, unknown_count]
    })

    print("\n=== Step 4: Split by Time Step (Train: 1-30, Val/Cal: 31-34, Test: 35-49) ===")
    # Elliptic has time_step values from 1 to 49
    # Let's count nodes per step
    step_counts = df_merged.groupby('time_step').size().reset_index(name='total_nodes')
    step_labeled = df_merged[df_merged['label'] != -1].groupby('time_step').size().reset_index(name='labeled_nodes')
    step_illicit = df_merged[df_merged['label'] == 1].groupby('time_step').size().reset_index(name='illicit_nodes')
    
    df_step_dist = pd.merge(step_counts, step_labeled, on='time_step', how='left').fillna(0)
    df_step_dist = pd.merge(df_step_dist, step_illicit, on='time_step', how='left').fillna(0)
    df_step_dist['labeled_nodes'] = df_step_dist['labeled_nodes'].astype(int)
    df_step_dist['illicit_nodes'] = df_step_dist['illicit_nodes'].astype(int)
    df_step_dist['fraud_ratio_labeled'] = df_step_dist['illicit_nodes'] / np.where(df_step_dist['labeled_nodes'] > 0, df_step_dist['labeled_nodes'], 1)
    
    # Add split information
    def get_split(step):
        if step <= 30:
            return 'Train'
        elif step <= 34:
            return 'Calibration'
        else:
            return 'Test'
    df_step_dist['split'] = df_step_dist['time_step'].apply(get_split)

    # Summary of splits
    train_nodes = df_merged[df_merged['time_step'] <= 30]
    val_nodes = df_merged[(df_merged['time_step'] >= 31) & (df_merged['time_step'] <= 34)]
    test_nodes = df_merged[df_merged['time_step'] >= 35]

    print(f"Train split (steps 1-30): {len(train_nodes)} nodes ({len(train_nodes[train_nodes['label'] != -1])} labeled)")
    print(f"Calibration split (steps 31-34): {len(val_nodes)} nodes ({len(val_nodes[val_nodes['label'] != -1])} labeled)")
    print(f"Test split (steps 35-49): {len(test_nodes)} nodes ({len(test_nodes[test_nodes['label'] != -1])} labeled)")

    df_splits_summary = pd.DataFrame({
        'Split': ['Train (1-30)', 'Calibration (31-34)', 'Test (35-49)', 'Total'],
        'Total Nodes': [len(train_nodes), len(val_nodes), len(test_nodes), len(df_merged)],
        'Labeled Nodes': [
            (train_nodes['label'] != -1).sum(),
            (val_nodes['label'] != -1).sum(),
            (test_nodes['label'] != -1).sum(),
            (df_merged['label'] != -1).sum()
        ],
        'Illicit Nodes (1)': [
            (train_nodes['label'] == 1).sum(),
            (val_nodes['label'] == 1).sum(),
            (test_nodes['label'] == 1).sum(),
            (df_merged['label'] == 1).sum()
        ],
        'Licit Nodes (0)': [
            (train_nodes['label'] == 0).sum(),
            (val_nodes['label'] == 0).sum(),
            (test_nodes['label'] == 0).sum(),
            (df_merged['label'] == 0).sum()
        ]
    })
    df_splits_summary['Fraud Ratio (Labeled)'] = df_splits_summary['Illicit Nodes (1)'] / df_splits_summary['Labeled Nodes']

    print("\n=== Step 5: Feature Scaling ===")
    # Extract feature columns (columns 2 to 166, which are feat_0 to feat_164)
    feat_cols = [f'feat_{i}' for i in range(165)]
    
    # 94 local features (feat_0 to feat_93)
    local_cols = feat_cols[:94]
    # 71 neighbor aggregated features (feat_94 to feat_164)
    neighbor_cols = feat_cols[94:]

    print(f"Local features count: {len(local_cols)}")
    print(f"Neighbor features count: {len(neighbor_cols)}")

    # We fit scaler on the training set features (excluding time_step and txId)
    scaler = StandardScaler()
    print("Fitting StandardScaler on Train set features...")
    scaler.fit(train_nodes[feat_cols])

    # Transform all splits
    print("Scaling features across all splits...")
    df_scaled = df_merged.copy()
    df_scaled[feat_cols] = scaler.transform(df_merged[feat_cols])

    # Compute pre/post scaling statistics for a subset of features to log/verify
    # We will pick feat_0, feat_50, feat_100, feat_150
    test_feats = ['feat_0', 'feat_50', 'feat_100', 'feat_150']
    scaling_stats_list = []
    for f in test_feats:
        scaling_stats_list.append({
            'Feature': f,
            'Train Mean (Pre)': df_merged.loc[df_merged['time_step'] <= 30, f].mean(),
            'Train Std (Pre)': df_merged.loc[df_merged['time_step'] <= 30, f].std(),
            'Train Mean (Post)': df_scaled.loc[df_scaled['time_step'] <= 30, f].mean(),
            'Train Std (Post)': df_scaled.loc[df_scaled['time_step'] <= 30, f].std(),
            'Test Mean (Post)': df_scaled.loc[df_scaled['time_step'] >= 35, f].mean(),
            'Test Std (Post)': df_scaled.loc[df_scaled['time_step'] >= 35, f].std()
        })
    df_scaling_stats = pd.DataFrame(scaling_stats_list)

    print("\n=== Step 6: PyTorch Geometric Graph Object Construction ===")
    # Create mapping from txId -> contiguous node index
    print("Creating txId to index mapping...")
    txId_to_idx = {txId: idx for idx, txId in enumerate(df_scaled['txId'].values)}
    
    # Map edges
    print("Mapping edgelist to edge_index tensor...")
    src_list = []
    dst_list = []
    dropped_edges = 0
    for s, d in zip(df_edgelist['txId1'], df_edgelist['txId2']):
        if s in txId_to_idx and d in txId_to_idx:
            src_list.append(txId_to_idx[s])
            dst_list.append(txId_to_idx[d])
        else:
            dropped_edges += 1
            
    print(f"Mapped {len(src_list)} edges. Dropped {dropped_edges} edges because of missing nodes in features.")

    # Convert to edge_index tensor (shape 2 x num_edges)
    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    
    # Feature matrix X
    x = torch.tensor(df_scaled[feat_cols].values, dtype=torch.float)
    
    # Target label Y
    y = torch.tensor(df_scaled['label'].values, dtype=torch.long)

    # Time steps
    time_steps_tensor = torch.tensor(df_scaled['time_step'].values, dtype=torch.long)
    
    # Masks
    train_mask = torch.tensor((df_scaled['time_step'] <= 30) & (df_scaled['label'] != -1), dtype=torch.bool)
    val_mask = torch.tensor((df_scaled['time_step'] >= 31) & (df_scaled['time_step'] <= 34) & (df_scaled['label'] != -1), dtype=torch.bool)
    test_mask = torch.tensor((df_scaled['time_step'] >= 35) & (df_scaled['label'] != -1), dtype=torch.bool)

    # PyG Data Object
    data = Data(x=x, edge_index=edge_index, y=y)
    data.time_step = time_steps_tensor
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    print("PyTorch Geometric Data Object created:")
    print(data)
    print(f"Number of nodes: {data.num_nodes}")
    print(f"Number of edges: {data.num_edges}")
    print(f"Number of training nodes (labeled): {data.train_mask.sum().item()}")
    print(f"Number of validation/calibration nodes (labeled): {data.val_mask.sum().item()}")
    print(f"Number of testing nodes (labeled): {data.test_mask.sum().item()}")

    # Save the PyG Data object
    pyg_output_path = 'elliptic_pyg_data.pt'
    print(f"Saving PyG data object to {pyg_output_path}...")
    torch.save(data, pyg_output_path)
    print("Successfully saved PyG data object.")

    # Graph stats for Excel
    avg_degree = data.num_edges / data.num_nodes
    # self loops check
    self_loops = (edge_index[0] == edge_index[1]).sum().item()
    df_graph_stats = pd.DataFrame({
        'Metric': ['Node Count', 'Edge Count', 'Average Node Degree', 'Self-loops Count', 'Isolated Nodes (approx)'],
        'Value': [data.num_nodes, data.num_edges, avg_degree, self_loops, data.num_nodes - len(set(src_list + dst_list))]
    })

    print("\n=== Step 7 & 8: Class Imbalance and Calibration Set Details ===")
    train_licit = (df_scaled.loc[df_scaled['time_step'] <= 30, 'label'] == 0).sum()
    train_illicit = (df_scaled.loc[df_scaled['time_step'] <= 30, 'label'] == 1).sum()
    imbalance_ratio = train_licit / train_illicit
    pos_weight = imbalance_ratio # weight for BCEWithLogitsLoss pos_weight

    print(f"Train Set Licit Nodes: {train_licit}")
    print(f"Train Set Illicit Nodes: {train_illicit}")
    print(f"Train Set Imbalance Ratio (Licit / Illicit): {imbalance_ratio:.2f}")
    print(f"Suggested Pos Weight for BCE Loss: {pos_weight:.4f}")

    df_imbalance = pd.DataFrame({
        'Metric': ['Train Licit Nodes', 'Train Illicit Nodes', 'Licit / Illicit Ratio', 'Suggested BCE Pos Weight'],
        'Value': [train_licit, train_illicit, imbalance_ratio, pos_weight]
    })

    # Calibration split stats
    cal_licit = (df_scaled.loc[(df_scaled['time_step'] >= 31) & (df_scaled['time_step'] <= 34), 'label'] == 0).sum()
    cal_illicit = (df_scaled.loc[(df_scaled['time_step'] >= 31) & (df_scaled['time_step'] <= 34), 'label'] == 1).sum()
    cal_total = len(val_nodes)
    cal_labeled = cal_licit + cal_illicit

    df_calibration = pd.DataFrame({
        'Metric': ['Calibration Total Nodes (steps 31-34)', 'Calibration Labeled Nodes', 'Calibration Licit (0)', 'Calibration Illicit (1)', 'Calibration Fraud Ratio (among labeled)'],
        'Value': [cal_total, cal_labeled, cal_licit, cal_illicit, cal_illicit / cal_labeled if cal_labeled > 0 else 0]
    })

    print("\n=== Step 9: Sanity Checks ===")
    # Sanity check 1: Verify no edges connect nodes across different time steps
    src_times = time_steps_tensor[edge_index[0]]
    dst_times = time_steps_tensor[edge_index[1]]
    cross_time_edges = (src_times != dst_times).sum().item()
    
    print(f"Number of cross-timestep edges: {cross_time_edges}")
    sc1_status = "PASS" if cross_time_edges == 0 else "FAIL"
    print(f"Sanity Check 1 (No cross-time-step edges): {sc1_status}")

    # Sanity check 2: Verify fraud ratios in train, val, and test splits are around ~2% of total nodes, or ~9.6% of labeled nodes
    train_fraud_ratio = train_illicit / (train_licit + train_illicit)
    cal_fraud_ratio = cal_illicit / (cal_licit + cal_illicit)
    
    test_licit = (df_scaled.loc[df_scaled['time_step'] >= 35, 'label'] == 0).sum()
    test_illicit = (df_scaled.loc[df_scaled['time_step'] >= 35, 'label'] == 1).sum()
    test_fraud_ratio = test_illicit / (test_licit + test_illicit)

    print(f"Train labeled fraud ratio: {train_fraud_ratio:.2%}")
    print(f"Calibration labeled fraud ratio: {cal_fraud_ratio:.2%}")
    print(f"Test labeled fraud ratio: {test_fraud_ratio:.2%}")

    # Total nodes fraud ratio (including unknown)
    train_total_fraud_ratio = train_illicit / len(train_nodes)
    cal_total_fraud_ratio = cal_illicit / len(val_nodes)
    test_total_fraud_ratio = test_illicit / len(test_nodes)

    print(f"Train total fraud ratio (incl. unknown): {train_total_fraud_ratio:.2%}")
    print(f"Calibration total fraud ratio (incl. unknown): {cal_total_fraud_ratio:.2%}")
    print(f"Test total fraud ratio (incl. unknown): {test_total_fraud_ratio:.2%}")

    # Standard fraud ratio for Elliptic is ~9-10% of labeled nodes, or ~2% of all nodes. Let's document both
    sc2_status = "PASS" if abs(train_fraud_ratio - 0.09) < 0.05 and abs(test_fraud_ratio - 0.06) < 0.05 else "WARNING"
    print(f"Sanity Check 2 (Fraud ratios match literature): {sc2_status}")

    df_sanity_checks = pd.DataFrame({
        'Sanity Check': [
            '1. No cross-timestep edges',
            '2. Train Labeled Fraud Ratio',
            '2. Calibration Labeled Fraud Ratio',
            '2. Test Labeled Fraud Ratio',
            '2. Train Total Fraud Ratio (incl. unknown)',
            '2. Calibration Total Fraud Ratio (incl. unknown)',
            '2. Test Total Fraud Ratio (incl. unknown)'
        ],
        'Expected / Constraint': [
            '0 cross-time edges',
            '~9-10% of labeled (or ~2% of total)',
            '~9-10% of labeled (or ~2% of total)',
            '~6-10% of labeled (or ~2% of total)',
            '~2% of total nodes',
            '~2% of total nodes',
            '~2% of total nodes'
        ],
        'Actual Value': [
            f"{cross_time_edges} edges",
            f"{train_fraud_ratio:.2%}",
            f"{cal_fraud_ratio:.2%}",
            f"{test_fraud_ratio:.2%}",
            f"{train_total_fraud_ratio:.2%}",
            f"{cal_total_fraud_ratio:.2%}",
            f"{test_total_fraud_ratio:.2%}"
        ],
        'Status': [
            sc1_status,
            "PASS" if (0.05 <= train_fraud_ratio <= 0.15) else "CHECK",
            "PASS" if (0.05 <= cal_fraud_ratio <= 0.15) else "CHECK",
            "PASS" if (0.03 <= test_fraud_ratio <= 0.15) else "CHECK",
            "PASS" if (0.01 <= train_total_fraud_ratio <= 0.03) else "CHECK",
            "PASS" if (0.01 <= cal_total_fraud_ratio <= 0.03) else "CHECK",
            "PASS" if (0.01 <= test_total_fraud_ratio <= 0.03) else "CHECK"
        ]
    })

    print("\n=== Exporting Preprocessing Report Workbook ===")
    report_path = 'preprocessing_report.xlsx'
    with pd.ExcelWriter(report_path, engine='openpyxl') as writer:
        df_merge_stats.to_excel(writer, sheet_name='1_Merge_Stats', index=False)
        df_node_handling.to_excel(writer, sheet_name='2_Node_Handling', index=False)
        df_class_mapping.to_excel(writer, sheet_name='3_Class_Mapping', index=False)
        df_step_dist.to_excel(writer, sheet_name='4_Temporal_Splits', index=False)
        df_scaling_stats.to_excel(writer, sheet_name='5_Feature_Scaling_Stats', index=False)
        df_graph_stats.to_excel(writer, sheet_name='6_Graph_Structure', index=False)
        df_imbalance.to_excel(writer, sheet_name='7_Imbalance_Handling', index=False)
        df_calibration.to_excel(writer, sheet_name='8_Calibration_Set', index=False)
        df_sanity_checks.to_excel(writer, sheet_name='9_Sanity_Checks', index=False)
    print(f"Report Excel saved at: {report_path}")

    # Now let's save the labeled dataset in a separate file (it has 46,564 nodes).
    # Having the full features scaled inside an Excel workbook for labeled nodes is very helpful.
    print("Preparing labeled node features spreadsheet (only labeled nodes)...")
    df_labeled = df_scaled[df_scaled['label'] != -1].copy()
    
    # We will export the txId, time_step, label and all features of the labeled nodes to Excel.
    # To prevent excel file being too large, let's export a clean subset of columns first or the whole labeled.
    # 46k rows by 168 columns is about 25MB. We'll save it to `preprocessed_nodes_labeled.xlsx`.
    labeled_output_path = 'preprocessed_nodes_labeled.xlsx'
    print(f"Saving preprocessed labeled node features ({df_labeled.shape[0]} nodes) to {labeled_output_path}...")
    
    # Select columns to save: txId, time_step, class, label, feat_0, feat_1, ...
    cols_to_save = ['txId', 'time_step', 'class', 'label'] + feat_cols
    df_labeled[cols_to_save].to_excel(labeled_output_path, sheet_name='Preprocessed_Labeled', index=False)
    print(f"Labeled features Excel saved at: {labeled_output_path}")
    
    print("\nPreprocessing finished successfully!")

if __name__ == '__main__':
    main()
