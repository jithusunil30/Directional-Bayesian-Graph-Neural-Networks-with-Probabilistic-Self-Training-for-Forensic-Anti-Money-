import os
import torch
import pandas as pd
import numpy as np

def convert_pt_to_csv(pt_path, output_dir):
    print(f"Loading {pt_path}...")
    try:
        data = torch.load(pt_path, weights_only=False)
    except Exception as e:
        print(f"Error loading {pt_path}: {e}")
        return

    print("PyG Data object loaded successfully.")
    os.makedirs(output_dir, exist_ok=True)

    # ---------------------------------------------------------
    # 1. Convert Nodes to DataFrame (elliptic_nodes_readable.csv)
    # ---------------------------------------------------------
    print("Preparing node features and metadata...")
    # Convert tensors to numpy arrays
    x_np = data.x.numpy()
    y_np = data.y.numpy()
    time_step_np = data.time_step.numpy()
    train_mask_np = data.train_mask.numpy().astype(int)
    val_mask_np = data.val_mask.numpy().astype(int)
    test_mask_np = data.test_mask.numpy().astype(int)

    num_nodes = x_np.shape[0]
    num_features = x_np.shape[1]

    # Create columns list
    columns = ['node_idx', 'time_step', 'label', 'train_mask', 'val_mask', 'test_mask']
    features_cols = [f'feat_{i}' for i in range(num_features)]
    all_columns = columns + features_cols

    # Build DataFrame rows
    # We want to also map contiguous node indices back to transaction IDs if possible
    # We can retrieve the original transaction IDs from the spreadsheet preprocessed_nodes_labeled.xlsx
    # or by keeping the order of df_scaled in preprocess.py
    # Since they are in the exact same row order as df_scaled from preprocess.py, we can map them back!
    # Let's check if the excel or features CSV is available
    features_csv = 'elliptic_bitcoin_dataset/elliptic_txs_features.csv'
    if not os.path.exists(features_csv) and os.path.exists('../elliptic_bitcoin_dataset/elliptic_txs_features.csv'):
        features_csv = '../elliptic_bitcoin_dataset/elliptic_txs_features.csv'

    txIds = None
    if os.path.exists(features_csv):
        print("Reading transaction IDs from raw features file to include original txIds...")
        # Only read the first column which is txId
        df_txs = pd.read_csv(features_csv, header=None, usecols=[0])
        txIds = df_txs[0].values
    else:
        print("Raw features file not found, txId column will not be available in node CSV (node_idx will be used instead).")

    # Construct dataframe
    data_dict = {
        'node_idx': np.arange(num_nodes),
        'time_step': time_step_np,
        'label': y_np,
        'train_mask': train_mask_np,
        'val_mask': val_mask_np,
        'test_mask': test_mask_np
    }
    
    if txIds is not None:
        data_dict['txId'] = txIds
        # Move txId to be the first column
        all_columns = ['node_idx', 'txId'] + columns[1:] + features_cols

    # Add features to the dictionary
    for i in range(num_features):
        data_dict[f'feat_{i}'] = x_np[:, i]

    df_nodes = pd.DataFrame(data_dict, columns=all_columns)
    
    nodes_csv_path = os.path.join(output_dir, 'elliptic_nodes_readable.csv')
    print(f"Saving nodes to {nodes_csv_path} (~300MB)...")
    df_nodes.to_csv(nodes_csv_path, index=False)
    print("Nodes CSV saved.")

    # ---------------------------------------------------------
    # 2. Convert Edges to DataFrame (elliptic_edges_readable.csv)
    # ---------------------------------------------------------
    print("Preparing edge connections list...")
    edge_index_np = data.edge_index.numpy()
    src_indices = edge_index_np[0]
    dst_indices = edge_index_np[1]

    edges_dict = {
        'source_node_idx': src_indices,
        'destination_node_idx': dst_indices
    }

    if txIds is not None:
        edges_dict['source_txId'] = txIds[src_indices]
        edges_dict['destination_txId'] = txIds[dst_indices]
        edge_columns = ['source_node_idx', 'destination_node_idx', 'source_txId', 'destination_txId']
    else:
        edge_columns = ['source_node_idx', 'destination_node_idx']

    df_edges = pd.DataFrame(edges_dict, columns=edge_columns)

    edges_csv_path = os.path.join(output_dir, 'elliptic_edges_readable.csv')
    print(f"Saving edges to {edges_csv_path} (~10MB)...")
    df_edges.to_csv(edges_csv_path, index=False)
    print("Edges CSV saved.")

    print("\nConversion successfully completed! Readable CSV files generated in: ", output_dir)
if __name__ == '__main__':
    pt_file = 'elliptic_pyg_data.pt'
    if not os.path.exists(pt_file) and os.path.exists('updated/elliptic_pyg_data.pt'):
        pt_file = 'updated/elliptic_pyg_data.pt'
    output_dir = '.'
    convert_pt_to_csv(pt_file, output_dir)
