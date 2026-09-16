import os
import sys
import torch
import numpy as np
import pandas as pd
from torch_geometric.data import Data

# Import load_data from src/utils.py
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from utils import load_data

def main():
    print("=== Creating Fully Labeled Updated Dataset ===", flush=True)
    
    # 1. Load Original PyG Graph Data
    data, _, _, _, _, _, _ = load_data()
    num_nodes = data.x.shape[0]
    num_edges = data.edge_index.shape[1]
    num_features = data.x.shape[1]
    
    # 2. Load GAT Predictions for Unknown Nodes
    csv_path = 'src/predicted_unknown_labels_gat.csv'
    if not os.path.exists(csv_path) and os.path.exists('predicted_unknown_labels_gat.csv'):
        csv_path = 'predicted_unknown_labels_gat.csv'
        
    print(f"Loading GAT predicted labels from: {csv_path}", flush=True)
    df_gat_preds = pd.read_csv(csv_path)
    
    # 3. Build Updated Target Array (y_updated)
    y_orig = data.y.numpy().copy()
    y_updated = y_orig.copy()
    
    unknown_mask = (y_orig == -1)
    unknown_indices = df_gat_preds['node_index'].values
    unknown_preds = df_gat_preds['predicted_label'].values
    unknown_probs = df_gat_preds['illicit_probability'].values
    confidence_cats = df_gat_preds['confidence_category'].values
    
    # Assign predicted labels to unknown nodes
    y_updated[unknown_indices] = unknown_preds
    
    # Verify no unlabeled nodes remain
    rem_unknown = np.sum(y_updated == -1)
    licit_total = np.sum(y_updated == 0)
    illicit_total = np.sum(y_updated == 1)
    
    print("\n--- Updated Node Label Distribution ---", flush=True)
    print(f"Total Nodes: {num_nodes:,}", flush=True)
    print(f"Remaining Unlabeled (-1): {rem_unknown}", flush=True)
    print(f"Final Licit Nodes (0): {licit_total:,} ({licit_total/num_nodes:.2%})", flush=True)
    print(f"Final Illicit Nodes (1): {illicit_total:,} ({illicit_total/num_nodes:.2%})", flush=True)
    
    # Create boolean indicator mask for predicted nodes
    is_predicted_np = np.zeros(num_nodes, dtype=bool)
    is_predicted_np[unknown_indices] = True
    
    # Probability array (ground truth nodes have 0.0 or 1.0 probability)
    prob_array = np.where(y_orig == 1, 1.0, np.where(y_orig == 0, 0.0, 0.5))
    prob_array[unknown_indices] = unknown_probs
    
    category_array = np.where(y_orig == 1, "Ground Truth Illicit", np.where(y_orig == 0, "Ground Truth Licit", "Predicted"))
    category_array[unknown_indices] = confidence_cats

    # 4. Save Updated PyTorch Geometric (.pt) Object
    data_updated = Data(
        x=data.x,
        edge_index=data.edge_index,
        y=torch.tensor(y_updated, dtype=torch.long)
    )
    data_updated.y_original = torch.tensor(y_orig, dtype=torch.long)
    data_updated.is_predicted = torch.tensor(is_predicted_np, dtype=torch.bool)
    data_updated.time_step = data.time_step
    data_updated.train_mask = data.train_mask
    data_updated.val_mask = data.val_mask
    data_updated.test_mask = data.test_mask
    
    pt_output_path = 'elliptic_pyg_data_updated.pt'
    print(f"\nSaving updated PyG Data object to: {pt_output_path}...", flush=True)
    torch.save(data_updated, pt_output_path)
    print(f"Successfully saved {pt_output_path} ({os.path.getsize(pt_output_path)/1e6:.1f} MB)", flush=True)

    # Also save to updated/ directory if present
    if os.path.exists('updated'):
        updated_pt_path = os.path.join('updated', 'elliptic_pyg_data_updated.pt')
        torch.save(data_updated, updated_pt_path)
        print(f"Also saved copy to: {updated_pt_path}", flush=True)

    # 5. Export Fully Labeled CSV Dataset
    print("\nExporting fully labeled CSV dataset...", flush=True)
    
    # Build Metadata DataFrame
    df_nodes = pd.DataFrame({
        'node_index': np.arange(num_nodes),
        'time_step': data.time_step.numpy(),
        'original_class': y_orig,
        'final_label': y_updated,
        'final_class_name': ['Illicit' if y == 1 else 'Licit' for y in y_updated],
        'is_predicted': is_predicted_np,
        'illicit_probability': np.round(prob_array, 4),
        'confidence_category': category_array
    })
    
    csv_output_path = 'elliptic_dataset_fully_labeled.csv'
    df_nodes.to_csv(csv_output_path, index=False)
    print(f"Saved node labels summary CSV to: {csv_output_path} ({os.path.getsize(csv_output_path)/1e6:.1f} MB)", flush=True)

    # 6. Export Excel Summary Report Workbook
    excel_output_path = 'updated_dataset_summary.xlsx'
    print(f"Saving dataset summary Excel workbook to: {excel_output_path}...", flush=True)
    
    df_summary_stats = pd.DataFrame({
        'Dataset Metric': [
            'Total Graph Nodes',
            'Total Graph Edges',
            'Total Features per Node',
            'Original Ground-Truth Licit (Class 0)',
            'Original Ground-Truth Illicit (Class 1)',
            'Original Unknown Nodes (Class -1)',
            'GAT Predicted Licit (for Unknowns)',
            'GAT Predicted Illicit (for Unknowns)',
            'Final Total Licit Nodes (0)',
            'Final Total Illicit Nodes (1)',
            'Final Fraud Ratio (%)'
        ],
        'Count / Value': [
            num_nodes,
            num_edges,
            num_features,
            np.sum(y_orig == 0),
            np.sum(y_orig == 1),
            np.sum(y_orig == -1),
            np.sum((y_orig == -1) & (y_updated == 0)),
            np.sum((y_orig == -1) & (y_updated == 1)),
            licit_total,
            illicit_total,
            f"{illicit_total / num_nodes:.2%}"
        ]
    })
    
    # Confidence breakdown table
    df_conf_breakdown = df_nodes['confidence_category'].value_counts().reset_index()
    df_conf_breakdown.columns = ['Confidence Category', 'Node Count']
    df_conf_breakdown['Percentage'] = (df_conf_breakdown['Node Count'] / num_nodes * 100).round(2).astype(str) + '%'
    
    with pd.ExcelWriter(excel_output_path, engine='openpyxl') as writer:
        df_summary_stats.to_excel(writer, sheet_name='Summary_Stats', index=False)
        df_conf_breakdown.to_excel(writer, sheet_name='Confidence_Breakdown', index=False)
        
    print(f"Saved Excel summary workbook at: {os.path.abspath(excel_output_path)}", flush=True)
    print("\nUpdated Dataset creation complete!", flush=True)

if __name__ == '__main__':
    main()
