import torch
import torch_geometric

def display_pt_file(file_path):
    print(f"Loading PyG Graph Data from: {file_path}...\n")
    try:
        # Load with weights_only=False to allow custom PyTorch Geometric classes to unpickle
        data = torch.load(file_path, weights_only=False)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return

    print("=== PyTorch Geometric Data Object ===")
    print(data)
    
    print("\n=== Attributes and Shapes ===")
    print(f"Feature Matrix (x) Shape:       {data.x.shape} (dtype: {data.x.dtype})")
    print(f"Edge Index (edge_index) Shape:  {data.edge_index.shape} (dtype: {data.edge_index.dtype})")
    print(f"Target Labels (y) Shape:        {data.y.shape} (dtype: {data.y.dtype})")
    print(f"Time Steps Shape:               {data.time_step.shape} (dtype: {data.time_step.dtype})")
    print(f"Train Mask Shape:               {data.train_mask.shape} (dtype: {data.train_mask.dtype}, Count True: {data.train_mask.sum().item()})")
    print(f"Validation Mask Shape:          {data.val_mask.shape} (dtype: {data.val_mask.dtype}, Count True: {data.val_mask.sum().item()})")
    print(f"Test Mask Shape:                {data.test_mask.shape} (dtype: {data.test_mask.dtype}, Count True: {data.test_mask.sum().item()})")

    print("\n=== Sample Values ===")
    print("Labels sample (y[:10]):")
    print(data.y[:10])
    print("\nTime steps sample (time_step[:10]):")
    print(data.time_step[:10])
    print("\nEdge Index sample (first 5 edges):")
    print(data.edge_index[:, :5])
    print("\nFeatures sample (x[0, :5] - first 5 features of node 0):")
    print(data.x[0, :5])

if __name__ == "__main__":
    import os
    pt_path = "elliptic_pyg_data.pt"
    if not os.path.exists(pt_path) and os.path.exists("updated/elliptic_pyg_data.pt"):
        pt_path = "updated/elliptic_pyg_data.pt"
    display_pt_file(pt_path)
