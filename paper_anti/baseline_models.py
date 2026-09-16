import os
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix, roc_curve, precision_recall_curve
)

# Set seeds
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# --- Calibration Error (ECE) Helper ---
def compute_ece(probs, labels, n_bins=10):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total_samples = len(labels)
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        in_bin = (probs >= bin_lower) & (probs < bin_upper if i < n_bins - 1 else probs <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        if in_bin.sum() > 0:
            accuracy_in_bin = np.mean(labels[in_bin])
            avg_confidence_in_bin = np.mean(probs[in_bin])
            ece += np.abs(accuracy_in_bin - avg_confidence_in_bin) * (in_bin.sum() / total_samples)
    return float(ece)

def find_optimal_threshold(val_probs, val_labels):
    best_thresh = 0.5
    best_f1 = 0.0
    for thresh in np.linspace(0.05, 0.95, 91):
        preds = (val_probs >= thresh).astype(int)
        score = f1_score(val_labels, preds, pos_label=1, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_thresh = float(thresh)
    return best_thresh, best_f1

def evaluate_predictions(probs, labels, threshold=0.5):
    preds = (probs >= threshold).astype(int)
    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, pos_label=1, zero_division=0)
    rec = recall_score(labels, preds, pos_label=1, zero_division=0)
    f1 = f1_score(labels, preds, pos_label=1, zero_division=0)
    try:
        roc_auc = roc_auc_score(labels, probs)
    except:
        roc_auc = 0.5
    try:
        pr_auc = average_precision_score(labels, probs)
    except:
        pr_auc = 0.0
    ece = compute_ece(probs, labels, n_bins=10)
    cm = confusion_matrix(labels, preds)
    return {
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1_Score': f1,
        'ROC_AUC': roc_auc,
        'PR_AUC': pr_auc,
        'ECE': ece,
        'Confusion_Matrix': cm,
        'Predictions': preds,
        'Probabilities': probs
    }

# --- PyTorch Models ---
class MLPModel(nn.Module):
    def __init__(self, in_features, hidden_dim=128, dropout_rate=0.3):
        super(MLPModel, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 64)
        self.fc3 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = self.dropout(h)
        h = F.relu(self.fc2(h))
        h = self.dropout(h)
        out = self.fc3(h).squeeze(-1)
        return out

class GCNModel(nn.Module):
    def __init__(self, in_features, hidden_dim=128, dropout_rate=0.3):
        super(GCNModel, self).__init__()
        self.conv1 = GCNConv(in_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, edge_index):
        h = F.relu(self.conv1(x, edge_index))
        h = self.dropout(h)
        out = self.conv2(h, edge_index).squeeze(-1)
        return out

class GraphSAGEModel(nn.Module):
    def __init__(self, in_features, hidden_dim=128, dropout_rate=0.3):
        super(GraphSAGEModel, self).__init__()
        self.conv1 = SAGEConv(in_features, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, edge_index):
        h = F.relu(self.conv1(x, edge_index))
        h = self.dropout(h)
        out = self.conv2(h, edge_index).squeeze(-1)
        return out

class GATModel(nn.Module):
    def __init__(self, in_features, hidden_dim=64, heads=4, dropout_rate=0.3):
        super(GATModel, self).__init__()
        self.conv1 = GATConv(in_features, hidden_dim, heads=heads, dropout=dropout_rate)
        self.conv2 = GATConv(hidden_dim * heads, 1, heads=1, concat=False, dropout=dropout_rate)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, edge_index):
        h = F.elu(self.conv1(x, edge_index))
        h = self.dropout(h)
        out = self.conv2(h, edge_index).squeeze(-1)
        return out

def run_baseline_benchmark(pyg_path=None, output_dir=None):
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(__file__))
    if pyg_path is None:
        pyg_path = os.path.join(output_dir, 'elliptic_pyg_data.pt')

    os.makedirs(output_dir, exist_ok=True)
    print("=" * 75)
    print("   ELLIPTIC AML PIPELINE: BASELINE MODEL BENCHMARKING (SUPERVISED)   ")
    print("=" * 75)

    print(f"Loading preprocessed PyG Data from: {pyg_path}...")
    data = torch.load(pyg_path, weights_only=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loaded {data.num_nodes:,} nodes, {data.num_edges:,} edges on device: {device}")

    # Extract NumPy splits for tabular models
    train_mask_np = data.train_mask.numpy()
    val_mask_np = data.val_mask.numpy()
    test_mask_np = data.test_mask.numpy()

    X_np = data.x.numpy()
    y_np = data.y.numpy().astype(int)

    X_train, y_train = X_np[train_mask_np], y_np[train_mask_np]
    X_val, y_val = X_np[val_mask_np], y_np[val_mask_np]
    X_test, y_test = X_np[test_mask_np], y_np[test_mask_np]

    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    pos_weight = float(neg_count / pos_count)
    print(f"Class imbalance pos_weight (Train): {pos_weight:.2f}")

    results_table = []
    saved_probs = {}

    # -------------------------------------------------------------
    # 1. Logistic Regression
    # -------------------------------------------------------------
    print("\n[1/7] Training Baseline: Logistic Regression (Balanced)...")
    lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    lr.fit(X_train, y_train)
    val_probs_lr = lr.predict_proba(X_val)[:, 1]
    test_probs_lr = lr.predict_proba(X_test)[:, 1]
    opt_th_lr, val_f1_lr = find_optimal_threshold(val_probs_lr, y_val)
    res_lr = evaluate_predictions(test_probs_lr, y_test, threshold=opt_th_lr)
    saved_probs['Logistic Regression'] = (test_probs_lr, opt_th_lr)
    results_table.append({
        'Model': 'Logistic Regression', 'Category': 'Non-Graph ML',
        'Accuracy': res_lr['Accuracy'], 'Precision': res_lr['Precision'], 'Recall': res_lr['Recall'],
        'F1_Score': res_lr['F1_Score'], 'ROC_AUC': res_lr['ROC_AUC'], 'PR_AUC': res_lr['PR_AUC'],
        'ECE': res_lr['ECE'], 'Opt_Threshold': opt_th_lr
    })
    print(f" -> LR Test F1: {res_lr['F1_Score']:.4f} | ROC-AUC: {res_lr['ROC_AUC']:.4f} | PR-AUC: {res_lr['PR_AUC']:.4f} | ECE: {res_lr['ECE']:.4f}")

    # -------------------------------------------------------------
    # 2. Random Forest
    # -------------------------------------------------------------
    print("\n[2/7] Training Baseline: Random Forest (Balanced, 100 Trees)...")
    rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    val_probs_rf = rf.predict_proba(X_val)[:, 1]
    test_probs_rf = rf.predict_proba(X_test)[:, 1]
    opt_th_rf, val_f1_rf = find_optimal_threshold(val_probs_rf, y_val)
    res_rf = evaluate_predictions(test_probs_rf, y_test, threshold=opt_th_rf)
    saved_probs['Random Forest'] = (test_probs_rf, opt_th_rf)
    results_table.append({
        'Model': 'Random Forest', 'Category': 'Non-Graph ML',
        'Accuracy': res_rf['Accuracy'], 'Precision': res_rf['Precision'], 'Recall': res_rf['Recall'],
        'F1_Score': res_rf['F1_Score'], 'ROC_AUC': res_rf['ROC_AUC'], 'PR_AUC': res_rf['PR_AUC'],
        'ECE': res_rf['ECE'], 'Opt_Threshold': opt_th_rf
    })
    print(f" -> RF Test F1: {res_rf['F1_Score']:.4f} | ROC-AUC: {res_rf['ROC_AUC']:.4f} | PR-AUC: {res_rf['PR_AUC']:.4f} | ECE: {res_rf['ECE']:.4f}")

    # -------------------------------------------------------------
    # 3. XGBoost
    # -------------------------------------------------------------
    print("\n[3/7] Training Baseline: XGBoost (scale_pos_weight)...")
    xgb = XGBClassifier(
        n_estimators=100, max_depth=6, learning_rate=0.1,
        scale_pos_weight=pos_weight, random_state=42, eval_metric='logloss', n_jobs=-1
    )
    xgb.fit(X_train, y_train)
    val_probs_xgb = xgb.predict_proba(X_val)[:, 1]
    test_probs_xgb = xgb.predict_proba(X_test)[:, 1]
    opt_th_xgb, val_f1_xgb = find_optimal_threshold(val_probs_xgb, y_val)
    res_xgb = evaluate_predictions(test_probs_xgb, y_test, threshold=opt_th_xgb)
    saved_probs['XGBoost'] = (test_probs_xgb, opt_th_xgb)
    results_table.append({
        'Model': 'XGBoost', 'Category': 'Non-Graph ML',
        'Accuracy': res_xgb['Accuracy'], 'Precision': res_xgb['Precision'], 'Recall': res_xgb['Recall'],
        'F1_Score': res_xgb['F1_Score'], 'ROC_AUC': res_xgb['ROC_AUC'], 'PR_AUC': res_xgb['PR_AUC'],
        'ECE': res_xgb['ECE'], 'Opt_Threshold': opt_th_xgb
    })
    print(f" -> XGBoost Test F1: {res_xgb['F1_Score']:.4f} | ROC-AUC: {res_xgb['ROC_AUC']:.4f} | PR-AUC: {res_xgb['PR_AUC']:.4f} | ECE: {res_xgb['ECE']:.4f}")

    # -------------------------------------------------------------
    # PyTorch Setup for Deep Learning Models
    # -------------------------------------------------------------
    x_dev = data.x.to(device)
    edge_index_dev = data.edge_index.to(device)
    train_mask_dev = data.train_mask.to(device)
    val_mask_dev = data.val_mask.to(device)
    test_mask_dev = data.test_mask.to(device)
    y_dev = data.y.to(device)

    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(device))
    in_feat = data.x.shape[1]

    # -------------------------------------------------------------
    # 4. Multi-Layer Perceptron (MLP)
    # -------------------------------------------------------------
    print("\n[4/7] Training Baseline: Multi-Layer Perceptron (MLP)...")
    mlp = MLPModel(in_features=in_feat, hidden_dim=128, dropout_rate=0.3).to(device)
    mlp_opt = optim.Adam(mlp.parameters(), lr=0.005, weight_decay=1e-4)
    best_val_loss = float('inf')
    best_mlp_state = None

    for epoch in range(1, 61):
        mlp.train()
        mlp_opt.zero_grad()
        out = mlp(x_dev[train_mask_dev])
        loss = loss_fn(out, y_dev[train_mask_dev])
        loss.backward()
        mlp_opt.step()

        mlp.eval()
        with torch.no_grad():
            v_out = mlp(x_dev[val_mask_dev])
            v_loss = loss_fn(v_out, y_dev[val_mask_dev]).item()
            if v_loss < best_val_loss:
                best_val_loss = v_loss
                best_mlp_state = mlp.state_dict().copy()

    mlp.load_state_dict(best_mlp_state)
    mlp.eval()
    with torch.no_grad():
        val_probs_mlp = torch.sigmoid(mlp(x_dev[val_mask_dev])).cpu().numpy()
        test_probs_mlp = torch.sigmoid(mlp(x_dev[test_mask_dev])).cpu().numpy()

    opt_th_mlp, _ = find_optimal_threshold(val_probs_mlp, y_val)
    res_mlp = evaluate_predictions(test_probs_mlp, y_test, threshold=opt_th_mlp)
    saved_probs['MLP'] = (test_probs_mlp, opt_th_mlp)
    results_table.append({
        'Model': 'MLP', 'Category': 'Neural Network',
        'Accuracy': res_mlp['Accuracy'], 'Precision': res_mlp['Precision'], 'Recall': res_mlp['Recall'],
        'F1_Score': res_mlp['F1_Score'], 'ROC_AUC': res_mlp['ROC_AUC'], 'PR_AUC': res_mlp['PR_AUC'],
        'ECE': res_mlp['ECE'], 'Opt_Threshold': opt_th_mlp
    })
    print(f" -> MLP Test F1: {res_mlp['F1_Score']:.4f} | ROC-AUC: {res_mlp['ROC_AUC']:.4f} | PR-AUC: {res_mlp['PR_AUC']:.4f} | ECE: {res_mlp['ECE']:.4f}")

    # -------------------------------------------------------------
    # 5. Graph Convolutional Network (GCN)
    # -------------------------------------------------------------
    print("\n[5/7] Training Graph Baseline: GCN (2 Layers, 128 Hidden)...")
    gcn = GCNModel(in_features=in_feat, hidden_dim=128, dropout_rate=0.3).to(device)
    gcn_opt = optim.Adam(gcn.parameters(), lr=0.01, weight_decay=1e-4)
    best_val_loss = float('inf')
    best_gcn_state = None

    for epoch in range(1, 61):
        gcn.train()
        gcn_opt.zero_grad()
        out = gcn(x_dev, edge_index_dev)
        loss = loss_fn(out[train_mask_dev], y_dev[train_mask_dev])
        loss.backward()
        gcn_opt.step()

        gcn.eval()
        with torch.no_grad():
            v_out = gcn(x_dev, edge_index_dev)
            v_loss = loss_fn(v_out[val_mask_dev], y_dev[val_mask_dev]).item()
            if v_loss < best_val_loss:
                best_val_loss = v_loss
                best_gcn_state = gcn.state_dict().copy()

    gcn.load_state_dict(best_gcn_state)
    gcn.eval()
    with torch.no_grad():
        all_out = gcn(x_dev, edge_index_dev)
        val_probs_gcn = torch.sigmoid(all_out[val_mask_dev]).cpu().numpy()
        test_probs_gcn = torch.sigmoid(all_out[test_mask_dev]).cpu().numpy()

    opt_th_gcn, _ = find_optimal_threshold(val_probs_gcn, y_val)
    res_gcn = evaluate_predictions(test_probs_gcn, y_test, threshold=opt_th_gcn)
    saved_probs['GCN'] = (test_probs_gcn, opt_th_gcn)
    results_table.append({
        'Model': 'GCN', 'Category': 'Graph Neural Network',
        'Accuracy': res_gcn['Accuracy'], 'Precision': res_gcn['Precision'], 'Recall': res_gcn['Recall'],
        'F1_Score': res_gcn['F1_Score'], 'ROC_AUC': res_gcn['ROC_AUC'], 'PR_AUC': res_gcn['PR_AUC'],
        'ECE': res_gcn['ECE'], 'Opt_Threshold': opt_th_gcn
    })
    print(f" -> GCN Test F1: {res_gcn['F1_Score']:.4f} | ROC-AUC: {res_gcn['ROC_AUC']:.4f} | PR-AUC: {res_gcn['PR_AUC']:.4f} | ECE: {res_gcn['ECE']:.4f}")

    # -------------------------------------------------------------
    # 6. GraphSAGE
    # -------------------------------------------------------------
    print("\n[6/7] Training Graph Baseline: GraphSAGE (2 Layers, 128 Hidden)...")
    sage = GraphSAGEModel(in_features=in_feat, hidden_dim=128, dropout_rate=0.3).to(device)
    sage_opt = optim.Adam(sage.parameters(), lr=0.01, weight_decay=1e-4)
    best_val_loss = float('inf')
    best_sage_state = None

    for epoch in range(1, 61):
        sage.train()
        sage_opt.zero_grad()
        out = sage(x_dev, edge_index_dev)
        loss = loss_fn(out[train_mask_dev], y_dev[train_mask_dev])
        loss.backward()
        sage_opt.step()

        sage.eval()
        with torch.no_grad():
            v_out = sage(x_dev, edge_index_dev)
            v_loss = loss_fn(v_out[val_mask_dev], y_dev[val_mask_dev]).item()
            if v_loss < best_val_loss:
                best_val_loss = v_loss
                best_sage_state = sage.state_dict().copy()

    sage.load_state_dict(best_sage_state)
    sage.eval()
    with torch.no_grad():
        all_out = sage(x_dev, edge_index_dev)
        val_probs_sage = torch.sigmoid(all_out[val_mask_dev]).cpu().numpy()
        test_probs_sage = torch.sigmoid(all_out[test_mask_dev]).cpu().numpy()

    opt_th_sage, _ = find_optimal_threshold(val_probs_sage, y_val)
    res_sage = evaluate_predictions(test_probs_sage, y_test, threshold=opt_th_sage)
    saved_probs['GraphSAGE'] = (test_probs_sage, opt_th_sage)
    results_table.append({
        'Model': 'GraphSAGE', 'Category': 'Graph Neural Network',
        'Accuracy': res_sage['Accuracy'], 'Precision': res_sage['Precision'], 'Recall': res_sage['Recall'],
        'F1_Score': res_sage['F1_Score'], 'ROC_AUC': res_sage['ROC_AUC'], 'PR_AUC': res_sage['PR_AUC'],
        'ECE': res_sage['ECE'], 'Opt_Threshold': opt_th_sage
    })
    print(f" -> GraphSAGE Test F1: {res_sage['F1_Score']:.4f} | ROC-AUC: {res_sage['ROC_AUC']:.4f} | PR-AUC: {res_sage['PR_AUC']:.4f} | ECE: {res_sage['ECE']:.4f}")

    # -------------------------------------------------------------
    # 7. Graph Attention Network (GAT)
    # -------------------------------------------------------------
    print("\n[7/7] Training Graph Baseline: GAT (4 Heads, 64 Hidden)...")
    gat = GATModel(in_features=in_feat, hidden_dim=64, heads=4, dropout_rate=0.3).to(device)
    gat_opt = optim.Adam(gat.parameters(), lr=0.005, weight_decay=1e-4)
    best_val_loss = float('inf')
    best_gat_state = None

    for epoch in range(1, 61):
        gat.train()
        gat_opt.zero_grad()
        out = gat(x_dev, edge_index_dev)
        loss = loss_fn(out[train_mask_dev], y_dev[train_mask_dev])
        loss.backward()
        gat_opt.step()

        gat.eval()
        with torch.no_grad():
            v_out = gat(x_dev, edge_index_dev)
            v_loss = loss_fn(v_out[val_mask_dev], y_dev[val_mask_dev]).item()
            if v_loss < best_val_loss:
                best_val_loss = v_loss
                best_gat_state = gat.state_dict().copy()

    gat.load_state_dict(best_gat_state)
    gat.eval()
    with torch.no_grad():
        all_out = gat(x_dev, edge_index_dev)
        val_probs_gat = torch.sigmoid(all_out[val_mask_dev]).cpu().numpy()
        test_probs_gat = torch.sigmoid(all_out[test_mask_dev]).cpu().numpy()

    opt_th_gat, _ = find_optimal_threshold(val_probs_gat, y_val)
    res_gat = evaluate_predictions(test_probs_gat, y_test, threshold=opt_th_gat)
    saved_probs['GAT'] = (test_probs_gat, opt_th_gat)
    results_table.append({
        'Model': 'GAT', 'Category': 'Graph Neural Network',
        'Accuracy': res_gat['Accuracy'], 'Precision': res_gat['Precision'], 'Recall': res_gat['Recall'],
        'F1_Score': res_gat['F1_Score'], 'ROC_AUC': res_gat['ROC_AUC'], 'PR_AUC': res_gat['PR_AUC'],
        'ECE': res_gat['ECE'], 'Opt_Threshold': opt_th_gat
    })
    print(f" -> GAT Test F1: {res_gat['F1_Score']:.4f} | ROC-AUC: {res_gat['ROC_AUC']:.4f} | PR-AUC: {res_gat['PR_AUC']:.4f} | ECE: {res_gat['ECE']:.4f}")

    # Save trained GAT weights for downstream pseudo-labeling
    torch.save(best_gat_state, os.path.join(output_dir, 'gat_supervised_model.pt'))
    np.save(os.path.join(output_dir, 'y_test.npy'), y_test)
    for model_name, (probs, th) in saved_probs.items():
        np.save(os.path.join(output_dir, f'test_probs_{model_name.lower().replace(" ", "_")}.npy'), probs)

    # Save Summary Report
    df_metrics = pd.DataFrame(results_table)
    report_xlsx = os.path.join(output_dir, 'baseline_performance_report.xlsx')
    with pd.ExcelWriter(report_xlsx, engine='openpyxl') as writer:
        df_metrics.to_excel(writer, sheet_name='Baseline_Performance', index=False)
    
    print("\n" + "=" * 75)
    print("   BASELINE BENCHMARK PERFORMANCE SUMMARY TABLE   ")
    print("=" * 75)
    print(df_metrics.to_string(index=False))
    print(f"\nSaved benchmark metrics to: {report_xlsx}")
    return df_metrics, saved_probs

if __name__ == '__main__':
    run_baseline_benchmark()
