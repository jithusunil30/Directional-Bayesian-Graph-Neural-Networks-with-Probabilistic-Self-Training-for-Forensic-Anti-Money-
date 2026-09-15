import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, log_loss, confusion_matrix,
    roc_curve, precision_recall_curve
)

import matplotlib.pyplot as plt
import seaborn as sns

# Set publication style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.dpi'] = 300


def compute_ece(probs, labels, n_bins=10):
    """Computes Expected Calibration Error (ECE)."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        bin_lower, bin_upper = bin_boundaries[i], bin_boundaries[i+1]
        in_bin = (probs >= bin_lower) & (probs < bin_upper)
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(labels[in_bin] == (probs[in_bin] >= 0.5))
            avg_confidence_in_bin = np.mean(probs[in_bin])
            ece += np.abs(accuracy_in_bin - avg_confidence_in_bin) * prop_in_bin
    return ece


# 1. PyTorch MLP Model
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, dropout=0.2):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.out = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        h = F.relu(self.bn1(self.fc1(x)))
        h = self.dropout(h)
        h = F.relu(self.bn2(self.fc2(h)))
        h = self.dropout(h)
        return self.out(h).squeeze(-1)


# 2. GCN Model
class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, dropout=0.2):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.out = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        h = F.relu(self.bn1(self.conv1(x, edge_index)))
        h = self.dropout(h)
        h = F.relu(self.bn2(self.conv2(h, edge_index)))
        h = self.dropout(h)
        return self.out(h).squeeze(-1)


# 3. GraphSAGE Model
class GraphSAGE(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, dropout=0.2):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.out = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        h = F.relu(self.bn1(self.conv1(x, edge_index)))
        h = self.dropout(h)
        h = F.relu(self.bn2(self.conv2(h, edge_index)))
        h = self.dropout(h)
        return self.out(h).squeeze(-1)


# 4. GAT Model (Multi-Head Attention)
class GAT(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, heads=2, dropout=0.2):
        super(GAT, self).__init__()
        self.conv1 = GATConv(input_dim, hidden_dim // heads, heads=heads)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.conv2 = GATConv(hidden_dim, hidden_dim, heads=1)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.out = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        h = F.elu(self.bn1(self.conv1(x, edge_index)))
        h = self.dropout(h)
        h = F.elu(self.bn2(self.conv2(h, edge_index)))
        h = self.dropout(h)
        return self.out(h).squeeze(-1)


# 5. Bayesian GNN (Proposed Model using Monte Carlo Dropout)
class BayesianGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, dropout=0.3):
        super(BayesianGNN, self).__init__()
        self.conv1 = SAGEConv(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.out = nn.Linear(hidden_dim, 1)
        self.dropout_rate = dropout

    def forward(self, x, edge_index, mc_dropout=False):
        h = F.relu(self.bn1(self.conv1(x, edge_index)))
        h = F.dropout(h, p=self.dropout_rate, training=self.training or mc_dropout)
        h = F.relu(self.bn2(self.conv2(h, edge_index)))
        h = F.dropout(h, p=self.dropout_rate, training=self.training or mc_dropout)
        return self.out(h).squeeze(-1)

    def predict_mc(self, x, edge_index, mask, num_samples=20):
        self.eval()
        probs_list = []
        with torch.no_grad():
            for _ in range(num_samples):
                logits = self.forward(x, edge_index, mc_dropout=True)
                probs = torch.sigmoid(logits[mask]).cpu().numpy()
                probs_list.append(probs)
        
        probs_matrix = np.stack(probs_list, axis=0)
        mean_probs = np.mean(probs_matrix, axis=0)
        uncertainties = np.var(probs_matrix, axis=0)
        return mean_probs, uncertainties

# 6. Bayesian GAT (Proposed Model using GATConv + Monte Carlo Dropout)
class BayesianGAT(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, heads=2, dropout=0.3):
        super(BayesianGAT, self).__init__()
        self.conv1 = GATConv(input_dim, hidden_dim // heads, heads=heads)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.conv2 = GATConv(hidden_dim, hidden_dim, heads=1)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.out = nn.Linear(hidden_dim, 1)
        self.dropout_rate = dropout

    def forward(self, x, edge_index, mc_dropout=False):
        h = F.elu(self.bn1(self.conv1(x, edge_index)))
        h = F.dropout(h, p=self.dropout_rate, training=self.training or mc_dropout)
        h = F.elu(self.bn2(self.conv2(h, edge_index)))
        h = F.dropout(h, p=self.dropout_rate, training=self.training or mc_dropout)
        return self.out(h).squeeze(-1)

    def predict_mc(self, x, edge_index, mask, num_samples=20):
        self.eval()
        probs_list = []
        with torch.no_grad():
            for _ in range(num_samples):
                logits = self.forward(x, edge_index, mc_dropout=True)
                probs = torch.sigmoid(logits[mask]).cpu().numpy()
                probs_list.append(probs)
        
        probs_matrix = np.stack(probs_list, axis=0)
        mean_probs = np.mean(probs_matrix, axis=0)
        uncertainties = np.var(probs_matrix, axis=0)
        return mean_probs, uncertainties



def main():
    print("==================================================")
    print("   LAB 5: Full 8-Model Benchmark & Evaluation      ")
    print("==================================================")

    pyg_path = 'elliptic_pyg_data.pt'
    if not os.path.exists(pyg_path):
        print(f"[ERROR] Could not find '{pyg_path}'. Please run preprocess.py first.")
        sys.exit(1)

    print(f"\n[1/9] Loading PyG Graph Data from '{pyg_path}'...")
    data = torch.load(pyg_path, weights_only=False)

    x_tensor = data.x
    y_tensor = data.y
    edge_index = data.edge_index
    train_mask = data.train_mask
    val_mask = data.val_mask
    test_mask = data.test_mask

    # Numpy arrays for scikit-learn & xgboost
    X_train_np = x_tensor[train_mask].numpy()
    y_train_np = y_tensor[train_mask].numpy()
    
    X_val_np = x_tensor[val_mask].numpy()
    y_val_np = y_tensor[val_mask].numpy()

    X_test_np = x_tensor[test_mask].numpy()
    y_test_np = y_tensor[test_mask].numpy()

    print(f"      Train Samples (Steps 1-30) : {X_train_np.shape[0]} (Illicit: {(y_train_np==1).sum()})")
    print(f"      Val Samples (Steps 31-34)   : {X_val_np.shape[0]} (Illicit: {(y_val_np==1).sum()})")
    print(f"      Test Samples (Steps 35-49)  : {X_test_np.shape[0]} (Illicit: {(y_test_np==1).sum()})")

    train_licit = (y_train_np == 0).sum()
    train_illicit = (y_train_np == 1).sum()
    pos_weight_val = train_licit / train_illicit
    pos_weight_tensor = torch.tensor([pos_weight_val], dtype=torch.float)
    print(f"      Train Imbalance Pos Weight : {pos_weight_val:.4f}")

    predictions = {}
    probabilities = {}
    training_histories = {}
    epistemic_uncertainties = None

    # --- 1. Logistic Regression ---
    print("\n[2/9] Training Model 1: Logistic Regression...")
    logreg = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
    logreg.fit(X_train_np, y_train_np)
    y_pred_lr = logreg.predict(X_test_np)
    y_prob_lr = logreg.predict_proba(X_test_np)[:, 1]
    predictions['Logistic Regression'] = y_pred_lr
    probabilities['Logistic Regression'] = y_prob_lr

    # --- 2. Random Forest ---
    print("[3/9] Training Model 2: Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1)
    rf.fit(X_train_np, y_train_np)
    y_pred_rf = rf.predict(X_test_np)
    y_prob_rf = rf.predict_proba(X_test_np)[:, 1]
    predictions['Random Forest'] = y_pred_rf
    probabilities['Random Forest'] = y_prob_rf

    # --- 3. XGBoost ---
    print("[4/9] Training Model 3: XGBoost...")
    xgb = XGBClassifier(n_estimators=100, scale_pos_weight=pos_weight_val, random_state=42, eval_metric='logloss', n_jobs=-1)
    xgb.fit(X_train_np, y_train_np, eval_set=[(X_val_np, y_val_np)], verbose=False)
    y_pred_xgb = xgb.predict(X_test_np)
    y_prob_xgb = xgb.predict_proba(X_test_np)[:, 1]
    predictions['XGBoost'] = y_pred_xgb
    probabilities['XGBoost'] = y_prob_xgb

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_dim = x_tensor.shape[1]
    data = data.to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor.to(device))

    # --- 4. MLP ---
    print("[5/9] Training Model 4: MLP...")
    mlp = MLP(input_dim=input_dim, hidden_dim=64, dropout=0.2).to(device)
    optimizer_mlp = torch.optim.Adam(mlp.parameters(), lr=0.005, weight_decay=1e-4)

    X_train_t = x_tensor[train_mask].to(device)
    y_train_t = y_tensor[train_mask].float().to(device)
    X_val_t = x_tensor[val_mask].to(device)
    y_val_t = y_tensor[val_mask].float().to(device)
    X_test_t = x_tensor[test_mask].to(device)

    tr_mlp, val_mlp = [], []
    best_mlp_val_loss = float('inf')
    best_mlp_state = None

    for epoch in range(1, 101):
        mlp.train()
        optimizer_mlp.zero_grad()
        out = mlp(X_train_t)
        loss = criterion(out, y_train_t)
        loss.backward()
        optimizer_mlp.step()
        tr_mlp.append(loss.item())

        mlp.eval()
        with torch.no_grad():
            val_out = mlp(X_val_t)
            val_loss = criterion(val_out, y_val_t).item()
            val_mlp.append(val_loss)
            if val_loss < best_mlp_val_loss:
                best_mlp_val_loss = val_loss
                best_mlp_state = mlp.state_dict()

    mlp.load_state_dict(best_mlp_state)
    mlp.eval()
    with torch.no_grad():
        test_logits = mlp(X_test_t)
        y_prob_mlp = torch.sigmoid(test_logits).cpu().numpy()
        y_pred_mlp = (y_prob_mlp >= 0.5).astype(int)

    predictions['MLP'] = y_pred_mlp
    probabilities['MLP'] = y_prob_mlp
    training_histories['MLP'] = (tr_mlp, val_mlp)

    # Helper function to train PyG GNN models
    def train_gnn_model(model_class, model_name, lr=0.005, epochs=100):
        print(f"Training Model: {model_name}...")
        model = model_class(input_dim=input_dim, hidden_dim=64, dropout=0.2).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

        train_losses, val_losses = [], []
        best_val_loss = float('inf')
        best_state = None

        for epoch in range(1, epochs + 1):
            model.train()
            optimizer.zero_grad()
            out_all = model(data.x, data.edge_index)
            loss = criterion(out_all[data.train_mask], data.y[data.train_mask].float())
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

            model.eval()
            with torch.no_grad():
                out_val = model(data.x, data.edge_index)
                val_loss = criterion(out_val[data.val_mask], data.y[data.val_mask].float()).item()
                val_losses.append(val_loss)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = model.state_dict()

        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            out_test = model(data.x, data.edge_index)
            y_prob = torch.sigmoid(out_test[data.test_mask]).cpu().numpy()
            y_pred = (y_prob >= 0.5).astype(int)

        return y_pred, y_prob, train_losses, val_losses

    # --- 5. GCN ---
    print("\n[6/9] Training Model 5: GCN...")
    y_pred_gcn, y_prob_gcn, tr_gcn, val_gcn = train_gnn_model(GCN, "GCN")
    predictions['GCN'] = y_pred_gcn
    probabilities['GCN'] = y_prob_gcn
    training_histories['GCN'] = (tr_gcn, val_gcn)

    # --- 6. GraphSAGE ---
    print("[7/9] Training Model 6: GraphSAGE...")
    y_pred_sage, y_prob_sage, tr_sage, val_sage = train_gnn_model(GraphSAGE, "GraphSAGE")
    predictions['GraphSAGE'] = y_pred_sage
    probabilities['GraphSAGE'] = y_prob_sage
    training_histories['GraphSAGE'] = (tr_sage, val_sage)

    # --- 7. GAT ---
    print("[8/9] Training Model 7: GAT...")
    y_pred_gat, y_prob_gat, tr_gat, val_gat = train_gnn_model(GAT, "GAT")
    predictions['GAT'] = y_pred_gat
    probabilities['GAT'] = y_prob_gat
    training_histories['GAT'] = (tr_gat, val_gat)

    # --- 8. Bayesian GNN (Proposed Model) ---
    print("\n[9/9] Training Model 8: Bayesian GNN (Proposed Model)...")
    bgnn = BayesianGNN(input_dim=input_dim, hidden_dim=64, dropout=0.3).to(device)
    optimizer_bgnn = torch.optim.Adam(bgnn.parameters(), lr=0.005, weight_decay=1e-4)

    tr_bgnn, val_bgnn = [], []
    best_bgnn_val_loss = float('inf')
    best_bgnn_state = None

    for epoch in range(1, 101):
        bgnn.train()
        optimizer_bgnn.zero_grad()
        out_all = bgnn(data.x, data.edge_index)
        loss = criterion(out_all[data.train_mask], data.y[data.train_mask].float())
        loss.backward()
        optimizer_bgnn.step()
        tr_bgnn.append(loss.item())

        bgnn.eval()
        with torch.no_grad():
            out_val = bgnn(data.x, data.edge_index)
            val_loss = criterion(out_val[data.val_mask], data.y[data.val_mask].float()).item()
            val_bgnn.append(val_loss)
            if val_loss < best_bgnn_val_loss:
                best_bgnn_val_loss = val_loss
                best_bgnn_state = bgnn.state_dict()

    bgnn.load_state_dict(best_bgnn_state)
    print("      Performing Monte Carlo Inference (M=20 stochastic forward passes)...")
    y_prob_bgnn, epistemic_uncertainties = bgnn.predict_mc(data.x, data.edge_index, data.test_mask, num_samples=20)
    y_pred_bgnn = (y_prob_bgnn >= 0.5).astype(int)

    predictions['Bayesian GNN (Proposed)'] = y_pred_bgnn
    probabilities['Bayesian GNN (Proposed)'] = y_prob_bgnn
    training_histories['Bayesian GNN (Proposed)'] = (tr_bgnn, val_bgnn)

    # --- 9. Bayesian GAT (Proposed Model) ---
    print("\n[10/10] Training Model 9: Bayesian GAT (Proposed Model)...")
    bgat = BayesianGAT(input_dim=input_dim, hidden_dim=64, heads=2, dropout=0.3).to(device)
    optimizer_bgat = torch.optim.Adam(bgat.parameters(), lr=0.005, weight_decay=1e-4)

    tr_bgat, val_bgat = [], []
    best_bgat_val_loss = float('inf')
    best_bgat_state = None

    for epoch in range(1, 101):
        bgat.train()
        optimizer_bgat.zero_grad()
        out_all = bgat(data.x, data.edge_index)
        loss = criterion(out_all[data.train_mask], data.y[data.train_mask].float())
        loss.backward()
        optimizer_bgat.step()
        tr_bgat.append(loss.item())

        bgat.eval()
        with torch.no_grad():
            out_val = bgat(data.x, data.edge_index)
            val_loss = criterion(out_val[data.val_mask], data.y[data.val_mask].float()).item()
            val_bgat.append(val_loss)
            if val_loss < best_bgat_val_loss:
                best_bgat_val_loss = val_loss
                best_bgat_state = bgat.state_dict()

    bgat.load_state_dict(best_bgat_state)
    print("      Performing Monte Carlo Inference for Bayesian GAT (M=20 stochastic passes)...")
    y_prob_bgat, _ = bgat.predict_mc(data.x, data.edge_index, data.test_mask, num_samples=20)
    y_pred_bgat = (y_prob_bgat >= 0.5).astype(int)

    predictions['Bayesian GAT (Proposed)'] = y_pred_bgat
    probabilities['Bayesian GAT (Proposed)'] = y_prob_bgat
    training_histories['Bayesian GAT (Proposed)'] = (tr_bgat, val_bgat)


    # --- Metrics Computation ---
    metrics_list = []
    for model_name in predictions:
        ypred = predictions[model_name]
        yprob = probabilities[model_name]
        
        acc = accuracy_score(y_test_np, ypred)
        prec = precision_score(y_test_np, ypred, pos_label=1, zero_division=0)
        rec = recall_score(y_test_np, ypred, pos_label=1, zero_division=0)
        f1_illicit = f1_score(y_test_np, ypred, pos_label=1, zero_division=0)
        f1_macro = f1_score(y_test_np, ypred, average='macro', zero_division=0)
        auc = roc_auc_score(y_test_np, yprob)
        pr_auc = average_precision_score(y_test_np, yprob)
        yprob_clipped = np.clip(yprob, 1e-15, 1 - 1e-15)
        ce_loss = log_loss(y_test_np, yprob_clipped)
        ece_score = compute_ece(yprob, y_test_np)

        metrics_list.append({
            'Model Name': model_name,
            'Accuracy': acc,
            'Precision (Illicit)': prec,
            'Recall (Illicit)': rec,
            'F1-Score (Illicit)': f1_illicit,
            'Macro F1-Score': f1_macro,
            'ROC-AUC': auc,
            'PR-AUC (Avg Prec)': pr_auc,
            'Cross-Entropy Loss': ce_loss,
            'Expected Calibration Error (ECE)': ece_score
        })

    df_metrics = pd.DataFrame(metrics_list)
    print("\n==================================================")
    print("      FULL 8-MODEL BENCHMARK PERFORMANCE TABLE     ")
    print("==================================================")
    print(df_metrics.to_string(index=False))

    # Save to Excel
    report_xlsx = 'baseline_performance_report.xlsx'
    df_metrics.to_excel(report_xlsx, index=False)
    print(f"\n[+] Saved detailed evaluation metrics to '{report_xlsx}'")

    # --- Plot Artifacts Generation ---
    artifact_dir = r"C:\Users\niran\.gemini\antigravity-ide\brain\b422ee4b-f2cf-45f8-8301-5ffb87f5b8f2\plots"
    os.makedirs(artifact_dir, exist_ok=True)
    print(f"\nGenerating 7 Visualization Artifacts in:\n      '{artifact_dir}'")

    model_colors = {
        'Logistic Regression': '#7f8c8d',
        'Random Forest': '#2ecc71',
        'XGBoost': '#f39c12',
        'MLP': '#8e44ad',
        'GCN': '#e74c3c',
        'GraphSAGE': '#3498db',
        'GAT': '#9b59b6',
        'Bayesian GNN (Proposed)': '#1abc9c',
        'Bayesian GAT (Proposed)': '#e67e22'
    }

    # Figure 9: Confusion Matrices Grid (3 x 3)
    print("      -> Figure 9: fig9_confusion_matrices.png...")
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    for idx, (mname, ypred) in enumerate(predictions.items()):
        ax = axes[idx // 3, idx % 3]
        cm = confusion_matrix(y_test_np, ypred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False,
                    xticklabels=['Licit', 'Illicit'], yticklabels=['Licit', 'Illicit'],
                    annot_kws={"size": 10, "weight": "bold"})
        ax.set_title(mname, fontweight='bold', fontsize=10)
        ax.set_xlabel('Predicted Label', fontweight='bold', fontsize=9)
        if idx % 3 == 0:
            ax.set_ylabel('True Label', fontweight='bold', fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(artifact_dir, 'fig9_confusion_matrices.png'), dpi=300)
    plt.close()

    # Figure 10: ROC Curves Overlay
    print("      -> Figure 10: fig10_roc_curves.png...")
    plt.figure(figsize=(9, 7))
    for mname, yprob in probabilities.items():
        fpr, tpr, _ = roc_curve(y_test_np, yprob)
        auc_val = roc_auc_score(y_test_np, yprob)
        lw = 3.0 if 'Proposed' in mname else 1.8
        plt.plot(fpr, tpr, label=f"{mname} (AUC = {auc_val:.3f})", color=model_colors[mname], linewidth=lw)
    plt.plot([0, 1], [0, 1], 'k--', label='Chance Level (AUC = 0.500)')
    plt.xlabel('False Positive Rate (1 - Specificity)', fontweight='bold', fontsize=11)
    plt.ylabel('True Positive Rate (Recall)', fontweight='bold', fontsize=11)
    plt.title('Receiver Operating Characteristic (ROC) Curves Benchmark', fontweight='bold', fontsize=13)
    plt.legend(loc='lower right', fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(artifact_dir, 'fig10_roc_curves.png'), dpi=300)
    plt.close()

    # Figure 11: Precision-Recall Curves Overlay
    print("      -> Figure 11: fig11_pr_curves.png...")
    plt.figure(figsize=(9, 7))
    for mname, yprob in probabilities.items():
        prec_curve, rec_curve, _ = precision_recall_curve(y_test_np, yprob)
        pr_auc_val = average_precision_score(y_test_np, yprob)
        lw = 3.0 if 'Proposed' in mname else 1.8
        plt.plot(rec_curve, prec_curve, label=f"{mname} (PR-AUC = {pr_auc_val:.3f})", color=model_colors[mname], linewidth=lw)
    plt.xlabel('Recall (Sensitivity)', fontweight='bold', fontsize=11)
    plt.ylabel('Precision (Positive Predictive Value)', fontweight='bold', fontsize=11)
    plt.title('Precision-Recall (PR) Curves Benchmark (Minority Class: Illicit)', fontweight='bold', fontsize=13)
    plt.legend(loc='upper right', fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(artifact_dir, 'fig11_pr_curves.png'), dpi=300)
    plt.close()

    # Figure 12: Performance Comparison Bar Chart
    print("      -> Figure 12: fig12_metrics_comparison_bar.png...")
    fig, ax = plt.subplots(figsize=(15, 6))
    sub_df = df_metrics[['Model Name', 'F1-Score (Illicit)', 'Precision (Illicit)', 'Recall (Illicit)', 'ROC-AUC', 'PR-AUC (Avg Prec)']].melt(id_vars='Model Name', var_name='Metric', value_name='Score')
    sns.barplot(data=sub_df, x='Model Name', y='Score', hue='Metric', palette='Blues_r', ax=ax, edgecolor='black')
    ax.set_title('Full 9-Model Performance Metrics Comparison', fontweight='bold', fontsize=13)
    ax.set_ylabel('Score (0.0 to 1.0)', fontweight='bold', fontsize=11)
    ax.set_xlabel('Model Name', fontweight='bold', fontsize=11)
    plt.xticks(rotation=20, ha='right', fontweight='bold', fontsize=9)
    plt.ylim(0, 1.05)
    for p in ax.patches:
        height = p.get_height()
        if height > 0.02:
            ax.annotate(f'{height:.2f}', (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom', fontsize=6.5, fontweight='bold', xytext=(0, 2), textcoords='offset points')
    plt.tight_layout()
    plt.savefig(os.path.join(artifact_dir, 'fig12_metrics_comparison_bar.png'), dpi=300)
    plt.close()

    # Figure 13: Neural & GNN Loss Progression
    print("      -> Figure 13: fig13_training_loss_curves.png...")
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    nn_names = ['MLP', 'GCN', 'GraphSAGE', 'GAT', 'Bayesian GNN (Proposed)', 'Bayesian GAT (Proposed)']
    epochs = np.arange(1, 101)

    for idx, gname in enumerate(nn_names):
        ax = axes[idx // 3, idx % 3]
        tr_l, val_l = training_histories[gname]
        ax.plot(epochs, tr_l, label='Train Loss', color=model_colors[gname], linewidth=2)
        ax.plot(epochs, val_l, label='Val Loss', color='#2c3e50', linestyle='--', linewidth=2)
        ax.set_title(f'{gname} Loss Progression', fontweight='bold', fontsize=10)
        ax.set_xlabel('Epoch', fontsize=8)
        ax.set_ylabel('BCE Loss', fontsize=8)
        ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(artifact_dir, 'fig13_training_loss_curves.png'), dpi=300)
    plt.close()

    # Figure 14: Predictive Uncertainty Distribution (Proposed Model)
    print("      -> Figure 14: fig14_uncertainty_distribution.png...")
    plt.figure(figsize=(9, 6))
    df_unc = pd.DataFrame({
        'Uncertainty_Variance': epistemic_uncertainties,
        'Class': np.where(y_test_np == 1, 'Illicit (Fraud)', 'Licit (Legit)')
    })
    sns.kdeplot(data=df_unc[df_unc['Class']=='Licit (Legit)']['Uncertainty_Variance'], label='Licit Transactions', color='#2ecc71', fill=True, alpha=0.4, linewidth=2)
    sns.kdeplot(data=df_unc[df_unc['Class']=='Illicit (Fraud)']['Uncertainty_Variance'], label='Illicit Transactions', color='#e74c3c', fill=True, alpha=0.4, linewidth=2)
    plt.title('Proposed Bayesian GNN Epistemic Uncertainty Distribution', fontweight='bold', fontsize=13)
    plt.xlabel('Predictive Probability Variance ($\sigma_{uncertainty}^2$)', fontweight='bold', fontsize=11)
    plt.ylabel('Density', fontweight='bold', fontsize=11)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(artifact_dir, 'fig14_uncertainty_distribution.png'), dpi=300)
    plt.close()

    # Figure 15: Expected Calibration Error (ECE) Reliability Bar Chart
    print("      -> Figure 15: fig15_reliability_calibration_curves.png...")
    plt.figure(figsize=(10, 5.5))
    ece_df = df_metrics[['Model Name', 'Expected Calibration Error (ECE)']]
    bars = plt.bar(ece_df['Model Name'], ece_df['Expected Calibration Error (ECE)'], color=[model_colors[m] for m in ece_df['Model Name']], edgecolor='black', alpha=0.85)
    plt.title('Expected Calibration Error (ECE) Comparison (Lower is Better)', fontweight='bold', fontsize=13)
    plt.ylabel('ECE Score', fontweight='bold', fontsize=11)
    plt.xticks(rotation=20, ha='right', fontweight='bold', fontsize=9)
    for bar in bars:
        h = bar.get_height()
        plt.annotate(f'{h:.3f}', xy=(bar.get_x() + bar.get_width()/2, h),
                     xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8.5, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(artifact_dir, 'fig15_reliability_calibration_curves.png'), dpi=300)
    plt.close()

    print("\nFull 8-Model Benchmark finished successfully!")
    print("==================================================")

if __name__ == '__main__':
    main()
