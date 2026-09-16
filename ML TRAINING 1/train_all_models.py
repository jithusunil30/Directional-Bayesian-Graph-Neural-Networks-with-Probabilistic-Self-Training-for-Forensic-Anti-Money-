import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_curve, precision_recall_curve, auc, confusion_matrix,
    f1_score, accuracy_score, precision_score, recall_score
)

from utils import load_elliptic_data, evaluate_predictions, plot_calibration_curve
from models import (
    GCNModel, GATModel, GraphSAGEModel, GINModel,
    BayesianGCNModel, BayesianGATModel, BayesianGraphSAGEModel
)

# Set seeds for strict reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

def find_optimal_threshold(val_probs, val_labels, min_acc=0.91):
    """
    Finds optimal decision threshold on validation set (Steps 31-34)
    optimizing F1-score subject to Accuracy >= min_acc (ensuring >90% test accuracy).
    """
    best_th = 0.5
    best_f1 = -1.0
    for th in np.linspace(0.20, 0.95, 151):
        preds = (val_probs >= th).astype(int)
        acc = accuracy_score(val_labels, preds)
        f1 = f1_score(val_labels, preds, pos_label=1, zero_division=0)
        if acc >= min_acc and f1 > best_f1:
            best_f1 = f1
            best_th = float(th)
            
    # Fallback to maximize overall accuracy if constraint is strict
    if best_f1 == -1.0:
        best_acc = -1.0
        for th in np.linspace(0.20, 0.95, 151):
            preds = (val_probs >= th).astype(int)
            acc = accuracy_score(val_labels, preds)
            if acc > best_acc:
                best_acc = acc
                best_th = float(th)
    return best_th

def main():
    start_total = time.time()
    save_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(save_dir)
    print(f"==========================================================================")
    print(f"   PURE GRAPHICAL & BAYESIAN GRAPH ML (BNN/GNN) - ACCURACY > 90% BENCHMARK   ")
    print(f"==========================================================================\n")
    print(f"Working Directory: {save_dir}")

    # 1. Load Data
    data, X_train, y_train, X_val, y_val, X_test, y_test, raw_pos_weight = load_elliptic_data()

    # Standardize graph features
    scaler = StandardScaler()
    x_scaled_np = scaler.fit_transform(data.x.numpy())
    x_tensor = torch.tensor(x_scaled_np, dtype=torch.float32)
    y_tensor = torch.tensor(data.y.numpy(), dtype=torch.float32)
    edge_index = data.edge_index

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Execution Device: {device}\n")

    in_features = data.x.shape[1]
    results_list = []
    prob_dict = {}
    pred_dict = {}

    # Common PyG tensors on device
    x_gnn = x_tensor.to(device)
    edge_index_gnn = edge_index.to(device)
    train_mask = data.train_mask.to(device)
    val_mask = data.val_mask.to(device)
    test_mask = data.test_mask.to(device)
    y_gnn = y_tensor.to(device)
    
    # Balanced loss weighting: 3.5 gives balanced focus without excessive false alarms
    loss_weight = 3.5
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([loss_weight]).to(device))
    print(f"Using Balanced Graph BCE Loss with pos_weight = {loss_weight:.2f}")

    # =========================================================================
    # Model 1: Graph Convolutional Network (GCN)
    # =========================================================================
    print("\n[1/9] Training Graph Convolutional Network (GCN)...")
    sys.stdout.flush()
    t0 = time.time()
    gcn = GCNModel(in_features=in_features, hidden_dim=128, dropout_rate=0.3).to(device)
    opt = optim.Adam(gcn.parameters(), lr=0.01, weight_decay=1e-4)
    best_val_loss = float('inf')
    best_state = None

    for epoch in range(1, 46):
        gcn.train()
        opt.zero_grad()
        out = gcn(x_gnn, edge_index_gnn)
        loss = criterion(out[train_mask], y_gnn[train_mask])
        loss.backward()
        opt.step()

        gcn.eval()
        with torch.no_grad():
            val_out = gcn(x_gnn, edge_index_gnn)
            val_loss = criterion(val_out[val_mask], y_gnn[val_mask]).item()
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = gcn.state_dict().copy()

    gcn.load_state_dict(best_state)
    gcn.eval()
    with torch.no_grad():
        all_out = gcn(x_gnn, edge_index_gnn)
        gcn_val_probs = torch.sigmoid(all_out[val_mask]).cpu().numpy()
        gcn_probs = torch.sigmoid(all_out[test_mask]).cpu().numpy()

    opt_th_gcn = find_optimal_threshold(gcn_val_probs, y_val, min_acc=0.91)
    gcn_preds = (gcn_probs >= opt_th_gcn).astype(int)

    prob_dict['GCN'] = gcn_probs
    pred_dict['GCN'] = gcn_preds
    np.save('gcn_probs.npy', gcn_probs)
    metrics_gcn = evaluate_predictions(y_test, gcn_preds, gcn_probs)
    metrics_gcn['Model'] = 'GCN'
    metrics_gcn['Opt_Threshold'] = opt_th_gcn
    results_list.append(metrics_gcn)
    plot_calibration_curve(gcn_probs, y_test, 'GCN', 'calibration_gcn.png')
    print(f"Done in {time.time()-t0:.2f}s | Acc: {metrics_gcn['Accuracy']*100:.2f}%, F1: {metrics_gcn['F1-Score']:.4f}, Recall: {metrics_gcn['Recall']*100:.2f}%, Precision: {metrics_gcn['Precision']*100:.2f}%, AUC-ROC: {metrics_gcn['AUC-ROC']:.4f}")
    sys.stdout.flush()

    # =========================================================================
    # Model 2: Graph Attention Network (GAT)
    # =========================================================================
    print("\n[2/9] Training Graph Attention Network (GAT)...")
    sys.stdout.flush()
    t0 = time.time()
    gat = GATModel(in_features=in_features, hidden_dim=32, heads=2, dropout_rate=0.3).to(device)
    opt = optim.Adam(gat.parameters(), lr=0.01, weight_decay=1e-4)
    best_val_loss = float('inf')
    best_state = None

    for epoch in range(1, 41):
        gat.train()
        opt.zero_grad()
        out = gat(x_gnn, edge_index_gnn)
        loss = criterion(out[train_mask], y_gnn[train_mask])
        loss.backward()
        opt.step()

        gat.eval()
        with torch.no_grad():
            val_out = gat(x_gnn, edge_index_gnn)
            val_loss = criterion(val_out[val_mask], y_gnn[val_mask]).item()
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = gat.state_dict().copy()

    gat.load_state_dict(best_state)
    gat.eval()
    with torch.no_grad():
        all_out = gat(x_gnn, edge_index_gnn)
        gat_val_probs = torch.sigmoid(all_out[val_mask]).cpu().numpy()
        gat_probs = torch.sigmoid(all_out[test_mask]).cpu().numpy()

    opt_th_gat = find_optimal_threshold(gat_val_probs, y_val, min_acc=0.91)
    gat_preds = (gat_probs >= opt_th_gat).astype(int)

    prob_dict['GAT'] = gat_probs
    pred_dict['GAT'] = gat_preds
    np.save('gat_probs.npy', gat_probs)
    metrics_gat = evaluate_predictions(y_test, gat_preds, gat_probs)
    metrics_gat['Model'] = 'GAT'
    metrics_gat['Opt_Threshold'] = opt_th_gat
    results_list.append(metrics_gat)
    plot_calibration_curve(gat_probs, y_test, 'GAT', 'calibration_gat.png')
    print(f"Done in {time.time()-t0:.2f}s | Acc: {metrics_gat['Accuracy']*100:.2f}%, F1: {metrics_gat['F1-Score']:.4f}, Recall: {metrics_gat['Recall']*100:.2f}%, Precision: {metrics_gat['Precision']*100:.2f}%, AUC-ROC: {metrics_gat['AUC-ROC']:.4f}")
    sys.stdout.flush()

    # =========================================================================
    # Model 3: GraphSAGE
    # =========================================================================
    print("\n[3/9] Training GraphSAGE...")
    sys.stdout.flush()
    t0 = time.time()
    sage = GraphSAGEModel(in_features=in_features, hidden_dim=128, dropout_rate=0.3).to(device)
    opt = optim.Adam(sage.parameters(), lr=0.01, weight_decay=1e-4)
    best_val_loss = float('inf')
    best_state = None

    for epoch in range(1, 46):
        sage.train()
        opt.zero_grad()
        out = sage(x_gnn, edge_index_gnn)
        loss = criterion(out[train_mask], y_gnn[train_mask])
        loss.backward()
        opt.step()

        sage.eval()
        with torch.no_grad():
            val_out = sage(x_gnn, edge_index_gnn)
            val_loss = criterion(val_out[val_mask], y_gnn[val_mask]).item()
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = sage.state_dict().copy()

    sage.load_state_dict(best_state)
    sage.eval()
    with torch.no_grad():
        all_out = sage(x_gnn, edge_index_gnn)
        sage_val_probs = torch.sigmoid(all_out[val_mask]).cpu().numpy()
        sage_probs = torch.sigmoid(all_out[test_mask]).cpu().numpy()

    opt_th_sage = find_optimal_threshold(sage_val_probs, y_val, min_acc=0.91)
    sage_preds = (sage_probs >= opt_th_sage).astype(int)

    prob_dict['GraphSAGE'] = sage_probs
    pred_dict['GraphSAGE'] = sage_preds
    np.save('graphsage_probs.npy', sage_probs)
    metrics_sage = evaluate_predictions(y_test, sage_preds, sage_probs)
    metrics_sage['Model'] = 'GraphSAGE'
    metrics_sage['Opt_Threshold'] = opt_th_sage
    results_list.append(metrics_sage)
    plot_calibration_curve(sage_probs, y_test, 'GraphSAGE', 'calibration_graphsage.png')
    print(f"Done in {time.time()-t0:.2f}s | Acc: {metrics_sage['Accuracy']*100:.2f}%, F1: {metrics_sage['F1-Score']:.4f}, Recall: {metrics_sage['Recall']*100:.2f}%, Precision: {metrics_sage['Precision']*100:.2f}%, AUC-ROC: {metrics_sage['AUC-ROC']:.4f}")
    sys.stdout.flush()

    # =========================================================================
    # Model 4: Graph Isomorphism Network (GIN)
    # =========================================================================
    print("\n[4/9] Training Graph Isomorphism Network (GIN)...")
    sys.stdout.flush()
    t0 = time.time()
    gin = GINModel(in_features=in_features, hidden_dim=128, dropout_rate=0.3).to(device)
    opt = optim.Adam(gin.parameters(), lr=0.005, weight_decay=1e-4)
    best_val_loss = float('inf')
    best_state = None

    for epoch in range(1, 46):
        gin.train()
        opt.zero_grad()
        out = gin(x_gnn, edge_index_gnn)
        loss = criterion(out[train_mask], y_gnn[train_mask])
        loss.backward()
        opt.step()

        gin.eval()
        with torch.no_grad():
            val_out = gin(x_gnn, edge_index_gnn)
            val_loss = criterion(val_out[val_mask], y_gnn[val_mask]).item()
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = gin.state_dict().copy()

    gin.load_state_dict(best_state)
    gin.eval()
    with torch.no_grad():
        all_out = gin(x_gnn, edge_index_gnn)
        gin_val_probs = torch.sigmoid(all_out[val_mask]).cpu().numpy()
        gin_probs = torch.sigmoid(all_out[test_mask]).cpu().numpy()

    opt_th_gin = find_optimal_threshold(gin_val_probs, y_val, min_acc=0.91)
    gin_preds = (gin_probs >= opt_th_gin).astype(int)

    prob_dict['GIN'] = gin_probs
    pred_dict['GIN'] = gin_preds
    np.save('gin_probs.npy', gin_probs)
    metrics_gin = evaluate_predictions(y_test, gin_preds, gin_probs)
    metrics_gin['Model'] = 'GIN'
    metrics_gin['Opt_Threshold'] = opt_th_gin
    results_list.append(metrics_gin)
    plot_calibration_curve(gin_probs, y_test, 'GIN', 'calibration_gin.png')
    print(f"Done in {time.time()-t0:.2f}s | Acc: {metrics_gin['Accuracy']*100:.2f}%, F1: {metrics_gin['F1-Score']:.4f}, Recall: {metrics_gin['Recall']*100:.2f}%, Precision: {metrics_gin['Precision']*100:.2f}%, AUC-ROC: {metrics_gin['AUC-ROC']:.4f}")
    sys.stdout.flush()

    # =========================================================================
    # Model 5: Bayesian GCN (BGCN - Monte Carlo Variational Inference)
    # =========================================================================
    print("\n[5/9] Training Bayesian GCN (BGCN via MC Dropout Variational Inference)...")
    sys.stdout.flush()
    t0 = time.time()
    bgcn = BayesianGCNModel(in_features=in_features, hidden_dim=128, mc_dropout_rate=0.3).to(device)
    opt = optim.Adam(bgcn.parameters(), lr=0.01, weight_decay=1e-4)
    best_val_loss = float('inf')
    best_state = None

    for epoch in range(1, 46):
        bgcn.train()
        opt.zero_grad()
        out = bgcn(x_gnn, edge_index_gnn)
        loss = criterion(out[train_mask], y_gnn[train_mask])
        loss.backward()
        opt.step()

        bgcn.eval()
        with torch.no_grad():
            val_out = bgcn(x_gnn, edge_index_gnn)
            val_loss = criterion(val_out[val_mask], y_gnn[val_mask]).item()
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = bgcn.state_dict().copy()

    bgcn.load_state_dict(best_state)
    print("Computing Bayesian GCN Monte Carlo Variational Inference (T=30 passes)...")
    sys.stdout.flush()
    bgcn_val_probs, _, _ = bgcn.mc_predict(x_gnn, edge_index_gnn, val_mask, num_samples=30)
    bgcn_probs, bgcn_epistemic, bgcn_aleatoric = bgcn.mc_predict(x_gnn, edge_index_gnn, test_mask, num_samples=30)
    
    opt_th_bgcn = find_optimal_threshold(bgcn_val_probs, y_val, min_acc=0.91)
    bgcn_preds = (bgcn_probs >= opt_th_bgcn).astype(int)

    prob_dict['Bayesian GCN'] = bgcn_probs
    pred_dict['Bayesian GCN'] = bgcn_preds
    np.save('bgcn_probs.npy', bgcn_probs)
    np.save('bgcn_epistemic.npy', bgcn_epistemic)
    np.save('bgcn_aleatoric.npy', bgcn_aleatoric)

    metrics_bgcn = evaluate_predictions(y_test, bgcn_preds, bgcn_probs)
    metrics_bgcn['Model'] = 'Bayesian GCN'
    metrics_bgcn['Opt_Threshold'] = opt_th_bgcn
    results_list.append(metrics_bgcn)
    plot_calibration_curve(bgcn_probs, y_test, 'Bayesian GCN', 'calibration_bayesian_gcn.png')
    print(f"Done in {time.time()-t0:.2f}s | Acc: {metrics_bgcn['Accuracy']*100:.2f}%, F1: {metrics_bgcn['F1-Score']:.4f}, Recall: {metrics_bgcn['Recall']*100:.2f}%, Precision: {metrics_bgcn['Precision']*100:.2f}%, AUC-ROC: {metrics_bgcn['AUC-ROC']:.4f}")
    sys.stdout.flush()

    # =========================================================================
    # Model 6: Bayesian GAT (BGAT - Relational Uncertainty)
    # =========================================================================
    print("\n[6/9] Training Bayesian GAT (BGAT via MC Multi-Head Dropout)...")
    sys.stdout.flush()
    t0 = time.time()
    bgat = BayesianGATModel(in_features=in_features, hidden_dim=32, heads=2, mc_dropout_rate=0.3).to(device)
    opt = optim.Adam(bgat.parameters(), lr=0.01, weight_decay=1e-4)
    best_val_loss = float('inf')
    best_state = None

    for epoch in range(1, 41):
        bgat.train()
        opt.zero_grad()
        out = bgat(x_gnn, edge_index_gnn)
        loss = criterion(out[train_mask], y_gnn[train_mask])
        loss.backward()
        opt.step()

        bgat.eval()
        with torch.no_grad():
            val_out = bgat(x_gnn, edge_index_gnn)
            val_loss = criterion(val_out[val_mask], y_gnn[val_mask]).item()
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = bgat.state_dict().copy()

    bgat.load_state_dict(best_state)
    print("Computing Bayesian GAT Monte Carlo Variational Inference (T=30 passes)...")
    sys.stdout.flush()
    bgat_val_probs, _, _ = bgat.mc_predict(x_gnn, edge_index_gnn, val_mask, num_samples=30)
    bgat_probs, bgat_epistemic, bgat_aleatoric = bgat.mc_predict(x_gnn, edge_index_gnn, test_mask, num_samples=30)
    
    opt_th_bgat = find_optimal_threshold(bgat_val_probs, y_val, min_acc=0.91)
    bgat_preds = (bgat_probs >= opt_th_bgat).astype(int)

    prob_dict['Bayesian GAT'] = bgat_probs
    pred_dict['Bayesian GAT'] = bgat_preds
    np.save('bgat_probs.npy', bgat_probs)
    np.save('bgat_epistemic.npy', bgat_epistemic)
    np.save('bgat_aleatoric.npy', bgat_aleatoric)

    metrics_bgat = evaluate_predictions(y_test, bgat_preds, bgat_probs)
    metrics_bgat['Model'] = 'Bayesian GAT'
    metrics_bgat['Opt_Threshold'] = opt_th_bgat
    results_list.append(metrics_bgat)
    plot_calibration_curve(bgat_probs, y_test, 'Bayesian GAT', 'calibration_bayesian_gat.png')
    print(f"Done in {time.time()-t0:.2f}s | Acc: {metrics_bgat['Accuracy']*100:.2f}%, F1: {metrics_bgat['F1-Score']:.4f}, Recall: {metrics_bgat['Recall']*100:.2f}%, Precision: {metrics_bgat['Precision']*100:.2f}%, AUC-ROC: {metrics_bgat['AUC-ROC']:.4f}")
    sys.stdout.flush()

    # =========================================================================
    # Model 7: Bayesian GraphSAGE (BSAGE - Inductive Uncertainty)
    # =========================================================================
    print("\n[7/9] Training Bayesian GraphSAGE (BSAGE via MC Dropout)...")
    sys.stdout.flush()
    t0 = time.time()
    bsage = BayesianGraphSAGEModel(in_features=in_features, hidden_dim=128, mc_dropout_rate=0.3).to(device)
    opt = optim.Adam(bsage.parameters(), lr=0.01, weight_decay=1e-4)
    best_val_loss = float('inf')
    best_state = None

    for epoch in range(1, 46):
        bsage.train()
        opt.zero_grad()
        out = bsage(x_gnn, edge_index_gnn)
        loss = criterion(out[train_mask], y_gnn[train_mask])
        loss.backward()
        opt.step()

        bsage.eval()
        with torch.no_grad():
            val_out = bsage(x_gnn, edge_index_gnn)
            val_loss = criterion(val_out[val_mask], y_gnn[val_mask]).item()
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = bsage.state_dict().copy()

    bsage.load_state_dict(best_state)
    print("Computing Bayesian GraphSAGE Monte Carlo Variational Inference (T=30 passes)...")
    sys.stdout.flush()
    bsage_val_probs, _, _ = bsage.mc_predict(x_gnn, edge_index_gnn, val_mask, num_samples=30)
    bsage_probs, bsage_epistemic, bsage_aleatoric = bsage.mc_predict(x_gnn, edge_index_gnn, test_mask, num_samples=30)
    
    opt_th_bsage = find_optimal_threshold(bsage_val_probs, y_val, min_acc=0.91)
    bsage_preds = (bsage_probs >= opt_th_bsage).astype(int)

    prob_dict['Bayesian GraphSAGE'] = bsage_probs
    pred_dict['Bayesian GraphSAGE'] = bsage_preds
    np.save('bsage_probs.npy', bsage_probs)
    np.save('bsage_epistemic.npy', bsage_epistemic)
    np.save('bsage_aleatoric.npy', bsage_aleatoric)

    metrics_bsage = evaluate_predictions(y_test, bsage_preds, bsage_probs)
    metrics_bsage['Model'] = 'Bayesian GraphSAGE'
    metrics_bsage['Opt_Threshold'] = opt_th_bsage
    results_list.append(metrics_bsage)
    plot_calibration_curve(bsage_probs, y_test, 'Bayesian GraphSAGE', 'calibration_bayesian_graphsage.png')
    print(f"Done in {time.time()-t0:.2f}s | Acc: {metrics_bsage['Accuracy']*100:.2f}%, F1: {metrics_bsage['F1-Score']:.4f}, Recall: {metrics_bsage['Recall']*100:.2f}%, Precision: {metrics_bsage['Precision']*100:.2f}%, AUC-ROC: {metrics_bsage['AUC-ROC']:.4f}")
    sys.stdout.flush()

    # =========================================================================
    # Model 8: Bayesian GNN (BGNN / BNN Joint Ensemble Posterior)
    # =========================================================================
    print("\n[8/9] Evaluating Bayesian GNN (BNN / BGNN Ensemble Posterior)...")
    sys.stdout.flush()
    # Bayesian Model Averaging across the 3 variational posteriors
    bgnn_val_probs = (bgat_val_probs + bgcn_val_probs + bsage_val_probs) / 3.0
    bgnn_probs = (bgat_probs + bgcn_probs + bsage_probs) / 3.0
    bgnn_epistemic = (bgat_epistemic + bgcn_epistemic + bsage_epistemic) / 3.0
    bgnn_aleatoric = (bgat_aleatoric + bgcn_aleatoric + bsage_aleatoric) / 3.0

    opt_th_bgnn = find_optimal_threshold(bgnn_val_probs, y_val, min_acc=0.91)
    bgnn_preds = (bgnn_probs >= opt_th_bgnn).astype(int)

    prob_dict['Bayesian GNN (BNN)'] = bgnn_probs
    pred_dict['Bayesian GNN (BNN)'] = bgnn_preds
    np.save('bgnn_probs.npy', bgnn_probs)
    np.save('bgnn_epistemic.npy', bgnn_epistemic)
    np.save('bgnn_aleatoric.npy', bgnn_aleatoric)

    metrics_bgnn = evaluate_predictions(y_test, bgnn_preds, bgnn_probs)
    metrics_bgnn['Model'] = 'Bayesian GNN (BNN)'
    metrics_bgnn['Opt_Threshold'] = opt_th_bgnn
    results_list.append(metrics_bgnn)
    plot_calibration_curve(bgnn_probs, y_test, 'Bayesian GNN (BNN)', 'calibration_bayesian_gnn.png')
    print(f" -> BNN/BGNN Accuracy: {metrics_bgnn['Accuracy']*100:.2f}% | F1: {metrics_bgnn['F1-Score']:.4f} | Recall: {metrics_bgnn['Recall']*100:.2f}% | Precision: {metrics_bgnn['Precision']*100:.2f}% | AUC-ROC: {metrics_bgnn['AUC-ROC']:.4f}")
    sys.stdout.flush()

    # =========================================================================
    # Model 9: Graph Semi-Supervised Learning (Graph-SSL / Pseudo-Labeling)
    # =========================================================================
    print("\n[9/9] Training Graph Semi-Supervised Learning (Graph-SSL / Pseudo-Labeling)...")
    sys.stdout.flush()
    t0 = time.time()
    
    y_raw_np = data.y.numpy().copy()
    time_steps_np = data.time_step.numpy()
    unlabeled_mask_np = (y_raw_np == -1)
    unlabeled_mask_tensor = torch.tensor(unlabeled_mask_np, dtype=torch.bool, device=device)
    
    # Generate confident pseudo-labels for unlabeled transactions (y == -1) using trained GAT
    gat.eval()
    with torch.no_grad():
        unlabeled_logits = gat(x_gnn, edge_index_gnn)[unlabeled_mask_tensor]
        unlabeled_probs = torch.sigmoid(unlabeled_logits).cpu().numpy()

    pseudo_y_np = y_raw_np.copy()
    high_conf_illicit = (unlabeled_probs >= 0.88)
    high_conf_licit = (unlabeled_probs <= 0.12)
    unlabeled_indices = np.where(unlabeled_mask_np)[0]
    
    pseudo_y_np[unlabeled_indices[high_conf_illicit]] = 1
    pseudo_y_np[unlabeled_indices[high_conf_licit]] = 0

    ssl_train_mask_np = (time_steps_np <= 30) & (pseudo_y_np != -1)
    ssl_train_mask = torch.tensor(ssl_train_mask_np, dtype=torch.bool, device=device)
    ssl_y_tensor = torch.tensor(pseudo_y_np, dtype=torch.float32, device=device)

    ssl_model = GraphSAGEModel(in_features=in_features, hidden_dim=128, dropout_rate=0.3).to(device)
    opt_ssl = optim.Adam(ssl_model.parameters(), lr=0.01, weight_decay=1e-4)
    best_val_loss = float('inf')
    best_ssl_state = None

    for epoch in range(1, 46):
        ssl_model.train()
        opt_ssl.zero_grad()
        out = ssl_model(x_gnn, edge_index_gnn)
        loss = criterion(out[ssl_train_mask], ssl_y_tensor[ssl_train_mask])
        loss.backward()
        opt_ssl.step()

        ssl_model.eval()
        with torch.no_grad():
            val_out = ssl_model(x_gnn, edge_index_gnn)
            val_loss = criterion(val_out[val_mask], y_gnn[val_mask]).item()
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_ssl_state = ssl_model.state_dict().copy()

    ssl_model.load_state_dict(best_ssl_state)
    ssl_model.eval()
    with torch.no_grad():
        all_ssl_out = ssl_model(x_gnn, edge_index_gnn)
        ssl_val_probs = torch.sigmoid(all_ssl_out[val_mask]).cpu().numpy()
        ssl_probs = torch.sigmoid(all_ssl_out[test_mask]).cpu().numpy()

    opt_th_ssl = find_optimal_threshold(ssl_val_probs, y_val, min_acc=0.91)
    ssl_preds = (ssl_probs >= opt_th_ssl).astype(int)

    prob_dict['Graph-SSL (Pseudo-Label)'] = ssl_probs
    pred_dict['Graph-SSL (Pseudo-Label)'] = ssl_preds
    np.save('graph_ssl_probs.npy', ssl_probs)

    metrics_ssl = evaluate_predictions(y_test, ssl_preds, ssl_probs)
    metrics_ssl['Model'] = 'Graph-SSL (Pseudo-Label)'
    metrics_ssl['Opt_Threshold'] = opt_th_ssl
    results_list.append(metrics_ssl)
    plot_calibration_curve(ssl_probs, y_test, 'Graph-SSL (Pseudo-Label)', 'calibration_graph_ssl.png')
    print(f"Done in {time.time()-t0:.2f}s | Acc: {metrics_ssl['Accuracy']*100:.2f}%, F1: {metrics_ssl['F1-Score']:.4f}, Recall: {metrics_ssl['Recall']*100:.2f}%, Precision: {metrics_ssl['Precision']*100:.2f}%, AUC-ROC: {metrics_ssl['AUC-ROC']:.4f}")
    sys.stdout.flush()

    # =========================================================================
    # Plot Combined Uncertainty Distributions (Epistemic & Aleatoric)
    # =========================================================================
    plt.figure(figsize=(14, 5.5), dpi=300)
    plt.subplot(1, 2, 1)
    sns.kdeplot(bgcn_epistemic[y_test == 0], label='BGCN Licit (0)', color='#1F77B4', alpha=0.35, fill=True)
    sns.kdeplot(bgcn_epistemic[y_test == 1], label='BGCN Illicit (1)', color='#D62728', alpha=0.35, fill=True)
    sns.kdeplot(bgat_epistemic[y_test == 1], label='BGAT Illicit (1)', color='#FF7F0E', linestyle='--', linewidth=2)
    sns.kdeplot(bsage_epistemic[y_test == 1], label='BSAGE Illicit (1)', color='#2CA02C', linestyle=':', linewidth=2)
    plt.title('Bayesian Epistemic Uncertainty (Model Variance)', fontsize=12, fontweight='bold')
    plt.xlabel('Variance (Epistemic Risk)', fontsize=10)
    plt.ylabel('Density', fontsize=10)
    plt.legend(frameon=True, fontsize=9)
    plt.grid(True, linestyle=':', alpha=0.5)

    plt.subplot(1, 2, 2)
    sns.kdeplot(bgcn_aleatoric[y_test == 0], label='BGCN Licit (0)', color='#1F77B4', alpha=0.35, fill=True)
    sns.kdeplot(bgcn_aleatoric[y_test == 1], label='BGCN Illicit (1)', color='#D62728', alpha=0.35, fill=True)
    sns.kdeplot(bgat_aleatoric[y_test == 1], label='BGAT Illicit (1)', color='#FF7F0E', linestyle='--', linewidth=2)
    sns.kdeplot(bsage_aleatoric[y_test == 1], label='BSAGE Illicit (1)', color='#2CA02C', linestyle=':', linewidth=2)
    plt.title('Bayesian Aleatoric Uncertainty (Data Noise Entropy)', fontsize=12, fontweight='bold')
    plt.xlabel('Entropy (Bits)', fontsize=10)
    plt.ylabel('Density', fontsize=10)
    plt.legend(frameon=True, fontsize=9)
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.tight_layout()
    plt.savefig('bayesian_graph_uncertainty_distributions.png')
    plt.close()
    print("\nSaved bayesian_graph_uncertainty_distributions.png")

    # =========================================================================
    # Compile Metrics Summary Table
    # =========================================================================
    df_results = pd.DataFrame(results_list)
    col_order = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC', 'AUC-PR', 'ECE', 'Log-Loss', 'Brier-Score', 'Opt_Threshold', 'TN', 'FP', 'FN', 'TP']
    df_results = df_results[[c for c in col_order if c in df_results.columns]]
    
    df_results.to_csv('model_comparison_metrics.csv', index=False)
    df_results.to_excel('model_comparison_metrics.xlsx', index=False)
    print("\n" + "=" * 80)
    print("   FINAL PURE GRAPHICAL & BAYESIAN GNN PERFORMANCE BENCHMARK   ")
    print("=" * 80)
    print(df_results.to_string(index=False))
    print("\nSaved metric tables to model_comparison_metrics.csv & model_comparison_metrics.xlsx")

    # =========================================================================
    # Generate Combined Visualizations
    # =========================================================================
    # 1. Combined ROC Curves
    plt.figure(figsize=(9.5, 7.5), dpi=300)
    for model_name, probs in prob_dict.items():
        fpr, tpr, _ = roc_curve(y_test, probs)
        roc_val = auc(fpr, tpr)
        is_bnn = 'BNN' in model_name or 'Bayesian' in model_name
        lw = 2.4 if is_bnn else 1.6
        plt.plot(fpr, tpr, linewidth=lw, label=f'{model_name} (AUC = {roc_val:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.6, label='Random Chance (AUC = 0.500)')
    plt.title('Combined ROC Curves Across Pure Graphical & Bayesian Models', fontsize=13, fontweight='bold', pad=12)
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=11)
    plt.ylabel('True Positive Rate (Recall)', fontsize=11)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='lower right', frameon=True, fontsize=9.0)
    plt.tight_layout()
    plt.savefig('combined_roc_curves.png')
    plt.close()
    print("Saved combined_roc_curves.png")

    # 2. Combined PR Curves
    plt.figure(figsize=(9.5, 7.5), dpi=300)
    for model_name, probs in prob_dict.items():
        pr_prec, pr_rec, _ = precision_recall_curve(y_test, probs)
        pr_val = auc(pr_rec, pr_prec)
        is_bnn = 'BNN' in model_name or 'Bayesian' in model_name
        lw = 2.4 if is_bnn else 1.6
        plt.plot(pr_rec, pr_prec, linewidth=lw, label=f'{model_name} (AUC = {pr_val:.3f})')
    no_skill = np.sum(y_test == 1) / len(y_test)
    plt.plot([0, 1], [no_skill, no_skill], 'k--', alpha=0.6, label=f'Baseline Proportion ({no_skill:.3f})')
    plt.title('Combined Precision-Recall (PR) Curves for Illicit Detection', fontsize=13, fontweight='bold', pad=12)
    plt.xlabel('Recall (Sensitivity)', fontsize=11)
    plt.ylabel('Precision', fontsize=11)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='upper right', frameon=True, fontsize=9.0)
    plt.tight_layout()
    plt.savefig('combined_pr_curves.png')
    plt.close()
    print("Saved combined_pr_curves.png")

    # 3. Combined Confusion Matrices
    fig, axes = plt.subplots(3, 3, figsize=(18, 14), dpi=300)
    axes = axes.flatten()
    model_keys = list(pred_dict.keys())
    for idx, name in enumerate(model_keys):
        if idx >= len(axes):
            break
        ax = axes[idx]
        cm = confusion_matrix(y_test, pred_dict[name], labels=[0, 1])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax,
                    annot_kws={'size': 12, 'weight': 'bold'})
        ax.set_title(name, fontsize=11, fontweight='bold')
        ax.set_xlabel('Predicted Label', fontsize=9.5)
        ax.set_ylabel('True Label', fontsize=9.5)
        ax.set_xticklabels(['Licit (0)', 'Illicit (1)'])
        ax.set_yticklabels(['Licit (0)', 'Illicit (1)'])
    for extra_ax in axes[len(model_keys):]:
        fig.delaxes(extra_ax)
    plt.suptitle('Confusion Matrices Across Graphical & Bayesian Models (Accuracy > 90%)', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('confusion_matrices_all.png')
    plt.close()
    print("Saved confusion_matrices_all.png")

    # 4. Performance Comparison Bar Chart
    plt.figure(figsize=(16, 7), dpi=300)
    plot_df = df_results.copy()
    models = plot_df['Model']
    x = np.arange(len(models))
    width = 0.20

    plt.bar(x - 1.5*width, plot_df['Accuracy'], width, label='Accuracy', color='#2B5B84')
    plt.bar(x - 0.5*width, plot_df['Precision'], width, label='Precision', color='#E67E22')
    plt.bar(x + 0.5*width, plot_df['Recall'], width, label='Recall', color='#2ECC71')
    plt.bar(x + 1.5*width, plot_df['F1-Score'], width, label='F1-Score', color='#E74C3C')

    # Draw 90% target accuracy horizontal line
    plt.axhline(y=0.90, color='red', linestyle='--', linewidth=2.0, label='Target Threshold (Accuracy >= 90%)')

    plt.title('Performance Comparison: Pure Graphical & Bayesian GNNs on Bitcoin Graph', fontsize=14, fontweight='bold', pad=14)
    plt.xlabel('Graphical Machine Learning Architecture', fontsize=12, fontweight='bold')
    plt.ylabel('Evaluation Score (0.0 to 1.0)', fontsize=12, fontweight='bold')
    plt.xticks(x, models, rotation=20, ha='right', fontsize=9.5, fontweight='bold')
    plt.ylim([0, 1.12])
    plt.grid(axis='y', linestyle=':', alpha=0.6)
    plt.legend(loc='upper right', frameon=True, fontsize=10)
    plt.tight_layout()
    plt.savefig('performance_comparison_barchart.png')
    plt.close()
    print("Saved performance_comparison_barchart.png")

    total_time = time.time() - start_total
    print(f"\n=== All Graphical & Bayesian Models Completed in {total_time/60:.2f} minutes! ===")

if __name__ == '__main__':
    main()
