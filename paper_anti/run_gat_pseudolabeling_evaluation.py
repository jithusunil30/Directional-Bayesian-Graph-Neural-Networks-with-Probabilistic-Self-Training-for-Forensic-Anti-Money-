import os
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from baseline_models import (
    GATModel, GCNModel, GraphSAGEModel, MLPModel,
    evaluate_predictions, find_optimal_threshold, compute_ece
)

np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

def run_gat_pseudolabeling(pyg_path=None, output_dir=None, conf_thresh_high=0.90, conf_thresh_low=0.10):
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(__file__))
    if pyg_path is None:
        pyg_path = os.path.join(output_dir, 'elliptic_pyg_data.pt')

    os.makedirs(output_dir, exist_ok=True)
    print("=" * 80)
    print("   SEMI-SUPERVISED PROBABILISTIC GAT PSEUDO-LABELING & RETRAINING   ")
    print("=" * 80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading Graph Data from {pyg_path} on {device}...")
    data = torch.load(pyg_path, weights_only=False)

    # 1. Train GAT on Labeled Train Nodes (timesteps 1-30)
    in_feat = data.x.shape[1]
    x_dev = data.x.to(device)
    edge_index_dev = data.edge_index.to(device)
    train_mask_dev = data.train_mask.to(device)
    val_mask_dev = data.val_mask.to(device)
    test_mask_dev = data.test_mask.to(device)
    unlabeled_mask_dev = data.unlabeled_mask.to(device)
    y_dev = data.y.to(device)

    y_train_np = data.y[data.train_mask].numpy().astype(int)
    pos_weight = float((y_train_np == 0).sum() / (y_train_np == 1).sum())
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(device))

    gat = GATModel(in_features=in_feat, hidden_dim=64, heads=4, dropout_rate=0.3).to(device)
    gat_opt = optim.Adam(gat.parameters(), lr=0.005, weight_decay=1e-4)

    gat_weights_path = os.path.join(output_dir, 'gat_supervised_model.pt')
    if os.path.exists(gat_weights_path):
        print(f"Loading existing supervised GAT weights from {gat_weights_path}...")
        gat.load_state_dict(torch.load(gat_weights_path, map_location=device, weights_only=False))
    else:
        print("Training GAT model on labeled train set...")
        best_val_loss = float('inf')
        best_state = None
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
                    best_state = gat.state_dict().copy()
        gat.load_state_dict(best_state)

    # 2. Forward Inference over 157,205 Unlabeled Nodes
    print("\n[Step 1] Generating Posterior Probabilities on Unlabeled Graph Nodes...")
    gat.eval()
    with torch.no_grad():
        full_logits = gat(x_dev, edge_index_dev)
        full_probs = torch.sigmoid(full_logits).cpu().numpy()

    unlabeled_indices = np.where(data.unlabeled_mask.numpy())[0]
    unlabeled_probs = full_probs[unlabeled_indices]
    unlabeled_timesteps = data.time_step.numpy()[unlabeled_indices]

    # 3. High-Confidence Filtering (Train timesteps 1-30 unlabeled nodes for augmentation)
    train_unlabeled_mask = (data.time_step.numpy() <= 30) & data.unlabeled_mask.numpy()
    train_unlabeled_idx = np.where(train_unlabeled_mask)[0]
    train_unlabeled_probs = full_probs[train_unlabeled_idx]

    pseudo_illicit_mask = train_unlabeled_probs >= conf_thresh_high
    pseudo_licit_mask = train_unlabeled_probs <= conf_thresh_low
    
    pseudo_illicit_idx = train_unlabeled_idx[pseudo_illicit_mask]
    pseudo_licit_idx = train_unlabeled_idx[pseudo_licit_mask]
    
    pseudo_total_idx = np.concatenate([pseudo_illicit_idx, pseudo_licit_idx])
    pseudo_labels = np.concatenate([np.ones(len(pseudo_illicit_idx)), np.zeros(len(pseudo_licit_idx))])

    print(f" -> Total Unlabeled Nodes:          {len(unlabeled_indices):,}")
    print(f" -> Train-Period Unlabeled (t<=30): {len(train_unlabeled_idx):,}")
    print(f" -> High-Conf Pseudo Illicit (>= {conf_thresh_high:.2f}): {len(pseudo_illicit_idx):,}")
    print(f" -> High-Conf Pseudo Licit (<= {conf_thresh_low:.2f}):   {len(pseudo_licit_idx):,}")
    print(f" -> Total Pseudo-Labels Selected:   {len(pseudo_total_idx):,} ({len(pseudo_total_idx)/len(train_unlabeled_idx)*100:.2f}% of t<=30 unlabeled)")

    # 4. Construct Augmented Training Set
    labeled_train_idx = np.where(data.train_mask.numpy())[0]
    labeled_train_y = data.y.numpy()[labeled_train_idx]
    
    augmented_train_idx = np.concatenate([labeled_train_idx, pseudo_total_idx])
    augmented_train_y = np.concatenate([labeled_train_y, pseudo_labels])
    augmented_train_X = data.x.numpy()[augmented_train_idx]

    val_idx = np.where(data.val_mask.numpy())[0]
    val_X = data.x.numpy()[val_idx]
    val_y = data.y.numpy()[val_idx].astype(int)

    test_idx = np.where(data.test_mask.numpy())[0]
    test_X = data.x.numpy()[test_idx]
    test_y = data.y.numpy()[test_idx].astype(int)

    print(f"\n[Step 2] Augmented Training Set Size: {len(augmented_train_idx):,} nodes (Original Labeled: {len(labeled_train_idx):,})")

    # Save pseudo-label stats summary
    df_pseudo_summary = pd.DataFrame({
        'Category': ['Original Labeled Train', 'Pseudo Illicit Added', 'Pseudo Licit Added', 'Total Augmented Train'],
        'Count': [len(labeled_train_idx), len(pseudo_illicit_idx), len(pseudo_licit_idx), len(augmented_train_idx)],
        'Illicit Count': [int((labeled_train_y == 1).sum()), len(pseudo_illicit_idx), 0, int((augmented_train_y == 1).sum())],
        'Licit Count': [int((labeled_train_y == 0).sum()), 0, len(pseudo_licit_idx), int((augmented_train_y == 0).sum())]
    })

    # 5. Retrain Models on Augmented Data
    print("\n[Step 3] Retraining Classifiers on Augmented Dataset...")
    aug_results = []
    aug_probs_dict = {}

    # (A) Retrained Logistic Regression
    print(" -> Retraining Logistic Regression + Pseudo-Labels...")
    lr_aug = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    lr_aug.fit(augmented_train_X, augmented_train_y)
    val_p_lr = lr_aug.predict_proba(val_X)[:, 1]
    test_p_lr = lr_aug.predict_proba(test_X)[:, 1]
    th_lr, _ = find_optimal_threshold(val_p_lr, val_y)
    res_lr = evaluate_predictions(test_p_lr, test_y, threshold=th_lr)
    aug_probs_dict['Logistic Regression (Pseudo)'] = test_p_lr
    aug_results.append({
        'Model': 'Logistic Regression + Pseudo-Labels', 'Accuracy': res_lr['Accuracy'],
        'Precision': res_lr['Precision'], 'Recall': res_lr['Recall'], 'F1_Score': res_lr['F1_Score'],
        'ROC_AUC': res_lr['ROC_AUC'], 'PR_AUC': res_lr['PR_AUC'], 'ECE': res_lr['ECE'], 'Opt_Threshold': th_lr
    })

    # (B) Retrained Random Forest
    print(" -> Retraining Random Forest + Pseudo-Labels...")
    rf_aug = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1)
    rf_aug.fit(augmented_train_X, augmented_train_y)
    val_p_rf = rf_aug.predict_proba(val_X)[:, 1]
    test_p_rf = rf_aug.predict_proba(test_X)[:, 1]
    th_rf, _ = find_optimal_threshold(val_p_rf, val_y)
    res_rf = evaluate_predictions(test_p_rf, test_y, threshold=th_rf)
    aug_probs_dict['Random Forest (Pseudo)'] = test_p_rf
    aug_results.append({
        'Model': 'Random Forest + Pseudo-Labels', 'Accuracy': res_rf['Accuracy'],
        'Precision': res_rf['Precision'], 'Recall': res_rf['Recall'], 'F1_Score': res_rf['F1_Score'],
        'ROC_AUC': res_rf['ROC_AUC'], 'PR_AUC': res_rf['PR_AUC'], 'ECE': res_rf['ECE'], 'Opt_Threshold': th_rf
    })

    # (C) Retrained XGBoost
    print(" -> Retraining XGBoost + Pseudo-Labels...")
    aug_pos_w = float((augmented_train_y == 0).sum() / (augmented_train_y == 1).sum())
    xgb_aug = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, scale_pos_weight=aug_pos_w, random_state=42, eval_metric='logloss', n_jobs=-1)
    xgb_aug.fit(augmented_train_X, augmented_train_y)
    val_p_xgb = xgb_aug.predict_proba(val_X)[:, 1]
    test_p_xgb = xgb_aug.predict_proba(test_X)[:, 1]
    th_xgb, _ = find_optimal_threshold(val_p_xgb, val_y)
    res_xgb = evaluate_predictions(test_p_xgb, test_y, threshold=th_xgb)
    aug_probs_dict['XGBoost (Pseudo)'] = test_p_xgb
    aug_results.append({
        'Model': 'XGBoost + Pseudo-Labels', 'Accuracy': res_xgb['Accuracy'],
        'Precision': res_xgb['Precision'], 'Recall': res_xgb['Recall'], 'F1_Score': res_xgb['F1_Score'],
        'ROC_AUC': res_xgb['ROC_AUC'], 'PR_AUC': res_xgb['PR_AUC'], 'ECE': res_xgb['ECE'], 'Opt_Threshold': th_xgb
    })

    # (D) Retrained GAT on Augmented Graph
    print(" -> Retraining GAT with Semi-Supervised Pseudo-Labels...")
    aug_train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    aug_train_mask[augmented_train_idx] = True
    aug_y_tensor = data.y.clone()
    aug_y_tensor[pseudo_total_idx] = torch.tensor(pseudo_labels, dtype=torch.float32)

    gat_aug = GATModel(in_features=in_feat, hidden_dim=64, heads=4, dropout_rate=0.3).to(device)
    gat_aug_opt = optim.Adam(gat_aug.parameters(), lr=0.005, weight_decay=1e-4)
    aug_loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([aug_pos_w]).to(device))
    
    aug_train_mask_dev = aug_train_mask.to(device)
    aug_y_dev = aug_y_tensor.to(device)
    best_aug_val_loss = float('inf')
    best_gat_aug_state = None

    for epoch in range(1, 61):
        gat_aug.train()
        gat_aug_opt.zero_grad()
        out = gat_aug(x_dev, edge_index_dev)
        loss = aug_loss_fn(out[aug_train_mask_dev], aug_y_dev[aug_train_mask_dev])
        loss.backward()
        gat_aug_opt.step()

        gat_aug.eval()
        with torch.no_grad():
            v_out = gat_aug(x_dev, edge_index_dev)
            v_loss = loss_fn(v_out[val_mask_dev], y_dev[val_mask_dev]).item()
            if v_loss < best_aug_val_loss:
                best_aug_val_loss = v_loss
                best_gat_aug_state = gat_aug.state_dict().copy()

    gat_aug.load_state_dict(best_gat_aug_state)
    gat_aug.eval()
    with torch.no_grad():
        all_out_aug = gat_aug(x_dev, edge_index_dev)
        val_p_gat_aug = torch.sigmoid(all_out_aug[val_mask_dev]).cpu().numpy()
        test_p_gat_aug = torch.sigmoid(all_out_aug[test_mask_dev]).cpu().numpy()

    th_gat_aug, _ = find_optimal_threshold(val_p_gat_aug, val_y)
    res_gat_aug = evaluate_predictions(test_p_gat_aug, test_y, threshold=th_gat_aug)
    aug_probs_dict['GAT (Pseudo)'] = test_p_gat_aug
    aug_results.append({
        'Model': 'GAT + Pseudo-Labels (Ours)', 'Accuracy': res_gat_aug['Accuracy'],
        'Precision': res_gat_aug['Precision'], 'Recall': res_gat_aug['Recall'], 'F1_Score': res_gat_aug['F1_Score'],
        'ROC_AUC': res_gat_aug['ROC_AUC'], 'PR_AUC': res_gat_aug['PR_AUC'], 'ECE': res_gat_aug['ECE'], 'Opt_Threshold': th_gat_aug
    })

    # Save model weights & probabilities
    torch.save(best_gat_aug_state, os.path.join(output_dir, 'gat_pseudolabeled_model.pt'))
    for name, p in aug_probs_dict.items():
        np.save(os.path.join(output_dir, f'test_probs_{name.lower().replace(" ", "_").replace("(", "").replace(")", "")}.npy'), p)

    df_aug_metrics = pd.DataFrame(aug_results)
    print("\n" + "=" * 80)
    print("   SEMI-SUPERVISED PSEUDO-LABELING PERFORMANCE COMPARISON TABLE   ")
    print("=" * 80)
    print(df_aug_metrics.to_string(index=False))

    # Save comparison to Excel
    comp_xlsx = os.path.join(output_dir, 'pseudolabeling_performance_comparison.xlsx')
    with pd.ExcelWriter(comp_xlsx, engine='openpyxl') as writer:
        df_aug_metrics.to_excel(writer, sheet_name='PseudoLabeling_Performance', index=False)
        df_pseudo_summary.to_excel(writer, sheet_name='PseudoLabel_Statistics', index=False)

    print(f"\nSaved pseudo-labeling metrics to: {comp_xlsx}")
    return df_aug_metrics, df_pseudo_summary

if __name__ == '__main__':
    run_gat_pseudolabeling()
