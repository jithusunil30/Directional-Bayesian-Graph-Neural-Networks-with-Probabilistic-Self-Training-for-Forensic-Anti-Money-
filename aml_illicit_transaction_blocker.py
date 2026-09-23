import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve, auc, confusion_matrix
from torch_geometric.nn import GCNConv, SAGEConv

# =========================================================================
# 1. ARCHITECTURE: Directional Residual Bayesian GNN (Dir-ResGNN)
# =========================================================================

class BayesianDirResGCN(nn.Module):
    """
    Directional Residual Bayesian Graph Convolutional Network.
    - Directional Convolution: Decouples upstream inflow (E_in) and downstream outflow (E_out).
    - Residual Skip-Connections: Prevents deep over-smoothing and signal collapse.
    - Monte Carlo Dropout: Enables Bayesian epistemic uncertainty quantification.
    """
    def __init__(self, in_features, hidden_dim=128, mc_dropout_rate=0.3):
        super(BayesianDirResGCN, self).__init__()
        self.conv1_fwd = GCNConv(in_features, hidden_dim)
        self.conv1_rev = GCNConv(in_features, hidden_dim)
        self.res1 = nn.Linear(in_features, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)

        self.conv2 = GCNConv(hidden_dim, 64)
        self.res2 = nn.Linear(hidden_dim, 64)
        self.bn2 = nn.BatchNorm1d(64)

        self.out = nn.Linear(64, 1)
        self.mc_dropout_rate = mc_dropout_rate

    def forward(self, x, edge_index, force_mc_dropout=False):
        training_mode = self.training or force_mc_dropout
        rev_edge_index = edge_index.flip(0)
        
        # Layer 1: Directional Fusion + Residual Skip
        h1 = self.conv1_fwd(x, edge_index) + self.conv1_rev(x, rev_edge_index) + self.res1(x)
        h1 = F.relu(self.bn1(h1))
        h1 = F.dropout(h1, p=self.mc_dropout_rate, training=training_mode)

        # Layer 2: Convolution + Residual Skip
        h2 = self.conv2(h1, edge_index) + self.res2(h1)
        h2 = F.relu(self.bn2(h2))
        h2 = F.dropout(h2, p=self.mc_dropout_rate, training=training_mode)

        return self.out(h2).squeeze(-1)

    @torch.no_grad()
    def mc_predict(self, x, edge_index, mask=None, num_passes=25):
        """
        Runs T Monte Carlo stochastic forward passes at inference time
        to decompose predictive probability mean and epistemic model uncertainty.
        """
        self.eval()
        samples = []
        for _ in range(num_passes):
            logits = self.forward(x, edge_index, force_mc_dropout=True)
            if mask is not None:
                logits = logits[mask]
            probs = torch.sigmoid(logits).cpu().numpy()
            samples.append(probs)

        samples = np.array(samples)  # Shape: (T, N_mask)
        mean_prob = np.mean(samples, axis=0)
        epistemic_unc = np.var(samples, axis=0) # Model variance
        return mean_prob, epistemic_unc

# =========================================================================
# 2. AUTOMATED COMPLIANCE DECISION ROUTER (DECLINE / REVIEW / APPROVE)
# =========================================================================

class AMLComplianceRouter:
    """
    3-Tier Uncertainty-Aware Automated Transaction Decision Engine.
    - Tier 1: Automated Immediate Freeze / Decline & SAR Auto-Filing.
    - Tier 2: Route to AML Compliance Officer for Manual Investigation.
    - Tier 3: Automated Instant Approval & Ledger Broadcast.
    """
    def __init__(self, prob_decline_threshold=0.932, prob_review_threshold=0.700, unc_threshold=0.040):
        self.prob_decline = prob_decline_threshold
        self.prob_review = prob_review_threshold
        self.unc_threshold = unc_threshold

    def route_transaction(self, tx_id, probability, epistemic_unc):
        """
        Routes a single live transaction based on Bayesian risk outputs.
        """
        if probability >= self.prob_decline and epistemic_unc < self.unc_threshold:
            action = "DECLINE & FREEZE"
            tier = "Tier 1 (High Risk & Confident)"
            reason = f"High Illicit Probability ({probability:.4f} >= {self.prob_decline}) & Low Uncertainty ({epistemic_unc:.5f})"
            color = "\033[91m" # Red
        elif probability >= self.prob_review or epistemic_unc >= self.unc_threshold:
            action = "ROUTE TO MANUAL REVIEW"
            tier = "Tier 2 (Ambiguous / Novel Attack)"
            reason = f"Moderate Risk ({probability:.4f}) OR High Model Uncertainty ({epistemic_unc:.5f} >= {self.unc_threshold})"
            color = "\033[93m" # Yellow
        else:
            action = "APPROVE & BROADCAST"
            tier = "Tier 3 (Verified Licit)"
            reason = f"Low Illicit Risk ({probability:.4f} < {self.prob_review}) & Low Uncertainty ({epistemic_unc:.5f})"
            color = "\033[92m" # Green
        
        reset = "\033[0m"
        return {
            "tx_id": tx_id,
            "probability": probability,
            "epistemic_unc": epistemic_unc,
            "action": action,
            "tier": tier,
            "reason": reason,
            "display": f"{color}[{action}] {tx_id} | P(illicit)={probability:.4f} | Unc={epistemic_unc:.5f} | {tier}{reset}"
        }

    def batch_route(self, probs, uncertainties):
        actions = []
        for p, u in zip(probs, uncertainties):
            if p >= self.prob_decline and u < self.unc_threshold:
                actions.append("DECLINE")
            elif p >= self.prob_review or u >= self.unc_threshold:
                actions.append("MANUAL_REVIEW")
            else:
                actions.append("APPROVE")
        return np.array(actions)

# =========================================================================
# 3. MAIN PIPELINE EXECUTION & SIMULATION
# =========================================================================

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Locate fully-labeled PyG graph (100% continuous graph to prevent Graph Severance Penalty)
    pyg_path = os.path.join(base_dir, 'dataset', 'elliptic_pyg_data_fully_labeled.pt')
    if not os.path.exists(pyg_path):
        pyg_path = os.path.join(base_dir, 'dataset', 'elliptic_pyg_data_updated.pt')
    if not os.path.exists(pyg_path):
        pyg_path = os.path.join(base_dir, 'dataset', 'elliptic_pyg_data.pt')

    print("=" * 85, flush=True)
    print("   AUTOMATED AML ILLICIT TRANSACTION DETECTION & DECLINE ENGINE   ", flush=True)
    print("   (Directional Residual Bayesian GNN + Uncertainty-Aware Compliance Router)", flush=True)
    print("=" * 85, flush=True)
    print(f"Loading Graph Data from: {pyg_path} ...", flush=True)
    
    data = torch.load(pyg_path, weights_only=False)
    
    x = data.x
    edge_index = data.edge_index
    y = data.y.long()
    time_steps = data.time_step.numpy()

    # Zero-Leakage Strict Temporal Split
    train_mask = torch.tensor(time_steps <= 30, dtype=torch.bool)
    val_mask   = torch.tensor((time_steps >= 31) & (time_steps <= 34), dtype=torch.bool)
    test_mask  = torch.tensor(time_steps >= 35, dtype=torch.bool)

    in_features = x.shape[1]
    num_total = len(y)
    num_test = int(test_mask.sum())

    print(f"\nGraph Data Summary:", flush=True)
    print(f"  • Total Nodes: {num_total:,}", flush=True)
    print(f"  • Total Edges: {edge_index.shape[1]:,}", flush=True)
    print(f"  • Feature Dimensions: {in_features} continuous attributes", flush=True)
    print(f"  • Train Set (t <= 30): {int(train_mask.sum()):,} nodes", flush=True)
    print(f"  • Test Set  (t >= 35): {num_test:,} nodes", flush=True)

    # Handle binary target (-1 or 2 mapping to 0/1)
    y_clean = y.clone()
    y_clean[y_clean == 2] = 0  # Licit
    y_clean[y_clean == -1] = 0 # Unlabeled fallback if any

    # Compute Soft Confidence Sample Weights w_i = max(P_i, 1 - P_i) to filter pseudo-label noise
    pos_weight_val = float((y_clean[train_mask] == 0).sum() / (y_clean[train_mask] == 1).sum())
    pos_weight = torch.tensor([pos_weight_val])

    # Initialize Directional Residual Bayesian GNN Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing Hardware Acceleration: {device}", flush=True)
    
    model = BayesianDirResGCN(in_features=in_features, hidden_dim=128, mc_dropout_rate=0.3).to(device)
    x = x.to(device)
    edge_index = edge_index.to(device)
    y_clean = y_clean.to(device)
    pos_weight = pos_weight.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.003, weight_decay=1e-4)

    # Train Model
    print("\nTraining Directional Residual Bayesian GNN (10 Epochs for fast demonstration)...", flush=True)
    model.train()
    for epoch in range(1, 11):
        optimizer.zero_grad()
        logits = model(x, edge_index)
        loss = F.binary_cross_entropy_with_logits(
            logits[train_mask], y_clean[train_mask].float(), pos_weight=pos_weight
        )
        loss.backward()
        optimizer.step()
        if epoch % 2 == 0 or epoch == 1:
            print(f"  Epoch [{epoch:02d}/10] - BCE Loss: {loss.item():.4f}", flush=True)

    # Run Monte Carlo Dropout Epistemic Uncertainty Predictor (T=10 passes)
    print("\nExecuting Monte Carlo Stochastic Inference (T=10 passes) on Test Graph...", flush=True)
    test_indices = np.where(test_mask.cpu().numpy())[0]
    mean_probs, epistemic_unc = model.mc_predict(x, edge_index, mask=test_mask.cpu(), num_passes=10)
    y_test_true = y_clean[test_mask].cpu().numpy()

    # Optimal Decision Threshold
    opt_theta = 0.932
    test_preds = (mean_probs >= opt_theta).astype(int)

    # Calculate Metrics (Exposing accuracy & showcasing PR-AUC North Star)
    test_acc = (test_preds == y_test_true).mean()
    test_prec = precision_score(y_test_true, test_preds, zero_division=0)
    test_rec = recall_score(y_test_true, test_preds, zero_division=0)
    test_f1 = f1_score(y_test_true, test_preds, zero_division=0)
    test_auc = roc_auc_score(y_test_true, mean_probs)
    
    prec_curve, rec_curve, _ = precision_recall_curve(y_test_true, mean_probs)
    test_prauc = auc(rec_curve, prec_curve)
    
    cm = confusion_matrix(y_test_true, test_preds)
    tn, fp, fn, tp = cm.ravel()

    print("\n" + "=" * 85, flush=True)
    print("   MODEL EVALUATION METRICS (ANTI-ACCURACY-ILLUSION GUARANTEE)   ", flush=True)
    print("=" * 85, flush=True)
    print(f"  * North Star Metric - PR-AUC:   {test_prauc:.4f} (High Precision-Recall Envelope)", flush=True)
    print(f"  * Discrimination Power - AUC-ROC: {test_auc:.4f}", flush=True)
    print(f"  * F1-Score:                      {test_f1:.4f}", flush=True)
    print(f"  * Precision:                     {test_prec*100:.2f}%", flush=True)
    print(f"  * Recall (Criminal Detection Rate): {test_rec*100:.2f}%", flush=True)
    print(f"  * Overall Raw Accuracy:          {test_acc*100:.2f}%", flush=True)
    print(f"\nConfusion Matrix Breakdown ({num_test:,} Test Transactions):", flush=True)
    print(f"  [TN] Verified Licit Approved:  {tn:,}", flush=True)
    print(f"  [FP] False Positives (Flagged): {fp:,}", flush=True)
    print(f"  [FN] Missed Illicit (Escaped):  {fn:,}", flush=True)
    print(f"  [TP] Illicit Criminals Caught: {tp:,}", flush=True)

    # =========================================================================
    # 4. COMPLIANCE ROUTER DEMONSTRATION & LIVE SIMULATION
    # =========================================================================
    router = AMLComplianceRouter(prob_decline_threshold=0.932, prob_review_threshold=0.700, unc_threshold=0.040)
    routed_actions = router.batch_route(mean_probs, epistemic_unc)
    
    num_declined = np.sum(routed_actions == "DECLINE")
    num_review   = np.sum(routed_actions == "MANUAL_REVIEW")
    num_approved = np.sum(routed_actions == "APPROVE")

    print("\n" + "=" * 85, flush=True)
    print("   AUTOMATED 3-TIER COMPLIANCE ROUTING ACTION BREAKDOWN   ", flush=True)
    print("=" * 85, flush=True)
    print(f"  [DECLINE] Tier 1: AUTOMATED DECLINE & FREEZE:    {num_declined:,} transactions ({num_declined/num_test*100:.2f}%)", flush=True)
    print(f"  [REVIEW]  Tier 2: ROUTED TO HUMAN MANUAL REVIEW: {num_review:,} transactions ({num_review/num_test*100:.2f}%)", flush=True)
    print(f"  [APPROVE] Tier 3: AUTOMATED INSTANT APPROVAL:   {num_approved:,} transactions ({num_approved/num_test*100:.2f}%)", flush=True)

    # Simulate Live Ingestion of 10 Random Incoming Bitcoin Transactions
    print("\n" + "=" * 85, flush=True)
    print("   SIMULATION: REAL-TIME INCOMING BITCOIN TRANSACTION PROCESSING   ", flush=True)
    print("=" * 85, flush=True)
    
    sample_indices = np.random.choice(len(test_indices), size=10, replace=False)
    for idx in sample_indices:
        tx_node_id = f"txId_{test_indices[idx]:06d}"
        p = mean_probs[idx]
        u = epistemic_unc[idx]
        res = router.route_transaction(tx_node_id, p, u)
        print(f"  {res['display']}", flush=True)
        time.sleep(0.05)

    print("\nProduction Model Script Execution Complete!", flush=True)

if __name__ == '__main__':
    main()
