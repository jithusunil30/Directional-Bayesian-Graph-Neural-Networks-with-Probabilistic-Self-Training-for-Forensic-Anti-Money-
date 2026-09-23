import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve, auc, confusion_matrix
from torch_geometric.nn import GCNConv

# =========================================================================
# 1. ARCHITECTURE: Directional Residual Bayesian GNN (Dir-ResGNN)
# =========================================================================

class BayesianDirResGCN(nn.Module):
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
        
        h1 = self.conv1_fwd(x, edge_index) + self.conv1_rev(x, rev_edge_index) + self.res1(x)
        h1 = F.relu(self.bn1(h1))
        h1 = F.dropout(h1, p=self.mc_dropout_rate, training=training_mode)

        h2 = self.conv2(h1, edge_index) + self.res2(h1)
        h2 = F.relu(self.bn2(h2))
        h2 = F.dropout(h2, p=self.mc_dropout_rate, training=training_mode)

        return self.out(h2).squeeze(-1)

    @torch.no_grad()
    def mc_predict(self, x, edge_index, mask=None, num_passes=10):
        self.eval()
        samples = []
        for _ in range(num_passes):
            logits = self.forward(x, edge_index, force_mc_dropout=True)
            if mask is not None:
                logits = logits[mask]
            probs = torch.sigmoid(logits).cpu().numpy()
            samples.append(probs)

        samples = np.array(samples)  # (T, N)
        mean_prob = np.mean(samples, axis=0)
        epistemic_unc = np.var(samples, axis=0)
        return mean_prob, epistemic_unc

# =========================================================================
# 2. AUTOMATED COMPLIANCE DECISION ROUTER (DECLINE / REVIEW / APPROVE)
# =========================================================================

class AMLComplianceRouter:
    def __init__(self, prob_decline_threshold=0.932, prob_review_threshold=0.700, unc_threshold=0.040):
        self.prob_decline = prob_decline_threshold
        self.prob_review = prob_review_threshold
        self.unc_threshold = unc_threshold

    def route_transaction(self, tx_id, probability, epistemic_unc):
        if probability >= self.prob_decline and epistemic_unc < self.unc_threshold:
            action = "DECLINE & FREEZE"
            tier = "Tier 1 (High Risk & Confident)"
            reason = f"High Illicit Probability ({probability:.4f} >= {self.prob_decline}) & Low Uncertainty ({epistemic_unc:.5f})"
            status_code = "DECLINED"
        elif probability >= self.prob_review or epistemic_unc >= self.unc_threshold:
            action = "ROUTE TO MANUAL REVIEW"
            tier = "Tier 2 (Ambiguous / Novel Attack)"
            reason = f"Moderate Risk ({probability:.4f}) OR High Model Uncertainty ({epistemic_unc:.5f} >= {self.unc_threshold})"
            status_code = "REVIEW"
        else:
            action = "APPROVE & BROADCAST"
            tier = "Tier 3 (Verified Licit)"
            reason = f"Low Illicit Risk ({probability:.4f} < {self.prob_review}) & Low Uncertainty ({epistemic_unc:.5f})"
            status_code = "APPROVED"
        
        return {
            "tx_id": tx_id,
            "probability": float(probability),
            "epistemic_unc": float(epistemic_unc),
            "action": action,
            "tier": tier,
            "status_code": status_code,
            "reason": reason
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
