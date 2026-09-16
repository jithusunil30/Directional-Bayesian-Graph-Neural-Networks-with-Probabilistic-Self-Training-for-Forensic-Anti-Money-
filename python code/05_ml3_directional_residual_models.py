import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GINConv

# =========================================================================
# 1. Directional Residual Graph Neural Networks (Dir-ResGNN)
# =========================================================================

class DirResGCNModel(nn.Module):
    """
    Directional Residual Graph Convolutional Network (Dir-ResGCN).
    Processes both forward payment flow (inflow) and reverse flow (dispersion)
    with residual skip-connections to prevent over-smoothing.
    """
    def __init__(self, in_features, hidden_dim=128, dropout_rate=0.3):
        super(DirResGCNModel, self).__init__()
        self.conv1_fwd = GCNConv(in_features, hidden_dim)
        self.conv1_rev = GCNConv(in_features, hidden_dim)
        self.res1 = nn.Linear(in_features, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)

        self.conv2 = GCNConv(hidden_dim, 64)
        self.res2 = nn.Linear(hidden_dim, 64)
        self.bn2 = nn.BatchNorm1d(64)

        self.out = nn.Linear(64, 1)
        self.dropout_rate = dropout_rate

    def forward(self, x, edge_index):
        rev_edge_index = edge_index.flip(0)
        h1 = self.conv1_fwd(x, edge_index) + self.conv1_rev(x, rev_edge_index) + self.res1(x)
        h1 = F.relu(self.bn1(h1))
        h1 = F.dropout(h1, p=self.dropout_rate, training=self.training)

        h2 = self.conv2(h1, edge_index) + self.res2(h1)
        h2 = F.relu(self.bn2(h2))
        h2 = F.dropout(h2, p=self.dropout_rate, training=self.training)

        return self.out(h2).squeeze(-1)

class DirResGATModel(nn.Module):
    """
    Directional Residual Graph Attention Network (Dir-ResGAT).
    Multi-head dynamic attention coefficients with residual projection.
    """
    def __init__(self, in_features, hidden_dim=32, heads=2, dropout_rate=0.3):
        super(DirResGATModel, self).__init__()
        self.gat1 = GATConv(in_features, hidden_dim, heads=heads, dropout=dropout_rate)
        self.res1 = nn.Linear(in_features, hidden_dim * heads)
        self.bn1 = nn.BatchNorm1d(hidden_dim * heads)

        self.gat2 = GATConv(hidden_dim * heads, 32, heads=1, concat=False, dropout=dropout_rate)
        self.res2 = nn.Linear(hidden_dim * heads, 32)
        self.bn2 = nn.BatchNorm1d(32)

        self.out = nn.Linear(32, 1)
        self.dropout_rate = dropout_rate

    def forward(self, x, edge_index):
        h1 = self.gat1(x, edge_index) + self.res1(x)
        h1 = F.elu(self.bn1(h1))
        h1 = F.dropout(h1, p=self.dropout_rate, training=self.training)

        h2 = self.gat2(h1, edge_index) + self.res2(h1)
        h2 = F.elu(self.bn2(h2))
        h2 = F.dropout(h2, p=self.dropout_rate, training=self.training)

        return self.out(h2).squeeze(-1)

class DirResGraphSAGEModel(nn.Module):
    """
    Directional Residual GraphSAGE (Dir-ResSAGE).
    Aggregates upstream inflows and downstream outflows via dual spatial pooling.
    """
    def __init__(self, in_features, hidden_dim=128, dropout_rate=0.3):
        super(DirResGraphSAGEModel, self).__init__()
        self.sage1_fwd = SAGEConv(in_features, hidden_dim, aggr='mean')
        self.sage1_rev = SAGEConv(in_features, hidden_dim, aggr='mean')
        self.res1 = nn.Linear(in_features, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)

        self.sage2 = SAGEConv(hidden_dim, 64, aggr='mean')
        self.res2 = nn.Linear(hidden_dim, 64)
        self.bn2 = nn.BatchNorm1d(64)

        self.out = nn.Linear(64, 1)
        self.dropout_rate = dropout_rate

    def forward(self, x, edge_index):
        rev_edge_index = edge_index.flip(0)
        h1 = self.sage1_fwd(x, edge_index) + self.sage1_rev(x, rev_edge_index) + self.res1(x)
        h1 = F.relu(self.bn1(h1))
        h1 = F.dropout(h1, p=self.dropout_rate, training=self.training)

        h2 = self.sage2(h1, edge_index) + self.res2(h1)
        h2 = F.relu(self.bn2(h2))
        h2 = F.dropout(h2, p=self.dropout_rate, training=self.training)

        return self.out(h2).squeeze(-1)

class DirGINModel(nn.Module):
    def __init__(self, in_features, hidden_dim=128, dropout_rate=0.3):
        super(DirGINModel, self).__init__()
        mlp1 = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
        self.gin1 = GINConv(mlp1, train_eps=True)
        self.res1 = nn.Linear(in_features, hidden_dim)

        mlp2 = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        self.gin2 = GINConv(mlp2, train_eps=True)
        self.res2 = nn.Linear(hidden_dim, 64)

        self.out = nn.Linear(64, 1)
        self.dropout_rate = dropout_rate

    def forward(self, x, edge_index):
        h1 = self.gin1(x, edge_index) + self.res1(x)
        h1 = F.dropout(h1, p=self.dropout_rate, training=self.training)

        h2 = self.gin2(h1, edge_index) + self.res2(h1)
        h2 = F.dropout(h2, p=self.dropout_rate, training=self.training)

        return self.out(h2).squeeze(-1)

# =========================================================================
# 2. Bayesian Directional Residual Graph Neural Networks (MC Variational)
# =========================================================================

class BayesianDirGCNModel(nn.Module):
    def __init__(self, in_features, hidden_dim=128, mc_dropout_rate=0.3):
        super(BayesianDirGCNModel, self).__init__()
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
    def mc_predict(self, x, edge_index, mask, num_samples=30):
        self.eval()
        samples = []
        for _ in range(num_samples):
            logits = self.forward(x, edge_index, force_mc_dropout=True)[mask]
            probs = torch.sigmoid(logits).cpu().numpy()
            samples.append(probs)

        samples = np.array(samples)
        mean_prob = np.mean(samples, axis=0)
        epistemic_unc = np.var(samples, axis=0)
        
        eps = 1e-7
        p_clipped = np.clip(samples, eps, 1.0 - eps)
        sample_entropy = -(p_clipped * np.log2(p_clipped) + (1.0 - p_clipped) * np.log2(1.0 - p_clipped))
        aleatoric_unc = np.mean(sample_entropy, axis=0)
        return mean_prob, epistemic_unc, aleatoric_unc

class BayesianDirGATModel(nn.Module):
    def __init__(self, in_features, hidden_dim=32, heads=2, mc_dropout_rate=0.3):
        super(BayesianDirGATModel, self).__init__()
        self.gat1 = GATConv(in_features, hidden_dim, heads=heads, dropout=mc_dropout_rate)
        self.res1 = nn.Linear(in_features, hidden_dim * heads)
        self.bn1 = nn.BatchNorm1d(hidden_dim * heads)

        self.gat2 = GATConv(hidden_dim * heads, 32, heads=1, concat=False, dropout=mc_dropout_rate)
        self.res2 = nn.Linear(hidden_dim * heads, 32)
        self.bn2 = nn.BatchNorm1d(32)

        self.out = nn.Linear(32, 1)
        self.mc_dropout_rate = mc_dropout_rate

    def forward(self, x, edge_index, force_mc_dropout=False):
        training_mode = self.training or force_mc_dropout
        h1 = self.gat1(x, edge_index) + self.res1(x)
        h1 = F.elu(self.bn1(h1))
        h1 = F.dropout(h1, p=self.mc_dropout_rate, training=training_mode)

        h2 = self.gat2(h1, edge_index) + self.res2(h1)
        h2 = F.elu(self.bn2(h2))
        h2 = F.dropout(h2, p=self.mc_dropout_rate, training=training_mode)

        return self.out(h2).squeeze(-1)

    @torch.no_grad()
    def mc_predict(self, x, edge_index, mask, num_samples=30):
        self.eval()
        samples = []
        for _ in range(num_samples):
            logits = self.forward(x, edge_index, force_mc_dropout=True)[mask]
            probs = torch.sigmoid(logits).cpu().numpy()
            samples.append(probs)

        samples = np.array(samples)
        mean_prob = np.mean(samples, axis=0)
        epistemic_unc = np.var(samples, axis=0)
        
        eps = 1e-7
        p_clipped = np.clip(samples, eps, 1.0 - eps)
        sample_entropy = -(p_clipped * np.log2(p_clipped) + (1.0 - p_clipped) * np.log2(1.0 - p_clipped))
        aleatoric_unc = np.mean(sample_entropy, axis=0)
        return mean_prob, epistemic_unc, aleatoric_unc

class BayesianDirGraphSAGEModel(nn.Module):
    def __init__(self, in_features, hidden_dim=128, mc_dropout_rate=0.3):
        super(BayesianDirGraphSAGEModel, self).__init__()
        self.sage1_fwd = SAGEConv(in_features, hidden_dim, aggr='mean')
        self.sage1_rev = SAGEConv(in_features, hidden_dim, aggr='mean')
        self.res1 = nn.Linear(in_features, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)

        self.sage2 = SAGEConv(hidden_dim, 64, aggr='mean')
        self.res2 = nn.Linear(hidden_dim, 64)
        self.bn2 = nn.BatchNorm1d(64)

        self.out = nn.Linear(64, 1)
        self.mc_dropout_rate = mc_dropout_rate

    def forward(self, x, edge_index, force_mc_dropout=False):
        training_mode = self.training or force_mc_dropout
        rev_edge_index = edge_index.flip(0)
        h1 = self.sage1_fwd(x, edge_index) + self.sage1_rev(x, rev_edge_index) + self.res1(x)
        h1 = F.relu(self.bn1(h1))
        h1 = F.dropout(h1, p=self.mc_dropout_rate, training=training_mode)

        h2 = self.sage2(h1, edge_index) + self.res2(h1)
        h2 = F.relu(self.bn2(h2))
        h2 = F.dropout(h2, p=self.mc_dropout_rate, training=training_mode)

        return self.out(h2).squeeze(-1)

    @torch.no_grad()
    def mc_predict(self, x, edge_index, mask, num_samples=30):
        self.eval()
        samples = []
        for _ in range(num_samples):
            logits = self.forward(x, edge_index, force_mc_dropout=True)[mask]
            probs = torch.sigmoid(logits).cpu().numpy()
            samples.append(probs)

        samples = np.array(samples)
        mean_prob = np.mean(samples, axis=0)
        epistemic_unc = np.var(samples, axis=0)
        
        eps = 1e-7
        p_clipped = np.clip(samples, eps, 1.0 - eps)
        sample_entropy = -(p_clipped * np.log2(p_clipped) + (1.0 - p_clipped) * np.log2(1.0 - p_clipped))
        aleatoric_unc = np.mean(sample_entropy, axis=0)
        return mean_prob, epistemic_unc, aleatoric_unc
