import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GINConv

# =========================================================================
# 1. Deterministic Graph Neural Networks (GNNs)
# =========================================================================

class GCNModel(nn.Module):
    def __init__(self, in_features, hidden_dim=128, dropout_rate=0.3):
        super(GCNModel, self).__init__()
        self.conv1 = GCNConv(in_features, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.conv2 = GCNConv(hidden_dim, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.out = nn.Linear(64, 1)
        self.dropout_rate = dropout_rate

    def forward(self, x, edge_index):
        h = F.relu(self.bn1(self.conv1(x, edge_index)))
        h = F.dropout(h, p=self.dropout_rate, training=self.training)
        h = F.relu(self.bn2(self.conv2(h, edge_index)))
        h = F.dropout(h, p=self.dropout_rate, training=self.training)
        return self.out(h).squeeze(-1)

class GATModel(nn.Module):
    def __init__(self, in_features, hidden_dim=32, heads=2, dropout_rate=0.3):
        super(GATModel, self).__init__()
        self.gat1 = GATConv(in_features, hidden_dim, heads=heads, dropout=dropout_rate)
        self.bn1 = nn.BatchNorm1d(hidden_dim * heads)
        self.gat2 = GATConv(hidden_dim * heads, 32, heads=1, concat=False, dropout=dropout_rate)
        self.bn2 = nn.BatchNorm1d(32)
        self.out = nn.Linear(32, 1)
        self.dropout_rate = dropout_rate

    def forward(self, x, edge_index):
        h = F.elu(self.bn1(self.gat1(x, edge_index)))
        h = F.dropout(h, p=self.dropout_rate, training=self.training)
        h = F.elu(self.bn2(self.gat2(h, edge_index)))
        h = F.dropout(h, p=self.dropout_rate, training=self.training)
        return self.out(h).squeeze(-1)

class GraphSAGEModel(nn.Module):
    def __init__(self, in_features, hidden_dim=128, dropout_rate=0.3):
        super(GraphSAGEModel, self).__init__()
        self.sage1 = SAGEConv(in_features, hidden_dim, aggr='mean')
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.sage2 = SAGEConv(hidden_dim, 64, aggr='mean')
        self.bn2 = nn.BatchNorm1d(64)
        self.out = nn.Linear(64, 1)
        self.dropout_rate = dropout_rate

    def forward(self, x, edge_index):
        h = F.relu(self.bn1(self.sage1(x, edge_index)))
        h = F.dropout(h, p=self.dropout_rate, training=self.training)
        h = F.relu(self.bn2(self.sage2(h, edge_index)))
        h = F.dropout(h, p=self.dropout_rate, training=self.training)
        return self.out(h).squeeze(-1)

class GINModel(nn.Module):
    def __init__(self, in_features, hidden_dim=128, dropout_rate=0.3):
        super(GINModel, self).__init__()
        mlp1 = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
        self.gin1 = GINConv(mlp1, train_eps=True)
        mlp2 = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        self.gin2 = GINConv(mlp2, train_eps=True)
        self.out = nn.Linear(64, 1)
        self.dropout_rate = dropout_rate

    def forward(self, x, edge_index):
        h = self.gin1(x, edge_index)
        h = F.dropout(h, p=self.dropout_rate, training=self.training)
        h = self.gin2(h, edge_index)
        h = F.dropout(h, p=self.dropout_rate, training=self.training)
        return self.out(h).squeeze(-1)

# =========================================================================
# 2. Bayesian Graph Neural Networks (Monte Carlo Variational Inference)
# =========================================================================

class BayesianGCNModel(nn.Module):
    def __init__(self, in_features, hidden_dim=128, mc_dropout_rate=0.3):
        super(BayesianGCNModel, self).__init__()
        self.conv1 = GCNConv(in_features, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.conv2 = GCNConv(hidden_dim, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.out = nn.Linear(64, 1)
        self.mc_dropout_rate = mc_dropout_rate

    def forward(self, x, edge_index, force_mc_dropout=False):
        training_mode = self.training or force_mc_dropout
        h = F.relu(self.bn1(self.conv1(x, edge_index)))
        h = F.dropout(h, p=self.mc_dropout_rate, training=training_mode)
        h = F.relu(self.bn2(self.conv2(h, edge_index)))
        h = F.dropout(h, p=self.mc_dropout_rate, training=training_mode)
        return self.out(h).squeeze(-1)

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

class BayesianGATModel(nn.Module):
    def __init__(self, in_features, hidden_dim=32, heads=2, mc_dropout_rate=0.3):
        super(BayesianGATModel, self).__init__()
        self.gat1 = GATConv(in_features, hidden_dim, heads=heads, dropout=mc_dropout_rate)
        self.bn1 = nn.BatchNorm1d(hidden_dim * heads)
        self.gat2 = GATConv(hidden_dim * heads, 32, heads=1, concat=False, dropout=mc_dropout_rate)
        self.bn2 = nn.BatchNorm1d(32)
        self.out = nn.Linear(32, 1)
        self.mc_dropout_rate = mc_dropout_rate

    def forward(self, x, edge_index, force_mc_dropout=False):
        training_mode = self.training or force_mc_dropout
        h = F.elu(self.bn1(self.gat1(x, edge_index)))
        h = F.dropout(h, p=self.mc_dropout_rate, training=training_mode)
        h = F.elu(self.bn2(self.gat2(h, edge_index)))
        h = F.dropout(h, p=self.mc_dropout_rate, training=training_mode)
        return self.out(h).squeeze(-1)

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

class BayesianGraphSAGEModel(nn.Module):
    def __init__(self, in_features, hidden_dim=128, mc_dropout_rate=0.3):
        super(BayesianGraphSAGEModel, self).__init__()
        self.sage1 = SAGEConv(in_features, hidden_dim, aggr='mean')
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.sage2 = SAGEConv(hidden_dim, 64, aggr='mean')
        self.bn2 = nn.BatchNorm1d(64)
        self.out = nn.Linear(64, 1)
        self.mc_dropout_rate = mc_dropout_rate

    def forward(self, x, edge_index, force_mc_dropout=False):
        training_mode = self.training or force_mc_dropout
        h = F.relu(self.bn1(self.sage1(x, edge_index)))
        h = F.dropout(h, p=self.mc_dropout_rate, training=training_mode)
        h = F.relu(self.bn2(self.sage2(h, edge_index)))
        h = F.dropout(h, p=self.mc_dropout_rate, training=training_mode)
        return self.out(h).squeeze(-1)

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
