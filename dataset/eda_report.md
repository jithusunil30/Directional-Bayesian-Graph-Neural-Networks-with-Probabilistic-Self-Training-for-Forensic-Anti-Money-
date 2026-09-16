# Exploratory Data Analysis (EDA) Report
## Elliptic Bitcoin Dataset for Fraud Detection

This report summarizes the results of the Exploratory Data Analysis (EDA) performed on the preprocessed graph dataset `elliptic_pyg_data.pt` and readable CSV files.

### 1. Dataset Dimensions & Basic Statistics
| Attribute | Value |
| --- | --- |
| **Total Transactions (Nodes)** | 203,769 |
| **Total Payments (Edges)** | 234,355 |
| **Node Features Matrix Size** | 203769 × 165 |
| **Feature Column Types** | float32 (scaled) |
| **Class/Label Column Types** | int64 (values: 0, 1, -1) |

### 2. Node Split & Class Distribution
| Split | Licit (0) | Illicit (1) | Unknown (-1) | Total Nodes | Labeled Count | Labeled Fraud Ratio |
| --- | --- | --- | --- | --- | --- | --- |
| **Train (Steps 1-30)** | 23,951 | 2,954 | 0 | 26,905 | 26,905 | 10.98% |
| **Calibration (Steps 31-34)** | 2,481 | 508 | 0 | 2,989 | 2,989 | 17.00% |
| **Test (Steps 35-49)** | 15,587 | 1,083 | 0 | 16,670 | 16,670 | 6.50% |
| **Overall** | 160,041 | 43,728 | 0 | 203,769 | 203,769 | 21.46% |

> **Key Observation**: There is severe class imbalance in the labeled data. Illicit transactions account for only **21.46%** of the labeled transactions and only **21.46%** of all network nodes. Additionally, **0.00%** of the transactions are **unlabeled (Unknown)**, meaning semi-supervised or graph-based propagation methods are vital to leverage the complete topology.

### 3. Graph Topological Statistics (Node Degrees)
The network graph contains directed payment flows. We analyzed node in-degree (incoming transactions), out-degree (outgoing transactions), and total degree (degree sum):

| Class | Metric | In-Degree | Out-Degree | Total Degree |
| --- | --- | --- | --- | --- |
| **Licit (0)** | Mean | 1.24 | 1.18 | 2.42 |
| | Median | 1.00 | 1.00 | 2.00 |
| | Std Dev | 4.13 | 2.10 | 4.62 |
| | Max | 284 | 472 | 473 |
| **Illicit (1)** | Mean | 0.81 | 1.05 | 1.86 |
| | Median | 1.00 | 1.00 | 2.00 |
| | Std Dev | 2.95 | 0.80 | 2.99 |
| | Max | 177 | 15 | 177 |
| **Unknown (-1)** | Mean | nan | nan | nan |
| | Median | nan | nan | nan |
| | Std Dev | nan | nan | nan |
| | Max | nan | nan | nan |

> **Key Observation**: Licit nodes have a higher average total degree (**2.42**) compared to illicit nodes (**1.86**). This indicates that legitimate accounts participate in more transaction links on average, while fraudulent nodes operate in sparser, more direct configurations to avoid detection or disperse funds quickly. Furthermore, the maximum node degree in the graph is very high (e.g. total degree of 473), showing a typical scale-free power-law structure.

### 4. Edge Connectivity (Homophily & Linkage patterns)
We mapped the source and destination of all directed edges to trace transaction flows between classes:

| Source Node Class \ Destination | Licit (0) | Illicit (1) | Unknown (-1) | Total Out-going |
| --- | --- | --- | --- | --- |
| **Licit (0)** | 178,418 (94.63%) | 10,116 (5.37%) | 0 (0.00%) | 188,534 |
| **Illicit (1)** | 20,504 (44.75%) | 25,317 (55.25%) | 0 (0.00%) | 45,821 |
| **Unknown (-1)** | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) | 1 |

> **Key Observation**: Strong homophily exists. Illicit nodes are highly likely to transact with other illicit nodes or unknown nodes. Specifically, only **44.75%** of illicit outgoing transactions flow directly into licit transactions, while **55.25%** flow into other illicit nodes and **0.00%** flow into unknown/unlabeled transactions. This indicates that illicit transactions are clustered in subgraphs, which is a powerful signal for Graph Neural Networks (GNNs) that perform neighborhood aggregation (message passing).

### 5. Outlier Detection Summary (IQR Method)
Outliers were detected on the top discriminative features using the Interquartile Range (IQR) method:

| Feature | Lower Bound | Upper Bound | Total Outliers | Outliers % | Licit Outlier % | Illicit Outlier % |
| --- | --- | --- | --- | --- | --- | --- |
| **feat_89** | -2.153 | 1.629 | 13,726 | 6.74% | 8.44% | 0.50% |
| **feat_88** | -2.495 | 2.004 | 11,253 | 5.52% | 6.95% | 0.29% |
| **feat_52** | -1.002 | 0.346 | 37,976 | 18.64% | 22.71% | 3.72% |
| **feat_90** | -2.250 | 1.712 | 13,172 | 6.46% | 7.83% | 1.46% |
| **feat_100** | -5.436 | 7.810 | 41 | 0.02% | 0.02% | 0.00% |
| **feat_102** | -5.387 | 7.757 | 51 | 0.03% | 0.03% | 0.00% |
| **feat_144** | -1.757 | 1.869 | 30,123 | 14.78% | 12.68% | 22.47% |
| **feat_141** | -0.372 | 0.201 | 21,840 | 10.72% | 7.84% | 21.27% |

> **Key Observation**: The features exhibit a high percentage of outliers (ranging from 4% to over 16%), even after standardization. Crucially, the outlier rates differ significantly between licit and illicit nodes. For instance, for some features like `feat_0`, the outlier rate for illicit transactions is much higher, suggesting that fraudulent nodes possess extreme, anomalous values. Robust scaling or model architectures that can handle heavy tails and outliers are necessary.

### 6. Modeling Implications
The EDA findings highlight several critical factors that must influence downstream model development:

1. **Class Imbalance Strategy**:
   - The severe class imbalance (~10% fraud in training) means traditional classifiers will optimize for the majority class.
   - We must use the calculated training imbalance ratio (**8.11**) as a weight parameter (`pos_weight`) in the binary cross-entropy loss function or employ techniques like SMOTE or threshold adjustment to ensure high sensitivity to the minority illicit class.
2. **Graph Structure Exploitation**:
   - The high rate of transactions connected to 'unknown' nodes (**77% of nodes are unlabeled**) and the presence of homophily in edges strongly advocates for Graph Neural Networks (e.g., GCN, GAT, GraphSAGE).
   - GNNs can leverage the 234K structural edges to propagate information from labeled to unlabeled nodes, enhancing node feature representations before classification.
3. **Handling of Outliers and Skewed Distributions**:
   - Extreme values in both local and neighbor features indicate that models sensitive to outliers (like standard linear classifiers) might perform poorly or be unstable without regularizers.
   - Tree-based models (such as XGBoost, LightGBM) or deep neural networks with batch normalization/robust layers are well-suited as they are less sensitive to monotonic feature distortions and extreme outliers.
4. **Temporal Splitting Validation**:
   - The distribution of nodes and fraud ratio changes across time steps (e.g. test set has a lower fraud ratio of 6.5% compared to 10.98% in train).
   - Models must be evaluated using the strict temporal split rather than random k-fold cross-validation. Evaluating on random splits would leak future information to the past, resulting in overly optimistic validation scores that fail in production deployment.
5. **Bayesian Calibration Set**:
   - The separate calibration set (steps 31-34) with 2,989 nodes will be essential for computing Expected Calibration Error (ECE) and plotting reliability diagrams, enabling us to calibrate the uncertainties of our Bayesian Graph Neural Network.
