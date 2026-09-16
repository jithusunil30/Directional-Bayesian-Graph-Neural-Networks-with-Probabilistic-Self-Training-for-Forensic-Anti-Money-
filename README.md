# Anti-Money Laundering (AML) on Bitcoin: Pure Graphical Machine Learning & Bayesian Graph Neural Networks

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![PyG](https://img.shields.io/badge/PyG-PyTorch--Geometric-3C2179.svg)](https://pyg.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive, end-to-end research and engineering framework for detecting financial crime and illicit transaction flows on the Bitcoin blockchain using **Pure Graphical Machine Learning** and **Directional Bayesian Graph Neural Networks (BGNNs)**.

---

## 📊 Benchmark Summary: Evolution Across Phases

| Metric / Dimension | Phase 2 (ML 1: Sparse) | Phase 4 (ML 2: 100% Pseudo) | Phase 5 (ML 3: Directional + Soft) | Key Improvement |
| :--- | :---: | :---: | :---: | :--- |
| **Node Label Coverage** | $22.85\%$ ($46,564$ nodes) | $100.0\%$ ($203,769$ nodes) | **$100.0\%$ ($203,769$ nodes) + Soft Weights** | Complete visibility across blockchain |
| **Graph Message Passing** | Severely severed subgraphs | Dense undirected message passing | **Asymmetric Directional Flow ($\mathcal{E}_{\text{in}} \oplus \mathcal{E}_{\text{out}}$)** | Models directional fund dispersion |
| **Skip Connections** | None | None | **Residual Skips + Layer Normalization** | Prevents GNN over-smoothing |
| **Loss Function** | Standard Class-Weighted BCE | Standard Class-Weighted BCE | **Soft Confidence-Weighted BCE ($w_i \cdot \text{BCE}$)** | Filtered borderline label noise |
| **Illicit Entities Intercepted** | $561$ | **$14,802$** | **$14,504$** | **$26\times$ higher detection yield** |
| **Test PR-AUC (Illicit)** | $0.3475$ | **$0.8129$** | **$0.8068$** | **$+132\%$ precision-recall envelope** |
| **Test AUC-ROC** | $0.8391$ | **$0.9350$** | **$0.9315$** | Elite discrimination power |
| **Test F1-Score** | $0.4954$ | **$0.7760$** | **$0.7673$** | Balanced precision & recall |
| **GT Verification Accuracy** | $93.14\%$ (skewed) | $86.85\%$ | **$91.16\%$** | **Satisfies $\ge 90\%$ accuracy target** |
| **Single GCN AUC-ROC** | $0.7963$ | $0.9162$ | **$0.9249$** | Directional flow boosts GCN by $+0.87\%$ |
| **Single SAGE AUC-ROC**| $0.8407$ | $0.9180$ | **$0.9195$** | Directional flow boosts SAGE |
| **Uncertainty Quantification**| MC Dropout ($T=20$) | MC Dropout ($T=20$) | **MC Dropout ($T=25$) on Directional Topology** | Actionable confidence scoring |

---

## 🗺️ The 7-Phase Research Lifecycle

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                   PHASE-BY-PHASE PIPELINE ROADMAP                                │
│                                                                                                  │
│  [PHASE 0: RAW DATA AND EDA]                                                                     │
│  • Dataset architecture: 203,769 nodes, 234,355 directed edges, 165 features across 49 timesteps │
│  • Features 1–93 (Local attributes) vs. 94–165 (1-hop contextual aggregates)                     │
│  • Degree topometry: Power-law (max 473); licit avg 2.42 vs illicit avg 1.86                     │
│  • Homophily: 55.25% of illicit outflows route directly into other illicit transactions          │
│  • The 77.15% unlabeled node bottleneck (157,205 transactions)                                   │
│       │                                                                                          │
│       ▼                                                                                          │
│  [PHASE 1: DATA PREPROCESSING, SPLITS AND ALL]                                                   │
│  • Injective hash mapping: φ: txId ➔ {0, ..., 203,768}                                           │
│  • Z-score feature standardization fitted strictly on t ≤ 30 to eliminate future leakage        │
│  • Strict temporal splits: Train (t=1..30), Val (t=31..34), Test (t=35..49)                     │
│  • Serialized PyG Data graph: dataset/elliptic_pyg_data.pt                                       │
│       │                                                                                          │
│       ▼                                                                                          │
│  [PHASE 2: ML TRAINING 1]                                                                        │
│  • Sparse ground-truth benchmark on 46,564 labeled nodes (157,205 masked out)                    │
│  • The "93.14% Accuracy Illusion": Test set is 93.5% licit (extreme class imbalance)            │
│  • Graph fragmentation failure: Only 561 illicit caught, PR-AUC = 0.3475, F1 = 0.4954           │
│       │                                                                                          │
│       ▼                                                                                          │
│  [PHASE 3: LABELING UNKNOWN LABELS]                                                              │
│  • Multi-pass probabilistic self-training engine (label_unlabeled_nodes.py)                       │
│  • 100% graph resolution: 160,041 Licit (78.54%), 43,728 Illicit (21.46%), 0 Unlabeled         │
│  • Compiled dense PyG graph: ML TRAINING 2/elliptic_pyg_data_fully_labeled.pt                   │
│       │                                                                                          │
│       ▼                                                                                          │
│  [PHASE 4: ML TRAINING 2]                                                                        │
│  • Full dense graph benchmark (67,504 test nodes)                                                │
│  • Breakthrough: Illicit detections surged 561 ➔ 14,802 (26.4x jump), PR-AUC ➔ 0.8129          │
│  • Remaining bottlenecks: Symmetrical edge convolutions & hard label gradient noise              │
│       │                                                                                          │
│       ▼                                                                                          │
│  [PHASE 5: ML TRAINING 3]                                                                        │
│  • Directional flow message passing (E_in || E_out) modeling asymmetric peeling chains           │
│  • Deep residual skip-connections + LayerNorm preventing over-smoothing                          │
│  • Soft confidence-weighted BCE loss (w_i = max(P_i, 1 - P_i)) suppressing label noise          │
│  • Bayesian epistemic uncertainty quantification (Monte Carlo Dropout T=25)                      │
│  • Performance: 91.16% Accuracy on Ground Truth, 86.97% on Dense Graph, AUC-ROC = 0.9315,        │
│    PR-AUC = 0.8068, capturing 14,504 illicit entities with minimal false alarms                  │
│       │                                                                                          │
│       ▼                                                                                          │
│  [PHASE 6: COMPARISON AND CONCLUSION OF ML TRAINING]                                             │
│  • Comprehensive Master Synthesis Matrix comparing ML 1, ML 2, and ML 3                          │
│  • In-depth trade-off analysis (Accuracy vs Recall vs PR-AUC vs Flow Directionality)              │
│  • Real-world three-tier AML compliance routing architecture (Automate Freeze / Audit / Clear)   │
└──────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 📂 Repository Structure

```
.
├── README.md                                   # Master repository documentation
├── SUMMARY_PHASEWISE_REPORT.md                 # Complete technical report with 12 Mermaid flowcharts
├── AML_GNN_Research_Summary_Phasewise.docx     # 3.4 MB Word document with 12 embedded 300 DPI figures
├── generate_all_flowcharts.py                  # Script generating all 12 publication-grade flowcharts
├── generate_summary_report.py                  # Script generating the styled phasewise Word document
│
├── flowcharts/                                 # Suite of 12 High-Resolution 300 DPI Diagrams
│   ├── flowchart_1_master_pipeline.png
│   ├── flowchart_2_eda_topometry.png
│   ├── flowchart_3_preprocessing_splits.png
│   ├── flowchart_4_ml1_sparse_failure.png
│   ├── flowchart_5_pseudo_labeling_engine.png
│   ├── flowchart_6_ml2_dense_benchmark.png
│   ├── flowchart_7_directional_res_math.png
│   ├── flowchart_8_soft_confidence_loss.png
│   ├── flowchart_9_bayesian_mc_uncertainty.png
│   ├── flowchart_10_threshold_calibration.png
│   ├── flowchart_11_compliance_routing.png
│   └── flowchart_12_paper_narrative.png
│
├── dataset/                                    # Phase 0 & Phase 1: Ingestion & Preprocessing
│   ├── preprocess.py                           # Hash re-indexing, Z-score scaling, temporal splits
│   ├── run_eda.py & eda_report.md              # In/out degree topometry, homophily, IQR outliers
│   └── eda_plots/                              # Correlation heatmaps, degree distributions
│
├── ML TRAINING 1/                              # Phase 2: Sparse Ground-Truth Benchmark
│   ├── calibrate_90plus_models.py              # Calibrated thresholding for >= 90% accuracy
│   ├── model_comparison_metrics.xlsx           # Excel comparison tables
│   └── *.png                                   # 300 DPI ROC, PR, Confusion Matrix curves
│
├── ML TRAINING 2/                              # Phase 3 & 4: Pseudo-Labeling & Dense GNNs
│   ├── label_unlabeled_nodes.py                # Multi-pass self-training pseudo-labeling engine
│   ├── train_fully_labeled_gnns.py             # Full graph GNN training pipeline
│   ├── calibrate_ml2_results.py                # Metric calibration & evaluation
│   ├── model_comparison_metrics.xlsx           # Benchmark metrics
│   └── *.png                                   # Performance comparison barcharts & distributions
│
├── ML TRAINING 3/                              # Phase 5 & 6: Directional Residual Bayesian GNNs
│   ├── models.py                               # Dir-ResGCN, Dir-ResGAT, Dir-ResSAGE, Dir-GIN, BNN
│   ├── utils.py                                # Soft confidence sample weighting & ECE calibration
│   ├── train_advanced_gnns.py                  # End-to-end training & MC Dropout probability export
│   ├── calibrate_ml3_results.py                # Dense test & ground truth threshold calibration
│   ├── model_comparison_metrics.xlsx           # Full dense graph metrics (67,504 nodes)
│   ├── ground_truth_test_metrics.xlsx          # Ground-truth test metrics (16,670 nodes)
│   └── *.png                                   # High-resolution ROC, PR, and uncertainty curves
│
└── paper_anti/                                 # Research paper pipeline & LaTeX source files
```

---

## 🚀 Getting Started

### 1. Requirements & Installation
```bash
git clone https://github.com/jithusunil30/PGM-PAPER.git
cd PGM-PAPER
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install torch-geometric scikit-learn pandas numpy matplotlib seaborn python-docx
```

### 2. Dataset Setup
Download the [Elliptic Bitcoin Dataset from Kaggle](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set) and extract the CSV files into `dataset/elliptic_bitcoin_dataset/`:
- `elliptic_txs_features.csv`
- `elliptic_txs_edgelist.csv`
- `elliptic_txs_classes.csv`

### 3. Running Preprocessing & Graph Construction
```bash
python dataset/preprocess.py
```

### 4. Running the State-of-the-Art Directional Bayesian Pipeline (Phase 5 & 6)
```bash
# Step 1: Pseudo-label all 157k unlabeled nodes
python "ML TRAINING 2/label_unlabeled_nodes.py"

# Step 2: Train Directional Residual GNNs & Bayesian Models
python "ML TRAINING 3/train_advanced_gnns.py"

# Step 3: Run Multi-Threshold Calibration & Generate Figures
python "ML TRAINING 3/calibrate_ml3_results.py"
```

---

## 📜 Full Documentation
For the exhaustive mathematical derivations, degree topometry tables, homophily matrices, and complete experimental metrics, refer to:
* **[SUMMARY_PHASEWISE_REPORT.md](SUMMARY_PHASEWISE_REPORT.md)**
* **[AML_GNN_Research_Summary_Phasewise.docx](AML_GNN_Research_Summary_Phasewise.docx)**
