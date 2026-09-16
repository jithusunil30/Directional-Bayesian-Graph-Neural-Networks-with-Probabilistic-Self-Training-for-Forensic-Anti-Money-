# Anti-Money Laundering (AML) on Bitcoin: Datasets & Code

[![Branch: Data-and-Code](https://img.shields.io/badge/branch-Data--and--Code-blue.svg)](https://github.com/jithusunil30/PGM-PAPER/tree/Data-and-Code)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![PyG](https://img.shields.io/badge/PyG-PyTorch--Geometric-3C2179.svg)](https://pyg.org/)

This branch (**`Data-and-Code`**) contains **strictly code, datasets, metric tables, trained probability arrays, and model definitions** across all experimental phases, completely free of any `.doc`, `.docx`, `.txt`, or document files.

> 📘 **Note:** Full narrative documentation, phase-by-phase walkthroughs, and Word research documents are maintained on the [`main`](https://github.com/jithusunil30/PGM-PAPER/tree/main) branch.

---

## 📂 Code & Data Architecture

```
.
├── dataset/                                    # Phase 0 & Phase 1: Preprocessing & EDA
│   ├── preprocess.py                           # Z-score standardization & strict temporal split
│   ├── run_eda.py                              # Degree calculation, homophily & outlier statistics
│   ├── eda_report.md                           # EDA metrics & structural findings
│   └── src/                                    # Connectivity & utility modules
│
├── ML TRAINING 1/                              # Phase 2: Sparse Ground-Truth Benchmark
│   ├── models.py                               # GCN, GAT, GraphSAGE, GIN, BNN architectures
│   ├── train_all_models.py                     # Training pipeline on sparse ground-truth nodes
│   ├── calibrate_90plus_models.py              # 90%+ calibrated threshold evaluation
│   ├── model_comparison_metrics.xlsx & .csv   # Benchmark performance metrics
│   ├── *.npy                                   # Saved probability & uncertainty arrays
│   └── *.png                                   # 300-DPI publication curves (ROC, PR, Confusion)
│
├── ML TRAINING 2/                              # Phase 3 & 4: Pseudo-Labeling & Dense Graph GNNs
│   ├── label_unlabeled_nodes.py                # Self-training probabilistic pseudo-labeling engine
│   ├── models.py                               # Graph neural network model definitions
│   ├── train_fully_labeled_gnns.py             # Training on 100% dense graph (203k nodes)
│   ├── calibrate_ml2_results.py                # Multi-threshold calibration script
│   ├── model_comparison_metrics.xlsx & .csv   # Benchmark performance metrics
│   ├── *.npy                                   # Probabilities and epistemic variance arrays
│   └── *.png                                   # 300-DPI curves and uncertainty distributions
│
├── ML TRAINING 3/                              # Phase 5 & 6: Directional Residual Bayesian GNNs
│   ├── models.py                               # Dir-ResGCN, Dir-ResGAT, Dir-ResSAGE, Dir-GIN, BNN
│   ├── utils.py                                # Soft confidence sample weighting & ECE calculation
│   ├── train_advanced_gnns.py                  # End-to-end directional Bayesian training
│   ├── calibrate_ml3_results.py                # Calibration on full graph & ground-truth subset
│   ├── model_comparison_metrics.xlsx & .csv   # Dense test set metrics (67,504 nodes)
│   ├── ground_truth_test_metrics.xlsx & .csv   # Ground-truth verified metrics (16,670 nodes)
│   ├── *.npy                                   # Directional probabilities & uncertainty arrays
│   └── *.png                                   # ROC, PR, Confusion, and Epistemic curves
│
├── paper_anti/                                 # Research pipeline scripts & evaluation tables
├── generate_all_flowcharts.py                  # Flowchart rendering script
├── generate_summary_report.py                  # Document compilation utility
└── flowcharts/                                 # 12 High-Resolution 300-DPI Architecture Flowcharts
```

---

## 🚀 Execution Quickstart

### 1. Environment Setup
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install torch-geometric scikit-learn pandas numpy matplotlib seaborn
```

### 2. Run Data Preprocessing & Splitting (Phase 1)
```bash
python dataset/preprocess.py
```

### 3. Run Pseudo-Labeling (Phase 3)
```bash
python "ML TRAINING 2/label_unlabeled_nodes.py"
```

### 4. Train Directional Residual GNNs (Phase 5)
```bash
python "ML TRAINING 3/train_advanced_gnns.py"
```

### 5. Evaluate & Calibrate Metrics (Phase 6)
```bash
python "ML TRAINING 3/calibrate_ml3_results.py"
```
