# Python Research Codebase Index
**Project:** Pure Graphical Machine Learning & Directional Bayesian Graph Neural Networks for Anti-Money Laundering on Bitcoin  
**Dataset:** Elliptic Bitcoin Transaction Dataset ($203,769$ transaction nodes, $234,355$ directed payment edges, $165$ continuous features across $49$ discrete timesteps)  
**Location:** [`python code/`](file:///c:/Users/USER/OneDrive/Desktop/Probalistic%20Graphical%20Lab/python%20code) / Desktop mirror: `c:\Users\USER\OneDrive\Desktop\python code`  

---

## 📋 Execution Roadmap & Script Directory

The scripts in this folder are organized in sequential order to replicate the full 7-phase research pipeline from end to end:

| Script Name | Phase / Module | Core Functionality |
| :--- | :---: | :--- |
| **`00_run_complete_pipeline.py`** | Master Pipeline | End-to-end master orchestrator running all preprocessing, training, calibration, and report generation workflows. |
| **`01_preprocess_and_splits.py`** | Phase 0 & 1 | Injective hash mapping $\phi(\text{txId})$, zero-lookahead Z-score normalization strictly on $t \le 30$, temporal splits (Train $1\dots30$, Val $31\dots34$, Test $35\dots49$), and PyTorch Geometric binary serialization. |
| **`01_run_eda_analysis.py`** | Phase 0 | Computes graph topometry, degree power-law distributions (max in-degree 473, max out-degree 177), and relational edge homophily ($55.25\%$ illicit clustering). |
| **`02_ml1_models.py`** | Phase 2 | Architecture definitions for GCN, GAT, GraphSAGE, GIN, Bayesian GCN, Bayesian GAT, Bayesian GraphSAGE, and Bayesian GNN (BNN) with Monte Carlo Dropout. |
| **`02_ml1_train_models.py`** | Phase 2 | Training loop for 8 baseline graphical models on sparse ground-truth nodes with dynamic class weighting. |
| **`02_ml1_calibrate_and_evaluate.py`** | Phase 2 | 191-point threshold calibration, Expected Calibration Error (ECE), ROC/PR evaluation, and confusion matrix generation on sparse test nodes. |
| **`02_ml1_utils.py`** | Phase 2 | Evaluation metrics and data loader helper utilities for ML Training 1. |
| **`03_pseudo_label_unlabeled_nodes.py`** | Phase 3 | Inductive out-of-fold GNN ensemble self-training engine, temperature scaling, and margin weighting ($w_i = \max(P_i, 1-P_i)$) to pseudo-label all $157,205$ unlabeled nodes into a $100\%$ resolved graph ($160,041$ Licit, $43,728$ Illicit, $0$ Unlabeled). |
| **`04_ml2_models.py`** | Phase 4 | Model architectures optimized for dense graph neighborhood aggregation. |
| **`04_ml2_train_fully_labeled.py`** | Phase 4 | Training engine across the continuous dense graph ($203,769$ nodes, $234,355$ edges) across $123,287$ training nodes. |
| **`04_ml2_calibrate_and_evaluate.py`** | Phase 4 | Evaluation and threshold calibration across all $67,504$ dense test transactions, documenting the $26.4\times$ detection surge ($14,802$ illicit captured). |
| **`04_ml2_utils.py`** | Phase 4 | Evaluation metrics, confusion matrix plotting, and ROC/PR generation for dense graphs. |
| **`05_ml3_directional_residual_models.py`** | Phase 5 | State-of-the-Art models: Dir-ResGCN, Dir-ResGAT, Dir-ResSAGE, Dir-GIN, Bayesian Dir-GCN, Bayesian Dir-GAT, Bayesian Dir-SAGE, Bayesian GNN (BNN). Implements decoupled incoming/outgoing flow message passing ($\mathcal{E}_{\text{in}} \oplus \mathcal{E}_{\text{out}}$), residual skip projections, and Layer Normalization. |
| **`05_ml3_soft_loss_and_metrics.py`** | Phase 5 | Soft confidence-weighted binary cross-entropy loss ($\mathcal{L}_{\text{soft}} = - \frac{1}{N} \sum w_i \cdot \text{BCE}$), Brier score, ECE calibration, and Monte Carlo Dropout ($T=25$) epistemic uncertainty calculation. |
| **`05_ml3_train_advanced_gnns.py`** | Phase 5 | Training loop for directional residual models utilizing soft pseudo-label loss weighting. |
| **`05_ml3_calibrate_and_verify.py`** | Phase 5 | Dual-evaluation calibration engine testing both across the full dense test graph ($67,504$ nodes) and ground-truth forensic test verification ($16,670$ nodes) achieving $91.16\%$ Accuracy. |
| **`06_generate_all_flowcharts.py`** | Phase 6 / Visual | Renders the complete suite of 12 high-resolution 300-DPI publication flowchart PNGs into `flowcharts/`. |
| **`06_generate_summary_report.py`** | Reporting | Compiles `AML_GNN_Research_Summary_Phasewise.docx` loaded with all 12 embedded figures. |
| **`06_create_overall_draft_1.py`** | Reporting | Builds the complete research compendium `overall draft 1.docx` and `.doc` with full phase write-ups, equations, and tables. |
| **`06_create_visualization_and_outputs_doc.py`** | Reporting | Builds `Visualization and outputs.docx` and `.doc` compiling all 41 evaluation plots, 12 flowcharts, and metric tables into a single document. |

---

## 🚀 Execution Instructions (PowerShell)

To run the complete pipeline from scratch or re-execute specific components:

```powershell
# Step 1: Preprocessing and Temporal Graph Partitioning
python "python code/01_preprocess_and_splits.py"

# Step 2: ML Training 1 Sparse Benchmark & Calibration
python "python code/02_ml1_train_models.py"
python "python code/02_ml1_calibrate_and_evaluate.py"

# Step 3: Probabilistic Pseudo-Labeling Engine (Resolving 157k Unknowns)
python "python code/03_pseudo_label_unlabeled_nodes.py"

# Step 4: ML Training 2 Dense Graph Benchmark & Calibration
python "python code/04_ml2_train_fully_labeled.py"
python "python code/04_ml2_calibrate_and_evaluate.py"

# Step 5: ML Training 3 Directional Residual GNNs & Soft Loss Optimization
python "python code/05_ml3_train_advanced_gnns.py"
python "python code/05_ml3_calibrate_and_verify.py"

# Step 6: Render All 12 High-Resolution Visual Flowcharts
python "python code/06_generate_all_flowcharts.py"

# Step 7: Generate Master Documents
python "python code/06_create_overall_draft_1.py"
python "python code/06_create_visualization_and_outputs_doc.py"
```
