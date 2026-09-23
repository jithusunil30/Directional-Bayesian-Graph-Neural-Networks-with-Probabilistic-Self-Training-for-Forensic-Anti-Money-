import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def create_box(ax, text, xy, width, height, box_color='#1B365D', text_color='white', fontsize=9, bold=True):
    rect = patches.FancyBboxPatch(
        xy, width, height,
        boxstyle="round,pad=0.04,rounding_size=0.08",
        facecolor=box_color, edgecolor='#0B1B3D', linewidth=1.5
    )
    ax.add_patch(rect)
    weight = 'bold' if bold else 'normal'
    ax.text(
        xy[0] + width / 2.0, xy[1] + height / 2.0, text,
        ha='center', va='center', color=text_color,
        fontsize=fontsize, weight=weight, multialignment='center'
    )

def create_arrow(ax, start, end, color='#2C3E50', width=1.5):
    ax.annotate(
        '', xy=end, xytext=start,
        arrowprops=dict(
            arrowstyle="-|>", color=color, lw=width,
            mutation_scale=14, shrinkA=0, shrinkB=0
        )
    )

def generate_master_flowchart(save_path):
    fig, ax = plt.subplots(figsize=(14, 8), dpi=300)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')

    # Title
    ax.text(7, 7.6, "End-to-End AML Graph Intelligence Pipeline: Master Architecture", 
            ha='center', va='center', fontsize=15, weight='bold', color='#1B365D')
    ax.text(7, 7.25, "Full-Lifecycle Flow from Raw Blockchain Data to Directional Bayesian GNN Compliance Deployment", 
            ha='center', va='center', fontsize=10, style='italic', color='#555555')

    # Row 1 (Phases 0, 1, 2)
    create_box(ax, "Phase 0: Raw Data & EDA\n203k Nodes, 234k Edges\nDegree & Homophily Analysis\n77.15% Unlabeled Bottleneck", (0.5, 5.2), 3.6, 1.4, '#2C3E50')
    create_arrow(ax, (4.1, 5.9), (4.9, 5.9))

    create_box(ax, "Phase 1: Preprocessing & Splits\nInjective Hash Mapping φ\nZ-score Scaler (t <= 30)\nStrict Temporal Split (Train/Val/Test)\nPyG Graph Construction", (4.9, 5.2), 4.2, 1.4, '#1B365D')
    create_arrow(ax, (9.1, 5.9), (9.9, 5.9))

    create_box(ax, "Phase 2: ML Training 1\nSparse Ground-Truth Benchmark\n46,564 Labeled Nodes Active\nAccuracy: 93.14% (Skewed)\nRecall: 51.8% | 561 Caught", (9.9, 5.2), 3.6, 1.4, '#C0392B')

    # Arrow Down to Row 2
    create_arrow(ax, (11.7, 5.2), (11.7, 4.3))
    ax.text(11.8, 4.75, "Diagnostic: Severed Graph Paths\n522 Criminals Undetected", fontsize=8, color='#C0392B', weight='bold')

    # Row 2 (Phases 3, 4) - flows right to left
    create_box(ax, "Phase 3: Labeling Unknown Labels\nProbabilistic Self-Training Engine\n157,205 Nodes Pseudo-Labeled\nMargin & Confidence Calibration\n100% Graph Coverage (203k Nodes)", (8.5, 2.9), 5.0, 1.4, '#27AE60')
    create_arrow(ax, (8.5, 3.6), (7.3, 3.6))

    create_box(ax, "Phase 4: ML Training 2\n100% Dense Graph Benchmark\nIllicit Caught: 14,802 (26x Surge)\nF1-Score: 0.7760 | PR-AUC: 0.8129\nAUC-ROC: 0.9350", (2.3, 2.9), 5.0, 1.4, '#2980B9')

    # Arrow Down to Row 3
    create_arrow(ax, (4.8, 2.9), (4.8, 2.0))
    ax.text(4.9, 2.45, "Flaw: Undirected Symmetry & Label Noise", fontsize=8, color='#E67E22', weight='bold')

    # Row 3 (Phases 5, 6)
    create_box(ax, "Phase 5: ML Training 3 (State of the Art)\nDirectional Flow Passing (Ein || Eout)\nResidual Skips + Layer Normalization\nSoft Confidence-Weighted Loss (w_i)\nBayesian MC Dropout (T=25 passes)\nAcc (GT): 91.16% | AUC: 0.9315 | 14,504 Caught", (1.0, 0.4), 6.0, 1.6, '#8E44AD')
    create_arrow(ax, (7.0, 1.2), (7.8, 1.2))

    create_box(ax, "Phase 6: Comparison & Deployment\nCross-Phase Master Evaluation\nAutomated Risk-Tier Routing\nImmediate Freeze vs. Human Review\nAcademic Paper Publication", (7.8, 0.4), 5.2, 1.6, '#16A085')

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Master Flowchart generated: {save_path}")

def generate_phase_flowcharts(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Master Flowchart
    generate_master_flowchart(os.path.join(output_dir, "flowchart_master_pipeline.png"))

    # 2. Phase 0 & 1 Flowchart (Data Engineering)
    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis('off')
    ax.text(6, 5.6, "Phases 0 & 1: Raw Data Ingestion, EDA, and Preprocessing Flow", ha='center', va='center', fontsize=13, weight='bold', color='#1B365D')

    create_box(ax, "Raw CSV Sources\n• features.csv (203k x 167)\n• edgelist.csv (234k edges)\n• classes.csv (labels)", (0.4, 3.4), 3.2, 1.6, '#34495E')
    create_arrow(ax, (3.6, 4.2), (4.4, 4.2))

    create_box(ax, "Exploratory Data Analysis\n• In/Out Degree Asymmetry\n• Homophily (55.25% Illicit-to-Illicit)\n• IQR Outlier Profiling\n• 77.15% Unlabeled Discovery", (4.4, 3.4), 3.6, 1.6, '#2980B9')
    create_arrow(ax, (8.0, 4.2), (8.8, 4.2))

    create_box(ax, "Preprocessing & Splits\n• Hash Re-Indexing (φ: txId -> int)\n• Z-score Scaling (Fitted on t <= 30)\n• Temporal Train / Val / Test\n• PyG Data Object Serialization", (8.8, 3.4), 3.0, 1.6, '#1B365D')

    # Lower summary cards
    create_box(ax, "Train Split (Steps 1–30)\n123,287 Total Nodes\n26,905 Labeled Seed Nodes", (1.0, 0.8), 3.0, 1.4, '#27AE60')
    create_box(ax, "Val Split (Steps 31–34)\n12,978 Total Nodes\n2,989 Labeled Nodes (ECE Calibration)", (4.6, 0.8), 3.4, 1.4, '#E67E22')
    create_box(ax, "Test Split (Steps 35–49)\n67,504 Total Nodes\n16,670 Labeled Forensic Nodes", (8.5, 0.8), 3.0, 1.4, '#C0392B')

    create_arrow(ax, (10.3, 3.4), (10.0, 2.2))
    create_arrow(ax, (10.3, 3.4), (6.3, 2.2))
    create_arrow(ax, (10.3, 3.4), (2.5, 2.2))

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "flowchart_phase0_1_data.png"), bbox_inches='tight', dpi=300)
    plt.close()

    # 3. Phase 2 & 3 Flowchart (Sparse Benchmark & Labeling)
    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis('off')
    ax.text(6, 5.6, "Phases 2 & 3: Sparse Masking Failure & Pseudo-Labeling Solution", ha='center', va='center', fontsize=13, weight='bold', color='#1B365D')

    create_box(ax, "Phase 2: ML Training 1\n• 46,564 Labeled Nodes Active\n• 157,205 Unlabeled Masked\n• Message Passing Severed", (0.5, 3.3), 3.4, 1.6, '#C0392B')
    create_arrow(ax, (3.9, 4.1), (4.7, 4.1))

    create_box(ax, "Diagnostic Bottleneck\n• High Accuracy (93.14%) via Skew\n• Low Recall (51.8%) | 561 Caught\n• PR-AUC: 0.3475 (Failure in Depth)\n• 522 Criminal Transactions Missed", (4.7, 3.3), 3.8, 1.6, '#7F8C8D')
    create_arrow(ax, (8.5, 4.1), (9.3, 4.1))

    create_box(ax, "Phase 3: Labeling Engine\n• Multi-Pass Self-Training\n• Out-of-Fold Probabilities\n• Soft Confidence Margins\n• 100% Graph Restored", (9.3, 3.3), 2.4, 1.6, '#27AE60')

    # Bottom flow for Phase 3 output
    create_arrow(ax, (10.5, 3.3), (10.5, 2.0))
    create_box(ax, "Fully Labeled Graph Dataset (elliptic_pyg_data_fully_labeled.pt)\nTotal: 203,769 Nodes | Licit: 160,041 (78.54%) | Illicit: 43,728 (21.46%) | Unlabeled: 0 (0.00%)\nContinuous Multi-Hop Message Passing Enabled Across All 234,355 Edges", (0.5, 0.6), 11.0, 1.4, '#1B365D')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "flowchart_phase2_3_labeling.png"), bbox_inches='tight', dpi=300)
    plt.close()

    # 4. Phase 4 & 5 Flowchart (Model Evolution: Undirected GNN -> Directional Bayesian GNN)
    fig, ax = plt.subplots(figsize=(13, 6.5), dpi=300)
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 6.5)
    ax.axis('off')
    ax.text(6.5, 6.0, "Phases 4 & 5: GNN Evolution to Directional Residual Bayesian GNNs", ha='center', va='center', fontsize=13, weight='bold', color='#1B365D')

    create_box(ax, "Phase 4: ML Training 2\n• Dense 100% Graph\n• Undirected GNNs\n• Illicit TP: 14,802\n• PR-AUC: 0.8129\n• AUC-ROC: 0.9350", (0.5, 3.8), 3.4, 1.6, '#2980B9')
    create_arrow(ax, (3.9, 4.6), (4.7, 4.6))

    create_box(ax, "Identified Architectural Flaws\n• Flow Asymmetry Ignored (Inflow != Outflow)\n• Over-Smoothing over Multi-Hop Paths\n• Hard Pseudo-Labels Injected Noise (w=1.0)", (4.7, 3.8), 4.2, 1.6, '#E67E22')
    create_arrow(ax, (8.9, 4.6), (9.7, 4.6))

    create_box(ax, "Phase 5: ML Training 3\n• State of the Art\n• Directional Flow Decoupling\n• Residual Skips + LayerNorm\n• Soft Confidence BCE (w_i)\n• Bayesian MC Dropout (T=25)", (9.7, 3.8), 3.0, 1.6, '#8E44AD')

    # Bottom detailed architecture block - Arrow ends cleanly on top edge Y=2.3
    create_arrow(ax, (11.2, 3.8), (11.2, 2.3))
    create_box(ax, "Directional Residual Bayesian Convolution Formulation:\n"
                  "h_v^(l+1) = LayerNorm( sigma( W_in * Agg_{u in N_in(v)}(h_u) || W_out * Agg_{w in N_out(v)}(h_w) ) + W_res * h_v^(l) )\n"
                  "Loss: L = - sum w_i * [ y_i * log(p_i) + pos_weight * (1 - y_i) * log(1 - p_i) ] where w_i = max(P_i, 1 - P_i)\n"
                  "Result: 91.16% Accuracy on Ground Truth, 0.9315 AUC-ROC, 0.8068 PR-AUC, 14,504 Illicit Caught with Elite Precision",
                  (0.5, 0.5), 12.0, 1.8, '#1B365D')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "flowchart_phase4_5_gnn_evolution.png"), bbox_inches='tight', dpi=300)
    plt.close()

    # 5. Phase 6 Flowchart: Operational Compliance & Risk Routing
    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis('off')
    ax.text(6, 5.6, "Phase 6: Uncertainty-Aware AML Compliance & Transaction Routing Flow", ha='center', va='center', fontsize=13, weight='bold', color='#1B365D')

    create_box(ax, "New Bitcoin Transaction\nNode v (x_v in R^165)\n+ Local Neighborhood N(v)", (0.4, 2.2), 2.6, 1.6, '#34495E')
    create_arrow(ax, (3.0, 3.0), (3.8, 3.0))

    create_box(ax, "Bayesian GNN Inference\nMonte Carlo Dropout (T=25)\n• Predictive Probability P(illicit)\n• Epistemic Uncertainty σ_epi^2", (3.8, 2.2), 3.4, 1.6, '#8E44AD')

    # Three routing branches - cleanly centered on each target box
    create_arrow(ax, (7.2, 3.4), (8.2, 4.55))
    create_arrow(ax, (7.2, 3.0), (8.2, 2.95))
    create_arrow(ax, (7.2, 2.6), (8.2, 1.35))

    create_box(ax, "Tier 1: High Risk & Confident\nP >= 0.93 & Low Uncertainty\n-> Immediate Automated Freeze\n-> SAR Auto-Generated", (8.2, 3.9), 3.5, 1.3, '#C0392B')
    create_box(ax, "Tier 2: Ambiguous / Novel Attack\nP in [0.70, 0.93] OR High Uncertainty\n-> Route to Compliance Officer\n-> Manual Forensic Investigation", (8.2, 2.3), 3.5, 1.3, '#E67E22')
    create_box(ax, "Tier 3: Verified Licit\nP < 0.70 & Low Uncertainty\n-> Auto-Approve Transaction\n-> Broadcast to Ledger", (8.2, 0.7), 3.5, 1.3, '#27AE60')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "flowchart_phase6_compliance.png"), bbox_inches='tight', dpi=300)
    plt.close()
    print("All 5 publication-grade flowcharts generated successfully!")

if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.abspath(__file__))
    img_dir = os.path.join(base_dir, "flowcharts")
    generate_phase_flowcharts(img_dir)
