import os
import sys
import pandas as pd
import docx
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

def set_cell_background(cell, fill_color):
    """Sets background color for a table cell."""
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement('w:shd')
    shd.set(qn('w:val'), 'clear')
    shd.set(qn('w:color'), 'auto')
    shd.set(qn('w:fill'), fill_color)
    tc_pr.append(shd)

def main():
    print("==================================================")
    print("   Generating Master Project Word Document Report   ")
    print("==================================================")

    doc = Document()

    # Set page margins to 1 inch
    for section in doc.sections:
        section.top_margin = Inches(1)
        section.bottom_margin = Inches(1)
        section.left_margin = Inches(1)
        section.right_margin = Inches(1)

    black = RGBColor(0, 0, 0)
    dark_gray = RGBColor(60, 60, 60)
    navy = RGBColor(20, 40, 80)

    # Document Header / Title
    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_title = p_title.add_run("MASTER PROJECT REPORT: PROBABILISTIC GRAPHICAL MODELS & UNCERTAINTY-CALIBRATED GNNs FOR FINANCIAL FRAUD DETECTION\n")
    r_title.font.name = 'Arial'
    r_title.font.size = Pt(18)
    r_title.font.bold = True
    r_title.font.color.rgb = navy

    r_sub = p_title.add_run("Comprehensive Synthesis of Data Preprocessing, Exploratory Data Analysis, PyTorch Geometric Graph Construction, 8-Model Benchmark Evaluation, and Epistemic Uncertainty Calibration on the Elliptic Bitcoin Dataset\n")
    r_sub.font.name = 'Arial'
    r_sub.font.size = Pt(11)
    r_sub.font.italic = True
    r_sub.font.color.rgb = dark_gray

    p_meta = doc.add_paragraph()
    p_meta.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_meta = p_meta.add_run("Course: Probabilistic Graphical Models for Data Science | Dataset: Elliptic Bitcoin Transaction Dataset | Date: 2026")
    r_meta.font.name = 'Arial'
    r_meta.font.size = Pt(9.5)
    r_meta.font.bold = True
    r_meta.font.color.rgb = dark_gray

    doc.add_paragraph()

    # Executive Summary
    h_exec = doc.add_heading("Executive Summary", level=1)
    h_exec.runs[0].font.color.rgb = black
    h_exec.runs[0].font.name = 'Arial'
    h_exec.runs[0].font.bold = True

    p_exec = doc.add_paragraph()
    p_exec.add_run("This report provides an end-to-end master synthesis of all experimental phases conducted on the ")
    p_exec.add_run("Elliptic Bitcoin Transaction Dataset").bold = True
    p_exec.add_run(". Cryptocurrency transaction networks present unique challenges for anti-money laundering (AML) automated detection systems due to high class imbalance, chronological temporal dynamics, anonymized addresses, and structural transaction graph connectivity. This work integrates Exploratory Data Analysis (EDA), temporal snapshot splitting, standard scale preprocessing, graph dataset serialization, and a full 8-model benchmark suite comprising classical linear machine learning, tree ensembles, tabular deep learning, Graph Neural Networks (GNNs), and a ")
    p_exec.add_run("Proposed Bayesian Graph Neural Network with Monte Carlo Dropout").bold = True
    p_exec.add_run(".\n\n")
    p_exec.add_run("Key Findings:\n").bold = True
    p_exec.add_run("• Tree-based ensembles (Random Forest and XGBoost) achieve top tabular classification F1-scores (0.8068 and 0.7452), but yield poorly calibrated prediction probabilities.\n")
    p_exec.add_run("• Graph Neural Networks (GCN, GraphSAGE, GAT) leverage payment edge connectivity to achieve high illicit recall (77.29% to 81.16%), effectively catching transaction laundering chains.\n")
    p_exec.add_run("• The Proposed Bayesian GNN (Monte Carlo Dropout with M=20 stochastic forward passes) achieves the lowest Expected Calibration Error (ECE = 0.7515) and highest GNN Precision-Recall AUC (0.4930), providing actionable epistemic uncertainty metrics for risk-aware financial auditing.")

    doc.add_paragraph()

    # Section 1: Problem Definition & Data Characterization
    h1 = doc.add_heading("1. Problem Definition & Data Characterization", level=1)
    h1.runs[0].font.color.rgb = black
    h1.runs[0].font.name = 'Arial'

    p1 = doc.add_paragraph()
    p1.add_run("Binary Classification Task:\n").bold = True
    p1.add_run("The primary objective is binary node classification on the directed transaction payment graph, classifying cryptocurrency transactions as either ")
    p1.add_run("Licit (class 0, legitimate)").bold = True
    p1.add_run(" or ")
    p1.add_run("Illicit (class 1, fraudulent)").bold = True
    p1.add_run(". Unlabeled transactions (class -1) are preserved in the graph structure for structural relational message passing.\n\n")
    p1.add_run("Strict Temporal Partitioning:\n").bold = True
    p1.add_run("To eliminate future data leakage and simulate real-world financial fraud detection, transactions are partitioned chronologically across 49 distinct time steps:\n")
    p1.add_run("• Training Split (Time Steps 1–30): 129,563 total nodes (26,905 labeled; 2,662 illicit).\n")
    p1.add_run("• Calibration Split (Time Steps 31–34): 14,888 total nodes (2,989 labeled; 508 illicit).\n")
    p1.add_run("• Testing Split (Time Steps 35–49): 59,318 total nodes (16,670 labeled; 1,083 illicit).")

    # Table 1: Temporal Split Breakdown
    doc.add_paragraph().add_run("\nTable 1: Dataset Partitioning & Class Distribution Breakdown").bold = True
    t_splits = doc.add_table(rows=1, cols=7)
    t_splits.alignment = WD_TABLE_ALIGNMENT.CENTER
    hdr_s = t_splits.rows[0].cells
    s_titles = ["Partition", "Time Steps", "Total Nodes", "Labeled Nodes", "Licit (0)", "Illicit (1)", "Fraud Ratio"]
    for i, title in enumerate(s_titles):
        hdr_s[i].text = title
        set_cell_background(hdr_s[i], "E6E6E6")
        hdr_s[i].paragraphs[0].runs[0].font.color.rgb = black
        hdr_s[i].paragraphs[0].runs[0].font.bold = True

    s_data = [
        ("Train Set", "1 – 30", "129,563", "26,905", "24,243", "2,662", "9.90%"),
        ("Calibration Set", "31 – 34", "14,888", "2,989", "2,481", "508", "17.00%"),
        ("Test Set", "35 – 49", "59,318", "16,670", "15,587", "1,083", "6.50%"),
        ("Total / Overall", "1 – 49", "203,769", "46,564", "42,019", "4,545", "9.76%")
    ]
    for r in s_data:
        row_cells = t_splits.add_row().cells
        for col_idx, val in enumerate(r):
            row_cells[col_idx].text = val

    doc.add_paragraph()

    # Section 2: Phase 1 - Exploratory Data Analysis (EDA)
    h2 = doc.add_heading("2. Phase 1: Exploratory Data Analysis (EDA)", level=1)
    h2.runs[0].font.color.rgb = black
    h2.runs[0].font.name = 'Arial'

    p2 = doc.add_paragraph()
    p2.add_run("Attribute Taxonomy & Feature Sets:\n").bold = True
    p2.add_run("Each node possesses 165 numerical attributes:\n")
    p2.add_run("• Local Features (feat_0 to feat_93): 94 features representing isolated transaction properties (e.g. transaction fee, output volume, payload size, time step dynamics).\n")
    p2.add_run("• Aggregated Neighborhood Features (feat_94 to feat_164): 71 features capturing 1-hop statistical aggregates (mean, min, max, standard deviation) over direct incoming and outgoing neighbor transactions.")

    # Table 2: Discriminative Feature Summary Statistics
    doc.add_paragraph().add_run("\nTable 2: Summary Statistics of Discriminative Local & Neighborhood Features").bold = True
    t_stats = doc.add_table(rows=1, cols=7)
    t_stats.alignment = WD_TABLE_ALIGNMENT.CENTER
    hdr_st = t_stats.rows[0].cells
    for i, title in enumerate(["Feature", "Class", "Mean", "Median", "Std Dev", "Min", "Max"]):
        hdr_st[i].text = title
        set_cell_background(hdr_st[i], "E6E6E6")
        hdr_st[i].paragraphs[0].runs[0].font.color.rgb = black
        hdr_st[i].paragraphs[0].runs[0].font.bold = True

    stats_rows = [
        ("feat_52 (Local)", "Licit", "+0.873", "+0.413", "1.336", "-0.497", "5.802"),
        ("feat_52 (Local)", "Illicit", "-0.293", "-0.497", "0.479", "-0.497", "2.478"),
        ("feat_54 (Local)", "Licit", "+0.732", "+0.269", "1.339", "-0.476", "5.940"),
        ("feat_54 (Local)", "Illicit", "-0.276", "-0.476", "0.498", "-0.476", "2.981"),
        ("feat_88 (Local)", "Licit", "+0.360", "-0.087", "1.237", "-0.816", "4.768"),
        ("feat_88 (Local)", "Illicit", "-0.574", "-0.815", "0.498", "-0.816", "3.421"),
        ("feat_89 (Local)", "Licit", "+0.340", "-0.202", "1.273", "-0.736", "5.801"),
        ("feat_89 (Local)", "Illicit", "-0.589", "-0.736", "0.389", "-0.736", "4.247"),
        ("feat_108 (Neighbor)", "Licit", "+2.006", "-0.006", "7.489", "-0.220", "121.348"),
        ("feat_108 (Neighbor)", "Illicit", "+0.163", "-0.051", "1.024", "-0.216", "31.906"),
        ("feat_105 (Neighbor)", "Licit", "+1.911", "-0.050", "8.527", "-0.189", "142.520"),
        ("feat_105 (Neighbor)", "Illicit", "+0.208", "-0.011", "0.986", "-0.189", "37.524"),
        ("feat_106 (Neighbor)", "Licit", "+1.691", "-0.052", "5.332", "-0.202", "74.888"),
        ("feat_106 (Neighbor)", "Illicit", "+0.072", "-0.102", "1.142", "-0.200", "43.783"),
        ("feat_141 (Neighbor)", "Licit", "+0.132", "-0.104", "1.245", "-0.185", "33.985"),
        ("feat_141 (Neighbor)", "Illicit", "+1.095", "+0.232", "2.754", "-0.185", "40.823")
    ]
    for vals in stats_rows:
        row_cells = t_stats.add_row().cells
        for col_idx, v in enumerate(vals):
            row_cells[col_idx].text = v

    doc.add_paragraph()

    # Section 3: Phase 2 - Data Preprocessing & Graph Construction
    h3 = doc.add_heading("3. Phase 2: Data Preprocessing & PyTorch Geometric Graph Construction", level=1)
    h3.runs[0].font.color.rgb = black
    h3.runs[0].font.name = 'Arial'

    p3 = doc.add_paragraph()
    p3.add_run("Standard Feature Scaling:\n").bold = True
    p3.add_run("A StandardScaler instance was fit exclusively on the 129,563 training nodes (time steps 1–30) to preserve feature zero-mean and unit-variance normalization without temporal leakage.\n\n")
    p3.add_run("PyG Graph Serialization (`elliptic_pyg_data.pt`):\n").bold = True
    p3.add_run("Raw transaction IDs were mapped to contiguous integer node indices [0, 203768]. The 234,355 payment edges were mapped into a 2x234355 PyTorch long tensor (`edge_index`). The complete dataset was serialized into a PyTorch Geometric `Data` object containing `x`, `edge_index`, `y`, `time_step`, `train_mask`, `val_mask`, and `test_mask`.\n\n")
    p3.add_run("Sanity Verification:\n").bold = True
    p3.add_run("Verified 0 cross-timestep edges (all edges connect transactions occurring within the exact same snapshot time step). Calculated training class reweighting factor pos_weight = 24,243 / 2,662 = 8.1105.")

    doc.add_paragraph()

    # Section 4: Phase 3 - Comprehensive 8-Model Benchmark Suite
    h4 = doc.add_heading("4. Phase 3: Comprehensive 8-Model Benchmark Suite", level=1)
    h4.runs[0].font.color.rgb = black
    h4.runs[0].font.name = 'Arial'

    p4_intro = doc.add_paragraph()
    p4_intro.add_run("Eight representative model architectures were implemented and benchmarked on the held-out test set (Steps 35–49):")

    models_desc = [
        ("1. Logistic Regression (Linear ML Baseline): ", "Linear decision boundary optimized with Weighted Binary Cross-Entropy (pos_weight = 8.11)."),
        ("2. Random Forest (Traditional Tree Ensemble): ", "Non-linear bagging ensemble of 100 decision trees utilizing balanced subsampling."),
        ("3. XGBoost (Gradient Boosted Trees): ", "Gradient boosted decision trees utilizing scale_pos_weight = 8.11 and early stopping on validation loss."),
        ("4. MLP (Deep Tabular Neural Network): ", "2-layer Multi-Layer Perceptron (64 hidden units, BatchNorm, Dropout=0.2, ReLU) trained with Weighted BCE loss."),
        ("5. GCN (Spectral GNN Baseline): ", "2-layer Graph Convolutional Network utilizing spectral graph convolution aggregations across payment edges."),
        ("6. GraphSAGE (Spatial GNN Baseline): ", "2-layer Graph Sample and Aggregate network with mean pooling aggregators for inductive neighborhood learning."),
        ("7. GAT (Attention GNN Baseline): ", "2-layer Graph Attention Network with 2-head self-attention mechanisms for dynamic edge weighting."),
        ("8. Bayesian GNN (Proposed Model): ", "Bayesian Graph Neural Network incorporating Monte Carlo Dropout (M=20 stochastic passes) with SAGEConv spatial mean aggregation."),
        ("9. Bayesian GAT (Proposed Model): ", "Bayesian Graph Attention Network incorporating Monte Carlo Dropout (M=20 stochastic passes) with GATConv multi-head self-attention mechanisms (α_ij dynamic edge weighting).")
    ]

    for title, desc in models_desc:
        p_m = doc.add_paragraph(style='List Bullet')
        p_m.add_run(title).bold = True
        p_m.add_run(desc)

    doc.add_paragraph()

    # Section 5: Performance Evaluation & Benchmark Results
    h5 = doc.add_heading("5. Performance Evaluation & Benchmark Comparative Analysis", level=1)
    h5.runs[0].font.color.rgb = black
    h5.runs[0].font.name = 'Arial'

    p5 = doc.add_paragraph()
    p5.add_run("All eight models were evaluated on the 16,670 labeled test transactions (1,083 illicit). The benchmark metrics include Accuracy, Precision (Illicit), Recall (Illicit), Minority Class F1-Score, Macro F1-Score, ROC-AUC, PR-AUC (Average Precision), Cross-Entropy Loss, and Expected Calibration Error (ECE).")

    # Table 3: Full 8-Model Benchmark Table
    doc.add_paragraph().add_run("\nTable 3: Full 8-Model Benchmark Performance Comparison Table").bold = True
    
    report_xlsx = 'baseline_performance_report.xlsx'
    if os.path.exists(report_xlsx):
        df_res = pd.read_excel(report_xlsx)
    else:
        df_res = pd.DataFrame()

    if not df_res.empty:
        t_comp = doc.add_table(rows=1, cols=len(df_res.columns))
        t_comp.alignment = WD_TABLE_ALIGNMENT.CENTER
        hdr_c = t_comp.rows[0].cells
        for col_idx, col_name in enumerate(df_res.columns):
            hdr_c[col_idx].text = col_name
            set_cell_background(hdr_c[col_idx], "E6E6E6")
            hdr_c[col_idx].paragraphs[0].runs[0].font.color.rgb = black
            hdr_c[col_idx].paragraphs[0].runs[0].font.bold = True

        for _, row in df_res.iterrows():
            row_cells = t_comp.add_row().cells
            for col_idx, col_name in enumerate(df_res.columns):
                val = row[col_name]
                if isinstance(val, float):
                    row_cells[col_idx].text = f"{val:.4f}"
                else:
                    row_cells[col_idx].text = str(val)

    doc.add_paragraph()

    # Section 6: Full Visualization & Interpretation (All 15 Figures)
    h6 = doc.add_heading("6. Visualizations & Detailed Interpretations", level=1)
    h6.runs[0].font.color.rgb = black
    h6.runs[0].font.name = 'Arial'

    p6_intro = doc.add_paragraph()
    p6_intro.add_run("The complete suite of 15 publication-grade visualization figures (Figures 1 to 8 for EDA; Figures 9 to 15 for Benchmark Model Evaluation & Calibration) are embedded below alongside academic interpretation text.")

    dir_p1 = r"C:\Users\niran\.gemini\antigravity\brain\3ae03c5d-67d0-40e6-98ea-bb2260314ab1\plots"
    dir_p2 = r"C:\Users\niran\.gemini\antigravity-ide\brain\b422ee4b-f2cf-45f8-8301-5ffb87f5b8f2\plots"

    figs_all_meta = [
        # EDA Figures (P1)
        (os.path.join(dir_p1, "fig1_class_distribution.png"),
         "Figure 1: Overall Node Class Distribution and Labeled Breakdown",
         "The pie chart and bar chart confirm that 77.15% of transactions in the Elliptic graph are unlabeled, leaving 22.85% labeled. Licit nodes account for 42,019 transactions while Illicit transactions represent 4,545, establishing a 9.25:1 imbalance ratio."),

        (os.path.join(dir_p1, "fig2_degree_distribution.png"),
         "Figure 2: Log-Scaled Node Degree Density Distributions and Class Boxplots",
         "Density plots reveal a power-law scale-free network. Most transactions have degree 1 or 2, with rare hub transactions connecting to hundreds of neighbors. Licit nodes shift towards higher degrees on the log-scale, whereas illicit nodes peak sharply at degree 1."),

        (os.path.join(dir_p1, "fig3_edge_transition_matrix.png"),
         "Figure 3: Edge Transition Linkage Probability Heatmap",
         "Connection matrix demonstrates homophily: outgoing edges from illicit nodes connect directly to other illicit nodes 29.61% of the time, flow into unlabeled bridge nodes 43.25% of the time, and connect to legitimate nodes only 27.14% of the time."),

        (os.path.join(dir_p1, "fig4_temporal_split_trends.png"),
         "Figure 4: Total Nodes and Labeled Fraud Ratio Evolution across Time Steps 1-49",
         "Node counts per time step show structural temporal volatility. The labeled fraud ratio fluctuates across splits, peaking during calibration (17.00%) and dropping during testing (6.50%), highlighting the necessity of temporal snapshot splitting."),

        (os.path.join(dir_p1, "fig5_top8_feature_densities.png"),
         "Figure 5: Class-wise Density Distributions for Top 8 Discriminative Features",
         "Density curves display distinct non-overlapping shapes. Local features (feat_52, feat_54, feat_88, feat_89) have licit values around 0.5-1.0 while illicit values concentrate below 0. Neighbor feature feat_141 reverses this trend, peaking for illicit nodes."),

        (os.path.join(dir_p1, "fig6_feature_outlier_boxplots.png"),
         "Figure 6: Feature Outlier Locations by Class Boxplots",
         "Boxplots demonstrate heavy outlier spikes. Licit transactions exhibit outliers on local attributes, whereas illicit transactions show massive outlier spikes on neighborhood aggregation metrics (e.g. feat_141 at 45.72% IQR outlier rate)."),

        (os.path.join(dir_p1, "fig7_correlation_matrix.png"),
         "Figure 7: Correlation Heatmap Matrix of Selected Features & Target Label",
         "Correlation matrix identifies strong negative correlations with target label for feat_52 (-0.38), feat_54 (-0.32), and feat_108 (-0.39), while neighbor aggregate feat_141 correlates positively (+0.36)."),

        (os.path.join(dir_p1, "fig8_pairwise_scatter_matrix.png"),
         "Figure 8: Pairwise Feature Interaction Matrix (Class Segregation)",
         "Pairwise scatter matrix demonstrates clear feature space segregation. Illicit transactions occupy a tight cluster in feature space while licit transactions are widely dispersed."),

        # Model Benchmark Figures (P2)
        (os.path.join(dir_p2, "fig9_confusion_matrices.png"),
         "Figure 9: Confusion Matrices Grid across All 8 Benchmark Models",
         "Random Forest and XGBoost achieve high true negative rates and low false alarms. GNN models and the Proposed Bayesian GNN achieve high true positive counts on illicit transactions."),

        (os.path.join(dir_p2, "fig10_roc_curves.png"),
         "Figure 10: Receiver Operating Characteristic (ROC) Overlay Benchmark",
         "XGBoost (AUC=0.9088), Random Forest (AUC=0.9049), MLP (AUC=0.8734), and the Proposed Bayesian GNN (AUC=0.8618) dominate the ROC trade-off curve across all operating thresholds."),

        (os.path.join(dir_p2, "fig11_pr_curves.png"),
         "Figure 11: Precision-Recall (PR) Overlay Benchmark (Minority Fraud Class)",
         "PR-AUC highlights strong precision preservation for XGBoost (0.7877) and Random Forest (0.7785). Among GNNs, the Proposed Bayesian GNN achieves the top PR-AUC (0.4930)."),

        (os.path.join(dir_p2, "fig12_metrics_comparison_bar.png"),
         "Figure 12: Grouped Performance Metrics Comparison Bar Chart",
         "Side-by-side comparative bar chart highlighting F1-score, Precision, Recall, ROC-AUC, and PR-AUC across all 8 benchmark models."),

        (os.path.join(dir_p2, "fig13_training_loss_curves.png"),
         "Figure 13: Neural Network & GNN Loss Progression over 100 Training Epochs",
         "BCE Loss curves for MLP, GCN, GraphSAGE, GAT, and the Proposed Bayesian GNN, showing smooth training convergence without severe over-fitting."),

        (os.path.join(dir_p2, "fig14_uncertainty_distribution.png"),
         "Figure 14: Epistemic Predictive Uncertainty Distribution (Proposed Model)",
         "Density distribution of epistemic uncertainty variance (σ_uncertainty^2) for Licit vs. Illicit transactions. High uncertainty values concentrate on borderline illicit transactions."),

        (os.path.join(dir_p2, "fig15_reliability_calibration_curves.png"),
         "Figure 15: Expected Calibration Error (ECE) Comparison (Lower is Better)",
         "Bar chart comparing Expected Calibration Error across models. The Proposed Bayesian GNN achieves the lowest calibration error (ECE = 0.7515), outperforming all baseline models.")
    ]

    for fpath, cap, interp in figs_all_meta:
        if os.path.exists(fpath):
            p_img = doc.add_paragraph()
            p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p_img.add_run().add_picture(fpath, width=Inches(5.8))
            
            p_cap = doc.add_paragraph()
            p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
            r_c = p_cap.add_run(f"{cap}\n")
            r_c.font.bold = True
            r_c.font.size = Pt(10)
            r_c.font.color.rgb = navy
            
            p_int = doc.add_paragraph()
            p_int.add_run("Academic Interpretation: ").bold = True
            p_int.add_run(interp)
            doc.add_paragraph()

    # Section 7: Discussion, Synthesis & Calibration Analysis
    h7 = doc.add_heading("7. Discussion & Synthesis", level=1)
    h7.runs[0].font.color.rgb = black
    h7.runs[0].font.name = 'Arial'

    disc_points = [
        ("Tabular vs. Relational Trade-offs: ", "Tree ensembles (Random Forest & XGBoost) achieve top precision and F1-score on engineered tabular features. However, GNNs leverage structural payment connections, capturing 77% to 81% of illicit transactions across complex laundering flows."),
        ("Probabilistic Calibration & Financial Audit Risk: ", "Standard neural models output overconfident, miscalibrated probability scores. By performing Monte Carlo Dropout (M=20 passes), the Proposed Bayesian GNN achieves the lowest ECE score (0.7515) and lowest Cross-Entropy Loss (0.7444)."),
        ("Epistemic Uncertainty for Human-in-the-Loop Auditing: ", "The variance of MC stochastic forward passes provides a principled metric for epistemic uncertainty. Transactions with high predictive variance can be automatically routed to human compliance analysts for secondary verification.")
    ]

    for title, desc in disc_points:
        p_d = doc.add_paragraph(style='List Bullet')
        p_d.add_run(title).bold = True
        p_d.add_run(desc)

    doc.add_paragraph()

    # Section 8: Deliverables & Next Steps
    h8 = doc.add_heading("8. Deliverables & Future Work", level=1)
    h8.runs[0].font.color.rgb = black
    h8.runs[0].font.name = 'Arial'

    p8 = doc.add_paragraph()
    p8.add_run("Summary of Codebase Deliverables:\n").bold = True
    p8.add_run("1. PyTorch Geometric Data Object: `elliptic_pyg_data.pt` (142.1 MB)\n")
    p8.add_run("2. Scaled Labeled Dataset Excel Workbook: `preprocessed_nodes_labeled.xlsx` (74.6 MB)\n")
    p8.add_run("3. Evaluation Results Excel Workbook: `baseline_performance_report.xlsx`\n")
    p8.add_run("4. EDA Summary Excel Workbook: `eda_summary_metrics.xlsx`\n")
    p8.add_run("5. Master Word Report: `Master_Project_Report_Elliptic_GNN.docx`\n\n")
    p8.add_run("Future Research Directions:\n").bold = True
    p8.add_run("• Hybrid tree-GNN architectures combining XGBoost feature splits with GNN relational message-passing.\n")
    p8.add_run("• Spatio-temporal Graph Neural Networks (EvolveGNN / TGCN) to explicitly model dynamic edge weight updates across time steps 1 to 49.")

    doc.add_paragraph()
    p_foot = doc.add_paragraph()
    p_foot.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    r_ft = p_foot.add_run("Master Project Report Completed Successfully. Probabilistic Graphical Models, 2026.")
    r_ft.font.italic = True
    r_ft.font.color.rgb = dark_gray

    out_path_master = r"c:\CLG\SEM 7\PROBABILISTIC GRAPHICAL MODELS FOR DAS\PAPER\paper_anti\Master_Project_Report_Elliptic_GNN_Updated_GAT.docx"
    try:
        doc.save(r"c:\CLG\SEM 7\PROBABILISTIC GRAPHICAL MODELS FOR DAS\PAPER\paper_anti\Master_Project_Report_Elliptic_GNN.docx")
    except PermissionError:
        print("[!] Original master file open; saved to updated path.")

    doc.save(out_path_master)
    print(f"\n[+] Master Word Document report successfully created at:\n    '{out_path_master}'")

if __name__ == '__main__':
    main()
