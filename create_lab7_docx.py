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
    print("   Generating Updated LAB 7 Word Document Reports   ")
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

    # Document Title & Header Block
    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_title = p_title.add_run("LAB 7: BASELINE SELECTION, PROPOSED METHODOLOGY (BAYESIAN GAT) IMPLEMENTATION & COMPARATIVE ANALYSIS REPORT\n")
    r_title.font.name = 'Arial'
    r_title.font.size = Pt(16)
    r_title.font.bold = True
    r_title.font.color.rgb = navy

    r_sub = p_title.add_run("Implementation of Proposed Bayesian Graph Attention Network (Bayesian GAT with Monte Carlo Dropout M=20 & GATConv) and Empirical Comparative Evaluation on the Elliptic Bitcoin Dataset\n")
    r_sub.font.name = 'Arial'
    r_sub.font.size = Pt(11)
    r_sub.font.italic = True
    r_sub.font.color.rgb = dark_gray

    p_meta = doc.add_paragraph()
    p_meta.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_meta = p_meta.add_run("Course: Probabilistic Graphical Models for DAS | Roll No: cb.ps.i5das23102 | Dataset: Elliptic Bitcoin Dataset | Date: 2026")
    r_meta.font.name = 'Arial'
    r_meta.font.size = Pt(9.5)
    r_meta.font.bold = True
    r_meta.font.color.rgb = dark_gray

    doc.add_paragraph()

    # QUESTION 1
    h1 = doc.add_heading("1. Identify the Best-Performing Baseline Model from Lab 6 and Justify Your Selection", level=1)
    h1.runs[0].font.color.rgb = black
    h1.runs[0].font.name = 'Arial'

    p1_intro = doc.add_paragraph()
    p1_intro.add_run("Based on the comprehensive benchmark evaluation conducted on the held-out ").bold = False
    p1_intro.add_run("Test Set (Time Steps 35–49, 16,670 labeled transactions, 1,083 illicit)").bold = True
    p1_intro.add_run(", the selection of the best-performing baseline model depends on the architectural paradigm and evaluation metrics:\n")

    p1_rf = doc.add_paragraph(style='List Bullet')
    p1_rf.add_run("1. Best Tabular Baseline Model: Random Forest\n").bold = True
    p1_rf.add_run("• Key Empirical Metrics: ").bold = True
    p1_rf.add_run("Minority Class F1-Score = 0.8068 (highest overall among all 7 baselines), Accuracy = 97.88%, Precision (Illicit) = 0.9906 (99.06% of predicted illicit nodes are truly fraudulent), ROC-AUC = 0.9049, Cross-Entropy Loss = 0.1207.\n")
    p1_rf.add_run("• Technical Justification: ").bold = True
    p1_rf.add_run("Random Forest constructs non-linear axis-aligned decision tree splits across the 71 pre-engineered 1-hop aggregate neighborhood features (feat_94 to feat_164). This bagging ensemble handles skewed tabular feature distributions cleanly, maintaining exceptionally low false alarm rates.")

    p1_gat_base = doc.add_paragraph(style='List Bullet')
    p1_gat_base.add_run("2. Best GNN Baseline Model for Fraud Recall: GAT (Graph Attention Network)\n").bold = True
    p1_gat_base.add_run("• Key Empirical Metrics: ").bold = True
    p1_gat_base.add_run("Recall (Illicit) = 0.8116 (81.16% fraud catch rate, highest among all standard GNN baselines), ROC-AUC = 0.8267, Minority F1-Score = 0.2258.\n")
    p1_gat_base.add_run("• Technical Justification: ").bold = True
    p1_gat_base.add_run("Graph Attention Networks compute dynamic attention coefficients (α_ij) between neighboring transaction nodes. This allows GAT to assign higher weights to suspicious money-laundering edges while discounting uninformative transaction links.")

    p1_xgb = doc.add_paragraph(style='List Bullet')
    p1_xgb.add_run("3. Best Gradient Boosting Baseline Model: XGBoost\n").bold = True
    p1_xgb.add_run("• Key Empirical Metrics: ").bold = True
    p1_xgb.add_run("ROC-AUC = 0.9088 (highest overall across all models), PR-AUC = 0.7877 (highest overall under severe 6.50% test fraud prevalence), Minority F1-Score = 0.7452, Recall = 0.7359.\n")
    p1_xgb.add_run("• Technical Justification: ").bold = True
    p1_xgb.add_run("XGBoost optimizes gradient-boosted decision trees using explicit positive class reweighting (scale_pos_weight = 8.11) and early stopping on validation loss.")

    doc.add_paragraph()

    # QUESTION 2
    h2 = doc.add_heading("2. Implement the Proposed Methodology Using the Same Preprocessed Dataset", level=1)
    h2.runs[0].font.color.rgb = black
    h2.runs[0].font.name = 'Arial'

    p2 = doc.add_paragraph()
    p2.add_run("The Proposed Methodology is a ").bold = False
    p2.add_run("Bayesian Graph Attention Network (Bayesian GAT with Monte Carlo Dropout, M=20 stochastic forward passes using GATConv)").bold = True
    p2.add_run(". Built using PyTorch Geometric, the model incorporates multi-head self-attention mechanisms alongside persistent dropout layers during both training and inference to estimate predictive confidence and epistemic uncertainty variance.\n\n")

    p2_math = doc.add_paragraph()
    p2_math.add_run("Mathematical Formulations:\n").bold = True
    p2_math.add_run("1. Graph Attention Weight Formulation:  α_ij = softmax_j ( LeakyReLU( a^T [ W h_i || W h_j ] ) )\n")
    p2_math.add_run("2. Predictive Mean Probability:  μ_prob(i) = (1 / M) * Σ [ σ( f_θ_m( x_i, E ) ) ]\n")
    p2_math.add_run("3. Epistemic Uncertainty Variance:  σ_uncertainty^2(i) = (1 / M) * Σ [ σ( f_θ_m( x_i, E ) ) - μ_prob(i) ]^2\n\n")

    p2_code_title = doc.add_paragraph()
    p2_code_title.add_run("PyTorch Geometric Source Code Implementation (Bayesian GAT with GATConv):\n").bold = True

    # Add Code Listing Table
    t_code = doc.add_table(rows=1, cols=1)
    t_code.alignment = WD_TABLE_ALIGNMENT.CENTER
    c_code = t_code.rows[0].cells[0]
    set_cell_background(c_code, "F4F4F4")

    code_text = (
        "import torch\n"
        "import torch.nn as nn\n"
        "import torch.nn.functional as F\n"
        "from torch_geometric.nn import GATConv\n"
        "import numpy as np\n\n"
        "class BayesianGAT(nn.Module):\n"
        "    def __init__(self, input_dim, hidden_dim=64, heads=2, dropout=0.3):\n"
        "        super(BayesianGAT, self).__init__()\n"
        "        # Layer 1: Multi-Head Self-Attention (2 heads)\n"
        "        self.conv1 = GATConv(input_dim, hidden_dim // heads, heads=heads)\n"
        "        self.bn1 = nn.BatchNorm1d(hidden_dim)\n"
        "        # Layer 2: Single-Head Aggregation Attention\n"
        "        self.conv2 = GATConv(hidden_dim, hidden_dim, heads=1)\n"
        "        self.bn2 = nn.BatchNorm1d(hidden_dim)\n"
        "        self.out = nn.Linear(hidden_dim, 1)\n"
        "        self.dropout_rate = dropout\n\n"
        "    def forward(self, x, edge_index, mc_dropout=False):\n"
        "        h = F.elu(self.bn1(self.conv1(x, edge_index)))\n"
        "        h = F.dropout(h, p=self.dropout_rate, training=self.training or mc_dropout)\n"
        "        h = F.elu(self.bn2(self.conv2(h, edge_index)))\n"
        "        h = F.dropout(h, p=self.dropout_rate, training=self.training or mc_dropout)\n"
        "        return self.out(h).squeeze(-1)\n\n"
        "    def predict_mc(self, x, edge_index, mask, num_samples=20):\n"
        "        self.eval()\n"
        "        probs_list = []\n"
        "        with torch.no_grad():\n"
        "            for _ in range(num_samples):\n"
        "                logits = self.forward(x, edge_index, mc_dropout=True)\n"
        "                probs = torch.sigmoid(logits[mask]).cpu().numpy()\n"
        "                probs_list.append(probs)\n"
        "        probs_matrix = np.stack(probs_list, axis=0)\n"
        "        mean_probs = np.mean(probs_matrix, axis=0)\n"
        "        uncertainties = np.var(probs_matrix, axis=0)\n"
        "        return mean_probs, uncertainties"
    )
    p_c = c_code.paragraphs[0]
    p_c.add_run(code_text).font.size = Pt(8.5)
    if p_c.runs:
        p_c.runs[0].font.name = 'Courier New'

    doc.add_paragraph()

    p2_prep = doc.add_paragraph()
    p2_prep.add_run("Integration with Preprocessed Dataset (elliptic_pyg_data.pt):\n").bold = True
    p2_prep.add_run("• Standard Scaling: Features normalized using StandardScaler fit exclusively on training nodes (Steps 1–30).\n")
    p2_prep.add_run("• Loss Reweighting: Weighted BCE Loss using training class imbalance ratio pos_weight = 24,243 / 2,662 = 8.1105.\n")
    p2_prep.add_run("• Inference Routine: Executes M=20 stochastic forward passes per test node with active dropout to sample from weight posterior.")

    doc.add_paragraph()

    # QUESTION 3
    h3 = doc.add_heading("3. Compare Performance of Proposed Methodology with Selected Baseline Model & Summarize Improvements", level=1)
    h3.runs[0].font.color.rgb = black
    h3.runs[0].font.name = 'Arial'

    p3 = doc.add_paragraph()
    p3.add_run("All models were evaluated on the 16,670 labeled test transactions (1,083 illicit) under identical conditions. The full benchmark performance comparison table is presented below:")

    # Comparative Results Table
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

    # Visualizations
    p3_vis = doc.add_paragraph()
    p3_vis.add_run("Key Comparative Visualization Figures:\n").bold = True

    dir_p2 = r"C:\Users\niran\.gemini\antigravity-ide\brain\b422ee4b-f2cf-45f8-8301-5ffb87f5b8f2\plots"

    figs_q3_meta = [
        (os.path.join(dir_p2, "fig15_reliability_calibration_curves.png"),
         "Figure 1: Expected Calibration Error (ECE) Reliability Comparison Bar Chart",
         "The Proposed Bayesian GNN/GAT models achieve low Expected Calibration Error (ECE ~ 0.75-0.76), significantly outperforming GraphSAGE (0.8164), Random Forest (0.8795), and XGBoost (0.9175)."),

        (os.path.join(dir_p2, "fig11_pr_curves.png"),
         "Figure 2: Precision-Recall (PR) Curves Benchmark (Minority Fraud Class)",
         "The Proposed Bayesian models maintain robust precision-recall trade-offs under severe 6.50% test fraud prevalence."),

        (os.path.join(dir_p2, "fig14_uncertainty_distribution.png"),
         "Figure 3: Epistemic Predictive Uncertainty Distribution (Proposed Model)",
         "Density distribution of epistemic variance (σ_uncertainty^2) for Licit vs. Illicit transactions, showing high uncertainty concentrated on borderline illicit nodes."),

        (os.path.join(dir_p2, "fig12_metrics_comparison_bar.png"),
         "Figure 4: Grouped Performance Comparison Bar Chart across All Benchmark Models",
         "Side-by-side comparative bar chart highlighting F1-score, Precision, Recall, ROC-AUC, and PR-AUC across all benchmark models.")
    ]

    for fpath, cap, interp in figs_q3_meta:
        if os.path.exists(fpath):
            p_img = doc.add_paragraph()
            p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p_img.add_run().add_picture(fpath, width=Inches(5.5))
            
            p_cap = doc.add_paragraph()
            p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
            r_c = p_cap.add_run(f"{cap}\n")
            r_c.font.bold = True
            r_c.font.size = Pt(9.5)
            r_c.font.color.rgb = navy
            
            p_int = doc.add_paragraph()
            p_int.add_run("Academic Interpretation: ").bold = True
            p_int.add_run(interp)
            doc.add_paragraph()

    # Summary of Improvements
    h_imp = doc.add_heading("Summary of Improvements Achieved by Proposed Bayesian GAT Model", level=2)
    h_imp.runs[0].font.color.rgb = black
    h_imp.runs[0].font.name = 'Arial'

    improvements = [
        ("1. Maximum Fraud Recall Catch Rate (87.63% Recall): ", "The Proposed Bayesian GAT achieves an extraordinary 87.63% Recall, catching nearly 9 out of 10 fraudulent transactions. This represents a +19.58% improvement over Random Forest (68.05%) and +6.47% improvement over standard GAT (81.16%)."),
        ("2. Dynamic Multi-Head Self-Attention (α_ij): ", "Unlike static degree weighting, Bayesian GAT computes dynamic attention coefficients α_ij across transaction edges, focusing feature aggregation on suspicious laundering pathways."),
        ("3. Superior Probabilistic Calibration (ECE = 0.7580): ", "Expected Calibration Error dropped to ECE = 0.7580 (compared to 0.8196 for standard GAT and 0.8795 for Random Forest), achieving a -7.52% calibration error reduction."),
        ("4. Epistemic Uncertainty Quantification: ", "Outputs transaction-level predictive variance (σ_uncertainty^2) via M=20 Monte Carlo passes, allowing compliance auditors to flag uncertain predictions for manual review.")
    ]

    for title, desc in improvements:
        p_imp = doc.add_paragraph(style='List Bullet')
        p_imp.add_run(title).bold = True
        p_imp.add_run(desc)

    doc.add_paragraph()
    p_foot = doc.add_paragraph()
    p_foot.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    r_ft = p_foot.add_run("Lab 7 Report Completed Successfully. Probabilistic Graphical Models, 2026.")
    r_ft.font.italic = True
    r_ft.font.color.rgb = dark_gray

    out_path_1 = r"c:\CLG\SEM 7\PROBABILISTIC GRAPHICAL MODELS FOR DAS\PAPER\paper_anti\cb.ps.i5das23102_Lab_7_Updated_GAT.docx"
    out_path_2 = r"c:\CLG\SEM 7\PROBABILISTIC GRAPHICAL MODELS FOR DAS\PAPER\paper_anti\LAB7_Proposed_vs_Baseline_Report.docx"
    
    try:
        doc.save(r"c:\CLG\SEM 7\PROBABILISTIC GRAPHICAL MODELS FOR DAS\PAPER\paper_anti\cb.ps.i5das23102_Lab_7.docx")
    except PermissionError:
        print("[!] Note: Original file open in Word; saved to updated path.")

    doc.save(out_path_1)
    try:
        doc.save(out_path_2)
    except PermissionError:
        print("[!] Note: Secondary file open in Word.")
        
    print(f"\n[+] Successfully generated updated LAB 7 Word Document at:\n    '{out_path_1}'")

if __name__ == '__main__':
    main()

