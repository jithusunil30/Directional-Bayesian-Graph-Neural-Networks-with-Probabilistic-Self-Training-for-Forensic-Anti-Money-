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
    print("   Generating Lab 6 Question Answer Word Report    ")
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
    r_title = p_title.add_run("LAB 6: BASELINE SELECTION, PROPOSED METHODOLOGY IMPLEMENTATION & COMPARATIVE ANALYSIS REPORT\n")
    r_title.font.name = 'Arial'
    r_title.font.size = Pt(16)
    r_title.font.bold = True
    r_title.font.color.rgb = navy

    r_sub = p_title.add_run("Detailed Solutions and Empirical Analysis on the Elliptic Bitcoin Dataset\n")
    r_sub.font.name = 'Arial'
    r_sub.font.size = Pt(11)
    r_sub.font.italic = True
    r_sub.font.color.rgb = dark_gray

    p_meta = doc.add_paragraph()
    p_meta.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_meta = p_meta.add_run("Course: Probabilistic Graphical Models for DAS | Dataset: Elliptic Bitcoin Dataset | Date: 2026")
    r_meta.font.name = 'Arial'
    r_meta.font.size = Pt(9.5)
    r_meta.font.bold = True
    r_meta.font.color.rgb = dark_gray

    doc.add_paragraph()

    # SECTION 1
    h1 = doc.add_heading("Question 1: Best-Performing Baseline Model Identification & Justification", level=1)
    h1.runs[0].font.color.rgb = black
    h1.runs[0].font.name = 'Arial'

    p1 = doc.add_paragraph()
    p1.add_run("From the empirical benchmarks evaluated on the held-out Test Set (Time Steps 35–49, 16,670 labeled transactions), the selection of the best-performing baseline model depends on the architectural paradigm and optimization objective:\n\n")

    p1_rf = doc.add_paragraph(style='List Bullet')
    p1_rf.add_run("1. Best Tabular Baseline Model: Random Forest\n").bold = True
    p1_rf.add_run("• Performance Metrics: Minority F1-Score = 0.8068 (highest overall), Accuracy = 97.88%, Precision (Illicit) = 0.9906 (99.06% of predicted illicit nodes are truly fraudulent), ROC-AUC = 0.9049.\n")
    p1_rf.add_run("• Technical Justification: Random Forest effectively creates axis-aligned decision tree splits across the 71 pre-engineered 1-hop aggregate neighborhood features (feat_94 to feat_164). This tabular ensemble handles skewed features cleanly, maintaining extremely low false alarm rates.")

    p1_xgb = doc.add_paragraph(style='List Bullet')
    p1_xgb.add_run("2. Best Gradient Boosting Baseline Model: XGBoost\n").bold = True
    p1_xgb.add_run("• Performance Metrics: ROC-AUC = 0.9088 (highest overall across all 8 models), PR-AUC = 0.7877 (highest overall), F1-Score = 0.7452, Recall = 0.7359.\n")
    p1_xgb.add_run("• Technical Justification: XGBoost optimizes gradient-boosted decision trees using explicit class weighting (scale_pos_weight = 8.11) and early stopping, maximizing precision-recall operating trade-offs.")

    p1_sage = doc.add_paragraph(style='List Bullet')
    p1_sage.add_run("3. Best Graph Neural Network Baseline Model: GraphSAGE\n").bold = True
    p1_sage.add_run("• Performance Metrics: Recall = 0.7932, PR-AUC = 0.3883, ROC-AUC = 0.8525, Minority F1-Score = 0.2433.\n")
    p1_sage.add_run("• Technical Justification: Among deterministic GNN baselines (GCN, GraphSAGE, GAT), GraphSAGE achieves the top PR-AUC and ROC-AUC scores. Its spatial mean pooling aggregator aggregates neighborhood representations effectively without suffering from oversmoothing on dynamic scale-free graphs.")

    doc.add_paragraph()

    # SECTION 2
    h2 = doc.add_heading("Question 2: Implementation of Proposed Methodology", level=1)
    h2.runs[0].font.color.rgb = black
    h2.runs[0].font.name = 'Arial'

    p2 = doc.add_paragraph()
    p2.add_run("The Proposed Methodology is a ").bold = True
    p2.add_run("Bayesian Graph Neural Network (Bayesian GNN with Monte Carlo Dropout, M=20 stochastic forward passes)").bold = True
    p2.add_run(". Built using PyTorch Geometric, the model incorporates persistent dropout layers during both training and inference to approximate Bayesian posterior inference over graph network weights.\n\n")

    p2_math = doc.add_paragraph()
    p2_math.add_run("Monte Carlo Sampling Formulations:\n").bold = True
    p2_math.add_run("1. Predictive Mean Probability:  μ_prob(i) = (1 / M) * Σ [ σ( f_θ_m( x_i, E ) ) ]\n")
    p2_math.add_run("2. Epistemic Uncertainty Variance:  σ_uncertainty^2(i) = (1 / M) * Σ [ σ( f_θ_m( x_i, E ) ) - μ_prob(i) ]^2\n\n")

    p2_code = doc.add_paragraph()
    p2_code.add_run("Code Implementation Listing (from baseline_models.py):\n").bold = True

    # Add code block table
    t_code = doc.add_table(rows=1, cols=1)
    t_code.alignment = WD_TABLE_ALIGNMENT.CENTER
    c_code = t_code.rows[0].cells[0]
    set_cell_background(c_code, "F4F4F4")
    
    code_text = (
        "class BayesianGNN(nn.Module):\n"
        "    def __init__(self, input_dim, hidden_dim=64, dropout=0.3):\n"
        "        super(BayesianGNN, self).__init__()\n"
        "        self.conv1 = SAGEConv(input_dim, hidden_dim)\n"
        "        self.bn1 = nn.BatchNorm1d(hidden_dim)\n"
        "        self.conv2 = SAGEConv(hidden_dim, hidden_dim)\n"
        "        self.bn2 = nn.BatchNorm1d(hidden_dim)\n"
        "        self.out = nn.Linear(hidden_dim, 1)\n"
        "        self.dropout_rate = dropout\n\n"
        "    def forward(self, x, edge_index, mc_dropout=False):\n"
        "        h = F.relu(self.bn1(self.conv1(x, edge_index)))\n"
        "        h = F.dropout(h, p=self.dropout_rate, training=self.training or mc_dropout)\n"
        "        h = F.relu(self.bn2(self.conv2(h, edge_index)))\n"
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

    # SECTION 3
    h3 = doc.add_heading("Question 3: Performance Comparison & Summary of Improvements", level=1)
    h3.runs[0].font.color.rgb = black
    h3.runs[0].font.name = 'Arial'

    p3 = doc.add_paragraph()
    p3.add_run("All eight models were benchmarked under identical conditions on the 16,670 labeled test transactions (1,083 illicit). The results are summarized below:")

    # Comparative Table
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
    p3_vis.add_run("Key Comparison Figures & Visualizations:\n").bold = True

    dir_p2 = r"C:\Users\niran\.gemini\antigravity-ide\brain\b422ee4b-f2cf-45f8-8301-5ffb87f5b8f2\plots"

    figs_q3_meta = [
        (os.path.join(dir_p2, "fig15_reliability_calibration_curves.png"),
         "Figure 1: Expected Calibration Error (ECE) Reliability Comparison Bar Chart",
         "The Proposed Bayesian GNN achieves the lowest Expected Calibration Error (ECE = 0.7515), outperforming GraphSAGE (0.8164), Random Forest (0.8795), and XGBoost (0.9175)."),

        (os.path.join(dir_p2, "fig11_pr_curves.png"),
         "Figure 2: Precision-Recall (PR) Curves Benchmark (Minority Class: Illicit)",
         "The Proposed Bayesian GNN reaches PR-AUC = 0.4930, demonstrating a +26.96% relative improvement over standard GraphSAGE (0.3883)."),

        (os.path.join(dir_p2, "fig14_uncertainty_distribution.png"),
         "Figure 3: Epistemic Predictive Uncertainty Distribution (Proposed Model)",
         "Density distribution of epistemic variance (σ_uncertainty^2) for Licit vs. Illicit transactions, showing high uncertainty concentrated on borderline illicit nodes.")
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
            p_int.add_run("Interpretation: ").bold = True
            p_int.add_run(interp)
            doc.add_paragraph()

    # Detailed Summary of Improvements
    h_imp = doc.add_heading("Summary of Improvements Achieved", level=2)
    h_imp.runs[0].font.color.rgb = black
    h_imp.runs[0].font.name = 'Arial'

    improvements = [
        ("1. Superior Probabilistic Calibration (Lowest ECE): ", "Expected Calibration Error dropped to ECE = 0.7515 (compared to 0.8164 for GraphSAGE and 0.8795 for Random Forest), achieving a -7.94% calibration error reduction over GraphSAGE and -14.55% reduction over Random Forest."),
        ("2. Highest PR-AUC Among GNN Architectures: ", "PR-AUC increased from 0.3883 (GraphSAGE) to 0.4930 (Proposed Bayesian GNN), representing a +26.96% improvement in precision-recall trade-off under severe minority fraud prevalence (6.50%)."),
        ("3. 32.82% Reduction in Cross-Entropy Loss: ", "Cross-Entropy Loss dropped from 1.1080 (GraphSAGE) to 0.7444 (Proposed Bayesian GNN)."),
        ("4. High Fraud Recall Catch Rate: ", "Achieved 79.87% Recall, catching +11.82% more fraudulent transactions than Random Forest (68.05%)."),
        ("5. Epistemic Uncertainty Quantification: ", "Outputs transaction-level predictive variance (σ_uncertainty^2), enabling risk-aware human compliance auditing.")
    ]

    for title, desc in improvements:
        p_imp = doc.add_paragraph(style='List Bullet')
        p_imp.add_run(title).bold = True
        p_imp.add_run(desc)

    doc.add_paragraph()
    p_foot = doc.add_paragraph()
    p_foot.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    r_ft = p_foot.add_run("Lab 6 Report Generated Successfully. Probabilistic Graphical Models, 2026.")
    r_ft.font.italic = True
    r_ft.font.color.rgb = dark_gray

    output_docx_path = r"c:\CLG\SEM 7\PROBABILISTIC GRAPHICAL MODELS FOR DAS\PAPER\paper_anti\Lab6_Proposed_Vs_Baseline_Report.docx"
    doc.save(output_docx_path)
    print(f"\n[+] Lab 6 Question Answer Word Document created at:\n    '{output_docx_path}'")

if __name__ == '__main__':
    main()
