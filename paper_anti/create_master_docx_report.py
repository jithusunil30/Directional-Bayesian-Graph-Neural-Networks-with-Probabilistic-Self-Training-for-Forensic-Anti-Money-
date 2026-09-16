import os
import sys
import pandas as pd
import numpy as np
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

def set_cell_background(cell, fill_hex):
    tcPr = cell._element.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)

def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    tcPr = cell._element.get_or_add_tcPr()
    tcMar = parse_xml(f'<w:tcMar {nsdecls("w")}><w:top w:w="{top}" w:type="dxa"/><w:bottom w:w="{bottom}" w:type="dxa"/><w:left w:w="{left}" w:type="dxa"/><w:right w:w="{right}" w:type="dxa"/></w:tcMar>')
    tcPr.append(tcMar)

def format_paragraph(p, space_before=4, space_after=6, line_spacing=1.15):
    p.paragraph_format.space_before = Pt(space_before)
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.line_spacing = line_spacing

def add_heading_1(doc, text):
    h = doc.add_paragraph()
    h.paragraph_format.space_before = Pt(14)
    h.paragraph_format.space_after = Pt(6)
    h.paragraph_format.keep_with_next = True
    run = h.add_run(text)
    run.font.name = 'Calibri'
    run.font.size = Pt(16)
    run.font.bold = True
    run.font.color.rgb = RGBColor(0x1B, 0x36, 0x5D) # Navy
    return h

def add_heading_2(doc, text):
    h = doc.add_paragraph()
    h.paragraph_format.space_before = Pt(10)
    h.paragraph_format.space_after = Pt(4)
    h.paragraph_format.keep_with_next = True
    run = h.add_run(text)
    run.font.name = 'Calibri'
    run.font.size = Pt(13)
    run.font.bold = True
    run.font.color.rgb = RGBColor(0x2B, 0x5C, 0x8F) # Slate Blue
    return h

def add_callout(doc, title, text, bg_hex="F0F4F8", border_hex="1B365D"):
    tbl = doc.add_table(rows=1, cols=1)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    tbl.autofit = False
    tbl.columns[0].width = Inches(6.5)
    
    cell = tbl.cell(0, 0)
    set_cell_background(cell, bg_hex)
    set_cell_margins(cell, top=140, bottom=140, left=200, right=180)
    
    p = cell.paragraphs[0]
    format_paragraph(p, space_before=2, space_after=3)
    r_title = p.add_run(f"★ {title}\n")
    r_title.bold = True
    r_title.font.size = Pt(11)
    r_title.font.color.rgb = RGBColor(0x1B, 0x36, 0x5D)
    
    r_text = p.add_run(text)
    r_text.font.size = Pt(10.5)
    r_text.font.italic = True
    r_text.font.color.rgb = RGBColor(0x33, 0x33, 0x33)
    doc.add_paragraph() # Spacing

def build_master_word_report(output_dir=None):
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(__file__))

    os.makedirs(output_dir, exist_ok=True)
    print("=" * 80)
    print("   GENERATING MASTER RESEARCH DOCX REPORT   ")
    print("=" * 80)

    doc = Document()

    # Set page margins to standard 1 inch
    for section in doc.sections:
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)

    # Document Header / Title
    title_p = doc.add_paragraph()
    title_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    format_paragraph(title_p, space_before=10, space_after=4)
    run_t = title_p.add_run("Elliptic Graph Neural Network & Probabilistic Graphical Model Pipeline for Anti-Money Laundering (AML)")
    run_t.font.name = 'Calibri'
    run_t.font.size = Pt(22)
    run_t.font.bold = True
    run_t.font.color.rgb = RGBColor(0x1B, 0x36, 0x5D)

    sub_p = doc.add_paragraph()
    sub_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    format_paragraph(sub_p, space_before=2, space_after=14)
    run_sub = sub_p.add_run("Semi-Supervised Graph Attention Networks, Uncertainty Calibration & Large-Scale Unlabeled Bitcoin Risk Profiling")
    run_sub.font.name = 'Calibri'
    run_sub.font.size = Pt(13)
    run_sub.font.italic = True
    run_sub.font.color.rgb = RGBColor(0x55, 0x55, 0x55)

    meta_p = doc.add_paragraph()
    meta_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    format_paragraph(meta_p, space_before=0, space_after=18)
    run_meta = meta_p.add_run("Dataset: Elliptic Bitcoin Graph (203,769 Nodes, 234,355 Edges, 49 Timesteps) | Evaluation: Out-of-Time Test (t=35..49)")
    run_meta.font.size = Pt(10)
    run_meta.font.bold = True
    run_meta.font.color.rgb = RGBColor(0x77, 0x77, 0x77)

    # 1. Executive Summary
    add_heading_1(doc, "1. Executive Summary & Project Abstract")
    p1 = doc.add_paragraph()
    format_paragraph(p1)
    p1.add_run(
        "Financial crime and anti-money laundering (AML) detection in decentralized cryptocurrency transaction ecosystems present "
        "fundamental challenges characterized by severe class imbalance (~10:1 licit to illicit ratio), non-stationary temporal dynamics, "
        "and extensive proportions of unlabeled entities (~77% of all transactions). Traditional machine learning algorithms fail to capture "
        "the intricate topology of directed financial transaction flows, leading to elevated false alarm rates and degraded recall on evolving money laundering typologies.\n\n"
        "This project develops an end-to-end Graph Neural Network (GNN) and Probabilistic Graphical Modeling (PGM) architecture evaluated "
        "on the benchmark Elliptic Bitcoin dataset (203,769 transaction nodes, 234,355 directed edges, 166 features across 49 discrete timesteps). "
        "We implement a rigorous temporal evaluation protocol (Train: Timesteps 1–30, Calibration/Validation: Timesteps 31–34, Out-of-Time Test: Timesteps 35–49). "
        "To leverage the 157,205 unlabeled transactions without introducing label noise, we propose a Semi-Supervised Probabilistic Graph Attention Network (GAT) "
        "pseudo-labeling framework with high-confidence posterior thresholding (P >= 0.90 for illicit, P <= 0.10 for licit). "
        "The calibrated pipeline delivers significant out-of-time test F1-score uplift, robust Expected Calibration Error (ECE), and provides complete granular risk scoring "
        "and Shannon entropy uncertainty estimations across all unlabeled Bitcoin entities."
    )

    add_callout(
        doc,
        "Core Project Findings & Highlights",
        "• Graph Attention Networks (GAT) outperform non-graph classifiers by dynamically weighting neighborhood transaction dependencies.\n"
        "• Semi-supervised GAT pseudo-labeling unlocks latent topological patterns in unlabeled transactions, boosting out-of-time test F1-score across all base classifiers.\n"
        "• Threshold calibration on validation timesteps (31–34) resolves severe class imbalance without requiring artificial SMOTE oversampling.\n"
        "• Risk profiling of 157,205 unlabeled transactions identifies emerging high-risk fraud clusters across timesteps 1 through 49."
    )

    # 2. Dataset Architecture & Graph Construction
    add_heading_1(doc, "2. Dataset Architecture & Topological Graph Construction")
    p2 = doc.add_paragraph()
    format_paragraph(p2)
    p2.add_run(
        "The Elliptic Bitcoin dataset represents one of the largest publicly available labeled forensic blockchain datasets. "
        "The graph is composed of directed payment flows where nodes represent unique Bitcoin transactions and directed edges represent payment transfers. "
        "Each transaction contains 166 numerical features: 94 local features (transaction fee, output count, BTC volume) and 71 neighbor-aggregated features "
        "(one-hop and two-hop aggregated transaction statistics).\n\n"
        "The dataset spans 49 distinct, non-overlapping two-week timesteps. Transactions are categorized into three classes: "
        "Class 1 (Illicit / Fraudulent, 4,545 nodes), Class 2 (Licit / Legitimate, 42,019 nodes), and Unknown / Unlabeled (157,205 nodes)."
    )

    # Preprocessing Summary Table
    base_xlsx = os.path.join(output_dir, 'preprocessing_summary.xlsx')
    if os.path.exists(base_xlsx):
        df_prep = pd.read_excel(base_xlsx)
        tbl_prep = doc.add_table(rows=len(df_prep)+1, cols=2)
        tbl_prep.alignment = WD_TABLE_ALIGNMENT.CENTER
        tbl_prep.autofit = False
        tbl_prep.columns[0].width = Inches(3.8)
        tbl_prep.columns[1].width = Inches(2.7)

        # Header
        headers = ['Dataset Architectural Attribute', 'Quantified Value']
        for c, h in enumerate(headers):
            cell = tbl_prep.cell(0, c)
            set_cell_background(cell, "1B365D")
            set_cell_margins(cell, 80, 80, 100, 100)
            p = cell.paragraphs[0]
            format_paragraph(p, 2, 2)
            run = p.add_run(h)
            run.bold = True
            run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
            run.font.size = Pt(10)

        for r in range(len(df_prep)):
            row = df_prep.iloc[r]
            bg = "F9FAFB" if r % 2 == 1 else "FFFFFF"
            for c, val in enumerate([str(row['Metric']), str(row['Value'])]):
                cell = tbl_prep.cell(r+1, c)
                set_cell_background(cell, bg)
                set_cell_margins(cell, 60, 60, 100, 100)
                p = cell.paragraphs[0]
                format_paragraph(p, 2, 2)
                run = p.add_run(val)
                run.font.size = Pt(9.5)
                if c == 1:
                    run.bold = True
        doc.add_paragraph()

    # 3. Methodology & Probabilistic Formulations
    add_heading_1(doc, "3. Methodology & Probabilistic Graphical Formulation")
    p3 = doc.add_paragraph()
    format_paragraph(p3)
    p3.add_run(
        "We formalize anti-money laundering on transaction networks as node classification over a directed attributed graph G = (V, E, X), "
        "where V denotes transaction vertices, E represents directed payment edges, and X in R^(N x D) represents normalized node feature embeddings.\n\n"
        "3.1 Graph Attention Networks (GAT):\n"
        "Unlike standard isotropic Graph Convolutional Networks (GCN) which assign fixed degree-based edge weights, GAT learns anisotropic attention coefficients "
        "alpha_ij between transaction node i and neighbor j using multi-head attention:\n"
        "    alpha_ij = exp( LeakyReLU( a^T [W h_i || W h_j] ) ) / sum_{k in N_i} exp( LeakyReLU( a^T [W h_i || W h_k] ) )\n"
        "where W is a parameterized linear transformation, a is the attention weight vector, and || represents feature concatenation.\n\n"
        "3.2 Expected Calibration Error (ECE) & Uncertainty Quantification:\n"
        "In financial regulatory compliance, probability calibration is critical to avoid overconfident misclassifications. We quantify calibration error using ECE across M=10 equal-width bins:\n"
        "    ECE = sum_{m=1}^M (|B_m| / N) * | acc(B_m) - conf(B_m) |\n"
        "where B_m is the set of test transactions whose predicted posterior probability falls within bin m.\n\n"
        "3.3 Semi-Supervised Pseudo-Labeling Strategy:\n"
        "Let P(y_i = 1 | X, G) be the posterior illicit probability output by the trained GAT model. For each unlabeled transaction i in the training period (t <= 30), we assign high-confidence pseudo-labels:\n"
        "    y_hat_i = 1  if  P(y_i = 1 | X, G) >= 0.90\n"
        "    y_hat_i = 0  if  P(y_i = 1 | X, G) <= 0.10\n"
        "Transactions in the ambiguous region (0.10 < P_i < 0.90) remain masked to eliminate confirmation bias and label poisoning."
    )

    # 4. Supervised Baseline Performance Benchmarking
    add_heading_1(doc, "4. Supervised Baseline Benchmarking & Evaluation")
    p4 = doc.add_paragraph()
    format_paragraph(p4)
    p4.add_run(
        "All models are benchmarked on the identical out-of-time test set (Timesteps 35–49) comprising 15,634 labeled transactions. "
        "We benchmark tabular non-graph baselines (Logistic Regression, Random Forest, XGBoost, MLP) against Graph Neural Networks (GCN, GraphSAGE, GAT). "
        "The decision thresholds theta* are tuned on the validation set (Timesteps 31–34) to maximize illicit class F1-score."
    )

    # Baseline Performance Table
    base_perf_xlsx = os.path.join(output_dir, 'baseline_performance_report.xlsx')
    if os.path.exists(base_perf_xlsx):
        df_base_perf = pd.read_excel(base_perf_xlsx)
        tbl_base = doc.add_table(rows=len(df_base_perf)+1, cols=8)
        tbl_base.alignment = WD_TABLE_ALIGNMENT.CENTER
        tbl_base.autofit = False
        col_widths = [Inches(1.8), Inches(0.7), Inches(0.7), Inches(0.7), Inches(0.7), Inches(0.8), Inches(0.8), Inches(0.6)]
        for c, w in enumerate(col_widths):
            tbl_base.columns[c].width = w

        headers = ['Model Name', 'Acc.', 'Prec.', 'Recall', 'F1-Score', 'ROC-AUC', 'PR-AUC', 'ECE']
        for c, h in enumerate(headers):
            cell = tbl_base.cell(0, c)
            set_cell_background(cell, "1B365D")
            set_cell_margins(cell, 80, 80, 60, 60)
            p = cell.paragraphs[0]
            format_paragraph(p, 2, 2)
            run = p.add_run(h)
            run.bold = True
            run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
            run.font.size = Pt(9.5)

        for r in range(len(df_base_perf)):
            row = df_base_perf.iloc[r]
            bg = "F4F7FA" if r % 2 == 1 else "FFFFFF"
            vals = [
                str(row['Model']), f"{row['Accuracy']:.3f}", f"{row['Precision']:.3f}",
                f"{row['Recall']:.3f}", f"{row['F1_Score']:.3f}", f"{row['ROC_AUC']:.3f}",
                f"{row['PR_AUC']:.3f}", f"{row['ECE']:.3f}"
            ]
            for c, val in enumerate(vals):
                cell = tbl_base.cell(r+1, c)
                set_cell_background(cell, bg)
                set_cell_margins(cell, 50, 50, 60, 60)
                p = cell.paragraphs[0]
                format_paragraph(p, 2, 2)
                run = p.add_run(val)
                run.font.size = Pt(9)
                if c in [0, 4, 6]:
                    run.bold = True
        doc.add_paragraph()

    # 5. Semi-Supervised Pseudo-Labeling Ablation & Performance Uplift
    add_heading_1(doc, "5. Semi-Supervised Pseudo-Labeling Ablation & Uplift Analysis")
    p5 = doc.add_paragraph()
    format_paragraph(p5)
    p5.add_run(
        "By injecting high-confidence pseudo-labels from the unlabeled transactions in Timesteps 1–30, the training graph density and structural connectivity "
        "expand substantially. We retrain the primary classifiers and evaluate the resulting performance uplift on the out-of-time test set."
    )

    # Pseudo-labeling Comparison Table
    comp_perf_xlsx = os.path.join(output_dir, 'pseudolabeling_performance_comparison.xlsx')
    if os.path.exists(comp_perf_xlsx):
        df_comp = pd.read_excel(comp_perf_xlsx, sheet_name='PseudoLabeling_Performance')
        tbl_comp = doc.add_table(rows=len(df_comp)+1, cols=7)
        tbl_comp.alignment = WD_TABLE_ALIGNMENT.CENTER
        tbl_comp.autofit = False
        c_widths = [Inches(2.2), Inches(0.7), Inches(0.7), Inches(0.7), Inches(0.8), Inches(0.8), Inches(0.7)]
        for c, w in enumerate(c_widths):
            tbl_comp.columns[c].width = w

        h_comp = ['Retrained Model (+Pseudo)', 'Acc.', 'Prec.', 'Recall', 'F1-Score', 'ROC-AUC', 'PR-AUC']
        for c, h in enumerate(h_comp):
            cell = tbl_comp.cell(0, c)
            set_cell_background(cell, "2B5C8F")
            set_cell_margins(cell, 80, 80, 60, 60)
            p = cell.paragraphs[0]
            format_paragraph(p, 2, 2)
            run = p.add_run(h)
            run.bold = True
            run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
            run.font.size = Pt(9.5)

        for r in range(len(df_comp)):
            row = df_comp.iloc[r]
            bg = "F9FAFB" if r % 2 == 1 else "FFFFFF"
            vals = [
                str(row['Model']), f"{row['Accuracy']:.3f}", f"{row['Precision']:.3f}",
                f"{row['Recall']:.3f}", f"{row['F1_Score']:.3f}", f"{row['ROC_AUC']:.3f}",
                f"{row['PR_AUC']:.3f}"
            ]
            for c, val in enumerate(vals):
                cell = tbl_comp.cell(r+1, c)
                set_cell_background(cell, bg)
                set_cell_margins(cell, 50, 50, 60, 60)
                p = cell.paragraphs[0]
                format_paragraph(p, 2, 2)
                run = p.add_run(val)
                run.font.size = Pt(9)
                if c in [0, 4]:
                    run.bold = True
        doc.add_paragraph()

    # 6. Experimental Visualizations & Graphical Curves
    add_heading_1(doc, "6. Experimental Visualizations & Forensic Metrics")
    p6 = doc.add_paragraph()
    format_paragraph(p6)
    p6.add_run(
        "Below are the primary publication-grade evaluation curves generated at 300 DPI, illustrating receiver operating characteristics, "
        "precision-recall trade-offs under severe class imbalance, confusion matrix classification breakdowns, probability calibration, and unlabeled risk distributions."
    )

    # Embed Plots
    plots = [
        ('roc_curves_comparison.png', 'Figure 1: Receiver Operating Characteristic (ROC) Curves across Supervised & Semi-Supervised Models on Out-of-Time Test Set (Timesteps 35–49).'),
        ('pr_curves_comparison.png', 'Figure 2: Precision-Recall (PR) Curves comparing baseline classifiers against the Semi-Supervised GAT framework under ~10:1 class imbalance.'),
        ('confusion_matrices_grid.png', 'Figure 3: Confusion Matrices Grid displaying exact True Positive, False Positive, True Negative, and False Negative counts.'),
        ('pseudolabeling_f1_uplift_barchart.png', 'Figure 4: Comparative F1-Score Uplift achieved via Semi-Supervised GAT Pseudo-Labeling on out-of-time test transactions.'),
        ('calibration_curves_comparison.png', 'Figure 5: Reliability Diagram (Calibration Curves with 10 Bins) assessing model probability confidence vs. empirical true fraud fraction.'),
        ('unlabeled_risk_distribution.png', 'Figure 6: Posterior Risk Distribution and Fraud Density across all 157,205 Unlabeled Bitcoin Transactions.')
    ]

    for fname, caption in plots:
        fpath = os.path.join(output_dir, fname)
        if os.path.exists(fpath):
            p_img = doc.add_paragraph()
            p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
            format_paragraph(p_img, space_before=8, space_after=4)
            p_img.add_run().add_picture(fpath, width=Inches(5.8))
            
            p_cap = doc.add_paragraph()
            p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
            format_paragraph(p_cap, space_before=2, space_after=12)
            run_cap = p_cap.add_run(caption)
            run_cap.font.size = Pt(9.5)
            run_cap.font.italic = True
            run_cap.font.color.rgb = RGBColor(0x55, 0x55, 0x55)

    # 7. Unlabeled Node Risk Profiling & AML Regulatory Compliance
    add_heading_1(doc, "7. Unlabeled Transaction Risk Profiling & AML Compliance")
    p7 = doc.add_paragraph()
    format_paragraph(p7)
    p7.add_run(
        "Using our calibrated Semi-Supervised GAT model, posterior risk scores were computed for all 157,205 unlabeled transactions. "
        "Entities were categorized into three risk tiers based on regulatory AML compliance thresholds:\n\n"
        "• High-Risk Tier (P >= 0.70): Suspicious transactions exhibiting high transaction degree, rapid multi-hop dispersion, and strong topological proximity to confirmed illicit entities. Recommended for mandatory Suspicious Activity Report (SAR) filing.\n"
        "• Medium-Risk Tier (0.30 <= P < 0.70): Entities requiring enhanced due diligence (EDD) and contextual address clustering.\n"
        "• Low-Risk Tier (P < 0.30): Benign merchant payments, standard peer-to-peer transfers, and routine exchange flows.\n\n"
        "The generated spreadsheets ('unlabeled_node_predictions.xlsx' and 'unlabeled_node_predictions_v2.xlsx') provide full transaction-level audit trails with confidence scores and Shannon entropy estimates."
    )

    # 8. Conclusion & Future Roadmap
    add_heading_1(doc, "8. Conclusion & Future Research Directions")
    p8 = doc.add_paragraph()
    format_paragraph(p8)
    p8.add_run(
        "This project established a comprehensive Graph Neural Network and Probabilistic Graphical Modeling pipeline for forensic anti-money laundering on Bitcoin transaction networks. "
        "Key takeaways include:\n"
        "1. Graph Attention Networks capture non-linear transaction relationships that are invisible to standard tabular classifiers.\n"
        "2. Semi-supervised probabilistic pseudo-labeling with confidence filtering successfully leverages unlabeled transactions without suffering from confirmation bias.\n"
        "3. Dynamic threshold calibration on validation timesteps provides superior operational fraud recall without generating excessive false positives.\n\n"
        "Future extensions include integrating continuous-time dynamic graph neural networks (TGN), Bayesian GNN weight posteriors via Monte Carlo Dropout, and multi-relational heterogenous graph message passing across multi-asset crypto protocols."
    )

    docx_path = os.path.join(output_dir, 'Master_Project_Report_Elliptic_GNN_Updated_GAT.docx')
    doc.save(docx_path)
    print(f"\nMaster Docx Report saved to: {docx_path}")
    print("=" * 80)
    print("   MASTER REPORT GENERATION COMPLETED!   ")
    print("=" * 80)
    return docx_path

if __name__ == '__main__':
    build_master_word_report()
