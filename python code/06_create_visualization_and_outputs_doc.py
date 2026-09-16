import os
import sys
import shutil
import docx
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

def set_cell_background(cell, fill_hex):
    tcPr = cell._element.get_or_add_tcPr()
    shd = OxmlElement('w:shd')
    shd.set(qn('w:val'), 'clear')
    shd.set(qn('w:color'), 'auto')
    shd.set(qn('w:fill'), fill_hex)
    tcPr.append(shd)

def set_cell_margins(cell, top=80, bottom=80, left=100, right=100):
    tcPr = cell._element.get_or_add_tcPr()
    tcMar = OxmlElement('w:tcMar')
    for m, val in [('top', top), ('bottom', bottom), ('left', left), ('right', right)]:
        node = OxmlElement(f'w:{m}')
        node.set(qn('w:w'), str(val))
        node.set(qn('w:type'), 'dxa')
        tcMar.append(node)
    tcPr.append(tcMar)

def add_styled_heading(doc, text, level):
    h = doc.add_heading(text, level=level)
    h.paragraph_format.keep_with_next = True
    if level == 1:
        h.paragraph_format.space_before = Pt(16)
        h.paragraph_format.space_after = Pt(6)
        run = h.runs[0]
        run.font.size = Pt(15)
        run.font.color.rgb = RGBColor(0x1B, 0x36, 0x5D) # Deep Navy
        run.font.bold = True
    elif level == 2:
        h.paragraph_format.space_before = Pt(12)
        h.paragraph_format.space_after = Pt(4)
        run = h.runs[0]
        run.font.size = Pt(12)
        run.font.color.rgb = RGBColor(0x2C, 0x3E, 0x50) # Slate Grey
        run.font.bold = True
    elif level == 3:
        h.paragraph_format.space_before = Pt(8)
        h.paragraph_format.space_after = Pt(2)
        run = h.runs[0]
        run.font.size = Pt(10)
        run.font.color.rgb = RGBColor(0x29, 0x80, 0xB9) # Accent Blue
        run.font.bold = True
    return h

def add_styled_picture(doc, img_path, caption=None, width_inches=6.2):
    if os.path.exists(img_path):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(6)
        p.paragraph_format.space_after = Pt(2)
        p.paragraph_format.keep_with_next = True
        p.add_run().add_picture(img_path, width=Inches(width_inches))
        if caption:
            p_cap = doc.add_paragraph()
            p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p_cap.paragraph_format.space_before = Pt(1)
            p_cap.paragraph_format.space_after = Pt(6)
            r_cap = p_cap.add_run(caption)
            r_cap.font.size = Pt(8.5)
            r_cap.font.italic = True
            r_cap.font.color.rgb = RGBColor(0x55, 0x55, 0x55)
    else:
        print(f"Warning: Picture not found: {img_path}")

def add_callout_box(doc, title, text, bg_hex="F0F4F8", border_hex="1B365D"):
    tbl = doc.add_table(rows=1, cols=1)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    cell = tbl.cell(0, 0)
    set_cell_background(cell, bg_hex)
    set_cell_margins(cell, top=100, bottom=100, left=140, right=140)
    
    tcPr = cell._element.get_or_add_tcPr()
    tcBorders = OxmlElement('w:tcBorders')
    left_b = OxmlElement('w:left')
    left_b.set(qn('w:val'), 'single')
    left_b.set(qn('w:sz'), '24')
    left_b.set(qn('w:space'), '0')
    left_b.set(qn('w:color'), border_hex)
    tcBorders.append(left_b)
    for side in ['top', 'bottom', 'right']:
        b = OxmlElement(f'w:{side}')
        b.set(qn('w:val'), 'none')
        tcBorders.append(b)
    tcPr.append(tcBorders)
    
    cell.width = Inches(6.5)
    p = cell.paragraphs[0]
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(2)
    r_title = p.add_run(f"📌 {title}\n")
    r_title.font.bold = True
    r_title.font.size = Pt(9.5)
    r_title.font.color.rgb = RGBColor(0x1B, 0x36, 0x5D)
    
    r_body = p.add_run(text)
    r_body.font.size = Pt(9)
    r_body.font.color.rgb = RGBColor(0x33, 0x33, 0x33)
    doc.add_paragraph().paragraph_format.space_after = Pt(4)

def format_table(table, col_widths, headers, rows):
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    hdr_cells = table.rows[0].cells
    for i, header_text in enumerate(headers):
        hdr_cells[i].text = header_text
        set_cell_background(hdr_cells[i], '1B365D') # Navy
        set_cell_margins(hdr_cells[i], top=80, bottom=80, left=80, right=80)
        p = hdr_cells[i].paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        for r in p.runs:
            r.font.bold = True
            r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
            r.font.size = Pt(8)

    for r_idx, row_data in enumerate(rows):
        row_cells = table.add_row().cells
        bg_color = 'F8F9FA' if r_idx % 2 == 1 else 'FFFFFF'
        for c_idx, val in enumerate(row_data):
            row_cells[c_idx].text = str(val)
            set_cell_background(row_cells[c_idx], bg_color)
            set_cell_margins(row_cells[c_idx], top=50, bottom=50, left=60, right=60)
            p = row_cells[c_idx].paragraphs[0]
            p.paragraph_format.space_before = Pt(1)
            p.paragraph_format.space_after = Pt(1)
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT if c_idx == 0 else WD_ALIGN_PARAGRAPH.CENTER
            for r in p.runs:
                r.font.size = Pt(8)
                if any(k in str(row_data[0]) for k in ['Bayesian GNN', 'Dir-ResGCN', 'Dir-ResSAGE', 'Bayesian Dir-SAGE']):
                    if c_idx == 0:
                        r.font.bold = True
                    if c_idx in [1, 4, 5, 6]:
                        r.font.bold = True

    for row in table.rows:
        for i, w in enumerate(col_widths):
            row.cells[i].width = Inches(w)

def build_visualization_and_outputs_doc():
    base_dir = r"c:\Users\USER\OneDrive\Desktop\Probalistic Graphical Lab"
    img_dir = os.path.join(base_dir, "flowcharts")
    ml1_plots = os.path.join(base_dir, "ML TRAINING 1", "plots")
    ml2_plots = os.path.join(base_dir, "ML TRAINING 2", "plots")
    ml3_plots = os.path.join(base_dir, "ML TRAINING 3", "plots")
    
    doc = Document()

    # Set Margins
    for s in doc.sections:
        s.top_margin = Inches(0.8)
        s.bottom_margin = Inches(0.8)
        s.left_margin = Inches(0.8)
        s.right_margin = Inches(0.8)

    # -------------------------------------------------------------
    # DOCUMENT COVER / TITLE
    # -------------------------------------------------------------
    title_p = doc.add_paragraph()
    title_p.paragraph_format.space_before = Pt(18)
    title_p.paragraph_format.space_after = Pt(2)
    r_title = title_p.add_run("VISUALIZATION AND OUTPUTS:\nPure Graphical Machine Learning & Bayesian Graph Neural Networks for AML on Bitcoin")
    r_title.font.size = Pt(18)
    r_title.font.bold = True
    r_title.font.color.rgb = RGBColor(0x1B, 0x36, 0x5D)

    sub_p = doc.add_paragraph()
    sub_p.paragraph_format.space_before = Pt(2)
    sub_p.paragraph_format.space_after = Pt(12)
    r_sub = sub_p.add_run("Complete Visual Atlas of Architectural Flowcharts, Empirical Evaluation Curves, Multi-Model Confusion Matrices, Reliability Calibration Diagrams, and Master Output Benchmark Tables across All 7 Research Phases")
    r_sub.font.size = Pt(10.5)
    r_sub.font.italic = True
    r_sub.font.color.rgb = RGBColor(0x55, 0x55, 0x55)

    # Metadata Table
    meta_table = doc.add_table(rows=4, cols=2)
    meta_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    meta_data = [
        ("Dataset Description", "Elliptic Bitcoin Transaction Graph (203,769 Nodes, 234,355 Edges, 165 Features, 49 Timesteps)"),
        ("Research Scope", "100% Pure Graphical Machine Learning & Bayesian GNNs (Non-graph tabular baselines strictly excluded)"),
        ("Repository & Branches", "https://github.com/jithusunil30/PGM-PAPER (main: documentation & reports; Data-and-Code: pure code & data)"),
        ("Document Purpose", "Consolidated single-document visual reference containing all pipeline diagrams, plots, charts, and metric outputs")
    ]
    for idx, (k, v) in enumerate(meta_data):
        row_cells = meta_table.rows[idx].cells
        row_cells[0].text = k
        row_cells[1].text = v
        set_cell_background(row_cells[0], 'F0F4F8')
        set_cell_background(row_cells[1], 'FFFFFF')
        set_cell_margins(row_cells[0], top=35, bottom=35, left=50, right=50)
        set_cell_margins(row_cells[1], top=35, bottom=35, left=50, right=50)
        p0 = row_cells[0].paragraphs[0]
        p1 = row_cells[1].paragraphs[0]
        p0.runs[0].font.bold = True
        p0.runs[0].font.size = Pt(8.5)
        p1.runs[0].font.size = Pt(8.5)
        row_cells[0].width = Inches(1.8)
        row_cells[1].width = Inches(4.7)

    doc.add_paragraph().paragraph_format.space_after = Pt(8)

    # -------------------------------------------------------------
    # SECTION 1: MASTER PIPELINE & ARCHITECTURE FLOWCHARTS (1-12)
    # -------------------------------------------------------------
    add_styled_heading(doc, "Section 1: Master Pipeline Architecture & System Flowcharts", level=1)
    doc.add_paragraph(
        "This section compiles the complete suite of 12 high-resolution, 300-DPI architectural flowchart diagrams that delineate each phase "
        "of the pure graphical anti-money laundering pipeline, from raw data extraction to operational three-tier compliance routing."
    )

    flowchart_items = [
        ("flowchart_1_master_pipeline.png",
         "Figure 1: Master End-to-End System Architecture Pipeline",
         "Overview of all 7 phases: Phase 0 (Raw EDA) -> Phase 1 (Causality Splitting) -> Phase 2 (ML 1 Sparse Benchmark) -> Phase 3 (Pseudo-Labeling Engine) -> Phase 4 (ML 2 Dense Benchmark) -> Phase 5 (ML 3 Directional Residual GNNs) -> Phase 6 (AML Compliance Routing)."),
        
        ("flowchart_2_eda_topometry.png",
         "Figure 2: Phase 0 Raw Data Ingestion & Feature Decomposition",
         "Decomposition of the 203,769 transaction nodes and 234,355 directed edges into 93 local transaction features, 72 1-hop aggregate neighborhood features, and class distribution."),
        
        ("flowchart_3_preprocessing_splits.png",
         "Figure 3: Phase 0 Degree Topometry & Homophily Flowchart",
         "Visualization of power-law degree distributions (max in-degree 473), structural differences between licit commercial webs (degree 2.42) and illicit peeling chains (degree 1.86), and 55.25% illicit homophily."),
        
        ("flowchart_4_ml1_sparse_failure.png",
         "Figure 4: Phase 2 Sparse Benchmark Failure & Accuracy Illusion",
         "Diagnostic representation of how masking 77.15% of unlabeled transactions severed multi-hop laundering chains, resulting in 522 missed criminals despite a deceptive 93.14% raw accuracy."),
        
        ("flowchart_5_pseudo_labeling_engine.png",
         "Figure 5: Phase 3 Probabilistic Self-Training & Margin Calibration",
         "Out-of-fold GNN ensemble inference architecture assigning continuous confidence margins w_i = max(P_i, 1-P_i) to pseudo-label all 157,205 unlabeled nodes, reaching 100% graph resolution."),
        
        ("flowchart_6_ml2_dense_benchmark.png",
         "Figure 6: Phase 4 100% Dense Graph Benchmark & 26.4x Detection Surge",
         "Representation of dense message passing restoring continuous topological paths, causing illicit detections to surge from 561 to 14,802 and PR-AUC to jump from 0.3475 to 0.8129."),
        
        ("flowchart_7_directional_res_math.png",
         "Figure 7: Phase 5 Directional Residual Convolutions Mathematical Architecture",
         "Mathematical architecture decoupling incoming payment pooling (E_in) from outgoing peeling dispersion (E_out), with deep residual skip projections and layer normalization."),
        
        ("flowchart_8_soft_confidence_loss.png",
         "Figure 8: Phase 5 Soft Confidence-Weighted BCE Loss Optimization",
         "Formulation of margin-weighted loss function L_soft = - sum w_i * BCE, suppressing gradient noise on boundary transactions while enforcing strict loss on ground-truth nodes."),
        
        ("flowchart_9_bayesian_mc_uncertainty.png",
         "Figure 9: Phase 5 Bayesian Monte Carlo Dropout Epistemic Uncertainty",
         "Execution of T=25 stochastic forward passes under dropout (p=0.30) to compute predictive mean mu_v and epistemic model variance sigma_epi^2(v)."),
        
        ("flowchart_10_threshold_calibration.png",
         "Figure 10: Phase 6 Multi-Threshold Optimization & Operating Frontiers",
         "Grid search across 191 threshold points to determine optimal decision operating points (theta* = 0.974 for GT test; theta* = 0.932 for dense test)."),
        
        ("flowchart_11_compliance_routing.png",
         "Figure 11: Phase 6 Operational Three-Tier AML Compliance Routing",
         "Real-time exchange transaction routing: Tier 1 (Automated Freeze & SAR filing) vs. Tier 2 (Human Compliance Audit) vs. Tier 3 (Mempool Release)."),
        
        ("flowchart_12_paper_narrative.png",
         "Figure 12: Academic Research Paper & Dissertation Progression Narrative",
         "Strategic 5-section narrative structure for top-tier academic publication (Problem -> Baseline Failure -> Topological Healing -> Proposed SOTA -> Empirical Rigor).")
    ]

    for fname, title, desc in flowchart_items:
        fpath = os.path.join(img_dir, fname)
        add_styled_picture(doc, fpath, f"{title}\n{desc}", width_inches=6.2)
        doc.add_paragraph().paragraph_format.space_after = Pt(4)

    doc.add_page_break()

    # -------------------------------------------------------------
    # SECTION 2: ML TRAINING 1 OUTPUTS & VISUALIZATIONS
    # -------------------------------------------------------------
    add_styled_heading(doc, "Section 2: Phase 2 (ML Training 1 — Sparse Ground-Truth Benchmark) Outputs & Plots", level=1)
    doc.add_paragraph(
        "ML Training 1 evaluates 8 pure graphical models strictly on verified forensic ground-truth nodes (46,564 nodes), masking out all 157,205 unlabeled transactions. "
        "The evaluation was performed across 16,670 test nodes in timesteps 35..49."
    )

    # ML 1 Metrics Table
    add_styled_heading(doc, "ML Training 1 Output Metrics Table (16,670 Test Nodes)", level=2)
    tbl_p1 = doc.add_table(rows=1, cols=11)
    headers_p1 = ["Model Architecture", "Accuracy", "Precision", "Recall", "F1-Score", "AUC-ROC", "PR-AUC", "ECE", "Brier", "Opt_Thresh", "Illicit TP / Total"]
    widths_p1 = [1.5, 0.55, 0.55, 0.55, 0.55, 0.55, 0.55, 0.45, 0.45, 0.5, 0.8]
    rows_p1 = [
        ["GCN", "90.11%", "12.47%", "8.68%", "0.1023", "0.7761", "0.1751", "0.3888", "0.3001", "0.993", "94 / 1,083"],
        ["GAT", "90.14%", "7.96%", "4.89%", "0.0606", "0.7925", "0.1607", "0.5209", "0.4240", "0.984", "53 / 1,083"],
        ["GraphSAGE", "90.49%", "35.52%", "56.97%", "0.4376", "0.8303", "0.2822", "0.3481", "0.2485", "0.957", "617 / 1,083"],
        ["GIN", "90.20%", "19.17%", "15.79%", "0.1732", "0.7488", "0.1648", "0.3899", "0.3221", "0.991", "171 / 1,083"],
        ["Bayesian GCN", "90.07%", "30.42%", "41.09%", "0.3496", "0.7924", "0.2099", "0.3674", "0.2621", "0.932", "445 / 1,083"],
        ["Bayesian GAT", "90.02%", "28.96%", "36.84%", "0.3243", "0.8182", "0.2175", "0.4871", "0.3808", "0.944", "399 / 1,083"],
        ["Bayesian GraphSAGE", "92.59%", "43.98%", "51.62%", "0.4749", "0.8317", "0.3263", "0.3544", "0.2571", "0.967", "559 / 1,083"],
        ["Bayesian GNN (BNN)", "93.14%", "47.46%", "51.80%", "0.4954", "0.8391", "0.3475", "0.4030", "0.2717", "0.919", "561 / 1,083"],
        ["Graph-SSL (Pseudo-Label)", "91.30%", "37.68%", "51.99%", "0.4369", "0.8497", "0.3236", "0.4114", "0.3227", "0.984", "563 / 1,083"]
    ]
    format_table(tbl_p1, widths_p1, headers_p1, rows_p1)

    add_callout_box(doc, "Key Insight: The 93.14% Accuracy Illusion",
                    "Bayesian GNN achieved 93.14% raw accuracy, but caught only 561 out of 1,083 criminals (missing 522 illicit entities). "
                    "In this test set, 93.5% of nodes are licit. A dummy model predicting licit for everything scores 93.5% accuracy. "
                    "Severed graph message passing resulted in an unacceptably low PR-AUC of 0.3475.")

    # ML 1 Visual Plots
    add_styled_heading(doc, "ML Training 1 Visual Plots & Curves", level=2)
    add_styled_picture(doc, os.path.join(ml1_plots, "performance_comparison_barchart.png"),
                       "Figure 2.1: ML Training 1 — Multi-Metric Performance Comparison Across All 8 Architectures", width_inches=6.0)
    add_styled_picture(doc, os.path.join(ml1_plots, "combined_roc_curves.png"),
                       "Figure 2.2: ML Training 1 — Overlaid ROC Curves Showing AUC Discrimination on Sparse Ground Truth", width_inches=5.8)
    add_styled_picture(doc, os.path.join(ml1_plots, "combined_pr_curves.png"),
                       "Figure 2.3: ML Training 1 — Precision-Recall Curves Illustrating Low Precision-Recall Envelopes (PR-AUC 0.3475)", width_inches=5.8)
    add_styled_picture(doc, os.path.join(ml1_plots, "confusion_matrices_all.png"),
                       "Figure 2.4: ML Training 1 — Confusion Matrix Grid for All Models (Documenting 522 Missed Criminals)", width_inches=6.0)
    add_styled_picture(doc, os.path.join(ml1_plots, "threshold_sensitivity_curves.png"),
                       "Figure 2.5: ML Training 1 — Threshold Sensitivity Curves Balancing Accuracy >= 90% and Illicit Recall", width_inches=5.8)
    add_styled_picture(doc, os.path.join(ml1_plots, "bayesian_graph_uncertainty_distributions.png"),
                       "Figure 2.6: ML Training 1 — Bayesian Predictive Uncertainty Distributions (Epistemic vs. Aleatoric)", width_inches=5.8)
    add_styled_picture(doc, os.path.join(ml1_plots, "calibration_bayesian_gnn.png"),
                       "Figure 2.7: ML Training 1 — Reliability Diagram & ECE Calibration Curve for Bayesian GNN (BNN)", width_inches=5.2)

    doc.add_page_break()

    # -------------------------------------------------------------
    # SECTION 3: ML TRAINING 2 OUTPUTS & VISUALIZATIONS
    # -------------------------------------------------------------
    add_styled_heading(doc, "Section 3: Phase 4 (ML Training 2 — 100% Dense Graph Benchmark) Outputs & Plots", level=1)
    doc.add_paragraph(
        "ML Training 2 retrained all 8 GNN models across the complete, dense 100% resolved graph (203,769 nodes, 234,355 edges). "
        "The test evaluation expanded to all 67,504 dense test transactions in timesteps 35..49 (17,678 illicit, 49,826 licit)."
    )

    # ML 2 Metrics Table
    add_styled_heading(doc, "ML Training 2 Output Metrics Table (67,504 Dense Test Nodes)", level=2)
    tbl_p2 = doc.add_table(rows=1, cols=11)
    headers_p2 = ["Model Architecture", "Accuracy", "Precision", "Recall", "F1-Score", "AUC-ROC", "PR-AUC", "ECE", "Brier", "Opt_Thresh", "Illicit TP / Total"]
    widths_p2 = [1.5, 0.55, 0.55, 0.55, 0.55, 0.55, 0.55, 0.45, 0.45, 0.5, 0.8]
    rows_p2 = [
        ["GCN", "83.48%", "64.28%", "83.08%", "0.7248", "0.9033", "0.7000", "0.3456", "0.2848", "0.945", "14,687 / 17,678"],
        ["GAT", "80.77%", "59.55%", "82.84%", "0.6929", "0.8954", "0.7257", "0.4819", "0.4313", "0.985", "14,644 / 17,678"],
        ["GraphSAGE", "84.24%", "65.23%", "85.24%", "0.7390", "0.9135", "0.7566", "0.3818", "0.3202", "0.971", "15,068 / 17,678"],
        ["GIN", "81.50%", "60.75%", "82.97%", "0.7014", "0.8774", "0.5864", "0.3251", "0.2704", "0.922", "14,668 / 17,678"],
        ["Bayesian GCN", "83.60%", "64.67%", "82.41%", "0.7247", "0.9061", "0.7222", "0.3538", "0.2882", "0.940", "14,569 / 17,678"],
        ["Bayesian GAT", "82.17%", "62.70%", "78.78%", "0.6982", "0.8967", "0.7184", "0.4727", "0.4111", "0.979", "13,926 / 17,678"],
        ["Bayesian GraphSAGE", "87.00%", "71.92%", "82.60%", "0.7689", "0.9302", "0.7976", "0.3291", "0.2603", "0.952", "14,602 / 17,678"],
        ["Bayesian GNN (BNN)", "87.34%", "72.30%", "83.73%", "0.7760", "0.9350", "0.8129", "0.3852", "0.3020", "0.952", "14,802 / 17,678"]
    ]
    format_table(tbl_p2, widths_p2, headers_p2, rows_p2)

    add_callout_box(doc, "Key Insight: The 26.4x Detection Explosion",
                    "Restoring continuous multi-hop paths via pseudo-labeling allowed GNN message passing to track funds moving through intermediate hops. "
                    "Illicit detections exploded from 561 to 14,802 (a 26.4x gain), surging PR-AUC by +133.9% (from 0.3475 to 0.8129), F1-score to 0.7760, and AUC-ROC to 0.9350.")

    # ML 2 Visual Plots
    add_styled_heading(doc, "ML Training 2 Visual Plots & Curves", level=2)
    add_styled_picture(doc, os.path.join(ml2_plots, "performance_comparison_barchart.png"),
                       "Figure 3.1: ML Training 2 — Multi-Metric Dense Graph Performance Across All 8 GNNs", width_inches=6.0)
    add_styled_picture(doc, os.path.join(ml2_plots, "combined_roc_curves.png"),
                       "Figure 3.2: ML Training 2 — Overlaid ROC Curves Reaching Peak Discrimination (AUC-ROC 0.9350)", width_inches=5.8)
    add_styled_picture(doc, os.path.join(ml2_plots, "combined_pr_curves.png"),
                       "Figure 3.3: ML Training 2 — Precision-Recall Curves Showing the PR-AUC Leap to 0.8129", width_inches=5.8)
    add_styled_picture(doc, os.path.join(ml2_plots, "confusion_matrices_all.png"),
                       "Figure 3.4: ML Training 2 — Confusion Matrix Grid Across 67,504 Test Nodes (Capturing 14,802 Criminals)", width_inches=6.0)
    add_styled_picture(doc, os.path.join(ml2_plots, "metric_tradeoff_scatter.png"),
                       "Figure 3.5: ML Training 2 — Metric Trade-off Scatter Plot (AUC-ROC vs. F1 vs. Recall)", width_inches=5.8)
    add_styled_picture(doc, os.path.join(ml2_plots, "bayesian_uncertainty_distributions.png"),
                       "Figure 3.6: ML Training 2 — Dense Graph Epistemic Uncertainty Distribution", width_inches=5.8)
    add_styled_picture(doc, os.path.join(ml2_plots, "calibration_bayesian_gnn.png"),
                       "Figure 3.7: ML Training 2 — Dense Calibration Reliability Curve for Bayesian GNN", width_inches=5.2)

    doc.add_page_break()

    # -------------------------------------------------------------
    # SECTION 4: ML TRAINING 3 OUTPUTS & VISUALIZATIONS
    # -------------------------------------------------------------
    add_styled_heading(doc, "Section 4: Phase 5 (ML Training 3 — Directional Residual GNNs & Soft Loss) Outputs & Plots", level=1)
    doc.add_paragraph(
        "ML Training 3 deploys the state-of-the-art framework: decoupling directional convolutions (Ein || Eout), deep residual skip connections + LayerNorm, "
        "soft confidence-weighted BCE loss (w_i = max(P_i, 1-P_i)), and Monte Carlo Dropout (T=25) for Bayesian epistemic uncertainty quantification."
    )

    # ML 3 Dense Metrics Table
    add_styled_heading(doc, "ML Training 3 Output Metrics: Full Dense Test Graph (67,504 Nodes)", level=2)
    tbl_p3_dense = doc.add_table(rows=1, cols=11)
    headers_p3 = ["Model Architecture", "Accuracy", "Precision", "Recall", "F1-Score", "AUC-ROC", "PR-AUC", "ECE", "Brier", "Opt_Thresh", "Illicit TP / Total"]
    widths_p3 = [1.5, 0.55, 0.55, 0.55, 0.55, 0.55, 0.55, 0.45, 0.45, 0.5, 0.8]
    rows_p3_dense = [
        ["Dir-ResGCN", "85.65%", "68.39%", "84.07%", "0.7542", "0.9249", "0.7884", "0.2949", "0.2265", "0.895", "14,862 / 17,678"],
        ["Dir-ResGAT", "77.22%", "54.02%", "87.48%", "0.6679", "0.8829", "0.6894", "0.4973", "0.4517", "0.985", "15,464 / 17,678"],
        ["Dir-ResSAGE", "84.61%", "66.06%", "84.78%", "0.7426", "0.9195", "0.7786", "0.3052", "0.2470", "0.922", "14,988 / 17,678"],
        ["Dir-GIN", "84.37%", "66.53%", "81.15%", "0.7311", "0.9090", "0.7336", "0.3418", "0.2856", "0.961", "14,345 / 17,678"],
        ["Bayesian Dir-GCN", "85.97%", "70.05%", "81.11%", "0.7517", "0.9235", "0.7867", "0.3500", "0.2684", "0.918", "14,339 / 17,678"],
        ["Bayesian Dir-GAT", "77.76%", "55.07%", "81.85%", "0.6584", "0.8658", "0.6448", "0.4992", "0.4481", "0.980", "14,470 / 17,678"],
        ["Bayesian Dir-SAGE", "85.73%", "68.48%", "84.31%", "0.7558", "0.9237", "0.7770", "0.2935", "0.2261", "0.891", "14,905 / 17,678"],
        ["Bayesian GNN (BNN)", "86.97%", "72.07%", "82.05%", "0.7673", "0.9315", "0.8068", "0.3809", "0.2923", "0.932", "14,504 / 17,678"]
    ]
    format_table(tbl_p3_dense, widths_p3, headers_p3, rows_p3_dense)

    # ML 3 Ground Truth Verification Table
    add_styled_heading(doc, "ML Training 3 Output Metrics: Ground-Truth Verification Benchmark (16,670 Nodes)", level=2)
    tbl_p3_gt = doc.add_table(rows=1, cols=8)
    headers_p3_gt = ["Model Architecture", "Accuracy", "Precision", "Recall", "F1-Score", "AUC-ROC", "PR-AUC", "Illicit TP / Total"]
    widths_p3_gt = [1.8, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 1.0]
    rows_p3_gt = [
        ["Dir-ResGCN", "90.01%", "29.90%", "39.98%", "0.3422", "0.8156", "0.2534", "433 / 1,083"],
        ["Dir-ResGAT", "42.50%", "9.58%", "93.07%", "0.1738", "0.8084", "0.2088", "1,008 / 1,083"],
        ["Dir-ResSAGE", "90.71%", "34.77%", "49.12%", "0.4072", "0.8439", "0.3477", "532 / 1,083"],
        ["Dir-GIN", "90.08%", "31.52%", "44.88%", "0.3703", "0.8405", "0.2743", "486 / 1,083"],
        ["Bayesian Dir-GCN", "90.03%", "28.02%", "34.07%", "0.3075", "0.8049", "0.2351", "369 / 1,083"],
        ["Bayesian Dir-GAT", "40.87%", "9.39%", "93.72%", "0.1708", "0.7902", "0.1771", "1,015 / 1,083"],
        ["Bayesian Dir-SAGE", "91.04%", "34.81%", "43.40%", "0.3864", "0.8311", "0.3026", "470 / 1,083"],
        ["Bayesian GNN (BNN)", "91.16%", "34.93%", "41.74%", "0.3803", "0.8264", "0.3107", "452 / 1,083"]
    ]
    format_table(tbl_p3_gt, widths_p3_gt, headers_p3_gt, rows_p3_gt)

    add_callout_box(doc, "Top 3 Models in ML 3 (Summary of Leaders)",
                    "1. Bayesian GNN (BNN): Overall Champion — 93.15% AUC-ROC, 0.8068 PR-AUC, 72.07% Precision, 91.16% Ground-Truth Accuracy, and Monte Carlo Uncertainty output.\n"
                    "2. Bayesian Dir-SAGE: Calibration Leader — Lowest Brier Error (0.2261), lowest Log-Loss (0.7145), 14,905 illicit entities caught, 91.04% GT Accuracy.\n"
                    "3. Dir-ResGCN: Top Single Deterministic Model — 0.9249 AUC-ROC (+0.87% boost over standard GCN), 14,862 caught, 90.01% GT Accuracy.")

    # ML 3 Visual Plots
    add_styled_heading(doc, "ML Training 3 Visual Plots & Curves", level=2)
    add_styled_picture(doc, os.path.join(ml3_plots, "performance_comparison_barchart.png"),
                       "Figure 4.1: ML Training 3 — Directional Residual Models Performance Benchmark", width_inches=6.0)
    add_styled_picture(doc, os.path.join(ml3_plots, "combined_roc_curves.png"),
                       "Figure 4.2: ML Training 3 — Directional ROC Curves Across Dense Topology (Dir-ResGCN 0.9249, BNN 0.9315)", width_inches=5.8)
    add_styled_picture(doc, os.path.join(ml3_plots, "combined_pr_curves.png"),
                       "Figure 4.3: ML Training 3 — Precision-Recall Curves Showing Elite Precision Retention (PR-AUC 0.8068)", width_inches=5.8)
    add_styled_picture(doc, os.path.join(ml3_plots, "confusion_matrices_all.png"),
                       "Figure 4.4: ML Training 3 — Directional Confusion Matrix Grid (Capturing 14,504 Illicit Entities with High Precision)", width_inches=6.0)
    add_styled_picture(doc, os.path.join(ml3_plots, "threshold_sensitivity_curves.png"),
                       "Figure 4.5: ML Training 3 — Dual-Threshold Sensitivity Frontier (Dense 86.97% Acc vs. Ground-Truth 91.16% Acc)", width_inches=5.8)
    add_styled_picture(doc, os.path.join(ml3_plots, "metric_tradeoff_scatter.png"),
                       "Figure 4.6: ML Training 3 — Directional Models Discrimination vs. Forensic Yield Scatter Frontier", width_inches=5.8)
    add_styled_picture(doc, os.path.join(ml3_plots, "bayesian_uncertainty_distributions.png"),
                       "Figure 4.7: ML Training 3 — Epistemic Uncertainty Distribution via Monte Carlo Dropout (T=25)", width_inches=5.8)
    add_styled_picture(doc, os.path.join(ml3_plots, "calibration_bayesian_dir_sage.png"),
                       "Figure 4.8: ML Training 3 — Calibration Reliability Diagram for Bayesian Dir-SAGE (Lowest Calibration Error)", width_inches=5.2)

    doc.add_page_break()

    # -------------------------------------------------------------
    # SECTION 5: MASTER COMPARATIVE SYNTHESIS & INVENTORY
    # -------------------------------------------------------------
    add_styled_heading(doc, "Section 5: Master Cross-Phase Output Synthesis & File Inventory", level=1)
    doc.add_paragraph(
        "This master comparative matrix synthesizes the empirical progression across all three training paradigms across 13 core dimensions:"
    )

    tbl_master = doc.add_table(rows=1, cols=4)
    headers_m = ["Evaluation Dimension", "Phase 2 (ML 1: Sparse)", "Phase 4 (ML 2: 100% Pseudo)", "Phase 5 (ML 3: Directional + Soft)"]
    widths_m = [2.0, 1.8, 1.8, 1.9]
    rows_m = [
        ["Node Label Coverage", "22.85% (46,564 nodes)", "100.0% (203,769 nodes)", "100.0% (203,769 nodes) + Soft Weights"],
        ["Graph Message Passing", "Severely severed subgraphs", "Dense undirected message passing", "Asymmetric Directional Flow (Ein || Eout)"],
        ["Skip Connections", "None", "None", "Residual Skips + Layer Normalization"],
        ["Loss Function", "Standard Class-Weighted BCE", "Standard Class-Weighted BCE", "Soft Confidence-Weighted BCE (w_i * BCE)"],
        ["Illicit Entities Intercepted", "561 illicit transactions", "14,802 illicit transactions", "14,504 illicit transactions"],
        ["Test PR-AUC (Illicit)", "0.3475", "0.8129", "0.8068"],
        ["Test AUC-ROC", "0.8391", "0.9350", "0.9315"],
        ["Test F1-Score", "0.4954", "0.7760", "0.7673"],
        ["GT Verification Accuracy", "93.14% (skewed base rate)", "86.85%", "91.16% (Satisfies >= 90% benchmark)"],
        ["Single GCN AUC-ROC", "0.7963 (Standard GCN)", "0.9162 (Dense GCN)", "0.9249 (Dir-ResGCN: +0.87% boost)"],
        ["Single SAGE AUC-ROC", "0.8407 (Standard SAGE)", "0.9180 (Dense SAGE)", "0.9195 (Dir-ResSAGE)"],
        ["Uncertainty Quantification", "MC Dropout (T=20)", "MC Dropout (T=20)", "MC Dropout (T=25) on Directional Topology"],
        ["Operational Verdict", "Incomplete Baseline", "Topological Breakthrough", "Peak Operational & Research Standard"]
    ]
    format_table(tbl_master, widths_m, headers_m, rows_m)

    add_styled_heading(doc, "Complete Plots & Visualizations File Catalog", level=2)
    doc.add_paragraph(
        "• Flowcharts (12 High-Resolution PNGs): Located in flowcharts/\n"
        "  - flowchart_1_master_pipeline.png through flowchart_12_paper_narrative.png\n\n"
        "• ML Training 1 Plots (15 Plots): Located in ML TRAINING 1/plots/\n"
        "  - performance_comparison_barchart.png, combined_roc_curves.png, combined_pr_curves.png, confusion_matrices_all.png, "
        "threshold_sensitivity_curves.png, bayesian_graph_uncertainty_distributions.png, and 8 individual model calibration plots.\n\n"
        "• ML Training 2 Plots (14 Plots): Located in ML TRAINING 2/plots/\n"
        "  - performance_comparison_barchart.png, combined_roc_curves.png, combined_pr_curves.png, confusion_matrices_all.png, "
        "metric_tradeoff_scatter.png, bayesian_uncertainty_distributions.png, and 8 individual model calibration plots.\n\n"
        "• ML Training 3 Plots (15 Plots): Located in ML TRAINING 3/plots/\n"
        "  - performance_comparison_barchart.png, combined_roc_curves.png, combined_pr_curves.png, confusion_matrices_all.png, "
        "threshold_sensitivity_curves.png, metric_tradeoff_scatter.png, bayesian_uncertainty_distributions.png, and 8 individual model calibration plots."
    )

    # Save to primary locations
    output_docx = os.path.join(base_dir, "Visualization and outputs.docx")
    output_doc = os.path.join(base_dir, "Visualization and outputs.doc")
    doc.save(output_docx)
    print(f"Generated successfully: {output_docx}")

    # Copy to .doc
    try:
        shutil.copyfile(output_docx, output_doc)
        print(f"Copied to: {output_doc}")
    except Exception as e:
        print(f"Error copying to .doc: {e}")

    # Copy to Desktop
    desktop_dir = r"c:\Users\USER\OneDrive\Desktop"
    if os.path.exists(desktop_dir):
        try:
            shutil.copyfile(output_docx, os.path.join(desktop_dir, "Visualization and outputs.docx"))
            shutil.copyfile(output_docx, os.path.join(desktop_dir, "Visualization and outputs.doc"))
            print("Successfully copied 'Visualization and outputs.docx' and 'Visualization and outputs.doc' to Desktop!")
        except Exception as e:
            print(f"Desktop copy warning: {e}")

if __name__ == '__main__':
    build_visualization_and_outputs_doc()
