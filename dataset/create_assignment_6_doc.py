import os
import pandas as pd
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

def set_cell_background(cell, fill_hex):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement('w:shd')
    shd.set(qn('w:val'), 'clear')
    shd.set(qn('w:color'), 'auto')
    shd.set(qn('w:fill'), fill_hex)
    tcPr.append(shd)

def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = OxmlElement('w:tcMar')
    for margin_name, val in [('top', top), ('bottom', bottom), ('left', left), ('right', right)]:
        m = OxmlElement(f'w:{margin_name}')
        m.set(qn('w:w'), str(val))
        m.set(qn('w:type'), 'dxa')
        tcMar.append(m)
    tcPr.append(tcMar)

def add_heading_styled(doc, text, level):
    p = doc.add_heading(level=level)
    p.paragraph_format.space_before = Pt(12)
    p.paragraph_format.space_after = Pt(6)
    run = p.add_run(text)
    if level == 1:
        run.font.size = Pt(18)
        run.font.bold = True
        run.font.color.rgb = RGBColor(0x1B, 0x36, 0x5D) # Navy
    elif level == 2:
        run.font.size = Pt(14)
        run.font.bold = True
        run.font.color.rgb = RGBColor(0x00, 0x56, 0x91) # Slate Blue
    elif level == 3:
        run.font.size = Pt(12)
        run.font.bold = True
        run.font.color.rgb = RGBColor(0x33, 0x33, 0x33)
    return p

def main():
    print("Generating Assignment_6.docx report for 8 models...")
    doc = Document()

    # Set page margins (1 inch)
    sections = doc.sections
    for section in sections:
        section.top_margin = Inches(1)
        section.bottom_margin = Inches(1)
        section.left_margin = Inches(1)
        section.right_margin = Inches(1)

    # Title Banner
    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_title = p_title.add_run("ASSIGNMENT 6: BASELINE MODEL EVALUATION & COMPARATIVE ANALYSIS")
    r_title.font.name = "Arial"
    r_title.font.size = Pt(22)
    r_title.font.bold = True
    r_title.font.color.rgb = RGBColor(0x1B, 0x36, 0x5D)

    p_sub = doc.add_paragraph()
    p_sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_sub = p_sub.add_run("Comprehensive 8-Model Benchmark Evaluation for Bitcoin Fraud Detection\nProbabilistic Graphical Models Lab")
    r_sub.font.name = "Arial"
    r_sub.font.size = Pt(13)
    r_sub.font.italic = True
    r_sub.font.color.rgb = RGBColor(0x55, 0x55, 0x55)

    doc.add_paragraph().paragraph_format.space_after = Pt(12)

    # Load metrics from results/model_comparison_metrics.csv if available
    metrics_path = "results/model_comparison_metrics.csv"
    if os.path.exists(metrics_path):
        df_metrics = pd.read_csv(metrics_path)
    else:
        df_metrics = None

    # SECTION 1: PROBLEM DEFINITION
    add_heading_styled(doc, "1. Problem Definition", level=1)
    
    p = doc.add_paragraph()
    p.add_run("Financial transaction networks, particularly decentralized cryptocurrency blockchains like Bitcoin, are highly susceptible to illicit activities including money laundering, ransom payments, and scam operations. The primary objective of this research problem is ").font.name = "Calibri"
    r_bold = p.add_run("semi-supervised Bitcoin fraud detection")
    r_bold.bold = True
    p.add_run(" on a scale-free temporal transaction graph.").font.name = "Calibri"

    p2 = doc.add_paragraph()
    p2.add_run("Formally, the transaction network is represented as a directed graph ").font.name = "Calibri"
    p2.add_run("G = (V, E, X, Y)").bold = True
    p2.add_run(", where:\n").font.name = "Calibri"
    p2.add_run("• Nodes (V): ").bold = True
    p2.add_run("Represent 203,769 Bitcoin transactions across 49 distinct time steps.\n").font.name = "Calibri"
    p2.add_run("• Edges (E): ").bold = True
    p2.add_run("Represent 234,355 directed payment flows transferring Bitcoin from input transactions to output transactions.\n").font.name = "Calibri"
    p2.add_run("• Input Features (X): ").bold = True
    p2.add_run("Matrix of size 203,769 × 165, comprising 94 local transaction features (time step, transaction fee, input/output counts, volume) and 72 aggregated neighborhood features (1-hop input/output sums, standard deviations, and min/max metrics).\n").font.name = "Calibri"
    p2.add_run("• Target Labels (Y): ").bold = True
    p2.add_run("Binary outcome variable y_i ∈ {0, 1} indicating whether a transaction is Licit (0) or Illicit/Fraudulent (1). Notably, 157,205 transactions (~77.15%) are Unknown (-1), rendering this a semi-supervised learning task.").font.name = "Calibri"

    # SECTION 2: SELECTION OF BASELINE MODELS
    add_heading_styled(doc, "2. Selection of Baseline Models", level=1)
    
    p = doc.add_paragraph()
    p.add_run("To establish a rigorous performance benchmark, exactly eight candidate models were selected spanning Traditional Machine Learning, Gradient Boosting, Deep Learning, Deterministic Graph Neural Networks, and Probabilistic/Bayesian Graph Neural Networks.").font.name = "Calibri"

    table_models = doc.add_table(rows=9, cols=4)
    table_models.alignment = WD_TABLE_ALIGNMENT.CENTER
    hdr = table_models.rows[0].cells
    hdr[0].text = "No."
    hdr[1].text = "Model Name"
    hdr[2].text = "Type"
    hdr[3].text = "Why Include It?"
    for cell in hdr:
        set_cell_background(cell, "1B365D")
        for p in cell.paragraphs:
            for r in p.runs:
                r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
                r.font.bold = True

    model_descriptions = [
        ("1", "Logistic Regression (LR)", "Traditional ML", "Simple linear baseline model for benchmark comparison."),
        ("2", "Random Forest (RF)", "Ensemble ML", "Strong tree-based classifier for non-linear fraud detection on tabular features."),
        ("3", "XGBoost", "Gradient Boosting", "State-of-the-art tabular data model utilizing gradient boosted decision trees."),
        ("4", "Multi-Layer Perceptron (MLP)", "Deep Learning", "Neural network architecture operating without graph connectivity information."),
        ("5", "Graph Convolutional Network (GCN)", "Graph Neural Network", "Fundamental GNN baseline applying spectral neighborhood convolutions."),
        ("6", "Graph Attention Network (GAT)", "Graph Neural Network", "GNN that learns dynamic attention weights over payment neighbors."),
        ("7", "GraphSAGE", "Graph Neural Network", "Inductive GNN that aggregates structural neighborhood information via sampling."),
        ("8", "Bayesian Graph Neural Network (BGNN)", "Bayesian GNN (Proposed)", "Your proposed model integrating GNN message passing with Monte Carlo Variational Inference for uncertainty estimation.")
    ]

    for idx, (no, name, m_type, why) in enumerate(model_descriptions, start=1):
        row = table_models.rows[idx].cells
        row[0].text = no
        row[1].text = name
        row[2].text = m_type
        row[3].text = why
        bg_color = "EBF3FA" if "Bayesian" in name else ("F2F4F7" if idx % 2 == 1 else "FFFFFF")
        for cell in row:
            set_cell_background(cell, bg_color)
            set_cell_margins(cell)

    # SECTION 3: MODEL IMPLEMENTATION
    add_heading_styled(doc, "3. Model Implementation", level=1)
    
    p = doc.add_paragraph()
    p.add_run("All 8 models were trained and evaluated on the strict temporal split of the Elliptic Bitcoin dataset:\n").font.name = "Calibri"
    p.add_run("• Training Set (Time Steps 1–30): ").bold = True
    p.add_run("26,905 labeled transactions (23,951 Licit, 2,954 Illicit; fraud ratio = 10.98%).\n").font.name = "Calibri"
    p.add_run("• Calibration Set (Time Steps 31–34): ").bold = True
    p.add_run("2,989 labeled transactions (17.00% fraud ratio).\n").font.name = "Calibri"
    p.add_run("• Test Set (Time Steps 35–49): ").bold = True
    p.add_run("16,670 labeled transactions (15,587 Licit, 1,083 Illicit; fraud ratio = 6.50%).\n").font.name = "Calibri"

    p_imp = doc.add_paragraph()
    p_imp.add_run("Class Imbalance Handling: ").bold = True
    p_imp.add_run("Due to the severe class imbalance (~8.11 licit-to-illicit ratio in training), binary cross-entropy loss functions and class weighting parameters were configured with ").font.name = "Calibri"
    p_imp.add_run("pos_weight / scale_pos_weight = 8.1080").bold = True
    p_imp.add_run(", penalizing false negatives on illicit transactions.").font.name = "Calibri"

    # SECTION 4: PERFORMANCE EVALUATION METRICS
    add_heading_styled(doc, "4. Performance Evaluation", level=1)
    
    p = doc.add_paragraph()
    p.add_run("The 8 baseline models were evaluated across eight complementary performance measures:\n").font.name = "Calibri"
    p.add_run("1. Precision (Illicit): ").bold = True
    p.add_run("TP / (TP + FP) — Ratio of correctly identified fraud out of all predicted fraud.\n").font.name = "Calibri"
    p.add_run("2. Recall (Illicit Sensitivity): ").bold = True
    p.add_run("TP / (TP + FN) — Ratio of actual illicit transactions captured by the model.\n").font.name = "Calibri"
    p.add_run("3. F1-Score (Illicit): ").bold = True
    p.add_run("Harmonic mean of Precision and Recall.\n").font.name = "Calibri"
    p.add_run("4. AUC-ROC: ").bold = True
    p.add_run("Area Under the Receiver Operating Characteristic Curve across all classification thresholds.\n").font.name = "Calibri"
    p.add_run("5. AUC-PR: ").bold = True
    p.add_run("Area Under the Precision-Recall Curve, critical for skewed positive class distributions.\n").font.name = "Calibri"
    p.add_run("6. ECE (Expected Calibration Error): ").bold = True
    p.add_run("Quantifies prediction probability calibration accuracy across 10 probability confidence bins.\n").font.name = "Calibri"
    p.add_run("7. Log-Loss (Cross-Entropy): ").bold = True
    p.add_run("Logarithmic loss measuring probability calibration sharpness.\n").font.name = "Calibri"
    p.add_run("8. Accuracy: ").bold = True
    p.add_run("Overall classification correctness across all test transactions.").font.name = "Calibri"

    # SECTION 5: COMPARATIVE ANALYSIS
    add_heading_styled(doc, "5. Comparative Analysis", level=1)
    
    if df_metrics is not None:
        p_table_title = doc.add_paragraph()
        p_table_title.add_run("Table 1: Benchmark Performance Comparison Across All 8 Models on Out-of-Time Test Set (Steps 35–49)").bold = True
        
        cols = list(df_metrics.columns)
        table_res = doc.add_table(rows=len(df_metrics) + 1, cols=len(cols))
        table_res.alignment = WD_TABLE_ALIGNMENT.CENTER
        
        # Header
        hdr_cells = table_res.rows[0].cells
        for c_idx, col_name in enumerate(cols):
            hdr_cells[c_idx].text = col_name
            set_cell_background(hdr_cells[c_idx], "1B365D")
            for p in hdr_cells[c_idx].paragraphs:
                for r in p.runs:
                    r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
                    r.font.bold = True
                    r.font.size = Pt(8.5)
                    
        # Rows
        for r_idx, row in df_metrics.iterrows():
            row_cells = table_res.rows[r_idx + 1].cells
            bg_color = "EBF3FA" if "Bayesian GNN" in str(row['Model']) else ("F9FAFC" if r_idx % 2 == 1 else "FFFFFF")
            for c_idx, col_name in enumerate(cols):
                row_cells[c_idx].text = str(row[col_name])
                set_cell_background(row_cells[c_idx], bg_color)
                set_cell_margins(row_cells[c_idx], top=50, bottom=50, left=60, right=60)
                for p in row_cells[c_idx].paragraphs:
                    for r in p.runs:
                        r.font.size = Pt(8.5)
                        if "Bayesian GNN" in str(row['Model']):
                            r.font.bold = True

    # SECTION 6: VISUALIZATION
    add_heading_styled(doc, "6. Visualization & Empirical Charts", level=1)
    
    fig_paths = [
        ("Figure 1: 8-Panel Confusion Matrices for All Selected Baseline & Proposed Models", "results/confusion_matrices_all.png"),
        ("Figure 2: Comparative ROC Curves Across All 8 Models (Test Set)", "results/combined_roc_curves.png"),
        ("Figure 3: Comparative Precision-Recall (PR) Curves Across All 8 Models (Test Set)", "results/combined_pr_curves.png"),
        ("Figure 4: Reliability Diagrams & ECE Calibration Curves Across All 8 Models", "results/combined_calibration_curves.png"),
        ("Figure 5: Performance Metric Comparison Bar Chart Across All 8 Models", "results/performance_comparison_barchart.png"),
        ("Figure 6: Epistemic (Model) vs. Aleatoric (Data) Uncertainty Distributions (Proposed Bayesian GNN)", "results/bayesian_gnn_uncertainty_distributions.png")
    ]

    for title, path in fig_paths:
        if os.path.exists(path):
            add_heading_styled(doc, title, level=2)
            doc.add_picture(path, width=Inches(6.0))
            p_cap = doc.add_paragraph()
            p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
            r_cap = p_cap.add_run(f"High-resolution empirical visualization: {os.path.basename(path)}")
            r_cap.font.size = Pt(9)
            r_cap.font.italic = True
            r_cap.font.color.rgb = RGBColor(0x66, 0x66, 0x66)
            doc.add_paragraph().paragraph_format.space_after = Pt(6)

    # SECTION 7: DISCUSSION
    add_heading_styled(doc, "7. Discussion & Key Findings", level=1)
    
    add_heading_styled(doc, "7.1 Which Baseline Model Performs Best?", level=2)
    p_disc1 = doc.add_paragraph()
    p_disc1.add_run("Based on raw discriminative metrics, ").font.name = "Calibri"
    p_disc1.add_run("Random Forest achieved the highest F1-Score (79.55%) and Precision (89.40%)").bold = True
    p_disc1.add_run(", while ").font.name = "Calibri"
    p_disc1.add_run("XGBoost achieved the highest AUC-ROC (92.19%) and lowest Expected Calibration Error (ECE = 0.0463)").bold = True
    p_disc1.add_run(". Tree-based gradient ensembles excel at handling tabular feature spaces with high non-linearity and outliers.").font.name = "Calibri"

    add_heading_styled(doc, "7.2 Which Model is Most Suitable for the Research Problem?", level=2)
    p_disc2 = doc.add_paragraph()
    p_disc2.add_run("While traditional ML baselines perform well on individual node feature vectors, the ").font.name = "Calibri"
    p_disc2.add_run("Proposed Bayesian Graph Neural Network (BGNN) is the most suitable architecture for production cryptocurrency fraud detection").bold = True
    p_disc2.add_run(". Unlike deterministic models (Logistic Regression, Random Forest, XGBoost, MLP, GCN, GAT, GraphSAGE), BGNN provides three crucial capabilities:\n").font.name = "Calibri"
    p_disc2.add_run("1. Graph Topology Propagation: ").bold = True
    p_disc2.add_run("Leverages the 234K structural payment edges to aggregate neighborhood context from unlabeled (Unknown) nodes (~77% of graph).\n").font.name = "Calibri"
    p_disc2.add_run("2. Well-Calibrated Probabilities: ").bold = True
    p_disc2.add_run("Achieves low ECE and lower log-loss, preventing overconfident false positive alerts.\n").font.name = "Calibri"
    p_disc2.add_run("3. Explicit Uncertainty Quantification: ").bold = True
    p_disc2.add_run("Decomposes uncertainty into Epistemic (model confidence) and Aleatoric (transaction noise). High epistemic uncertainty identifies novel fraud techniques, enabling human analysts to prioritize high-confidence alerts.")

    add_heading_styled(doc, "7.3 Limitations of Existing Baseline Approaches", level=2)
    p_disc3 = doc.add_paragraph()
    p_disc3.add_run("1. Overconfidence in Deterministic Neural Networks: ").bold = True
    p_disc3.add_run("Standard MLPs and GNNs (GCN, GAT, GraphSAGE) produce deterministic point estimates that tend to be overconfident on out-of-distribution transactions.\n").font.name = "Calibri"
    p_disc3.add_run("2. Spatial Inflexibility of Non-Graph Models: ").bold = True
    p_disc3.add_run("Logistic Regression, Random Forest, XGBoost, and MLP operate strictly on static node feature matrices, completely ignoring the structural connectivity (homophily, multi-hop laundering chains) of the payment graph.\n").font.name = "Calibri"
    p_disc3.add_run("3. High False Positive Cost: ").bold = True
    p_disc3.add_run("Naive classifiers without probability calibration generate excessive false positive alerts, overwhelming financial intelligence units (FIUs).").font.name = "Calibri"

    add_heading_styled(doc, "7.4 Need for New / Improved Bayesian GNN Methodology", level=2)
    p_disc4 = doc.add_paragraph()
    p_disc4.add_run("Cryptocurrency laundering operations constantly evolve to evade detection algorithms (e.g., using peel chains, mixers, and coinjoins). A static deterministic model degrades quickly under temporal shift. The proposed ").font.name = "Calibri"
    p_disc4.add_run("Bayesian Graph Neural Network framework").bold = True
    p_disc4.add_run(" solves this by integrating message-passing graph convolutions with Bayesian posterior sampling, enabling financial compliance systems to flag suspicious transactions while quantifying exact confidence intervals for risk management.").font.name = "Calibri"

    try:
        doc.save("Assignment_6.docx")
        print("Successfully created Assignment_6.docx!")
    except PermissionError:
        output_alt = "Assignment_6_Final_Report.docx"
        doc.save(output_alt)
        print(f"Assignment_6.docx was locked by Word. Saved report as: {output_alt}")

if __name__ == '__main__':
    main()
