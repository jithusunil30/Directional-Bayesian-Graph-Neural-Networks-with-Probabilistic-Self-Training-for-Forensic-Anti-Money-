import os
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
    doc = Document()
    
    # Page Margins (1 inch)
    for s in doc.sections:
        s.top_margin = Inches(1)
        s.bottom_margin = Inches(1)
        s.left_margin = Inches(1)
        s.right_margin = Inches(1)

    # Colors - Clean Simple Black & Gray
    black = RGBColor(0, 0, 0)
    dark_gray = RGBColor(50, 50, 50)

    # Document Title
    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run_title = p_title.add_run("LAB 4: EXPLORATORY DATA ANALYSIS (EDA) REPORT\n")
    run_title.font.name = 'Arial'
    run_title.font.size = Pt(18)
    run_title.font.bold = True
    run_title.font.color.rgb = black

    run_sub = p_title.add_run("Comprehensive Structural and Topological Analysis of the Elliptic Bitcoin Dataset")
    run_sub.font.name = 'Arial'
    run_sub.font.size = Pt(12)
    run_sub.font.italic = True
    run_sub.font.color.rgb = dark_gray

    doc.add_paragraph()

    # Section 1
    h1 = doc.add_heading("1. Examine the Dataset (Dimensions, Attributes & Statistics)", level=1)
    h1.runs[0].font.color.rgb = black
    h1.runs[0].font.name = 'Arial'
    h1.runs[0].font.bold = True

    p1 = doc.add_paragraph()
    p1.add_run("Number of Samples and Features\n").bold = True
    p1.add_run("The preprocessed Elliptic Bitcoin dataset contains a total of ")
    p1.add_run("203,769 samples (nodes)").bold = True
    p1.add_run(" and ")
    p1.add_run("165 numerical attributes (features)").bold = True
    p1.add_run(". Additionally, there are ")
    p1.add_run("234,355 directed payment edges").bold = True
    p1.add_run(" defining the payment flow structure of the transaction graph.")

    # Attributes Table (Simple Black & White / Light Gray Header)
    doc.add_paragraph().add_run("Data Types of Attributes").bold = True
    t_attr = doc.add_table(rows=1, cols=3)
    t_attr.alignment = WD_TABLE_ALIGNMENT.CENTER
    hdr = t_attr.rows[0].cells
    hdr[0].text = "Attribute Name"
    hdr[1].text = "Data Type"
    hdr[2].text = "Role & Description"
    for c in hdr:
        set_cell_background(c, "E6E6E6")
        c.paragraphs[0].runs[0].font.color.rgb = black
        c.paragraphs[0].runs[0].font.bold = True

    attr_data = [
        ("node_idx", "int64 / int32", "Contiguous index of the transaction in the PyG graph [0, N-1]"),
        ("time_step", "int64 (1 to 49)", "Chronological snapshot index showing when the transaction occurred"),
        ("label", "int64 (0, 1, -1)", "Class target: 0 = Licit (legit), 1 = Illicit (fraud), -1 = Unknown (unlabeled)"),
        ("train_mask / val_mask / test_mask", "boolean / uint8", "Train, calibration, and test split designations"),
        ("feat_0 to feat_93", "float32 (scaled)", "94 local features capturing individual transaction dynamics"),
        ("feat_94 to feat_164", "float32 (scaled)", "71 aggregate features capturing immediate neighbor metrics")
    ]
    for r_idx, (a_name, d_type, desc) in enumerate(attr_data):
        row_cells = t_attr.add_row().cells
        row_cells[0].text = a_name
        row_cells[1].text = d_type
        row_cells[2].text = desc

    doc.add_paragraph()

    # Summary Statistics Table (Simple Header)
    doc.add_paragraph().add_run("Summary Statistics of Selected Discriminative Features").bold = True
    t_stats = doc.add_table(rows=1, cols=7)
    t_stats.alignment = WD_TABLE_ALIGNMENT.CENTER
    hdr2 = t_stats.rows[0].cells
    for i, title in enumerate(["Feature", "Class", "Mean", "Median", "Std Dev", "Min", "Max"]):
        hdr2[i].text = title
        set_cell_background(hdr2[i], "E6E6E6")
        hdr2[i].paragraphs[0].runs[0].font.color.rgb = black
        hdr2[i].paragraphs[0].runs[0].font.bold = True

    stats_rows = [
        ("feat_52", "Licit", "0.873", "0.413", "1.336", "-0.497", "5.802"),
        ("feat_52", "Illicit", "-0.293", "-0.497", "0.479", "-0.497", "2.478"),
        ("feat_54", "Licit", "0.732", "0.269", "1.339", "-0.476", "5.940"),
        ("feat_54", "Illicit", "-0.276", "-0.476", "0.498", "-0.476", "2.981"),
        ("feat_88", "Licit", "0.360", "-0.087", "1.237", "-0.816", "4.768"),
        ("feat_88", "Illicit", "-0.574", "-0.815", "0.498", "-0.816", "3.421"),
        ("feat_89", "Licit", "0.340", "-0.202", "1.273", "-0.736", "5.801"),
        ("feat_89", "Illicit", "-0.589", "-0.736", "0.389", "-0.736", "4.247"),
        ("feat_108", "Licit", "2.006", "-0.006", "7.489", "-0.220", "121.348"),
        ("feat_108", "Illicit", "0.163", "-0.051", "1.024", "-0.216", "31.906"),
        ("feat_105", "Licit", "1.911", "-0.050", "8.527", "-0.189", "142.520"),
        ("feat_105", "Illicit", "0.208", "-0.011", "0.986", "-0.189", "37.524"),
        ("feat_106", "Licit", "1.691", "-0.052", "5.332", "-0.202", "74.888"),
        ("feat_106", "Illicit", "0.072", "-0.102", "1.142", "-0.200", "43.783"),
        ("feat_141", "Licit", "0.132", "-0.104", "1.245", "-0.185", "33.985"),
        ("feat_141", "Illicit", "1.095", "0.232", "2.754", "-0.185", "40.823")
    ]
    for r_idx, vals in enumerate(stats_rows):
        row_cells = t_stats.add_row().cells
        for col_idx, v in enumerate(vals):
            row_cells[col_idx].text = v

    doc.add_paragraph()

    # Class Imbalance text
    p_imb = doc.add_paragraph()
    p_imb.add_run("Class Distribution and Class Imbalance\n").bold = True
    p_imb.add_run("The dataset experiences severe class imbalance. Licit nodes account for ")
    p_imb.add_run("42,019 (20.62%)").bold = True
    p_imb.add_run(" of all nodes, whereas Illicit transactions account for only ")
    p_imb.add_run("4,545 (2.23%)").bold = True
    p_imb.add_run(". The remaining ")
    p_imb.add_run("157,205 (77.15%)").bold = True
    p_imb.add_run(" nodes are unlabeled. Focusing strictly on labeled transactions (46,564 total), ")
    p_imb.add_run("90.24%").bold = True
    p_imb.add_run(" are legitimate and ")
    p_imb.add_run("9.76%").bold = True
    p_imb.add_run(" are fraudulent, showing an imbalance ratio of ")
    p_imb.add_run("8.11 to 1").bold = True
    p_imb.add_run(" in training.")

    # Section 2
    h2 = doc.add_heading("2. Perform Exploratory Data Analysis (EDA)", level=1)
    h2.runs[0].font.color.rgb = black
    h2.runs[0].font.name = 'Arial'
    h2.runs[0].font.bold = True

    doc.add_paragraph().add_run("Identifying Feature Distributions").bold = True
    doc.add_paragraph("By comparing feature means, we observe feature anisotropy. Licit nodes occupy positive value spaces on discriminative local features (e.g., feat_52 mean is +0.873, feat_54 mean is +0.732), while illicit nodes occupy negative spaces (feat_52 mean is -0.293, feat_54 mean is -0.276). In contrast, neighbor features show a reverse pattern (e.g., neighbor aggregated feat_141 has a mean of +0.132 for licit but peaks at +1.095 for illicit). The density distributions exhibit significant skewness and heavy tails, common in monetary transactions.")

    doc.add_paragraph().add_run("Analyzing Correlations Between Numerical Features").bold = True
    doc.add_paragraph("We computed correlation coefficients between discriminative features and the target label. Features like feat_52 (-0.38), feat_54 (-0.32), and neighbor feature feat_108 (-0.39) show notable negative correlation with the binary target (lower values align with illicit class 1). Conversely, neighbor aggregate feature feat_141 exhibits a strong positive correlation (+0.36) with illicit transactions, indicating that connection to active transaction groups is a key indicator of fraud. Multicollinearity is also visible among neighboring aggregated features (e.g., feat_105 and feat_106 are correlated at +0.97).")

    doc.add_paragraph().add_run("Detecting Outliers").bold = True
    doc.add_paragraph("Using the Interquartile Range (IQR) method, outliers are heavily concentrated in the neighbor features. Licit nodes have high outlier rates in local metrics (e.g., 19.97% on feat_106). Conversely, illicit transactions have extremely high outlier percentages on neighbor feature feat_141 (45.72%), displaying severe structural anomalies compared to the general population.")

    doc.add_paragraph().add_run("Class Imbalance Check").bold = True
    doc.add_paragraph("Class imbalance is consistent across chronological splits. The fraud ratio is 10.98% in training (steps 1-30), peaks at 17.00% in calibration (steps 31-34), and declines to 6.50% in testing (steps 35-49).")

    # Section 3: Visualizations & Embedded Pictures
    h3 = doc.add_heading("3. Visualizations & Interpretations", level=1)
    h3.runs[0].font.color.rgb = black
    h3.runs[0].font.name = 'Arial'
    h3.runs[0].font.bold = True

    plots_dir = r"C:\Users\niran\.gemini\antigravity\brain\3ae03c5d-67d0-40e6-98ea-bb2260314ab1\plots"

    figs_meta = [
        ("fig1_class_distribution.png", "Figure 1: Overall Node Class Distribution and Labeled Node Split-wise Distributions",
         "The pie chart confirms that 77.15% of transactions are unlabeled, leaving only 22.85% labeled. The stacked bar chart shows the training set contains 26,905 labeled transactions, calibration contains 2,989, and testing contains 16,670. The fraud proportions represent a tiny minority in all partitions."),
        
        ("fig2_degree_distribution.png", "Figure 2: Log-Scaled Node Degree Density Distributions and Class Boxplots",
         "The density plot highlights a classic scale-free network profile: most transactions have a degree of 1 or 2, while a small number of hub transactions connect to hundreds of other transactions. Licit nodes show a shift towards higher degrees on the log-scale, whereas illicit nodes have a sharp peak at degree 1."),
        
        ("fig3_edge_transition_matrix.png", "Figure 3: Heatmap Matrix of Edge Transition Connection Probabilities",
         "This connection matrix reveals strong community clustering. Outgoing links from illicit transactions connect to other illicit nodes 29.61% of the time, flow into unlabeled bridge nodes 43.25% of the time, and connect to legitimate nodes only 27.14% of the time."),
        
        ("fig4_temporal_split_trends.png", "Figure 4: Total Nodes and Labeled Fraud Ratio across Time Steps 1-49",
         "The node distribution over time is highly volatile. The labeled fraud ratio fluctuates significantly, peaking during calibration (steps 31-34 at 17.00%) and dipping in testing (steps 35-49 at 6.50%), validating the need for chronological temporal splitting."),
        
        ("fig5_top8_feature_densities.png", "Figure 5: Class-wise Density plots for top 8 discriminative features",
         "The density plots display distinct, non-overlapping shapes between licit (green) and illicit (red) classes. Local features (feat_52, feat_54, feat_88, feat_89) have values concentrated around 0.5 to 1.0 for licit, while illicit values concentrate below 0."),
        
        ("fig6_feature_outlier_boxplots.png", "Figure 6: Boxplots demonstrating feature outlier locations by Class",
         "The boxplots demonstrate heavy outlier profiles. Licit nodes exhibit dense outliers on local features, while illicit nodes display extensive, extreme outlier spikes on neighborhood aggregation metrics (e.g., feat_141 at 45.72% IQR outlier rate)."),
        
        ("fig7_correlation_matrix.png", "Figure 7: Correlation Heatmap Matrix of Selected features and binary Label",
         "The correlation matrix identifies negative correlations with the label for feat_52, feat_54, feat_88, feat_89, feat_108, feat_105, and feat_106 (coefficients from -0.30 to -0.39), while feat_141 correlates positively (+0.36)."),
        
        ("fig8_pairwise_scatter_matrix.png", "Figure 8: Pairwise Scatter Matrix showing feature interactions split by Class",
         "The pairplot illustrates clear class segregation: illicit transactions occupy a tight cluster in the feature space, while licit transactions are more widely dispersed, indicating distinct transactional behaviors.")
    ]

    for fname, cap, interp in figs_meta:
        fpath = os.path.join(plots_dir, fname)
        if os.path.exists(fpath):
            p_img = doc.add_paragraph()
            p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p_img.add_run().add_picture(fpath, width=Inches(6.0))
            
            p_cap = doc.add_paragraph()
            p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
            r_c = p_cap.add_run(f"{cap}\n")
            r_c.font.bold = True
            r_c.font.size = Pt(10)
            r_c.font.color.rgb = black
            
            p_int = doc.add_paragraph()
            p_int.add_run("Interpretation: ").bold = True
            p_int.add_run(interp)
            doc.add_paragraph()

    # Section 4
    h4 = doc.add_heading("4. Influence of EDA Findings on Model Selection", level=1)
    h4.runs[0].font.color.rgb = black
    h4.runs[0].font.name = 'Arial'
    h4.runs[0].font.bold = True

    points = [
        ("Handling Class Imbalance: ", "The training data features a severe imbalance ratio (10.98% fraud, which translates to a licit-to-illicit ratio of 8.11). Standard binary loss functions will bias predictions to the majority class. We must use the calculated training imbalance ratio (8.11) as a class weight ('pos_weight') in loss computation during optimization."),
        ("Graph Topology Propagation: ", "Over 77% of nodes are unlabeled. The connectivity matrix documents high homophily—illicit transactions connect to other illicit or unknown nodes 72.86% of the time. We recommend using Graph Neural Networks (such as GCN, GAT, or GraphSAGE) to aggregate neighbor features across structural edges."),
        ("Robustness to Heavy Tails & Outliers: ", "Outlier rates range from 4% up to 45.72% (especially for neighbor aggregates on illicit nodes). Standard linear models can be highly unstable under such distributions. Downstream classifiers must either use robust architectures (e.g. tree-based XGBoost/LightGBM models) or deep neural networks with Batch Normalization."),
        ("Chronological Snapshot Validation: ", "Class distributions shift over time (e.g. fraud ratio drops from 10.98% to 6.50% in the test split). Models must be evaluated strictly according to the temporal split (Train: Steps 1-30, Calibration: Steps 31-34, Test: Steps 35-49) to eliminate data leakage."),
        ("Bayesian Calibration Set: ", "Steps 31-34 (2,989 nodes) are held out exclusively as a calibration set. This allows us to calculate Expected Calibration Error (ECE) and plot reliability curves, ensuring confidence intervals output by the Bayesian neural network are accurate.")
    ]

    for title_prefix, desc in points:
        p_pt = doc.add_paragraph(style='List Bullet')
        p_pt.add_run(title_prefix).bold = True
        p_pt.add_run(desc)

    doc.add_paragraph()
    p_footer = doc.add_paragraph()
    p_footer.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    r_foot = p_footer.add_run("Report Generated Successfully. Probabilistic Graphical Models Lab, 2026.")
    r_foot.font.italic = True
    r_foot.font.color.rgb = dark_gray

    # Save output Word document
    out_docx_path = r"C:\CLG\SEM 7\PROBABILISTIC GRAPHICAL MODELS FOR DAS\PAPER\paper_anti\LAB4_EDA_Report.docx"
    doc.save(out_docx_path)
    print(f"\nSuccessfully generated simple Word Document at: '{out_docx_path}'")

if __name__ == '__main__':
    main()
