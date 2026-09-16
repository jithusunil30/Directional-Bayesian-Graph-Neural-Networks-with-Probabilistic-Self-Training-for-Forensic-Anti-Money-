import os
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn
import torch
import pandas as pd
import numpy as np

def set_cell_background(cell, hex_color):
    """Sets the background color of a table cell."""
    shading_elm = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{hex_color}"/>')
    cell._tc.get_or_add_tcPr().append(shading_elm)

def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    """Sets margins (padding) for a table cell."""
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = OxmlElement('w:tcMar')
    for m, val in [('w:top', top), ('w:bottom', bottom), ('w:left', left), ('w:right', right)]:
        node = OxmlElement(m)
        node.set(qn('w:w'), str(val))
        node.set(qn('w:type'), 'dxa')
        tcMar.append(node)
    tcPr.append(tcMar)

def set_table_borders(table):
    """Applies clean, professional borders (light gray horizontal lines only)."""
    tblPr = table._tbl.tblPr
    borders = parse_xml(
        f'<w:tblBorders {nsdecls("w")}>'
        '<w:top w:val="single" w:sz="4" w:space="0" w:color="CCCCCC"/>'
        '<w:bottom w:val="single" w:sz="4" w:space="0" w:color="CCCCCC"/>'
        '<w:left w:val="none"/>'
        '<w:right w:val="none"/>'
        '<w:insideH w:val="single" w:sz="4" w:space="0" w:color="E0E0E0"/>'
        '<w:insideV w:val="none"/>'
        '</w:tblBorders>'
    )
    tblPr.append(borders)

def main():
    print("=== Loading Data ===")
    pt_path = 'elliptic_pyg_data.pt'
    if not os.path.exists(pt_path) and os.path.exists('updated/elliptic_pyg_data.pt'):
        pt_path = 'updated/elliptic_pyg_data.pt'
        
    data = torch.load(pt_path, weights_only=False)
    
    num_nodes = data.x.shape[0]
    num_features = data.x.shape[1]
    num_edges = data.edge_index.shape[1]
    
    x_np = data.x.numpy()
    y_np = data.y.numpy()
    time_steps = data.time_step.numpy()
    train_mask = data.train_mask.numpy()
    val_mask = data.val_mask.numpy()
    test_mask = data.test_mask.numpy()
    
    # Class stats
    unknown_count = np.sum(y_np == -1)
    licit_count = np.sum(y_np == 0)
    illicit_count = np.sum(y_np == 1)
    total_labeled = licit_count + illicit_count
    
    train_licit = np.sum((y_np == 0) & train_mask)
    train_illicit = np.sum((y_np == 1) & train_mask)
    val_licit = np.sum((y_np == 0) & val_mask)
    val_illicit = np.sum((y_np == 1) & val_mask)
    test_licit = np.sum((y_np == 0) & test_mask)
    test_illicit = np.sum((y_np == 1) & test_mask)
    
    # Node Degrees
    src = data.edge_index[0].numpy()
    dst = data.edge_index[1].numpy()
    out_degree = np.bincount(src, minlength=num_nodes)
    in_degree = np.bincount(dst, minlength=num_nodes)
    total_degree = out_degree + in_degree
    
    df_deg = pd.DataFrame({
        'class': y_np,
        'in_degree': in_degree,
        'out_degree': out_degree,
        'total_degree': total_degree
    })
    
    # Edge Matrix
    src_mapped = np.where(y_np[src] == -1, 2, y_np[src])
    dst_mapped = np.where(y_np[dst] == -1, 2, y_np[dst])
    edge_matrix = np.zeros((3, 3), dtype=int)
    for s, d in zip(src_mapped, dst_mapped):
        edge_matrix[s, d] += 1
    row_sums = edge_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    edge_matrix_prob = edge_matrix / row_sums
    
    # Selected Top Features (from run_eda.py)
    labeled_indices = np.where(y_np != -1)[0]
    x_labeled = x_np[labeled_indices]
    y_labeled = y_np[labeled_indices]
    licit_feats = x_labeled[y_labeled == 0]
    illicit_feats = x_labeled[y_labeled == 1]
    mean_diff = np.abs(np.mean(licit_feats, axis=0) - np.mean(illicit_feats, axis=0))
    top_local_idx = np.argsort(mean_diff[:94])[-4:][::-1]
    top_neighbor_idx = np.argsort(mean_diff[94:])[-4:][::-1] + 94
    selected_features = list(top_local_idx) + list(top_neighbor_idx)
    selected_feature_names = [f'feat_{i}' for i in selected_features]
    df_feats = pd.DataFrame(x_labeled[:, selected_features], columns=selected_feature_names)
    df_feats['label'] = y_labeled
    
    # Calculate Correlation with Target Label
    correlations = df_feats.corr(numeric_only=True)['label'].drop('label')
    
    # Outliers calculation
    outlier_records = []
    for col in selected_feature_names:
        series = df_feats[col]
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        total_outliers = np.sum((series < lower_bound) | (series > upper_bound))
        pct_outliers = total_outliers / len(series)
        
        licit_series = df_feats[df_feats['label'] == 0][col]
        licit_outliers = np.sum((licit_series < lower_bound) | (licit_series > upper_bound))
        licit_pct = licit_outliers / len(licit_series)
        
        illicit_series = df_feats[df_feats['label'] == 1][col]
        illicit_outliers = np.sum((illicit_series < lower_bound) | (illicit_series > upper_bound))
        illicit_pct = illicit_outliers / len(illicit_series)
        
        outlier_records.append({
            'Feature': col,
            'Q1': q1,
            'Q3': q3,
            'IQR': iqr,
            'Lower': lower_bound,
            'Upper': upper_bound,
            'TotalOutliers': total_outliers,
            'TotalOutliersPct': f"{pct_outliers:.2%}",
            'LicitOutliersPct': f"{licit_pct:.2%}",
            'IllicitOutliersPct': f"{illicit_pct:.2%}"
        })

    print("Creating PGM_Lab_4_EDA_Report.docx Document...")
    doc = docx.Document()
    
    # Set margins
    for s in doc.sections:
        s.top_margin = Inches(1.0)
        s.bottom_margin = Inches(1.0)
        s.left_margin = Inches(1.0)
        s.right_margin = Inches(1.0)
        
    style_normal = doc.styles['Normal']
    font = style_normal.font
    font.name = 'Calibri'
    font.size = Pt(11)
    font.color.rgb = RGBColor(0x33, 0x33, 0x33)
    
    COLOR_PRIMARY = RGBColor(0x00, 0x33, 0x66)    # Deep Navy
    COLOR_SECONDARY = RGBColor(0x50, 0x50, 0x50)  # Slate Gray
    COLOR_SUCCESS = RGBColor(0x2E, 0x7D, 0x32)
    COLOR_WARNING = RGBColor(0xD8, 0x43, 0x15)
    
    # Document Header Title
    title_p = doc.add_paragraph()
    title_p.paragraph_format.space_before = Pt(0)
    title_p.paragraph_format.space_after = Pt(4)
    title_run = title_p.add_run("LAB 4: EXPLORATORY DATA ANALYSIS (EDA) REPORT")
    title_run.font.size = Pt(20)
    title_run.font.bold = True
    title_run.font.color.rgb = COLOR_PRIMARY
    
    sub_p = doc.add_paragraph()
    sub_p.paragraph_format.space_after = Pt(12)
    sub_run = sub_p.add_run("Comprehensive Structural and Topological Analysis of the Elliptic Bitcoin Dataset")
    sub_run.font.size = Pt(11)
    sub_run.font.italic = True
    sub_run.font.color.rgb = COLOR_SECONDARY
    
    p_hr = doc.add_paragraph()
    p_hr.paragraph_format.space_after = Pt(18)
    p_hr_border = parse_xml(f'<w:pBdr {nsdecls("w")}><w:bottom w:val="single" w:sz="6" w:space="1" w:color="003366"/></w:pBdr>')
    p_hr._p.get_or_add_pPr().append(p_hr_border)
    
    # ----------------------------------------------------
    # SECTION 1: EXAMINE THE DATASET
    # ----------------------------------------------------
    doc.add_heading("1. Examine the Dataset (Dimensions, Attributes & Statistics)", level=1)
    doc.paragraphs[-1].runs[0].font.color.rgb = COLOR_PRIMARY
    doc.paragraphs[-1].runs[0].font.bold = True
    
    # Number of samples and features
    doc.add_heading("Number of Samples and Features", level=2)
    doc.paragraphs[-1].runs[0].font.color.rgb = COLOR_SECONDARY
    doc.add_paragraph(
        f"The preprocessed Elliptic Bitcoin dataset contains a total of **{num_nodes:,} samples (nodes)** "
        f"and **{num_features} numerical attributes (features)**. Additionally, there are **{num_edges:,} directed payment "
        f"edges** defining the payment flow structure of the transaction graph."
    )
    
    # Data types of attributes
    doc.add_heading("Data Types of Attributes", level=2)
    doc.paragraphs[-1].runs[0].font.color.rgb = COLOR_SECONDARY
    doc.add_paragraph(
        "The graph is constructed with contiguous node mappings. In tabular form, the attributes and data types are mapped as follows:"
    )
    
    # Table of types
    table_types = doc.add_table(rows=7, cols=3)
    table_types.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(table_types)
    widths_t = [Inches(2.0), Inches(1.5), Inches(3.0)]
    
    hdr_t = table_types.rows[0].cells
    for i, t in enumerate(["Attribute Name", "Data Type", "Role & Description"]):
        hdr_t[i].text = t
        hdr_t[i].paragraphs[0].runs[0].font.bold = True
        hdr_t[i].paragraphs[0].runs[0].font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        set_cell_background(hdr_t[i], "003366")
        set_cell_margins(hdr_t[i], top=80, bottom=80)
        hdr_t[i].width = widths_t[i]
        
    type_data = [
        ["node_idx", "int64 / int32", "Contiguous index of the transaction in the PyG graph [0, N-1]"],
        ["time_step", "int64 (1 to 49)", "Chronological snapshot index showing when the transaction occurred"],
        ["label", "int64 (0, 1, -1)", "Class target: 0 = Licit (legit), 1 = Illicit (fraud), -1 = Unknown (unlabeled)"],
        ["train_mask / val_mask / test_mask", "boolean / uint8", "Train, calibration, and test split designations"],
        ["feat_0 to feat_93", "float32 (scaled)", "94 local features capturing individual transaction dynamics"],
        ["feat_94 to feat_164", "float32 (scaled)", "71 aggregate features capturing immediate neighbor metrics"]
    ]
    
    for r_idx, row in enumerate(type_data):
        cells = table_types.rows[r_idx+1].cells
        for c_idx, val in enumerate(row):
            cells[c_idx].text = val
            set_cell_margins(cells[c_idx], top=60, bottom=60)
            cells[c_idx].width = widths_t[c_idx]
            if r_idx % 2 == 1:
                set_cell_background(cells[c_idx], "F2F5F8")
                
    doc.add_paragraph().paragraph_format.space_after = Pt(12)
    
    # Summary statistics
    doc.add_heading("Summary Statistics of Selected Features", level=2)
    doc.paragraphs[-1].runs[0].font.color.rgb = COLOR_SECONDARY
    doc.add_paragraph(
        "All 165 features have been standardized (StandardScaler fit on training steps 1-30). This results in an overall mean of "
        "approximately 0 and a standard deviation of 1. To provide detailed statistics, the table below lists the calculated mean, "
        "median, standard deviation, minimum, and maximum for the top 8 discriminative features split by target class (Licit vs. Illicit):"
    )
    
    # Detailed stats table
    table_stats = doc.add_table(rows=17, cols=7)
    table_stats.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(table_stats)
    widths_s = [Inches(1.2), Inches(0.8), Inches(1.0), Inches(1.0), Inches(1.0), Inches(1.0), Inches(1.0)]
    
    hdr_s = table_stats.rows[0].cells
    for i, t in enumerate(["Feature", "Class", "Mean", "Median", "Std Dev", "Min", "Max"]):
        hdr_s[i].text = t
        hdr_s[i].paragraphs[0].runs[0].font.bold = True
        hdr_s[i].paragraphs[0].runs[0].font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        set_cell_background(hdr_s[i], "003366")
        set_cell_margins(hdr_s[i], top=80, bottom=80)
        hdr_s[i].width = widths_s[i]
        
    s_idx = 1
    for f in selected_feature_names:
        for c_val, c_lbl in [(0, "Licit"), (1, "Illicit")]:
            subset = df_feats[df_feats['label'] == c_val][f]
            cells = table_stats.rows[s_idx].cells
            cells[0].text = f if c_val == 0 else ""
            cells[1].text = c_lbl
            cells[2].text = f"{subset.mean():.3f}"
            cells[3].text = f"{subset.median():.3f}"
            cells[4].text = f"{subset.std():.3f}"
            cells[5].text = f"{subset.min():.3f}"
            cells[6].text = f"{subset.max():.3f}"
            
            for cell in cells:
                set_cell_margins(cell, top=50, bottom=50)
            if (s_idx // 2) % 2 == 1:
                for cell in cells:
                    set_cell_background(cell, "F2F5F8")
            s_idx += 1
            
    doc.add_paragraph().paragraph_format.space_after = Pt(12)
    
    # Class distribution
    doc.add_heading("Class Distribution and Class Imbalance", level=2)
    doc.paragraphs[-1].runs[0].font.color.rgb = COLOR_SECONDARY
    doc.add_paragraph(
        f"The dataset experiences severe class imbalance. Licit nodes account for **{licit_count:,} ({licit_count/num_nodes:.2%})** "
        f"of all nodes, whereas Illicit transactions account for only **{illicit_count:,} ({illicit_count/num_nodes:.2%})**. "
        f"The remaining **{unknown_count:,} ({unknown_count/num_nodes:.2%})** nodes are unlabeled. "
        f"Focusing strictly on labeled transactions ({total_labeled:,} total), **{licit_count/total_labeled:.2%}** are legitimate "
        f"and **{illicit_count/total_labeled:.2%}** are fraudulent, showing an imbalance ratio of **8.11 to 1** in training."
    )
    
    # ----------------------------------------------------
    # SECTION 2: PERFORM EXPLORATORY DATA ANALYSIS
    # ----------------------------------------------------
    doc.add_heading("2. Perform Exploratory Data Analysis (EDA)", level=1)
    doc.paragraphs[-1].runs[0].font.color.rgb = COLOR_PRIMARY
    doc.paragraphs[-1].runs[0].font.bold = True
    
    # Identifying feature distributions
    doc.add_heading("Identifying Feature Distributions", level=2)
    doc.paragraphs[-1].runs[0].font.color.rgb = COLOR_SECONDARY
    doc.add_paragraph(
        "By comparing feature means, we observe feature anisotropy. Licit nodes occupy positive value spaces on discriminative "
        "local features (e.g. feat_52 mean is +0.873, feat_54 is +0.732), while illicit nodes occupy negative spaces (feat_52 mean "
        "is -0.293, feat_54 is -0.276). In contrast, neighbor features show a reverse pattern (e.g., neighbor aggregated feat_141 "
        "has a mean of +0.132 for licit but peaks at +1.095 for illicit). The density distributions exhibit significant skewness "
        "and heavy tails, common in monetary transactions."
    )
    
    # Analyzing correlations
    doc.add_heading("Analyzing Correlations Between Numerical Features", level=2)
    doc.paragraphs[-1].runs[0].font.color.rgb = COLOR_SECONDARY
    doc.add_paragraph(
        "We computed the correlation coefficients between the discriminative features and the target label. "
        "Features like feat_52 (-0.38), feat_54 (-0.32), and neighbor feature feat_108 (-0.39) show notable negative correlation "
        "with the binary target (meaning lower values align with illicit class 1). Conversely, neighbor aggregate feature feat_141 "
        "exhibits a strong positive correlation (+0.36) with illicit transactions, indicating that connection to active transaction "
        "groups is a key indicator of fraud. Multi-collinearity is also visible among neighboring aggregated features (e.g., feat_105 "
        "and feat_106 are correlated at +0.97, reflecting redundant neighbor statistics)."
    )
    
    # Detecting outliers
    doc.add_heading("Detecting Outliers", level=2)
    doc.paragraphs[-1].runs[0].font.color.rgb = COLOR_SECONDARY
    doc.add_paragraph(
        "Using the Interquartile Range (IQR) method, outliers are heavily concentrated in the neighbor features. "
        "Licit nodes have high outlier rates in local metrics (e.g. 19.97% on feat_106). Conversely, illicit transactions "
        "have extremely high outlier percentages on neighbor feature feat_141 (**45.72%**), displaying severe structural anomalies "
        "compared to the general population. This reflects concentrated, high-frequency money-laundering transaction structures."
    )
    
    # Checking class imbalance
    doc.add_heading("Class Imbalance Check", level=2)
    doc.paragraphs[-1].runs[0].font.color.rgb = COLOR_SECONDARY
    doc.add_paragraph(
        "Class imbalance is consistent across chronological splits. The fraud ratio is 10.98% in training (steps 1-30), "
        "peaks at 17.00% in calibration (steps 31-34), and declines to 6.50% in testing (steps 35-49). This temporal fluctuation "
        "necessitates robust evaluation and training-imbalance scaling."
    )
    
    # ----------------------------------------------------
    # SECTION 3: VISUALIZATIONS & INTERPRETATIONS
    # ----------------------------------------------------
    doc.add_heading("3. Create Visualizations & Interpretations", level=1)
    doc.paragraphs[-1].runs[0].font.color.rgb = COLOR_PRIMARY
    doc.paragraphs[-1].runs[0].font.bold = True
    
    # Visualization 1
    doc.add_heading("Pie & Stacked Bar Charts for Class Distribution", level=2)
    doc.paragraphs[-1].runs[0].font.color.rgb = COLOR_SECONDARY
    if os.path.exists("eda_plots/class_distribution.png"):
        doc.add_picture("eda_plots/class_distribution.png", width=Inches(5.8))
        p_cap = doc.add_paragraph("Figure 1: Overall Node Class Distribution and Labeled Node Split-wise Distributions")
        p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_cap.runs[0].font.italic = True
        p_cap.runs[0].font.size = Pt(9)
    doc.add_paragraph(
        "**Interpretation**: The pie chart confirms that 77.1% of transactions are unlabelled, leaving only 22.9% labeled. "
        "The stacked bar chart shows the training set contains 26,905 labeled transactions, calibration contains 2,989, and testing "
        "contains 16,670. The fraud proportions (red bars) represent a tiny minority in all partitions, verifying that class imbalance "
        "remains a critical hurdle across all splits."
    )
    
    # Visualization 2
    doc.add_heading("Node Degree Distributions (Densities & Boxplots)", level=2)
    doc.paragraphs[-1].runs[0].font.color.rgb = COLOR_SECONDARY
    if os.path.exists("eda_plots/node_degree_distributions.png"):
        doc.add_picture("eda_plots/node_degree_distributions.png", width=Inches(5.8))
        p_cap = doc.add_paragraph("Figure 2: Log-Scaled Node Degree Density Distributions and Class Boxplots")
        p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_cap.runs[0].font.italic = True
        p_cap.runs[0].font.size = Pt(9)
    doc.add_paragraph(
        "**Interpretation**: The density plot highlights a classic scale-free network profile: most transactions have a degree "
        "of 1 or 2, while a small number of hub transactions connect to hundreds of other transactions. Licit nodes show a shift "
        "towards higher degrees on the log-scale, whereas illicit nodes have a sharp, narrow peak at degree 1, verifying that fraudulent "
        "actors restrict transaction degrees to bypass monitoring."
    )
    
    # Visualization 3
    doc.add_heading("Edge Class Linkage Probabilities", level=2)
    doc.paragraphs[-1].runs[0].font.color.rgb = COLOR_SECONDARY
    if os.path.exists("eda_plots/edge_type_connectivity.png"):
        doc.add_picture("eda_plots/edge_type_connectivity.png", width=Inches(4.5))
        p_cap = doc.add_paragraph("Figure 3: Heatmap Matrix of Edge Transition Connection Probabilities")
        p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_cap.runs[0].font.italic = True
        p_cap.runs[0].font.size = Pt(9)
    doc.add_paragraph(
        "**Interpretation**: This connection matrix reveals strong community clustering. Outgoing links from illicit transactions "
        "connect to other illicit nodes 29.61% of the time, and flow into unlabeled bridge nodes 43.25% of the time, with only 27.14% "
        "connecting directly to legitimate nodes. This confirms the presence of money-laundering cycles and cliques, indicating that "
        "graph message passing will be highly effective."
    )
    
    # Visualization 4
    doc.add_heading("Temporal Split Trends and Fraud Ratio Evolution", level=2)
    doc.paragraphs[-1].runs[0].font.color.rgb = COLOR_SECONDARY
    if os.path.exists("eda_plots/temporal_trends.png"):
        doc.add_picture("eda_plots/temporal_trends.png", width=Inches(5.8))
        p_cap = doc.add_paragraph("Figure 4: Total Nodes and Labeled Fraud Ratio across Time Steps 1-49")
        p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_cap.runs[0].font.italic = True
        p_cap.runs[0].font.size = Pt(9)
    doc.add_paragraph(
        "**Interpretation**: The node distribution over time is highly volatile, showing periodic waves. The labeled fraud ratio "
        "(red line) fluctuates significantly, peaking during the calibration split (steps 31-34) and dipping sharply in the test split "
        "(steps 35-49). Because the distribution changes over time, random validation splits would create data leakage, validating "
        "the need for chronological temporal splitting."
    )
    
    # Visualization 5 & 6
    doc.add_heading("Feature Distributions and Outlier Boxplots", level=2)
    doc.paragraphs[-1].runs[0].font.color.rgb = COLOR_SECONDARY
    if os.path.exists("eda_plots/feature_distributions.png"):
        doc.add_picture("eda_plots/feature_distributions.png", width=Inches(5.8))
        p_cap = doc.add_paragraph("Figure 5: Class-wise Density plots for top 8 discriminative features")
        p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_cap.runs[0].font.italic = True
        p_cap.runs[0].font.size = Pt(9)
        
    if os.path.exists("eda_plots/feature_boxplots.png"):
        doc.add_picture("eda_plots/feature_boxplots.png", width=Inches(5.8))
        p_cap = doc.add_paragraph("Figure 6: Boxplots demonstrating feature outlier locations by Class")
        p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_cap.runs[0].font.italic = True
        p_cap.runs[0].font.size = Pt(9)
    doc.add_paragraph(
        "**Interpretation**: The density plots display distinct, non-overlapping shapes between licit (blue) and illicit (red) classes. "
        "Local features (feat_52, feat_54, feat_88, feat_89) have values concentrated around 0.5 to 1.0 for licit, while illicit values "
        "are concentrated below 0. The boxplots demonstrate heavy outlier profiles. Licit nodes exhibit dense outliers on local features, "
        "while illicit nodes display extensive, extreme outlier spikes on neighborhood aggregation metrics (e.g. feat_141)."
    )
    
    # Visualization 7 & 8
    doc.add_heading("Feature Correlations & Pairwise Scatter Plots", level=2)
    doc.paragraphs[-1].runs[0].font.color.rgb = COLOR_SECONDARY
    if os.path.exists("eda_plots/feature_correlation_heatmap.png"):
        doc.add_picture("eda_plots/feature_correlation_heatmap.png", width=Inches(4.8))
        p_cap = doc.add_paragraph("Figure 7: Correlation Heatmap Matrix of Selected features and binary Label")
        p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_cap.runs[0].font.italic = True
        p_cap.runs[0].font.size = Pt(9)
        
    if os.path.exists("eda_plots/pairplot.png"):
        doc.add_picture("eda_plots/pairplot.png", width=Inches(5.8))
        p_cap = doc.add_paragraph("Figure 8: Pairwise Scatter Matrix showing feature interactions split by Class")
        p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_cap.runs[0].font.italic = True
        p_cap.runs[0].font.size = Pt(9)
    doc.add_paragraph(
        "**Interpretation**: The correlation matrix identifies negative correlations with the label for feat_52, feat_54, feat_88, feat_89, "
        "feat_108, feat_105, and feat_106 (coefficients from -0.30 to -0.39), while feat_141 correlates positively (+0.36). "
        "The pairplot illustrates clear class segregation: illicit transactions occupy a tight cluster in the feature space, "
        "while licit transactions are more widely dispersed, indicating distinct transactional behaviors."
    )
    
    # ----------------------------------------------------
    # SECTION 4: MODELING IMPLICATIONS
    # ----------------------------------------------------
    doc.add_heading("4. Influence of EDA Findings on Model Selection", level=1)
    doc.paragraphs[-1].runs[0].font.color.rgb = COLOR_PRIMARY
    doc.paragraphs[-1].runs[0].font.bold = True
    
    implication_text = [
        ("Handling Class Imbalance", 
         "The training data features a severe imbalance ratio (10.98% fraud, which translates to a licit-to-illicit ratio of 8.11). "
         "Standard binary loss functions will bias predictions to the majority class. We must use the calculated training "
         "imbalance ratio (8.11) as a class weight ('pos_weight') in loss computation during optimization."),
        ("Graph Topology Propagation", 
         "Over 77% of nodes are unlabeled. The connectivity matrix documents high homophily—illicit transactions are highly likely to transact "
         "with other illicit or unknown nodes. We recommend using Graph Neural Networks (such as GCN, GAT, or GraphSAGE) "
         "to aggregate neighbor features across structural edges. This allows representation vectors to capture connectivity patterns, "
         "vastly improving classification performance over purely tabular baselines."),
        ("Robustness to Heavy Tails & Outliers", 
         "Outlier rates range from 4% up to 45% (especially for neighbor aggregates on illicit nodes). Standard linear models "
         "can be highly unstable under such distributions. Downstream classifiers must either use robust architectures (e.g. tree-based "
         "XGBoost/LightGBM models) or deep neural networks that employ Batch Normalization or robust activation functions to resist outlier distortion."),
        ("Chronological Snapshot Validation", 
         "Class distributions shift over time (e.g. fraud ratio drops from 10.98% to 6.50% in the test split). Models must be evaluated "
         "strictly according to the temporal split (Train: Steps 1-30, Calibration: Steps 31-34, Test: Steps 35-49) to eliminate data leakage. "
         "Random splits would leak future nodes to the past, resulting in artificial overperformance that fails in live systems."),
        ("Bayesian Calibration set", 
         "Steps 31-34 (2,989 nodes) are held out exclusively as a calibration set. This allows us to calculate Expected Calibration Error (ECE) "
         "and plot reliability curves, ensuring the confidence intervals output by the Bayesian neural network are accurate.")
    ]
    
    for title, desc in implication_text:
        p_imp = doc.add_paragraph()
        run_title = p_imp.add_run(f"• {title}: ")
        run_title.bold = True
        run_title.font.color.rgb = COLOR_SECONDARY
        p_imp.add_run(desc)
        p_imp.paragraph_format.space_after = Pt(8)
        
    doc.add_paragraph().paragraph_format.space_after = Pt(24)
    
    # Footer
    p_footer = doc.add_paragraph()
    p_footer.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    run_f = p_footer.add_run("Report Generated Successfully. Probabilistic Graphical Models Lab, 2026.")
    run_f.font.size = Pt(8.5)
    run_f.font.italic = True
    run_f.font.color.rgb = COLOR_SECONDARY
    
    output_docx = "PGM_Lab_4_EDA_Report.docx"
    doc.save(output_docx)
    print(f"\nWord Document saved successfully at: {output_docx}")

if __name__ == '__main__':
    main()
