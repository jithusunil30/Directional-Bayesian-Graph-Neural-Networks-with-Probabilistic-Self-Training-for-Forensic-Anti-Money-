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
    print("=== Loading Data for Word Document Report ===")
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
    
    # Degrees
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
    
    # Top features (computed same way as run_eda.py for consistency)
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
    
    # Outliers
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
        
        licit_series = df_feats[df_feats.index.map(lambda i: y_labeled[i] == 0)][col]
        licit_outliers = np.sum((licit_series < lower_bound) | (licit_series > upper_bound))
        licit_pct = licit_outliers / len(licit_series)
        
        illicit_series = df_feats[df_feats.index.map(lambda i: y_labeled[i] == 1)][col]
        illicit_outliers = np.sum((illicit_series < lower_bound) | (illicit_series > upper_bound))
        illicit_pct = illicit_outliers / len(illicit_series)
        
        outlier_records.append({
            'Feature': col,
            'Lower': lower_bound,
            'Upper': upper_bound,
            'TotalOutliers': total_outliers,
            'TotalOutliersPct': f"{pct_outliers:.2%}",
            'LicitOutliersPct': f"{licit_pct:.2%}",
            'IllicitOutliersPct': f"{illicit_pct:.2%}"
        })

    # Start Document
    print("Creating Word Document structure...")
    doc = docx.Document()
    
    # Set Margins
    for section in doc.sections:
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)
        
    # Typography Setup
    style_normal = doc.styles['Normal']
    font = style_normal.font
    font.name = 'Calibri'
    font.size = Pt(11)
    font.color.rgb = RGBColor(0x33, 0x33, 0x33) # Charcoal
    
    COLOR_PRIMARY = RGBColor(0x00, 0x33, 0x66)    # Deep Navy
    COLOR_SECONDARY = RGBColor(0x50, 0x50, 0x50)  # Slate Gray
    COLOR_SUCCESS = RGBColor(0x2E, 0x7D, 0x32)    # Green
    COLOR_WARNING = RGBColor(0xD8, 0x43, 0x15)    # Orange
    
    # Document Title
    title_p = doc.add_paragraph()
    title_p.paragraph_format.space_before = Pt(0)
    title_p.paragraph_format.space_after = Pt(6)
    title_run = title_p.add_run("EXPLORATORY DATA ANALYSIS (EDA) REPORT")
    title_run.font.size = Pt(22)
    title_run.font.bold = True
    title_run.font.color.rgb = COLOR_PRIMARY
    
    subtitle_p = doc.add_paragraph()
    subtitle_p.paragraph_format.space_after = Pt(18)
    sub_run = subtitle_p.add_run("Elliptic Bitcoin Dataset Properties, Network Topology & Modeling Implications")
    sub_run.font.size = Pt(12)
    sub_run.font.italic = True
    sub_run.font.color.rgb = COLOR_SECONDARY
    
    # Divider Line
    p_hr = doc.add_paragraph()
    p_hr.paragraph_format.space_after = Pt(18)
    p_hr_border = parse_xml(f'<w:pBdr {nsdecls("w")}><w:bottom w:val="single" w:sz="6" w:space="1" w:color="003366"/></w:pBdr>')
    p_hr._p.get_or_add_pPr().append(p_hr_border)
    
    # Executive Summary
    doc.add_heading("Executive Summary", level=1)
    doc.paragraphs[-1].runs[0].font.color.rgb = COLOR_PRIMARY
    doc.paragraphs[-1].runs[0].font.bold = True
    
    p = doc.add_paragraph(
        "This report delivers a thorough Exploratory Data Analysis (EDA) on the preprocessed graph dataset "
        "representing Bitcoin transactions. Through quantitative checks and network topological profiling, "
        "we characterize the severe class imbalance, high proportion of unlabeled nodes, localized homophilic "
        "subgraphs, feature distributions, and outlier statistics. We translate these findings into specific "
        "modeling recommendations for constructing a robust Bayesian Graph Neural Network classifier."
    )
    p.paragraph_format.space_after = Pt(12)
    
    # Section 1: Dimensions & Class distribution
    doc.add_heading("1. Preprocessed Dataset Profile & Dimensions", level=1)
    doc.paragraphs[-1].runs[0].font.color.rgb = COLOR_PRIMARY
    doc.paragraphs[-1].runs[0].font.bold = True
    
    doc.add_paragraph(
        "The graph is composed of 203,769 transaction nodes and 234,355 directed payment edges. Each node contains "
        "165 scaled numeric features. The ground truth targets are binary classes: Licit (Class 0), "
        "Illicit (Class 1), and Unlabeled (Class -1). The table below outlines the overall dimensions and class split distributions:"
    )
    
    # Table 1: Split distributions
    table1 = doc.add_table(rows=5, cols=8)
    table1.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(table1)
    
    headers1 = ["Split", "Licit (0)", "Illicit (1)", "Unknown (-1)", "Total Nodes", "Labeled Nodes", "Labeled Fraud Ratio", "Time Steps"]
    widths1 = [Inches(1.2), Inches(0.8), Inches(0.8), Inches(0.9), Inches(0.9), Inches(1.0), Inches(1.3), Inches(0.8)]
    
    hdr_cells = table1.rows[0].cells
    for idx, text in enumerate(headers1):
        hdr_cells[idx].text = text
        hdr_cells[idx].paragraphs[0].runs[0].font.bold = True
        hdr_cells[idx].paragraphs[0].runs[0].font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        set_cell_background(hdr_cells[idx], "003366")
        set_cell_margins(hdr_cells[idx], top=100, bottom=100)
        hdr_cells[idx].width = widths1[idx]
        
    data1 = [
        ["Train Split", f"{train_licit:,}", f"{train_illicit:,}", "0", f"{np.sum(train_mask):,}", f"{train_licit+train_illicit:,}", f"{train_illicit/(train_licit+train_illicit):.2%}", "Steps 1-30"],
        ["Calibration Split", f"{val_licit:,}", f"{val_illicit:,}", "0", f"{np.sum(val_mask):,}", f"{val_licit+val_illicit:,}", f"{val_illicit/(val_licit+val_illicit):.2%}", "Steps 31-34"],
        ["Test Split", f"{test_licit:,}", f"{test_illicit:,}", "0", f"{np.sum(test_mask):,}", f"{test_licit+test_illicit:,}", f"{test_illicit/(test_licit+test_illicit):.2%}", "Steps 35-49"],
        ["Overall Dataset", f"{licit_count:,}", f"{illicit_count:,}", f"{unknown_count:,}", f"{num_nodes:,}", f"{total_labeled:,}", f"{illicit_count/total_labeled:.2%}", "Steps 1-49"]
    ]
    
    for i, row_data in enumerate(data1):
        row_cells = table1.rows[i+1].cells
        for j, val in enumerate(row_data):
            row_cells[j].text = val
            set_cell_margins(row_cells[j], top=70, bottom=70)
            row_cells[j].width = widths1[j]
            if i % 2 == 1:
                set_cell_background(row_cells[j], "F2F5F8")
            if i == 3: # Bold overall row
                row_cells[j].paragraphs[0].runs[0].font.bold = True
                
    doc.add_paragraph().paragraph_format.space_after = Pt(12)
    
    # Embed Plot 1
    if os.path.exists("eda_plots/class_distribution.png"):
        doc.add_paragraph().paragraph_format.space_after = Pt(6)
        doc.add_picture("eda_plots/class_distribution.png", width=Inches(6.0))
        p_cap = doc.add_paragraph("Figure 1: Overall Node Class Pie Chart and Stacked Split Distributions")
        p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_cap.runs[0].font.italic = True
        p_cap.runs[0].font.size = Pt(9)
        p_cap.paragraph_format.space_after = Pt(18)

    # Section 2: Graph degree
    doc.add_heading("2. Graph Topological Statistics (Node Degrees)", level=1)
    doc.paragraphs[-1].runs[0].font.color.rgb = COLOR_PRIMARY
    doc.paragraphs[-1].runs[0].font.bold = True
    
    doc.add_paragraph(
        "By calculating in-degree (incoming transactions) and out-degree (outgoing transaction outputs), "
        "we profile the connectivity structure. Licit nodes participate in more transactions on average (total degree 3.10) "
        "than illicit nodes (total degree 2.01). Legitimate entities exhibit larger hub sizes (max degree 473), "
        "while illicit nodes are bound to sparse paths:"
    )
    
    # Table 2: Node degrees
    table2 = doc.add_table(rows=10, cols=5)
    table2.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(table2)
    
    headers2 = ["Node Class", "Metric", "In-Degree", "Out-Degree", "Total Degree"]
    widths2 = [Inches(1.8), Inches(1.2), Inches(1.2), Inches(1.2), Inches(1.2)]
    
    hdr_cells2 = table2.rows[0].cells
    for idx, text in enumerate(headers2):
        hdr_cells2[idx].text = text
        hdr_cells2[idx].paragraphs[0].runs[0].font.bold = True
        hdr_cells2[idx].paragraphs[0].runs[0].font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        set_cell_background(hdr_cells2[idx], "003366")
        set_cell_margins(hdr_cells2[idx], top=100, bottom=100)
        hdr_cells2[idx].width = widths2[idx]
        
    classes_list = [
        (0, "Licit (Class 0)"),
        (1, "Illicit (Class 1)"),
        (-1, "Unknown (Class -1)")
    ]
    
    row_idx = 1
    for class_val, class_name in classes_list:
        sub_df = df_deg[df_deg['class'] == class_val]
        metrics = [
            ("Mean", f"{sub_df['in_degree'].mean():.2f}", f"{sub_df['out_degree'].mean():.2f}", f"{sub_df['total_degree'].mean():.2f}"),
            ("Median", f"{sub_df['in_degree'].median():.2f}", f"{sub_df['out_degree'].median():.2f}", f"{sub_df['total_degree'].median():.2f}"),
            ("Max Value", f"{sub_df['in_degree'].max():,}", f"{sub_df['out_degree'].max():,}", f"{sub_df['total_degree'].max():,}")
        ]
        for m_name, in_val, out_val, tot_val in metrics:
            cells = table2.rows[row_idx].cells
            cells[0].text = class_name if m_name == "Mean" else ""
            cells[1].text = m_name
            cells[2].text = in_val
            cells[3].text = out_val
            cells[4].text = tot_val
            
            for cell in cells:
                set_cell_margins(cell, top=60, bottom=60)
            if row_idx % 2 == 1:
                for cell in cells:
                    set_cell_background(cell, "F2F5F8")
            row_idx += 1
            
    doc.add_paragraph().paragraph_format.space_after = Pt(12)
    
    # Embed Plot 2
    if os.path.exists("eda_plots/node_degree_distributions.png"):
        doc.add_picture("eda_plots/node_degree_distributions.png", width=Inches(6.0))
        p_cap = doc.add_paragraph("Figure 2: Log-Scaled Degree Density distributions and Boxplots (excluding outliers)")
        p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_cap.runs[0].font.italic = True
        p_cap.runs[0].font.size = Pt(9)
        p_cap.paragraph_format.space_after = Pt(18)

    # Section 3: Linkage patterns
    doc.add_heading("3. Edge Connectivity (Class Linkages & Homophily)", level=1)
    doc.paragraphs[-1].runs[0].font.color.rgb = COLOR_PRIMARY
    doc.paragraphs[-1].runs[0].font.bold = True
    
    doc.add_paragraph(
        "By tracking transaction flows (Source Node -> Destination Node), we observe distinct structural clustering. "
        "Illicit nodes are highly likely to transact with other illicit transactions (29.61%) or unknown nodes (43.25%), "
        "and rarely interact directly with licit accounts (27.14%):"
    )
    
    # Table 3: Transition Matrix
    table3 = doc.add_table(rows=4, cols=5)
    table3.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(table3)
    
    headers3 = ["Source \\ Destination", "Licit (0)", "Illicit (1)", "Unknown (-1)", "Total Out-going"]
    widths3 = [Inches(1.8), Inches(1.2), Inches(1.2), Inches(1.2), Inches(1.4)]
    
    hdr_cells3 = table3.rows[0].cells
    for idx, text in enumerate(headers3):
        hdr_cells3[idx].text = text
        hdr_cells3[idx].paragraphs[0].runs[0].font.bold = True
        hdr_cells3[idx].paragraphs[0].runs[0].font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        set_cell_background(hdr_cells3[idx], "003366")
        set_cell_margins(hdr_cells3[idx], top=100, bottom=100)
        hdr_cells3[idx].width = widths3[idx]
        
    data3 = [
        ["Licit (0)", f"{edge_matrix[0,0]:,} ({edge_matrix_prob[0,0]:.2%})", f"{edge_matrix[0,1]:,} ({edge_matrix_prob[0,1]:.2%})", f"{edge_matrix[0,2]:,} ({edge_matrix_prob[0,2]:.2%})", f"{row_sums[0][0]:,}"],
        ["Illicit (1)", f"{edge_matrix[1,0]:,} ({edge_matrix_prob[1,0]:.2%})", f"{edge_matrix[1,1]:,} ({edge_matrix_prob[1,1]:.2%})", f"{edge_matrix[1,2]:,} ({edge_matrix_prob[1,2]:.2%})", f"{row_sums[1][0]:,}"],
        ["Unknown (-1)", f"{edge_matrix[2,0]:,} ({edge_matrix_prob[2,0]:.2%})", f"{edge_matrix[2,1]:,} ({edge_matrix_prob[2,1]:.2%})", f"{edge_matrix[2,2]:,} ({edge_matrix_prob[2,2]:.2%})", f"{row_sums[2][0]:,}"]
    ]
    
    for i, row_data in enumerate(data3):
        row_cells = table3.rows[i+1].cells
        for j, val in enumerate(row_data):
            row_cells[j].text = val
            set_cell_margins(row_cells[j], top=70, bottom=70)
            row_cells[j].width = widths3[j]
            if i % 2 == 1:
                set_cell_background(row_cells[j], "F2F5F8")
                
    doc.add_paragraph().paragraph_format.space_after = Pt(12)
    
    # Embed Plot 3
    if os.path.exists("eda_plots/edge_type_connectivity.png"):
        doc.add_picture("eda_plots/edge_type_connectivity.png", width=Inches(5.0))
        p_cap = doc.add_paragraph("Figure 3: Heatmap of Outgoing Edge Connection Probabilities by Node Class")
        p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_cap.runs[0].font.italic = True
        p_cap.runs[0].font.size = Pt(9)
        p_cap.paragraph_format.space_after = Pt(18)

    # Section 4: Temporal dynamics
    doc.add_heading("4. Temporal Dynamics & Data Splits", level=1)
    doc.paragraphs[-1].runs[0].font.color.rgb = COLOR_PRIMARY
    doc.paragraphs[-1].runs[0].font.bold = True
    
    doc.add_paragraph(
        "Bitcoin transactions are split chronologically into 49 time steps. To avoid temporal data leakage, "
        "splits are strictly partitioned. The train set spans steps 1-30, the calibration set spans steps 31-34 "
        "and the test set spans steps 35-49. The graph below displays the node count and fraud ratios over the steps, "
        "documenting shifting dynamics (e.g. Test set has a lower fraud ratio of 6.50% vs. 10.98% in training):"
    )
    
    # Embed Plot 4
    if os.path.exists("eda_plots/temporal_trends.png"):
        doc.add_picture("eda_plots/temporal_trends.png", width=Inches(6.0))
        p_cap = doc.add_paragraph("Figure 4: Node Counts (Bars) and Labeled Fraud Ratios (Line) over the 49 Time Steps")
        p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_cap.runs[0].font.italic = True
        p_cap.runs[0].font.size = Pt(9)
        p_cap.paragraph_format.space_after = Pt(18)

    # Section 5: Feature analysis & Outliers
    doc.add_heading("5. Discriminative Features & Outlier Analysis", level=1)
    doc.paragraphs[-1].runs[0].font.color.rgb = COLOR_PRIMARY
    doc.paragraphs[-1].runs[0].font.bold = True
    
    doc.add_paragraph(
        "By looking at the absolute difference in means between licit and illicit classes, we identified "
        "the most discriminative features. The top local features are index 52, 54, 88, and 89. The top neighbor "
        "aggregation features are index 108, 105, 106, and 141. The table below summaries their boundaries "
        "and outlier rates using the Interquartile Range (IQR) method:"
    )
    
    # Table 4: Outliers
    table4 = doc.add_table(rows=9, cols=8)
    table4.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(table4)
    
    headers4 = ["Feature Name", "Type", "Lower Bound", "Upper Bound", "Total Outliers", "Total Outliers %", "Licit Outlier %", "Illicit Outlier %"]
    widths4 = [Inches(1.2), Inches(0.9), Inches(1.1), Inches(1.1), Inches(1.1), Inches(1.1), Inches(1.2), Inches(1.2)]
    
    hdr_cells4 = table4.rows[0].cells
    for idx, text in enumerate(headers4):
        hdr_cells4[idx].text = text
        hdr_cells4[idx].paragraphs[0].runs[0].font.bold = True
        hdr_cells4[idx].paragraphs[0].runs[0].font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        set_cell_background(hdr_cells4[idx], "003366")
        set_cell_margins(hdr_cells4[idx], top=100, bottom=100)
        hdr_cells4[idx].width = widths4[idx]
        
    for i, record in enumerate(outlier_records):
        row_cells = table4.rows[i+1].cells
        row_cells[0].text = record['Feature']
        row_cells[1].text = "Local" if i < 4 else "Neighbor"
        row_cells[2].text = f"{record['Lower']:.3f}"
        row_cells[3].text = f"{record['Upper']:.3f}"
        row_cells[4].text = f"{record['TotalOutliers']:,}"
        row_cells[5].text = record['TotalOutliersPct']
        row_cells[6].text = record['LicitOutliersPct']
        row_cells[7].text = record['IllicitOutliersPct']
        
        for cell in row_cells:
            set_cell_margins(cell, top=60, bottom=60)
        if i % 2 == 1:
            for cell in row_cells:
                set_cell_background(cell, "F2F5F8")
                
    doc.add_paragraph().paragraph_format.space_after = Pt(12)
    
    # Embed Plot 5
    if os.path.exists("eda_plots/feature_distributions.png"):
        doc.add_picture("eda_plots/feature_distributions.png", width=Inches(6.0))
        p_cap = doc.add_paragraph("Figure 5: Density Distributions of Discriminative Features split by Licit vs. Illicit")
        p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_cap.runs[0].font.italic = True
        p_cap.runs[0].font.size = Pt(9)
        p_cap.paragraph_format.space_after = Pt(18)

    # Embed Plot 6
    if os.path.exists("eda_plots/feature_boxplots.png"):
        doc.add_picture("eda_plots/feature_boxplots.png", width=Inches(6.0))
        p_cap = doc.add_paragraph("Figure 6: Outlier Boxplots for selected Local and Neighbor features")
        p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_cap.runs[0].font.italic = True
        p_cap.runs[0].font.size = Pt(9)
        p_cap.paragraph_format.space_after = Pt(18)
        
    # Embed Plot 7
    if os.path.exists("eda_plots/feature_correlation_heatmap.png"):
        doc.add_picture("eda_plots/feature_correlation_heatmap.png", width=Inches(5.0))
        p_cap = doc.add_paragraph("Figure 7: Correlation Heatmap of top discriminative features and Target Label")
        p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_cap.runs[0].font.italic = True
        p_cap.runs[0].font.size = Pt(9)
        p_cap.paragraph_format.space_after = Pt(18)

    # Section 6: Modeling Implications
    doc.add_heading("6. Discussion of Modeling Implications", level=1)
    doc.paragraphs[-1].runs[0].font.color.rgb = COLOR_PRIMARY
    doc.paragraphs[-1].runs[0].font.bold = True
    
    implication_text = [
        ("Class Imbalance Strategy", 
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
    
    output_docx = "Exploratory_Data_Analysis_Report.docx"
    doc.save(output_docx)
    print(f"\nWord Document saved successfully at: {output_docx}")

if __name__ == '__main__':
    main()
