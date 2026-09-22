import os, sys
import numpy as np
import torch
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

def generate_balanced_colored_topology():
    base_dir = r"c:\Users\USER\OneDrive\Desktop\Probalistic Graphical Lab"
    pt_path = os.path.join(base_dir, "dataset", "elliptic_pyg_data.pt")
    output_path = os.path.join(base_dir, "dataset", "eda_plots", "transaction_network_graph.png")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if not os.path.exists(pt_path):
        print(f"Error: {pt_path} not found.")
        return

    data = torch.load(pt_path, weights_only=False)
    y = data.y.numpy()
    edge_index = data.edge_index.numpy()
    time_step = data.time_step.numpy()
    
    edge_src, edge_dst = edge_index[0], edge_index[1]

    # Target: Exactly EQUAL number of Licit (0), Illicit (1), and Unknown (-1) nodes
    N_PER_CLASS = 20  # 20 Licit + 20 Illicit + 20 Unknown = 60 Nodes Total

    # Search for optimal window with maximum edge density among balanced nodes
    best_edges = -1
    best_selected = []

    np.random.seed(42)
    for ts in range(1, 49):
        ts_nodes = np.where((time_step >= ts) & (time_step <= ts+1))[0]
        ill = [n for n in ts_nodes if y[n] == 1]
        lic = [n for n in ts_nodes if y[n] == 0]
        unk = [n for n in ts_nodes if y[n] == -1]
        
        if len(ill) >= N_PER_CLASS and len(lic) >= N_PER_CLASS and len(unk) >= N_PER_CLASS:
            mask = np.isin(edge_src, ts_nodes) & np.isin(edge_dst, ts_nodes)
            sub_src, sub_dst = edge_src[mask], edge_dst[mask]
            
            G_temp = nx.DiGraph()
            for u, v in zip(sub_src, sub_dst):
                G_temp.add_edge(u, v)
                
            degs = dict(G_temp.degree())
            
            ill_sorted = sorted(ill, key=lambda n: degs.get(n, 0), reverse=True)[:N_PER_CLASS]
            lic_sorted = sorted(lic, key=lambda n: degs.get(n, 0), reverse=True)[:N_PER_CLASS]
            unk_sorted = sorted(unk, key=lambda n: degs.get(n, 0), reverse=True)[:N_PER_CLASS]
            
            selected = ill_sorted + lic_sorted + unk_sorted
            
            sub_mask = np.isin(edge_src, selected) & np.isin(edge_dst, selected)
            edge_cnt = np.sum(sub_mask)
            
            if edge_cnt > best_edges:
                best_edges = edge_cnt
                best_selected = selected

    selected_nodes = best_selected
    print(f"Balanced Selection: {len(selected_nodes)} Nodes ({N_PER_CLASS} Licit, {N_PER_CLASS} Illicit, {N_PER_CLASS} Unknown)")

    # Build directed graph
    sub_mask = np.isin(edge_src, selected_nodes) & np.isin(edge_dst, selected_nodes)
    sub_src = edge_src[sub_mask]
    sub_dst = edge_dst[sub_mask]

    G = nx.DiGraph()
    for n in selected_nodes:
        G.add_node(n, cls=y[n])
    for u, v in zip(sub_src, sub_dst):
        G.add_edge(u, v)

    # Remove isolated nodes if any, while preserving class balance log
    print(f"Graph Construction: {G.number_of_nodes()} Nodes, {G.number_of_edges()} Directed Edges")

    # Compute spring layout
    pos = nx.spring_layout(G, k=0.48, seed=42, iterations=180)

    fig, ax = plt.subplots(figsize=(15, 10), dpi=300)
    ax.set_facecolor('#F9FAFC')
    fig.patch.set_facecolor('#F9FAFC')

    # Color definitions: Distinct colors for Nodes & Edges
    # Licit (Class 0): Emerald Green
    # Illicit (Class 1): Crimson Red
    # Unknown (Class -1): Vibrant Amethyst Purple
    
    COLOR_LICIT_NODE = '#2ECC71'
    BORDER_LICIT_NODE = '#145A32'
    COLOR_LICIT_EDGE = '#27AE60'

    COLOR_ILLICIT_NODE = '#E74C3C'
    BORDER_ILLICIT_NODE = '#78281F'
    COLOR_ILLICIT_EDGE = '#C0392B'

    COLOR_UNKNOWN_NODE = '#9B59B6'
    BORDER_UNKNOWN_NODE = '#4A235A'
    COLOR_UNKNOWN_EDGE = '#8E44AD'

    node_colors = []
    border_colors = []
    node_sizes = []

    total_degrees = dict(G.degree())

    for n in G.nodes():
        cls = y[n]
        deg = total_degrees.get(n, 1)
        size = max(240, min(850, 240 + deg * 50))
        
        if cls == 1: # Illicit
            node_colors.append(COLOR_ILLICIT_NODE)
            border_colors.append(BORDER_ILLICIT_NODE)
            node_sizes.append(size + 60)
        elif cls == 0: # Licit
            node_colors.append(COLOR_LICIT_NODE)
            border_colors.append(BORDER_LICIT_NODE)
            node_sizes.append(size)
        else: # Unknown (-1)
            node_colors.append(COLOR_UNKNOWN_NODE)
            border_colors.append(BORDER_UNKNOWN_NODE)
            node_sizes.append(size)

    # Assign Edge Colors based on transaction flow category
    edge_colors = []
    for u, v in G.edges():
        src_y = y[u]
        dst_y = y[v]
        
        if src_y == 1 or dst_y == 1:
            edge_colors.append(COLOR_ILLICIT_EDGE) # Red for Illicit flow
        elif src_y == 0 and dst_y == 0:
            edge_colors.append(COLOR_LICIT_EDGE)   # Green for Licit flow
        else:
            edge_colors.append(COLOR_UNKNOWN_EDGE) # Purple for Unknown / Mixed flow

    # Draw Colored Edges with Curved Arrows
    nx.draw_networkx_edges(
        G, pos,
        ax=ax,
        arrowstyle='->',
        arrowsize=14,
        edge_color=edge_colors,
        width=1.6,
        alpha=0.75,
        connectionstyle='arc3,rad=0.08'
    )

    # Draw Nodes
    nx.draw_networkx_nodes(
        G, pos,
        ax=ax,
        node_color=node_colors,
        node_size=node_sizes,
        edgecolors=border_colors,
        linewidths=1.8,
        alpha=0.95
    )

    # Node ID Labels
    node_labels = {}
    for n in G.nodes():
        cls = y[n]
        if cls == 1:
            node_labels[n] = f"Ill-{n}"
        elif cls == 0:
            node_labels[n] = f"Lic-{n}"
        else:
            node_labels[n] = f"Unk-{n}"

    nx.draw_networkx_labels(
        G, pos,
        labels=node_labels,
        font_size=6.5,
        font_color='#111111',
        font_weight='bold',
        ax=ax
    )

    # Title & Subtitle
    plt.title("Bitcoin UTXO Balanced Transaction Network Topology", fontsize=16, fontweight='bold', pad=18, color='#1B365D')
    plt.suptitle("Equal Representation (20 Licit, 20 Illicit, 20 Unknown) with Distinct Node & Edge Color Coding", fontsize=10.5, style='italic', color='#444444', y=0.925)

    # Custom Multi-Section Legend for Nodes & Edges
    legend_elements = [
        # Node Categories
        Patch(facecolor=COLOR_ILLICIT_NODE, edgecolor=BORDER_ILLICIT_NODE, label='Illicit Node (20 Nodes / 33.3%)'),
        Patch(facecolor=COLOR_LICIT_NODE, edgecolor=BORDER_LICIT_NODE, label='Licit Node (20 Nodes / 33.3%)'),
        Patch(facecolor=COLOR_UNKNOWN_NODE, edgecolor=BORDER_UNKNOWN_NODE, label='Unknown Node (20 Nodes / 33.3%)'),
        # Edge Categories
        Line2D([0], [0], color=COLOR_ILLICIT_EDGE, lw=2.5, label='Illicit Transaction Edge (Red)'),
        Line2D([0], [0], color=COLOR_LICIT_EDGE, lw=2.5, label='Licit Transaction Edge (Green)'),
        Line2D([0], [0], color=COLOR_UNKNOWN_EDGE, lw=2.5, label='Unknown Transaction Edge (Purple)')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9.5, frameon=True, facecolor='#FFFFFF', edgecolor='#D5D8DC')

    # Information Callout Box
    info_box = (
        f"Equal 1:1:1 Balanced Graph Topology:\n"
        f"• Licit Transactions   : 20 Nodes (Green)\n"
        f"• Illicit Transactions : 20 Nodes (Red)\n"
        f"• Unknown Transactions : 20 Nodes (Purple)\n"
        f"• Total Subgraph Nodes : 60 Nodes\n"
        f"• Total Directed Edges : {G.number_of_edges()} Edges\n"
        f"• Edge Color Coding    : Green (Licit) | Red (Illicit) | Purple (Unknown)"
    )
    ax.text(0.02, 0.98, info_box, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=dict(boxstyle='round,pad=0.6', facecolor='#F4F6F9', alpha=0.92, edgecolor='#1B365D'))

    plt.axis('off')
    plt.tight_layout()

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Successfully generated balanced, colored topology plot at: {output_path}")

if __name__ == '__main__':
    generate_balanced_colored_topology()
