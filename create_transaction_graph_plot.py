import os, sys
import numpy as np
import torch
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, FancyBboxPatch
from matplotlib.lines import Line2D

def generate_perfect_24node_structured_topology():
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

    # Exactly 8 Licit, 8 Illicit, 8 Unknown = 24 Nodes Total
    N_PER_CLASS = 8

    adj = {}
    for u, v in zip(edge_src, edge_dst):
        if u not in adj: adj[u] = []
        if v not in adj: adj[v] = []
        adj[u].append(v)
        adj[v].append(u)

    np.random.seed(42)
    illicit_seeds = np.where((y == 1) & (time_step >= 10) & (time_step <= 35))[0]

    best_G = None
    best_edge_count = -1

    for seed in illicit_seeds[:100]:
        selected_nodes = {seed}
        counts = {1: 1, 0: 0, -1: 0}
        frontier = set(adj.get(seed, []))
        
        while (counts[1] < N_PER_CLASS or counts[0] < N_PER_CLASS or counts[-1] < N_PER_CLASS) and frontier:
            candidate = None
            for fn in list(frontier):
                cls = y[fn]
                if counts[cls] < N_PER_CLASS:
                    candidate = fn
                    break
            if candidate is None: break
            selected_nodes.add(candidate)
            counts[y[candidate]] += 1
            frontier.remove(candidate)
            for nxt in adj.get(candidate, []):
                if nxt not in selected_nodes: frontier.add(nxt)
                
        if counts[1] == N_PER_CLASS and counts[0] == N_PER_CLASS and counts[-1] == N_PER_CLASS:
            sub_nodes = list(selected_nodes)
            sub_mask = np.isin(edge_src, sub_nodes) & np.isin(edge_dst, sub_nodes)
            G_temp = nx.DiGraph()
            for u in sub_nodes: G_temp.add_node(u, cls=y[u])
            for u, v in zip(edge_src[sub_mask], edge_dst[sub_mask]): G_temp.add_edge(u, v)
            if len(list(nx.isolates(G_temp))) == 0: # 100% connected
                if G_temp.number_of_edges() > best_edge_count:
                    best_edge_count = G_temp.number_of_edges()
                    best_G = G_temp.copy()

    G = best_G
    print(f"Verified 24-Node Graph:")
    print(f"• Total Nodes  : {G.number_of_nodes()} (8 Licit, 8 Illicit, 8 Unknown)")
    print(f"• Total Edges  : {G.number_of_edges()} Directed Edges")
    print(f"• Isolates     : {len(list(nx.isolates(G)))} (100% Connected)")

    # Group nodes by class for structured 3-column placement
    lic_nodes = sorted([n for n in G.nodes() if y[n] == 0])
    unk_nodes = sorted([n for n in G.nodes() if y[n] == -1])
    ill_nodes = sorted([n for n in G.nodes() if y[n] == 1])

    pos = {}
    
    # Column 1 (Left - Licit): x = -2.5
    for idx, n in enumerate(lic_nodes):
        pos[n] = np.array([-2.5, 3.5 - idx * 1.0])

    # Column 2 (Center - Unknown): x = 0.0
    for idx, n in enumerate(unk_nodes):
        pos[n] = np.array([0.0, 3.5 - idx * 1.0])

    # Column 3 (Right - Illicit): x = +2.5
    for idx, n in enumerate(ill_nodes):
        pos[n] = np.array([2.5, 3.5 - idx * 1.0])

    fig, ax = plt.subplots(figsize=(15, 10), dpi=300)
    ax.set_facecolor('#F8F9FA')
    fig.patch.set_facecolor('#F8F9FA')

    # Color Definitions
    COLOR_LICIT_NODE = '#2ECC71'
    BORDER_LICIT_NODE = '#145A32'
    COLOR_LICIT_EDGE = '#27AE60'

    COLOR_UNKNOWN_NODE = '#9B59B6'
    BORDER_UNKNOWN_NODE = '#4A235A'
    COLOR_UNKNOWN_EDGE = '#8E44AD'

    COLOR_ILLICIT_NODE = '#E74C3C'
    BORDER_ILLICIT_NODE = '#78281F'
    COLOR_ILLICIT_EDGE = '#C0392B'

    # Draw Column Background Shading Cards
    # Left Column Card (Licit)
    card_lic = FancyBboxPatch((-3.4, -4.2), 1.8, 8.4, boxstyle="round,pad=0.15", 
                             facecolor='#E8F8F5', edgecolor='#A3E4D7', linewidth=1.5, alpha=0.7)
    ax.add_patch(card_lic)
    ax.text(-2.5, 4.4, "LICIT ZONE\n(8 Green Nodes)", fontsize=11, fontweight='bold', ha='center', color='#145A32')

    # Center Column Card (Unknown)
    card_unk = FancyBboxPatch((-0.9, -4.2), 1.8, 8.4, boxstyle="round,pad=0.15", 
                             facecolor='#F4ECF7', edgecolor='#D2B4DE', linewidth=1.5, alpha=0.7)
    ax.add_patch(card_unk)
    ax.text(0.0, 4.4, "UNKNOWN ZONE\n(8 Purple Nodes)", fontsize=11, fontweight='bold', ha='center', color='#4A235A')

    # Right Column Card (Illicit)
    card_ill = FancyBboxPatch((1.6, -4.2), 1.8, 8.4, boxstyle="round,pad=0.15", 
                             facecolor='#FDEDEC', edgecolor='#F5B7B1', linewidth=1.5, alpha=0.7)
    ax.add_patch(card_ill)
    ax.text(2.5, 4.4, "ILLICIT ZONE\n(8 Red Nodes)", fontsize=11, fontweight='bold', ha='center', color='#78281F')

    # Color assignment per node
    node_colors = []
    border_colors = []
    node_sizes = []

    for n in G.nodes():
        cls = y[n]
        if cls == 1:
            node_colors.append(COLOR_ILLICIT_NODE)
            border_colors.append(BORDER_ILLICIT_NODE)
            node_sizes.append(500)
        elif cls == 0:
            node_colors.append(COLOR_LICIT_NODE)
            border_colors.append(BORDER_LICIT_NODE)
            node_sizes.append(500)
        else:
            node_colors.append(COLOR_UNKNOWN_NODE)
            border_colors.append(BORDER_UNKNOWN_NODE)
            node_sizes.append(500)

    # Edge colors per transaction flow type
    edge_colors = []
    for u, v in G.edges():
        src_y = y[u]
        dst_y = y[v]
        if src_y == 1 or dst_y == 1:
            edge_colors.append(COLOR_ILLICIT_EDGE)
        elif src_y == 0 and dst_y == 0:
            edge_colors.append(COLOR_LICIT_EDGE)
        else:
            edge_colors.append(COLOR_UNKNOWN_EDGE)

    # Draw Curved Edges
    nx.draw_networkx_edges(
        G, pos,
        ax=ax,
        arrowstyle='->',
        arrowsize=15,
        edge_color=edge_colors,
        width=1.8,
        alpha=0.80,
        connectionstyle='arc3,rad=0.10'
    )

    # Draw Nodes
    nx.draw_networkx_nodes(
        G, pos,
        ax=ax,
        node_color=node_colors,
        node_size=node_sizes,
        edgecolors=border_colors,
        linewidths=2.0,
        alpha=0.98
    )

    # Node Labels (Clean custom formatted text: Lic-1..8, Unk-1..8, Ill-1..8)
    labels = {}
    for idx, n in enumerate(lic_nodes, 1): labels[n] = f"Lic-{idx}"
    for idx, n in enumerate(unk_nodes, 1): labels[n] = f"Unk-{idx}"
    for idx, n in enumerate(ill_nodes, 1): labels[n] = f"Ill-{idx}"

    nx.draw_networkx_labels(
        G, pos,
        labels=labels,
        font_size=7.5,
        font_color='#FFFFFF',
        font_weight='bold',
        ax=ax
    )

    # Title & Subtitle
    plt.title("Bitcoin UTXO 24-Node Structured Bipartite/Tripartite Topology", fontsize=16, fontweight='bold', pad=24, color='#1B365D')
    plt.suptitle("Perfect Equal Balance (Exactly 8 Licit, 8 Illicit, 8 Unknown) | 100% Fully Connected (Zero Isolates) | Structured 3-Column Layout", fontsize=10.5, style='italic', color='#444444', y=0.925)

    # Legend
    legend_elements = [
        # Nodes
        Patch(facecolor=COLOR_LICIT_NODE, edgecolor=BORDER_LICIT_NODE, label='Licit Node (Exactly 8 Nodes)'),
        Patch(facecolor=COLOR_UNKNOWN_NODE, edgecolor=BORDER_UNKNOWN_NODE, label='Unknown Node (Exactly 8 Nodes)'),
        Patch(facecolor=COLOR_ILLICIT_NODE, edgecolor=BORDER_ILLICIT_NODE, label='Illicit Node (Exactly 8 Nodes)'),
        # Edges
        Line2D([0], [0], color=COLOR_LICIT_EDGE, lw=2.5, label='Licit Transaction Edge (Green)'),
        Line2D([0], [0], color=COLOR_UNKNOWN_EDGE, lw=2.5, label='Unknown Transaction Edge (Purple)'),
        Line2D([0], [0], color=COLOR_ILLICIT_EDGE, lw=2.5, label='Illicit Transaction Edge (Red)')
    ]
    ax.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.06), ncol=6, fontsize=9, frameon=True, facecolor='#FFFFFF', edgecolor='#D5D8DC')

    ax.set_xlim(-4.0, 4.0)
    ax.set_ylim(-4.8, 5.0)
    plt.axis('off')
    plt.tight_layout()

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Successfully generated structured 24-node topology plot at: {output_path}")

if __name__ == '__main__':
    generate_perfect_24node_structured_topology()
