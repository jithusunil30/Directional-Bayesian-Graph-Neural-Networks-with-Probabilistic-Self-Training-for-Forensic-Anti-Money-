import os, sys
import numpy as np
import torch
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

def generate_transaction_network_plot():
    base_dir = r"c:\Users\USER\OneDrive\Desktop\Probalistic Graphical Lab"
    pt_path = os.path.join(base_dir, "dataset", "elliptic_pyg_data.pt")
    output_path = os.path.join(base_dir, "dataset", "eda_plots", "transaction_network_graph.png")

    if not os.path.exists(pt_path):
        print(f"Error: {pt_path} not found.")
        return

    data = torch.load(pt_path, weights_only=False)
    y = data.y.numpy()
    edge_index = data.edge_index.numpy()
    time_step = data.time_step.numpy()

    # Find a connected multi-hop subgraph containing illicit, licit, and unknown nodes
    np.random.seed(42)
    illicit_indices = np.where((y == 1) & (time_step == 20))[0]
    if len(illicit_indices) == 0:
        illicit_indices = np.where(y == 1)[0]
        
    seed_nodes = illicit_indices[:6]
    
    selected_nodes = set(seed_nodes)
    edge_src, edge_dst = edge_index[0], edge_index[1]

    # Collect 1-hop neighbors
    mask_1 = np.isin(edge_src, list(selected_nodes)) | np.isin(edge_dst, list(selected_nodes))
    neighbors_1 = list(set(edge_src[mask_1]).union(set(edge_dst[mask_1])))
    selected_nodes.update(neighbors_1[:35])

    # Collect 2-hop neighbors
    mask_2 = np.isin(edge_src, list(selected_nodes)) | np.isin(edge_dst, list(selected_nodes))
    neighbors_2 = list(set(edge_src[mask_2]).union(set(edge_dst[mask_2])))
    selected_nodes.update(neighbors_2[:55])

    # Extract all directed edges between the selected node set
    mask_sub = np.isin(edge_src, list(selected_nodes)) & np.isin(edge_dst, list(selected_nodes))
    sub_src = edge_src[mask_sub]
    sub_dst = edge_dst[mask_sub]

    G = nx.DiGraph()
    for n in selected_nodes:
        G.add_node(n, label=y[n])
    for u, v in zip(sub_src, sub_dst):
        G.add_edge(u, v)

    # Remove isolated singletons to keep the graph compact and dense
    G.remove_nodes_from(list(nx.isolates(G)))
    
    print(f"Graph Extracted: {G.number_of_nodes()} Nodes, {G.number_of_edges()} Directed Edges")

    # Compute spring layout
    pos = nx.spring_layout(G, k=0.45, seed=42, iterations=120)

    fig, ax = plt.subplots(figsize=(13, 9), dpi=300)
    ax.set_facecolor('#F8F9FA')
    fig.patch.set_facecolor('#F8F9FA')

    # Assign node properties
    node_colors = []
    node_sizes = []
    border_colors = []

    for n in G.nodes():
        cls = y[n]
        if cls == 1: # Illicit
            node_colors.append('#E74C3C') # Crimson Red
            node_sizes.append(340)
            border_colors.append('#78281F')
        elif cls == 0: # Licit
            node_colors.append('#2ECC71') # Emerald Green
            node_sizes.append(220)
            border_colors.append('#145A32')
        else: # Unknown (-1)
            node_colors.append('#95A5A6') # Slate Gray
            node_sizes.append(180)
            border_colors.append('#34495E')

    # Draw Edges with Curved Arrows
    nx.draw_networkx_edges(
        G, pos,
        ax=ax,
        arrowstyle='->',
        arrowsize=14,
        edge_color='#BDC3C7',
        width=1.3,
        alpha=0.75,
        connectionstyle='arc3,rad=0.06'
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

    # Node ID labels for Illicit Nodes to highlight laundering hubs
    illicit_labels = {n: f"Ill-{n}" for n in G.nodes() if y[n] == 1}
    nx.draw_networkx_labels(
        G, pos,
        labels=illicit_labels,
        font_size=7.5,
        font_color='#5B2C6F',
        font_weight='bold',
        ax=ax
    )

    # Title & Subtitle
    plt.title("Bitcoin UTXO Transaction Topology & Relational Network Subgraph", fontsize=15, fontweight='bold', pad=15, color='#1B365D')
    plt.suptitle("Direct Visualization of Payment Connections Between Licit (Green), Illicit (Red), and Unlabeled Unknown (Gray) Transaction Nodes", fontsize=10, style='italic', color='#555555', y=0.925)

    # Custom Legend
    legend_elements = [
        Patch(facecolor='#E74C3C', edgecolor='#78281F', label='Illicit Transaction (Class 1)'),
        Patch(facecolor='#2ECC71', edgecolor='#145A32', label='Licit Transaction (Class 0)'),
        Patch(facecolor='#95A5A6', edgecolor='#34495E', label='Unlabeled / Unknown Node (Class -1)'),
        Line2D([0], [0], color='#BDC3C7', lw=2, label='Directed UTXO Payment Flow (u -> v)')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9.5, frameon=True, facecolor='#FFFFFF', edgecolor='#D5D8DC')

    plt.axis('off')
    plt.tight_layout()

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Successfully generated graph plot at: {output_path}")

if __name__ == '__main__':
    generate_transaction_network_plot()
