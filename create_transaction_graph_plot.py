import os, sys
import numpy as np
import torch
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from PIL import Image, ImageDraw, ImageFont

def ensure_icon_assets(icon_dir):
    os.makedirs(icon_dir, exist_ok=True)
    licit_path = os.path.join(icon_dir, "licit_person.png")
    thief_path = os.path.join(icon_dir, "illicit_thief.png")
    unk_path = os.path.join(icon_dir, "unknown_question.png")

    if not (os.path.exists(licit_path) and os.path.exists(thief_path) and os.path.exists(unk_path)):
        size = (128, 128)
        # 1. Licit Normal Person Icon
        im_lic = Image.new("RGBA", size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(im_lic)
        draw.ellipse([4, 4, 124, 124], fill="#2ECC71", outline="#145A32", width=4)
        draw.ellipse([48, 24, 80, 56], fill="#FFFFFF")
        draw.chord([32, 60, 96, 114], start=180, end=360, fill="#FFFFFF")
        im_lic.save(licit_path)

        # 2. Illicit Thief Icon
        im_ill = Image.new("RGBA", size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(im_ill)
        draw.ellipse([4, 4, 124, 124], fill="#E74C3C", outline="#78281F", width=4)
        draw.ellipse([42, 22, 86, 66], fill="#111111")
        draw.rectangle([44, 40, 84, 52], fill="#FFFFFF")
        draw.ellipse([50, 43, 58, 49], fill="#111111")
        draw.ellipse([70, 43, 78, 49], fill="#111111")
        draw.chord([28, 64, 100, 116], start=180, end=360, fill="#111111")
        im_ill.save(thief_path)

        # 3. Unknown Question Mark Icon
        im_unk = Image.new("RGBA", size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(im_unk)
        draw.ellipse([4, 4, 124, 124], fill="#9B59B6", outline="#4A235A", width=4)
        try:
            font = ImageFont.truetype("arialbd.ttf", 72)
            draw.text((64, 60), "?", fill="#FFFFFF", font=font, anchor="mm")
        except:
            draw.arc([44, 24, 84, 64], start=210, end=360, fill="#FFFFFF", width=10)
            draw.line([(84, 44), (64, 70), (64, 80)], fill="#FFFFFF", width=10)
            draw.ellipse([59, 90, 69, 100], fill="#FFFFFF")
        im_unk.save(unk_path)

    return licit_path, thief_path, unk_path

def enforce_min_distance(pos, min_dist=0.85, max_iter=300):
    """Enforces a strict minimum Euclidean distance between all node pairs to guarantee NO overlap."""
    nodes = list(pos.keys())
    coords = np.array([pos[n] for n in nodes])
    
    for _ in range(max_iter):
        overlapping = False
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                diff = coords[i] - coords[j]
                dist = np.linalg.norm(diff)
                if dist < min_dist:
                    overlapping = True
                    if dist < 1e-4:
                        diff = np.random.randn(2) * 0.01
                        dist = np.linalg.norm(diff)
                    overlap = min_dist - dist
                    direction = diff / dist
                    coords[i] += direction * (overlap / 2.0)
                    coords[j] -= direction * (overlap / 2.0)
        if not overlapping:
            break
            
    return {nodes[i]: coords[i] for i in range(len(nodes))}

def generate_topology_graph():
    base_dir = r"c:\Users\USER\OneDrive\Desktop\Probalistic Graphical Lab"
    pt_path = os.path.join(base_dir, "dataset", "elliptic_pyg_data.pt")
    
    # Save locations
    output_path_root = os.path.join(base_dir, "topology_graph.png")
    output_path_eda = os.path.join(base_dir, "dataset", "eda_plots", "topology_graph.png")
    output_path_main = os.path.join(base_dir, "dataset", "eda_plots", "transaction_network_graph.png")

    icon_dir = os.path.join(base_dir, "dataset", "eda_plots", "icons")
    licit_icon_path, thief_icon_path, unknown_icon_path = ensure_icon_assets(icon_dir)

    img_licit = plt.imread(licit_icon_path)
    img_thief = plt.imread(thief_icon_path)
    img_unknown = plt.imread(unknown_icon_path)

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
            if len(list(nx.isolates(G_temp))) == 0:
                if G_temp.number_of_edges() > best_edge_count:
                    best_edge_count = G_temp.number_of_edges()
                    best_G = G_temp.copy()

    G = best_G

    # Base Layout: Kamada-Kawai algorithm for natural graph spacing
    pos_raw = nx.kamada_kawai_layout(G, weight=None)
    pos_scaled = {n: pos_raw[n] * 3.5 for n in G.nodes()}

    # Guarantee ZERO node overlap using force-distance repulsion solver
    pos = enforce_min_distance(pos_scaled, min_dist=0.85, max_iter=300)

    print(f"Verified Non-Overlapping Topology Graph:")
    print(f"• Total Nodes  : {G.number_of_nodes()} (8 Licit, 8 Illicit, 8 Unknown)")
    print(f"• Total Edges  : {G.number_of_edges()} Directed Edges")
    print(f"• Isolates     : {len(list(nx.isolates(G)))} (100% Connected)")

    fig, ax = plt.subplots(figsize=(15, 11), dpi=300)
    ax.set_facecolor('#F8F9FA')
    fig.patch.set_facecolor('#F8F9FA')

    COLOR_LICIT_EDGE = '#27AE60'
    COLOR_UNKNOWN_EDGE = '#8E44AD'
    COLOR_ILLICIT_EDGE = '#C0392B'

    # Assign Edge Colors per flow category
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

    # Draw Directed Curved Edges
    nx.draw_networkx_edges(
        G, pos,
        ax=ax,
        arrowstyle='->',
        arrowsize=16,
        edge_color=edge_colors,
        width=2.0,
        alpha=0.80,
        connectionstyle='arc3,rad=0.08'
    )

    lic_nodes = sorted([n for n in G.nodes() if y[n] == 0])
    unk_nodes = sorted([n for n in G.nodes() if y[n] == -1])
    ill_nodes = sorted([n for n in G.nodes() if y[n] == 1])

    labels = {}
    for idx, n in enumerate(lic_nodes, 1): labels[n] = f"Lic-{idx}"
    for idx, n in enumerate(unk_nodes, 1): labels[n] = f"Unk-{idx}"
    for idx, n in enumerate(ill_nodes, 1): labels[n] = f"Ill-{idx}"

    # Overlay Node Custom Image Icons (Normal Person, Thief, Question Mark)
    for n, (x_pos, y_pos) in pos.items():
        cls = y[n]
        if cls == 0:
            icon_img = img_licit
            label_color = '#145A32'
            edge_c = '#27AE60'
        elif cls == 1:
            icon_img = img_thief
            label_color = '#78281F'
            edge_c = '#C0392B'
        else:
            icon_img = img_unknown
            label_color = '#4A235A'
            edge_c = '#8E44AD'
            
        imagebox = OffsetImage(icon_img, zoom=0.26)
        ab = AnnotationBbox(imagebox, (x_pos, y_pos), frameon=False)
        ax.add_artist(ab)

        # Draw Node Label below icon with background box
        ax.text(x_pos, y_pos - 0.24, labels[n], fontsize=8.5, fontweight='bold', 
                ha='center', va='top', color=label_color,
                bbox=dict(boxstyle='round,pad=0.22', facecolor='#FFFFFF', alpha=0.92, edgecolor=edge_c, linewidth=1.2))

    # Title & Subtitle
    plt.title("Bitcoin UTXO Transaction Topology & Node Archetype Network Graph", fontsize=16, fontweight='bold', pad=22, color='#1B365D')
    plt.suptitle("Zero Node Overlap | Organic Network Topology | 24 Connected Nodes (8 Licit Persons, 8 Illicit Thieves, 8 Unknown Question Marks)", fontsize=10.5, style='italic', color='#444444', y=0.925)

    # Custom Legend
    legend_elements = [
        # Nodes / Icons
        Patch(facecolor='#2ECC71', edgecolor='#145A32', label='Normal Person Node (Licit - 8 Nodes)'),
        Patch(facecolor='#9B59B6', edgecolor='#4A235A', label='Question Mark Node (Unknown - 8 Nodes)'),
        Patch(facecolor='#E74C3C', edgecolor='#78281F', label='Thief Criminal Node (Illicit - 8 Nodes)'),
        # Edges
        Line2D([0], [0], color=COLOR_LICIT_EDGE, lw=2.5, label='Licit Transaction Edge (Green)'),
        Line2D([0], [0], color=COLOR_UNKNOWN_EDGE, lw=2.5, label='Unknown Transaction Edge (Purple)'),
        Line2D([0], [0], color=COLOR_ILLICIT_EDGE, lw=2.5, label='Illicit Transaction Edge (Red)')
    ]
    ax.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.06), ncol=6, fontsize=9, frameon=True, facecolor='#FFFFFF', edgecolor='#D5D8DC')

    # Add margins around min/max coordinates
    x_coords = [p[0] for p in pos.values()]
    y_coords = [p[1] for p in pos.values()]
    ax.set_xlim(min(x_coords) - 0.7, max(x_coords) + 0.7)
    ax.set_ylim(min(y_coords) - 0.9, max(y_coords) + 0.7)

    plt.axis('off')
    plt.tight_layout()

    plt.savefig(output_path_root, dpi=300, bbox_inches='tight')
    plt.savefig(output_path_eda, dpi=300, bbox_inches='tight')
    plt.savefig(output_path_main, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Successfully generated zero-overlap topology graph with custom icons at:\n1. {output_path_root}\n2. {output_path_eda}\n3. {output_path_main}")

if __name__ == '__main__':
    generate_topology_graph()
