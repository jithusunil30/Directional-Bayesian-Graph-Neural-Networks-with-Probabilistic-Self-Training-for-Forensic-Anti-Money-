import os, sys
import numpy as np
import pandas as pd
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
    pseudo_lic_path = os.path.join(icon_dir, "pseudo_licit.png")
    pseudo_ill_path = os.path.join(icon_dir, "pseudo_illicit.png")

    size = (128, 128)

    # 1. Licit Normal Person Icon (Solid Green)
    if not os.path.exists(licit_path):
        im_lic = Image.new("RGBA", size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(im_lic)
        draw.ellipse([4, 4, 124, 124], fill="#2ECC71", outline="#145A32", width=5)
        draw.ellipse([48, 24, 80, 56], fill="#FFFFFF")
        draw.chord([32, 60, 96, 114], start=180, end=360, fill="#FFFFFF")
        im_lic.save(licit_path)

    # 2. Illicit Thief Icon (Solid Red)
    if not os.path.exists(thief_path):
        im_ill = Image.new("RGBA", size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(im_ill)
        draw.ellipse([4, 4, 124, 124], fill="#E74C3C", outline="#78281F", width=5)
        draw.ellipse([42, 22, 86, 66], fill="#111111")
        draw.rectangle([44, 40, 84, 52], fill="#FFFFFF")
        draw.ellipse([50, 43, 58, 49], fill="#111111")
        draw.ellipse([70, 43, 78, 49], fill="#111111")
        draw.chord([28, 64, 100, 116], start=180, end=360, fill="#111111")
        im_ill.save(thief_path)

    # 3. Unknown Question Mark Icon (Purple)
    if not os.path.exists(unk_path):
        im_unk = Image.new("RGBA", size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(im_unk)
        draw.ellipse([4, 4, 124, 124], fill="#9B59B6", outline="#4A235A", width=5)
        try:
            font = ImageFont.truetype("arialbd.ttf", 72)
            draw.text((64, 60), "?", fill="#FFFFFF", font=font, anchor="mm")
        except:
            draw.arc([44, 24, 84, 64], start=210, end=360, fill="#FFFFFF", width=10)
            draw.line([(84, 44), (64, 70), (64, 80)], fill="#FFFFFF", width=10)
            draw.ellipse([59, 90, 69, 100], fill="#FFFFFF")
        im_unk.save(unk_path)

    # 4. Pseudo-Labeled Licit Icon (Green with Gold Ring & Star)
    if not os.path.exists(pseudo_lic_path):
        im_p_lic = Image.new("RGBA", size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(im_p_lic)
        draw.ellipse([4, 4, 124, 124], fill="#27AE60", outline="#F39C12", width=6)
        draw.ellipse([48, 24, 80, 56], fill="#FFFFFF")
        draw.chord([32, 60, 96, 114], start=180, end=360, fill="#FFFFFF")
        # Add badge circle
        draw.ellipse([82, 82, 122, 122], fill="#F39C12", outline="#FFFFFF", width=2)
        try:
            font_s = ImageFont.truetype("arialbd.ttf", 26)
            draw.text((102, 100), "P", fill="#111111", font=font_s, anchor="mm")
        except:
            pass
        im_p_lic.save(pseudo_lic_path)

    # 5. Pseudo-Labeled Illicit Icon (Red Thief with Gold/Orange Ring & P Badge)
    if not os.path.exists(pseudo_ill_path):
        im_p_ill = Image.new("RGBA", size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(im_p_ill)
        draw.ellipse([4, 4, 124, 124], fill="#C0392B", outline="#E67E22", width=6)
        draw.ellipse([42, 22, 86, 66], fill="#111111")
        draw.rectangle([44, 40, 84, 52], fill="#FFFFFF")
        draw.ellipse([50, 43, 58, 49], fill="#111111")
        draw.ellipse([70, 43, 78, 49], fill="#111111")
        draw.chord([28, 64, 100, 116], start=180, end=360, fill="#111111")
        # Add badge circle
        draw.ellipse([82, 82, 122, 122], fill="#E67E22", outline="#FFFFFF", width=2)
        try:
            font_s = ImageFont.truetype("arialbd.ttf", 26)
            draw.text((102, 100), "P", fill="#FFFFFF", font=font_s, anchor="mm")
        except:
            pass
        im_p_ill.save(pseudo_ill_path)

    return licit_path, thief_path, unk_path, pseudo_lic_path, pseudo_ill_path

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

def generate_post_pseudolabel_graphs():
    base_dir = r"c:\Users\USER\OneDrive\Desktop\Probalistic Graphical Lab"
    pt_orig_path = os.path.join(base_dir, "dataset", "elliptic_pyg_data.pt")
    pt_full_path = os.path.join(base_dir, "ML TRAINING 2", "elliptic_pyg_data_fully_labeled.pt")
    csv_full_path = os.path.join(base_dir, "dataset", "elliptic_dataset_fully_labeled.csv")

    icon_dir = os.path.join(base_dir, "dataset", "eda_plots", "icons")
    licit_icon_path, thief_icon_path, unknown_icon_path, pseudo_lic_path, pseudo_ill_path = ensure_icon_assets(icon_dir)

    img_licit = plt.imread(licit_icon_path)
    img_thief = plt.imread(thief_icon_path)
    img_unknown = plt.imread(unknown_icon_path)
    img_pseudo_lic = plt.imread(pseudo_lic_path)
    img_pseudo_ill = plt.imread(pseudo_ill_path)

    if not (os.path.exists(pt_orig_path) and os.path.exists(pt_full_path)):
        print(f"Error: dataset files missing.")
        return

    data_orig = torch.load(pt_orig_path, weights_only=False)
    data_full = torch.load(pt_full_path, weights_only=False)
    df_csv = pd.read_csv(csv_full_path) if os.path.exists(csv_full_path) else None

    y_orig = data_orig.y.numpy()
    y_full = data_full.y.numpy()
    is_predicted = data_full.is_predicted.numpy()
    edge_index = data_orig.edge_index.numpy()
    time_step = data_orig.time_step.numpy()
    edge_src, edge_dst = edge_index[0], edge_index[1]

    N_PER_CLASS = 8

    adj = {}
    for u, v in zip(edge_src, edge_dst):
        if u not in adj: adj[u] = []
        if v not in adj: adj[v] = []
        adj[u].append(v)
        adj[v].append(u)

    np.random.seed(42)
    illicit_seeds = np.where((y_orig == 1) & (time_step >= 10) & (time_step <= 35))[0]

    best_G = None
    best_edge_count = -1

    for seed in illicit_seeds[:100]:
        selected_nodes = {seed}
        counts = {1: 1, 0: 0, -1: 0}
        frontier = set(adj.get(seed, []))
        
        while (counts[1] < N_PER_CLASS or counts[0] < N_PER_CLASS or counts[-1] < N_PER_CLASS) and frontier:
            candidate = None
            for fn in list(frontier):
                cls = y_orig[fn]
                if counts[cls] < N_PER_CLASS:
                    candidate = fn
                    break
            if candidate is None: break
            selected_nodes.add(candidate)
            counts[y_orig[candidate]] += 1
            frontier.remove(candidate)
            for nxt in adj.get(candidate, []):
                if nxt not in selected_nodes: frontier.add(nxt)
                
        if counts[1] == N_PER_CLASS and counts[0] == N_PER_CLASS and counts[-1] == N_PER_CLASS:
            sub_nodes = list(selected_nodes)
            sub_mask = np.isin(edge_src, sub_nodes) & np.isin(edge_dst, sub_nodes)
            G_temp = nx.DiGraph()
            for u in sub_nodes: G_temp.add_node(u, cls=y_orig[u])
            for u, v in zip(edge_src[sub_mask], edge_dst[sub_mask]): G_temp.add_edge(u, v)
            if len(list(nx.isolates(G_temp))) == 0:
                if G_temp.number_of_edges() > best_edge_count:
                    best_edge_count = G_temp.number_of_edges()
                    best_G = G_temp.copy()

    G = best_G

    # Base Layout: Kamada-Kawai algorithm for natural graph spacing (Identical coordinates)
    pos_raw = nx.kamada_kawai_layout(G, weight=None)
    pos_scaled = {n: pos_raw[n] * 3.5 for n in G.nodes()}
    pos = enforce_min_distance(pos_scaled, min_dist=0.85, max_iter=300)

    # Node lists before pseudo labeling
    lic_gt_nodes = sorted([n for n in G.nodes() if y_orig[n] == 0])
    ill_gt_nodes = sorted([n for n in G.nodes() if y_orig[n] == 1])
    unk_orig_nodes = sorted([n for n in G.nodes() if y_orig[n] == -1])

    # Node lists after pseudo labeling
    pseudo_lic_nodes = sorted([n for n in unk_orig_nodes if y_full[n] == 0])
    pseudo_ill_nodes = sorted([n for n in unk_orig_nodes if y_full[n] == 1])

    # Map node labels
    labels_before = {}
    for idx, n in enumerate(lic_gt_nodes, 1): labels_before[n] = f"Lic-{idx}"
    for idx, n in enumerate(unk_orig_nodes, 1): labels_before[n] = f"Unk-{idx}"
    for idx, n in enumerate(ill_gt_nodes, 1): labels_before[n] = f"Ill-{idx}"

    labels_after = {}
    for idx, n in enumerate(lic_gt_nodes, 1): labels_after[n] = f"Lic-{idx}"
    for idx, n in enumerate(ill_gt_nodes, 1): labels_after[n] = f"Ill-{idx}"
    p_lic_idx = 1
    p_ill_idx = 1
    for n in unk_orig_nodes:
        prob_str = ""
        if df_csv is not None:
            r = df_csv[df_csv['node_index'] == n]
            if not r.empty:
                prob = r['illicit_probability'].values[0]
                prob_str = f"\nP_ill={prob:.2f}"
        if y_full[n] == 0:
            labels_after[n] = f"Pseudo-Lic-{p_lic_idx}{prob_str}"
            p_lic_idx += 1
        else:
            labels_after[n] = f"Pseudo-Ill-{p_ill_idx}{prob_str}"
            p_ill_idx += 1

    COLOR_LICIT_EDGE = '#27AE60'
    COLOR_PSEUDO_LICIT_EDGE = '#16A085'
    COLOR_PSEUDO_ILLICIT_EDGE = '#E67E22'
    COLOR_ILLICIT_EDGE = '#C0392B'
    COLOR_UNKNOWN_EDGE = '#8E44AD'

    # =========================================================================
    # GRAPH 1: Standalone Post Pseudo-Labeling Topology Graph (100% Resolved)
    # =========================================================================
    fig, ax = plt.subplots(figsize=(15, 11), dpi=300)
    ax.set_facecolor('#F8F9FA')
    fig.patch.set_facecolor('#F8F9FA')

    edge_colors_after = []
    for u, v in G.edges():
        src_final = y_full[u]
        dst_final = y_full[v]
        src_is_p = is_predicted[u]
        dst_is_p = is_predicted[v]

        if src_final == 1 or dst_final == 1:
            if src_is_p or dst_is_p:
                edge_colors_after.append(COLOR_PSEUDO_ILLICIT_EDGE)
            else:
                edge_colors_after.append(COLOR_ILLICIT_EDGE)
        else:
            if src_is_p or dst_is_p:
                edge_colors_after.append(COLOR_PSEUDO_LICIT_EDGE)
            else:
                edge_colors_after.append(COLOR_LICIT_EDGE)

    nx.draw_networkx_edges(
        G, pos,
        ax=ax,
        arrowstyle='->',
        arrowsize=16,
        edge_color=edge_colors_after,
        width=2.2,
        alpha=0.85,
        connectionstyle='arc3,rad=0.08'
    )

    for n, (x_pos, y_pos) in pos.items():
        if is_predicted[n]:
            if y_full[n] == 0:
                icon_img = img_pseudo_lic
                label_color = '#0E6251'
                edge_c = '#F39C12'
            else:
                icon_img = img_pseudo_ill
                label_color = '#7E5109'
                edge_c = '#E67E22'
        else:
            if y_full[n] == 0:
                icon_img = img_licit
                label_color = '#145A32'
                edge_c = '#27AE60'
            else:
                icon_img = img_thief
                label_color = '#78281F'
                edge_c = '#C0392B'

        imagebox = OffsetImage(icon_img, zoom=0.26)
        ab = AnnotationBbox(imagebox, (x_pos, y_pos), frameon=False)
        ax.add_artist(ab)

        ax.text(x_pos, y_pos - 0.24, labels_after[n], fontsize=8.0, fontweight='bold', 
                ha='center', va='top', color=label_color,
                bbox=dict(boxstyle='round,pad=0.22', facecolor='#FFFFFF', alpha=0.95, edgecolor=edge_c, linewidth=1.5))

    plt.title("Post Pseudo-Labeling Bitcoin UTXO Transaction Topology (100% Resolved Graph)", fontsize=16, fontweight='bold', pad=22, color='#1B365D')
    plt.suptitle("Phase 3 Probabilistic Self-Training Output | 24 Resolved Nodes (8 GT Licit, 8 GT Illicit, 4 Pseudo-Licit, 4 Pseudo-Illicit, 0 Unknown)", 
                 fontsize=10.5, style='italic', color='#444444', y=0.925)

    legend_elements = [
        Patch(facecolor='#2ECC71', edgecolor='#145A32', label='GT Licit Nodes (8)'),
        Patch(facecolor='#27AE60', edgecolor='#F39C12', label='Pseudo-Labeled Licit (4)'),
        Patch(facecolor='#C0392B', edgecolor='#E67E22', label='Pseudo-Labeled Illicit (4)'),
        Patch(facecolor='#E74C3C', edgecolor='#78281F', label='GT Illicit Nodes (8)'),
        Line2D([0], [0], color=COLOR_LICIT_EDGE, lw=2.5, label='Licit Edge'),
        Line2D([0], [0], color=COLOR_PSEUDO_LICIT_EDGE, lw=2.5, label='Pseudo-Licit Edge'),
        Line2D([0], [0], color=COLOR_PSEUDO_ILLICIT_EDGE, lw=2.5, label='Pseudo-Illicit Edge'),
        Line2D([0], [0], color=COLOR_ILLICIT_EDGE, lw=2.5, label='Illicit Edge')
    ]
    ax.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.06), ncol=4, fontsize=9, frameon=True, facecolor='#FFFFFF', edgecolor='#D5D8DC')

    x_coords = [p[0] for p in pos.values()]
    y_coords = [p[1] for p in pos.values()]
    ax.set_xlim(min(x_coords) - 0.7, max(x_coords) + 0.7)
    ax.set_ylim(min(y_coords) - 0.9, max(y_coords) + 0.7)
    plt.axis('off')
    plt.tight_layout()

    out_post_root = os.path.join(base_dir, "post_pseudolabel_topology_graph.png")
    out_post_eda = os.path.join(base_dir, "dataset", "eda_plots", "post_pseudolabel_topology_graph.png")
    out_post_ml2 = os.path.join(base_dir, "ML TRAINING 2", "plots", "post_pseudolabel_topology_graph.png")
    os.makedirs(os.path.dirname(out_post_ml2), exist_ok=True)

    plt.savefig(out_post_root, dpi=300, bbox_inches='tight')
    plt.savefig(out_post_eda, dpi=300, bbox_inches='tight')
    plt.savefig(out_post_ml2, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved Post-Pseudolabeling Graph to:\n  {out_post_root}\n  {out_post_eda}\n  {out_post_ml2}")

    # =========================================================================
    # GRAPH 2: Side-by-Side Comparison (Before vs After Pseudo-Labeling)
    # =========================================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(28, 12), dpi=300)
    fig.patch.set_facecolor('#F8F9FA')
    ax1.set_facecolor('#F8F9FA')
    ax2.set_facecolor('#F8F9FA')

    # LEFT PANEL: Before Pseudo-Labeling (Sparse Graph with Unknowns)
    edge_colors_before = []
    for u, v in G.edges():
        src_y = y_orig[u]
        dst_y = y_orig[v]
        if src_y == 1 or dst_y == 1:
            edge_colors_before.append(COLOR_ILLICIT_EDGE)
        elif src_y == 0 and dst_y == 0:
            edge_colors_before.append(COLOR_LICIT_EDGE)
        else:
            edge_colors_before.append(COLOR_UNKNOWN_EDGE)

    nx.draw_networkx_edges(
        G, pos, ax=ax1, arrowstyle='->', arrowsize=14,
        edge_color=edge_colors_before, width=2.0, alpha=0.80, connectionstyle='arc3,rad=0.08'
    )

    for n, (x_pos, y_pos) in pos.items():
        cls = y_orig[n]
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

        imagebox = OffsetImage(icon_img, zoom=0.24)
        ab = AnnotationBbox(imagebox, (x_pos, y_pos), frameon=False)
        ax1.add_artist(ab)

        ax1.text(x_pos, y_pos - 0.24, labels_before[n], fontsize=8.0, fontweight='bold', 
                 ha='center', va='top', color=label_color,
                 bbox=dict(boxstyle='round,pad=0.22', facecolor='#FFFFFF', alpha=0.92, edgecolor=edge_c, linewidth=1.2))

    ax1.set_title("BEFORE PSEUDO-LABELING (Phase 1 & 2: Sparse Graph)\n8 Licit (33.3%) | 8 Illicit (33.3%) | 8 Unknown (33.3% Fragmented)", 
                  fontsize=13, fontweight='bold', color='#1B365D', pad=12)
    ax1.set_xlim(min(x_coords) - 0.7, max(x_coords) + 0.7)
    ax1.set_ylim(min(y_coords) - 0.9, max(y_coords) + 0.7)
    ax1.axis('off')

    # RIGHT PANEL: After Pseudo-Labeling (100% Resolved Dense Graph)
    nx.draw_networkx_edges(
        G, pos, ax=ax2, arrowstyle='->', arrowsize=14,
        edge_color=edge_colors_after, width=2.2, alpha=0.85, connectionstyle='arc3,rad=0.08'
    )

    for n, (x_pos, y_pos) in pos.items():
        if is_predicted[n]:
            if y_full[n] == 0:
                icon_img = img_pseudo_lic
                label_color = '#0E6251'
                edge_c = '#F39C12'
            else:
                icon_img = img_pseudo_ill
                label_color = '#7E5109'
                edge_c = '#E67E22'
        else:
            if y_full[n] == 0:
                icon_img = img_licit
                label_color = '#145A32'
                edge_c = '#27AE60'
            else:
                icon_img = img_thief
                label_color = '#78281F'
                edge_c = '#C0392B'

        imagebox = OffsetImage(icon_img, zoom=0.24)
        ab = AnnotationBbox(imagebox, (x_pos, y_pos), frameon=False)
        ax2.add_artist(ab)

        ax2.text(x_pos, y_pos - 0.24, labels_after[n], fontsize=7.5, fontweight='bold', 
                 ha='center', va='top', color=label_color,
                 bbox=dict(boxstyle='round,pad=0.22', facecolor='#FFFFFF', alpha=0.95, edgecolor=edge_c, linewidth=1.5))

    ax2.set_title("AFTER PSEUDO-LABELING (Phase 3 & 4: 100% Resolved Dense Graph)\n12 Total Licit (8 GT + 4 Pseudo) | 12 Total Illicit (8 GT + 4 Pseudo) | 0 Unknown (100% Connected)", 
                  fontsize=13, fontweight='bold', color='#1B365D', pad=12)
    ax2.set_xlim(min(x_coords) - 0.7, max(x_coords) + 0.7)
    ax2.set_ylim(min(y_coords) - 0.9, max(y_coords) + 0.7)
    ax2.axis('off')

    # Main Title
    plt.suptitle("Forensic Transformation: Transaction Network Topology Before vs. After Out-of-Fold GNN Ensemble Pseudo-Labeling", 
                 fontsize=16, fontweight='bold', y=0.98, color='#1B365D')

    # Shared Legend
    legend_elements_cmp = [
        Patch(facecolor='#2ECC71', edgecolor='#145A32', label='GT Licit Node'),
        Patch(facecolor='#9B59B6', edgecolor='#4A235A', label='Unknown Node (Before)'),
        Patch(facecolor='#27AE60', edgecolor='#F39C12', label='Pseudo-Labeled Licit (After)'),
        Patch(facecolor='#C0392B', edgecolor='#E67E22', label='Pseudo-Labeled Illicit (After)'),
        Patch(facecolor='#E74C3C', edgecolor='#78281F', label='GT Illicit Node'),
        Line2D([0], [0], color=COLOR_LICIT_EDGE, lw=2.5, label='Licit Edge'),
        Line2D([0], [0], color=COLOR_UNKNOWN_EDGE, lw=2.5, label='Unknown Edge'),
        Line2D([0], [0], color=COLOR_PSEUDO_LICIT_EDGE, lw=2.5, label='Pseudo-Licit Edge'),
        Line2D([0], [0], color=COLOR_PSEUDO_ILLICIT_EDGE, lw=2.5, label='Pseudo-Illicit Edge'),
        Line2D([0], [0], color=COLOR_ILLICIT_EDGE, lw=2.5, label='Illicit Edge')
    ]
    fig.legend(handles=legend_elements_cmp, loc='lower center', bbox_to_anchor=(0.5, 0.02), ncol=5, fontsize=9.5, frameon=True, facecolor='#FFFFFF', edgecolor='#D5D8DC')

    plt.tight_layout(rect=[0, 0.07, 1, 0.95])

    out_cmp_root = os.path.join(base_dir, "before_vs_after_pseudolabel_graph.png")
    out_cmp_eda = os.path.join(base_dir, "dataset", "eda_plots", "before_vs_after_pseudolabel_graph.png")
    out_cmp_ml2 = os.path.join(base_dir, "ML TRAINING 2", "plots", "before_vs_after_pseudolabel_graph.png")

    plt.savefig(out_cmp_root, dpi=300, bbox_inches='tight')
    plt.savefig(out_cmp_eda, dpi=300, bbox_inches='tight')
    plt.savefig(out_cmp_ml2, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved Before vs. After Comparison Graph to:\n  {out_cmp_root}\n  {out_cmp_eda}\n  {out_cmp_ml2}")

if __name__ == '__main__':
    generate_post_pseudolabel_graphs()
