import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def create_cross_phase_plot():
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2), dpi=300)
    
    models = ['GCN', 'GAT', 'GraphSAGE', 'GIN', 'Bayesian SAGE', 'Bayesian GNN', 'Dir-ResGCN', 'Dir-ResSAGE']
    x_indices = list(range(len(models)))
    
    # Phase 2 (Sparse, Ground-Truth 16,670 nodes) - Indices 0 to 5
    x_p2 = [0, 1, 2, 3, 4, 5]
    f1_p2 = [0.102, 0.060, 0.437, 0.172, 0.474, 0.495]
    pr_p2 = [0.174, 0.160, 0.282, 0.165, 0.326, 0.348]
    tp_p2 = [83, 41, 608, 172, 541, 561]
    
    # Phase 4 (Dense 100% Graph, 67,504 nodes) - Indices 0 to 5
    x_p4 = [0, 1, 2, 3, 4, 5]
    f1_p4 = [0.725, 0.692, 0.739, 0.702, 0.768, 0.776]
    pr_p4 = [0.700, 0.726, 0.755, 0.587, 0.798, 0.813]
    tp_p4 = [14704, 14670, 15069, 14678, 14600, 14802]
    
    # Phase 4 Ground-Truth Eval Trendline - Indices 0 to 5
    f1_p4_gt = [0.754, 0.742, 0.768, 0.710, 0.765, 0.768]
    pr_p4_gt = [0.789, 0.778, 0.807, 0.734, 0.787, 0.807]
    tp_p4_gt = [14862, 14988, 14504, 14345, 14905, 14504]
    
    # Phase 5 (State-of-the-Art Directional Residual GNNs) - Indices 5, 6, 7
    x_p5 = [5, 6, 7]
    f1_p5 = [0.767, 0.754, 0.743]
    pr_p5 = [0.807, 0.788, 0.779]
    tp_p5 = [14504, 14862, 14988]
    
    # Colors & Styles
    color_p2 = '#E74C3C'  # Coral Red
    color_p4 = '#F39C12'  # Warm Orange
    color_p4_gt = '#A3E4D7' # Mint Light Green
    color_p5 = '#2ECC71'  # Emerald Green
    
    # --- Subplot 1: F1-Score ---
    ax1 = axes[0]
    ax1.plot(x_p4, f1_p4_gt, color=color_p4_gt, linestyle='-', linewidth=1.8, label='Phase 4 (Dense GT)')
    ax1.plot(x_p2, f1_p2, color=color_p2, marker='o', linestyle='-', linewidth=2, markersize=8, label='Phase 2 (Sparse)')
    ax1.plot(x_p4, f1_p4, color=color_p4, marker='s', linestyle='-', linewidth=2, markersize=8, label='Phase 4 (Dense)')
    ax1.plot(x_p5, f1_p5, color=color_p5, marker='^', linestyle='-', linewidth=2.5, markersize=10, label='Phase 5 (Dir-Res)')
    
    ax1.set_title('F1-Score', fontsize=13, weight='bold', pad=10)
    ax1.set_ylabel('F1-Score', fontsize=11, weight='bold')
    ax1.set_xticks(x_indices)
    ax1.set_xticklabels(models, rotation=28, ha='right', fontsize=9.5)
    ax1.set_ylim(0.0, 0.82)
    ax1.grid(True, linestyle=':', alpha=0.5)
    ax1.legend(loc='lower right', fontsize=8.5, framealpha=0.9)

    # --- Subplot 2: PR-AUC ---
    ax2 = axes[1]
    ax2.plot(x_p4, pr_p4_gt, color=color_p4_gt, linestyle='-', linewidth=1.8, label='Phase 4 (Dense GT)')
    ax2.plot(x_p2, pr_p2, color=color_p2, marker='o', linestyle='-', linewidth=2, markersize=8, label='Phase 2 (Sparse)')
    ax2.plot(x_p4, pr_p4, color=color_p4, marker='s', linestyle='-', linewidth=2, markersize=8, label='Phase 4 (Dense)')
    ax2.plot(x_p5, pr_p5, color=color_p5, marker='^', linestyle='-', linewidth=2.5, markersize=10, label='Phase 5 (Dir-Res)')
    
    ax2.set_title('PR-AUC', fontsize=13, weight='bold', pad=10)
    ax2.set_ylabel('PR-AUC', fontsize=11, weight='bold')
    ax2.set_xticks(x_indices)
    ax2.set_xticklabels(models, rotation=28, ha='right', fontsize=9.5)
    ax2.set_ylim(0.1, 0.85)
    ax2.grid(True, linestyle=':', alpha=0.5)
    ax2.legend(loc='lower right', fontsize=8.5, framealpha=0.9)

    # --- Subplot 3: Criminals Caught (TP) ---
    ax3 = axes[2]
    ax3.plot(x_p4, tp_p4_gt, color=color_p4_gt, linestyle='-', linewidth=1.8, label='Phase 4 (Dense GT)')
    ax3.plot(x_p2, tp_p2, color=color_p2, marker='o', linestyle='-', linewidth=2, markersize=8, label='Phase 2 (Sparse)')
    ax3.plot(x_p4, tp_p4, color=color_p4, marker='s', linestyle='-', linewidth=2, markersize=8, label='Phase 4 (Dense)')
    ax3.plot(x_p5, tp_p5, color=color_p5, marker='^', linestyle='-', linewidth=2.5, markersize=10, label='Phase 5 (Dir-Res)')
    
    ax3.set_title('Criminals Caught (TP)', fontsize=13, weight='bold', pad=10)
    ax3.set_ylabel('Criminals Caught (TP)', fontsize=11, weight='bold')
    ax3.set_xticks(x_indices)
    ax3.set_xticklabels(models, rotation=28, ha='right', fontsize=9.5)
    ax3.set_ylim(-500, 16200)
    ax3.grid(True, linestyle=':', alpha=0.5)
    ax3.legend(loc='lower right', fontsize=8.5, framealpha=0.9)

    fig.suptitle('Cross-Phase ML Metrics Comparison (Published Paper Results)', fontsize=15, weight='bold', y=0.98, color='#1B365D')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    out_path = "nb_cross_phase.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"nb_cross_phase.png updated successfully with connected Phase 5 lines at: {os.path.abspath(out_path)}")

if __name__ == '__main__':
    create_cross_phase_plot()
