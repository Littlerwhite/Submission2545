"""Figure: 4 attack graph topologies (2x2, double-column)."""
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

graphs = {
    "G10": {
        "edges": [(0,1),(0,2),(0,3),(0,4),(1,5),(1,6),(2,6),(2,7),
                  (3,7),(3,8),(4,8),(4,9),(5,9),(6,9),(7,9),(8,9)],
        "pos": {0:(0,0), 1:(-3,1), 2:(-1,1), 3:(1,1), 4:(3,1),
                5:(-3,2), 6:(-1,2), 7:(1,2), 8:(3,2), 9:(0,3)},
    },
    "G100-shallow (4 layers)": {"edges": [(0,1),(0,2),(0,3),(0,4)], "layers": 4},
    "G100-complex (6 layers)": {"edges": [(0,1),(0,2),(0,3),(0,4)], "layers": 6},
    "G100-deep (8 layers)":    {"edges": [(0,1),(0,2),(0,3),(0,4)], "layers": 8},
}

fig, axes = plt.subplots(2, 2, figsize=(12, 9), dpi=150)
axes = axes.flatten()

# ============ (a) G10 ============
G = nx.DiGraph()
G.add_edges_from(graphs["G10"]["edges"])
pos = graphs["G10"]["pos"]
colors = ['#4472C4' if n not in [0,9] else ('#70AD47' if n==0 else '#C00000')
          for n in G.nodes()]
nx.draw(G, pos, ax=axes[0], with_labels=True, node_color=colors,
        node_size=400, font_size=9, font_color='white',
        arrowsize=12, edge_color='gray', width=1.2)
axes[0].set_title("(a) $G_{10}$", fontsize=13, pad=8)
axes[0].axis('off')
axes[0].margins(0.05)

# ============ (b-d) 100-node graphs ============
for idx, (name, cfg) in enumerate(list(graphs.items())[1:], start=1):
    ax = axes[idx]
    layers = cfg["layers"]
    n_per_layer = 25 if layers == 4 else (16 if layers == 6 else 12)

    for l in range(layers):
        y = -l
        x_start = -n_per_layer / 2
        # 节点
        for i in range(n_per_layer):
            color = ('#70AD47' if l == 0 else
                     '#C00000' if l == layers - 1 else '#4472C4')
            ax.scatter(x_start + i, y, s=22, color=color, zorder=3,
                       edgecolor='white', linewidth=0.4)
        # 层间连线（加粗）
        if l < layers - 1:
            for i in range(n_per_layer):
                for offset in [-1, 0, 1]:
                    j = i + offset
                    if 0 <= j < n_per_layer:
                        ax.plot([x_start + i, -n_per_layer / 2 + j],
                                [y - 0.1, -l - 1 + 0.1],
                                color='gray', lw=1.0, alpha=0.7, zorder=1)

    ax.set_title(f"({chr(98+idx-1)}) {name}", fontsize=12, pad=8)
    ax.axis('off')
    # 收紧边缘留白
    ax.set_xlim(-n_per_layer / 2 - 0.3, n_per_layer / 2 + 0.3)
    ax.margins(0.02)

# 边缘留白小，子图之间留白大
plt.tight_layout(pad=0.4, w_pad=4.0, h_pad=3.5)

plt.savefig("paper/figures/figure_topologies.pdf", bbox_inches='tight')
plt.savefig("paper/figures/figure_topologies.png", bbox_inches='tight', dpi=300)
print("已保存 figure_topologies.pdf")