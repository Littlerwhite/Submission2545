"""Figure: 4 attack graph topologies with structural annotations."""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np
import pyspiel
import open_spiel.python.games.attack_graph_scalable

# ============================================================
# 颜色配置
# ============================================================
C_ENTRY   = '#2ca02c'   # 绿色：入口节点
C_TERM    = '#d62728'   # 红色：终点/关键资产
C_MID     = '#1f77b4'   # 蓝色：中间节点
C_CUT     = '#ff7f0e'   # 橙色：最小割节点
C_EDGE    = '#aaaaaa'   # 灰色：边

# ============================================================
# (a) G10：精确绘制
# ============================================================
def draw_g10(ax):
    edges = {
        0: [1, 2, 3, 4],
        1: [5, 6], 2: [6, 7], 3: [7, 8], 4: [8, 9],
        5: [9], 6: [9], 7: [9], 8: [9], 9: [],
    }
    G = nx.DiGraph()
    for u, vs in edges.items():
        for v in vs:
            G.add_edge(u, v)

    pos = {0: (0, 0), 1: (-4, 1.2), 2: (-1.5, 1.2),
           3: (1.5, 1.2), 4: (4, 1.2),
           5: (-4, 2.5), 6: (-1.5, 2.5), 7: (1.5, 2.5), 8: (4, 2.5),
           9: (0, 3.8)}

    # 节点颜色
    node_colors = []
    for n in G.nodes():
        if n == 0:
            node_colors.append(C_ENTRY)
        elif n == 9:
            node_colors.append(C_TERM)
        else:
            node_colors.append(C_MID)

    # 边
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color=C_EDGE,
                            arrows=True, arrowsize=10, width=1.0)
    # 节点
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                            node_size=380, edgecolors='black', linewidths=0.8)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=9, font_color='white')

    # 高亮入口分支（最小割 {1,2,3,4}）
    for n in [1, 2, 3, 4]:
        ax.add_patch(plt.Circle(pos[n], 0.35, fill=False,
                                 edgecolor=C_CUT, linewidth=2.5,
                                 linestyle='--'))

    ax.set_xlim(-5, 5)
    ax.set_ylim(-0.8, 4.6)
    ax.axis('off')

    # 标注
    ax.text(0, 4.5, r"(a) $G_{10}$: |V|=10, depth=3, $\kappa_v$=4",
            ha='center', fontsize=9, fontweight='bold')
    ax.text(0, -0.5, r"B=1,  $\rho$=0.25",
            ha='center', fontsize=8.5, color='#333')


# ============================================================
# (b)-(d)：100 节点图，分层着色 + 最小割标注
# ============================================================
def draw_100(ax, layers_config, title):
    """layers_config: (层数, 每层节点数)"""
    n_layers, n_per_layer = layers_config

    # 生成层级坐标
    x_start = -n_per_layer / 2
    layer_y = [-(i) for i in range(n_layers + 1)]  # 顶部 0，底部 -n_layers

    # 节点位置
    pos = {}
    # 入口（顶部一个点）
    pos['entry'] = (0, 0)
    # 中间层
    for l in range(n_layers):
        for i in range(n_per_layer):
            pos[(l, i)] = (x_start + i, layer_y[l + 1])

    # 边
    for l in range(n_layers):
        for i in range(n_per_layer):
            # 每层节点连到下一层的 3 个相邻节点
            if l == 0:
                # 第一层与入口相连
                ax.plot([0, x_start + i], [0, layer_y[1]],
                        color=C_EDGE, lw=0.25, alpha=0.5, zorder=1)
            else:
                for offset in [-1, 0, 1]:
                    j = i + offset
                    if 0 <= j < n_per_layer:
                        ax.plot([x_start + i, x_start + j],
                                [layer_y[l], layer_y[l + 1]],
                                color=C_EDGE, lw=0.2, alpha=0.4, zorder=1)

    # 节点（着色：入口绿、中间蓝、最后层橙、终点红）
    # 入口
    ax.scatter(0, 0, s=45, color=C_ENTRY, edgecolors='black',
               linewidths=0.6, zorder=3)
    # 中间层
    for l in range(n_layers):
        for i in range(n_per_layer):
            x = x_start + i
            y = layer_y[l + 1]
            if l == 0:
                # 第一层（最小割节点）用橙色高亮
                ax.scatter(x, y, s=22, color=C_CUT, edgecolors='black',
                           linewidths=0.4, zorder=3)
            else:
                ax.scatter(x, y, s=12, color=C_MID, edgecolors='none',
                           zorder=3)
    # 终点
    ax.scatter(0, layer_y[-1] - 0.6, s=45, color=C_TERM,
               edgecolors='black', linewidths=0.6, zorder=3)

    # 缩放到合适范围
    ax.set_xlim(x_start - 1.5, x_start + n_per_layer + 0.5)
    ax.set_ylim(layer_y[-1] - 1.5, 1.0)
    ax.axis('off')

    # 标题
    ax.set_title(title, fontsize=9.5, fontweight='bold', pad=4)

    # 底部信息
    depth = n_layers + 1
    ax.text(x_start + n_per_layer / 2, layer_y[-1] - 1.1,
            f"depth={depth},  B=8,  $\\rho$=2.00",
            ha='center', fontsize=8.5, color='#333')


# ============================================================
# 主绘图
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), dpi=150)

# (a) G10
draw_g10(axes[0, 0])

# (b) G100-shallow
draw_100(axes[0, 1], (4, 25),
         r"(b) $G_{100}^{\mathrm{shallow}}$: 4 layers")

# (c) G100-complex
draw_100(axes[1, 0], (6, 17),
         r"(c) $G_{100}^{\mathrm{complex}}$: 6 layers")

# (d) G100-deep
draw_100(axes[1, 1], (8, 12),
         r"(d) $G_{100}^{\mathrm{deep}}$: 8 layers")

# ============================================================
# 全局图例（放在底部）
# ============================================================
legend_elements = [
    mpatches.Patch(color=C_ENTRY, label='Entry node'),
    mpatches.Patch(color=C_TERM,  label='Terminal (critical asset)'),
    mpatches.Patch(color=C_MID,   label='Intermediate node'),
    mpatches.Patch(color=C_CUT,   label=r'Minimum-cut node $\kappa_v(s,t)$'),
    mpatches.Patch(facecolor='white', edgecolor=C_EDGE,
                   label='Directed edge'),
]

fig.legend(handles=legend_elements, loc='lower center',
           ncol=5, fontsize=9, frameon=False,
           bbox_to_anchor=(0.5, -0.02))

plt.tight_layout(rect=[0, 0.04, 1, 1])
plt.savefig("paper/figures/figure_topologies.pdf", bbox_inches='tight')
plt.savefig("paper/figures/figure_topologies.png", bbox_inches='tight', dpi=300)
print("Saved: paper/figures/figure_topologies.pdf")