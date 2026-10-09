"""Figure: Ablation study bar chart."""
import matplotlib.pyplot as plt
import numpy as np
import glob, json

# 从 results_ablation/ 读数据
attackers = ["random", "shortest", "highest_benefit", "risk_averse"]
groups = {}
for f in sorted(glob.glob("results_ablation/*/result.json")):
    with open(f) as fp:
        r = json.load(fp)
    game = r["game"]
    if game not in groups:
        groups[game] = {a: [] for a in attackers}
    for a, hit in r["hit_rates"].items():
        groups[game][a].append(hit)

labels_map = {
    "attack_graph_10": "Full",
    "attack_graph_10_nodynamic": "w/o dyn.",
    "attack_graph_10_nohidden": "w/o rich\nencoding",
}
order = ["attack_graph_10", "attack_graph_10_nodynamic", "attack_graph_10_nohidden"]

# 只展示 3 种 smart 攻击者
smart = ["shortest", "highest_benefit", "risk_averse"]
x = np.arange(len(smart))
width = 0.25
colors = ['#4472C4', '#ED7D31', '#70AD47']

fig, ax = plt.subplots(figsize=(7, 3.5), dpi=150)
for i, g in enumerate(order):
    means = [np.mean(groups[g][a]) for a in smart]
    stds = [np.std(groups[g][a]) for a in smart]
    ax.bar(x + (i - 1) * width, means, width, yerr=stds,
           label=labels_map[g], color=colors[i], capsize=3,
           edgecolor='black', linewidth=0.5)

ax.set_xticks(x)
ax.set_xticklabels(["Shortest", "Highest-benefit", "Risk-averse"], fontsize=9)
ax.set_ylabel("Hit rate", fontsize=10)
ax.set_ylim(0, 1.0)
ax.set_title("Ablation on $G_{10}$ (SDCFR, 3 seeds)", fontsize=11)
ax.legend(fontsize=8, loc='lower right')
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig("paper/figures/figure_ablation.pdf", bbox_inches='tight')
plt.savefig("paper/figures/figure_ablation.png", bbox_inches='tight', dpi=300)
print("已保存 figure_ablation.pdf")
