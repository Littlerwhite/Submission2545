"""Figure: Attacker type radar chart."""
import matplotlib.pyplot as plt
import numpy as np

categories = ['Random', 'Shortest', 'Highest', 'Risk-averse']
N = len(categories)
angles = [n / N * 2 * np.pi for n in range(N)]
angles += angles[:1]

# Table 1 data
methods = {
    "DeepCFR":  [0.920, 0.611, 0.624, 0.652],
    "Dueling":  [0.958, 0.707, 0.727, 0.710],
    "SDCFR":    [1.000, 0.824, 0.824, 0.825],
    "Tabular":  [1.000, 1.000, 0.998, 1.000],
}
colors = ['#4472C4', '#ED7D31', '#70AD47', '#C00000']

fig, ax = plt.subplots(figsize=(5.5, 5.5), dpi=150,
                        subplot_kw=dict(polar=True))

for (name, vals), color in zip(methods.items(), colors):
    vals = vals + vals[:1]
    ax.plot(angles, vals, linewidth=2, label=name, color=color)
    ax.fill(angles, vals, alpha=0.1, color=color)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=10)
ax.set_ylim(0, 1.25)
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=8)
ax.tick_params(axis='x', pad=18)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=9)
ax.set_title("Performance across attacker types ($G_{10}$)", fontsize=11, pad=20)

plt.tight_layout()
plt.savefig("paper/figures/figure_radar.pdf", bbox_inches='tight')
plt.savefig("paper/figures/figure_radar.png", bbox_inches='tight', dpi=300)
print("已保存 figure_radar.pdf")
