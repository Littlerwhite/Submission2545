"""Figure: Training time comparison."""
import matplotlib.pyplot as plt
import numpy as np
import glob, json

# 从 result.json 提取训练时间
def extract_time(pattern):
    times = []
    for f in glob.glob(pattern):
        with open(f) as fp:
            times.append(json.load(fp)["train_time"])
    return times

configs = [
    ("DeepCFR", "results_final/attack_graph_10_deepcfr_seed*/result.json"),
    ("Dueling", "results_final/attack_graph_10_dueling_deepcfr_seed*/result.json"),
    ("SDCFR",   "results_final/attack_graph_10_sdcfr_seed*/result.json"),
]

names, means, stds = [], [], []
for name, pattern in configs:
    t = extract_time(pattern)
    if t:
        names.append(name)
        means.append(np.mean(t))
        stds.append(np.std(t))

fig, ax = plt.subplots(figsize=(5, 3.5), dpi=150)
colors = ['#4472C4', '#ED7D31', '#70AD47']
bars = ax.bar(names, means, yerr=stds, color=colors, capsize=5,
              edgecolor='black', linewidth=0.5)

for bar, m in zip(bars, means):
    ax.text(bar.get_x() + bar.get_width()/2, m + 2,
            f"{m:.1f}s", ha='center', fontsize=9)

ax.set_ylabel("Training time (s)", fontsize=11)
ax.set_title("Training time on $G_{10}$ (3 seeds)", fontsize=11)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig("paper/figures/figure_traintime.pdf", bbox_inches='tight')
plt.savefig("paper/figures/figure_traintime.png", bbox_inches='tight', dpi=300)
print("已保存 figure_traintime.pdf")
