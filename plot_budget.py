"""Figure: Budget vs hit rate."""
import matplotlib.pyplot as plt
import numpy as np

# 从 Table 3 数据（已有）
budgets = [1, 2, 3, 3]
hits = [0.825, 0.997, 1.000, 1.000]
labels = ["$G_{10}$", "$G_{100}^{sh}$", "$G_{100}^{cx}$", "$G_{100}^{dp}$"]

fig, ax = plt.subplots(figsize=(6, 3.5), dpi=150)
ax.plot(budgets[:3], hits[:3], marker='o', linewidth=2.5,
        color='#4472C4', markersize=10)
# 标出每个点
for b, h, l in zip(budgets, hits, labels):
    ax.annotate(l, (b, h), textcoords="offset points",
                xytext=(0, 10), ha='center', fontsize=9)

ax.axvline(x=4, color='red', linestyle='--', alpha=0.5)
ax.text(4.1, 0.85, r'$\kappa_v(s,t)=4$', color='red', fontsize=9)

ax.set_xlabel("Honeypot budget $B$", fontsize=11)
ax.set_ylabel("Risk-averse hit rate", fontsize=11)
ax.set_ylim(0.7, 1.05)
ax.set_xticks([1, 2, 3, 4])
ax.set_title("Budget drives defense effectiveness", fontsize=11)
ax.grid(alpha=0.3, linestyle='--')
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig("paper/figures/figure_budget.pdf", bbox_inches='tight')
plt.savefig("paper/figures/figure_budget.png", bbox_inches='tight', dpi=300)
print("已保存 figure_budget.pdf")
