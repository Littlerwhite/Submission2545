"""按游戏分组汇总 results_final/ 下所有实验。"""
import glob, json
import numpy as np

attackers = ["random", "shortest", "highest_benefit", "risk_averse"]

groups = {}
for f in glob.glob("results_final/*/result.json"):
    with open(f) as fp:
        r = json.load(fp)
    key = (r["game"], r["algo"])
    if key not in groups:
        groups[key] = {a: [] for a in attackers}
    for a, hit in r["hit_rates"].items():
        groups[key][a].append(hit)

games = sorted(set(k[0] for k in groups))
algos = ["deepcfr", "dueling_deepcfr", "sdcfr"]

for game in games:
    print(f"\n{'='*95}")
    print(f"游戏: {game}")
    print(f"{'='*95}")
    print(f"{'algo':<20}", end="")
    for a in attackers:
        print(f"{a:>18}", end="")
    print()
    print("-" * 95)
    for algo in algos:
        key = (game, algo)
        if key not in groups:
            continue
        print(f"{algo:<20}", end="")
        for a in attackers:
            arr = np.array(groups[key][a])
            if len(arr) > 0:
                print(f"{arr.mean():>7.4f}±{arr.std():.4f}   ", end="")
            else:
                print(f"{'-':>18}", end="")
        print()
