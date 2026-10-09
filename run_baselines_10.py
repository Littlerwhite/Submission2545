"""10 节点图上的经典基线对比：Tabular CFR / NFSP / External MCCFR / Outcome MCCFR。"""
import os
import json
import time
import random
import numpy as np
import torch
import pyspiel
import open_spiel.python.games.attack_graph_scalable
from open_spiel.python import rl_environment

GAME = "attack_graph_10"
ITER = 1000              # 训练轮数
EVAL_GAMES = 300         # 每种攻击者评估局数
SEEDS = [42, 123, 456]   # 3 个种子
ATTACKERS = ["random", "shortest", "highest_benefit", "risk_averse"]


# ============================================================
# 攻击者构造
# ============================================================
def build_dist(edges, terminal):
    dist = {}
    for n in edges:
        visited, queue = {n}, [(n, 0)]
        d = float('inf')
        while queue:
            cur, dd = queue.pop(0)
            if cur == terminal:
                d = dd
                break
            for nxt in edges.get(cur, []):
                if nxt not in visited:
                    visited.add(nxt)
                    queue.append((nxt, dd + 1))
        dist[n] = d
    return dist


def make_attackers(edges, terminal):
    dist = build_dist(edges, terminal)

    def random_attacker(state):
        return int(np.random.choice(state.legal_actions(0)))

    def shortest_attacker(state):
        legal = state.legal_actions(0)
        moves = [a for a in legal if a != 0]
        if not moves:
            return 0
        neighbors = edges.get(state.attacker_pos, [])
        best_action, best_d = 0, float('inf')
        for a in moves:
            idx = a - 1
            if 0 <= idx < len(neighbors):
                d = dist.get(neighbors[idx], float('inf'))
                if d < best_d:
                    best_d = d
                    best_action = a
        return best_action

    def highest_benefit_attacker(state):
        legal = state.legal_actions(0)
        neighbors = edges.get(state.attacker_pos, [])
        best_action, best_score = 0, -float('inf')
        for a in legal:
            if a == 0:
                score = -1.0
            else:
                idx = a - 1
                if 0 <= idx < len(neighbors):
                    target = neighbors[idx]
                    if target in state.honeypot_nodes:
                        score = -1.0
                    elif target == terminal:
                        score = 1.0
                    else:
                        score = 0.5 - dist.get(target, 10) * 0.1
                else:
                    score = -1.0
            if score > best_score:
                best_score = score
                best_action = a
        return best_action

    def risk_averse_attacker(state):
        legal = state.legal_actions(0)
        neighbors = edges.get(state.attacker_pos, [])
        safe = []
        for a in legal:
            if a == 0:
                continue
            idx = a - 1
            if 0 <= idx < len(neighbors):
                if neighbors[idx] not in state.honeypot_nodes:
                    safe.append(a)
        if safe:
            return min(safe, key=lambda a: dist.get(neighbors[a-1], float('inf')))
        return 0

    return {
        "random": random_attacker,
        "shortest": shortest_attacker,
        "highest_benefit": highest_benefit_attacker,
        "risk_averse": risk_averse_attacker,
    }


# ============================================================
# 通用评估：给定"防守策略函数"，跑四种攻击者
# ============================================================
def evaluate_defender(prob_func, game, attackers, num_games=EVAL_GAMES):
    results = {}
    for name, atk in attackers.items():
        hits = 0
        for _ in range(num_games):
            state = game.new_initial_state()
            while not state.is_terminal():
                p = state.current_player()
                if p == 0:
                    action = atk(state)
                else:
                    probs_dict = prob_func(state)
                    legal = state.legal_actions(1)
                    probs = np.array([probs_dict.get(a, 0.0) for a in legal])
                    if probs.sum() > 0:
                        probs /= probs.sum()
                    else:
                        probs = np.ones(len(legal)) / len(legal)
                    action = np.random.choice(legal, p=probs)
                state.apply_action(action)
            if state.returns()[0] == -1.0:
                hits += 1
        results[name] = hits / num_games
    return results


# ============================================================
# 四种基线：各自返回一个 prob_func(state) -> dict
# ============================================================
def run_tabular_cfr(game, seed):
    from open_spiel.python.algorithms import cfr
    random.seed(seed); np.random.seed(seed)
    solver = cfr.CFRSolver(game)
    for _ in range(ITER):
        solver.evaluate_and_update_policy()
    avg_policy = solver.average_policy()

    def prob(state):
        player = state.current_player()
        legal = state.legal_actions(player)
        probs_dict = avg_policy.action_probabilities(state)
        probs = {a: probs_dict.get(a, 0.0) for a in legal}
        s = sum(probs.values())
        return {a: p / s for a, p in probs.items()} if s > 0 else \
               {a: 1.0 / len(legal) for a in legal}
    return prob


def run_nfsp(game, seed):
    from open_spiel.python.pytorch import nfsp
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    env = rl_environment.Environment(game)
    info_size = env.observation_spec()["info_state"][0]
    num_actions = env.action_spec()["num_actions"]
    agents = [
        nfsp.NFSP(
            player_id=p,
            state_representation_size=info_size,
            num_actions=num_actions,
            hidden_layers_sizes=[64, 64],
            reservoir_buffer_capacity=2000,
            anticipatory_param=0.1,
            batch_size=32,
            rl_learning_rate=0.01,
            sl_learning_rate=0.01,
            min_buffer_size_to_learn=100,
            optimizer_str="sgd",
        ) for p in range(2)
    ]
    for _ in range(3000):
        ts = env.reset()
        while not ts.last():
            pid = ts.observations["current_player"]
            out = agents[pid].step(ts)
            ts = env.step([out.action])
        for ag in agents:
            ag.step(ts)

    def prob(state):
        player = state.current_player()
        legal = state.legal_actions(player)
        info = state.information_state_tensor(player)
        agent = agents[player]
        with torch.no_grad():
            info_t = torch.FloatTensor(info).unsqueeze(0).to(agent._device)
            logits = agent._avg_network(info_t)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        probs_dict = {a: float(probs[a]) for a in legal}
        s = sum(probs_dict.values())
        return {a: p / s for a, p in probs_dict.items()} if s > 0 else \
               {a: 1.0 / len(legal) for a in legal}
    return prob


def run_external_mccfr(game, seed):
    from open_spiel.python.algorithms import external_sampling_mccfr
    random.seed(seed); np.random.seed(seed)
    solver = external_sampling_mccfr.ExternalSamplingSolver(
        game, external_sampling_mccfr.AverageType.SIMPLE)
    for _ in range(ITER):
        solver.iteration()
    avg_policy = solver.average_policy()

    def prob(state):
        player = state.current_player()
        legal = state.legal_actions(player)
        probs_dict = avg_policy.action_probabilities(state)
        probs = {a: probs_dict.get(a, 0.0) for a in legal}
        s = sum(probs.values())
        return {a: p / s for a, p in probs.items()} if s > 0 else \
               {a: 1.0 / len(legal) for a in legal}
    return prob


def run_outcome_mccfr(game, seed):
    from open_spiel.python.algorithms import outcome_sampling_mccfr
    random.seed(seed); np.random.seed(seed)
    solver = outcome_sampling_mccfr.OutcomeSamplingSolver(game)
    for _ in range(ITER):
        solver.iteration()
    avg_policy = solver.average_policy()

    def prob(state):
        player = state.current_player()
        legal = state.legal_actions(player)
        probs_dict = avg_policy.action_probabilities(state)
        probs = {a: probs_dict.get(a, 0.0) for a in legal}
        s = sum(probs.values())
        return {a: p / s for a, p in probs.items()} if s > 0 else \
               {a: 1.0 / len(legal) for a in legal}
    return prob


# ============================================================
# 主流程
# ============================================================
def main():
    game = pyspiel.load_game(GAME)
    game_cls = type(game)
    attackers = make_attackers(game_cls.edges, game_cls.terminal_node)

    methods = {
        "TabularCFR":     run_tabular_cfr,
        "NFSP":           run_nfsp,
        "ExternalMCCFR":  run_external_mccfr,
        "OutcomeMCCFR":   run_outcome_mccfr,
    }

    os.makedirs("results_baselines", exist_ok=True)

    for name, func in methods.items():
        print(f"\n{'='*60}\n{name}\n{'='*60}")
        all_results = {a: [] for a in ATTACKERS}
        times = []

        for seed in SEEDS:
            print(f"\n--- seed {seed} ---")
            t0 = time.time()
            try:
                prob_func = func(game, seed)
                elapsed = time.time() - t0
                times.append(elapsed)
                r = evaluate_defender(prob_func, game, attackers)
                for a in ATTACKERS:
                    all_results[a].append(r[a])
                print(f"  {r}  ({elapsed:.1f}s)")
            except Exception as e:
                import traceback
                traceback.print_exc()
                times.append(0.0)
                for a in ATTACKERS:
                    all_results[a].append(0.0)

        # 汇总
        print(f"\n--- {name} 汇总 ---")
        summary = {}
        for a in ATTACKERS:
            arr = np.array(all_results[a])
            print(f"  {a:16s}: {arr.mean():.4f} ± {arr.std():.4f}")
            summary[a] = {"mean": float(arr.mean()),
                          "std": float(arr.std()),
                          "raw": [float(x) for x in arr]}
        summary["train_time"] = float(np.mean(times))
        with open(f"results_baselines/{name}.json", "w") as f:
            json.dump(summary, f, indent=2)

    print("\n全部完成！结果保存在 results_baselines/")


if __name__ == "__main__":
    main()
