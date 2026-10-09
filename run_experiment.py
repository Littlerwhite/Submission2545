"""统一实验运行器，支持多种子、多算法、多图。"""
import argparse
import os
import json
import time
import random
import numpy as np
import torch
import pyspiel
import open_spiel.python.games.attack_graph_scalable

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train_deepcfr(game, iterations, seed, out_dir):
    from open_spiel.python.pytorch import deep_cfr
    solver = deep_cfr.DeepCFRSolver(
        game,
        num_iterations=iterations,
        num_traversals=50,
        learning_rate=0.001,
        batch_size_advantage=512,
        batch_size_strategy=512,
        memory_capacity=10000,
        seed=seed,
    )
    t0 = time.time()
    for i in range(iterations):
        for p in range(solver._num_players):
            for _ in range(solver._num_traversals):
                solver._traverse_game_tree(solver._root_node, p)
            if solver._reinitialize_advantage_networks:
                solver._reinitialize_advantage_network(p)
            solver._learn_advantage_network(p)
        solver._learn_strategy_network()
        solver._iteration += 1
        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{iterations}]")
    elapsed = time.time() - t0
    torch.save(solver._policy_network.state_dict(),
               os.path.join(out_dir, "policy_network.pt"))
    return solver, elapsed

def train_dueling_deepcfr(game, iterations, seed, out_dir):
    from algorithms import deep_cfr_dueling
    solver = deep_cfr_dueling.DeepCFRSolver(
        game,
        num_iterations=iterations,
        num_traversals=50,
        learning_rate=0.001,
        batch_size_advantage=512,
        batch_size_strategy=512,
        memory_capacity=10000,
        seed=seed,
    )
    t0 = time.time()
    for i in range(iterations):
        for p in range(solver._num_players):
            for _ in range(solver._num_traversals):
                solver._traverse_game_tree(solver._root_node, p)
            if solver._reinitialize_advantage_networks:
                solver._reinitialize_advantage_network(p)
            solver._learn_advantage_network(p)
        solver._learn_strategy_network()
        solver._iteration += 1
    elapsed = time.time() - t0
    torch.save(solver._policy_network.state_dict(),
               os.path.join(out_dir, "policy_network.pt"))
    return solver, elapsed


def train_sdcfr(game, iterations, seed, out_dir):
    from algorithms import single_deep_cfr
    save_dir = os.path.join(out_dir, "sdcfr_ckpts")
    solver = single_deep_cfr.SingleDeepCFRSolver(
        game,
        save_dir=save_dir,
        num_iterations=iterations,
        num_traversals=50,
        learning_rate=0.001,
        batch_size_advantage=512,
        memory_capacity=10000,
        seed=seed,
    )
    t0 = time.time()
    solver.solve()
    elapsed = time.time() - t0
    return solver, elapsed

def build_dist(edges, terminal):
    """BFS 计算每个节点到终点的最短距离。"""
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
    """构造四种攻击者策略函数。"""
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
            return min(safe,
                       key=lambda a: dist.get(neighbors[a-1], float('inf')))
        return 0

    return {
        "random": random_attacker,
        "shortest": shortest_attacker,
        "highest_benefit": highest_benefit_attacker,
        "risk_averse": risk_averse_attacker,
    }

def evaluate_defense_multi(game, solver, attackers, num_games=300,
                           is_sdcfr=False):
    """对四种攻击者分别评估，返回字典 {attacker_name: hit_rate}。"""
    results = {}
    for name, attacker_func in attackers.items():
        hits = 0
        for _ in range(num_games):
            state = game.new_initial_state()
            while not state.is_terminal():
                p = state.current_player()
                if p == 0:
                    action = attacker_func(state)
                else:
                    if is_sdcfr:
                        probs_dict = solver.action_probabilities(state, 1)
                        legal = state.legal_actions(1)
                        probs = np.array([probs_dict.get(a, 0.0)
                                          for a in legal])
                        if probs.sum() > 0:
                            probs /= probs.sum()
                        else:
                            probs = np.ones(len(legal)) / len(legal)
                        action = np.random.choice(legal, p=probs)
                    else:
                        info = state.information_state_tensor(1)
                        with torch.no_grad():
                            logits = solver._policy_network(
                                torch.FloatTensor(info).unsqueeze(0))
                        probs = torch.softmax(
                            logits, dim=-1).squeeze().numpy()
                        legal = state.legal_actions(1)
                        masked = np.zeros_like(probs)
                        for a in legal:
                            masked[a] = probs[a]
                        if masked.sum() > 0:
                            masked /= masked.sum()
                        else:
                            masked[legal[0]] = 1.0
                        action = np.random.choice(len(masked), p=masked)
                state.apply_action(action)
            if state.returns()[0] == -1.0:
                hits += 1
        results[name] = hits / num_games
        print(f"    vs {name:16s}: {results[name]:.4f}")
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", default="attack_graph_10")
    parser.add_argument("--algo", default="deepcfr",
                        choices=["deepcfr", "dueling_deepcfr", "sdcfr", "random"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--output_dir", default="results")
    args = parser.parse_args()

    set_seed(args.seed)
    game = pyspiel.load_game(args.game)
    out_dir = os.path.join(args.output_dir,
                           f"{args.game}_{args.algo}_seed{args.seed}")
    os.makedirs(out_dir, exist_ok=True)
    print(f"=== {args.game} | {args.algo} | seed={args.seed} ===")
    game_cls = type(game)
    attackers = make_attackers(game_cls.edges, game_cls.terminal_node)

    if args.algo == "deepcfr":
        solver, elapsed = train_deepcfr(game, args.iterations, args.seed, out_dir)
        hit_rates = evaluate_defense_multi(game, solver, attackers, num_games=300)
        result = {
            "game": args.game, "algo": args.algo, "seed": args.seed,
            "iterations": args.iterations,
            "hit_rates": hit_rates,
            "train_time": elapsed,
        }
    elif args.algo == "dueling_deepcfr":
        solver, elapsed = train_dueling_deepcfr(game, args.iterations, args.seed, out_dir)
        hit_rates = evaluate_defense_multi(game, solver, attackers, num_games=300)
        result = {
            "game": args.game, "algo": args.algo, "seed": args.seed,
            "iterations": args.iterations,
            "hit_rates": hit_rates,
            "train_time": elapsed,
        }
    elif args.algo == "sdcfr":
        solver, elapsed = train_sdcfr(game, args.iterations, args.seed, out_dir)
        hit_rates = evaluate_defense_multi(game, solver, attackers,
                                           num_games=300, is_sdcfr=True)
        result = {
            "game": args.game, "algo": args.algo, "seed": args.seed,
            "iterations": args.iterations,
            "hit_rates": hit_rates,
            "train_time": elapsed,
        }
    elif args.algo == "random":
        # 随机防守基线（训练时间为 0）
        class RandomSolver:
            def __init__(self):
                self._policy_network = None
        hit_rate = 0.5  # 占位
        result = {"game": args.game, "algo": "random", "seed": args.seed,
                  "hit_rate": hit_rate, "train_time": 0.0}

    with open(os.path.join(out_dir, "result.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(f"结果: {result}")

if __name__ == "__main__":
    main()