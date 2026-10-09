"""计算每张攻击图的预算-关键割比率 ρ = B / κ_v(s,t)。"""
import pyspiel
import open_spiel.python.games.attack_graph_scalable
from collections import deque


def min_vertex_cut(edges, s, t):
    """用最大流/最小割思想求 s-t 最小顶点割大小。

    简化实现：用节点分裂法将顶点割转为边割，然后跑 Edmonds-Karp。
    节点 v 分裂为 v_in -> v_out，容量 1（s 和 t 除外，容量 inf）。
    """
    nodes = set(edges.keys())
    for v in edges.values():
        nodes.update(v)
    N = max(nodes) + 1

    # 构建残量图
    # 每个节点 v 拆成 2v (in) 和 2v+1 (out)
    cap = {}

    def add_edge(u, v, c):
        cap[(u, v)] = cap.get((u, v), 0) + c
        cap.setdefault((v, u), 0)

    for v in nodes:
        if v == s or v == t:
            add_edge(2 * v, 2 * v + 1, float('inf'))
        else:
            add_edge(2 * v, 2 * v + 1, 1)
    for u in edges:
        for v in edges[u]:
            add_edge(2 * u + 1, 2 * v, float('inf'))

    source = 2 * s + 1
    sink = 2 * t

    # BFS 增广
    def bfs():
        parent = {source: None}
        queue = deque([source])
        while queue:
            u = queue.popleft()
            for (a, b) in cap:
                if a == u and cap[(a, b)] > 0 and b not in parent:
                    parent[b] = u
                    if b == sink:
                        return parent
                    queue.append(b)
        return None

    flow = 0
    while True:
        parent = bfs()
        if parent is None:
            break
        # 找瓶颈
        path = []
        cur = sink
        while cur is not None:
            path.append(cur)
            cur = parent[cur]
        path = path[::-1]
        bottleneck = min(cap[(path[i], path[i + 1])] for i in range(len(path) - 1))
        for i in range(len(path) - 1):
            cap[(path[i], path[i + 1])] -= bottleneck
            cap[(path[i + 1], path[i])] += bottleneck
        flow += bottleneck
    return flow


GAMES = [
    "attack_graph_10",
    "attack_graph_100",
    "attack_graph_100_complex",
    "attack_graph_100_deep",
]

print(f"{'Game':<30} {'|V|':>5} {'κ_v(s,t)':>10} {'Budget':>8} {'ρ=B/κ':>8}")
print("-" * 70)

for name in GAMES:
    g = pyspiel.load_game(name)
    kappa = min_vertex_cut(g.edges, 0, g.terminal_node)
    budget = g.defender_budget_init
    rho = budget / kappa if kappa > 0 else float('inf')
    print(f"{name:<30} {g.max_node+1:>5} {kappa:>10} {budget:>8} {rho:>8.3f}")
