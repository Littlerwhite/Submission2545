# -*- coding: utf-8 -*-
"""可扩展攻击图：支持任意节点数，动作与节点ID解耦。"""
import pyspiel


class ScalableAttackGraphState(pyspiel.State):
    def __init__(self, game):
        super().__init__(game)
        self._game = game
        self.attacker_pos = 0
        self.defender_budget = game.defender_budget_init
        self.attacker_budget = game.attacker_budget_init
        self.honeypot_nodes = set()
        self.terminated = False
        self._returns = [0.0, 0.0]
        self._turn = 0
        self._step_count = 0
        self._defender_turn_count = 0
        self._ablate_dynamic = getattr(game, 'ablate_dynamic', False)
        self._ablate_hidden = getattr(game, 'ablate_hidden', False)

    def current_player(self):
        if self.terminated:
            return pyspiel.PlayerId.TERMINAL
        return 0 if self._turn == 0 else 1

    def _legal_actions(self, player):
        """C++ 层调用的实现（Tabular CFR 等需要）。"""
        if player == 0:
            if self.attacker_pos == self._game.terminal_node:
                return [0]
            if self.attacker_budget <= 0:
                return [0]
            neighbors = self._game.edges.get(self.attacker_pos, [])
            return [0] + list(range(1, len(neighbors) + 1))
        else:
            if self._ablate_dynamic and self._defender_turn_count > 0:
                return [0]
            if self.defender_budget > 0:
                return [0, 1]
            return [0]

    def legal_actions(self, player=None):
        """Python 层调用的包装器（DeepCFR 等需要）。"""
        if player is None:
            player = self.current_player()
        return self._legal_actions(player)

    def _apply_action(self, action):
        if self.terminated:
            return
        self._step_count += 1
        if self._step_count > self._game.max_game_length:
            self.terminated = True
            self._returns = [0.0, 0.0]
            return

        if self._turn == 0:
            if self.attacker_pos in self.honeypot_nodes:
                self.terminated = True
                self._returns = [-1.0, 1.0]
                return
            if action == 0:
                self.terminated = True
                self._returns = [-1.0, 1.0]
                return
            neighbors = self._game.edges.get(self.attacker_pos, [])
            idx = action - 1
            if 0 <= idx < len(neighbors):
                self.attacker_pos = neighbors[idx]
                self.attacker_budget -= 1
                if self.attacker_pos == self._game.terminal_node:
                    self.terminated = True
                    self._returns = [1.0, -1.0]
                    return
        else:
            if action == 1 and self.defender_budget > 0:
                self.honeypot_nodes.add(self.attacker_pos)
                self.defender_budget -= 1
            self._defender_turn_count += 1

        self._turn = 1 - self._turn

    def apply_action(self, action):
        self._apply_action(action)

    def is_terminal(self):
        return self.terminated

    def returns(self):
        return self._returns

    def information_state_tensor(self, player=None):
        """固定 5 维信息状态向量。

        正常版本：位置、防守预算、蜜罐数、回合、是否在蜜罐。
        消融版本（ablate_hidden=True）：只保留位置和回合。
        """
        if self._ablate_hidden:
            return [
                self.attacker_pos / float(self._game.max_node),
                0.0,
                0.0,
                float(self._turn),
                0.0,
            ]
        return [
            self.attacker_pos / float(self._game.max_node),
            self.defender_budget / float(self._game.defender_budget_init),
            len(self.honeypot_nodes) / float(self._game.defender_budget_init),
            float(self._turn),
            1.0 if self.attacker_pos in self.honeypot_nodes else 0.0,
        ]

    def observation_tensor(self, player):
        return self.information_state_tensor(player)

    def information_state_string(self, player=None):
        """返回信息集字符串。"""
        if player is None:
            player = self.current_player()
        legal = self._legal_actions(player)
        legal_key = ",".join(map(str, legal))
        if player == 0:
            return (f"A|pos={self.attacker_pos}|budget={self.attacker_budget}"
                    f"|turn={self._turn}|legal={legal_key}")
        else:
            hp = sorted(self.honeypot_nodes)
            return (f"D|pos={self.attacker_pos}|budget={self.defender_budget}"
                    f"|hp={hp}|turn={self._turn}|legal={legal_key}")

    def clone(self):
        new_state = ScalableAttackGraphState(self._game)
        new_state.attacker_pos = self.attacker_pos
        new_state.defender_budget = self.defender_budget
        new_state.attacker_budget = self.attacker_budget
        new_state.honeypot_nodes = set(self.honeypot_nodes)
        new_state.terminated = self.terminated
        new_state._returns = self._returns.copy()
        new_state._turn = self._turn
        new_state._step_count = self._step_count
        new_state._defender_turn_count = self._defender_turn_count
        return new_state

    def __str__(self):
        return f"pos:{self.attacker_pos}, def_budget:{self.defender_budget}"


class _ScalableAttackGraphBase(pyspiel.Game):
    edges = {}
    max_node = 9
    terminal_node = 9
    defender_budget_init = 2
    attacker_budget_init = 4
    max_game_length = 200
    max_out_degree = 4
    _registered_game_type = None

    def __init__(self, params=None):
        game_info = pyspiel.GameInfo(
            num_distinct_actions=self.max_out_degree + 1,
            max_chance_outcomes=0,
            num_players=2,
            min_utility=-1.0,
            max_utility=1.0,
            utility_sum=0.0,
            max_game_length=self.max_game_length,
        )
        super().__init__(self._registered_game_type, game_info, params)

    def num_players(self):
        return 2

    def num_distinct_actions(self):
        return self.max_out_degree + 1

    def get_parameters(self):
        return {}

    def new_initial_state(self):
        return ScalableAttackGraphState(self)

    def make_py_observer(self, iig_obs_type=None, params=None):
        return None

    def information_state_tensor_size(self):
        return 5

    def observation_tensor_size(self):
        return 5

    def information_state_tensor_size(self):
        return 5

    def observation_tensor_size(self):
        return 5


# ============================================================
# 10 节点图
# ============================================================
class AttackGraph10(_ScalableAttackGraphBase):
    edges = {
        0: [1, 2, 3, 4],
        1: [5, 6], 2: [6, 7], 3: [7, 8], 4: [8, 9],
        5: [9], 6: [9], 7: [9], 8: [9], 9: [],
    }
    max_node = 9
    terminal_node = 9
    defender_budget_init = 1
    attacker_budget_init = 4
    max_out_degree = 4
    max_game_length = 60


# ============================================================
# 100 节点浅图（4 层）
# ============================================================
def _build_100_node_edges():
    """100 节点浅层图：4 层结构，深度约 5 步。"""
    edges = {}
    edges[0] = [1, 2, 3, 4]

    layer1 = list(range(1, 26))
    layer2 = list(range(26, 51))
    layer3 = list(range(51, 76))
    layer4 = list(range(76, 99))
    layers = [layer1, layer2, layer3, layer4]

    for li in range(len(layers) - 1):
        cur, nxt = layers[li], layers[li + 1]
        for i, node in enumerate(cur):
            neighbors = []
            for offset in [i - 1, i, i + 1]:
                if 0 <= offset < len(nxt):
                    neighbors.append(nxt[offset])
            edges[node] = list(dict.fromkeys(neighbors))

    for node in layer4:
        edges[node] = [99]
    edges[99] = []
    return edges


class AttackGraph100(_ScalableAttackGraphBase):
    edges = _build_100_node_edges()
    max_node = 99
    terminal_node = 99
    defender_budget_init = 2
    attacker_budget_init = 12
    max_out_degree = 4
    max_game_length = 100


# ============================================================
# 100 节点复杂图（6 层 + 交叉边）
# ============================================================
def _build_100_complex_node_edges():
    """复杂 100 节点图：6 层，含交叉边。"""
    edges = {}
    edges[0] = [1, 2, 3, 4]

    layer1 = list(range(1, 17))
    layer2 = list(range(17, 33))
    layer3 = list(range(33, 49))
    layer4 = list(range(49, 65))
    layer5 = list(range(65, 81))
    layer6 = list(range(81, 99))
    layers = [layer1, layer2, layer3, layer4, layer5, layer6]

    for li in range(len(layers) - 1):
        cur, nxt = layers[li], layers[li + 1]
        for i, node in enumerate(cur):
            neighbors = []
            for offset in [i - 1, i, i + 1]:
                if 0 <= offset < len(nxt):
                    neighbors.append(nxt[offset])
            if i % 4 == 0:
                cross = (i + 5) % len(nxt)
                neighbors.append(nxt[cross])
            edges[node] = list(dict.fromkeys(neighbors))

    for node in layer6:
        edges[node] = [99]
    edges[99] = []
    return edges


class AttackGraph100Complex(_ScalableAttackGraphBase):
    edges = _build_100_complex_node_edges()
    max_node = 99
    terminal_node = 99
    defender_budget_init = 3
    attacker_budget_init = 15
    max_out_degree = 4
    max_game_length = 150


# ============================================================
# 注册
# ============================================================
def _make_game_type(short_name, long_name):
    return pyspiel.GameType(
        short_name=short_name, long_name=long_name,
        dynamics=pyspiel.GameType.Dynamics.SEQUENTIAL,
        chance_mode=pyspiel.GameType.ChanceMode.DETERMINISTIC,
        information=pyspiel.GameType.Information.IMPERFECT_INFORMATION,
        utility=pyspiel.GameType.Utility.ZERO_SUM,
        reward_model=pyspiel.GameType.RewardModel.TERMINAL,
        max_num_players=2, min_num_players=2,
        provides_information_state_string=True,
        provides_information_state_tensor=True,
        provides_observation_string=True,
        provides_observation_tensor=True,
        parameter_specification={},
    )




# ============================================================
# 100 节点深图（8 层 × 12 节点）
# ============================================================
def _build_100_deep_node_edges():
    """100 节点深图：8 层，深度约 9 步。"""
    edges = {}
    edges[0] = [1, 2, 3, 4]

    layer1 = list(range(1, 13))
    layer2 = list(range(13, 25))
    layer3 = list(range(25, 37))
    layer4 = list(range(37, 49))
    layer5 = list(range(49, 61))
    layer6 = list(range(61, 73))
    layer7 = list(range(73, 85))
    layer8 = list(range(85, 99))
    layers = [layer1, layer2, layer3, layer4, layer5, layer6, layer7, layer8]

    for li in range(len(layers) - 1):
        cur, nxt = layers[li], layers[li + 1]
        for i, node in enumerate(cur):
            neighbors = []
            for offset in [i - 1, i, i + 1]:
                if 0 <= offset < len(nxt):
                    neighbors.append(nxt[offset])
            edges[node] = list(dict.fromkeys(neighbors))

    for node in layer8:
        edges[node] = [99]
    edges[99] = []
    return edges


class AttackGraph100Deep(_ScalableAttackGraphBase):
    edges = _build_100_deep_node_edges()
    max_node = 99
    terminal_node = 99
    defender_budget_init = 3
    attacker_budget_init = 12
    max_out_degree = 4
    max_game_length = 150


AttackGraph10._registered_game_type = _make_game_type(
    "attack_graph_10", "Attack Graph 10")
AttackGraph100._registered_game_type = _make_game_type(
    "attack_graph_100", "Attack Graph 100")
AttackGraph100Complex._registered_game_type = _make_game_type(
    "attack_graph_100_complex", "Attack Graph 100 Complex")

pyspiel.register_game(AttackGraph10._registered_game_type, AttackGraph10)
pyspiel.register_game(AttackGraph100._registered_game_type, AttackGraph100)
pyspiel.register_game(
    AttackGraph100Complex._registered_game_type, AttackGraph100Complex)

AttackGraph100Deep._registered_game_type = _make_game_type(
    "attack_graph_100_deep", "Attack Graph 100 Deep")
pyspiel.register_game(
    AttackGraph100Deep._registered_game_type, AttackGraph100Deep)


# ============================================================
# 消融实验变体
# ============================================================
class AttackGraph10NoDynamic(AttackGraph10):
    """消融 1：防守者只能在第一步部署，之后不能再部署。"""
    ablate_dynamic = True


class AttackGraph10NoHidden(AttackGraph10):
    """消融 2：去除隐藏攻击者类型（占位对照）。"""
    ablate_hidden = True


AttackGraph10NoDynamic._registered_game_type = _make_game_type(
    "attack_graph_10_nodynamic", "Attack Graph 10 No Dynamic")
AttackGraph10NoHidden._registered_game_type = _make_game_type(
    "attack_graph_10_nohidden", "Attack Graph 10 No Hidden")

pyspiel.register_game(AttackGraph10NoDynamic._registered_game_type,
                      AttackGraph10NoDynamic)
pyspiel.register_game(AttackGraph10NoHidden._registered_game_type,
                      AttackGraph10NoHidden)

# ============================================================
# 100 节点消融变体
# ============================================================
class AttackGraph100NoDynamic(AttackGraph100):
    """消融：防守者只能在第一步部署。"""
    ablate_dynamic = True


class AttackGraph100NoHidden(AttackGraph100):
    """消融：去除丰富状态编码，只保留位置和回合。"""
    ablate_hidden = True


AttackGraph100NoDynamic._registered_game_type = _make_game_type(
    "attack_graph_100_nodynamic", "Attack Graph 100 No Dynamic")
AttackGraph100NoHidden._registered_game_type = _make_game_type(
    "attack_graph_100_nohidden", "Attack Graph 100 No Hidden")

pyspiel.register_game(AttackGraph100NoDynamic._registered_game_type,
                      AttackGraph100NoDynamic)
pyspiel.register_game(AttackGraph100NoHidden._registered_game_type,
                      AttackGraph100NoHidden)
