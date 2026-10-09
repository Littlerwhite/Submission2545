"""Single Deep CFR (SDCFR)：不训练策略网络，只保存每轮 advantage 网络。
决策时随机采样一个 advantage 网络（线性加权，越新权重越高）做 regret matching。
"""
import os
import random
import collections
import numpy as np
import torch
from torch import nn

from open_spiel.python.pytorch import deep_cfr


class SingleDeepCFRSolver(deep_cfr.DeepCFRSolver):
    """SDCFR：保存每轮 advantage 网络，决策时随机采样。"""

    def __init__(self, game, save_dir="./sdcfr_checkpoints",
                 advantage_network_layers=(128, 128), **kwargs):
        super().__init__(game, **kwargs)
        self._save_dir = save_dir
        self._adv_layers = tuple(advantage_network_layers)
        os.makedirs(save_dir, exist_ok=True)

    # ---------- 工具方法 ----------
    def _as_float_tensor(self, value):
        if isinstance(value, torch.Tensor):
            return value.to(device=self._device, dtype=torch.float32)
        return torch.as_tensor(
            np.asarray(value, dtype=np.float32), device=self._device)

    def _matched_regrets_from_raw_advantages(self, raw_advantages, legal_actions):
        legal_action_tensor = torch.as_tensor(
            legal_actions, dtype=torch.long, device=raw_advantages.device)
        positive_advantages = torch.clamp(raw_advantages, min=0.0)
        legal_positive = positive_advantages.index_select(
            0, legal_action_tensor)
        matched_regrets = torch.zeros(
            self._num_actions, dtype=raw_advantages.dtype,
            device=raw_advantages.device)
        cumulative = legal_positive.sum()
        if cumulative.item() > 0.0:
            matched_regrets.index_copy_(
                0, legal_action_tensor, legal_positive / cumulative)
        else:
            best_pos = torch.argmax(
                raw_advantages.index_select(0, legal_action_tensor))
            matched_regrets[legal_action_tensor[best_pos]] = 1.0
        return positive_advantages, matched_regrets

    # ---------- 训练 ----------
    def solve(self):
        """训练并保存每轮 advantage 网络。"""
        advantage_losses = collections.defaultdict(list)
        for _ in range(self._num_iterations):
            for p in range(self._num_players):
                for _ in range(self._num_traversals):
                    self._traverse_game_tree(self._root_node, p)
                if self._reinitialize_advantage_networks:
                    self._reinitialize_advantage_network(p)
                advantage_losses[p].append(self._learn_advantage_network(p))
            self._save_advantage_networks(self._iteration)
            self._iteration += 1
            if self._iteration % 50 == 0:
                print(f"  SDCFR iteration {self._iteration}")
        return advantage_losses

    def _save_advantage_networks(self, iteration):
        for p in range(self._num_players):
            path = os.path.join(self._save_dir,
                                f"adv_iter{iteration}_p{p}.pt")
            torch.save(self._advantage_networks[p].state_dict(), path)

    # ---------- 决策 ----------
    def _sample_advantage_network(self, player):
        iterations = list(range(1, self._iteration))
        if not iterations:
            return self._advantage_networks[player]
        chosen = random.choices(iterations, weights=iterations, k=1)[0]
        path = os.path.join(self._save_dir,
                            f"adv_iter{chosen}_p{player}.pt")
        net = type(self._advantage_networks[player])(
            self._embedding_size,
            list(self._adv_layers),
            self._num_actions,
        )
        net.load_state_dict(torch.load(path, map_location=self._device))
        net.to(self._device)
        net.eval()
        return net

    @torch.inference_mode()
    def action_probabilities(self, state, player_id=None):
        player = state.current_player()
        net = self._sample_advantage_network(player)
        info = state.information_state_tensor(player)
        legal = state.legal_actions(player)
        state_tensor = self._as_float_tensor(
            np.expand_dims(info, axis=0))
        raw = net(state_tensor)[0]
        _, matched_regrets = self._matched_regrets_from_raw_advantages(
            raw, legal)
        return {a: float(matched_regrets[a].item()) for a in legal}

    @torch.inference_mode()
    def average_policy_probabilities(self, state, player_id=None):
        """所有 advantage 网络的加权平均策略（用于 exploitability）。"""
        player = state.current_player()
        info = state.information_state_tensor(player)
        legal = state.legal_actions(player)
        state_tensor = self._as_float_tensor(np.expand_dims(info, axis=0))

        total_probs = np.zeros(self._num_actions)
        total_weight = 0.0
        for it in range(1, self._iteration):
            path = os.path.join(self._save_dir,
                                f"adv_iter{it}_p{player}.pt")
            if not os.path.exists(path):
                continue
            net = type(self._advantage_networks[player])(
                self._embedding_size,
                list(self._adv_layers),
                self._num_actions,
            )
            net.load_state_dict(torch.load(path, map_location=self._device))
            net.to(self._device)
            net.eval()
            raw = net(state_tensor)[0]
            _, mr = self._matched_regrets_from_raw_advantages(raw, legal)
            total_probs += it * mr.cpu().numpy()
            total_weight += it
        if total_weight > 0:
            total_probs /= total_weight
        return {a: float(total_probs[a]) for a in legal}