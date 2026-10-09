# Copyright 2019 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Implements Deep CFR Algorithm (Dueling MLP variant).

See https://arxiv.org/abs/1811.00164.
"""

# pylint: disable=g-explicit-length-test

import collections
from typing import Iterable, NamedTuple

import numpy as np
import torch
from torch import nn
import tree as np_tree

from open_spiel.python import policy
from open_spiel.python.algorithms import exploitability
import pyspiel
from algorithms.dueling_mlp import DuelingMLP


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class AdvantageMemory(NamedTuple):
    """Advantage network memory buffer."""
    info_state: np.ndarray
    iteration: np.ndarray
    advantage: np.ndarray


class StrategyMemory(NamedTuple):
    """Strategy network memory buffer."""
    info_state: np.ndarray
    iteration: np.ndarray
    strategy_action_probs: np.ndarray


class MLP(nn.Module):
    """A simple network built from nn.linear layers."""

    def __init__(
        self,
        input_size: int,
        hidden_sizes: Iterable[int],
        output_size: int,
        final_activation: nn.Module = None,
        seed: int = 42,
    ) -> None:
        super().__init__()
        set_seed(seed)

        layers_ = []

        def _create_linear_block(in_features, out_features):
            return nn.Sequential(nn.Linear(in_features, out_features), nn.ReLU())

        for size in hidden_sizes:
            layers_.append(_create_linear_block(input_size, size))
            input_size = size
        layers_.append(nn.LayerNorm(input_size))
        layers_.append(nn.Linear(input_size, output_size))
        if final_activation:
            layers_.append(final_activation)
        self.model = nn.Sequential(*layers_)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def reset(self):
        @torch.no_grad()
        def weight_reset(m: nn.Module):
            reset_parameters = getattr(m, "reset_parameters", None)
            if callable(reset_parameters):
                m.reset_parameters()
        self.apply(fn=weight_reset)


class ReservoirBuffer:
    """Allows uniform sampling over a stream of data."""

    def __init__(self, capacity: np.ndarray,
                 experience: AdvantageMemory):
        self.capacity = capacity
        self.experience = experience
        self.add_calls = np.array(0)

    def __len__(self) -> int:
        return min(self.add_calls.item(), self.capacity.item())

    def __getitem__(self, idx):
        return np_tree.map_structure(lambda data: data[idx], self.experience)

    @classmethod
    def init(cls, capacity: int,
             experience: AdvantageMemory) -> "ReservoirBuffer":
        experience_ = np_tree.map_structure(
            lambda x: np.empty((capacity, *x.shape), dtype=x.dtype),
            experience
        )
        return cls(np.array(capacity), experience_)

    def append(self, experience) -> None:
        idx = np.random.randint(0, self.add_calls + 1)
        is_full = self.add_calls >= self.capacity
        write_idx = np.where(is_full, idx, self.add_calls)
        should_update = write_idx < self.capacity

        def _inplace(arr, idx, val):
            arr[idx] = val

        if should_update:
            np_tree.map_structure(
                lambda buf_leaf, exp_leaf: _inplace(buf_leaf, write_idx, exp_leaf),
                self.experience,
                experience,
            )
        self.add_calls += 1

    def sample(self, num_samples: int):
        max_size = len(self)
        if max_size < num_samples:
            raise ValueError(
                f"{num_samples} elements could not be sampled from size {max_size}"
            )
        indices = np.random.choice(max_size, size=(num_samples,), replace=False)
        return np_tree.map_structure(lambda data: data[indices], self.experience)

    def shuffle(self) -> None:
        np_tree.map_structure(
            lambda x: np.random.shuffle(x[: len(self)]), self.experience
        )

    def clear(self) -> None:
        self.add_calls = np.array(0)


class DeepCFRSolver(policy.Policy):
    """Deep CFR with Dueling MLP advantage networks."""

    def __init__(
        self,
        game,
        policy_network_layers=(256, 256),
        advantage_network_layers=(128, 128),
        num_iterations: int = 100,
        num_traversals: int = 20,
        learning_rate: float = 1e-4,
        batch_size_advantage: int = None,
        batch_size_strategy: int = None,
        memory_capacity: int = int(1e6),
        policy_network_train_steps: int = 1,
        advantage_network_train_steps: int = 1,
        reinitialize_advantage_networks: bool = True,
        device: str = "cpu",
        seed: int = 42,
        print_nash_convs: bool = False,
    ) -> None:
        all_players = list(range(game.num_players()))
        super(DeepCFRSolver, self).__init__(game, all_players)

        self._game = game
        if game.get_type().dynamics == pyspiel.GameType.Dynamics.SIMULTANEOUS:
            raise ValueError("Simulatenous games are not supported.")
        self._batch_size_advantage = batch_size_advantage
        self._batch_size_strategy = batch_size_strategy
        self._policy_network_train_steps = policy_network_train_steps
        self._advantage_network_train_steps = advantage_network_train_steps
        self._num_players = game.num_players()
        self._root_node = self._game.new_initial_state()
        self._embedding_size = len(self._root_node.information_state_tensor(0))
        self._num_iterations = num_iterations
        self._num_traversals = num_traversals
        self._memory_capacity = int(memory_capacity)
        self._reinitialize_advantage_networks = reinitialize_advantage_networks
        self._num_actions = game.num_distinct_actions()
        self._iteration = 1
        self._learning_rate = learning_rate
        self._print_nash_convs = print_nash_convs
        self._device = torch.device(device)

        # ★ Dueling MLP for advantage networks
        self._advantage_memories = [None] * self._num_players
        self._advantage_networks = [
            DuelingMLP(
                self._embedding_size,
                list(advantage_network_layers),
                self._num_actions,
                None,
                seed + p,
            ).to(self._device)
            for p in range(self._num_players)
        ]

        # ★ Standard MLP for policy network
        self._strategy_memories = None
        self._policy_network = MLP(
            self._embedding_size,
            list(policy_network_layers),
            self._num_actions,
            nn.Softmax(-1),
            seed,
        ).to(self._device)

        self._loss_policy = nn.MSELoss()
        self._optimizer_policy = None
        self._reinitialize_policy_network()

        self._loss_advantages = nn.MSELoss(reduction="mean")
        self._optimizer_advantages = [None] * self._num_players
        for p in range(self._num_players):
            self._reinitialize_advantage_network(p)

    def _get_buffer_init(self, capacity: int, data) -> ReservoirBuffer:
        return ReservoirBuffer.init(capacity, data)

    @property
    def advantage_buffers(self):
        return self._advantage_memories

    @property
    def strategy_buffer(self):
        return self._strategy_memories

    def _reinitialize_advantage_network(self, player):
        self._advantage_networks[player].reset()
        self._optimizer_advantages[player] = torch.optim.Adam(
            self._advantage_networks[player].parameters(), lr=self._learning_rate)

    def _reinitialize_policy_network(self):
        self._policy_network.reset()
        self._optimizer_policy = torch.optim.Adam(
            self._policy_network.parameters(), lr=self._learning_rate
        )

    def solve(self):
        advantage_losses = collections.defaultdict(list)
        for _ in range(self._num_iterations):
            if self._print_nash_convs:
                policy_loss = self._learn_strategy_network()
                average_policy = policy.tabular_policy_from_callable(
                    self._game, self.action_probabilities
                )
                conv = exploitability.nash_conv(self._game, average_policy)
                print(f"NashConv @ {self._iteration} = {conv} | "
                      f"Policy loss = {policy_loss}")
                self._reinitialize_policy_network()

            for p in range(self._num_players):
                for _ in range(self._num_traversals):
                    self._traverse_game_tree(self._root_node, p)
                if self._reinitialize_advantage_networks:
                    self._reinitialize_advantage_network(p)
                advantage_losses[p].append(self._learn_advantage_network(p))
            self._iteration += 1
        policy_loss = self._learn_strategy_network()
        return self._policy_network, advantage_losses, policy_loss

    def _append_to_stategy_buffer(self, data: StrategyMemory) -> None:
        if self._strategy_memories is None:
            self._strategy_memories = self._get_buffer_init(
                self._memory_capacity, data)
        self._strategy_memories.append(data)

    def _append_to_advantage_buffer(self, player: int,
                                    data: AdvantageMemory) -> None:
        if self._advantage_memories[player] is None:
            self._advantage_memories[player] = self._get_buffer_init(
                self._memory_capacity, data)
        self._advantage_memories[player].append(data)

    def _traverse_game_tree(self, state, player):
        expected_payoff = collections.defaultdict(float)
        if state.is_terminal():
            return state.returns()[player]
        elif state.is_chance_node():
            chance_outcome, chance_proba = zip(*state.chance_outcomes())
            action = np.random.choice(chance_outcome, p=chance_proba)
            return self._traverse_game_tree(state.child(action), player)
        elif state.current_player() == player:
            sampled_regret = collections.defaultdict(float)
            _, strategy = self._sample_action_from_advantage(state, player)
            for action in state.legal_actions():
                expected_payoff[action] = self._traverse_game_tree(
                    state.child(action), player)
            cfv = 0
            for a_ in state.legal_actions():
                cfv += strategy[a_] * expected_payoff[a_]
            for action in state.legal_actions():
                sampled_regret[action] = expected_payoff[action]
                sampled_regret[action] -= cfv
            sampled_regret_arr = [0] * self._num_actions
            for action in sampled_regret:
                sampled_regret_arr[action] = sampled_regret[action]

            data = AdvantageMemory(
                np.array(state.information_state_tensor(), dtype=np.float32),
                np.array(self._iteration, dtype=int).reshape(1,),
                np.array(sampled_regret_arr, dtype=np.float32),
            )
            self._append_to_advantage_buffer(player, data)
            return cfv
        else:
            other_player = state.current_player()
            _, strategy = self._sample_action_from_advantage(state, other_player)
            probs = np.array(strategy)
            probs /= probs.sum()
            sampled_action = np.random.choice(range(self._num_actions), p=probs)

            data = StrategyMemory(
                np.array(state.information_state_tensor(other_player),
                         dtype=np.float32),
                np.array(self._iteration, dtype=int).reshape(1,),
                np.array(strategy, dtype=np.float32),
            )
            self._append_to_stategy_buffer(data)
            return self._traverse_game_tree(state.child(sampled_action), player)

    @torch.inference_mode()
    def _sample_action_from_advantage(self, state, player):
        info_state = state.information_state_tensor(player)
        legal_actions = state.legal_actions(player)
        with torch.no_grad():
            state_tensor = torch.FloatTensor(
                np.expand_dims(info_state, axis=0), device=self._device)
            raw_advantages = (
                self._advantage_networks[player](state_tensor)[0].cpu().numpy()
            )
        advantages = [max(0., a) for a in raw_advantages]
        cumulative_regret = np.sum([advantages[a] for a in legal_actions])
        matched_regrets = np.array([0.] * self._num_actions)
        if cumulative_regret > 0.:
            for action in legal_actions:
                matched_regrets[action] = advantages[action] / cumulative_regret
        else:
            matched_regrets[max(legal_actions,
                                key=lambda a: raw_advantages[a])] = 1
        return advantages, matched_regrets

    @torch.inference_mode()
    def action_probabilities(self, state, player_id=None):
        del player_id
        cur_player = state.current_player()
        legal_actions = state.legal_actions(cur_player)
        info_state_vector = np.array(state.information_state_tensor())
        if len(info_state_vector.shape) == 1:
            info_state_vector = np.expand_dims(info_state_vector, axis=0)
        probs = (
            self._policy_network(
                torch.FloatTensor(info_state_vector, device=self._device)
            ).cpu().numpy()
        )
        return {action: probs[0][action] for action in legal_actions}

    def _learn_advantage_network(self, player):
        for _ in range(self._advantage_network_train_steps):

            if self._advantage_memories[player] is None:
                return None

            if self._batch_size_advantage:
                if self._batch_size_advantage > len(
                        self._advantage_memories[player]):
                    return None
                samples = self._advantage_memories[player].sample(
                    self._batch_size_advantage)
            else:
                self._advantage_memories[player].shuffle()
                samples = self._advantage_memories[player].experience

            if len(samples.info_state) == 0:
                return None

            self._optimizer_advantages[player].zero_grad()
            iters = torch.FloatTensor(samples.iteration,
                                      device=self._device).sqrt()
            outputs = self._advantage_networks[player](
                torch.FloatTensor(samples.info_state, device=self._device)
            )
            advantages = torch.FloatTensor(samples.advantage,
                                          device=self._device)
            loss_advantages = self._loss_advantages(
                iters * outputs, iters * advantages)
            loss_advantages.backward()
            self._optimizer_advantages[player].step()

        return loss_advantages.detach().cpu().item()

    def _learn_strategy_network(self):
        if self._strategy_memories is None:
            return None

        for _ in range(self._policy_network_train_steps):
            if self._batch_size_strategy:
                if self._batch_size_strategy > len(self._strategy_memories):
                    return None
                samples = self._strategy_memories.sample(
                    self._batch_size_strategy)
            else:
                self._strategy_memories.shuffle()
                samples = self._strategy_memories.experience

            if len(samples.info_state) == 0:
                return None

            self._optimizer_policy.zero_grad()
            iters = torch.FloatTensor(samples.iteration,
                                      device=self._device).sqrt()
            outputs = self._policy_network(
                torch.FloatTensor(samples.info_state, device=self._device)
            )
            ac_probs = torch.FloatTensor(
                samples.strategy_action_probs, device=self._device).squeeze()
            loss_strategy = self._loss_policy(iters * outputs, iters * ac_probs)
            loss_strategy.backward()
            self._optimizer_policy.step()

        return loss_strategy.detach().cpu().item()