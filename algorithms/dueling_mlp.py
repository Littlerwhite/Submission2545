"""Dueling advantage network definitions."""

import math

from scipy import stats
import torch
from torch import nn
import torch.nn.functional as F


class SonnetLinear(nn.Module):
    def __init__(self, in_size, out_size, activate_relu=True):
        super(SonnetLinear, self).__init__()
        self._activate_relu = activate_relu
        self._in_size = in_size
        self._out_size = out_size
        self._weight = None
        self._bias = None
        self.reset()

    def forward(self, tensor):
        y = F.linear(tensor, self._weight, self._bias)
        return F.relu(y) if self._activate_relu else y

    def reset(self):
        stddev = 1.0 / math.sqrt(self._in_size)
        mean = 0
        lower = (-2 * stddev - mean) / stddev
        upper = (2 * stddev - mean) / stddev
        self._weight = nn.Parameter(
            torch.Tensor(
                stats.truncnorm.rvs(
                    lower,
                    upper,
                    loc=mean,
                    scale=stddev,
                    size=[self._out_size, self._in_size],
                )
            )
        )
        self._bias = nn.Parameter(torch.zeros([self._out_size]))


class DuelingMLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size,
                 final_activation=None, seed=None):
        super(DuelingMLP, self).__init__()
        self._layers = []
        for size in hidden_sizes[:-1]:
            self._layers.append(SonnetLinear(in_size=input_size, out_size=size))
            input_size = size
        self.state_layer = SonnetLinear(in_size=input_size, out_size=hidden_sizes[-1])
        self.adv_layer = SonnetLinear(in_size=input_size, out_size=hidden_sizes[-1])
        input_size = hidden_sizes[-1]
        self.state = SonnetLinear(in_size=input_size, out_size=1, activate_relu=False)
        self.adv = SonnetLinear(
            in_size=input_size,
            out_size=output_size,
            activate_relu=False,
        )
        self.model = nn.ModuleList(self._layers)

    def forward(self, x):
        for layer in self.model:
            x = layer(x)

        advantage = self.adv(self.adv_layer(x))
        advantage = advantage - advantage.mean(dim=1, keepdim=True)
        state = self.state(self.state_layer(x)).expand_as(advantage)
        x = state + advantage

        return x

    def reset(self):
        for layer in self._layers:
            layer.reset()
