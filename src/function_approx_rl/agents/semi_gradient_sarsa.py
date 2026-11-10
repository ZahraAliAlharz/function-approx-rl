"""Episodic semi-gradient SARSA with linear function approximation
(Sutton & Barto, 2018, §10.1), driven by a TileCoder's sparse features.

Q(s, a) is a linear function of the active tiles' weights for action a.
Since tile features are binary, the gradient of Q(s, a) with respect to
w[a] is just the indicator of which weights are active, so the usual
semi-gradient TD update reduces to: nudge the active tiles' weights for
the taken action toward the TD target, leave every other weight alone.
"""
from __future__ import annotations

import numpy as np

from function_approx_rl.agents.tile_coding import TileCoder


class SemiGradientSarsa:
    def __init__(
        self,
        tile_coder: TileCoder,
        n_actions: int,
        epsilon: float = 0.1,
        alpha: float = 0.5,
        gamma: float = 1.0,
        seed: int | None = None,
    ):
        self.tc = tile_coder
        self.n_actions = n_actions
        self.epsilon = epsilon
        # Standard convention: divide the overall step size across tilings,
        # since each tiling contributes one active feature per update.
        self.alpha = alpha / tile_coder.num_tilings
        self.gamma = gamma
        self.rng = np.random.default_rng(seed)
        self.weights = np.zeros((n_actions, tile_coder.n_features))

    def q_values(self, state) -> np.ndarray:
        tiles = self.tc.active_tiles(state)
        return self.weights[:, tiles].sum(axis=1)

    def select_action(self, state) -> int:
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))
        return int(np.argmax(self.q_values(state)))

    def update(self, state, action: int, reward: float, next_state, next_action: int, done: bool) -> None:
        tiles = self.tc.active_tiles(state)
        current_q = self.weights[action, tiles].sum()
        if done:
            target = reward
        else:
            next_tiles = self.tc.active_tiles(next_state)
            target = reward + self.gamma * self.weights[next_action, next_tiles].sum()
        error = target - current_q
        self.weights[action, tiles] += self.alpha * error
