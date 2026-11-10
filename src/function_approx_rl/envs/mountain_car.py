"""Mountain Car (Moore, 1990; Sutton & Barto, 2018, Example 10.1).

A car is stuck in a valley between two hills and cannot drive straight up
the far hill — the engine is too weak. It has to build momentum by
reversing up the near hill first. This is the standard introductory
continuous-state control problem: a tabular agent cannot represent its
2D (position, velocity) state space, which is exactly why it is used here
to motivate function approximation.

Actions: 0 = reverse, 1 = no throttle, 2 = forward.
"""
from __future__ import annotations

import numpy as np

MIN_POSITION, MAX_POSITION = -1.2, 0.5
MAX_SPEED = 0.07
GOAL_POSITION = 0.5
START_POSITION_RANGE = (-0.6, -0.4)


class MountainCar:
    def __init__(self, seed: int | None = None):
        self.rng = np.random.default_rng(seed)
        self.actions = (0, 1, 2)
        self.position = 0.0
        self.velocity = 0.0

    def reset(self) -> tuple[float, float]:
        self.position = self.rng.uniform(*START_POSITION_RANGE)
        self.velocity = 0.0
        return self.position, self.velocity

    def step(self, action: int) -> tuple[tuple[float, float], float, bool]:
        force = action - 1  # 0 -> -1 (reverse), 1 -> 0, 2 -> +1 (forward)
        velocity = self.velocity + 0.001 * force - 0.0025 * np.cos(3 * self.position)
        velocity = np.clip(velocity, -MAX_SPEED, MAX_SPEED)
        position = self.position + velocity
        position = np.clip(position, MIN_POSITION, MAX_POSITION)
        if position == MIN_POSITION and velocity < 0:
            velocity = 0.0  # inelastic collision with the left wall

        self.position, self.velocity = float(position), float(velocity)
        done = self.position >= GOAL_POSITION
        reward = -1.0
        return (self.position, self.velocity), reward, done
