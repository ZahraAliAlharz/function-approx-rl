"""Episode training loop for semi-gradient SARSA on Mountain Car."""
from __future__ import annotations

import numpy as np

from function_approx_rl.agents.semi_gradient_sarsa import SemiGradientSarsa
from function_approx_rl.envs.mountain_car import MountainCar


def run_episode(env: MountainCar, agent: SemiGradientSarsa, max_steps: int = 2000) -> int:
    """Run one episode; returns the number of steps taken (= -return, since reward is -1/step)."""
    state = env.reset()
    action = agent.select_action(state)
    for step in range(1, max_steps + 1):
        next_state, reward, done = env.step(action)
        next_action = agent.select_action(next_state)
        agent.update(state, action, reward, next_state, next_action, done)
        if done:
            return step
        state, action = next_state, next_action
    return max_steps


def train(env: MountainCar, agent: SemiGradientSarsa, episodes: int, max_steps: int = 2000) -> np.ndarray:
    steps_per_episode = np.zeros(episodes, dtype=int)
    for i in range(episodes):
        steps_per_episode[i] = run_episode(env, agent, max_steps)
    return steps_per_episode
