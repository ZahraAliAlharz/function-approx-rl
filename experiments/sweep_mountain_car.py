"""Learning curve for semi-gradient SARSA + tile coding on Mountain Car,
and a step-size sweep reproducing the step-size sensitivity Sutton & Barto
(2018), Figure 10.2 shows for this exact combination of method and problem.

Generates results/mountain_car_learning_curve.png and
results/mountain_car_step_size_sweep.png.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from function_approx_rl.agents import SemiGradientSarsa, TileCoder
from function_approx_rl.envs import MountainCar
from function_approx_rl.metrics import train

POSITION_RANGE = (-1.2, 0.5)
VELOCITY_RANGE = (-0.07, 0.07)
NUM_TILINGS = 8
TILES_PER_DIM = 8
EPISODES = 150
SEEDS = range(10)
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def make_agent(alpha: float, epsilon: float, seed: int) -> SemiGradientSarsa:
    tc = TileCoder(NUM_TILINGS, TILES_PER_DIM, [POSITION_RANGE, VELOCITY_RANGE])
    return SemiGradientSarsa(tc, n_actions=3, epsilon=epsilon, alpha=alpha, gamma=1.0, seed=seed)


def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    # 1. Learning curve at a single, reasonable step size.
    all_steps = []
    for seed in SEEDS:
        env = MountainCar(seed=seed)
        agent = make_agent(alpha=0.3, epsilon=0.0, seed=seed)
        all_steps.append(train(env, agent, episodes=EPISODES))
    mean_steps = np.mean(all_steps, axis=0)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(mean_steps)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Steps to reach the goal")
    ax.set_yscale("log")
    ax.set_title(f"Mountain Car: semi-gradient SARSA learning curve (mean over {len(SEEDS)} seeds)")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "mountain_car_learning_curve.png", dpi=150)

    print(f"Learning curve: steps first 5 episodes mean={mean_steps[:5].mean():.0f}, "
          f"last 20 episodes mean={mean_steps[-20:].mean():.0f}")

    # 2. Step-size sweep: too-small alpha learns slowly, too-large alpha is unstable.
    alphas = [0.05, 0.1, 0.3, 0.5, 0.9, 1.5]
    final_perf = []
    for alpha in alphas:
        runs = []
        for seed in SEEDS:
            env = MountainCar(seed=seed)
            agent = make_agent(alpha=alpha, epsilon=0.0, seed=seed)
            steps = train(env, agent, episodes=EPISODES)
            runs.append(steps[-20:].mean())
        final_perf.append(np.mean(runs))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(alphas, final_perf, marker="o")
    ax.set_xlabel("Step size (alpha)")
    ax.set_ylabel("Mean steps to goal, last 20 of 200 episodes")
    ax.set_title(f"Mountain Car: step-size sensitivity (mean over {len(SEEDS)} seeds)")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "mountain_car_step_size_sweep.png", dpi=150)

    print("Step-size sweep (alpha -> mean steps to goal, last 20 episodes):")
    for alpha, perf in zip(alphas, final_perf):
        print(f"  alpha={alpha:<4} -> {perf:.1f}")


if __name__ == "__main__":
    main()
