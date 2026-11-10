"""CLI entry point: train semi-gradient SARSA on Mountain Car."""
from __future__ import annotations

import argparse

from function_approx_rl.agents import SemiGradientSarsa, TileCoder
from function_approx_rl.envs import MountainCar
from function_approx_rl.metrics import train

POSITION_RANGE = (-1.2, 0.5)
VELOCITY_RANGE = (-0.07, 0.07)


def main():
    parser = argparse.ArgumentParser(description="Train semi-gradient SARSA with tile coding on Mountain Car.")
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--num-tilings", type=int, default=8)
    parser.add_argument("--tiles-per-dim", type=int, default=8)
    parser.add_argument("--epsilon", type=float, default=0.0)
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    env = MountainCar(seed=args.seed)
    tc = TileCoder(args.num_tilings, args.tiles_per_dim, [POSITION_RANGE, VELOCITY_RANGE])
    agent = SemiGradientSarsa(tc, n_actions=3, epsilon=args.epsilon, alpha=args.alpha, gamma=1.0, seed=args.seed)

    steps = train(env, agent, episodes=args.episodes)
    print(f"Steps per episode, first 5: {steps[:5]}")
    print(f"Steps per episode, last 5:  {steps[-5:]}")
    print(f"Mean steps, last 20 episodes: {steps[-20:].mean():.1f}")


if __name__ == "__main__":
    main()
