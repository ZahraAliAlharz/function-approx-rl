import numpy as np

from function_approx_rl.agents import SemiGradientSarsa, TileCoder
from function_approx_rl.envs import MountainCar
from function_approx_rl.metrics import run_episode, train

POSITION_RANGE = (-1.2, 0.5)
VELOCITY_RANGE = (-0.07, 0.07)


def make_agent(epsilon=0.0, alpha=0.3, seed=0):
    tc = TileCoder(num_tilings=8, tiles_per_dim=8, value_ranges=[POSITION_RANGE, VELOCITY_RANGE])
    return SemiGradientSarsa(tc, n_actions=3, epsilon=epsilon, alpha=alpha, gamma=1.0, seed=seed)


def test_q_values_start_at_zero():
    agent = make_agent()
    assert np.allclose(agent.q_values((0.0, 0.0)), 0.0)


def test_select_action_in_range():
    agent = make_agent(epsilon=0.1)
    for _ in range(50):
        action = agent.select_action((0.0, 0.0))
        assert action in (0, 1, 2)


def test_update_changes_weights_for_active_tiles_only():
    agent = make_agent()
    tiles_before = agent.tc.active_tiles((0.0, 0.0))
    agent.update((0.0, 0.0), action=2, reward=-1.0, next_state=(0.01, 0.01), next_action=2, done=False)
    assert not np.allclose(agent.weights[2, tiles_before], 0.0)
    other_action_tiles = agent.weights[0, tiles_before]
    assert np.allclose(other_action_tiles, 0.0)  # untouched action/tile combination


def test_single_episode_terminates_within_max_steps():
    env = MountainCar(seed=0)
    agent = make_agent(epsilon=0.1)
    steps = run_episode(env, agent, max_steps=2000)
    assert 0 < steps <= 2000


def test_training_improves_steps_to_goal():
    env = MountainCar(seed=0)
    agent = make_agent(epsilon=0.0, alpha=0.5, seed=0)
    steps = train(env, agent, episodes=100, max_steps=2000)
    # Early episodes should be slow (optimistic/zero-initialized weights
    # make every action look equally good); later episodes should be
    # meaningfully faster once the agent has learned to pump momentum.
    assert steps[:5].mean() > steps[-20:].mean()
    assert steps[-20:].mean() < 500
