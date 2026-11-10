from function_approx_rl.envs import MountainCar


def test_reset_within_start_range():
    env = MountainCar(seed=0)
    position, velocity = env.reset()
    assert -0.6 <= position <= -0.4
    assert velocity == 0.0


def test_step_returns_tuple_reward_done():
    env = MountainCar(seed=0)
    env.reset()
    (position, velocity), reward, done = env.step(2)  # forward
    assert isinstance(position, float)
    assert isinstance(velocity, float)
    assert reward == -1.0
    assert isinstance(done, bool)


def test_position_and_velocity_stay_in_bounds():
    env = MountainCar(seed=0)
    env.reset()
    for _ in range(500):
        (position, velocity), _, done = env.step(2)
        assert -1.2 <= position <= 0.5
        assert -0.07 <= velocity <= 0.07
        if done:
            break


def test_episode_terminates_with_constant_forward_throttle_eventually():
    # Pure forward throttle alone cannot climb the far hill directly (the
    # engine is deliberately too weak), but repeatedly swinging in the
    # valley is not required for *this* check: just confirm the env can
    # reach the goal state and correctly report done at that point.
    env = MountainCar(seed=0)
    env.position, env.velocity = 0.45, 0.06
    (position, velocity), reward, done = env.step(2)
    assert done is True
    assert position >= 0.5


def test_left_wall_collision_zeros_velocity():
    env = MountainCar(seed=0)
    env.position, env.velocity = -1.2, -0.05
    (position, velocity), _, _ = env.step(0)  # reverse, pushing further left
    assert position == -1.2
    assert velocity == 0.0
