import numpy as np

from function_approx_rl.agents import TileCoder


def test_feature_vector_has_one_active_tile_per_tiling():
    tc = TileCoder(num_tilings=8, tiles_per_dim=8, value_ranges=[(-1.2, 0.5), (-0.07, 0.07)])
    tiles = tc.active_tiles((0.0, 0.0))
    assert len(tiles) == 8
    assert len(set(tiles)) == 8  # each tiling contributes a distinct global index


def test_feature_vector_length_matches_expected_size():
    tc = TileCoder(num_tilings=8, tiles_per_dim=8, value_ranges=[(-1.2, 0.5), (-0.07, 0.07)])
    assert tc.n_features == 8 * 8 * 8  # num_tilings * tiles_per_dim ** n_dims
    vec = tc.features((0.0, 0.0))
    assert vec.shape == (tc.n_features,)
    assert vec.sum() == 8


def test_nearby_states_share_most_active_tiles():
    tc = TileCoder(num_tilings=8, tiles_per_dim=8, value_ranges=[(-1.2, 0.5), (-0.07, 0.07)])
    tiles_a = set(tc.active_tiles((0.0, 0.0)))
    tiles_b = set(tc.active_tiles((0.001, 0.0001)))  # a tiny nudge
    assert len(tiles_a & tiles_b) >= 6  # most tilings should still agree


def test_distant_states_share_no_active_tiles():
    tc = TileCoder(num_tilings=8, tiles_per_dim=8, value_ranges=[(-1.2, 0.5), (-0.07, 0.07)])
    tiles_a = set(tc.active_tiles((-1.1, -0.06)))
    tiles_b = set(tc.active_tiles((0.4, 0.06)))
    assert len(tiles_a & tiles_b) == 0


def test_indices_stay_within_bounds_at_range_edges():
    tc = TileCoder(num_tilings=8, tiles_per_dim=8, value_ranges=[(-1.2, 0.5), (-0.07, 0.07)])
    for state in [(-1.2, -0.07), (0.5, 0.07), (-1.2, 0.07), (0.5, -0.07)]:
        tiles = tc.active_tiles(state)
        assert all(0 <= t < tc.n_features for t in tiles)
