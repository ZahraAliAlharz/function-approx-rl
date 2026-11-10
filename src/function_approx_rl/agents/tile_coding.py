"""Tile coding: a coarse-coded binary feature representation for continuous
state spaces (Sutton & Barto, 2018, §9.5.4).

Each tiling overlays a grid on the (normalized) state space, offset from
the others by a fraction of a tile width. A state activates exactly one
tile per tiling, so the resulting feature vector is sparse (``num_tilings``
active features out of ``num_tilings * tiles_per_dim**n_dims``) and
generalizes locally: nearby states share most of their active tiles,
distant states share none.
"""
from __future__ import annotations

import numpy as np


class TileCoder:
    def __init__(
        self,
        num_tilings: int,
        tiles_per_dim: int,
        value_ranges: list[tuple[float, float]],
    ):
        self.num_tilings = num_tilings
        self.tiles_per_dim = tiles_per_dim
        self.value_ranges = value_ranges
        self.n_dims = len(value_ranges)
        self.tiles_per_tiling = tiles_per_dim**self.n_dims
        self.n_features = num_tilings * self.tiles_per_tiling

    def active_tiles(self, state: tuple[float, ...]) -> list[int]:
        """Return the (sparse) list of ``num_tilings`` active feature indices for ``state``."""
        normalized = [
            (state[d] - lo) / (hi - lo) for d, (lo, hi) in enumerate(self.value_ranges)
        ]
        indices = []
        for tiling in range(self.num_tilings):
            offset = tiling / self.num_tilings / self.tiles_per_dim
            bin_indices = [
                min(self.tiles_per_dim - 1, max(0, int((x + offset) * self.tiles_per_dim)))
                for x in normalized
            ]
            flat = 0
            for b in bin_indices:
                flat = flat * self.tiles_per_dim + b
            indices.append(tiling * self.tiles_per_tiling + flat)
        return indices

    def features(self, state: tuple[float, ...]) -> np.ndarray:
        """Dense 0/1 feature vector version of ``active_tiles``, mostly for tests/inspection."""
        vec = np.zeros(self.n_features)
        vec[self.active_tiles(state)] = 1.0
        return vec
