"""Tests for the A* navigation module."""

from __future__ import annotations

import numpy as np
import pytest

from warehouse_rl.navigation.astar import DistanceOracle, astar, path_cost, path_exists


@pytest.fixture
def open_grid() -> np.ndarray:
    return np.full((5, 5), True, dtype=bool)


def test_astar_open_grid_shortest_path(open_grid):
    path = astar(open_grid, (0, 0), (0, 4), connectivity=4)
    assert path == [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)]
    assert path_cost(path, connectivity=4) == 4


def test_astar_returns_none_for_blocked_endpoints(open_grid):
    blocked = open_grid.copy()
    blocked[2, 2] = False
    assert astar(blocked, (0, 0), (2, 2)) is None
    assert astar(open_grid, (0, 0), (0, 4), connectivity=4) is not None


def test_astar_detours_around_obstacle():
    grid = np.ones((5, 5), dtype=bool)
    grid[2, 1:4] = False  # horizontal wall
    path = astar(grid, (0, 0), (4, 4), connectivity=4)
    assert path is not None
    # Path must not step onto a blocked cell.
    for cell in path:
        assert grid[cell]
    assert path_cost(path, connectivity=4) == 8


def test_astar_trivial_path(open_grid):
    assert astar(open_grid, (2, 2), (2, 2), connectivity=4) == [(2, 2)]


def test_astar_unreachable_returns_none():
    grid = np.ones((3, 3), dtype=bool)
    grid[1, :] = False  # wall separating (0,0) from (2,2)
    assert astar(grid, (0, 0), (2, 2), connectivity=4) is None
    assert path_exists(grid, (0, 0), (2, 2)) is False


def test_connectivity8_uses_diagonals(open_grid):
    path4 = astar(open_grid, (0, 0), (4, 4), connectivity=4)
    path8 = astar(open_grid, (0, 0), (4, 4), connectivity=8)
    assert path4 is not None and path8 is not None
    assert path_cost(path8, connectivity=8) < path_cost(path4, connectivity=4)


def test_astar_out_of_bounds_returns_none(open_grid):
    assert astar(open_grid, (-1, 0), (2, 2)) is None
    assert astar(open_grid, (2, 2), (99, 99)) is None


def test_distance_oracle_caches_and_is_symmetric(open_grid):
    oracle = DistanceOracle(open_grid, connectivity=4)
    assert oracle.distance((0, 0), (0, 3)) == 3.0
    assert oracle.distance((0, 3), (0, 0)) == 3.0  # symmetric
    assert oracle.distance((0, 0), (0, 3)) == 3.0  # cached path
    assert oracle.reachable((0, 0), (4, 4))
    # Cache: both the direct and symmetric reverse entries were stored.
    assert len(oracle._dist_cache) == 4


def test_distance_oracle_unreachable(open_grid):
    grid = open_grid.copy()
    grid[1, :] = False
    oracle = DistanceOracle(grid, connectivity=4)
    assert oracle.distance((0, 0), (2, 2)) is None
    assert not oracle.reachable((0, 0), (2, 2))