"""Low-level grid path planning (A*).

The RL policy decides *which robot serves which order*; this module solves the
classical navigation sub-problem of getting a robot from one cell to another on
a static grid. A* with an admissible heuristic is optimal and fast here.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

Cell = Tuple[int, int]  # (row, col)

_MOVES4: Tuple[Cell, ...] = ((-1, 0), (1, 0), (0, -1), (0, 1))
_DIAGONALS: Tuple[Cell, ...] = ((-1, -1), (-1, 1), (1, -1), (1, 1))
_MOVES8: Tuple[Cell, ...] = _MOVES4 + _DIAGONALS
_DIAG_COST = 2.0 ** 0.5


def _moves(connectivity: int) -> Tuple[Tuple[Cell, ...], Dict[Cell, float]]:
    if connectivity == 4:
        return _MOVES4, {m: 1.0 for m in _MOVES4}
    if connectivity == 8:
        costs = {m: 1.0 for m in _MOVES4}
        costs.update({m: _DIAG_COST for m in _DIAGONALS})
        return _MOVES8, costs
    raise ValueError(f"connectivity must be 4 or 8, got {connectivity}")


def _heuristic(a: Cell, b: Cell, connectivity: int) -> float:
    dr = abs(a[0] - b[0])
    dc = abs(a[1] - b[1])
    if connectivity == 4:
        return float(dr + dc)
    # Octile distance: admissible for 8-connected grids with sqrt(2) diagonals.
    return float(max(dr, dc) + (_DIAG_COST - 1.0) * min(dr, dc))


def astar(
    walkable: np.ndarray,
    start: Cell,
    goal: Cell,
    *,
    connectivity: int = 4,
) -> Optional[List[Cell]]:
    """Return a minimum-cost path from ``start`` to ``goal`` (inclusive).

    Returns ``None`` if no path exists or either endpoint is blocked. The path
    includes both endpoints; a trivial path ``[start]`` is returned when
    ``start == goal``.
    """
    height, width = walkable.shape
    if not (0 <= start[0] < height and 0 <= start[1] < width):
        return None
    if not (0 <= goal[0] < height and 0 <= goal[1] < width):
        return None
    if not walkable[start] or not walkable[goal]:
        return None
    if start == goal:
        return [start]

    moves, costs = _moves(connectivity)
    open_heap: List[Tuple[float, float, Cell]] = [
        (_heuristic(start, goal, connectivity), 0.0, start)
    ]
    g_score: Dict[Cell, float] = {start: 0.0}
    came_from: Dict[Cell, Cell] = {}
    closed: set[Cell] = set()

    while open_heap:
        import heapq

        _, g, current = heapq.heappop(open_heap)
        if current == goal:
            return _reconstruct(came_from, current)
        if current in closed:
            continue
        closed.add(current)

        r, c = current
        for (dr, dc), step_cost in costs.items():
            nr, nc = r + dr, c + dc
            if not (0 <= nr < height and 0 <= nc < width):
                continue
            neighbor = (nr, nc)
            if not walkable[neighbor] or neighbor in closed:
                continue
            tentative = g + step_cost
            if tentative < g_score.get(neighbor, float("inf")):
                g_score[neighbor] = tentative
                came_from[neighbor] = current
                f = tentative + _heuristic(neighbor, goal, connectivity)
                import heapq

                heapq.heappush(open_heap, (f, tentative, neighbor))
    return None


def _reconstruct(came_from: Dict[Cell, Cell], current: Cell) -> List[Cell]:
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


def path_cost(path: Optional[Sequence[Cell]], *, connectivity: int = 4) -> Optional[float]:
    """Cost (in movement steps/units) of a path produced by :func:`astar`."""
    if path is None:
        return None
    if len(path) <= 1:
        return 0.0
    total = 0.0
    for prev, cur in zip(path, path[1:]):
        dr = abs(cur[0] - prev[0])
        dc = abs(cur[1] - prev[1])
        if connectivity == 4:
            total += dr + dc
        else:
            total += _DIAG_COST if (dr and dc) else float(dr + dc)
    return total


def path_exists(walkable: np.ndarray, start: Cell, goal: Cell, *, connectivity: int = 4) -> bool:
    return astar(walkable, start, goal, connectivity=connectivity) is not None


class DistanceOracle:
    """Memoized shortest-path distances over a *static* grid.

    Layouts never change during an episode, so distances can be cached across
    allocation decisions. This keeps the greedy baseline (which evaluates many
    robot/order pairs per step) cheap.
    """

    def __init__(self, walkable: np.ndarray, *, connectivity: int = 4) -> None:
        self.walkable = np.asarray(walkable, dtype=bool)
        self.connectivity = int(connectivity)
        self._dist_cache: Dict[Tuple[Cell, Cell], Optional[float]] = {}
        self._path_cache: Dict[Tuple[Cell, Cell], Optional[List[Cell]]] = {}

    def path(self, start: Cell, goal: Cell) -> Optional[List[Cell]]:
        key = (start, goal)
        if key not in self._path_cache:
            self._path_cache[key] = astar(
                self.walkable, start, goal, connectivity=self.connectivity
            )
        return self._path_cache[key]

    def distance(self, start: Cell, goal: Cell) -> Optional[float]:
        """Shortest-path cost, or ``None`` if unreachable. Symmetric for grids."""
        key = (start, goal)
        if key in self._dist_cache:
            return self._dist_cache[key]
        value = path_cost(self.path(start, goal), connectivity=self.connectivity)
        self._dist_cache[key] = value
        reverse = (goal, start)
        self._dist_cache.setdefault(reverse, value)
        return value

    def reachable(self, start: Cell, goal: Cell) -> bool:
        return self.distance(start, goal) is not None

    def clear(self) -> None:
        self._dist_cache.clear()
        self._path_cache.clear()
