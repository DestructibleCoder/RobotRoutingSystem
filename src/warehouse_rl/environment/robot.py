"""Homogeneous mobile robot model.

All robots share identical speed, capacity, and capabilities. A robot is a
small state machine driven by :class:`~warehouse_rl.environment.warehouse.WarehouseSimulation`:

``idle`` -> ``to_pickup`` -> ``picking`` -> ``to_dropoff`` -> ``dropping`` -> ``idle``

The RL policy never issues per-step movement commands; it only decides which
order a robot should serve. Movement along the A* path is deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

Cell = Tuple[int, int]

# Robot status constants (kept as plain strings for easy logging/CSV export).
IDLE = "idle"
TO_PICKUP = "to_pickup"
PICKING = "picking"
TO_DROPOFF = "to_dropoff"
DROPPING = "dropping"
BUSY_STATUSES = frozenset({TO_PICKUP, PICKING, TO_DROPOFF, DROPPING})


@dataclass
class Robot:
    """State of a single robot within an episode."""

    id: int
    position: Cell
    start_position: Cell
    status: str = IDLE
    order_id: Optional[int] = None

    # Current path (inclusive of the cell the robot is standing on) and cursor.
    path: List[Cell] = field(default_factory=list)
    path_index: int = 0

    # Countdown for pick/drop dwell operations.
    timer: int = 0

    # Cumulative statistics tracked across the episode.
    distance_travelled: float = 0.0
    busy_steps: int = 0
    orders_completed: int = 0

    @property
    def is_idle(self) -> bool:
        return self.status == IDLE

    @property
    def is_busy(self) -> bool:
        return self.status in BUSY_STATUSES

    @property
    def available(self) -> bool:
        """True when the robot can accept a new order."""
        return self.is_idle

    def reset(self) -> None:
        self.position = self.start_position
        self.status = IDLE
        self.order_id = None
        self.path = []
        self.path_index = 0
        self.timer = 0
        self.distance_travelled = 0.0
        self.busy_steps = 0
        self.orders_completed = 0

    def set_path(self, path: List[Cell]) -> None:
        self.path = list(path)
        self.path_index = 0

    @property
    def goal_reached(self) -> bool:
        return self.path_index >= len(self.path) - 1

    def step_along_path(self, *, connectivity: int = 4) -> bool:
        """Advance one grid move. Returns True when the path is exhausted."""
        if self.goal_reached:
            return True
        prev = self.path[self.path_index]
        self.path_index += 1
        nxt = self.path[self.path_index]
        diag = abs(nxt[0] - prev[0]) and abs(nxt[1] - prev[1])
        self.distance_travelled += (2.0 ** 0.5) if (diag and connectivity == 8) else 1.0
        self.position = nxt
        return self.goal_reached
