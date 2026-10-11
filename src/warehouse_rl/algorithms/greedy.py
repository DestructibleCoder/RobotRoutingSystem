"""Greedy task-allocation baseline.

The greedy policy is the classical non-learning reference point. Two variants
are provided:

``min_completion_time`` (default)
    Process pending orders oldest-first; for each, assign the idle robot whose
    estimated completion time is smallest.

``nearest_robot``
    For each idle robot, assign the pending order whose pickup is nearest.

Neither variant learns: they are deterministic functions of the current state
and are evaluated with the exact same metrics as the RL policies.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Set, Tuple

from warehouse_rl.config import GreedyConfig
from warehouse_rl.environment.orders import Order
from warehouse_rl.environment.robot import Robot
from warehouse_rl.environment.warehouse import Assignment, WarehouseSimulation

_INF = float("inf")


class GreedyPolicy:
    """Deterministic greedy allocation policy."""

    name = "greedy"

    def __init__(self, config: Optional[GreedyConfig] = None) -> None:
        self.config = config or GreedyConfig()
        self.strategy = self.config.strategy

    # -- estimation -------------------------------------------------------- #
    def estimate_completion_time(
        self, sim: WarehouseSimulation, robot: Robot, order: Order
    ) -> Optional[float]:
        """Estimated steps for ``robot`` to complete ``order`` if assigned now.

        ``None`` when either leg is unreachable from the current layout.
        """
        to_pickup = sim.oracle.distance(robot.position, order.pickup)
        if to_pickup is None:
            return None
        to_dropoff = sim.oracle.distance(order.pickup, order.dropoff)
        if to_dropoff is None:
            return None
        wcfg = sim.config.warehouse
        return to_pickup + wcfg.pick_dwell_steps + to_dropoff + wcfg.drop_dwell_steps

    # -- policy interface -------------------------------------------------- #
    def assign(self, sim: WarehouseSimulation) -> List[Assignment]:
        if self.strategy == "nearest_robot":
            return self._assign_nearest_robot(sim)
        return self._assign_min_completion_time(sim)

    def _assign_min_completion_time(self, sim: WarehouseSimulation) -> List[Assignment]:
        pending = sim.pending_orders()  # oldest first
        available: Set[int] = {r.id for r in sim.idle_robots()}
        assignments: List[Assignment] = []
        for order in pending:
            if not available:
                break
            best_robot: Optional[int] = None
            best_cost = _INF
            for robot_id in sorted(available):
                robot = sim.robots[robot_id]
                cost = self.estimate_completion_time(sim, robot, order)
                if cost is None:
                    continue
                if cost < best_cost:
                    best_cost = cost
                    best_robot = robot_id
            if best_robot is not None:
                assignments.append((best_robot, order.id))
                available.discard(best_robot)
        return assignments

    def _assign_nearest_robot(self, sim: WarehouseSimulation) -> List[Assignment]:
        pending = sim.pending_orders()
        available: Set[int] = {r.id for r in sim.idle_robots()}
        assignments: List[Assignment] = []
        for robot_id in sorted(available):
            robot = sim.robots[robot_id]
            best_order: Optional[int] = None
            best_dist = _INF
            for order in pending:
                dist = sim.oracle.distance(robot.position, order.pickup)
                if dist is None:
                    continue
                if dist < best_dist:
                    best_dist = dist
                    best_order = order.id
            if best_order is not None:
                assignments.append((robot_id, best_order))
                pending = [o for o in pending if o.id != best_order]
        return assignments


def make_policy(name: str, config: Optional[GreedyConfig] = None):
    """Factory dispatching on algorithm name.

    Only the greedy baseline exists today. ``ppo`` and ``dqn`` raise a clear
    error until Milestones 4-6 implement them.
    """
    if name == "greedy":
        return GreedyPolicy(config)
    if name in ("ppo", "dqn"):
        raise NotImplementedError(
            f"Policy '{name}' is not implemented yet (see Milestones 4-6 in the project brief)."
        )
    raise ValueError(f"Unknown policy '{name}'")
