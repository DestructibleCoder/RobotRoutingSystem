"""Core grid warehouse simulation.

This module contains :class:`WarehouseSimulation`, a framework-agnostic
discrete-step simulator. It deliberately does *not* depend on Gymnasium: the
RL environment (Milestone 3) is a thin wrapper around this class, which keeps
the simulation testable and reusable.

Timing model
------------
Each call to :meth:`WarehouseSimulation.step` advances the world by one step:

1. New orders arrive (Bernoulli stream).
2. The allocation policy is queried (every ``decision_interval`` steps) and
   returns ``(robot_id, order_id)`` assignments.
3. Robots advance along their A* paths / service timers.
4. Unassigned orders older than ``timeout_steps`` expire.

Assignments are validated: a robot must be idle, the order must be pending, and
a path from the robot to the pickup location must exist.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from warehouse_rl.config import ExperimentConfig
from warehouse_rl.environment.layout_generator import Layout
from warehouse_rl.environment.orders import (
    ASSIGNED,
    COMPLETED,
    PENDING,
    PICKED,
    TIMEOUT,
    Order,
    OrderGenerator,
)
from warehouse_rl.environment.robot import (
    DROPPING,
    IDLE,
    PICKING,
    TO_DROPOFF,
    TO_PICKUP,
    Robot,
)
from warehouse_rl.navigation.astar import DistanceOracle

Cell = Tuple[int, int]
Assignment = Tuple[int, int]  # (robot_id, order_id)

FAILED = "failed"


@dataclass
class StepResult:
    """Outcome of a single simulation step."""

    step: int
    assignments: List[Assignment] = field(default_factory=list)
    arrived_order_ids: List[int] = field(default_factory=list)
    completed_order_ids: List[int] = field(default_factory=list)
    timed_out_order_ids: List[int] = field(default_factory=list)
    failed_order_ids: List[int] = field(default_factory=list)
    done: bool = False
    truncated: bool = False


class WarehouseSimulation:
    """Discrete-event warehouse simulator with pluggable task allocation.

    Parameters
    ----------
    layout:
        Static geometry (walls/racks, pick stations, drop-offs, robot starts).
    config:
        An :class:`~warehouse_rl.config.ExperimentConfig` (or subclass).
    seed:
        Seed used for order arrivals and any other stochasticity.
    """

    def __init__(
        self,
        layout: Layout,
        config: ExperimentConfig,
        *,
        seed: int = 0,
    ) -> None:
        self.layout = layout
        self.config = config
        self.base_seed = int(seed)

        self.walkable: np.ndarray = layout.walkable_mask()
        self.pick_locations: List[Cell] = layout.pick_locations
        self.dropoff_locations: List[Cell] = layout.dropoff_locations
        self.start_positions: List[Cell] = layout.robot_starts
        self.oracle = DistanceOracle(self.walkable, connectivity=config.warehouse.connectivity)

        n_robots = int(config.warehouse.n_robots)
        if n_robots < 1:
            raise ValueError("n_robots must be >= 1")
        if n_robots > len(self.start_positions):
            raise ValueError(
                f"Layout '{layout.name}' provides {len(self.start_positions)} robot start "
                f"cells but n_robots={n_robots}"
            )
        self.n_robots = n_robots

        # Populated by reset().
        self.robots: List[Robot] = []
        self.orders: Dict[int, Order] = {}
        self.order_generator: Optional[OrderGenerator] = None
        self.step_count = 0
        self.n_orders_created = 0
        self.total_travel_distance = 0.0
        self.robot_busy_steps = 0
        self.n_failed = 0
        self.completed_order_ids: List[int] = []
        self.timed_out_order_ids: List[int] = []
        self.failed_order_ids: List[int] = []
        self.reset()

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    def reset(self, *, seed: Optional[int] = None) -> "WarehouseSimulation":
        """Reset the world to step 0. Optionally reseed the order stream."""
        if seed is not None:
            self.base_seed = int(seed)
        self.step_count = 0
        self.n_orders_created = 0
        self.total_travel_distance = 0.0
        self.robot_busy_steps = 0
        self.n_failed = 0
        self.completed_order_ids = []
        self.timed_out_order_ids = []
        self.failed_order_ids = []
        self.orders = {}

        starts = sorted(self.start_positions)
        self.robots = [
            Robot(id=i, position=starts[i], start_position=starts[i]) for i in range(self.n_robots)
        ]

        orders_cfg = self.config.orders
        self.order_generator = OrderGenerator(
            pick_locations=self.pick_locations,
            dropoff_locations=self.dropoff_locations,
            arrival_rate=orders_cfg.arrival_rate,
            max_pending=orders_cfg.max_pending,
            max_orders=orders_cfg.max_orders,
            seed=self.base_seed,
        )
        return self

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #
    @property
    def done(self) -> bool:
        return self.step_count >= self.config.warehouse.max_steps

    def pending_orders(self) -> List[Order]:
        return sorted(
            (o for o in self.orders.values() if o.status == PENDING),
            key=lambda o: (o.created_step, o.id),
        )

    def open_orders(self) -> List[Order]:
        return [o for o in self.orders.values() if o.is_open]

    def idle_robots(self) -> List[Robot]:
        return [r for r in self.robots if r.available]

    # ------------------------------------------------------------------ #
    # Assignment
    # ------------------------------------------------------------------ #
    def can_assign(self, robot_id: int, order_id: int) -> bool:
        if not (0 <= robot_id < len(self.robots)):
            return False
        robot = self.robots[robot_id]
        order = self.orders.get(order_id)
        if order is None or order.status != PENDING or not robot.available:
            return False
        return self.oracle.reachable(robot.position, order.pickup)

    def assign(self, robot_id: int, order_id: int) -> bool:
        """Bind ``order_id`` to ``robot_id`` if valid. Returns success."""
        if not self.can_assign(robot_id, order_id):
            return False
        robot = self.robots[robot_id]
        order = self.orders[order_id]
        path = self.oracle.path(robot.position, order.pickup)
        assert path is not None  # guaranteed by can_assign
        order.status = ASSIGNED
        order.assigned_step = self.step_count
        order.robot_id = robot_id
        robot.status = TO_PICKUP
        robot.order_id = order_id
        robot.set_path(path)
        return True

    def _apply_assignments(self, assignments: Sequence[Assignment]) -> List[Assignment]:
        applied: List[Assignment] = []
        for robot_id, order_id in assignments:
            if self.assign(robot_id, order_id):
                applied.append((int(robot_id), int(order_id)))
        return applied

    # ------------------------------------------------------------------ #
    # Stepping
    # ------------------------------------------------------------------ #
    def step(
        self,
        policy: Optional[object] = None,
        assignments: Optional[Sequence[Assignment]] = None,
    ) -> StepResult:
        """Advance one step.

        ``assignments`` takes precedence when provided; otherwise ``policy``
        (any object exposing ``assign(sim) -> Sequence[(robot_id, order_id)]``)
        is queried on decision steps.
        """
        if self.step_count >= self.config.warehouse.max_steps:
            raise RuntimeError("Episode already finished; call reset() first")

        self.step_count += 1
        step = self.step_count
        result = StepResult(step=step)

        # 1. Arrivals.
        gen = self.order_generator
        assert gen is not None
        new_orders = gen.maybe_generate(step, pending_count=len(self.pending_orders()))
        for order in new_orders:
            self.orders[order.id] = order
            self.n_orders_created += 1
            result.arrived_order_ids.append(order.id)

        # 2. Allocation.
        interval = max(1, int(self.config.warehouse.decision_interval))
        chosen: Sequence[Assignment] = []
        if assignments is not None:
            chosen = assignments
        elif policy is not None and step % interval == 0:
            chosen = policy.assign(self)  # type: ignore[attr-defined]
        result.assignments = self._apply_assignments(chosen)

        # 3. Robot dynamics.
        for robot in self.robots:
            self._advance_robot(robot, result)

        # 4. Expire stale unassigned orders.
        self._handle_timeouts(step, result)

        # 5. Bookkeeping.
        for robot in self.robots:
            if robot.is_busy:
                robot.busy_steps += 1
                self.robot_busy_steps += 1
        self.total_travel_distance = float(sum(r.distance_travelled for r in self.robots))

        result.done = self.done
        result.truncated = self.done
        return result

    def run(
        self,
        policy: Optional[object] = None,
        *,
        max_steps: Optional[int] = None,
    ) -> List[StepResult]:
        """Run the episode to completion, returning per-step results."""
        limit = max_steps if max_steps is not None else self.config.warehouse.max_steps
        history: List[StepResult] = []
        while not self.done and self.step_count < limit:
            history.append(self.step(policy=policy))
        return history

    # ------------------------------------------------------------------ #
    # Internal dynamics
    # ------------------------------------------------------------------ #
    def _advance_robot(self, robot: Robot, result: StepResult) -> None:
        cfg = self.config.warehouse
        if robot.status == IDLE:
            return

        if robot.status in (TO_PICKUP, TO_DROPOFF):
            arrived = robot.goal_reached
            for _ in range(max(1, int(cfg.speed))):
                arrived = robot.step_along_path(connectivity=cfg.connectivity)
                if arrived:
                    break
            if not arrived:
                return
            order = self.orders.get(robot.order_id)  # type: ignore[arg-type]
            if order is None:
                self._release_robot(robot)
                return
            if robot.status == TO_PICKUP:
                if cfg.pick_dwell_steps <= 0:
                    self._start_dropoff_leg(robot, order, result)
                else:
                    robot.status = PICKING
                    robot.timer = int(cfg.pick_dwell_steps)
            else:  # TO_DROPOFF
                if cfg.drop_dwell_steps <= 0:
                    self._complete_order(robot, order, result)
                else:
                    robot.status = DROPPING
                    robot.timer = int(cfg.drop_dwell_steps)
            return

        if robot.status == PICKING:
            robot.timer -= 1
            if robot.timer <= 0:
                order = self.orders.get(robot.order_id)  # type: ignore[arg-type]
                if order is None:
                    self._release_robot(robot)
                    return
                order.status = PICKED
                order.picked_step = self.step_count
                self._start_dropoff_leg(robot, order, result)
            return

        if robot.status == DROPPING:
            robot.timer -= 1
            if robot.timer <= 0:
                order = self.orders.get(robot.order_id)  # type: ignore[arg-type]
                if order is None:
                    self._release_robot(robot)
                    return
                self._complete_order(robot, order, result)
            return

    def _start_dropoff_leg(self, robot: Robot, order: Order, result: StepResult) -> None:
        path = self.oracle.path(robot.position, order.dropoff)
        if path is None:
            # Connected layouts should never hit this; treat as a failure.
            order.status = FAILED
            self.n_failed += 1
            self.failed_order_ids.append(order.id)
            result.failed_order_ids.append(order.id)
            self._release_robot(robot)
            return
        robot.status = TO_DROPOFF
        robot.set_path(path)

    def _complete_order(self, robot: Robot, order: Order, result: StepResult) -> None:
        order.status = COMPLETED
        order.completed_step = self.step_count
        robot.orders_completed += 1
        self.completed_order_ids.append(order.id)
        result.completed_order_ids.append(order.id)
        self._release_robot(robot)

    def _release_robot(self, robot: Robot) -> None:
        robot.status = IDLE
        robot.order_id = None
        robot.path = []
        robot.path_index = 0
        robot.timer = 0

    def _handle_timeouts(self, step: int, result: StepResult) -> None:
        timeout = int(self.config.orders.timeout_steps)
        if timeout <= 0:
            return
        for order in list(self.orders.values()):
            if order.status == PENDING and (step - order.created_step) >= timeout:
                order.status = TIMEOUT
                self.timed_out_order_ids.append(order.id)
                result.timed_out_order_ids.append(order.id)

    # ------------------------------------------------------------------ #
    # Rendering helpers
    # ------------------------------------------------------------------ #
    def ascii_render(self) -> str:
        """Return an ASCII snapshot with live robot positions overlaid."""
        grid = [list(row) for row in self.layout.rows]
        # Idle/spawn cells lose their label once a robot stands on them.
        overlay: Dict[Cell, str] = {}
        for robot in self.robots:
            overlay[robot.position] = str(robot.id % 10)
        for (r, c), ch in overlay.items():
            grid[r][c] = ch
        return "\n".join("".join(row) for row in grid)
