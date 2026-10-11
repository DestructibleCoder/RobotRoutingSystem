"""Order representation and the dynamic order stream.

An order is a pickup-and-delivery job: a robot must travel to ``pickup``,
service it, travel to ``dropoff``, and service it again. Orders arrive
dynamically according to a Bernoulli stream (configurable via
``OrderConfig.arrival_rate``).

Each order records timestamps so that the metrics module can derive waiting
time and completion time without extra bookkeeping.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

Cell = Tuple[int, int]

# Order lifecycle statuses.
PENDING = "pending"
ASSIGNED = "assigned"
PICKED = "picked"
COMPLETED = "completed"
TIMEOUT = "timeout"


@dataclass
class Order:
    id: int
    pickup: Cell
    dropoff: Cell
    created_step: int
    status: str = PENDING

    assigned_step: Optional[int] = None
    picked_step: Optional[int] = None
    completed_step: Optional[int] = None
    robot_id: Optional[int] = None

    # -- derived timings (None until the corresponding event happens) ------ #
    @property
    def waiting_time(self) -> Optional[int]:
        """Steps between creation and assignment."""
        if self.assigned_step is None:
            return None
        return self.assigned_step - self.created_step

    @property
    def completion_time(self) -> Optional[int]:
        """Steps between creation and completion (lead time)."""
        if self.completed_step is None:
            return None
        return self.completed_step - self.created_step

    @property
    def service_time(self) -> Optional[int]:
        """Steps between assignment and completion."""
        if self.completed_step is None or self.assigned_step is None:
            return None
        return self.completed_step - self.assigned_step

    @property
    def is_open(self) -> bool:
        return self.status in (PENDING, ASSIGNED, PICKED)


class OrderGenerator:
    """Stochastic pick/drop order stream.

    Parameters
    ----------
    pick_locations, dropoff_locations:
        Candidate cells for the two legs of each order.
    arrival_rate:
        Probability of spawning an order on any given step.
    max_pending:
        Upper bound on unassigned orders; new arrivals are dropped when full.
    max_orders:
        Optional cap on total generated orders (0 = unlimited).
    """

    def __init__(
        self,
        *,
        pick_locations: Sequence[Cell],
        dropoff_locations: Sequence[Cell],
        arrival_rate: float,
        max_pending: int = 25,
        max_orders: int = 0,
        seed: int = 0,
    ) -> None:
        if not pick_locations:
            raise ValueError("OrderGenerator needs at least one pick location")
        if not dropoff_locations:
            raise ValueError("OrderGenerator needs at least one drop-off location")
        if not 0.0 <= arrival_rate <= 1.0:
            raise ValueError(f"arrival_rate must be in [0, 1], got {arrival_rate}")
        self.pick_locations = list(pick_locations)
        self.dropoff_locations = list(dropoff_locations)
        self.arrival_rate = float(arrival_rate)
        self.max_pending = int(max_pending)
        self.max_orders = int(max_orders)
        self.rng = np.random.default_rng(seed)
        self.n_generated = 0

    def reset(self, seed: Optional[int] = None) -> None:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.n_generated = 0

    def maybe_generate(self, step: int, pending_count: int) -> List[Order]:
        """Return zero or more new orders for ``step``."""
        if self.max_orders and self.n_generated >= self.max_orders:
            return []
        if self.max_pending and pending_count >= self.max_pending:
            return []
        if self.rng.random() >= self.arrival_rate:
            return []
        return [self._make_order(step)]

    def _make_order(self, step: int) -> Order:
        pick_idx = int(self.rng.integers(len(self.pick_locations)))
        drop_idx = int(self.rng.integers(len(self.dropoff_locations)))
        order_id = self.n_generated
        self.n_generated += 1
        return Order(
            id=order_id,
            pickup=self.pick_locations[pick_idx],
            dropoff=self.dropoff_locations[drop_idx],
            created_step=step,
        )
