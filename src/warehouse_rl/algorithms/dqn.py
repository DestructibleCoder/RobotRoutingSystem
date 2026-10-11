"""DQN task-allocation policy (Milestone 6, optional).

NOT IMPLEMENTED YET.

DQN will only be added if the sequential-allocation MDP has a compact action
space that makes a value-based approach practical. The module will provide a
:class:`DQNPolicy` mirroring the :class:`PPOPolicy` interface
(``assign`` model agnostic, SB3 checkpoint save/load).
"""

from __future__ import annotations

from typing import List

from warehouse_rl.environment.warehouse import Assignment, WarehouseSimulation


class DQNPolicy:
    """Placeholder for the DQN-based allocation policy."""

    name = "dqn"

    def __init__(self, *args, **kwargs) -> None:  # pragma: no cover
        raise NotImplementedError(
            "DQN is optional (Milestone 6) and is not implemented yet. "
            "It will be added only if the action/observation design makes it practical."
        )

    def assign(self, sim: WarehouseSimulation) -> List[Assignment]:
        raise NotImplementedError

    def save(self, path: str) -> None:  # pragma: no cover
        raise NotImplementedError

    @classmethod
    def load(cls, path: str, **kwargs) -> "DQNPolicy":  # pragma: no cover
        raise NotImplementedError