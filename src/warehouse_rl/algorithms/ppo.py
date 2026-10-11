"""PPO task-allocation policy (Milestone 4+).

NOT IMPLEMENTED YET. This module defines the *interface contract* that the
PPO-based allocation policy will fulfil:

* observed state: normalized robot + order features per decision step,
* action: allocate a pending order to a robot (or wait),
* reward: from :class:`warehouse_rl.config.RewardConfig`,
* checkpoint format: Stable-Baselines3 ``.zip`` under
  ``results/training/ppo/``.

Training and evaluation will reuse the gymnasium wrapper (Milestone 3) and the
policy-agnostic runners in ``warehouse_rl.experiments``.
"""

from __future__ import annotations

from typing import List, Sequence

from warehouse_rl.environment.warehouse import Assignment, WarehouseSimulation


class PPOPolicy:
    """Placeholder for the PPO-based allocation policy."""

    name = "ppo"

    def __init__(self, *args, **kwargs) -> None:  # pragma: no cover
        raise NotImplementedError(
            "PPO is planned for Milestone 4 and is not implemented yet. "
            "Milestones 1-3 (simulator, greedy baseline, gymnasium env) must land first."
        )

    def assign(self, sim: WarehouseSimulation) -> List[Assignment]:
        raise NotImplementedError

    def save(self, path: str) -> None:  # pragma: no cover
        raise NotImplementedError

    @classmethod
    def load(cls, path: str, **kwargs) -> "PPOPolicy":  # pragma: no cover
        raise NotImplementedError