"""Task-allocation algorithms.

Currently only the classical greedy baseline is implemented. PPO and DQN are
planned (Milestones 4-6) and will be added in ``ppo.py`` / ``dqn.py``.
"""

from warehouse_rl.algorithms.greedy import GreedyPolicy, make_policy

__all__ = ["GreedyPolicy", "make_policy"]
