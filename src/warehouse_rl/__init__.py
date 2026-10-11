"""warehouse_rl: RL task-allocation research framework for warehouse layouts.

Package layout
--------------
environment   Grid warehouse simulation (layout, robots, orders, dynamics).
navigation    Classical low-level path planning (A*).
algorithms    Task-allocation policies (greedy baseline, PPO/DQN later).
experiments   Train / evaluate / compare runners and CSV plumbing.
metrics       Episodic metric collection and aggregation.
config        Typed, YAML-backed configuration system.
"""

from warehouse_rl.config import (
    ExperimentConfig,
    EvaluationConfig,
    TrainingConfig,
    load_config,
)
from warehouse_rl.environment.layout_generator import Layout, generate_layout
from warehouse_rl.environment.warehouse import WarehouseSimulation

__all__ = [
    "ExperimentConfig",
    "EvaluationConfig",
    "TrainingConfig",
    "load_config",
    "Layout",
    "generate_layout",
    "WarehouseSimulation",
]

__version__ = "0.1.0"
