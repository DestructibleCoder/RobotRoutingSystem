"""Training runner.

Only the greedy baseline exists at this milestone, and it does not train. The
function below establishes the interface and the on-disk artifact layout
(config snapshot + manifest) that the PPO/DQN runners will reuse in
Milestones 4-6. It intentionally raises for unimplemented algorithms rather
than pretending to train.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional

from warehouse_rl.config import TrainingConfig, load_config, save_config
from warehouse_rl.environment.layout_generator import Layout, load_suite
from warehouse_rl.algorithms.greedy import GreedyPolicy


@dataclass
class TrainingResult:
    """Artifacts produced by a (possibly no-op) training run."""

    algorithm: str
    policy: Any
    training_layouts: List[str] = field(default_factory=list)
    run_dir: Optional[Path] = None
    message: str = ""


def load_training_layouts(config: TrainingConfig) -> List[Layout]:
    """Return up to ``n_training_layouts`` layouts from the training split."""
    layouts = load_suite(config.layout.train_dir)
    if not layouts:
        raise FileNotFoundError(
            f"No training layouts in '{config.layout.train_dir}'. Run "
            f"`python scripts/generate_layouts.py` first."
        )
    n = config.n_training_layouts
    return layouts[:n] if n and n < len(layouts) else layouts


def train(
    config: TrainingConfig,
    *,
    run_dir: Optional[str | Path] = None,
) -> TrainingResult:
    """Train (or instantiate) a policy according to ``config.algorithm``."""
    layouts = load_training_layouts(config)
    run_dir = Path(run_dir) if run_dir else Path(config.output_dir) / config.algorithm
    run_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, run_dir / "config.yaml")

    if config.algorithm == "greedy":
        policy = GreedyPolicy(config.greedy)
        (run_dir / "MANIFEST.txt").write_text(
            "algorithm: greedy (non-learning baseline)\n"
            f"training_layouts: {[l.name for l in layouts]}\n"
            "note: greedy has no trainable parameters; no checkpoint written.\n",
            encoding="utf-8",
        )
        return TrainingResult(
            algorithm="greedy",
            policy=policy,
            training_layouts=[l.name for l in layouts],
            run_dir=run_dir,
            message="Greedy is a non-learning baseline; no training performed.",
        )

    if config.algorithm in ("ppo", "dqn"):
        raise NotImplementedError(
            f"Training for '{config.algorithm}' is planned for Milestones 4-6 and is not "
            f"implemented yet. The run directory and config snapshot were created at {run_dir}."
        )

    raise ValueError(f"Unknown algorithm '{config.algorithm}'")


def train_from_config(config_path: str | Path = "configs/training.yaml") -> TrainingResult:
    config = load_config(config_path, config_cls=TrainingConfig)
    return train(config)
