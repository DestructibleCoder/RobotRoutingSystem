#!/usr/bin/env python3
"""Train a task-allocation policy.

Only the greedy baseline is implemented so far (a no-op "training" that
instantiates the policy and snapshots the config). PPO/DQN will be added in
Milestones 4-6.

Usage
-----
    python scripts/train.py --config configs/training.yaml
    python scripts/train.py --algorithm greedy --n-training-layouts 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from warehouse_rl.config import TrainingConfig, load_config  # noqa: E402
from warehouse_rl.experiments.train import train  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/training.yaml")
    parser.add_argument("--algorithm", default=None, choices=["greedy", "ppo", "dqn"])
    parser.add_argument("--n-training-layouts", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    overrides = {}
    if args.seed is not None:
        overrides["seed"] = args.seed
    config: TrainingConfig = load_config(args.config, config_cls=TrainingConfig, overrides=overrides)
    if args.algorithm:
        config.algorithm = args.algorithm
    if args.n_training_layouts is not None:
        config.n_training_layouts = args.n_training_layouts

    print(
        f"Training algorithm='{config.algorithm}' on "
        f"{config.n_training_layouts} layout(s), seed={config.seed}"
    )
    try:
        result = train(config)
    except NotImplementedError as exc:
        print(f"[not implemented] {exc}")
        return 2
    print(f"Run directory: {result.run_dir}")
    print(f"Training layouts: {result.training_layouts}")
    print(result.message)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
