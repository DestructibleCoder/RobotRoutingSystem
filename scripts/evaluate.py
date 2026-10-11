#!/usr/bin/env python3
"""Evaluate a task-allocation policy on the test layouts.

Usage
-----
    python scripts/evaluate.py                       # greedy, configs/evaluation.yaml
    python scripts/evaluate.py --algorithm greedy --episodes 3
    python scripts/evaluate.py --algorithm ppo --model results/training/ppo/ppo.zip

Per-episode results are written to CSV under ``results/evaluation``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from warehouse_rl.algorithms.greedy import make_policy  # noqa: E402
from warehouse_rl.config import EvaluationConfig, load_config  # noqa: E402
from warehouse_rl.experiments.compare import summarize_records  # noqa: E402
from warehouse_rl.experiments.evaluate import evaluate_from_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/evaluation.yaml")
    parser.add_argument("--algorithm", default="greedy", choices=["greedy", "ppo", "dqn"])
    parser.add_argument("--layout-dir", default=None, help="override test layout directory")
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--model", default=None, help="path to a saved RL checkpoint (future)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config: EvaluationConfig = load_config(args.config, config_cls=EvaluationConfig)
    if args.episodes is not None:
        config.n_episodes = args.episodes
    if args.seeds is not None:
        config.seeds = list(args.seeds)

    try:
        policy = make_policy(args.algorithm, config.greedy)
    except NotImplementedError as exc:
        print(f"[not implemented] {exc}")
        return 2

    records = evaluate_from_config(
        policy,
        config=config,
        layout_dir=args.layout_dir,
        policy_name=args.algorithm,
    )
    n = len(records)
    completed = sum(r.n_orders_completed for r in records)
    print(f"Evaluated '{args.algorithm}' on {n} episodes; {completed} orders completed.")
    summary = summarize_records(records)
    if summary is not None:
        print(summary.to_string())
    else:
        print("(install the 'analysis' extra for aggregated summaries: pip install pandas)")
    print(f"Raw results written under: {config.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
