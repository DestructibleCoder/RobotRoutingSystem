#!/usr/bin/env python3
"""End-to-end smoke demo: run one greedy episode and print the trajectory.

Usage
-----
    python scripts/demo.py
    python scripts/demo.py --layout layouts/train/train_00.txt --every 25
    python scripts/demo.py --snapshot results/demo.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from warehouse_rl.algorithms.greedy import GreedyPolicy  # noqa: E402
from warehouse_rl.config import ExperimentConfig, load_config  # noqa: E402
from warehouse_rl.environment.layout_generator import Layout, load_suite  # noqa: E402
from warehouse_rl.environment.warehouse import WarehouseSimulation  # noqa: E402
from warehouse_rl.metrics.metrics import compute_episode_metrics  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--layout", default=None, help="path to a layout file")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--every", type=int, default=50, help="print a frame every N steps")
    parser.add_argument("--snapshot", default=None, help="save a PNG snapshot here")
    parser.add_argument("--animate", default=None, help="save a GIF animation here")
    return parser.parse_args()


def resolve_layout(args) -> Layout:
    if args.layout:
        return Layout.read(args.layout)
    suite = load_suite("layouts/train")
    if not suite:
        raise SystemExit("No layouts found. Run `python scripts/generate_layouts.py` first.")
    return suite[0]


def main() -> int:
    args = parse_args()
    config: ExperimentConfig = load_config(args.config, config_cls=ExperimentConfig)
    layout = resolve_layout(args)
    policy = GreedyPolicy(config.greedy)

    print(f"Layout: {layout.name} ({layout.width}x{layout.height})")
    print(layout.to_ascii())
    print()

    sim = WarehouseSimulation(layout, config, seed=args.seed)
    print(f"step 0:")
    print(sim.ascii_render())
    while not sim.done:
        sim.step(policy=policy)
        if args.every and sim.step_count % args.every == 0:
            print(f"\nstep {sim.step_count}:")
            print(sim.ascii_render())

    metrics = compute_episode_metrics(sim, policy="greedy", seed=args.seed)
    print("\n=== Episode metrics ===")
    for key, value in metrics.to_row().items():
        print(f"  {key}: {value}")

    if args.snapshot:
        from warehouse_rl.environment.rendering import save_snapshot

        print(f"Saved snapshot to {save_snapshot(sim, args.snapshot)}")
    if args.animate:
        from warehouse_rl.environment.rendering import animate_episode

        path = animate_episode(layout, config, policy, args.animate, seed=args.seed)
        print(f"Saved animation to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
