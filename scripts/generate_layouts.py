#!/usr/bin/env python3
"""Generate the reproducible train/test layout suites.

Usage
-----
    python scripts/generate_layouts.py                 # use configs/default.yaml
    python scripts/generate_layouts.py --n-train 1 --n-test 10
    python scripts/generate_layouts.py --force         # regenerate existing dirs

Train and test suites always come from disjoint seed streams, so test layouts
are guaranteed to be unseen during training.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from warehouse_rl.config import ExperimentConfig, load_config  # noqa: E402
from warehouse_rl.environment.layout_generator import (  # noqa: E402
    generate_train_test_suites,
    load_suite,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--n-train", type=int, default=None)
    parser.add_argument("--n-test", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None, help="layout generation seed")
    parser.add_argument("--force", action="store_true", help="regenerate non-empty dirs")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config: ExperimentConfig = load_config(args.config, config_cls=ExperimentConfig)
    lcfg = config.layout
    if args.n_train is not None:
        lcfg.n_train = args.n_train
    if args.n_test is not None:
        lcfg.n_test = args.n_test
    if args.seed is not None:
        lcfg.generation_seed = args.seed

    train_dir, test_dir = Path(lcfg.train_dir), Path(lcfg.test_dir)
    existing = load_suite(train_dir) + load_suite(test_dir)
    if existing and not args.force:
        print(
            f"Layouts already exist ({len(existing)} files). Use --force to regenerate. "
            "Nothing to do."
        )
        return 0

    train, test = generate_train_test_suites(
        n_train=lcfg.n_train,
        n_test=lcfg.n_test,
        train_dir=train_dir,
        test_dir=test_dir,
        generation_seed=lcfg.generation_seed,
        min_width=lcfg.min_width,
        min_height=lcfg.min_height,
        max_width=lcfg.max_width,
        max_height=lcfg.max_height,
    )
    print(f"Generated {len(train)} training layouts in {train_dir}")
    for layout in train:
        print(f"  - {layout.summary()}")
    print(f"Generated {len(test)} test layouts in {test_dir}")
    for layout in test:
        print(f"  - {layout.summary()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
