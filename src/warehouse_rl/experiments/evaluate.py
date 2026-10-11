"""Evaluation runner: execute a policy on a set of layouts and collect metrics.

This module is deliberately policy-agnostic. Any object with an
``assign(sim) -> [(robot_id, order_id)]`` method (and optionally a ``name``
attribute) can be evaluated, which lets the greedy baseline and future RL
policies flow through identical code paths.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Sequence

from warehouse_rl.config import EvaluationConfig, load_config
from warehouse_rl.environment.layout_generator import Layout, load_suite
from warehouse_rl.environment.warehouse import WarehouseSimulation
from warehouse_rl.metrics.metrics import EpisodeMetrics, compute_episode_metrics

SEED_STRIDE = 1_000_003


def load_layouts(directory: str | Path) -> List[Layout]:
    """Load a layout suite directory, raising if empty."""
    layouts = load_suite(directory)
    if not layouts:
        raise FileNotFoundError(
            f"No layouts found in '{directory}'. Generate them first, e.g. "
            f"`python scripts/generate_layouts.py`."
        )
    return layouts


def run_episode(
    layout: Layout,
    config: EvaluationConfig,
    policy: object,
    *,
    seed: int = 0,
    policy_name: Optional[str] = None,
) -> EpisodeMetrics:
    """Run a single episode and return its metrics."""
    sim = WarehouseSimulation(layout, config, seed=seed)
    sim.run(policy=policy)
    return compute_episode_metrics(
        sim,
        policy=policy_name or getattr(policy, "name", "policy"),
        seed=seed,
        layout=layout.name,
    )


def evaluate_policy(
    policy: object,
    layouts: Sequence[Layout],
    config: EvaluationConfig,
    *,
    n_episodes: Optional[int] = None,
    seeds: Optional[Sequence[int]] = None,
    policy_name: Optional[str] = None,
) -> List[EpisodeMetrics]:
    """Evaluate a policy across layouts, seeds, and repeats.

    Episodes use deterministic, distinct seeds derived from the base seeds so
    results are reproducible and independent.
    """
    n_episodes = n_episodes or config.n_episodes
    seeds = list(seeds) if seeds is not None else list(config.seeds)
    records: List[EpisodeMetrics] = []
    for layout in layouts:
        for base_seed in seeds:
            for episode in range(n_episodes):
                ep_seed = int(base_seed) * SEED_STRIDE + episode
                records.append(
                    run_episode(
                        layout,
                        config,
                        policy,
                        seed=ep_seed,
                        policy_name=policy_name,
                    )
                )
    return records


def evaluate_from_config(
    policy: object,
    *,
    config_path: str | Path = "configs/evaluation.yaml",
    config: Optional[EvaluationConfig] = None,
    layout_dir: Optional[str | Path] = None,
    policy_name: Optional[str] = None,
    save_csv: Optional[bool] = None,
    output_path: Optional[str | Path] = None,
) -> List[EpisodeMetrics]:
    """Convenience wrapper used by the CLI scripts."""
    if config is None:
        config = load_config(config_path, config_cls=EvaluationConfig)
    directory = layout_dir or config.layout.test_dir
    layouts = load_layouts(directory)
    records = evaluate_policy(policy, layouts, config, policy_name=policy_name)

    should_save = config.save_csv if save_csv is None else save_csv
    if should_save:
        from warehouse_rl.metrics.metrics import save_records_csv

        name = policy_name or getattr(policy, "name", "policy")
        out = Path(output_path) if output_path else Path(config.output_dir) / f"{name}_episodes.csv"
        save_records_csv(records, out)
    return records
