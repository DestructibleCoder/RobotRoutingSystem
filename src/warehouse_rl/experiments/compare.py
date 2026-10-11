"""Algorithm comparison runner (Experiment C scaffolding).

Runs several allocation policies over the *same* test layouts and seeds so the
comparison is fair, writes raw per-episode results, and produces a grouped
summary. Only the greedy baseline is implemented today; PPO/DQN slot in via
:func:`warehouse_rl.algorithms.make_policy` once available.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

from warehouse_rl.algorithms.greedy import make_policy
from warehouse_rl.config import EvaluationConfig, load_config
from warehouse_rl.experiments.evaluate import evaluate_policy, load_layouts
from warehouse_rl.metrics.metrics import EpisodeMetrics, records_to_rows, save_records_csv


def compare_algorithms(
    config: EvaluationConfig,
    algorithms: Optional[Sequence[str]] = None,
    *,
    output_dir: Optional[str | Path] = None,
    n_episodes: Optional[int] = None,
    seeds: Optional[Sequence[int]] = None,
) -> List[EpisodeMetrics]:
    """Evaluate each algorithm on the same layouts/seeds and write results.

    Returns the combined list of per-episode records.
    """
    algorithms = list(algorithms) if algorithms else list(config.algorithms)
    layouts = load_layouts(config.layout.test_dir)
    out_dir = Path(output_dir) if output_dir else Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_records: List[EpisodeMetrics] = []
    for algo in algorithms:
        policy = make_policy(algo, config.greedy)
        records = evaluate_policy(
            policy,
            layouts,
            config,
            n_episodes=n_episodes,
            seeds=seeds,
            policy_name=algo,
        )
        all_records.extend(records)

    save_records_csv(all_records, out_dir / "episodes.csv")
    summary = summarize_records(all_records)
    if summary is not None:
        summary.to_csv(out_dir / "summary.csv")
    return all_records


def summarize_records(records: Sequence[EpisodeMetrics]):
    """Group metric means (and std) by policy. Requires pandas; else returns None."""
    try:
        import pandas as pd

        df = pd.DataFrame(records_to_rows(records))
        numeric = df.select_dtypes("number").drop(columns=["seed"], errors="ignore")
        summary = df.groupby("policy")[numeric.columns].agg(["mean", "std"]).round(4)
        summary.columns = ["_".join(c).strip("_") for c in summary.columns]
        return summary
    except ImportError:  # pragma: no cover - pandas is an optional extra
        return None


def compare_from_config(
    config_path: str | Path = "configs/evaluation.yaml",
) -> List[EpisodeMetrics]:
    config = load_config(config_path, config_cls=EvaluationConfig)
    return compare_algorithms(config)
