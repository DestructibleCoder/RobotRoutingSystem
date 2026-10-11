"""Episodic metrics and aggregation.

Definitions (all timings in simulation steps, all distances in grid units):

* **completion time**  ``completed_step - created_step`` (order lead time).
* **waiting time**     ``assigned_step - created_step`` (queueing before pickup).
* **service time**     ``completed_step - assigned_step``.
* **travel distance**  summed per-robot movement across the episode.
* **throughput**       completed orders per 100 steps.
* **robot utilization** fraction of robot-steps spent busy (moving/servicing).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Optional, Sequence

from warehouse_rl.environment.orders import COMPLETED, TIMEOUT
from warehouse_rl.environment.warehouse import FAILED, WarehouseSimulation


@dataclass
class EpisodeMetrics:
    """Metrics for a single evaluation episode."""

    layout: str = ""
    policy: str = ""
    seed: int = 0
    episode_duration: int = 0
    n_orders_created: int = 0
    n_orders_completed: int = 0
    n_orders_timed_out: int = 0
    n_orders_failed: int = 0
    avg_completion_time: float = 0.0
    avg_waiting_time: float = 0.0
    avg_service_time: float = 0.0
    total_travel_distance: float = 0.0
    throughput_per_100_steps: float = 0.0
    robot_utilization: float = 0.0

    def to_row(self) -> Dict[str, object]:
        return asdict(self)


def compute_episode_metrics(
    sim: WarehouseSimulation,
    *,
    policy: str = "",
    seed: int = 0,
    layout: Optional[str] = None,
) -> EpisodeMetrics:
    """Derive :class:`EpisodeMetrics` from a finished (or truncated) simulation."""
    completed = [o for o in sim.orders.values() if o.status == COMPLETED]
    timed_out = [o for o in sim.orders.values() if o.status == TIMEOUT]
    failed = [o for o in sim.orders.values() if o.status == FAILED]

    completion_times = [o.completion_time for o in completed if o.completion_time is not None]
    waiting_times = [o.waiting_time for o in completed if o.waiting_time is not None]
    service_times = [o.service_time for o in completed if o.service_time is not None]

    duration = int(sim.step_count)
    n_completed = len(completed)
    robot_steps = max(1, sim.n_robots * duration)

    return EpisodeMetrics(
        layout=layout if layout is not None else sim.layout.name,
        policy=policy,
        seed=seed,
        episode_duration=duration,
        n_orders_created=len(sim.orders),
        n_orders_completed=n_completed,
        n_orders_timed_out=len(timed_out),
        n_orders_failed=len(failed),
        avg_completion_time=_safe_mean(completion_times),
        avg_waiting_time=_safe_mean(waiting_times),
        avg_service_time=_safe_mean(service_times),
        total_travel_distance=float(sim.total_travel_distance),
        throughput_per_100_steps=(100.0 * n_completed / duration) if duration else 0.0,
        robot_utilization=float(sim.robot_busy_steps) / robot_steps,
    )


def _safe_mean(values: Sequence[float]) -> float:
    return float(mean(values)) if values else 0.0


def records_to_rows(records: Iterable[EpisodeMetrics]) -> List[Dict[str, object]]:
    return [r.to_row() for r in records]


def save_records_csv(records: Iterable[EpisodeMetrics], path: str | Path) -> Path:
    """Write per-episode records to CSV (uses pandas when available)."""
    rows = records_to_rows(records)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import pandas as pd

        pd.DataFrame(rows).to_csv(path, index=False)
    except ImportError:  # pragma: no cover - pandas is an optional extra
        import csv

        if not rows:
            path.write_text("", encoding="utf-8")
            return path
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    return path


def aggregate(records: Sequence[EpisodeMetrics]) -> Dict[str, Dict[str, float]]:
    """Mean/std of numeric metrics grouped by policy (and layout if present)."""
    try:
        import pandas as pd

        df = pd.DataFrame(records_to_rows(records))
        numeric = df.select_dtypes("number").drop(columns=["seed"], errors="ignore")
        grouped = df.groupby(["policy", "layout"])[numeric.columns]
        out = grouped.agg(["mean", "std"]).round(4)
        return out.to_dict()  # type: ignore[return-value]
    except ImportError:  # pragma: no cover
        grouped: Dict[str, Dict[str, float]] = {}
        for rec in records:
            grouped.setdefault(rec.policy, {})
            grouped[rec.policy]["n_records"] = grouped[rec.policy].get("n_records", 0) + 1
        return grouped
