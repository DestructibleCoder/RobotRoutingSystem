"""Typed configuration system.

Configurations are plain nested dicts in YAML (see ``configs/``). This module
parses them into dataclasses so the rest of the codebase gets:

* defaults for every field,
* IDE/attribute access instead of ``cfg["a"]["b"]``,
* tolerant loading (unknown keys are ignored but reported),
* dotted-key overrides from the CLI (``--set warehouse.n_robots=6``).

Keeping a single source of truth for all configurable knobs is essential for the
reproducibility requirements of the project (seeds, robot counts, layout sets).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Type, TypeVar, get_type_hints

import yaml

T = TypeVar("T")

# --------------------------------------------------------------------------- #
# Config dataclasses
# --------------------------------------------------------------------------- #


@dataclass
class WarehouseConfig:
    """Static + dynamic properties of the simulated warehouse."""

    # Grid used only as the default size when *generating* layouts. Layouts
    # loaded from disk may have different dimensions; the simulation always
    # follows the layout's own size.
    width: int = 20
    height: int = 14
    n_robots: int = 4

    # Dwell (service) times, in discrete simulation steps.
    pick_dwell_steps: int = 2
    drop_dwell_steps: int = 2

    # Episode horizon and how many steps a robot moves per step (speed).
    max_steps: int = 500
    speed: int = 1

    # 4- or 8-connected grid navigation.
    connectivity: int = 4

    # How often the allocation policy is queried (in steps).
    decision_interval: int = 1


@dataclass
class OrderConfig:
    """Dynamic order stream parameters."""

    # Probability that a new order arrives each step (Bernoulli stream).
    arrival_rate: float = 0.35

    # Maximum number of *unassigned* orders allowed to accumulate. 0 = unlimited.
    max_pending: int = 25

    # Optional cap on total orders spawned per episode. 0 = unlimited.
    max_orders: int = 0

    # Steps after which an unassigned order is dropped. 0 disables timeouts.
    timeout_steps: int = 0


@dataclass
class LayoutConfig:
    """Where layouts live and how many to generate per split."""

    train_dir: str = "layouts/train"
    test_dir: str = "layouts/test"
    n_train: int = 10
    n_test: int = 10
    # Seed for the *generator*; distinct from the simulation seed.
    generation_seed: int = 20240101
    # Grid size range used by the generator: (min_w, min_h) .. (max_w, max_h).
    min_width: int = 14
    min_height: int = 10
    max_width: int = 26
    max_height: int = 18


@dataclass
class GreedyConfig:
    """Greedy baseline behaviour."""

    # "min_completion_time": for each order, pick the idle robot with the lowest
    # estimated completion time. Alternative: "nearest_robot".
    strategy: str = "min_completion_time"


@dataclass
class RewardConfig:
    """Placeholder reward shaping for the RL environment (Milestone 3)."""

    completion_reward: float = 1.0
    travel_weight: float = 0.0
    waiting_weight: float = 0.0
    timeout_penalty: float = 0.0


@dataclass
class ExperimentConfig:
    """Base config shared by training and evaluation."""

    seed: int = 0
    warehouse: WarehouseConfig = field(default_factory=WarehouseConfig)
    orders: OrderConfig = field(default_factory=OrderConfig)
    layout: LayoutConfig = field(default_factory=LayoutConfig)
    greedy: GreedyConfig = field(default_factory=GreedyConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)


@dataclass
class TrainingConfig(ExperimentConfig):
    """Additional knobs for training runs."""

    algorithm: str = "greedy"  # "greedy" | "ppo" | "dqn"
    n_training_layouts: int = 1
    seeds: List[int] = field(default_factory=lambda: [0])
    total_timesteps: int = 100_000
    output_dir: str = "results/training"


@dataclass
class EvaluationConfig(ExperimentConfig):
    """Additional knobs for evaluation runs."""

    algorithms: List[str] = field(default_factory=lambda: ["greedy"])
    n_episodes: int = 5
    seeds: List[int] = field(default_factory=lambda: [0, 1, 2])
    output_dir: str = "results/evaluation"
    save_csv: bool = True


# --------------------------------------------------------------------------- #
# Parsing helpers
# --------------------------------------------------------------------------- #

_SECTION_TYPES: Dict[str, type] = {
    "warehouse": WarehouseConfig,
    "orders": OrderConfig,
    "layout": LayoutConfig,
    "greedy": GreedyConfig,
    "reward": RewardConfig,
}


def deep_merge(base: Dict[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``override`` into a copy of ``base``."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, Mapping):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _coerce_field(section: str, name: str, value: Any, target_type: Any) -> Any:
    """Best-effort coercion of a YAML value to the annotated field type."""
    if value is None:
        return None
    # List[int] fields (seeds).
    origin = getattr(target_type, "__origin__", None)
    if origin is list:
        args = getattr(target_type, "__args__", ())
        elem = args[0] if args else Any
        if not isinstance(value, (list, tuple)):
            raise TypeError(f"{section}.{name} must be a list, got {type(value).__name__}")
        return [elem(v) if elem in (int, float, str) else v for v in value]
    if target_type is bool:
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes", "on")
        return bool(value)
    if target_type in (int, float, str):
        return target_type(value)
    return value


def _build_section(section: str, cls: Type[T], data: Any) -> T:
    if data is None:
        return cls()
    if not isinstance(data, Mapping):
        raise TypeError(f"Config section '{section}' must be a mapping, got {type(data).__name__}")
    hints = get_type_hints(cls)
    valid = {f.name for f in fields(cls)}
    kwargs: Dict[str, Any] = {}
    unknown: List[str] = []
    for key, value in data.items():
        if key not in valid:
            unknown.append(key)
            continue
        kwargs[key] = _coerce_field(section, key, value, hints.get(key, Any))
    if unknown:
        # Tolerate but make typos visible.
        print(f"[config] warning: unknown keys in section '{section}': {unknown}")
    return cls(**kwargs)


def _build_config(cls: Type[T], data: Mapping[str, Any]) -> T:
    """Instantiate one of the top-level config dataclasses from a dict."""
    hints = get_type_hints(cls)
    valid = {f.name for f in fields(cls)}
    kwargs: Dict[str, Any] = {}
    unknown: List[str] = []
    for key, value in data.items():
        if key not in valid:
            unknown.append(key)
            continue
        if key in _SECTION_TYPES:
            kwargs[key] = _build_section(key, _SECTION_TYPES[key], value)
        else:
            kwargs[key] = _coerce_field("", key, value, hints.get(key, Any))
    if unknown:
        print(f"[config] warning: unknown top-level keys: {unknown}")
    return cls(**kwargs)


def apply_overrides(data: Dict[str, Any], overrides: Mapping[str, Any]) -> Dict[str, Any]:
    """Apply dotted-key overrides, e.g. ``{"warehouse.n_robots": 6}``."""
    result = copy.deepcopy(data)
    for dotted, value in overrides.items():
        parts = dotted.split(".")
        node = result
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return result


def _load_yaml_with_extends(path: Path, _seen: Optional[set] = None) -> Dict[str, Any]:
    """Load a YAML file, resolving an optional ``extends: other.yaml`` parent."""
    _seen = _seen or set()
    path = Path(path).resolve()
    if path in _seen:
        raise ValueError(f"Circular config 'extends' chain involving {path}")
    _seen.add(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        loaded = yaml.safe_load(fh) or {}
    if not isinstance(loaded, Mapping):
        raise TypeError(f"Config root must be a mapping, got {type(loaded).__name__}")
    loaded = dict(loaded)
    parent = loaded.pop("extends", None)
    if parent:
        base = _load_yaml_with_extends(path.parent / str(parent), _seen)
        return deep_merge(base, loaded)
    return loaded


def load_config(
    path: Optional[str | Path] = None,
    *,
    config_cls: Type[T] = ExperimentConfig,
    overrides: Optional[Mapping[str, Any]] = None,
) -> T:
    """Load a YAML config file, merge overrides, and build a typed config.

    Parameters
    ----------
    path:
        YAML file. ``None`` yields an all-defaults config. A top-level
        ``extends: base.yaml`` key merges another file underneath this one.
    config_cls:
        One of :class:`ExperimentConfig`, :class:`TrainingConfig`,
        :class:`EvaluationConfig`.
    overrides:
        Dotted-key overrides applied on top of the file, e.g.
        ``{"seed": 3, "warehouse.n_robots": 5}``.
    """
    data: Dict[str, Any] = {}
    if path is not None:
        data = _load_yaml_with_extends(Path(path))
    if overrides:
        data = apply_overrides(data, overrides)
    return _build_config(config_cls, data)


def save_config(config: Any, path: str | Path) -> None:
    """Persist a config dataclass (or dict) as YAML for reproducibility."""
    from dataclasses import asdict

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(config) if is_dataclass(config) else config
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(payload, fh, sort_keys=False)
