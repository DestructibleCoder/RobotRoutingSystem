"""Tests for the configuration system."""

from __future__ import annotations

from pathlib import Path

import pytest

from warehouse_rl.config import (
    EvaluationConfig,
    ExperimentConfig,
    TrainingConfig,
    load_config,
    save_config,
)

ROOT = Path(__file__).resolve().parents[1]


def test_default_config_loads():
    config = load_config(ROOT / "configs/default.yaml", config_cls=ExperimentConfig)
    assert isinstance(config, ExperimentConfig)
    assert config.warehouse.n_robots == 4
    assert config.warehouse.max_steps == 500
    assert config.orders.arrival_rate == 0.35
    assert config.layout.n_train == 10
    assert config.greedy.strategy == "min_completion_time"


def test_training_config_extends_default():
    config = load_config(ROOT / "configs/training.yaml", config_cls=TrainingConfig)
    assert isinstance(config, TrainingConfig)
    # Inherited from default.yaml.
    assert config.warehouse.n_robots == 4
    assert config.orders.arrival_rate == 0.35
    # Own fields.
    assert config.algorithm == "greedy"
    assert config.n_training_layouts == 1
    assert config.seeds == [0]


def test_evaluation_config_extends_default():
    config = load_config(ROOT / "configs/evaluation.yaml", config_cls=EvaluationConfig)
    assert isinstance(config, EvaluationConfig)
    assert config.algorithms == ["greedy"]
    assert config.n_episodes == 3
    assert config.seeds == [0, 1, 2]
    assert config.save_csv is True


def test_no_config_yields_defaults():
    config = load_config(None, config_cls=ExperimentConfig)
    assert config.seed == 0
    assert config.warehouse.n_robots == 4


def test_dotted_overrides():
    config = load_config(
        ROOT / "configs/default.yaml",
        config_cls=ExperimentConfig,
        overrides={
            "seed": 9,
            "warehouse.n_robots": 6,
            "orders.arrival_rate": 0.1,
            "greedy.strategy": "nearest_robot",
        },
    )
    assert config.seed == 9
    assert config.warehouse.n_robots == 6
    assert config.orders.arrival_rate == 0.1
    assert config.greedy.strategy == "nearest_robot"


def test_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_config("does/not/exist.yaml")


def test_invalid_section_type_raises():
    with pytest.raises((TypeError, ValueError)):
        load_config(
            ROOT / "configs/default.yaml",
            config_cls=ExperimentConfig,
            overrides={"warehouse.n_robots": "not-a-number"},
        )


def test_save_and_reload(tmp_path: Path):
    config = load_config(ROOT / "configs/default.yaml", config_cls=ExperimentConfig)
    out = tmp_path / "snapshot.yaml"
    save_config(config, out)
    reloaded = load_config(out, config_cls=ExperimentConfig)
    assert reloaded.warehouse.n_robots == config.warehouse.n_robots
    assert reloaded.layout.n_train == config.layout.n_train
    assert reloaded.seed == config.seed