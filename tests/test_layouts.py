"""Tests for layout representation and generation."""

from __future__ import annotations

from pathlib import Path

import pytest

from warehouse_rl.environment.layout_generator import (
    Layout,
    generate_layout,
    generate_layout_suite,
    generate_train_test_suites,
)


def test_layout_roundtrip(tmp_path: Path):
    rows = ["#####", "#P.D#", "#####", "#R..#", "#####"]
    layout = Layout(name="tiny", rows=rows)
    assert layout.width == 5
    assert layout.height == 5
    assert layout.pick_locations == [(1, 1)]
    assert layout.dropoff_locations == [(1, 3)]
    assert layout.robot_starts == [(3, 1)]

    path = layout.write(tmp_path / "tiny.txt")
    loaded = Layout.read(path)
    assert loaded.name == "tiny"
    assert loaded.rows == rows


def test_layout_rejects_non_rectangular():
    with pytest.raises(ValueError, match="rectangular"):
        Layout(name="bad", rows=["###", "##"])


def test_layout_rejects_invalid_characters():
    with pytest.raises(ValueError, match="invalid characters"):
        Layout(name="bad", rows=["###", "#X#", "###"])


def test_layout_rejects_empty():
    with pytest.raises(ValueError):
        Layout(name="bad", rows=[])


def test_walkable_mask():
    rows = ["###", "#P#", "###"]
    layout = Layout(name="m", rows=rows)
    mask = layout.walkable_mask()
    assert mask.shape == (3, 3)
    assert not mask[0, 0]  # wall
    assert mask[1, 1]  # pick station is walkable


def test_generate_layout_is_connected_and_has_semantics():
    layout = generate_layout("gen", width=20, height=14, seed=7)
    assert layout.is_connected()
    assert len(layout.pick_locations) >= 2
    assert len(layout.dropoff_locations) >= 1
    assert len(layout.robot_starts) >= 1


def test_generate_layout_deterministic():
    a = generate_layout("a", width=18, height=12, seed=42)
    b = generate_layout("a", width=18, height=12, seed=42)
    assert a.rows == b.rows


def test_generate_layout_minimum_size_check():
    with pytest.raises(ValueError, match="at least 8x6"):
        generate_layout("s", width=5, height=5, seed=0)


def test_generate_layout_suite_distinct_names_and_sizes():
    suite = generate_layout_suite(
        4, seed=10, min_width=14, min_height=10, max_width=24, max_height=16
    )
    assert len(suite) == 4
    assert len({l.name for l in suite}) == 4
    for layout in suite:
        assert 14 <= layout.width <= 24
        assert 10 <= layout.height <= 16
        assert layout.is_connected()


def test_train_test_suites_are_disjoint(tmp_path: Path):
    train_dir = tmp_path / "train"
    test_dir = tmp_path / "test"
    train, test = generate_train_test_suites(
        n_train=5,
        n_test=5,
        train_dir=train_dir,
        test_dir=test_dir,
        generation_seed=1234,
    )
    names = lambda layouts: {l.name for l in layouts}  # noqa: E731
    assert names(train).isdisjoint(names(test))
    assert sorted(p.stem for p in train_dir.glob("*.txt")) == sorted(names(train))
    assert sorted(p.stem for p in test_dir.glob("*.txt")) == sorted(names(test))


def test_layout_connectivity_checker(tmp_path: Path):
    split = Layout(name="split", rows=["#P#", "###", "#R#"])
    assert not split.is_connected()


def test_layout_to_ascii_roundtrip():
    rows = ["###", "#P#", "#R#"]
    layout = Layout(name="x", rows=rows)
    assert layout.to_ascii() == "\n".join(rows)