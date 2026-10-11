"""Tests for the warehouse simulation and greedy baseline end-to-end."""

from __future__ import annotations

import pytest

from warehouse_rl.algorithms.greedy import GreedyPolicy
from warehouse_rl.config import ExperimentConfig, load_config
from warehouse_rl.environment.layout_generator import generate_layout
from warehouse_rl.environment.orders import ASSIGNED, PENDING
from warehouse_rl.environment.warehouse import WarehouseSimulation
from warehouse_rl.metrics.metrics import compute_episode_metrics


@pytest.fixture(scope="module")
def layout():
    return generate_layout("testsim", width=16, height=10, seed=1, n_pick=3, n_dropoff=2)


def make_config(**overrides) -> ExperimentConfig:
    return load_config(None, config_cls=ExperimentConfig, overrides=overrides)


def _metrics_for(config, layout, seed):
    sim = WarehouseSimulation(layout, config, seed=seed)
    sim.run(policy=GreedyPolicy())
    return compute_episode_metrics(sim, policy="greedy", seed=seed)


def test_episode_runs_to_completion_with_greedy(layout):
    config = make_config(
        **{"warehouse.max_steps": 120, "warehouse.n_robots": 4, "orders.arrival_rate": 0.4}
    )
    sim = WarehouseSimulation(layout, config, seed=3)
    assert sim.step_count == 0
    sim.run(policy=GreedyPolicy())
    assert sim.step_count == config.warehouse.max_steps
    assert sim.done

    metrics = compute_episode_metrics(sim, policy="greedy", seed=3)
    # With arrival_rate=0.4 over 120 steps the stream should complete orders.
    assert metrics.n_orders_completed > 0
    assert metrics.n_orders_created >= metrics.n_orders_completed
    assert metrics.total_travel_distance > 0.0
    assert 0.0 <= metrics.robot_utilization <= 1.0
    assert metrics.avg_completion_time > 0.0
    assert metrics.throughput_per_100_steps >= 0.0
    assert metrics.episode_duration == config.warehouse.max_steps


def test_run_deterministic_across_instances(layout):
    config = make_config(
        **{"warehouse.max_steps": 80, "warehouse.n_robots": 3, "orders.arrival_rate": 0.3}
    )
    m1 = _metrics_for(config, layout, seed=7)
    m2 = _metrics_for(config, layout, seed=7)
    assert m1.to_row() == m2.to_row()
    m3 = _metrics_for(config, layout, seed=8)
    assert m1.to_row() != m3.to_row()


def test_reset_restores_initial_state(layout):
    config = make_config(**{"warehouse.n_robots": 2, "warehouse.max_steps": 20})
    sim = WarehouseSimulation(layout, config, seed=0)
    sim.run(policy=GreedyPolicy())
    assert sim.step_count == 20
    sim.reset(seed=0)
    assert sim.step_count == 0
    assert [r.position for r in sim.robots] == sorted(layout.robot_starts)[:2]
    assert sim.orders == {}


def test_assign_validation(layout):
    config = make_config(
        **{
            "warehouse.n_robots": 2,
            "warehouse.max_steps": 30,
            "orders.arrival_rate": 1.0,
            "orders.max_pending": 10,
        }
    )
    sim = WarehouseSimulation(layout, config, seed=0)
    sim.step()  # one order arrives (id 0), no policy -> stays pending
    order = sim.orders[0]
    assert order.status == PENDING

    assert sim.can_assign(0, order.id)  # robot 0 idle
    assert sim.assign(0, order.id) is True
    assert sim.assign(0, order.id) is False  # robot now busy
    assert sim.assign(1, order.id) is False  # order no longer pending
    assert sim.assign(0, 123) is False  # unknown order
    assert sim.assign(99, order.id) is False  # invalid robot id


def test_policy_assignments_are_legal(layout):
    """Greedy must only pair idle robots with pending orders."""
    config = make_config(
        **{
            "warehouse.n_robots": 3,
            "warehouse.max_steps": 30,
            "orders.arrival_rate": 0.9,
            "orders.max_pending": 20,
        }
    )
    sim = WarehouseSimulation(layout, config, seed=0)
    policy = GreedyPolicy()
    for _ in range(30):
        result = sim.step(policy=policy)
        for robot_id, order_id in result.assignments:
            assert sim.orders[order_id].status == ASSIGNED
            robot = sim.robots[robot_id]
            assert robot.status == "to_pickup"
            assert robot.order_id == order_id
        # Robots may be assigned at most one order per step.
        ids = [robot_id for robot_id, _ in result.assignments]
        assert len(ids) == len(set(ids))


def test_unassigned_orders_time_out(layout):
    config = make_config(
        **{
            "warehouse.n_robots": 1,
            "warehouse.max_steps": 12,
            "orders.arrival_rate": 1.0,
            "orders.max_pending": 30,
            "orders.timeout_steps": 3,
            "warehouse.decision_interval": 5,
        }
    )
    sim = WarehouseSimulation(layout, config, seed=0)
    sim.run(policy=GreedyPolicy())
    assert len(sim.timed_out_order_ids) > 0
    metrics = compute_episode_metrics(sim)
    assert metrics.n_orders_timed_out > 0


def test_greedy_requires_reachable_pickup():
    # A disconnected layout with an unreachable pick station forces the greedy
    # policy to skip that order (estimate_completion_time -> None).
    from warehouse_rl.environment.layout_generator import Layout

    split = Layout(
        name="split",
        rows=[
            "#########",
            "#P#######",
            "#########",
            "#D#####R#",
            "#########",
        ],
    )
    config = make_config(
        **{"warehouse.n_robots": 1, "warehouse.max_steps": 3, "orders.arrival_rate": 1.0}
    )
    sim = WarehouseSimulation(split, config, seed=0)
    # Order stream points at the only pick station (unreachable) so greedy can't
    # assign it; the episode still runs to completion.
    sim.run(policy=GreedyPolicy())
    assert sim.done
    metrics = compute_episode_metrics(sim)
    assert metrics.n_orders_completed == 0


def test_too_many_robots_rejected():
    tiny = generate_layout("tiny", width=8, height=8, seed=0, n_pick=2, n_dropoff=1, max_starts=1)
    config = make_config(**{"warehouse.n_robots": 5})
    with pytest.raises(ValueError, match="robot start"):
        WarehouseSimulation(tiny, config, seed=0)


def test_step_after_done_raises(layout):
    config = make_config(**{"warehouse.max_steps": 3})
    sim = WarehouseSimulation(layout, config, seed=0)
    sim.run(policy=GreedyPolicy())
    assert sim.done
    with pytest.raises(RuntimeError, match="already finished"):
        sim.step()


def test_metrics_zero_for_empty_episode(layout):
    config = make_config(
        **{"warehouse.max_steps": 10, "warehouse.n_robots": 2, "orders.arrival_rate": 0.0}
    )
    sim = WarehouseSimulation(layout, config, seed=0)
    sim.run()
    metrics = compute_episode_metrics(sim)
    assert metrics.n_orders_created == 0
    assert metrics.n_orders_completed == 0
    assert metrics.total_travel_distance == 0.0
    assert metrics.robot_utilization == 0.0


def test_waiting_time_equals_assigned_minus_created(layout):
    config = make_config(
        **{
            "warehouse.max_steps": 15,
            "warehouse.decision_interval": 3,  # assignment happens step 3, 6, ...
            "orders.arrival_rate": 1.0,
            "orders.max_pending": 10,
        }
    )
    sim = WarehouseSimulation(layout, config, seed=0)
    sim.run(policy=GreedyPolicy())
    for order in sim.orders.values():
        if order.status == ASSIGNED or order.assigned_step is not None:
            assert order.waiting_time == order.assigned_step - order.created_step
            assert order.waiting_time >= 0