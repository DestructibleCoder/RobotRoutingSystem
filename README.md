# warehouse-rl — Generalizing RL Task-Allocation Policies Across Warehouse Layouts

**Research question:** *Can a reinforcement learning policy trained on a limited
number of warehouse layouts efficiently solve tasks in previously unseen
warehouse layouts?*

This repository is a research framework for studying **generalization across
environments** rather than peak performance on a single warehouse. The learning
problem is **task allocation** — assigning incoming orders to homogeneous mobile
robots — while low-level navigation is handled classically with A*.

> **Status (iteration 1):** simulator, A* navigation, the greedy baseline, the
> configuration system, layout generation, metrics collection, and the
> evaluation pipeline are implemented and tested. RL (Milestones 3–6) is stubbed
> but not yet implemented. See [Roadmap](#roadmap).

---

## 1. Problem model and assumptions

| Aspect | Assumption |
|---|---|
| Robots | Fixed number of **homogeneous** mobile robots (identical speed/capacity). |
| Warehouse | Grid-based: racks/obstacles (`#`), pick stations (`P`), drop-off/packing stations (`D`), robot spawns (`R`). |
| Orders | Dynamically arriving *pickup-and-delivery* jobs (`P` → `D`). |
| Decisions | A policy assigns pending orders to idle robots every `decision_interval` steps. |
| Navigation | A* on the static grid (4- or 8-connected). Movement is deterministic; no collision avoidance between robots (documented simplification). |
| Learning | RL decides *allocation only*; it never emits per-step movements. |
| Generalization | Policies are evaluated on layouts **never seen** during training. |

Robots follow the state machine `idle → to_pickup → picking → to_dropoff →
dropping → idle`. Orders carry timestamps so waiting/completion/service times
are computed exactly.

### Why a custom simulator? (assessment of existing environments)

The well-known **RWARE** (`multi-robot-warehouse`, NeurIPS 2021 benchmark) was
reviewed. It is a gym-compatible multi-agent environment with good documentation,
but it was **not adopted**, because:

1. RL controls **low-level movement** (`TURN_LEFT/RIGHT/FORWARD/LOAD`); there is
   no external task-allocation layer and no classical A* sub-planner.
2. Orders are *shelf-retrieval* requests, not modular pickup-and-delivery jobs
   with measurable waiting/completion times.
3. Training layouts are fixed presets (`tiny/small/medium/large`), making a
   clean layout-generalization study awkward.

Since the research question hinges on **allocation decisions generalizing across
layouts** (with A* as a fixed low-level solver), a lean custom simulator gives
full control over layouts, order streams, timestamps, and metrics. This
simulator is deliberately framework-agnostic; a thin Gymnasium wrapper is planned
for Milestone 3.

---

## 2. Repository structure

```text
.
├── README.md
├── pyproject.toml
├── configs/
│   ├── default.yaml        # base configuration
│   ├── training.yaml       # extends default; algorithm + training knobs
│   └── evaluation.yaml     # extends default; evaluation/compare knobs
├── src/warehouse_rl/
│   ├── config.py           # typed YAML-backed configuration
│   ├── environment/
│   │   ├── warehouse.py    # core discrete-step simulation
│   │   ├── robot.py        # homogeneous robot state machine
│   │   ├── orders.py       # order model + stochastic order stream
│   │   ├── layout_generator.py  # layout repr + procedural generation
│   │   └── rendering.py    # ASCII/PNG/GIF rendering (optional)
│   ├── navigation/
│   │   └── astar.py        # A* + memoized distance oracle
│   ├── algorithms/
│   │   ├── greedy.py       # greedy baseline (implemented)
│   │   ├── ppo.py          # stub (Milestone 4)
│   │   └── dqn.py          # stub (Milestone 6, optional)
│   ├── experiments/
│   │   ├── train.py        # training runner (greedy → no-op; PPO/DQN later)
│   │   ├── evaluate.py     # policy-agnostic evaluation → CSV
│   │   └── compare.py      # algorithm comparison (same test layouts/seeds)
│   └── metrics/
│       └── metrics.py      # episodic metrics + aggregation
├── layouts/
│   ├── train/              # 10 generated training layouts
│   └── test/               # 10 generated test layouts (disjoint seeds)
├── tests/                  # pytest: navigation, layouts, environment, config
├── scripts/
│   ├── generate_layouts.py
│   ├── train.py
│   ├── evaluate.py
│   └── demo.py             # end-to-end smoke demo with rendering
├── results/                # evaluation CSVs, snapshots, run artifacts
└── notebooks/              # analysis notebooks (planned)
```

## 3. Installation

Requires Python ≥ 3.10 (developed on 3.14). Uses `uv`:

```bash
uv venv --python 3.14 .venv
uv pip install -e ".[dev,analysis]"     # core + pytest + pandas + matplotlib
# RL stack (Milestone 4+, heavy):
uv pip install -e ".[rl]"
```

`no` extra installs the full PyTorch + Stable-Baselines3 stack.

## 4. Quickstart

```bash
# 1) Generate the 10 training + 10 test layouts (disjoint seed streams)
python scripts/generate_layouts.py

# 2) End-to-end demo: one greedy episode, ASCII frames + PNG snapshot
python scripts/demo.py --every 100 --snapshot results/demo.png

# 3) Evaluate greedy on all test layouts (same test set for every algorithm)
python scripts/evaluate.py --episodes 3

# 4) "Train" (greedy is non-learning; instantiates policy + snapshots config)
python scripts/train.py

# 5) Run the test suite
python -m pytest
```

Every script is reproducible: seeds live in the configs and are logged.

## 5. Configuration

All knobs live in `configs/`. `training.yaml` and `evaluation.yaml` `extends:`
`default.yaml` (deep merge). Typed dataclasses (`ExperimentConfig`,
`TrainingConfig`, `EvaluationConfig`) expose them; unknown keys warn but do not
crash. CLI overrides use dotted keys, e.g.:

```bash
python scripts/evaluate.py \
  --algorithm greedy \
  --episodes 5 \
  && python -c "from warehouse_rl.config import load_config, EvaluationConfig; \
     c = load_config('configs/evaluation.yaml', config_cls=EvaluationConfig); print(c)"
```

Key parameters: `warehouse.{n_robots,width,height,max_steps,connectivity,
decision_interval,pick_dwell_steps,drop_dwell_steps}`,
`orders.{arrival_rate,max_pending,max_orders,timeout_steps}`,
`layout.{train_dir,test_dir,generation_seed,min/max dimensions}`,
`greedy.strategy` (`min_completion_time` | `nearest_robot`), `seed`.

## 6. Metrics

Collected per episode (in `metrics/metrics.py`, written to CSV):

| Metric | Definition |
|---|---|
| avg completion time | mean `completed − created` steps |
| avg waiting time | mean `assigned − created` steps |
| avg service time | mean `completed − assigned` steps |
| total travel distance | summed robot movement (grid units) |
| throughput | completed orders per 100 steps |
| robot utilization | busy robot-steps / (n_robots × duration) |

Plus `episode_duration`, `n_orders_created/completed/timed_out/failed`. Aggregate
mean/std across layouts and seeds are produced by `compare.summarize_records`.

## 7. Design notes and documented simplifications

- Robots do **not** avoid each other (shared cells allowed). Collision-free
  movement is future work; it does not affect the *allocation* research question.
- Episodes have a **fixed horizon** (`max_steps`); `done == truncated`.
- `timeout_steps` currently expires only *unassigned* orders.
- The greedy baseline is deterministic and, by design, **never tuned**. We do
  **not** assume RL will outperform it.

## 8. Reproducibility & research integrity

- Robot count, warehouse dimensions, layout counts, and every seed are
  config-driven.
- Train/test layouts come from **disjoint seed streams**
  (`configuration seed` with a large offset for test), so test layouts are
  strictly unseen.
- The **same test layouts and seeds** are used for all compared algorithms.
- Config snapshots (`config.yaml`) and raw per-episode CSVs are persisted
  automatically; model checkpoints will be saved once RL lands.
- No hardcoded results anywhere: evaluation runs the simulator from scratch.

## 9. Roadmap

| Milestone | Scope | Status |
|---|---|---|
| M1 | Grid environment, robots, order stream, A*, rendering | ✅ |
| M2 | Greedy baseline, verified order completion, metrics | ✅ |
| M3 | Gymnasium wrapper: observation/action/reward spaces | ⏳ next |
| M4 | PPO training on one layout, save/reload, basic eval | ⏳ |
| M5 | Training on N layouts → evaluate on unseen layouts, CSV | ⏳ |
| M6 | PPO vs greedy (+ DQN if practical) across training-set sizes | ⏳ |

Experiment A (train on 10, test on unseen), Experiment B (training-set sizes
1/3/5/10 on a fixed test set), and Experiment C (greedy vs PPO vs optional DQN)
will be implemented on top of this foundation.