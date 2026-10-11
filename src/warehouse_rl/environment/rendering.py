"""Optional rendering utilities (Milestone 1 visualization).

These helpers are intentionally optional and import Matplotlib lazily so the
core simulator has no plotting dependency.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

from warehouse_rl.config import ExperimentConfig
from warehouse_rl.environment.layout_generator import (
    DROPOFF,
    OBSTACLE,
    PICK,
    ROBOT_START,
    Layout,
)
from warehouse_rl.environment.warehouse import WarehouseSimulation

_CHAR_COLORS = {
    OBSTACLE: "#3b3b3b",
    PICK: "#4c9be8",
    DROPOFF: "#e8873c",
    ROBOT_START: "#7fbf7f",
    ".": "#f5f5f5",
}


def _draw(ax, sim: WarehouseSimulation, step: Optional[int] = None) -> None:
    rows = sim.layout.rows
    h, w = sim.layout.height, sim.layout.width
    ax.clear()
    for r in range(h):
        for c in range(w):
            color = _CHAR_COLORS.get(rows[r][c], "#f5f5f5")
            ax.add_patch(_rect(c, r, color))
    for robot in sim.robots:
        rr, cc = robot.position
        ax.add_patch(_circle(cc, rr, "#1a1a1a"))
        ax.text(cc + 0.5, rr + 0.5, str(robot.id), color="white", ha="center", va="center", fontsize=8)
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    title = f"{sim.layout.name}"
    if step is not None:
        title += f" — step {step}"
    ax.set_title(title, fontsize=9)


def _rect(x: float, y: float, color: str):
    from matplotlib.patches import Rectangle

    return Rectangle((x, y), 1, 1, facecolor=color, edgecolor="#cccccc", linewidth=0.3)


def _circle(x: float, y: float, color: str):
    from matplotlib.patches import Circle

    return Circle((x + 0.5, y + 0.5), 0.35, facecolor=color, edgecolor="white", linewidth=0.5)


def save_snapshot(sim: WarehouseSimulation, path: str | Path) -> Path:
    """Save a single-frame PNG of the current simulation state."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(sim.layout.width / 3, sim.layout.height / 3))
    _draw(ax, sim, step=sim.step_count)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def animate_episode(
    layout: Layout,
    config: ExperimentConfig,
    policy: object,
    path: str | Path,
    *,
    seed: int = 0,
    fps: int = 10,
) -> Path:
    """Run an episode and save an animated GIF of the whole trajectory."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.animation as animation
    import matplotlib.pyplot as plt

    sim = WarehouseSimulation(layout, config, seed=seed)
    frames: List[np.ndarray] = []
    fig, ax = plt.subplots(figsize=(layout.width / 3, layout.height / 3))

    sim.reset()
    _draw(ax, sim, step=0)
    fig.canvas.draw()
    frames.append(np.asarray(fig.canvas.buffer_rgba()))
    while not sim.done:
        sim.step(policy=policy)
        _draw(ax, sim, step=sim.step_count)
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba()))

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.close(fig)

    # Render the captured RGBA buffers as an animated GIF.
    fig2, ax2 = plt.subplots(figsize=(layout.width / 3, layout.height / 3))
    images = [[ax2.imshow(frame, animated=True)] for frame in frames]
    ax2.axis("off")
    anim = animation.ArtistAnimation(fig2, images, interval=1000 / fps, blit=True)
    anim.save(path, writer=animation.PillowWriter(fps=fps))
    plt.close(fig2)
    return path
