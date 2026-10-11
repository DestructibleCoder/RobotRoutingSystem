"""Warehouse layout representation and procedural generation.

A *layout* is a small ASCII grid. Character legend:

====  ================================================================
Char  Meaning
====  ================================================================
``.``  Free floor (walkable)
``#``  Obstacle / storage rack (blocked)
``P``  Order-picking station (walkable; orders are picked up here)
``D``  Drop-off / packing station (walkable; orders are delivered here)
``R``  Robot spawn cell (walkable)
====  ================================================================

Layouts are stored as plain text files under ``layouts/train`` and
``layouts/test``. Training and test layouts are generated from disjoint seed
streams so the generalization guarantee (unseen layouts) is actually testable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

Cell = Tuple[int, int]

FLOOR = "."
OBSTACLE = "#"
PICK = "P"
DROPOFF = "D"
ROBOT_START = "R"
_ALLOWED = frozenset({FLOOR, OBSTACLE, PICK, DROPOFF, ROBOT_START})


@dataclass
class Layout:
    """A rectangular warehouse layout with labelled cells."""

    name: str
    rows: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.rows = [str(r) for r in self.rows]
        if not self.rows:
            raise ValueError("Layout must have at least one row")
        width = len(self.rows[0])
        if width == 0:
            raise ValueError("Layout rows must be non-empty")
        for i, row in enumerate(self.rows):
            if len(row) != width:
                raise ValueError(
                    f"Layout '{self.name}' is not rectangular: row {i} has width "
                    f"{len(row)}, expected {width}"
                )
            bad = set(row) - _ALLOWED
            if bad:
                raise ValueError(f"Layout '{self.name}' has invalid characters: {sorted(bad)}")

    # -- geometry ---------------------------------------------------------- #
    @property
    def width(self) -> int:
        return len(self.rows[0])

    @property
    def height(self) -> int:
        return len(self.rows)

    def char_at(self, cell: Cell) -> str:
        return self.rows[cell[0]][cell[1]]

    def array(self) -> np.ndarray:
        return np.array([list(r) for r in self.rows], dtype="<U1")

    def walkable_mask(self) -> np.ndarray:
        arr = self.array()
        return arr != OBSTACLE

    def cells(self, char: str) -> List[Cell]:
        out: List[Cell] = []
        for r, row in enumerate(self.rows):
            for c, ch in enumerate(row):
                if ch == char:
                    out.append((r, c))
        return out

    # -- semantic accessors ------------------------------------------------ #
    @property
    def pick_locations(self) -> List[Cell]:
        return self.cells(PICK)

    @property
    def dropoff_locations(self) -> List[Cell]:
        return self.cells(DROPOFF)

    @property
    def robot_starts(self) -> List[Cell]:
        return self.cells(ROBOT_START)

    @property
    def free_cells(self) -> List[Cell]:
        mask = self.walkable_mask()
        return [(int(r), int(c)) for r, c in zip(*np.where(mask))]

    # -- I/O --------------------------------------------------------------- #
    def to_ascii(self) -> str:
        return "\n".join(self.rows)

    def write(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_ascii() + "\n", encoding="utf-8")
        return path

    @classmethod
    def read(cls, path: str | Path, name: Optional[str] = None) -> "Layout":
        path = Path(path)
        rows = [ln.rstrip("\n") for ln in path.read_text(encoding="utf-8").splitlines()]
        # Drop trailing blank lines but keep meaningful spaces (there are none).
        while rows and rows[-1] == "":
            rows.pop()
        return cls(name=name or path.stem, rows=rows)

    # -- validation -------------------------------------------------------- #
    def reachable_from(self, start: Cell, *, connectivity: int = 4) -> np.ndarray:
        """Boolean mask of cells reachable from ``start`` (BFS, 4/8-connected)."""
        walkable = self.walkable_mask()
        h, w = walkable.shape
        seen = np.zeros_like(walkable, dtype=bool)
        if not (0 <= start[0] < h and 0 <= start[1] < w) or not walkable[start]:
            return seen
        steps = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        if connectivity == 8:
            steps += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        stack = [start]
        seen[start] = True
        while stack:
            r, c = stack.pop()
            for dr, dc in steps:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and walkable[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        return seen

    def is_connected(self, *, connectivity: int = 4) -> bool:
        """True if all functional cells (P, D, R) reach each other."""
        functional = self.pick_locations + self.dropoff_locations + self.robot_starts
        if not functional:
            return False
        seen = self.reachable_from(functional[0], connectivity=connectivity)
        return all(seen[cell] for cell in functional)

    def summary(self) -> dict:
        return {
            "name": self.name,
            "width": self.width,
            "height": self.height,
            "n_obstacles": len(self.cells(OBSTACLE)),
            "n_pick": len(self.pick_locations),
            "n_dropoff": len(self.dropoff_locations),
            "n_robot_starts": len(self.robot_starts),
        }

    def __str__(self) -> str:  # pragma: no cover - convenience
        return f"Layout({self.name}, {self.width}x{self.height})"


# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #


def _carve_racks(rng: np.random.Generator, grid: List[List[str]], min_seg: int, max_seg: int) -> None:
    """Place horizontal rack segments separated by vertical aisles."""
    h = len(grid)
    w = len(grid[0])
    # Rack rows: interior, every other row, leaving the final interior row as a
    # full-width service corridor.
    for r in range(2, h - 2, 2):
        c = 2
        while c < w - 2:
            seg = int(rng.integers(min_seg, max_seg + 1))
            seg = min(seg, w - 2 - c)
            if seg <= 0:
                break
            for cc in range(c, c + seg):
                grid[r][cc] = OBSTACLE
            # Leave a vertical aisle of 1-2 cells before the next segment.
            c += seg + int(rng.integers(1, 3))


def _place_pick_stations(
    rng: np.random.Generator,
    grid: List[List[str]],
    n_pick: int,
) -> List[Cell]:
    h = len(grid)
    w = len(grid[0])
    candidates: List[Cell] = []
    for r in range(1, h - 2):
        for c in range(1, w - 1):
            if grid[r][c] != FLOOR:
                continue
            # Adjacent to a rack -> plausible pick face.
            adjacent_rack = any(
                0 <= r + dr < h and 0 <= c + dc < w and grid[r + dr][c + dc] == OBSTACLE
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
            )
            if adjacent_rack:
                candidates.append((r, c))
    rng.shuffle(candidates)  # type: ignore[arg-type]
    chosen = candidates[: min(n_pick, len(candidates))]
    for r, c in chosen:
        grid[r][c] = PICK
    return chosen


def _place_dropoffs_and_starts(
    rng: np.random.Generator,
    grid: List[List[str]],
    n_drop: int,
    max_starts: int,
) -> Tuple[List[Cell], List[Cell]]:
    """Place drop-off stations and robot starts along the service corridor."""
    h = len(grid)
    w = len(grid[0])
    corridor = h - 2
    free_cols = [c for c in range(1, w - 1) if grid[corridor][c] == FLOOR]
    rng.shuffle(free_cols)  # type: ignore[arg-type]
    if not free_cols:
        return [], []
    n_drop = max(1, min(n_drop, max(1, len(free_cols) // 2)))
    drop_cols = sorted(free_cols[:n_drop])
    drops = [(corridor, c) for c in drop_cols]
    for cell in drops:
        grid[cell[0]][cell[1]] = DROPOFF

    remaining = [c for c in range(1, w - 1) if grid[corridor][c] == FLOOR]
    # Start cells sit next to the drop-offs; prefer left-to-right spawning order.
    rng.shuffle(remaining)  # type: ignore[arg-type]
    start_cols = sorted(remaining[: min(max_starts, len(remaining))])
    starts = [(corridor, c) for c in start_cols]
    for cell in starts:
        grid[cell[0]][cell[1]] = ROBOT_START
    return drops, starts


def generate_layout(
    name: str,
    *,
    width: int = 20,
    height: int = 14,
    seed: int = 0,
    n_pick: int = 4,
    n_dropoff: int = 3,
    max_starts: int = 8,
    connectivity: int = 4,
    max_attempts: int = 25,
) -> Layout:
    """Generate one reproducible, connected warehouse layout.

    The generator uses rejection sampling: it carves racks/aisles, scatters
    pick stations, drop-offs, and robot starts, then verifies that every
    functional cell is mutually reachable. Failed candidates are discarded and
    a new layout is drawn from the same seed stream.
    """
    if width < 8 or height < 6:
        raise ValueError("Layouts must be at least 8x6")
    rng = np.random.default_rng(seed)
    last_error = "no candidate produced"
    for _ in range(max_attempts):
        grid = [[FLOOR] * width for _ in range(height)]
        # Border walls.
        for c in range(width):
            grid[0][c] = OBSTACLE
            grid[height - 1][c] = OBSTACLE
        for r in range(height):
            grid[r][0] = OBSTACLE
            grid[r][width - 1] = OBSTACLE
        _carve_racks(rng, grid, min_seg=3, max_seg=min(7, max(3, width // 4)))
        _place_pick_stations(rng, grid, n_pick)
        _place_dropoffs_and_starts(rng, grid, n_dropoff, max_starts)

        candidate = Layout(name=name, rows=["".join(row) for row in grid])
        if not candidate.pick_locations:
            last_error = "no pick stations placed"
            continue
        if not candidate.dropoff_locations:
            last_error = "no drop-off stations placed"
            continue
        if not candidate.robot_starts:
            last_error = "no robot start cells placed"
            continue
        if not candidate.is_connected(connectivity=connectivity):
            last_error = "generated layout not fully connected"
            continue
        return candidate
    raise RuntimeError(f"Failed to generate layout '{name}' after {max_attempts} attempts: {last_error}")


def generate_layout_suite(
    n: int,
    *,
    seed: int,
    prefix: str = "layout",
    min_width: int = 14,
    min_height: int = 10,
    max_width: int = 26,
    max_height: int = 18,
    connectivity: int = 4,
    n_pick: int = 4,
    n_dropoff: int = 3,
) -> List[Layout]:
    """Generate ``n`` distinct layouts with randomized dimensions."""
    if n <= 0:
        return []
    master = np.random.default_rng(seed)
    layouts: List[Layout] = []
    for i in range(n):
        width = int(master.integers(min_width, max_width + 1))
        height = int(master.integers(min_height, max_height + 1))
        layout_seed = int(master.integers(0, 2**31 - 1))
        layouts.append(
            generate_layout(
                f"{prefix}_{i:02d}",
                width=width,
                height=height,
                seed=layout_seed,
                n_pick=n_pick,
                n_dropoff=n_dropoff,
                connectivity=connectivity,
            )
        )
    return layouts


# --------------------------------------------------------------------------- #
# Suite I/O
# --------------------------------------------------------------------------- #


def write_suite(layouts: Sequence[Layout], directory: str | Path) -> List[Path]:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    written = []
    for layout in layouts:
        written.append(layout.write(directory / f"{layout.name}.txt"))
    return written


def load_suite(directory: str | Path) -> List[Layout]:
    directory = Path(directory)
    if not directory.exists():
        return []
    return [Layout.read(p, name=p.stem) for p in sorted(directory.glob("*.txt"))]


def generate_train_test_suites(
    *,
    n_train: int,
    n_test: int,
    train_dir: str | Path,
    test_dir: str | Path,
    generation_seed: int = 20240101,
    min_width: int = 14,
    min_height: int = 10,
    max_width: int = 26,
    max_height: int = 18,
    connectivity: int = 4,
) -> Tuple[List[Layout], List[Layout]]:
    """Generate and persist disjoint train/test suites.

    Train and test layouts come from *separate* seed streams (offset by a large
    constant) so no test layout can coincide with a training layout.
    """
    train = generate_layout_suite(
        n_train,
        seed=generation_seed,
        prefix="train",
        min_width=min_width,
        min_height=min_height,
        max_width=max_width,
        max_height=max_height,
        connectivity=connectivity,
    )
    test = generate_layout_suite(
        n_test,
        seed=generation_seed + 1_000_003,
        prefix="test",
        min_width=min_width,
        min_height=min_height,
        max_width=max_width,
        max_height=max_height,
        connectivity=connectivity,
    )
    write_suite(train, train_dir)
    write_suite(test, test_dir)
    return train, test
