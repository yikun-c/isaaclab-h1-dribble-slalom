"""Deterministic physical-wall layout shared by CPU planning and Isaac rendering."""

from __future__ import annotations

from dataclasses import dataclass

from .core import Heading, MazeTask


@dataclass(frozen=True)
class WallSpec:
    name: str
    translation: tuple[float, float, float]
    size: tuple[float, float, float]


def maze_wall_specs(
    task: MazeTask,
    cell_size: float = 1.8,
    wall_thickness: float = 0.12,
    wall_height: float = 1.2,
    origin_xy: tuple[float, float] = (0.0, 0.0),
) -> tuple[WallSpec, ...]:
    """Convert closed grid edges into non-duplicated cuboid walls.

    The task start cell is centered at ``origin_xy``. East and south edges are
    emitted for every cell; north/west outer boundaries are emitted once.
    """
    if cell_size <= wall_thickness or wall_height <= 0.0:
        raise ValueError("cell size must exceed wall thickness and wall height must be positive")
    ox, oy = origin_xy
    specs: list[WallSpec] = []

    def center(cell: tuple[int, int]) -> tuple[float, float]:
        return ox + cell[0] * cell_size, oy + cell[1] * cell_size

    for cell in task.layout.cells:
        cx, cy = center(cell)
        x, y = cell
        if x == 0 and not task.layout.can_move(cell, Heading.WEST):
            specs.append(
                WallSpec(
                    f"west_{x}_{y}",
                    (cx - cell_size / 2.0, cy, wall_height / 2.0),
                    (wall_thickness, cell_size + wall_thickness, wall_height),
                )
            )
        if y == 0 and not task.layout.can_move(cell, Heading.NORTH):
            specs.append(
                WallSpec(
                    f"north_{x}_{y}",
                    (cx, cy - cell_size / 2.0, wall_height / 2.0),
                    (cell_size + wall_thickness, wall_thickness, wall_height),
                )
            )
        if not task.layout.can_move(cell, Heading.EAST):
            specs.append(
                WallSpec(
                    f"east_{x}_{y}",
                    (cx + cell_size / 2.0, cy, wall_height / 2.0),
                    (wall_thickness, cell_size + wall_thickness, wall_height),
                )
            )
        if not task.layout.can_move(cell, Heading.SOUTH):
            specs.append(
                WallSpec(
                    f"south_{x}_{y}",
                    (cx, cy + cell_size / 2.0, wall_height / 2.0),
                    (cell_size + wall_thickness, wall_thickness, wall_height),
                )
            )
    return tuple(specs)
