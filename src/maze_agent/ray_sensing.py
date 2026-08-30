"""Axis-aligned local ray sensing against the generated physical wall cuboids."""

from __future__ import annotations

from dataclasses import dataclass

from .core import Heading, MazeTask
from .physical_maze import WallSpec, maze_wall_specs


@dataclass(frozen=True)
class LocalRayRanges:
    front_m: float
    left_m: float
    right_m: float
    rear_m: float

    def open_by_direction(self, minimum_clearance_m: float) -> dict[str, bool]:
        if minimum_clearance_m <= 0.0:
            raise ValueError("minimum_clearance_m must be positive")
        return {
            "front": self.front_m >= minimum_clearance_m,
            "left": self.left_m >= minimum_clearance_m,
            "right": self.right_m >= minimum_clearance_m,
            "rear": self.rear_m >= minimum_clearance_m,
        }


def ray_distance_to_wall(
    walls: tuple[WallSpec, ...], origin_xy: tuple[float, float], heading: Heading, maximum_distance_m: float = 100.0
) -> float:
    """Return first hit distance for an axis-aligned horizontal ray.

    Wall cuboids are treated as their 2D collision rectangles, so the result is
    derived from the same geometry that is spawned in Isaac rather than the
    planner querying a full maze map.
    """
    if maximum_distance_m <= 0.0:
        raise ValueError("maximum_distance_m must be positive")
    ox, oy = origin_xy
    candidates: list[float] = []
    for wall in walls:
        wx, wy, _ = wall.translation
        sx, sy, _ = wall.size
        min_x, max_x = wx - sx / 2.0, wx + sx / 2.0
        min_y, max_y = wy - sy / 2.0, wy + sy / 2.0
        if heading is Heading.EAST and min_y <= oy <= max_y and min_x >= ox:
            candidates.append(min_x - ox)
        elif heading is Heading.WEST and min_y <= oy <= max_y and max_x <= ox:
            candidates.append(ox - max_x)
        elif heading is Heading.SOUTH and min_x <= ox <= max_x and min_y >= oy:
            candidates.append(min_y - oy)
        elif heading is Heading.NORTH and min_x <= ox <= max_x and max_y <= oy:
            candidates.append(oy - max_y)
    valid = [distance for distance in candidates if distance >= 0.0]
    return min(valid, default=maximum_distance_m)


def sense_physical_maze(
    task: MazeTask,
    position: tuple[int, int],
    heading: Heading,
    cell_size: float = 1.8,
    wall_thickness: float = 0.12,
    origin_xy: tuple[float, float] = (0.0, 0.0),
) -> LocalRayRanges:
    """Compute planner-local ranges at a cell center from physical wall specs."""
    if cell_size <= wall_thickness:
        raise ValueError("cell_size must exceed wall_thickness")
    ox, oy = origin_xy
    center = (ox + position[0] * cell_size, oy + position[1] * cell_size)
    walls = maze_wall_specs(task, cell_size=cell_size, wall_thickness=wall_thickness, origin_xy=origin_xy)
    return LocalRayRanges(
        front_m=ray_distance_to_wall(walls, center, heading),
        left_m=ray_distance_to_wall(walls, center, heading.left()),
        right_m=ray_distance_to_wall(walls, center, heading.right()),
        rear_m=ray_distance_to_wall(walls, center, heading.opposite()),
    )
