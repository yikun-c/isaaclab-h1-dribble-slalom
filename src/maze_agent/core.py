"""Pure-Python maze task with deterministic semantics shared by later Isaac integration."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field, replace
from enum import Enum
from random import Random
from typing import Iterable


GridPos = tuple[int, int]


class Heading(str, Enum):
    NORTH = "north"
    EAST = "east"
    SOUTH = "south"
    WEST = "west"

    def left(self) -> "Heading":
        return {
            Heading.NORTH: Heading.WEST,
            Heading.WEST: Heading.SOUTH,
            Heading.SOUTH: Heading.EAST,
            Heading.EAST: Heading.NORTH,
        }[self]

    def right(self) -> "Heading":
        return {
            Heading.NORTH: Heading.EAST,
            Heading.EAST: Heading.SOUTH,
            Heading.SOUTH: Heading.WEST,
            Heading.WEST: Heading.NORTH,
        }[self]

    def opposite(self) -> "Heading":
        return {
            Heading.NORTH: Heading.SOUTH,
            Heading.SOUTH: Heading.NORTH,
            Heading.EAST: Heading.WEST,
            Heading.WEST: Heading.EAST,
        }[self]

    @property
    def delta(self) -> GridPos:
        return {
            Heading.NORTH: (0, -1),
            Heading.EAST: (1, 0),
            Heading.SOUTH: (0, 1),
            Heading.WEST: (-1, 0),
        }[self]


CARDINAL_HEADINGS = (Heading.NORTH, Heading.EAST, Heading.SOUTH, Heading.WEST)


class Action(str, Enum):
    MOVE_FORWARD = "MOVE_FORWARD"
    TURN_LEFT = "TURN_LEFT"
    TURN_RIGHT = "TURN_RIGHT"
    BACKTRACK = "BACKTRACK"
    STOP = "STOP"


@dataclass(frozen=True)
class MazeLayout:
    """A connected cell graph; absent edges are physical walls."""

    width: int
    height: int
    seed: int
    passages: dict[GridPos, frozenset[Heading]]

    def __post_init__(self) -> None:
        if self.width < 5 or self.height < 5:
            raise ValueError("maze dimensions must be at least 5 by 5")
        expected = {(x, y) for y in range(self.height) for x in range(self.width)}
        if set(self.passages) != expected:
            raise ValueError("passages must contain every logical cell exactly once")
        for position, exits in self.passages.items():
            for heading in exits:
                neighbor = self.neighbor(position, heading)
                if neighbor not in expected or heading.opposite() not in self.passages[neighbor]:
                    raise ValueError("passages must be in bounds and bidirectional")

    @property
    def cells(self) -> tuple[GridPos, ...]:
        return tuple((x, y) for y in range(self.height) for x in range(self.width))

    def in_bounds(self, position: GridPos) -> bool:
        return 0 <= position[0] < self.width and 0 <= position[1] < self.height

    def neighbor(self, position: GridPos, heading: Heading) -> GridPos:
        dx, dy = heading.delta
        return position[0] + dx, position[1] + dy

    def can_move(self, position: GridPos, heading: Heading) -> bool:
        return heading in self.passages[position]

    def accessible_neighbors(self, position: GridPos) -> tuple[tuple[Heading, GridPos], ...]:
        return tuple(
            (heading, self.neighbor(position, heading))
            for heading in CARDINAL_HEADINGS
            if self.can_move(position, heading)
        )

    def shortest_path(
        self, start: GridPos, goal: GridPos, blocked: Iterable[GridPos] = ()
    ) -> tuple[GridPos, ...] | None:
        blocked_set = set(blocked)
        if start in blocked_set or goal in blocked_set:
            return None
        parents: dict[GridPos, GridPos | None] = {start: None}
        queue: deque[GridPos] = deque([start])
        while queue:
            current = queue.popleft()
            if current == goal:
                route: list[GridPos] = []
                while current is not None:
                    route.append(current)
                    current = parents[current]
                return tuple(reversed(route))
            for _, neighbor in self.accessible_neighbors(current):
                if neighbor not in blocked_set and neighbor not in parents:
                    parents[neighbor] = current
                    queue.append(neighbor)
        return None

    def canonical_edges(self) -> tuple[tuple[int, int, str], ...]:
        """Stable representation used to prove same-seed replay."""
        edges: list[tuple[int, int, str]] = []
        for position in self.cells:
            for heading in (Heading.EAST, Heading.SOUTH):
                if self.can_move(position, heading):
                    edges.append((position[0], position[1], heading.value))
        return tuple(edges)


def build_layout(width: int, height: int, seed: int) -> MazeLayout:
    """Generate one perfect maze with randomized depth-first carving."""
    if width < 5 or height < 5:
        raise ValueError("maze dimensions must be at least 5 by 5")
    rng = Random(seed)
    passages: dict[GridPos, set[Heading]] = {
        (x, y): set() for y in range(height) for x in range(width)
    }
    start = (0, 0)
    visited = {start}
    stack = [start]
    while stack:
        current = stack[-1]
        candidates = []
        for heading in CARDINAL_HEADINGS:
            dx, dy = heading.delta
            neighbor = current[0] + dx, current[1] + dy
            if 0 <= neighbor[0] < width and 0 <= neighbor[1] < height and neighbor not in visited:
                candidates.append((heading, neighbor))
        if not candidates:
            stack.pop()
            continue
        heading, neighbor = rng.choice(candidates)
        passages[current].add(heading)
        passages[neighbor].add(heading.opposite())
        visited.add(neighbor)
        stack.append(neighbor)
    return MazeLayout(
        width=width,
        height=height,
        seed=seed,
        passages={position: frozenset(exits) for position, exits in passages.items()},
    )


@dataclass(frozen=True)
class MazeTask:
    layout: MazeLayout
    start: GridPos
    exit: GridPos
    checkpoint: GridPos
    forbidden: GridPos
    task_id: str
    instruction: str

    def __post_init__(self) -> None:
        special = (self.start, self.exit, self.checkpoint, self.forbidden)
        if len(set(special)) != len(special):
            raise ValueError("start, exit, checkpoint and forbidden cells must differ")
        if any(not self.layout.in_bounds(position) for position in special):
            raise ValueError("task landmarks must be in bounds")
        route_to_checkpoint = self.layout.shortest_path(self.start, self.checkpoint, (self.forbidden,))
        route_to_exit = self.layout.shortest_path(self.checkpoint, self.exit, (self.forbidden,))
        if route_to_checkpoint is None or route_to_exit is None:
            raise ValueError("semantic task must remain solvable while avoiding forbidden cell")

    @property
    def optimal_route(self) -> tuple[GridPos, ...]:
        first = self.layout.shortest_path(self.start, self.checkpoint, (self.forbidden,))
        second = self.layout.shortest_path(self.checkpoint, self.exit, (self.forbidden,))
        assert first is not None and second is not None
        return first + second[1:]


def build_task(width: int, height: int, seed: int) -> MazeTask:
    """Create a solvable detour task: visit blue, avoid red, then stop at exit."""
    layout = build_layout(width, height, seed)
    start = (0, 0)
    exit_cell = (width - 1, height - 1)
    direct_path = set(layout.shortest_path(start, exit_cell) or ())
    rng = Random(f"semantic:{width}:{height}:{seed}")
    candidates = [position for position in layout.cells if position not in direct_path]
    rng.shuffle(candidates)
    for checkpoint in candidates:
        route = set(layout.shortest_path(start, checkpoint) or ())
        route.update(layout.shortest_path(checkpoint, exit_cell) or ())
        forbidden_candidates = [
            position
            for position in layout.cells
            if position not in route and position not in {start, exit_cell, checkpoint}
        ]
        if forbidden_candidates:
            forbidden = rng.choice(forbidden_candidates)
            return MazeTask(
                layout=layout,
                start=start,
                exit=exit_cell,
                checkpoint=checkpoint,
                forbidden=forbidden,
                task_id=f"maze-v1-w{width}-h{height}-seed{seed}",
                instruction="Visit the blue checkpoint, avoid the red cell, then stop at the exit.",
            )
    raise RuntimeError("could not place non-trivial semantic landmarks; use a larger maze or another seed")


@dataclass(frozen=True)
class MazeState:
    position: GridPos
    heading: Heading = Heading.EAST
    checkpoint_complete: bool = False
    terminated: bool = False
    success: bool = False
    terminal_reason: str | None = None
    steps: int = 0
    collisions: int = 0
    path: tuple[GridPos, ...] = field(default_factory=tuple)
    last_action: Action | None = None
    last_result: str = "reset"


@dataclass(frozen=True)
class Observation:
    front_open: bool
    left_open: bool
    right_open: bool
    rear_open: bool
    heading: Heading
    current_landmarks: tuple[str, ...]
    adjacent_landmarks: tuple[str, ...]
    checkpoint_complete: bool
    last_result: str


def reset(task: MazeTask, heading: Heading = Heading.EAST) -> MazeState:
    return MazeState(position=task.start, heading=heading, path=(task.start,))


def observe(task: MazeTask, state: MazeState) -> Observation:
    landmarks: list[str] = []
    if state.position == task.checkpoint:
        landmarks.append("blue_checkpoint")
    if state.position == task.exit:
        landmarks.append("exit")
    if state.position == task.forbidden:
        landmarks.append("red_forbidden")
    adjacent: list[str] = []
    for label, target in (("blue_checkpoint", task.checkpoint), ("exit", task.exit), ("red_forbidden", task.forbidden)):
        if target in {neighbor for _, neighbor in task.layout.accessible_neighbors(state.position)}:
            adjacent.append(label)
    return Observation(
        front_open=task.layout.can_move(state.position, state.heading),
        left_open=task.layout.can_move(state.position, state.heading.left()),
        right_open=task.layout.can_move(state.position, state.heading.right()),
        rear_open=task.layout.can_move(state.position, state.heading.opposite()),
        heading=state.heading,
        current_landmarks=tuple(landmarks),
        adjacent_landmarks=tuple(adjacent),
        checkpoint_complete=state.checkpoint_complete,
        last_result=state.last_result,
    )


def step(task: MazeTask, state: MazeState, action: Action) -> MazeState:
    """Apply one high-level action; a successful episode requires an explicit STOP."""
    if state.terminated:
        raise RuntimeError("cannot step a terminated episode")
    common = {"steps": state.steps + 1, "last_action": action}
    if action is Action.TURN_LEFT:
        return replace(state, heading=state.heading.left(), last_result="turned_left", **common)
    if action is Action.TURN_RIGHT:
        return replace(state, heading=state.heading.right(), last_result="turned_right", **common)
    if action is Action.STOP:
        succeeded = state.position == task.exit and state.checkpoint_complete
        return replace(
            state,
            terminated=True,
            success=succeeded,
            terminal_reason="completed" if succeeded else "stopped_early",
            last_result="completed" if succeeded else "stopped_early",
            **common,
        )
    direction = state.heading if action is Action.MOVE_FORWARD else state.heading.opposite()
    if not task.layout.can_move(state.position, direction):
        return replace(
            state,
            collisions=state.collisions + 1,
            last_result="wall_collision",
            **common,
        )
    destination = task.layout.neighbor(state.position, direction)
    if destination == task.forbidden:
        return replace(
            state,
            position=destination,
            terminated=True,
            terminal_reason="forbidden_cell",
            last_result="forbidden_cell",
            path=state.path + (destination,),
            **common,
        )
    checkpoint_complete = state.checkpoint_complete or destination == task.checkpoint
    result = "checkpoint_reached" if destination == task.checkpoint else "moved"
    if destination == task.exit:
        result = "exit_ready" if checkpoint_complete else "exit_before_checkpoint"
    return replace(
        state,
        position=destination,
        checkpoint_complete=checkpoint_complete,
        path=state.path + (destination,),
        last_result=result,
        **common,
    )
