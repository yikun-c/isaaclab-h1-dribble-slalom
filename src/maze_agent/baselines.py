"""Classical planning baselines using the exact same maze action contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .core import Action, CARDINAL_HEADINGS, GridPos, Heading, MazeState, MazeTask, reset, step


def heading_between(current: GridPos, destination: GridPos) -> Heading:
    dx, dy = destination[0] - current[0], destination[1] - current[1]
    for heading in CARDINAL_HEADINGS:
        if heading.delta == (dx, dy):
            return heading
    raise ValueError(f"positions are not adjacent: {current} -> {destination}")


def turn_actions(current: Heading, target: Heading) -> tuple[Action, ...]:
    if current is target:
        return ()
    if current.right() is target:
        return (Action.TURN_RIGHT,)
    if current.left() is target:
        return (Action.TURN_LEFT,)
    return (Action.TURN_RIGHT, Action.TURN_RIGHT)


def actions_for_route(task: MazeTask, route: Iterable[GridPos]) -> tuple[Action, ...]:
    positions = tuple(route)
    if not positions or positions[0] != task.start:
        raise ValueError("route must start at task.start")
    heading = Heading.EAST
    actions: list[Action] = []
    for current, destination in zip(positions, positions[1:]):
        target = heading_between(current, destination)
        turns = turn_actions(heading, target)
        actions.extend(turns)
        for turn in turns:
            heading = heading.left() if turn is Action.TURN_LEFT else heading.right()
        actions.append(Action.MOVE_FORWARD)
    actions.append(Action.STOP)
    return tuple(actions)


def astar_plan(task: MazeTask) -> tuple[Action, ...]:
    """Global-map oracle, deliberately labelled an upper bound in reports."""
    return actions_for_route(task, task.optimal_route)


def _dfs_path(task: MazeTask, start: GridPos, goal: GridPos) -> tuple[GridPos, ...] | None:
    """Record actual DFS exploration, including physical returns from failed branches."""
    visited: set[GridPos] = set()
    walked: list[GridPos] = [start]

    def visit(position: GridPos) -> bool:
        visited.add(position)
        if position == goal:
            return True
        for _, neighbor in task.layout.accessible_neighbors(position):
            if neighbor == task.forbidden or neighbor in visited:
                continue
            walked.append(neighbor)
            if visit(neighbor):
                return True
            walked.append(position)
        return False

    return tuple(walked) if visit(start) else None


def _dfs_route(task: MazeTask) -> tuple[GridPos, ...] | None:
    """Respect ordered semantic subgoals instead of treating exit as terminal too early."""
    first = _dfs_path(task, task.start, task.checkpoint)
    if first is None:
        return None
    second = _dfs_path(task, task.checkpoint, task.exit)
    if second is None:
        return None
    return first + second[1:]


def dfs_plan(task: MazeTask) -> tuple[Action, ...]:
    route = _dfs_route(task)
    if route is None:
        raise RuntimeError("DFS could not find a valid semantic route")
    return actions_for_route(task, route)


def right_hand_plan(task: MazeTask, max_actions: int | None = None) -> tuple[Action, ...]:
    """Reactive wall follower; it may fail, which is useful as a baseline result."""
    state = reset(task)
    budget = max_actions or task.layout.width * task.layout.height * 16
    actions: list[Action] = []
    for _ in range(budget):
        right = state.heading.right()
        if task.layout.can_move(state.position, right):
            actions.append(Action.TURN_RIGHT)
            state = step(task, state, Action.TURN_RIGHT)
            actions.append(Action.MOVE_FORWARD)
            state = step(task, state, Action.MOVE_FORWARD)
        elif task.layout.can_move(state.position, state.heading):
            actions.append(Action.MOVE_FORWARD)
            state = step(task, state, Action.MOVE_FORWARD)
        else:
            actions.append(Action.TURN_LEFT)
            state = step(task, state, Action.TURN_LEFT)
        if state.terminated:
            return tuple(actions)
        if state.position == task.exit and state.checkpoint_complete:
            actions.append(Action.STOP)
            return tuple(actions)
    return tuple(actions + [Action.STOP])


@dataclass(frozen=True)
class RunResult:
    final_state: MazeState
    actions: tuple[Action, ...]
    exhausted_budget: bool

    @property
    def path_efficiency(self) -> float:
        moved_steps = max(1, len(self.final_state.path) - 1)
        return moved_steps


def run_actions(task: MazeTask, actions: Iterable[Action], max_actions: int | None = None) -> RunResult:
    state = reset(task)
    consumed: list[Action] = []
    budget = max_actions if max_actions is not None else task.layout.width * task.layout.height * 32
    for action in actions:
        if len(consumed) >= budget:
            return RunResult(state, tuple(consumed), True)
        consumed.append(action)
        state = step(task, state, action)
        if state.terminated:
            return RunResult(state, tuple(consumed), False)
    return RunResult(state, tuple(consumed), not state.terminated)
