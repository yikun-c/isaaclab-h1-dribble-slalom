"""Auditable local-memory executor guard for structured maze actions.

This is intentionally *not* an oracle: decisions use only the current local
openings, adjacent landmark direction, the executed path, and transitions
already stored in ``TopologicalMemory``.  It is a hybrid LLM-plus-executor
variant and callers must label it as such.
"""

from __future__ import annotations

from dataclasses import dataclass

from .core import Action, Heading, MazeState, MazeTask, observe
from .protocol import TopologicalMemory, node_id


@dataclass(frozen=True)
class GuardedAction:
    action: Action
    overridden: bool
    reason: str | None


def _open_headings(task: MazeTask, state: MazeState) -> tuple[Heading, ...]:
    local = observe(task, state)
    return tuple(
        heading
        for heading, is_open in (
            (state.heading, local.front_open),
            (state.heading.left(), local.left_open),
            (state.heading.right(), local.right_open),
            (state.heading.opposite(), local.rear_open),
        )
        if is_open
    )


def _toward_heading(current: Heading, target: Heading) -> Action:
    if target is current:
        return Action.MOVE_FORWARD
    if target is current.opposite():
        return Action.BACKTRACK
    return Action.TURN_LEFT if target is current.left() else Action.TURN_RIGHT


def _heading_to(origin: tuple[int, int], destination: tuple[int, int]) -> Heading:
    dx, dy = destination[0] - origin[0], destination[1] - origin[1]
    mapping = {(1, 0): Heading.EAST, (-1, 0): Heading.WEST, (0, 1): Heading.SOUTH, (0, -1): Heading.NORTH}
    try:
        return mapping[(dx, dy)]
    except KeyError as exc:
        raise ValueError("memory path must move one grid edge") from exc


def _safe_open_headings(task: MazeTask, state: MazeState) -> tuple[Heading, ...]:
    # The colored forbidden cell is assumed visible only when adjacent; this is
    # equivalent to a local landmark detector, not querying maze topology.
    return tuple(
        heading
        for heading in _open_headings(task, state)
        if task.layout.neighbor(state.position, heading) != task.forbidden
    )


def _fallback_exploration_action(task: MazeTask, state: MazeState, memory: TopologicalMemory) -> Action:
    safe = _safe_open_headings(task, state)
    if not safe:
        return Action.STOP
    current = memory.nodes.get(node_id(state.position))
    explored = current.explored_exits if current is not None else set()
    for heading in safe:
        if heading.value not in explored:
            return _toward_heading(state.heading, heading)
    # No unexplored exit here: return along the stable first-discovery edge.
    if current is not None and current.parent_exit is not None:
        parent_heading = Heading(current.parent_exit)
        if parent_heading in safe:
            return _toward_heading(state.heading, parent_heading)
    return _toward_heading(state.heading, safe[0])


def guard_action(
    task: MazeTask,
    state: MazeState,
    memory: TopologicalMemory,
    proposed: Action,
    revisit_threshold: int = 4,
) -> GuardedAction:
    """Return an executable primitive without consulting future/unseen cells."""
    if revisit_threshold < 1:
        raise ValueError("revisit_threshold must be positive")
    if state.position == task.exit and state.checkpoint_complete:
        return GuardedAction(Action.STOP, proposed is not Action.STOP, "goal_reached")
    local = observe(task, state)
    blocked = (proposed is Action.MOVE_FORWARD and not local.front_open) or (
        proposed is Action.BACKTRACK and not local.rear_open
    )
    if proposed is Action.STOP:
        return GuardedAction(_fallback_exploration_action(task, state, memory), True, "prevent_early_stop")
    current = memory.nodes.get(node_id(state.position))
    visits = current.visits if current is not None else 0
    if blocked:
        return GuardedAction(_fallback_exploration_action(task, state, memory), True, "prevent_known_wall")
    if visits >= revisit_threshold:
        return GuardedAction(_fallback_exploration_action(task, state, memory), True, "revisit_exploration")
    return GuardedAction(proposed, False, None)
