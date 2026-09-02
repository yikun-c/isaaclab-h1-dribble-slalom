"""Auditable local-memory executor guard for structured maze actions.

This is intentionally *not* an oracle: decisions use only the current local
openings, adjacent landmark direction, the executed path, and transitions
already stored in ``TopologicalMemory``.  It is a hybrid LLM-plus-executor
variant and callers must label it as such.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections import deque

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


def _frontier_recovery_action(task: MazeTask, state: MazeState, memory: TopologicalMemory) -> Action | None:
    """Route only across executed transitions to the nearest observed frontier."""
    current_id = node_id(state.position)
    adjacency: dict[str, list[tuple[str, Heading]]] = {}
    for origin, direction_raw, destination in memory.transitions:
        direction = Heading(direction_raw)
        adjacency.setdefault(origin, []).append((destination, direction))
        adjacency.setdefault(destination, []).append((origin, direction.opposite()))
    queue: deque[tuple[str, Heading | None]] = deque([(current_id, None)])
    visited = {current_id}
    while queue:
        node, first_heading = queue.popleft()
        record = memory.nodes[node]
        unseen = record.observed_safe_exits - record.explored_exits
        if unseen:
            heading = sorted((Heading(value) for value in unseen), key=lambda item: item.value)[0]
            return _toward_heading(state.heading, heading) if node == current_id else _toward_heading(state.heading, first_heading)
        for destination, heading in adjacency.get(node, []):
            if destination not in visited:
                visited.add(destination)
                queue.append((destination, heading if first_heading is None else first_heading))
    return None


def _known_landmark_recovery_action(
    state: MazeState, memory: TopologicalMemory, landmark: str
) -> Action | None:
    """Return the first executed edge on a route to an already observed landmark.

    This intentionally searches only the transition graph built by
    ``record_transition``.  In particular, it does not use ``task.exit`` as a
    coordinate or inspect the maze layout: the exit becomes routable only after
    the robot has physically visited and locally labelled it.
    """
    targets = {
        known_id
        for known_id, record in memory.nodes.items()
        if landmark in record.landmarks
    }
    current_id = node_id(state.position)
    if not targets or current_id in targets:
        return None
    adjacency: dict[str, list[tuple[str, Heading]]] = {}
    for origin, direction_raw, destination in memory.transitions:
        direction = Heading(direction_raw)
        adjacency.setdefault(origin, []).append((destination, direction))
        adjacency.setdefault(destination, []).append((origin, direction.opposite()))
    queue: deque[tuple[str, Heading | None]] = deque([(current_id, None)])
    visited = {current_id}
    while queue:
        node, first_heading = queue.popleft()
        if node in targets:
            assert first_heading is not None
            return _toward_heading(state.heading, first_heading)
        for destination, heading in adjacency.get(node, []):
            if destination not in visited:
                visited.add(destination)
                queue.append((destination, heading if first_heading is None else first_heading))
    return None


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
    if state.checkpoint_complete:
        # The exit may have been seen before the semantic checkpoint.  Once the
        # checkpoint is reached, prefer the shortest route across *executed*
        # edges rather than spending the remaining budget on unrelated frontier
        # discovery.  This is still a local-memory guard, not global planning.
        exit_route = _known_landmark_recovery_action(state, memory, "exit")
        if exit_route is not None:
            # This route is a higher-priority safety/goal constraint than the
            # revisit heuristic below.  Returning it even when Qwen proposed
            # the same primitive is important: otherwise the later revisit
            # recovery can replace a correct MOVE_FORWARD with a turn and
            # make the agent oscillate at the checkpoint.
            return GuardedAction(
                exit_route,
                exit_route is not proposed,
                "route_known_exit_after_checkpoint",
            )
    local = observe(task, state)
    blocked = (proposed is Action.MOVE_FORWARD and not local.front_open) or (
        proposed is Action.BACKTRACK and not local.rear_open
    )
    if proposed is Action.STOP:
        recovery = _frontier_recovery_action(task, state, memory)
        return GuardedAction(recovery or _fallback_exploration_action(task, state, memory), True, "frontier_recovery" if recovery else "prevent_early_stop")
    current = memory.nodes.get(node_id(state.position))
    visits = current.visits if current is not None else 0
    if blocked:
        recovery = _frontier_recovery_action(task, state, memory)
        return GuardedAction(recovery or _fallback_exploration_action(task, state, memory), True, "frontier_recovery" if recovery else "prevent_known_wall")
    if visits >= revisit_threshold:
        recovery = _frontier_recovery_action(task, state, memory)
        return GuardedAction(recovery or _fallback_exploration_action(task, state, memory), True, "frontier_recovery" if recovery else "revisit_exploration")
    return GuardedAction(proposed, False, None)
