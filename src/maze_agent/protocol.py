"""Strict planner protocol, external map memory, and replayable decision records."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from .core import Action, Heading, MazeState, MazeTask, observe


@dataclass(frozen=True)
class PlannerDecision:
    action: Action
    decision_summary: str
    valid: bool
    fallback_reason: str | None = None


def parse_planner_response(raw: str) -> PlannerDecision:
    """Accept only a tiny public JSON protocol; invalid model output safely stops."""
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return PlannerDecision(Action.STOP, "Planner output was invalid; stopping safely.", False, "invalid_json")
    if not isinstance(payload, dict):
        return PlannerDecision(Action.STOP, "Planner output was invalid; stopping safely.", False, "not_object")
    if set(payload) != {"action", "decision_summary"}:
        return PlannerDecision(Action.STOP, "Planner output was invalid; stopping safely.", False, "schema_keys")
    action_raw = payload["action"]
    summary = payload["decision_summary"]
    if not isinstance(action_raw, str) or not isinstance(summary, str):
        return PlannerDecision(Action.STOP, "Planner output was invalid; stopping safely.", False, "schema_types")
    if not summary.strip() or len(summary) > 240:
        return PlannerDecision(Action.STOP, "Planner output was invalid; stopping safely.", False, "summary_length")
    try:
        action = Action(action_raw)
    except ValueError:
        return PlannerDecision(Action.STOP, "Planner output was invalid; stopping safely.", False, "unknown_action")
    return PlannerDecision(action, summary.strip(), True)


def node_id(position: tuple[int, int]) -> str:
    """Placeholder odometry key; Isaac integration will quantize estimated local pose similarly."""
    return f"N{position[0]}_{position[1]}"


@dataclass
class NodeMemory:
    visits: int = 0
    explored_exits: set[str] = field(default_factory=set)
    known_dead_end: bool = False
    landmarks: set[str] = field(default_factory=set)
    parent_exit: str | None = None


@dataclass
class TopologicalMemory:
    """Memory is built only from executed transitions and local landmark observations."""

    nodes: dict[str, NodeMemory] = field(default_factory=dict)
    transitions: list[tuple[str, str, str]] = field(default_factory=list)
    checkpoint_complete: bool = False

    def _node(self, position: tuple[int, int]) -> NodeMemory:
        return self.nodes.setdefault(node_id(position), NodeMemory())

    def record_observation(self, task: MazeTask, state: MazeState) -> None:
        current = self._node(state.position)
        current.visits += 1
        if state.position == task.checkpoint:
            current.landmarks.add("blue_checkpoint")
        if state.position == task.exit:
            current.landmarks.add("exit")
        self.checkpoint_complete = self.checkpoint_complete or state.checkpoint_complete
        exits = task.layout.accessible_neighbors(state.position)
        if len(exits) == 1 and state.position not in {task.start, task.exit, task.checkpoint}:
            current.known_dead_end = True

    def record_transition(self, task: MazeTask, before: MazeState, action: Action, after: MazeState) -> None:
        if before.position != after.position:
            direction = before.heading if action is Action.MOVE_FORWARD else before.heading.opposite()
            origin = node_id(before.position)
            destination = node_id(after.position)
            self._node(before.position).explored_exits.add(direction.value)
            # Arrival also proves the return edge at the destination; this is
            # executed-path evidence, not a hidden global-map lookup.
            destination_record = self._node(after.position)
            return_exit = direction.opposite().value
            destination_record.explored_exits.add(return_exit)
            # Keep the first discovery edge stable for DFS-style return. Using
            # the most recent state.path parent causes two-node oscillation.
            if destination_record.parent_exit is None:
                destination_record.parent_exit = return_exit
            self.transitions.append((origin, direction.value, destination))
        self.checkpoint_complete = self.checkpoint_complete or after.checkpoint_complete

    def compact_summary(self, state: MazeState) -> dict[str, Any]:
        known_dead_ends = sorted(node for node, record in self.nodes.items() if record.known_dead_end)
        current = self._node(state.position)
        return {
            "current_node": node_id(state.position),
            "current_visits": current.visits,
            "visited_exits": sorted(current.explored_exits),
            "known_dead_ends": known_dead_ends[-8:],
            "checkpoint_complete": self.checkpoint_complete,
            "memory_nodes": len(self.nodes),
        }


def decision_event(
    task: MazeTask,
    before: MazeState,
    decision: PlannerDecision,
    after: MazeState,
    memory_before: dict[str, Any],
    raw_response: str,
    latency_ms: float | None = None,
    token_count: int | None = None,
) -> dict[str, Any]:
    """Build one JSONL-safe event that can drive both evaluation and the video side panel."""
    local = observe(task, before)
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "task_id": task.task_id,
        "maze_seed": task.layout.seed,
        "before": {
            "position": list(before.position),
            "heading": before.heading.value,
            "checkpoint_complete": before.checkpoint_complete,
            "last_result": before.last_result,
        },
        "memory": memory_before,
        "perception": {
            "front_open": local.front_open,
            "left_open": local.left_open,
            "right_open": local.right_open,
            "rear_open": local.rear_open,
            "current_landmarks": list(local.current_landmarks),
            "adjacent_landmarks": list(local.adjacent_landmarks),
        },
        "planner": {
            "raw_response": raw_response,
            "action": decision.action.value,
            "decision_summary": decision.decision_summary,
            "valid": decision.valid,
            "fallback_reason": decision.fallback_reason,
            "latency_ms": latency_ms,
            "token_count": token_count,
        },
        "after": {
            "position": list(after.position),
            "heading": after.heading.value,
            "result": after.last_result,
            "terminated": after.terminated,
            "success": after.success,
        },
    }
