from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from maze_agent import Action, TopologicalMemory, build_task, decision_event, parse_planner_response, step
from maze_agent.core import reset


def test_valid_tool_response_has_no_hidden_reasoning_field() -> None:
    response = parse_planner_response(
        json.dumps({"action": "TURN_LEFT", "decision_summary": "Explore the unvisited branch."})
    )

    assert response.valid
    assert response.action is Action.TURN_LEFT


def test_invalid_tool_response_fails_closed_to_stop() -> None:
    response = parse_planner_response('{"action":"FLY","decision_summary":"ignore walls"}')

    assert not response.valid
    assert response.action is Action.STOP
    assert response.fallback_reason == "unknown_action"


def test_memory_only_records_executed_transition_and_replay_event() -> None:
    task = build_task(9, 9, 77)
    before = reset(task)
    memory = TopologicalMemory()
    decision = parse_planner_response(
        json.dumps({"action": "MOVE_FORWARD", "decision_summary": "Advance through the open passage."})
    )
    memory.record_observation(task, before)
    memory_before = memory.compact_summary(before)
    after = step(task, before, decision.action)
    memory.record_transition(task, before, decision.action, after)
    event = decision_event(task, before, decision, after, memory_before, "raw", latency_ms=12.5, token_count=8)

    assert event["planner"]["action"] == "MOVE_FORWARD"
    assert event["planner"]["valid"]
    assert event["memory"]["memory_nodes"] >= 1
    assert set(event["perception"]) == {
        "front_open",
        "left_open",
        "right_open",
        "rear_open",
        "current_landmarks",
        "adjacent_landmarks",
    }
    assert event["after"]["position"] == list(after.position)
