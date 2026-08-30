"""Write a synchronized planner trace suitable for evaluation and video overlays."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from maze_agent import (
    TopologicalMemory,
    astar_plan,
    build_task,
    decision_event,
    parse_planner_response,
    step,
)
from maze_agent.core import Action, reset


def write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def write_jsonl_atomic(path: Path, events: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def public_summary(action: Action) -> str:
    if action is Action.STOP:
        return "The required checkpoint is complete and the exit is reached."
    if action in (Action.TURN_LEFT, Action.TURN_RIGHT):
        return "Align with the next planned route segment."
    return "Advance through the selected open passage."


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay a planner trace using the strict public tool protocol.")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--width", type=int, default=9)
    parser.add_argument("--height", type=int, default=9)
    parser.add_argument(
        "--trace-output",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "maze" / "replays" / "astar_seed2026_v1.jsonl",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "maze" / "replays" / "astar_seed2026_v1.summary.json",
    )
    args = parser.parse_args()

    task = build_task(args.width, args.height, args.seed)
    state = reset(task)
    memory = TopologicalMemory()
    events: list[dict] = []
    invalid_outputs = 0
    for action in astar_plan(task):
        memory.record_observation(task, state)
        memory_before = memory.compact_summary(state)
        raw_response = json.dumps(
            {"action": action.value, "decision_summary": public_summary(action)}, ensure_ascii=False
        )
        decision = parse_planner_response(raw_response)
        invalid_outputs += int(not decision.valid)
        after = step(task, state, decision.action)
        events.append(
            decision_event(
                task,
                state,
                decision,
                after,
                memory_before,
                raw_response,
                latency_ms=0.0,
                token_count=0,
            )
        )
        memory.record_transition(task, state, decision.action, after)
        state = after
        if state.terminated:
            break
    if not state.success:
        raise RuntimeError(f"replay did not complete: {state.terminal_reason}")
    write_jsonl_atomic(args.trace_output, events)
    summary = {
        "trace_version": "maze-planner-trace-v1",
        "task_id": task.task_id,
        "seed": args.seed,
        "trace_output": str(args.trace_output.resolve()),
        "decisions": len(events),
        "invalid_outputs": invalid_outputs,
        "collisions": state.collisions,
        "success": state.success,
        "memory_nodes": len(memory.nodes),
    }
    write_json_atomic(args.summary_output, summary)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
