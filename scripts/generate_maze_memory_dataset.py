"""Generate SFT examples with only causal, executed-history external memory."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from maze_agent import TopologicalMemory, astar_plan, build_task, observe, step
from maze_agent.core import Action, reset


def write_jsonl_atomic(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def summary(action: Action) -> str:
    if action is Action.STOP:
        return "The checkpoint is complete and the exit has been reached."
    if action in (Action.TURN_LEFT, Action.TURN_RIGHT):
        return "Align with the next route segment using the executed-history memory."
    return "Advance through the selected open passage without revisiting a known dead end."


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a causal-memory SFT smoke dataset from train-only maze seeds.")
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "artifacts" / "maze" / "splits_v1.json")
    parser.add_argument("--mazes", type=int, default=200)
    parser.add_argument(
        "--output", type=Path, default=PROJECT_ROOT / "assets" / "datasets" / "maze_sft_memory_smoke_v1.jsonl"
    )
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    seeds = manifest["train_seeds"][: args.mazes]
    if len(seeds) != args.mazes:
        raise ValueError("requested more mazes than the train split contains")

    rows: list[dict] = []
    for maze_seed in seeds:
        task = build_task(manifest["width"], manifest["height"], maze_seed)
        state = reset(task)
        memory = TopologicalMemory()
        for decision_index, action in enumerate(astar_plan(task)):
            memory.record_observation(task, state)
            local = observe(task, state)
            rows.append(
                {
                    "dataset_version": "maze-sft-memory-smoke-v1",
                    "task_id": task.task_id,
                    "maze_seed": maze_seed,
                    "source_split": "train",
                    "decision_index": decision_index,
                    "input": {
                        "instruction": task.instruction,
                        "local_perception": {
                            "front_open": local.front_open,
                            "left_open": local.left_open,
                            "right_open": local.right_open,
                            "rear_open": local.rear_open,
                            "current_landmarks": list(local.current_landmarks),
                            "adjacent_landmarks": list(local.adjacent_landmarks),
                        },
                        "state": {
                            "heading": state.heading.value,
                            "checkpoint_complete": state.checkpoint_complete,
                            "last_result": state.last_result,
                        },
                        "memory": memory.compact_summary(state),
                    },
                    "target": {"action": action.value, "decision_summary": summary(action)},
                    "expert": "astar_global_map_oracle",
                    "memory_policy": "causal_executed_history_only",
                }
            )
            after = step(task, state, action)
            memory.record_transition(task, state, action, after)
            state = after
        if not state.success:
            raise RuntimeError(f"expert did not complete {task.task_id}")
    write_jsonl_atomic(args.output, rows)
    print(json.dumps({"output": str(args.output), "mazes": len(seeds), "rows": len(rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
