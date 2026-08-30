"""Generate causal-memory recovery examples from deliberately perturbed train-only states."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from maze_agent import TopologicalMemory, build_task, observe, oracle_next_action, step
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


def legal_nonterminal_actions(task, state) -> list[Action]:
    actions = [Action.TURN_LEFT, Action.TURN_RIGHT]
    if task.layout.can_move(state.position, state.heading):
        actions.append(Action.MOVE_FORWARD)
    if task.layout.can_move(state.position, state.heading.opposite()):
        actions.append(Action.BACKTRACK)
    return actions


def summary(action: Action) -> str:
    if action is Action.STOP:
        return "The checkpoint is complete and the exit has been reached."
    if action in (Action.TURN_LEFT, Action.TURN_RIGHT):
        return "Turn toward the shortest recovery route using executed-history memory."
    return "Move along the shortest recovery route while avoiding the forbidden cell."


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate train-only off-expert recovery SFT states.")
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "artifacts" / "maze" / "splits_v1.json")
    parser.add_argument("--mazes", type=int, default=200)
    parser.add_argument("--rollouts-per-maze", type=int, default=24)
    parser.add_argument("--max-prefix-actions", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument(
        "--output", type=Path, default=PROJECT_ROOT / "assets" / "datasets" / "maze_sft_recovery_smoke_v1.jsonl"
    )
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    seeds = manifest["train_seeds"][: args.mazes]
    rng = random.Random(args.seed)
    rows: list[dict] = []
    for maze_seed in seeds:
        task = build_task(manifest["width"], manifest["height"], maze_seed)
        for rollout_index in range(args.rollouts_per_maze):
            state = reset(task)
            memory = TopologicalMemory()
            prefix_length = rng.randint(1, args.max_prefix_actions)
            for _ in range(prefix_length):
                memory.record_observation(task, state)
                action = rng.choice(legal_nonterminal_actions(task, state))
                after = step(task, state, action)
                memory.record_transition(task, state, action, after)
                state = after
                if state.terminated:
                    break
            if state.terminated:
                continue
            memory.record_observation(task, state)
            local = observe(task, state)
            target = oracle_next_action(task, state)
            rows.append(
                {
                    "dataset_version": "maze-sft-recovery-smoke-v1",
                    "task_id": task.task_id,
                    "maze_seed": maze_seed,
                    "source_split": "train",
                    "rollout_index": rollout_index,
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
                    "target": {"action": target.value, "decision_summary": summary(target)},
                    "expert": "astar_global_map_oracle",
                    "recovery_policy": "random_legal_prefix_then_oracle_next_action",
                }
            )
    write_jsonl_atomic(args.output, rows)
    print(json.dumps({"output": str(args.output), "rows": len(rows), "mazes": len(seeds)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
