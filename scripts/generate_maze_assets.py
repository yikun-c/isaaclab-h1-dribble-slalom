"""Create versioned split manifests and A* SFT smoke data without touching final seeds."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from maze_agent import Action, astar_plan, build_split_manifest, build_task, observe, step
from maze_agent.core import MazeState, reset


def write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def write_jsonl_atomic(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def observation_payload(task, state: MazeState) -> dict:
    local = observe(task, state)
    return {
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
            "heading": local.heading.value,
            "checkpoint_complete": local.checkpoint_complete,
            "last_result": local.last_result,
        },
    }


def public_summary(state: MazeState, action: Action) -> str:
    if action is Action.STOP:
        return "The blue checkpoint is complete and the exit has been reached."
    if action in (Action.TURN_LEFT, Action.TURN_RIGHT):
        return "Align with the next expert route segment."
    return "Advance along the expert route while preserving the task constraints."


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate deterministic maze manifests and A* smoke demonstrations.")
    parser.add_argument("--master-seed", type=int, default=20260830)
    parser.add_argument("--width", type=int, default=9)
    parser.add_argument("--height", type=int, default=9)
    parser.add_argument("--train-count", type=int, default=2000)
    parser.add_argument("--development-count", type=int, default=200)
    parser.add_argument("--iid-final-count", type=int, default=500)
    parser.add_argument("--ood-final-count", type=int, default=500)
    parser.add_argument("--smoke-mazes", type=int, default=200)
    parser.add_argument("--manifest-output", type=Path, default=PROJECT_ROOT / "artifacts" / "maze" / "splits_v1.json")
    parser.add_argument("--dataset-output", type=Path, default=PROJECT_ROOT / "assets" / "datasets" / "maze_sft_smoke_v1.jsonl")
    args = parser.parse_args()

    manifest = build_split_manifest(
        args.master_seed,
        train_count=args.train_count,
        development_count=args.development_count,
        iid_final_count=args.iid_final_count,
        ood_final_count=args.ood_final_count,
    )
    if args.smoke_mazes > len(manifest.train_seeds):
        raise ValueError("smoke mazes cannot exceed train split size")
    manifest_payload = {
        "generator_version": manifest.generator_version,
        "master_seed": manifest.master_seed,
        "width": args.width,
        "height": args.height,
        "train_seeds": list(manifest.train_seeds),
        "development_seeds": list(manifest.development_seeds),
        "iid_final_seeds": list(manifest.iid_final_seeds),
        "ood_final_seeds": list(manifest.ood_final_seeds),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "policy": "Final split seeds are sealed and must not be loaded by dataset or training code.",
    }
    write_json_atomic(args.manifest_output, manifest_payload)

    rows: list[dict] = []
    completed_mazes = 0
    for seed in manifest.train_seeds:
        if completed_mazes >= args.smoke_mazes:
            break
        manifest.assert_trainable(seed)
        task = build_task(args.width, args.height, seed)
        state = reset(task)
        for decision_index, action in enumerate(astar_plan(task)):
            rows.append(
                {
                    "dataset_version": "maze-sft-smoke-v1",
                    "task_id": task.task_id,
                    "maze_seed": seed,
                    "source_split": "train",
                    "generator_version": manifest.generator_version,
                    "decision_index": decision_index,
                    "input": observation_payload(task, state),
                    "target": {
                        "action": action.value,
                        "decision_summary": public_summary(state, action),
                    },
                    "expert": "astar_global_map_oracle",
                    "optimal_route_cells": len(task.optimal_route),
                }
            )
            state = step(task, state, action)
        if not state.success:
            raise RuntimeError(f"expert did not complete {task.task_id}")
        completed_mazes += 1
    write_jsonl_atomic(args.dataset_output, rows)
    print(
        json.dumps(
            {
                "manifest": str(args.manifest_output),
                "dataset": str(args.dataset_output),
                "smoke_mazes": completed_mazes,
                "decision_rows": len(rows),
                "final_seed_count": len(manifest.sealed_final_seeds),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
