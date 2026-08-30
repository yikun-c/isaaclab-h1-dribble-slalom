"""Evaluate classical maze baselines with sealed split awareness."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from maze_agent import astar_plan, build_task, dfs_plan, right_hand_plan, run_actions


def write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate A*, DFS and right-hand-rule maze baselines.")
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "artifacts" / "maze" / "splits_v1.json")
    parser.add_argument("--split", choices=("development", "iid_final", "ood_final"), default="development")
    parser.add_argument("--width", type=int, default=9)
    parser.add_argument("--height", type=int, default=9)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "artifacts" / "maze" / "baseline_development_v1.json")
    args = parser.parse_args()

    payload = json.loads(args.manifest.read_text(encoding="utf-8"))
    seeds = payload[f"{args.split}_seeds"]
    if args.limit is not None:
        seeds = seeds[: args.limit]
    planners = {
        "astar_global_oracle": astar_plan,
        "dfs_memory_baseline": dfs_plan,
        "right_hand_rule": right_hand_plan,
    }
    results: dict[str, dict] = {}
    for name, planner in planners.items():
        successes = 0
        collisions: list[int] = []
        efficiencies: list[float] = []
        terminal_reasons: dict[str, int] = {}
        for seed in seeds:
            task = build_task(args.width, args.height, seed)
            run = run_actions(task, planner(task))
            state = run.final_state
            successes += int(state.success)
            collisions.append(state.collisions)
            terminal_reasons[state.terminal_reason or "unterminated"] = terminal_reasons.get(
                state.terminal_reason or "unterminated", 0
            ) + 1
            if state.success:
                optimal_moves = len(task.optimal_route) - 1
                actual_moves = max(1, len(state.path) - 1)
                efficiencies.append(optimal_moves / actual_moves)
        results[name] = {
            "episodes": len(seeds),
            "successes": successes,
            "success_rate": successes / len(seeds),
            "mean_collisions": statistics.fmean(collisions),
            "mean_path_efficiency_success_only": statistics.fmean(efficiencies) if efficiencies else None,
            "terminal_reasons": terminal_reasons,
        }
    report = {
        "evaluation_version": "maze-baseline-v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "manifest": str(args.manifest.resolve()),
        "split": args.split,
        "width": args.width,
        "height": args.height,
        "episodes": len(seeds),
        "results": results,
        "notes": {
            "astar_global_oracle": "Uses the complete map and is an upper bound, not a fair partial-observation comparison.",
            "dfs_memory_baseline": "Uses deterministic graph exploration with explicit visited-state memory.",
            "right_hand_rule": "Reactive baseline without semantic planning or route memory.",
        },
    }
    write_json_atomic(args.output, report)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
