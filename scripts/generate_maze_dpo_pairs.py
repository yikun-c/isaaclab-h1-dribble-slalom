"""Generate auditable train-only DPO preference pairs and random-label controls."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from maze_agent import Action, Heading, MazeState, build_task, step


ACTION_ORDER = (Action.MOVE_FORWARD, Action.TURN_LEFT, Action.TURN_RIGHT, Action.BACKTRACK, Action.STOP)


def write_jsonl_atomic(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def state_from_row(row: dict) -> MazeState:
    payload = row["input"]["state"]
    node = row["input"]["memory"]["current_node"]
    x, y = node.removeprefix("N").split("_")
    return MazeState(
        position=(int(x), int(y)),
        heading=Heading(payload["heading"]),
        checkpoint_complete=bool(payload["checkpoint_complete"]),
        last_result=payload["last_result"],
        path=((int(x), int(y)),),
    )


def remaining_cost(task, state: MazeState) -> int:
    if state.terminated:
        return 0 if state.success else 10_000
    if state.checkpoint_complete:
        route = task.layout.shortest_path(state.position, task.exit, (task.forbidden,))
    else:
        first = task.layout.shortest_path(state.position, task.checkpoint, (task.forbidden,))
        second = task.layout.shortest_path(task.checkpoint, task.exit, (task.forbidden,))
        route = None if first is None or second is None else first + second[1:]
    return 10_000 if route is None else len(route) - 1


def choose_rejected(task, state: MazeState, chosen: Action) -> tuple[Action, int, int, str]:
    chosen_cost = 1 + remaining_cost(task, step(task, state, chosen))
    candidates: list[tuple[int, int, Action, str]] = []
    for order, action in enumerate(ACTION_ORDER):
        if action is chosen:
            continue
        after = step(task, state, action)
        cost = 1 + remaining_cost(task, after)
        kind = "continuing_suboptimal" if not after.terminated and after.collisions == state.collisions else "terminal_or_collision"
        candidates.append((cost, -order, action, kind))
    continuing = [candidate for candidate in candidates if candidate[3] == "continuing_suboptimal"]
    rejected_cost, _, rejected, kind = max(continuing or candidates)
    if rejected_cost < chosen_cost:
        raise RuntimeError("expert action unexpectedly has higher cost than every candidate")
    return rejected, chosen_cost, rejected_cost, kind


def output_payload(action: Action, summary: str) -> str:
    return json.dumps({"action": action.value, "decision_summary": summary}, ensure_ascii=False, sort_keys=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build verified DPO pairs from train-only A* memory examples.")
    parser.add_argument("--source", type=Path, default=PROJECT_ROOT / "assets/datasets/maze_sft_memory_smoke_v1.jsonl")
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "artifacts/maze/splits_v1.json")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "assets/datasets/maze_dpo_pairs_smoke_v1.jsonl")
    parser.add_argument("--random-control-output", type=Path, default=PROJECT_ROOT / "assets/datasets/maze_dpo_pairs_random_control_smoke_v1.jsonl")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    train_seeds = set(manifest["train_seeds"])
    source_rows = [json.loads(line) for line in args.source.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.limit is not None:
        source_rows = source_rows[: args.limit]
    pairs: list[dict] = []
    controls: list[dict] = []
    for row in source_rows:
        if row.get("source_split") != "train" or row["maze_seed"] not in train_seeds:
            raise PermissionError("DPO pair source must be train-only; sealed split leakage refused")
        task = build_task(manifest["width"], manifest["height"], row["maze_seed"])
        state = state_from_row(row)
        chosen_action = Action(row["target"]["action"])
        rejected_action, chosen_cost, rejected_cost, rejected_kind = choose_rejected(task, state, chosen_action)
        pair = {
            "dataset_version": "maze-dpo-pairs-smoke-v1",
            "task_id": row["task_id"],
            "maze_seed": row["maze_seed"],
            "source_split": "train",
            "decision_index": row["decision_index"],
            "prompt": row["input"],
            "chosen": output_payload(chosen_action, row["target"]["decision_summary"]),
            "rejected": output_payload(rejected_action, "A valid but non-expert alternative action."),
            "chosen_action": chosen_action.value,
            "rejected_action": rejected_action.value,
            "chosen_remaining_cost": chosen_cost,
            "rejected_remaining_cost": rejected_cost,
            "rejected_kind": rejected_kind,
            "preference_source": "A* expert one-step cost comparison on train-only layout",
        }
        pairs.append(pair)
        flip = int(hashlib.sha256(f"{row['task_id']}:{row['decision_index']}".encode()).hexdigest(), 16) % 2 == 0
        control = {**pair, "dataset_version": "maze-dpo-random-label-control-smoke-v1", "random_label_swapped": flip}
        if flip:
            control["chosen"], control["rejected"] = pair["rejected"], pair["chosen"]
            control["chosen_action"], control["rejected_action"] = pair["rejected_action"], pair["chosen_action"]
        controls.append(control)
    write_jsonl_atomic(args.output, pairs)
    write_jsonl_atomic(args.random_control_output, controls)
    summary = {
        "pairs": len(pairs),
        "output": str(args.output.resolve()),
        "random_control_output": str(args.random_control_output.resolve()),
        "mean_cost_gap": sum(pair["rejected_remaining_cost"] - pair["chosen_remaining_cost"] for pair in pairs) / max(1, len(pairs)),
        "random_control_swapped": sum(pair["random_label_swapped"] for pair in controls),
    }
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
