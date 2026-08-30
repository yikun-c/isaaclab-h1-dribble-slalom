"""Create a deterministic expert/recovery SFT mixture without touching final splits."""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl_atomic(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Deterministically merge expert and recovery maze SFT data.")
    parser.add_argument(
        "--expert", type=Path, default=PROJECT_ROOT / "assets" / "datasets" / "maze_sft_memory_smoke_v1.jsonl"
    )
    parser.add_argument(
        "--recovery", type=Path, default=PROJECT_ROOT / "assets" / "datasets" / "maze_sft_recovery_smoke_v1.jsonl"
    )
    parser.add_argument(
        "--output", type=Path, default=PROJECT_ROOT / "assets" / "datasets" / "maze_sft_memory_recovery_mix_v1.jsonl"
    )
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--recovery-repeat", type=int, default=1)
    args = parser.parse_args()
    if args.recovery_repeat <= 0:
        raise ValueError("recovery-repeat must be positive")
    expert_rows = read_jsonl(args.expert)
    recovery_rows = read_jsonl(args.recovery)
    rows = expert_rows + recovery_rows * args.recovery_repeat
    random.Random(args.seed).shuffle(rows)
    write_jsonl_atomic(args.output, rows)
    print(json.dumps({"output": str(args.output), "rows": len(rows), "expert_rows": len(expert_rows), "recovery_rows": len(recovery_rows) * args.recovery_repeat}, ensure_ascii=False))


if __name__ == "__main__":
    main()
