"""Fetch a pinned, license-recorded Qwen baseline into the project worktree."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import model_info, snapshot_download


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch a pinned Qwen model into D: project storage.")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--model-dir", type=Path, default=PROJECT_ROOT / "models" / "qwen2_5_1_5b_instruct")
    parser.add_argument("--cache-dir", type=Path, default=PROJECT_ROOT / "cache" / "huggingface")
    parser.add_argument(
        "--manifest-output", type=Path, default=PROJECT_ROOT / "artifacts" / "models" / "qwen2_5_1_5b_instruct_v1.json"
    )
    parser.add_argument("--artifact-version", default="qwen2_5_1_5b_instruct_v1")
    args = parser.parse_args()
    args.model_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    info = model_info(args.model_id)
    revision = info.sha
    snapshot_download(
        repo_id=args.model_id,
        revision=revision,
        local_dir=args.model_dir,
        cache_dir=args.cache_dir,
        allow_patterns=["*.json", "*.safetensors", "tokenizer*", "vocab.*", "merges.txt", "LICENSE", "README.md"],
    )
    files = sorted(path for path in args.model_dir.rglob("*") if path.is_file() and ".cache" not in path.parts)
    payload = {
        "artifact_version": args.artifact_version,
        "model_id": args.model_id,
        "revision": revision,
        "license": "Apache-2.0",
        "license_url": f"https://huggingface.co/{args.model_id}/blob/main/LICENSE",
        "model_card_url": f"https://huggingface.co/{args.model_id}",
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model_dir": str(args.model_dir.resolve()),
        "files": [
            {
                "path": str(path.relative_to(args.model_dir)).replace("\\", "/"),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in files
        ],
    }
    payload["total_bytes"] = sum(item["bytes"] for item in payload["files"])
    write_json_atomic(args.manifest_output, payload)
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
