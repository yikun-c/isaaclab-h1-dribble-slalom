"""Compose a short, truth-labelled evidence cut from validated source clips."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCES = (
    PROJECT_ROOT / "artifacts/video/h1_bridge_evidence_v2.mp4",
    PROJECT_ROOT / "artifacts/video/llm_training_evidence_v3.mp4",
    PROJECT_ROOT / "artifacts/video/baseline_comparison_v1.mp4",
    PROJECT_ROOT / "artifacts/video/maze_trace_overlay_prototype_v2.mp4",
    PROJECT_ROOT / "artifacts/video/qwen35_memory_guard_replay_v1.mp4",
)
OUTPUT = PROJECT_ROOT / "artifacts/video/llm_h1_maze_evidence_cut_v1.mp4"
METADATA = PROJECT_ROOT / "artifacts/video/llm_h1_maze_evidence_cut_v1.json"


def write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    missing = [str(path) for path in SOURCES if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing sources: " + ", ".join(missing))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    inputs: list[str] = []
    filters: list[str] = []
    for index, path in enumerate(SOURCES):
        inputs.extend(("-i", str(path)))
        filters.append(f"[{index}:v]fps=30,scale=1280:720:flags=lanczos,setsar=1,setpts=PTS-STARTPTS[v{index}]")
    joined = "".join(f"[v{index}]" for index in range(len(SOURCES)))
    filters.append(f"{joined}concat=n={len(SOURCES)}:v=1:a=0,format=yuv420p[v]")
    command = ["ffmpeg", "-y", *inputs, "-filter_complex", ";".join(filters), "-map", "[v]", "-c:v", "libx264", "-crf", "18", "-preset", "medium", "-movflags", "+faststart", str(OUTPUT)]
    subprocess.run(command, check=True)
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration,size", "-show_entries", "stream=codec_name,width,height,r_frame_rate", "-of", "json", str(OUTPUT)],
        check=True,
        capture_output=True,
        text=True,
    )
    metadata = {
        "asset_type": "evidence_rough_cut",
        "truth_label": "Short evidence rough cut, not final film. Contains a labelled H1 bridge data card, training/ablation evidence, baseline comparison, an A* layout prototype, and a Qwen-plus-memory-guard development replay.",
        "sources": [str(path.resolve()) for path in SOURCES],
        "output": str(OUTPUT.resolve()),
        "ffprobe": json.loads(probe.stdout),
    }
    write_json_atomic(METADATA, metadata)
    print(json.dumps(metadata, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
