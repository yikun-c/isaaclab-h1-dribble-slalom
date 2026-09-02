"""Compose a silent, annotation-first ordering of the final evidence film."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCES = (
    PROJECT_ROOT / "artifacts/video/qwen35_h1_physical_bridge_camera_v2.mp4",
    PROJECT_ROOT / "artifacts/video/live_h1_success_trace_visualization_v1.mp4",
    PROJECT_ROOT / "artifacts/video/h1_bridge_evidence_v3.mp4",
    PROJECT_ROOT / "artifacts/video/final_h1_bridge_evidence_card_v1.mp4",
    PROJECT_ROOT / "artifacts/video/final_baseline_evidence_card_v1.mp4",
    PROJECT_ROOT / "artifacts/video/baseline_comparison_v1.mp4",
    PROJECT_ROOT / "artifacts/video/llm_training_evidence_v4.mp4",
    PROJECT_ROOT / "artifacts/video/maze_trace_overlay_prototype_v2.mp4",
    PROJECT_ROOT / "artifacts/video/qwen35_memory_guard_replay_v1.mp4",
)
ANNOTATIONS = PROJECT_ROOT / "assets/video/final_film_v2_annotations.srt"
OUTPUT = PROJECT_ROOT / "artifacts/video/llm_h1_maze_final_film_v2_silent.mp4"
METADATA = PROJECT_ROOT / "artifacts/video/llm_h1_maze_final_film_v2_silent.json"


def probe_duration(path: Path) -> float:
    result = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", str(path)], check=True, capture_output=True, text=True)
    return float(json.loads(result.stdout)["format"]["duration"])


def main() -> None:
    missing = [str(path) for path in (*SOURCES, ANNOTATIONS) if not path.is_file()]
    if missing:
        raise FileNotFoundError(", ".join(missing))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temp = OUTPUT.with_suffix(".visual.mp4")
    try:
        inputs: list[str] = []
        filters: list[str] = []
        for index, source in enumerate(SOURCES):
            inputs += ["-i", str(source)]
            filters.append(f"[{index}:v]fps=30,scale=1280:720:flags=lanczos,setsar=1,setpts=PTS-STARTPTS[v{index}]")
        filters.append("".join(f"[v{index}]" for index in range(len(SOURCES))) + f"concat=n={len(SOURCES)}:v=1:a=0,format=yuv420p[v]")
        subprocess.run(["ffmpeg", "-y", *inputs, "-filter_complex", ";".join(filters), "-map", "[v]", "-c:v", "libx264", "-crf", "18", "-preset", "medium", str(temp)], check=True)
        subtitle_path = ANNOTATIONS.resolve().as_posix().replace(":", "\\:")
        subtitle_filter = f"subtitles=filename='{subtitle_path}':force_style='FontName=Microsoft YaHei,FontSize=22,PrimaryColour=&H00FFFFFF,OutlineColour=&H80000000,BorderStyle=3,Outline=2,Shadow=0,MarginV=28,Alignment=2'"
        subprocess.run(["ffmpeg", "-y", "-i", str(temp), "-vf", subtitle_filter, "-map", "0:v:0", "-c:v", "libx264", "-crf", "18", "-preset", "medium", "-an", "-movflags", "+faststart", str(OUTPUT)], check=True)
        metadata = {
            "asset_type": "silent_annotation_final_film",
            "truth_label": "Silent annotation-first film. Physical camera footage and report-derived live H1 trace visualization are explicitly distinguished.",
            "sources": [str(path.resolve()) for path in SOURCES],
            "annotations": str(ANNOTATIONS.resolve()),
            "output": str(OUTPUT.resolve()),
            "duration_seconds": probe_duration(OUTPUT),
            "audio": "none",
        }
        METADATA.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(metadata, ensure_ascii=False))
    finally:
        if temp.exists():
            os.remove(temp)


if __name__ == "__main__":
    main()
