"""Attach versioned Chinese subtitles as a selectable MP4 subtitle stream."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT = PROJECT_ROOT / "artifacts/video/llm_h1_maze_evidence_cut_v2_voiceover.mp4"
SUBTITLES = PROJECT_ROOT / "assets/video/evidence_cut_v1_subtitles.srt"
OUTPUT = PROJECT_ROOT / "artifacts/video/llm_h1_maze_evidence_cut_v3_voiceover_subs.mp4"
METADATA = PROJECT_ROOT / "artifacts/video/llm_h1_maze_evidence_cut_v3_voiceover_subs.json"


def main() -> None:
    if not INPUT.is_file() or not SUBTITLES.is_file():
        raise FileNotFoundError("narrated video or subtitle source missing")
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(INPUT), "-i", str(SUBTITLES), "-map", "0:v:0", "-map", "0:a:0", "-map", "1:0", "-c:v", "copy", "-c:a", "copy", "-c:s", "mov_text", "-metadata:s:s:0", "language=chi", "-metadata:s:s:0", "title=简体中文", "-movflags", "+faststart", str(OUTPUT)],
        check=True,
    )
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration,size", "-show_entries", "stream=codec_name,codec_type:stream_tags=language,title", "-of", "json", str(OUTPUT)],
        check=True,
        capture_output=True,
        text=True,
    )
    METADATA.write_text(
        json.dumps(
            {"asset_type": "narrated_evidence_rough_cut_with_subtitles", "truth_label": "Short narrated evidence rough cut with selectable Chinese subtitles; not final film.", "input": str(INPUT.resolve()), "subtitle_source": str(SUBTITLES.resolve()), "output": str(OUTPUT.resolve()), "ffprobe": json.loads(probe.stdout)},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(METADATA.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
