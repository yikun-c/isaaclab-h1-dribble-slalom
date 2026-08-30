"""Mux duration-matched evidence-cut voiceover with the validated silent cut."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VIDEO = PROJECT_ROOT / "artifacts/video/llm_h1_maze_evidence_cut_v3.mp4"
VOICE_MANIFEST = PROJECT_ROOT / "artifacts/audio/evidence_cut_v2/manifest.json"
OUTPUT = PROJECT_ROOT / "artifacts/video/llm_h1_maze_evidence_cut_v5_voiceover.mp4"
METADATA = PROJECT_ROOT / "artifacts/video/llm_h1_maze_evidence_cut_v5_voiceover.json"


def main() -> None:
    manifest = json.loads(VOICE_MANIFEST.read_text(encoding="utf-8"))
    audio_paths = [Path(item["output"]) for item in manifest["entries"]]
    if not VIDEO.is_file() or any(not path.is_file() for path in audio_paths):
        raise FileNotFoundError("video or one of the voiceover segments is missing")
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", str(VIDEO)],
        check=True,
        capture_output=True,
        text=True,
    )
    duration = float(json.loads(probe.stdout)["format"]["duration"])
    command: list[str] = ["ffmpeg", "-y", "-i", str(VIDEO)]
    for path in audio_paths:
        command.extend(("-i", str(path)))
    audio_inputs = "".join(f"[{index}:a]" for index in range(1, len(audio_paths) + 1))
    audio_filter = f"{audio_inputs}concat=n={len(audio_paths)}:v=0:a=1,apad=pad_dur=1[a]"
    command.extend(("-filter_complex", audio_filter, "-map", "0:v:0", "-map", "[a]", "-c:v", "copy", "-c:a", "aac", "-b:a", "128k", "-t", f"{duration:.6f}", "-movflags", "+faststart", str(OUTPUT)))
    subprocess.run(command, check=True)
    final_probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration,size", "-show_entries", "stream=codec_name,codec_type,sample_rate", "-of", "json", str(OUTPUT)],
        check=True,
        capture_output=True,
        text=True,
    )
    METADATA.write_text(
        json.dumps(
            {
                "asset_type": "narrated_evidence_rough_cut",
                "truth_label": "Narrated short evidence rough cut, not final film. The narration explicitly distinguishes the H1 bridge, A* oracle layout, Qwen plus memory guard hybrid, and development-only evidence.",
                "video_source": str(VIDEO.resolve()),
                "voice_manifest": str(VOICE_MANIFEST.resolve()),
                "output": str(OUTPUT.resolve()),
                "ffprobe": json.loads(final_probe.stdout),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(METADATA.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
