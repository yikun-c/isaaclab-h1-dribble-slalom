"""Generate duration-matched Chinese voiceover segments from versioned text."""

from __future__ import annotations

import json
import subprocess
import argparse
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def atempo_chain(factor: float) -> str:
    """Return ffmpeg atempo stages, each constrained to its supported range."""
    stages: list[float] = []
    while factor > 2.0:
        stages.append(2.0)
        factor /= 2.0
    while factor < 0.5:
        stages.append(0.5)
        factor /= 0.5
    stages.append(factor)
    return ",".join(f"atempo={stage:.8f}" for stage in stages)


def probe_duration(path: Path) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    return float(json.loads(result.stdout)["format"]["duration"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Render duration-matched evidence-cut voiceover.")
    parser.add_argument("--source", type=Path, default=PROJECT_ROOT / "assets/video/evidence_cut_v1_voiceover.json")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "artifacts/audio/evidence_cut_v1")
    args = parser.parse_args()
    entries = json.loads(args.source.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = args.output_dir / "manifest.json"
    manifest_entries: list[dict] = []
    for index, entry in enumerate(entries, start=1):
        raw = args.output_dir / f"{index:02d}_{entry['id']}.raw.mp3"
        output = args.output_dir / f"{index:02d}_{entry['id']}.mp3"
        subprocess.run(
            ["edge-tts", "--voice", "zh-CN-XiaoxiaoNeural", "--text", entry["text"], "--write-media", str(raw)],
            check=True,
        )
        raw_duration = probe_duration(raw)
        target = float(entry["target_seconds"])
        # atempo > 1 speeds up, so raw_duration/target is the required factor.
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(raw), "-filter:a", atempo_chain(raw_duration / target), "-ar", "24000", "-ac", "1", "-c:a", "libmp3lame", "-b:a", "96k", str(output)],
            check=True,
        )
        output_duration = probe_duration(output)
        manifest_entries.append({**entry, "raw": str(raw.resolve()), "raw_duration_seconds": raw_duration, "output": str(output.resolve()), "output_duration_seconds": output_duration})
    manifest.write_text(json.dumps({"voice": "zh-CN-XiaoxiaoNeural", "source": str(args.source.resolve()), "entries": manifest_entries}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"manifest": str(manifest.resolve()), "entries": manifest_entries}, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
