"""Compose the versioned, evidence-first final H1 maze film and subtitles."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VOICEOVER = PROJECT_ROOT / "assets/video/final_film_v1_voiceover.json"
SOURCES = (
    PROJECT_ROOT / "artifacts/video/qwen35_h1_physical_bridge_camera_v2.mp4",
    PROJECT_ROOT / "artifacts/video/h1_bridge_evidence_v3.mp4",
    PROJECT_ROOT / "artifacts/video/final_h1_bridge_evidence_card_v1.mp4",
    PROJECT_ROOT / "artifacts/video/final_baseline_evidence_card_v1.mp4",
    PROJECT_ROOT / "artifacts/video/baseline_comparison_v1.mp4",
    PROJECT_ROOT / "artifacts/video/llm_training_evidence_v4.mp4",
    PROJECT_ROOT / "artifacts/video/maze_trace_overlay_prototype_v2.mp4",
    PROJECT_ROOT / "artifacts/video/qwen35_memory_guard_replay_v1.mp4",
    PROJECT_ROOT / "artifacts/video/live_h1_success_trace_visualization_v1.mp4",
)
OUTPUT = PROJECT_ROOT / "artifacts/video/llm_h1_maze_final_film_v1.mp4"
METADATA = PROJECT_ROOT / "artifacts/video/llm_h1_maze_final_film_v1.json"


def duration(path: Path) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", str(path)],
        check=True,
        text=True,
        capture_output=True,
    )
    return float(json.loads(result.stdout)["format"]["duration"])


def timestamp(seconds: float) -> str:
    total_ms = round(seconds * 1000)
    hours, rem = divmod(total_ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, millis = divmod(rem, 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"


def subtitle_chunks(text: str, start: float, end: float) -> list[tuple[float, float, str]]:
    parts: list[str] = []
    current = ""
    for char in text:
        current += char
        if char in "。；！？" or len(current) >= 28:
            parts.append(current.strip())
            current = ""
    if current.strip():
        parts.append(current.strip())
    weight = sum(max(1, len(item)) for item in parts)
    cursor = start
    result = []
    for index, item in enumerate(parts):
        if index == len(parts) - 1:
            nxt = end
        else:
            nxt = cursor + (end - start) * max(1, len(item)) / weight
        result.append((cursor, nxt, item))
        cursor = nxt
    return result


def main() -> None:
    entries = json.loads(VOICEOVER.read_text(encoding="utf-8"))
    if len(entries) != len(SOURCES):
        raise ValueError("voiceover entries must exactly match final source clips")
    missing = [str(path) for path in SOURCES if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing video sources: " + ", ".join(missing))
    audio_dir = PROJECT_ROOT / "artifacts/audio/final_film_v1"
    audio_paths = [audio_dir / f"{index:02d}_{entry['id']}.mp3" for index, entry in enumerate(entries, start=1)]
    missing_audio = [str(path) for path in audio_paths if not path.is_file()]
    if missing_audio:
        raise FileNotFoundError("missing voiceover files: " + ", ".join(missing_audio))

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temp_video = OUTPUT.with_suffix(".visual.mp4")
    temp_audio = OUTPUT.with_suffix(".audio.mp3")
    subtitle = OUTPUT.with_suffix(".srt")
    audio_list = OUTPUT.with_suffix(".audio.txt")
    try:
        inputs: list[str] = []
        filters: list[str] = []
        for index, source in enumerate(SOURCES):
            inputs += ["-i", str(source)]
            filters.append(f"[{index}:v]fps=30,scale=1280:720:flags=lanczos,setsar=1,setpts=PTS-STARTPTS[v{index}]")
        joined = "".join(f"[v{index}]" for index in range(len(SOURCES)))
        filters.append(f"{joined}concat=n={len(SOURCES)}:v=1:a=0,format=yuv420p[v]")
        subprocess.run(
            ["ffmpeg", "-y", *inputs, "-filter_complex", ";".join(filters), "-map", "[v]", "-c:v", "libx264", "-crf", "18", "-preset", "medium", "-movflags", "+faststart", str(temp_video)],
            check=True,
        )
        audio_list.write_text("".join(f"file '{path.resolve().as_posix()}'\n" for path in audio_paths), encoding="utf-8")
        subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(audio_list), "-c:a", "libmp3lame", "-b:a", "128k", str(temp_audio)], check=True)
        cursor = 0.0
        lines: list[str] = []
        subtitle_index = 1
        source_durations = []
        for entry, source in zip(entries, SOURCES):
            clip_duration = duration(source)
            source_durations.append(clip_duration)
            for start, end, text in subtitle_chunks(entry["text"], cursor, cursor + clip_duration):
                lines.extend((str(subtitle_index), f"{timestamp(start)} --> {timestamp(end)}", text, ""))
                subtitle_index += 1
            cursor += clip_duration
        subtitle.write_text("\n".join(lines), encoding="utf-8")
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(temp_video), "-i", str(temp_audio), "-i", str(subtitle), "-map", "0:v:0", "-map", "1:a:0", "-map", "2:0", "-c:v", "copy", "-c:a", "aac", "-b:a", "160k", "-c:s", "mov_text", "-movflags", "+faststart", str(OUTPUT)],
            check=True,
        )
        metadata = {
            "asset_type": "final_evidence_film",
            "truth_label": "Evidence-first film. Physical-camera clips are labelled separately from report-derived visualizations. Final route visualization derives from the accepted live H1/Qwen/local-memory-guard report and is not camera footage.",
            "sources": [str(path.resolve()) for path in SOURCES],
            "voiceover": str(VOICEOVER.resolve()),
            "subtitle": str(subtitle.resolve()),
            "output": str(OUTPUT.resolve()),
            "source_durations_seconds": source_durations,
            "duration_seconds": duration(OUTPUT),
        }
        METADATA.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(metadata, ensure_ascii=False, sort_keys=True))
    finally:
        for path in (temp_video, temp_audio, audio_list):
            if path.exists():
                path.unlink()


if __name__ == "__main__":
    main()
