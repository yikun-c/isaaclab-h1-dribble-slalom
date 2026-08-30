"""Generate a machine-readable QC report for the narrated evidence rough cut."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VIDEO = PROJECT_ROOT / "artifacts/video/llm_h1_maze_evidence_cut_v6_voiceover_subs.mp4"
OUTPUT = PROJECT_ROOT / "artifacts/video/qc_llm_h1_evidence_cut_v6.json"


def run(command: list[str]) -> str:
    return subprocess.run(command, check=True, capture_output=True, text=True).stdout


def main() -> None:
    if not VIDEO.is_file():
        raise FileNotFoundError(VIDEO)
    probe = json.loads(
        run([
            "ffprobe", "-v", "error", "-show_entries", "format=duration,size", "-show_entries",
            "stream=index,codec_name,codec_type,width,height,r_frame_rate,sample_rate:stream_tags=language,title",
            "-of", "json", str(VIDEO),
        ])
    )
    black_log = subprocess.run(
        ["ffmpeg", "-v", "info", "-i", str(VIDEO), "-vf", "blackdetect=d=0.1:pix_th=0.02", "-an", "-f", "null", "NUL"],
        check=True,
        capture_output=True,
        text=True,
    ).stderr
    black_intervals = re.findall(r"black_start:([^ ]+) black_end:([^ ]+) black_duration:([^\s]+)", black_log)
    silence_log = subprocess.run(
        ["ffmpeg", "-v", "info", "-i", str(VIDEO), "-af", "silencedetect=n=-50dB:d=0.4", "-f", "null", "NUL"],
        check=True,
        capture_output=True,
        text=True,
    ).stderr
    silence_durations = [float(value) for value in re.findall(r"silence_duration: ([0-9.]+)", silence_log)]
    subtitle_file = OUTPUT.with_suffix(".subtitles.srt")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(VIDEO), "-map", "0:s:0", "-c:s", "srt", str(subtitle_file)], check=True)
    subtitles = subtitle_file.read_text(encoding="utf-8")
    subtitle_entries = len(re.findall(r"(?m)^\d+$", subtitles))
    duration = float(probe["format"]["duration"])
    report = {
        "asset": str(VIDEO.resolve()),
        "truth_label": "QC for a narrated evidence rough cut, not a final 8-12 minute film.",
        "duration_seconds": duration,
        "format_size_bytes": int(probe["format"]["size"]),
        "streams": probe["streams"],
        "black_intervals": black_intervals,
        "max_silence_seconds": max(silence_durations, default=0.0),
        "subtitle_entries": subtitle_entries,
        "rough_cut_acceptance": {
            "has_video_audio_subtitle": {stream["codec_type"] for stream in probe["streams"]} >= {"video", "audio", "subtitle"},
            "no_detected_black_interval": not black_intervals,
            "subtitle_stream_extractable": subtitle_entries == 5,
            "duration_match_expected": abs(duration - 121.267) < 0.1,
        },
        "final_film_acceptance": False,
        "known_final_gaps": [
            "Duration is about 2 minutes, below the planned 8-12 minute final film.",
            "No accepted long-horizon physical H1 maze navigation recording exists.",
            "Visible Isaac RTX camera recorder is blocked by native DLL startup failures.",
            "Final sealed IID/OOD and 50-episode Isaac suites have not been run.",
        ],
    }
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
