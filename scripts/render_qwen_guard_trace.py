"""Render a truth-labelled replay from actual Qwen plus memory-guard events."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from maze_agent import build_task
from maze_agent.core import Heading


def write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def draw_wrapped(draw, text: str, xy: tuple[int, int], max_width: int, font, fill, line_gap: int = 5) -> int:
    x, y = xy
    tokens = re.findall(r"[A-Za-z0-9_+.-]+|[^\sA-Za-z0-9_+.-]", text)
    line = ""
    prior_word = False
    for token in tokens:
        word = token[0].isascii() and token[0].isalnum()
        separator = " " if line and prior_word and word else ""
        candidate = line + separator + token
        if line and draw.textlength(candidate, font=font) > max_width:
            draw.text((x, y), line, font=font, fill=fill)
            y += font.size + line_gap
            line = token
        else:
            line = candidate
        prior_word = word
    if line:
        draw.text((x, y), line, font=font, fill=fill)
        y += font.size + line_gap
    return y


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a Qwen3.5 local-memory-guard development replay.")
    parser.add_argument(
        "--report", type=Path, default=PROJECT_ROOT / "artifacts/maze/eval_qwen35_closedloop_memory_guard_dev1_v2.json"
    )
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--frames-per-decision", type=int, default=3)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "artifacts/video/qwen35_memory_guard_replay_v1.mp4")
    parser.add_argument("--metadata-output", type=Path, default=PROJECT_ROOT / "artifacts/video/qwen35_memory_guard_replay_v1.json")
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    episode = report["episodes_detail"][args.episode_index]
    events = episode["events"]
    if not events or args.fps <= 0 or args.frames_per_decision <= 0:
        raise ValueError("replay needs events and positive timing")

    import cv2
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    task = build_task(9, 9, episode["maze_seed"])
    font_path = Path(r"C:\Windows\Fonts\msyh.ttc")
    if not font_path.is_file():
        raise FileNotFoundError(font_path)
    title_font = ImageFont.truetype(str(font_path), 25)
    body_font = ImageFont.truetype(str(font_path), 19)
    small_font = ImageFont.truetype(str(font_path), 15)
    width, height, scene_width = 1280, 720, 850
    cell, origin_x, origin_y = 64, 62, 125
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError("OpenCV could not open output")
    positions = [tuple(events[0]["before"]["position"])]
    guard_count = 0
    try:
        for index, event in enumerate(events, start=1):
            positions.append(tuple(event["after"]["position"]))
            guard_reason = event["planner"].get("guard_reason")
            guard_count += int(bool(guard_reason))
            image = Image.new("RGB", (width, height), (18, 23, 30))
            draw = ImageDraw.Draw(image)
            draw.rectangle((0, 0, scene_width, height), fill=(31, 38, 47))
            draw.rectangle((scene_width, 0, width, height), fill=(20, 24, 31))
            draw.text((30, 20), "LLM 迷宫回放：Qwen3.5 + 本地记忆执行守卫", font=title_font, fill=(240, 244, 250))
            draw.text((30, 56), "真实开发日志回放 · 非物理 H1 录像 · 非封存最终集", font=small_font, fill=(248, 181, 72))

            def center(pos: tuple[int, int]) -> tuple[int, int]:
                return origin_x + pos[0] * cell + cell // 2, origin_y + pos[1] * cell + cell // 2

            for position in task.layout.cells:
                x, y = origin_x + position[0] * cell, origin_y + position[1] * cell
                draw.rectangle((x, y, x + cell, y + cell), fill=(41, 49, 60))
                if not task.layout.can_move(position, Heading.NORTH):
                    draw.line((x, y, x + cell, y), fill=(219, 227, 235), width=3)
                if not task.layout.can_move(position, Heading.WEST):
                    draw.line((x, y, x, y + cell), fill=(219, 227, 235), width=3)
                if position[0] == task.layout.width - 1 and not task.layout.can_move(position, Heading.EAST):
                    draw.line((x + cell, y, x + cell, y + cell), fill=(219, 227, 235), width=3)
                if position[1] == task.layout.height - 1 and not task.layout.can_move(position, Heading.SOUTH):
                    draw.line((x, y + cell, x + cell, y + cell), fill=(219, 227, 235), width=3)
            for prior, current in zip(positions, positions[1:]):
                draw.line((*center(prior), *center(current)), fill=(105, 202, 229), width=5)
            for position, color, label in ((task.checkpoint, (54, 173, 255), "蓝"), (task.forbidden, (245, 91, 97), "禁"), (task.exit, (71, 208, 136), "终")):
                cx, cy = center(position)
                draw.ellipse((cx - 12, cy - 12, cx + 12, cy + 12), fill=color)
                draw.text((cx - 8, cy - 10), label, font=small_font, fill=(15, 20, 24))
            cx, cy = center(tuple(event["after"]["position"]))
            draw.ellipse((cx - 15, cy - 15, cx + 15, cy + 15), fill=(244, 244, 246), outline=(20, 20, 20), width=2)
            draw.text((cx - 10, cy - 10), "A", font=small_font, fill=(20, 20, 20))

            px = scene_width + 26
            draw.text((px, 26), "开发集回放", font=body_font, fill=(125, 190, 255))
            draw.text((px, 61), f"决策 {index} / {len(events)}", font=small_font, fill=(222, 228, 236))
            draw.text((px, 95), f"已记录守卫覆盖 {guard_count}", font=small_font, fill=(247, 181, 72))
            y = 145
            draw.text((px, y), "模型提议", font=body_font, fill=(125, 190, 255)); y += 31
            draw.text((px, y), event["planner"].get("proposed_action", event["planner"]["action"]), font=body_font, fill=(238, 242, 247)); y += 48
            draw.text((px, y), "实际执行", font=body_font, fill=(125, 190, 255)); y += 31
            execution_color = (85, 219, 157) if not guard_reason else (249, 179, 74)
            draw.text((px, y), event["planner"]["action"], font=body_font, fill=execution_color); y += 48
            draw.text((px, y), "守卫原因", font=body_font, fill=(125, 190, 255)); y += 31
            y = draw_wrapped(draw, guard_reason or "未覆盖：直接执行模型动作", (px, y), width - px - 26, small_font, execution_color)
            y += 20
            draw.text((px, y), "公开决策摘要", font=body_font, fill=(125, 190, 255)); y += 30
            y = draw_wrapped(draw, event["planner"]["decision_summary"], (px, y), width - px - 26, small_font, (232, 238, 245))
            y += 18
            draw.text((px, y), f"物理接口前的网格结果：{event['after']['result']}", font=small_font, fill=(175, 188, 201))
            frame = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
            for _ in range(args.frames_per_decision):
                writer.write(frame)
    finally:
        writer.release()
    metadata = {
        "asset_type": "development_grid_replay",
        "truth_label": "Qwen3.5 plus local-memory execution-guard development grid replay; not physical H1 footage or final evaluation",
        "source_report": str(args.report.resolve()),
        "maze_seed": episode["maze_seed"],
        "events": len(events),
        "guard_overrides": episode["guard_overrides"],
        "success": episode["success"],
        "output": str(args.output.resolve()),
        "fps": args.fps,
        "duration_seconds": len(events) * args.frames_per_decision / args.fps,
    }
    write_json_atomic(args.metadata_output, metadata)
    print(json.dumps(metadata, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
