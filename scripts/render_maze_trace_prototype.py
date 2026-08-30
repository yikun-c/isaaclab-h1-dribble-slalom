"""Render a labelled 2D trace prototype from real planner JSONL events.

This is a layout/QC asset only. It intentionally displays the source as an A* oracle
so it cannot be mistaken for an LLM or H1 final result.
"""

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
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def font(path: Path, size: int):
    from PIL import ImageFont

    return ImageFont.truetype(str(path), size=size)


def draw_wrapped(draw, text: str, xy: tuple[int, int], max_width: int, font_obj, fill, line_gap: int = 7) -> int:
    x, y = xy
    tokens = re.findall(r"[A-Za-z0-9_]+|[^\sA-Za-z0-9_]", text)
    line = ""
    previous_word = False
    for token in tokens:
        is_word = token[0].isascii() and token[0].isalnum()
        separator = " " if line and previous_word and is_word else ""
        candidate = line + separator + token
        if draw.textlength(candidate, font=font_obj) > max_width and line:
            draw.text((x, y), line, font=font_obj, fill=fill)
            y += font_obj.size + line_gap
            line = token
        else:
            line = candidate
        previous_word = is_word
    if line:
        draw.text((x, y), line, font=font_obj, fill=fill)
        y += font_obj.size + line_gap
    return y


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a truthful maze-trace information-panel prototype.")
    parser.add_argument(
        "--trace", type=Path, default=PROJECT_ROOT / "artifacts" / "maze" / "replays" / "astar_seed2026_v1.jsonl"
    )
    parser.add_argument("--width", type=int, default=9)
    parser.add_argument("--height", type=int, default=9)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--frames-per-decision", type=int, default=5)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--output", type=Path, default=PROJECT_ROOT / "artifacts" / "video" / "maze_trace_overlay_prototype_v1.mp4"
    )
    parser.add_argument(
        "--metadata-output",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "video" / "maze_trace_overlay_prototype_v1.json",
    )
    args = parser.parse_args()
    if args.fps <= 0 or args.frames_per_decision <= 0:
        raise ValueError("fps and frames-per-decision must be positive")
    events = [json.loads(line) for line in args.trace.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.limit is not None:
        events = events[: args.limit]
    if not events:
        raise ValueError("trace contains no events")

    import cv2
    import numpy as np
    from PIL import Image, ImageDraw

    task = build_task(args.width, args.height, args.seed)
    output_width, output_height = 1280, 720
    scene_width, panel_x = 860, 860
    cell = min(64, (scene_width - 130) // args.width, (output_height - 130) // args.height)
    origin_x = 70
    origin_y = (output_height - cell * args.height) // 2 + 25
    font_path = Path(r"C:\Windows\Fonts\msyh.ttc")
    if not font_path.is_file():
        raise FileNotFoundError(f"CJK font not found: {font_path}")
    title_font = font(font_path, 25)
    panel_font = font(font_path, 18)
    small_font = font(font_path, 15)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (output_width, output_height)
    )
    if not writer.isOpened():
        raise RuntimeError("OpenCV could not open the requested MP4 output")

    path_positions = [tuple(events[0]["before"]["position"])]
    try:
        for event in events:
            current = tuple(event["after"]["position"])
            path_positions.append(current)
            image = Image.new("RGB", (output_width, output_height), (20, 25, 31))
            draw = ImageDraw.Draw(image)
            draw.rectangle((0, 0, scene_width, output_height), fill=(31, 38, 47))
            draw.rectangle((panel_x, 0, output_width, output_height), fill=(20, 24, 31))
            draw.text((34, 22), "LLM Maze Agent — Trace Layout Prototype", font=title_font, fill=(238, 242, 247))
            draw.text((34, 55), "数据源：A* 全局地图上界，仅验证信息栏同步，不代表LLM结果", font=small_font, fill=(247, 182, 78))

            def center(position: tuple[int, int]) -> tuple[int, int]:
                return origin_x + position[0] * cell + cell // 2, origin_y + position[1] * cell + cell // 2

            for position in task.layout.cells:
                x = origin_x + position[0] * cell
                y = origin_y + position[1] * cell
                draw.rectangle((x, y, x + cell, y + cell), fill=(41, 49, 60))
                if not task.layout.can_move(position, Heading.NORTH):
                    draw.line((x, y, x + cell, y), fill=(220, 226, 234), width=3)
                if not task.layout.can_move(position, Heading.WEST):
                    draw.line((x, y, x, y + cell), fill=(220, 226, 234), width=3)
                if position[0] == task.layout.width - 1 and not task.layout.can_move(position, Heading.EAST):
                    draw.line((x + cell, y, x + cell, y + cell), fill=(220, 226, 234), width=3)
                if position[1] == task.layout.height - 1 and not task.layout.can_move(position, Heading.SOUTH):
                    draw.line((x, y + cell, x + cell, y + cell), fill=(220, 226, 234), width=3)

            for previous, current_path in zip(path_positions, path_positions[1:]):
                draw.line((*center(previous), *center(current_path)), fill=(119, 205, 230), width=5)
            landmarks = ((task.checkpoint, (52, 173, 255), "蓝"), (task.forbidden, (245, 91, 97), "禁"), (task.exit, (67, 207, 134), "终"))
            for position, color, label in landmarks:
                cx, cy = center(position)
                radius = max(11, cell // 5)
                draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=color)
                draw.text((cx - 8, cy - 10), label, font=small_font, fill=(15, 20, 24))
            robot_x, robot_y = center(current)
            draw.ellipse((robot_x - 15, robot_y - 15, robot_x + 15, robot_y + 15), fill=(242, 242, 244), outline=(20, 20, 20), width=2)
            draw.text((robot_x - 10, robot_y - 12), "H1", font=small_font, fill=(20, 20, 20))

            draw.text((panel_x + 28, 25), "TRACE PROTOTYPE · A* ORACLE", font=panel_font, fill=(247, 182, 78))
            y = 77
            draw.text((panel_x + 28, y), "感知输入", font=panel_font, fill=(124, 190, 255))
            y += 30
            perception = event["perception"]
            draw.text(
                (panel_x + 28, y),
                f"前 {'通' if perception['front_open'] else '墙'}  左 {'通' if perception['left_open'] else '墙'}  右 {'通' if perception['right_open'] else '墙'}",
                font=small_font,
                fill=(222, 228, 236),
            )
            y += 42
            draw.text((panel_x + 28, y), "外部记忆", font=panel_font, fill=(124, 190, 255))
            y += 30
            memory = event["memory"]
            draw.text((panel_x + 28, y), f"节点 {memory['memory_nodes']}  当前 {memory['current_node']}", font=small_font, fill=(222, 228, 236))
            y += 24
            draw.text((panel_x + 28, y), f"已知死路 {len(memory['known_dead_ends'])}  蓝点 {'完成' if memory['checkpoint_complete'] else '未到'}", font=small_font, fill=(222, 228, 236))
            y += 44
            draw.text((panel_x + 28, y), "公开决策摘要", font=panel_font, fill=(124, 190, 255))
            y = draw_wrapped(draw, event["planner"]["decision_summary"], (panel_x + 28, y + 30), output_width - panel_x - 56, panel_font, (238, 242, 247))
            y += 20
            draw.text((panel_x + 28, y), f"工具输出  {event['planner']['action']}", font=panel_font, fill=(90, 224, 152))
            y += 40
            draw.text((panel_x + 28, y), f"物理结果  {event['after']['result']}", font=small_font, fill=(222, 228, 236))
            y += 26
            draw.text((panel_x + 28, y), f"步数 {len(path_positions)-1}  碰撞 0", font=small_font, fill=(222, 228, 236))

            frame = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
            for _ in range(args.frames_per_decision):
                writer.write(frame)
    finally:
        writer.release()

    metadata = {
        "asset_type": "layout_prototype",
        "truth_label": "A* oracle trace; not LLM or H1 performance evidence",
        "source_trace": str(args.trace.resolve()),
        "output": str(args.output.resolve()),
        "width": output_width,
        "height": output_height,
        "fps": args.fps,
        "events": len(events),
        "frames_per_event": args.frames_per_decision,
        "duration_seconds": len(events) * args.frames_per_decision / args.fps,
    }
    write_json_atomic(args.metadata_output, metadata)
    print(json.dumps(metadata, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
