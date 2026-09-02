"""Render an auditable visualization of a completed live H1 maze trace.

This intentionally visualizes an exact saved report.  It is not physical
camera footage and says so in every frame; its purpose is to make the complete
blue -> exit -> STOP sequence and the guard's executed actions inspectable.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Render an auditable live-H1 success trace visualization.")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--frames-per-event", type=int, default=12)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata-output", type=Path, required=True)
    args = parser.parse_args()
    if args.fps <= 0 or args.frames_per_event <= 0:
        raise ValueError("fps and frames-per-event must be positive")

    import cv2
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    report = json.loads(args.report.read_text(encoding="utf-8"))
    events = report["events"]
    if not events or not report.get("final_logical_success"):
        raise ValueError("report is not a completed live success trace")

    font_path = Path(r"C:\Windows\Fonts\msyh.ttc")
    if not font_path.is_file():
        raise FileNotFoundError(font_path)
    title = ImageFont.truetype(str(font_path), 27)
    body = ImageFont.truetype(str(font_path), 20)
    small = ImageFont.truetype(str(font_path), 16)

    width, height, left_w = 1280, 720, 850
    cell, ox, oy = 64, 64, 125
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError("OpenCV could not open output")

    def centre(pos: tuple[int, int]) -> tuple[int, int]:
        return ox + pos[0] * cell + cell // 2, oy + pos[1] * cell + cell // 2

    positions = [tuple(events[0]["before"]["position"])]
    route_count = 0
    try:
        for index, event in enumerate(events, start=1):
            before = tuple(event["before"]["position"])
            after = tuple(event["after"]["position"])
            positions.append(after)
            planner = event["planner"]
            physical = event.get("physical_macro", {})
            guard = planner.get("guard_reason")
            route_count += int(guard == "route_known_exit_after_checkpoint")
            frame = Image.new("RGB", (width, height), (18, 23, 30))
            draw = ImageDraw.Draw(frame)
            draw.rectangle((0, 0, left_w, height), fill=(31, 38, 47))
            draw.rectangle((left_w, 0, width, height), fill=(20, 24, 31))
            draw.text((28, 18), "H1 实时成功轨迹 · 日志可视化", font=title, fill=(242, 245, 249))
            draw.text((28, 54), "基于已验收的 LIVE QWEN + LOCAL MEMORY GUARD 报告 · 非相机录像", font=small, fill=(248, 181, 72))

            for x in range(9):
                for y in range(9):
                    px, py = ox + x * cell, oy + y * cell
                    draw.rectangle((px, py, px + cell, py + cell), fill=(44, 52, 63), outline=(86, 98, 112), width=1)
            for pos, color, label in (((0, 8), (68, 171, 255), "蓝"), ((8, 8), (80, 208, 139), "终"), ((0, 0), (240, 202, 91), "起")):
                cx, cy = centre(pos)
                draw.ellipse((cx - 13, cy - 13, cx + 13, cy + 13), fill=color)
                draw.text((cx - 9, cy - 10), label, font=small, fill=(18, 23, 30))
            for old, new in zip(positions, positions[1:]):
                draw.line((*centre(old), *centre(new)), fill=(100, 202, 232), width=5)
            cx, cy = centre(after)
            draw.ellipse((cx - 16, cy - 16, cx + 16, cy + 16), fill=(245, 245, 246), outline=(20, 20, 20), width=2)
            draw.text((cx - 10, cy - 10), "H1", font=small, fill=(20, 20, 20))

            px = left_w + 25
            draw.text((px, 25), "轨迹证据面板", font=body, fill=(128, 192, 255))
            rows = [
                ("事件", f"{index} / {len(events)}"),
                ("模型提议", planner.get("proposed_action", planner["action"])),
                ("实际执行", planner["action"]),
                ("守卫原因", guard or "无覆盖"),
                ("逻辑结果", event["after"]["result"]),
                ("物理 ticks", str(physical.get("ticks", "-"))),
                ("转角残差", f"{abs(float(physical.get('yaw_error_rad', 0.0))):.3f} rad"),
                ("检查点", "已完成" if event["memory"].get("checkpoint_complete") else "未完成"),
                ("出口路由动作", str(route_count)),
            ]
            y = 72
            for label, value in rows:
                draw.text((px, y), label, font=small, fill=(143, 167, 190))
                color = (247, 181, 72) if label == "守卫原因" and guard else (236, 242, 247)
                draw.text((px, y + 23), value, font=small, fill=color)
                y += 57
            if index == len(events):
                draw.rectangle((px, 625, width - 24, 686), fill=(42, 120, 83))
                draw.text((px + 12, 643), "终态：蓝点 → 出口 → STOP，成功", font=body, fill=(245, 255, 249))
            bgr = cv2.cvtColor(np.asarray(frame), cv2.COLOR_RGB2BGR)
            for _ in range(args.frames_per_event):
                writer.write(bgr)
    finally:
        writer.release()

    metadata = {
        "asset_type": "auditable_live_h1_trace_visualization",
        "truth_label": "Visualization derived from an accepted live H1/Qwen/local-memory-guard report; not physical camera footage.",
        "report": str(args.report.resolve()),
        "output": str(args.output.resolve()),
        "events": len(events),
        "final_logical_success": True,
        "duration_seconds": len(events) * args.frames_per_event / args.fps,
    }
    args.metadata_output.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False))


if __name__ == "__main__":
    main()
