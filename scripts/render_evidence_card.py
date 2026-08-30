"""Render short truth-labelled cards from baseline or physical-bridge JSON evidence."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a source-backed video evidence card.")
    parser.add_argument("--mode", choices=("baselines", "h1_bridge"), required=True)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--seconds", type=int, default=9)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata-output", type=Path, required=True)
    args = parser.parse_args()
    if args.fps <= 0 or args.seconds <= 0:
        raise ValueError("fps and seconds must be positive")
    defaults = {
        "baselines": PROJECT_ROOT / "artifacts/maze/baseline_development_v2.json",
        "h1_bridge": PROJECT_ROOT / "artifacts/h1/qwen35_h1_multidecision_smoke_v2.json",
    }
    source = args.source or defaults[args.mode]
    data = json.loads(source.read_text(encoding="utf-8"))

    import cv2
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    font_path = Path(r"C:\Windows\Fonts\msyh.ttc")
    if not font_path.is_file():
        raise FileNotFoundError(font_path)
    title, body, small = (ImageFont.truetype(str(font_path), size) for size in (32, 24, 18))
    width, height = 1280, 720
    image = Image.new("RGB", (width, height), (18, 23, 30))
    draw = ImageDraw.Draw(image)
    text, muted, blue, green, orange, red, panel = (239, 244, 250), (174, 187, 200), (93, 183, 255), (85, 219, 157), (249, 179, 74), (244, 101, 104), (30, 38, 49)

    def card(box, label: str, value: str, detail: str, color):
        x1, y1, x2, y2 = box
        draw.rounded_rectangle(box, radius=15, fill=panel)
        draw.rectangle((x1, y1, x1 + 7, y2), fill=color)
        draw.text((x1 + 24, y1 + 18), label, font=small, fill=muted)
        draw.text((x1 + 24, y1 + 51), value, font=title, fill=text)
        draw.text((x1 + 24, y2 - 30), detail, font=small, fill=muted)

    if args.mode == "baselines":
        result = data["results"]
        draw.text((44, 27), "为什么不把迷宫成功全归功于 LLM？", font=title, fill=text)
        draw.text((44, 67), "同一 200 个开发迷宫：经典算法是必需对照，不是被隐藏的对手", font=small, fill=orange)
        rows = (("A* 全局地图上界", "astar_global_oracle", blue, "知道完整地图，不公平但给出上界"), ("DFS + 显式记忆", "dfs_memory_baseline", green, "经典探索基线"), ("右手规则", "right_hand_rule", red, "无语义规划或路线记忆"))
        y = 145
        for label, key, color, note in rows:
            entry = result[key]
            draw.rounded_rectangle((70, y, 1210, y + 135), radius=14, fill=panel)
            draw.rectangle((70, y, 78, y + 135), fill=color)
            draw.text((105, y + 20), label, font=body, fill=text)
            draw.text((400, y + 20), f"完成 {entry['successes']} / {entry['episodes']}", font=body, fill=color)
            efficiency = entry["mean_path_efficiency_success_only"]
            draw.text((740, y + 20), f"成功路径效率 {efficiency:.3f}", font=body, fill=text)
            draw.text((105, y + 73), note, font=small, fill=muted)
            y += 160
        draw.text((70, 644), "结论：LLM 的价值要从语义指令、有限感知与可审计记忆接口来评估，而不是和 A* 比最短路。", font=body, fill=text)
        truth = "development baseline comparison from 200 fixed mazes; no LLM performance claim"
    else:
        events = data["events"]
        draw.text((44, 27), "实体 H1 不是直接输出关节角：它执行 LLM 的宏动作", font=title, fill=text)
        draw.text((44, 67), "真实物理日志 · 无相机录像 · 开发 smoke，非完整迷宫完成", font=small, fill=orange)
        card((70, 135, 420, 300), "碰撞墙体", str(data["walls"]), "100 段均在 Isaac stage 中验证", blue)
        card((465, 135, 815, 300), "完成宏动作", f"{data['completed_macros']} / {data['requested_decisions']}", "每步等待实际位移或转角阈值", green)
        card((860, 135, 1210, 300), "最终逻辑格", str(tuple(data["final_logical_position"])), "从 (0, 0) 出发", green)
        if "physical_wall_ranges_m" in events[0]:
            first_ranges = events[0]["physical_wall_ranges_m"]
            draw.text((70, 326), f"规划输入来自实体墙四向射线：首步前方 {first_ranges['front']:.2f}m，其他方向 {first_ranges['left']:.2f}m / {first_ranges['right']:.2f}m / {first_ranges['rear']:.2f}m", font=small, fill=orange)
        y = 365
        for index, event in enumerate(events, start=1):
            macro = event["physical_macro"]
            draw.rounded_rectangle((70, y, 1210, y + 74), radius=12, fill=panel)
            draw.text((95, y + 13), f"{index}. Qwen 输出", font=body, fill=blue)
            draw.text((270, y + 13), event["planner"]["action"], font=body, fill=blue)
            draw.text((590, y + 13), f"物理宏动作 {macro['ticks']} ticks", font=body, fill=green)
            detail = f"{macro['criterion']}  |  完成={macro['completed']}"
            draw.text((95, y + 45), detail, font=small, fill=muted)
            y += 92
        draw.text((70, 658), "这证明“语言模型 → 宏动作 → H1 低层策略”的三步物理桥接可运行，不证明完整导航已完成。", font=body, fill=text)
        truth = "three-decision Qwen3.5-to-H1 physical bridge data card; not physical footage or full navigation"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError("OpenCV could not open output")
    frame = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    try:
        for _ in range(args.fps * args.seconds):
            writer.write(frame)
    finally:
        writer.release()
    metadata = {"asset_type": "evidence_card", "mode": args.mode, "truth_label": truth, "source": str(source.resolve()), "output": str(args.output.resolve()), "duration_seconds": args.seconds, "fps": args.fps}
    write_json_atomic(args.metadata_output, metadata)
    print(json.dumps(metadata, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
