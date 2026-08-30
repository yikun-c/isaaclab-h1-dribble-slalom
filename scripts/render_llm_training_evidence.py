"""Render a truth-labelled LLM training evidence clip from saved experiment artifacts.

This deliberately visualises real SFT/evaluation files rather than terminal text.
The recovery intervention is retained as a negative result: lower train loss did
not improve closed-loop navigation.  This makes the video evidence auditable and
avoids implying that loss alone proves an embodied-planning improvement.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temp, path)


def training_points(trainer_state: dict, stride: int = 5) -> list[tuple[int, float]]:
    history = [item for item in trainer_state["log_history"] if "loss" in item and "step" in item]
    sampled = [item for item in history if item["step"] % stride == 0]
    if history and history[-1] not in sampled:
        sampled.append(history[-1])
    return [(int(item["step"]), float(item["loss"])) for item in sampled]


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a real Qwen3.5 maze-SFT evidence video.")
    parser.add_argument(
        "--base-run",
        type=Path,
        default=PROJECT_ROOT / "runs/qwen35_sft/2026-08-30_21-50-28_qwen3_5_2b_maze_memory_sft_dev200_v1",
    )
    parser.add_argument(
        "--recovery-run",
        type=Path,
        default=PROJECT_ROOT / "runs/qwen35_sft/2026-08-30_22-35-08_qwen3_5_2b_maze_memory_recovery_sft_dev300_v1",
    )
    parser.add_argument(
        "--base-action-eval",
        type=Path,
        default=PROJECT_ROOT / "artifacts/maze/eval_qwen35_toolcalls_memory_dev200_dev64_v1.json",
    )
    parser.add_argument(
        "--base-closed-eval",
        type=Path,
        default=PROJECT_ROOT / "artifacts/maze/eval_qwen35_closedloop_memory_dev200_dev3_v1.json",
    )
    parser.add_argument(
        "--recovery-action-eval",
        type=Path,
        default=PROJECT_ROOT / "artifacts/maze/eval_qwen35_toolcalls_memory_recovery_dev64_v1.json",
    )
    parser.add_argument(
        "--recovery-closed-eval",
        type=Path,
        default=PROJECT_ROOT / "artifacts/maze/eval_qwen35_closedloop_memory_recovery_dev3_v1.json",
    )
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--output", type=Path, default=PROJECT_ROOT / "artifacts/video/llm_training_evidence_v2.mp4"
    )
    parser.add_argument(
        "--metadata-output", type=Path, default=PROJECT_ROOT / "artifacts/video/llm_training_evidence_v2.json"
    )
    args = parser.parse_args()
    if args.fps <= 0:
        raise ValueError("fps must be positive")

    base_summary = load_json(args.base_run / "train_summary.json")
    recovery_summary = load_json(args.recovery_run / "train_summary.json")
    base_points = training_points(load_json(args.base_run / "checkpoint-200/trainer_state.json"))
    recovery_points = training_points(load_json(args.recovery_run / "checkpoint-300/trainer_state.json"))
    base_action, base_closed = load_json(args.base_action_eval), load_json(args.base_closed_eval)
    recovery_action, recovery_closed = load_json(args.recovery_action_eval), load_json(args.recovery_closed_eval)

    import cv2
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    font_path = Path(r"C:\Windows\Fonts\msyh.ttc")
    if not font_path.is_file():
        raise FileNotFoundError(font_path)
    def font(size: int):
        return ImageFont.truetype(str(font_path), size=size)

    title_font, subtitle_font, body_font, small_font = font(34), font(22), font(25), font(18)
    width, height = 1280, 720
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError("OpenCV could not open requested MP4")

    bg, panel, text, muted, blue, green, orange, red = (
        (18, 23, 30), (30, 38, 49), (239, 244, 250), (171, 184, 197),
        (93, 183, 255), (85, 219, 157), (249, 179, 74), (244, 101, 104),
    )

    def draw_header(draw: ImageDraw.ImageDraw, section: str) -> None:
        draw.rectangle((0, 0, width, 86), fill=(25, 32, 42))
        draw.text((42, 20), "LLM 训练证据：Qwen3.5-2B + LoRA", font=title_font, fill=text)
        draw.text((42, 59), section, font=small_font, fill=blue)
        draw.text((1008, 59), "开发集，非最终测试", font=small_font, fill=orange)

    def card(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], label: str, value: str, detail: str, color=blue) -> None:
        x1, y1, x2, y2 = box
        draw.rounded_rectangle(box, radius=14, fill=panel)
        draw.rectangle((x1, y1, x1 + 6, y2), fill=color)
        draw.text((x1 + 24, y1 + 20), label, font=small_font, fill=muted)
        draw.text((x1 + 24, y1 + 51), value, font=title_font, fill=text)
        draw.text((x1 + 24, y2 - 31), detail, font=small_font, fill=muted)

    def render_data(frame_index: int) -> Image.Image:
        image = Image.new("RGB", (width, height), bg)
        draw = ImageDraw.Draw(image)
        draw_header(draw, "1 / 4  训练数据不是视频素材：由 A* 专家轨迹生成")
        card(draw, (58, 135, 592, 310), "专家轨迹样本", f"{base_summary['dataset_rows']:,}", "200 个训练迷宫 · 结构化状态 → 动作", blue)
        card(draw, (688, 135, 1222, 310), "恢复状态混合样本", f"{recovery_summary['dataset_rows']:,}", "额外加入随机合法前缀后的纠错状态", orange)
        draw.rounded_rectangle((58, 365, 1222, 615), radius=14, fill=panel)
        draw.text((91, 400), "每条样本包含：局部通路 + 已访问记忆 + 专家下一步工具调用", font=body_font, fill=text)
        prompt = '{"front_open": true, "memory": {"visited_exits": ["west"]}}'
        target = '{"action": "TURN_LEFT", "decision_summary": "探索未访问分支"}'
        draw.text((91, 455), "输入  " + prompt, font=small_font, fill=blue)
        draw.text((91, 505), "标签  " + target, font=small_font, fill=green)
        draw.text((91, 564), "训练只更新 LoRA 适配器；不把全局迷宫答案直接给模型。", font=small_font, fill=muted)
        return image

    def draw_curve(draw: ImageDraw.ImageDraw, points: list[tuple[int, float]], rect: tuple[int, int, int, int], color: tuple[int, int, int]) -> None:
        x1, y1, x2, y2 = rect
        draw.rectangle(rect, outline=(80, 93, 108), width=2)
        max_step = max(step for step, _ in points)
        max_loss = max(2.3, max(loss for _, loss in points))
        for loss_label in (0.0, 1.0, 2.0):
            y = y2 - int((loss_label / max_loss) * (y2 - y1))
            draw.line((x1, y, x2, y), fill=(55, 66, 78), width=1)
            draw.text((x1 - 52, y - 9), f"{loss_label:.0f}", font=small_font, fill=muted)
        xy = [(x1 + int(step / max_step * (x2 - x1)), y2 - int(loss / max_loss * (y2 - y1))) for step, loss in points]
        draw.line(xy, fill=color, width=4)
        draw.text((x1, y2 + 16), "step 0", font=small_font, fill=muted)
        draw.text((x2 - 80, y2 + 16), f"step {max_step}", font=small_font, fill=muted)

    def render_base(frame_index: int) -> Image.Image:
        image = Image.new("RGB", (width, height), bg)
        draw = ImageDraw.Draw(image)
        draw_header(draw, "2 / 4  第一次 LoRA SFT：训练损失下降，但需要独立闭环检查")
        draw.text((85, 122), "训练 loss（每 5 步抽样，原始 trainer_state.json）", font=subtitle_font, fill=text)
        draw_curve(draw, base_points, (132, 165, 1140, 470), blue)
        card(draw, (85, 545, 438, 675), "训练步数", str(base_summary["max_steps"]), f"耗时 {base_summary['metrics']['train_runtime']:.0f} 秒", blue)
        card(draw, (465, 545, 815, 675), "最终训练 loss", f"{base_summary['metrics']['train_loss']:.3f}", "不是成功率", orange)
        card(draw, (842, 545, 1195, 675), "可训练参数", "0.033%", "LoRA q/k/v/o 投影", green)
        return image

    def render_eval(frame_index: int) -> Image.Image:
        image = Image.new("RGB", (width, height), bg)
        draw = ImageDraw.Draw(image)
        draw_header(draw, "3 / 4  独立开发评测：格式正确 ≠ 走出迷宫")
        card(draw, (72, 150, 575, 340), "动作完全正确", f"{base_action['exact_action_accuracy'] * 100:.2f}%", "64 个未训练开发状态", blue)
        card(draw, (705, 150, 1208, 340), "有效 JSON", f"{base_action['valid_json_rate'] * 100:.0f}%", "输出接口已经稳定", green)
        card(draw, (72, 410, 575, 600), "闭环成功", f"{base_closed['successes']} / {base_closed['episodes']}", "同一开发环境 · 最多 128 次决策", orange)
        card(draw, (705, 410, 1208, 600), "平均回环观测", f"{base_closed['mean_loop_observations']:.2f}", "模型仍会重复访问状态", red)
        draw.text((72, 642), "结论：离线下一步预测不错，但长期记忆/执行接口仍会造成闭环失败。", font=body_font, fill=text)
        return image

    def render_recovery(frame_index: int) -> Image.Image:
        image = Image.new("RGB", (width, height), bg)
        draw = ImageDraw.Draw(image)
        draw_header(draw, "4 / 4  加入恢复样本后的反例：更低 loss，闭环反而更差")
        draw.text((85, 120), "恢复混合 SFT 的训练 loss（每 5 步抽样）", font=subtitle_font, fill=text)
        draw_curve(draw, recovery_points, (132, 160, 1140, 410), orange)
        card(draw, (72, 500, 430, 650), "最终训练 loss", f"{recovery_summary['metrics']['train_loss']:.3f}", "比第一次更低", orange)
        card(draw, (461, 500, 819, 650), "动作完全正确", f"{recovery_action['exact_action_accuracy'] * 100:.2f}%", "64 个开发状态", blue)
        card(draw, (850, 500, 1208, 650), "闭环成功", f"{recovery_closed['successes']} / {recovery_closed['episodes']}", "比 1/3 更差：保留负结果", red)
        draw.text((72, 676), "下一步不是继续堆数据，而是改进记忆—执行接口并重新验证。", font=small_font, fill=muted)
        return image

    sections = ((render_data, 4), (render_base, 6), (render_eval, 5), (render_recovery, 6))
    total_frames = 0
    try:
        for render, seconds in sections:
            frames = seconds * args.fps
            for index in range(frames):
                image = render(index)
                writer.write(cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR))
            total_frames += frames
    finally:
        writer.release()

    metadata = {
        "asset_type": "training_evidence",
        "truth_label": "Development-only Qwen3.5-2B LoRA SFT evidence; recovery result is a retained negative result.",
        "output": str(output),
        "fps": args.fps,
        "frames": total_frames,
        "duration_seconds": total_frames / args.fps,
        "sources": [str(path.resolve()) for path in (args.base_run, args.recovery_run, args.base_action_eval, args.base_closed_eval, args.recovery_action_eval, args.recovery_closed_eval)],
        "metrics": {
            "base_train_loss": base_summary["metrics"]["train_loss"],
            "base_action_exact_rate": base_action["exact_action_accuracy"],
            "base_closed_successes": base_closed["successes"],
            "base_closed_episodes": base_closed["episodes"],
            "recovery_train_loss": recovery_summary["metrics"]["train_loss"],
            "recovery_action_exact_rate": recovery_action["exact_action_accuracy"],
            "recovery_closed_successes": recovery_closed["successes"],
            "recovery_closed_episodes": recovery_closed["episodes"],
        },
    }
    write_json_atomic(args.metadata_output.resolve(), metadata)
    print(json.dumps(metadata, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
