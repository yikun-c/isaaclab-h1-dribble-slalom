"""Bounded multi-decision Qwen3.5-to-H1 physical macro-action smoke.

This records the planner's raw tool call and waits for measured H1 translation
or rotation before advancing the *logical* maze state.  It is a development
integration smoke only, not a sealed evaluation or a navigation-success claim.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

SENSOR_DLL_DIRECTORY_HANDLES = []
RUN_PROGRESS: dict[str, object] = {}


def preload_windows_rtx_sensor_dlls() -> None:
    """Make Isaac's optional RGB sensor DLL search path explicit on Windows."""
    if os.name != "nt":
        return
    directories = (
        Path(r"D:\IsaacLab\.venv\Library\bin"),
        Path(r"D:\IsaacLab\.venv\Lib\site-packages\isaacsim\kit\dev\libs\sensors\generic_model_output\bin"),
    )
    available = [path for path in directories if path.is_dir()]
    global SENSOR_DLL_DIRECTORY_HANDLES
    SENSOR_DLL_DIRECTORY_HANDLES = [os.add_dll_directory(str(path)) for path in available]
    os.environ["PATH"] = os.pathsep.join(str(path) for path in available) + os.pathsep + os.environ["PATH"]


preload_windows_rtx_sensor_dlls()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "runtime" / "qwen35_transformers"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Run a bounded multi-decision Qwen3.5 H1 maze bridge smoke.")
parser.add_argument("--seed", type=int, default=2026)
parser.add_argument("--cell-size", type=float, default=1.8, help="Physical maze cell width in metres.")
parser.add_argument("--locomotion-profile", choices=("rough", "flat"), default="rough", help="Official H1 checkpoint/config profile.")
parser.add_argument("--decisions", type=int, default=3)
parser.add_argument("--max-ticks-per-macro", type=int, default=1100)
parser.add_argument("--forward-settle-distance", type=float, default=1.6, help="Metres between short stand-recovery intervals during a long forward macro.")
parser.add_argument("--macro-controller", choices=("fixed", "pose-feedback"), default="fixed", help="Use fixed velocity primitives or bounded root-pose feedback around the official policy.")
parser.add_argument("--turn-tolerance-rad", type=float, default=0.26, help="Measured yaw residual allowed before pose-feedback forward control re-centres the next cell traversal.")
parser.add_argument("--feedback-max-lateral-mps", type=float, default=0.10, help="Bounded body-frame lateral correction for pose-feedback traversal.")
parser.add_argument("--cross-track-tolerance-m", type=float, default=0.50, help="Maximum physical lateral residual before advancing the logical maze state.")
parser.add_argument("--execution-guard", action="store_true", help="Use the labelled local-memory execution guard.")
parser.add_argument("--guard-revisit-threshold", type=int, default=2)
parser.add_argument("--adapter-dir", type=Path, default=PROJECT_ROOT / "runs/qwen35_sft/2026-08-30_21-50-28_qwen3_5_2b_maze_memory_sft_dev200_v1/adapter")
parser.add_argument("--model-dir", type=Path, default=PROJECT_ROOT / "models/qwen3_5_2b")
parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "artifacts/h1/qwen35_h1_multidecision_smoke_v2.json")
parser.add_argument("--video-output", type=Path, help="Optional versioned visible-D3D12 recording path.")
parser.add_argument("--capture-every-ticks", type=int, default=2)
parser.add_argument("--capture-warmup-ticks", type=int, default=30, help="Skip early noisy RTX accumulation frames after physics starts.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.video_output:
    args.enable_cameras = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app


def write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def wrapped_angle(error: float) -> float:
    return (error + math.pi) % (2.0 * math.pi) - math.pi


def planner_input(task, state, memory):
    from maze_agent import observe, sense_physical_maze

    local = observe(task, state)
    ranges = sense_physical_maze(task, state.position, state.heading, cell_size=args.cell_size)
    openings = ranges.open_by_direction(minimum_clearance_m=1.0)
    return {
        "instruction": task.instruction,
        "local_perception": {
            "front_open": openings["front"],
            "left_open": openings["left"],
            "right_open": openings["right"],
            "rear_open": openings["rear"],
            "current_landmarks": list(local.current_landmarks),
            "adjacent_landmarks": list(local.adjacent_landmarks),
        },
        "state": {"heading": state.heading.value, "checkpoint_complete": state.checkpoint_complete, "last_result": state.last_result},
        "memory": memory.compact_summary(state),
    }


def main() -> None:
    import torch
    from peft import PeftModel
    from rsl_rl.runners import OnPolicyRunner
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    import isaaclab.sim as sim_utils
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.sim.utils.stage import get_current_stage
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
    from isaaclab_tasks.manager_based.locomotion.velocity.config.h1.agents.rsl_rl_ppo_cfg import H1FlatPPORunnerCfg, H1RoughPPORunnerCfg
    from isaaclab_tasks.manager_based.locomotion.velocity.config.h1.flat_env_cfg import H1FlatEnvCfg_PLAY
    from isaaclab_tasks.manager_based.locomotion.velocity.config.h1.rough_env_cfg import H1RoughEnvCfg_PLAY

    from maze_agent import (
        Action,
        GRID_HEADING_WORLD_YAW,
        TopologicalMemory,
        build_task,
        decision_event,
        guard_action,
        pose_feedback_velocity,
        parse_planner_response,
        sense_physical_maze,
        step,
        turn_feedback_velocity,
        velocity_for_grid_action,
    )
    from maze_agent.core import reset
    from maze_agent.physical_maze import maze_wall_specs
    from maze_agent.sft import SYSTEM_PROMPT

    if args.decisions <= 0 or args.max_ticks_per_macro <= 0 or args.cell_size <= 0.2 or args.forward_settle_distance <= 0.0 or not 0.05 <= args.turn_tolerance_rad <= 0.50 or not 0.01 <= args.feedback_max_lateral_mps <= 0.30 or not 0.10 <= args.cross_track_tolerance_m < args.cell_size / 2.0:
        raise ValueError("decision/macro budgets and cell-size must be positive")
    # The high-level task is seed-controlled; seed PhysX/PyTorch as well so a
    # controller comparison is not confounded by a new random locomotion reset.
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    task = build_task(9, 9, args.seed)
    RUN_PROGRESS.update(
        {
            "status": "initializing",
            "seed": args.seed,
            "cell_size_m": args.cell_size,
            "requested_decisions": args.decisions,
            "execution_guard": args.execution_guard,
        }
    )
    walls = maze_wall_specs(task, cell_size=args.cell_size)
    for spec in walls:
        wall_cfg = sim_utils.CuboidCfg(
            size=spec.size,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.20, 0.40, 0.62), roughness=0.64),
        )
        wall_cfg.func(f"/World/Maze/{spec.name}", wall_cfg, translation=spec.translation)
    task_name = "Isaac-Velocity-Flat-H1-v0" if args.locomotion_profile == "flat" else "Isaac-Velocity-Rough-H1-v0"
    checkpoint = get_published_pretrained_checkpoint("rsl_rl", task_name)
    if checkpoint is None:
        raise RuntimeError("official H1 checkpoint unavailable")
    cfg = H1FlatEnvCfg_PLAY() if args.locomotion_profile == "flat" else H1RoughEnvCfg_PLAY()
    cfg.seed = args.seed
    cfg.scene.num_envs = 1
    cfg.scene.clone_in_fabric = False
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None
    cfg.curriculum = None
    cfg.sim.device = args.device or "cuda:0"
    # Macro completion can take hundreds of 20ms environment steps. Size the
    # episode from the requested bounded run so a silent time-limit reset can
    # never masquerade as continuous physical navigation.
    cfg.episode_length_s = max(60.0, args.decisions * (args.max_ticks_per_macro + 12) * 0.02 * 1.1)
    cfg.commands.base_velocity.resampling_time_range = (1000.0, 1000.0)
    cfg.commands.base_velocity.rel_standing_envs = 0.0
    cfg.commands.base_velocity.rel_heading_envs = 0.0
    cfg.commands.base_velocity.heading_command = False
    cfg.events.reset_base.params = {
        "pose_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)},
        "velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0), "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0)},
    }
    env = ManagerBasedRLEnv(cfg, render_mode="rgb_array" if args.video_output else None)
    wrapped = RslRlVecEnvWrapper(env)
    model = base = None
    writer = None
    temporary_video = None
    recorded_frames = 0
    simulation_ticks = 0
    capture_context = {"decision": 0, "proposed": "等待模型", "executed": "等待模型", "guard": None}
    try:
        runner_cfg = H1FlatPPORunnerCfg() if args.locomotion_profile == "flat" else H1RoughPPORunnerCfg()
        runner = OnPolicyRunner(wrapped, runner_cfg.to_dict(), log_dir=None, device=env.device)
        runner.load(checkpoint)
        locomotion = runner.get_inference_policy(device=env.device)
        observations, _ = wrapped.reset()
        stage = get_current_stage()
        spawned_walls = sum(stage.GetPrimAtPath(f"/World/Maze/{spec.name}").IsValid() for spec in walls)
        root_initial = env.scene["robot"].data.root_pos_w[0].detach().cpu().tolist()
        if spawned_walls != len(walls) or max(abs(root_initial[0]), abs(root_initial[1])) > 0.65:
            raise RuntimeError("H1 was not initialized at the physical maze start")
        processor = AutoProcessor.from_pretrained(args.adapter_dir, local_files_only=True)
        base = AutoModelForMultimodalLM.from_pretrained(args.model_dir, local_files_only=True, dtype=torch.bfloat16).to("cuda").eval()
        model = PeftModel.from_pretrained(base, args.adapter_dir, local_files_only=True).eval()
        term = env.command_manager.get_term("base_velocity")
        state = reset(task)
        memory = TopologicalMemory()
        events: list[dict] = []

        if args.video_output:
            if args.capture_every_ticks <= 0 or args.capture_warmup_ticks < 0:
                raise ValueError("capture timing values must be non-negative/positive")
            import cv2
            import numpy as np
            from PIL import Image, ImageDraw, ImageFont

            font_path = Path(r"C:\Windows\Fonts\msyh.ttc")
            if not font_path.is_file():
                raise FileNotFoundError(font_path)
            title_font = ImageFont.truetype(str(font_path), 24)
            body_font = ImageFont.truetype(str(font_path), 17)
            args.video_output.parent.mkdir(parents=True, exist_ok=True)
            temporary_video = args.video_output.with_suffix(".raw.mp4")
            # Do not write black Hydra warm-up buffers to an otherwise valid MP4.
            for _ in range(60):
                env.sim.render()

        def apply_velocity(target_values: tuple[float, float, float]):
            nonlocal observations, writer, recorded_frames, simulation_ticks
            target = torch.tensor([target_values], device=env.device, dtype=term.vel_command_b.dtype)
            term.vel_command_b[:] = target
            with torch.inference_mode():
                joint_actions = locomotion(observations)
                observations, _, done, _ = wrapped.step(joint_actions)
            if bool(done.any()):
                raise RuntimeError("H1 environment terminated during a macro action; refusing to stitch reset state into one trajectory")
            if not torch.isfinite(env.scene["robot"].data.root_pos_w).all():
                raise RuntimeError("non-finite H1 root state")
            simulation_ticks += 1
            if args.video_output and simulation_ticks > args.capture_warmup_ticks and simulation_ticks % args.capture_every_ticks == 0:
                root = env.scene["robot"].data.root_pos_w[0]
                root_x, root_y = float(root[0].item()), float(root[1].item())
                env.sim.set_camera_view(eye=(root_x - 5.2, root_y - 7.6, 6.2), target=(root_x + 3.2, root_y + 3.2, 0.75))
                env.sim.render()
                rgb = env.render(recompute=False)
                if rgb is not None and rgb.size:
                    image = Image.fromarray(rgb)
                    if float(np.asarray(image).mean()) >= 8.0:
                        if image.size != (1280, 720):
                            image = image.resize((1280, 720), Image.Resampling.LANCZOS)
                        draw = ImageDraw.Draw(image, "RGBA")
                        draw.rectangle((0, 0, 1280, 94), fill=(17, 23, 30, 225))
                        draw.text((20, 9), f"实体迷宫 · Qwen3.5 → H1 {args.decisions}决策桥接", font=title_font, fill=(246, 248, 250, 255))
                        draw.text((20, 44), f"决策 {capture_context['decision']}  提议 {capture_context['proposed']}  执行 {capture_context['executed']}", font=body_font, fill=(165, 219, 235, 255))
                        guard_text = capture_context["guard"] or "无守卫覆盖（物理宏动作仍需阈值完成）"
                        draw.text((20, 68), f"{guard_text}  |  开发 smoke，非完整迷宫成功", font=body_font, fill=(248, 181, 72, 255))
                        if writer is None:
                            writer = cv2.VideoWriter(str(temporary_video), cv2.VideoWriter_fourcc(*"mp4v"), 30, (1280, 720))
                            if not writer.isOpened():
                                raise RuntimeError("could not open physical bridge video writer")
                        writer.write(cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR))
                        recorded_frames += 1

        for decision_index in range(args.decisions):
            if state.terminated:
                break
            RUN_PROGRESS.update({"status": "planning", "decision_index": decision_index, "logical_position": list(state.position), "logical_heading": state.heading.value})
            memory.record_observation(task, state)
            memory_before = memory.compact_summary(state)
            payload = planner_input(task, state, memory)
            messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": [{"type": "text", "text": json.dumps(payload, ensure_ascii=False)}]}]
            prompt = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=prompt, return_tensors="pt").to("cuda")
            with torch.inference_mode():
                started = time.perf_counter()
                output_ids = model.generate(**inputs, max_new_tokens=64, do_sample=False)
                latency_ms = (time.perf_counter() - started) * 1000.0
            raw = processor.decode(output_ids[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True)
            proposed = parse_planner_response(raw)
            decision = proposed
            guard_reason = None
            if args.execution_guard:
                guarded = guard_action(task, state, memory, proposed.action, args.guard_revisit_threshold)
                guard_reason = guarded.reason
                if guarded.overridden:
                    decision = replace(proposed, action=guarded.action, fallback_reason=f"memory_guard:{guarded.reason}")
            capture_context.update({"decision": decision_index + 1, "proposed": proposed.action.value, "executed": decision.action.value, "guard": guard_reason})
            before_physical = env.scene["robot"].data.root_pos_w[0].detach().cpu().tolist()
            macro_ticks = 0
            physical_completed = False
            macro_detail: dict[str, object] = {}
            if decision.action is Action.STOP:
                for _ in range(12):
                    apply_velocity((0.0, 0.0, 0.0))
                macro_ticks, physical_completed, macro_detail = 12, True, {"criterion": "stand_for_12_ticks"}
            elif decision.action in {Action.MOVE_FORWARD, Action.BACKTRACK}:
                start_x, start_y = before_physical[0], before_physical[1]
                logical_target = step(task, state, decision.action)
                if logical_target.position == state.position:
                    raise RuntimeError("planner attempted physical translation into a blocked logical edge")
                physical_heading = state.heading if decision.action is Action.MOVE_FORWARD else state.heading.opposite()
                world_yaw = GRID_HEADING_WORLD_YAW[physical_heading]
                target_xy = (logical_target.position[0] * args.cell_size, logical_target.position[1] * args.cell_size)
                target_dx, target_dy = target_xy[0] - start_x, target_xy[1] - start_y
                target_forward = target_dx * math.cos(world_yaw) + target_dy * math.sin(world_yaw)
                target_cross = -target_dx * math.sin(world_yaw) + target_dy * math.cos(world_yaw)
                if target_forward <= 0.0:
                    raise RuntimeError("physical target is behind the requested macro direction")
                target_values = velocity_for_grid_action(decision.action).as_tuple()
                last_velocity_target = target_values
                max_lateral_command = 0.0
                max_yaw_command = 0.0
                progress = 0.0
                cross_track = 0.0
                next_settle_distance = args.forward_settle_distance
                recovery_intervals = 0
                for _ in range(args.max_ticks_per_macro):
                    if args.macro_controller == "pose-feedback":
                        root_before_command = env.scene["robot"].data.root_pos_w[0]
                        feedback = pose_feedback_velocity(
                            target_xy=target_xy,
                            # BACKTRACK preserves the logical body heading and
                            # therefore needs a negative base-x command toward
                            # its target, not a hidden 180-degree reorientation.
                            target_yaw=GRID_HEADING_WORLD_YAW[state.heading],
                            current_xy=(float(root_before_command[0].item()), float(root_before_command[1].item())),
                            current_yaw=float(env.scene["robot"].data.heading_w[0].item()),
                            max_lateral_mps=args.feedback_max_lateral_mps,
                        )
                        last_velocity_target = feedback.as_tuple()
                    else:
                        last_velocity_target = target_values
                    max_lateral_command = max(max_lateral_command, abs(last_velocity_target[1]))
                    max_yaw_command = max(max_yaw_command, abs(last_velocity_target[2]))
                    apply_velocity(last_velocity_target)
                    macro_ticks += 1
                    current = env.scene["robot"].data.root_pos_w[0].detach().cpu().tolist()
                    moved_x, moved_y = current[0] - start_x, current[1] - start_y
                    progress = moved_x * math.cos(world_yaw) + moved_y * math.sin(world_yaw)
                    cross_track = -moved_x * math.sin(world_yaw) + moved_y * math.cos(world_yaw) - target_cross
                    if progress >= target_forward - 0.25 and abs(cross_track) <= args.cross_track_tolerance_m:
                        physical_completed = True
                        break
                    if progress >= next_settle_distance and next_settle_distance < target_forward - 0.25:
                        for _ in range(12):
                            apply_velocity((0.0, 0.0, 0.0))
                            macro_ticks += 1
                        recovery_intervals += 1
                        next_settle_distance += args.forward_settle_distance
                macro_detail = {
                    "criterion": f"absolute_target_center_forward<=0.25m_and_cross<={args.cross_track_tolerance_m:.3f}m",
                    "signed_translation_m": progress,
                    "target_forward_m": target_forward,
                    "cross_track_error_m": cross_track,
                    "target_root_xy": target_xy,
                    "stand_recovery_intervals": recovery_intervals,
                    "expected_world_yaw": world_yaw,
                    "macro_controller": args.macro_controller,
                    "last_velocity_target": last_velocity_target,
                    "max_lateral_command_mps": max_lateral_command,
                    "configured_max_lateral_mps": args.feedback_max_lateral_mps,
                    "configured_cross_track_tolerance_m": args.cross_track_tolerance_m,
                    "max_yaw_command_rps": max_yaw_command,
                }
            else:
                logical_after_turn = step(task, state, decision.action)
                target_yaw = GRID_HEADING_WORLD_YAW[logical_after_turn.heading]
                target_values = velocity_for_grid_action(decision.action).as_tuple()
                last_velocity_target = target_values
                yaw_error = float("inf")
                for _ in range(args.max_ticks_per_macro):
                    if args.macro_controller == "pose-feedback":
                        last_velocity_target = turn_feedback_velocity(
                            current_yaw=float(env.scene["robot"].data.heading_w[0].item()), target_yaw=target_yaw
                        ).as_tuple()
                    apply_velocity(last_velocity_target)
                    macro_ticks += 1
                    current_yaw = float(env.scene["robot"].data.heading_w[0].item())
                    yaw_error = wrapped_angle(target_yaw - current_yaw)
                    if abs(yaw_error) <= args.turn_tolerance_rad:
                        physical_completed = True
                        break
                macro_detail = {
                    "criterion": f"abs_yaw_error_rad<={args.turn_tolerance_rad:.3f}",
                    "target_world_yaw": target_yaw,
                    "yaw_error_rad": yaw_error,
                    "macro_controller": args.macro_controller,
                    "last_velocity_target": last_velocity_target,
                }
            # Settle before the next planner observation and prove the macro's
            # completed state from physical feedback, not requested duration.
            for _ in range(12):
                apply_velocity((0.0, 0.0, 0.0))
            logical_after = step(task, state, decision.action) if physical_completed else replace(state, last_result="physical_macro_incomplete")
            event = decision_event(task, state, decision, logical_after, memory_before, raw, latency_ms=latency_ms, token_count=None)
            if args.execution_guard:
                event["planner"]["proposed_action"] = proposed.action.value
                event["planner"]["guard_reason"] = guard_reason
            ranges = sense_physical_maze(task, state.position, state.heading, cell_size=args.cell_size)
            event["physical_wall_ranges_m"] = {"front": ranges.front_m, "left": ranges.left_m, "right": ranges.right_m, "rear": ranges.rear_m}
            event["physical_macro"] = {
                "completed": physical_completed,
                "ticks": macro_ticks,
                "before_root_w": before_physical,
                "after_root_w": env.scene["robot"].data.root_pos_w[0].detach().cpu().tolist(),
                "nominal_velocity_target": velocity_for_grid_action(decision.action).as_tuple(),
                **macro_detail,
            }
            events.append(event)
            memory.record_transition(task, state, decision.action, logical_after)
            state = logical_after
            RUN_PROGRESS.update(
                {
                    "status": "macro_completed",
                    "decision_index": decision_index + 1,
                    "logical_position": list(state.position),
                    "logical_heading": state.heading.value,
                    "last_action": decision.action.value,
                    "last_macro_ticks": macro_ticks,
                    "last_root_w": event["physical_macro"]["after_root_w"],
                }
            )
            if not physical_completed:
                break
        report = {
            "result": "QWEN35_H1_MULTIDECISION_SMOKE_OK",
            "truth_label": "development-only bounded multi-decision Qwen3.5-to-H1 macro bridge; not a sealed or completed maze-navigation evaluation",
            "maze_seed": args.seed,
            "requested_decisions": args.decisions,
            "execution_interface": "Qwen plus local-memory guard" if args.execution_guard else "Qwen proposed action directly executed",
            "guard_revisit_threshold": args.guard_revisit_threshold if args.execution_guard else None,
            "guard_overrides": sum(event["planner"].get("guard_reason") is not None for event in events),
            "completed_macros": sum(event["physical_macro"]["completed"] for event in events),
            "final_logical_position": list(state.position),
            "final_logical_heading": state.heading.value,
            "final_logical_success": state.success,
            "final_logical_result": state.last_result,
            "events": events,
            "walls": len(walls),
            "physical_cell_size_m": args.cell_size,
            "spawned_walls": spawned_walls,
            "model_dir": str(args.model_dir.resolve()),
            "adapter_dir": str(args.adapter_dir.resolve()),
            "locomotion_checkpoint": str(checkpoint),
            "locomotion_profile": args.locomotion_profile,
        }
        if writer is not None:
            writer.release()
            writer = None
            assert temporary_video is not None
            temporary_video.replace(args.video_output)
            report["video"] = {
                "output": str(args.video_output.resolve()),
                "frames": recorded_frames,
                "fps": 30,
                "duration_seconds": recorded_frames / 30.0,
                "truth_label": "three-decision Qwen3.5-to-H1 physical development bridge; not completed maze navigation",
            }
        elif args.video_output:
            raise RuntimeError("camera recorder produced no non-black frame")
        write_json_atomic(args.output, report)
        print(json.dumps({key: value for key, value in report.items() if key != "events"}, ensure_ascii=False, sort_keys=True), flush=True)
    finally:
        del model, base
        if writer is not None:
            writer.release()
        wrapped.close()
        import torch
        torch.cuda.empty_cache()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        failure_path = args.output.with_suffix(".failure.json")
        write_json_atomic(
            failure_path,
            {
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "output_target": str(args.output.resolve()),
                "seed": args.seed,
                "cell_size_m": args.cell_size,
                "decisions": args.decisions,
                "max_ticks_per_macro": args.max_ticks_per_macro,
                "execution_guard": args.execution_guard,
                "locomotion_profile": args.locomotion_profile,
                "progress": RUN_PROGRESS,
            },
        )
        raise
    finally:
        simulation_app.close()
