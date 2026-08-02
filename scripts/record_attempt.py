from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py  # noqa: F401 - preload HDF5 DLLs before Kit initializes D3D12

from isaaclab.app import AppLauncher


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

parser = argparse.ArgumentParser(description="Record one complete deterministic dribble attempt.")
parser.add_argument("checkpoint", type=Path)
parser.add_argument("output", type=Path)
parser.add_argument("--stage", type=int, choices=(0, 1, 2, 3), required=True)
parser.add_argument("--seed", type=int, default=2026)
parser.add_argument("--iteration", type=int, required=True)
parser.add_argument("--phase", type=str, required=True)
parser.add_argument("--transitions", type=int)
parser.add_argument("--attempts", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from rsl_rl.runners import OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from dribble_agent.dribble_env import DribbleSlalomEnv, DribbleSlalomEnvCfg
from dribble_agent.ppo_cfg import DribbleSlalomPPORunnerCfg


FONT_PATH = Path(r"C:\Windows\Fonts\msyh.ttc")
FPS = 30
STAGE_NAMES = ("one-pole attempt", "two-pole attempt", "three-pole attempt", "four poles and shot")


def overlay_hud(
    rgb: np.ndarray,
    *,
    iteration: int,
    phase: str,
    transitions: int,
    ball_speed: float,
    route_index: int,
    active_poles: int,
    status: str,
) -> np.ndarray:
    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image, "RGBA")
    width = image.width
    draw.rectangle((0, 0, width, 78), fill=(20, 24, 29, 224))
    draw.rectangle((0, 75, width, 78), fill=(45, 188, 215, 255))
    title_font = ImageFont.truetype(str(FONT_PATH), 24)
    data_font = ImageFont.truetype(str(FONT_PATH), 17)
    small_font = ImageFont.truetype(str(FONT_PATH), 15)

    draw.text((22, 8), "AI learns dribble slalom", font=title_font, fill=(247, 249, 251, 255))
    draw.text((300, 13), f"Policy {iteration:04d}  |  {phase}", font=data_font, fill=(165, 218, 232, 255))
    status_bbox = draw.textbbox((0, 0), status, font=data_font)
    draw.text(
        (width - (status_bbox[2] - status_bbox[0]) - 22, 13),
        status,
        font=data_font,
        fill=(245, 207, 93, 255),
    )
    draw.text(
        (22, 48),
        f"Experience {transitions / 1_000_000:.1f}M  |  Gates {route_index}/{active_poles}  |  Ball {ball_speed:.1f} m/s",
        font=small_font,
        fill=(198, 207, 216, 255),
    )
    return np.asarray(image)


def current_status(env: DribbleSlalomEnv) -> tuple[str, str]:
    route_index = int(env._route_index[0].item())
    if bool(env._goal_achieved[0].item()):
        return "Goal", "goal"
    if bool(env._wrong_route_latched[0].item()):
        return "Wrong route", f"wrong route after {route_index} gates"
    if bool(env._fallen[0].item()):
        return "Fallen", f"fell after {route_index} gates"
    if bool(env._failure_latched[0].item()):
        return "Lost control", f"lost control after {route_index} gates"
    if bool(env._route_complete[0].item()):
        return "Route complete", f"completed {route_index} gates"
    if bool(env._has_touched_ball[0].item()):
        return "Dribbling", "attempt in progress"
    return "Approaching ball", "no touch"


def main() -> None:
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    env_cfg = DribbleSlalomEnvCfg()
    env_cfg.scene.num_envs = 1
    env_cfg.scene.clone_in_fabric = False
    env_cfg.forced_stage = args.stage
    env_cfg.seed = args.seed
    env_cfg.success_hold_s = 2.8
    env_cfg.failure_hold_s = 1.8
    env_cfg.stall_timeout_s = 10.0
    env_cfg.episode_length_s = 35.0
    env_cfg.sim.device = args.device or "cuda:0"
    env_cfg.viewer.resolution = (1280, 720)
    env_cfg.viewer.eye = (1.4, -8.8, 3.3)
    env_cfg.viewer.lookat = (2.4, 0.0, 0.72)

    agent_cfg = DribbleSlalomPPORunnerCfg()
    agent_cfg.device = args.device or "cuda:0"
    env = DribbleSlalomEnv(env_cfg, render_mode="rgb_array")
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(str(checkpoint))
    policy = runner.get_inference_policy(device=env.device)
    obs = wrapped.get_observations()

    for _ in range(30):
        env.render(recompute=True)

    if args.attempts < 1:
        raise ValueError("attempts must be at least 1")

    transitions = args.transitions or args.iteration * 8192 * agent_cfg.num_steps_per_env
    try:
        for attempt in range(1, args.attempts + 1):
            output = args.output
            if args.attempts > 1:
                output = output.with_name(f"{output.stem}_a{attempt:02d}{output.suffix}")
            temporary_video = output.with_suffix(".raw.mp4")
            writer = cv2.VideoWriter(
                str(temporary_video),
                cv2.VideoWriter_fourcc(*"mp4v"),
                FPS,
                tuple(env_cfg.viewer.resolution),
            )
            if not writer.isOpened():
                raise RuntimeError(f"Could not open video writer: {temporary_video}")

            frames = 0
            result = "timeout"
            max_ball_speed = 0.0
            maximum_route_index = 0
            camera_x = 0.0
            try:
                for _ in range(env.max_episode_length + 2):
                    robot_x = float(
                        (env._robot.data.root_pos_w[0, 0] - env.scene.env_origins[0, 0]).item()
                    )
                    camera_x = 0.94 * camera_x + 0.06 * max(0.0, robot_x)
                    env.sim.set_camera_view(
                        eye=(camera_x + 1.4, -8.8, 3.3),
                        target=(camera_x + 2.4, 0.0, 0.72),
                    )
                    rgb = env.render(recompute=True)
                    if rgb is None or rgb.size == 0:
                        continue
                    route_index = int(env._route_index[0].item())
                    maximum_route_index = max(maximum_route_index, route_index)
                    ball_speed = float(torch.linalg.norm(env._ball.data.root_lin_vel_w[0]).item())
                    max_ball_speed = max(max_ball_speed, ball_speed)
                    status, result = current_status(env)
                    frame = overlay_hud(
                        rgb,
                        iteration=args.iteration,
                        phase=args.phase,
                        transitions=transitions,
                        ball_speed=ball_speed,
                        route_index=route_index,
                        active_poles=env.active_poles,
                        status=status,
                    )
                    writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                    frames += 1

                    with torch.inference_mode():
                        actions = policy(obs)
                        obs, _, dones, _ = wrapped.step(actions)
                    if bool(dones[0].item()):
                        break
            finally:
                writer.release()

            metadata = {
                "checkpoint": str(checkpoint),
                "output": str(output.resolve()),
                "attempt": attempt,
                "iteration": args.iteration,
                "phase": args.phase,
                "stage": args.stage,
                "stage_name": STAGE_NAMES[args.stage],
                "seed": args.seed,
                "transitions": transitions,
                "frames": frames,
                "fps": FPS,
                "duration_seconds": frames / FPS,
                "result": result,
                "maximum_route_index": maximum_route_index,
                "max_ball_speed_mps": max_ball_speed,
            }
            output.with_suffix(".json").write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            temporary_video.replace(output)
            print(json.dumps(metadata, ensure_ascii=False, indent=2))
    finally:
        wrapped.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
