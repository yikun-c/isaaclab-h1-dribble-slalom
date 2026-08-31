"""Record a truthful physical-maze H1 locomotion setup clip.

The clip is explicitly labelled as a pretrained low-level locomotion setup, not
as LLM-controlled navigation. It validates maze geometry, camera framing and the
H1 command interface before the high-level agent is connected.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# The pip Isaac sensor extension expects the legacy oneTBB runtime by the
# exact name tbb.dll. Keep both dependency directories in this child process
# only; do not alter the user's global PATH.
if os.name == "nt":
    _dll_dirs = (
        Path(r"D:\IsaacLab\.venv\Library\bin"),
        Path(r"D:\IsaacLab\.venv\Lib\site-packages\isaacsim\kit\dev\libs\sensors\generic_model_output\bin"),
    )
    _dll_handles = [os.add_dll_directory(str(path)) for path in _dll_dirs if path.is_dir()]
    os.environ["PATH"] = os.pathsep.join(str(path) for path in _dll_dirs if path.is_dir()) + os.pathsep + os.environ["PATH"]

from isaaclab.app import AppLauncher


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

parser = argparse.ArgumentParser(description="Record H1 in a physical maze setup scene.")
parser.add_argument("--seed", type=int, default=2026)
parser.add_argument("--steps", type=int, default=180)
parser.add_argument("--forward-speed", type=float, default=0.4)
parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "artifacts" / "video" / "h1_physical_maze_setup_v3.mp4")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app


def main() -> None:
    import cv2
    import numpy as np
    import torch
    from PIL import Image, ImageDraw, ImageFont
    from rsl_rl.runners import OnPolicyRunner

    import isaaclab.sim as sim_utils
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
    from isaaclab_tasks.manager_based.locomotion.velocity.config.h1.agents.rsl_rl_ppo_cfg import H1RoughPPORunnerCfg
    from isaaclab_tasks.manager_based.locomotion.velocity.config.h1.rough_env_cfg import H1RoughEnvCfg_PLAY

    from maze_agent import build_task, maze_wall_specs

    task = build_task(9, 9, args.seed)
    walls = maze_wall_specs(task)
    for spec in walls:
        cfg = sim_utils.CuboidCfg(
            size=spec.size,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.20, 0.40, 0.62), roughness=0.64),
        )
        cfg.func(f"/World/Maze/{spec.name}", cfg, translation=spec.translation)
    checkpoint = get_published_pretrained_checkpoint("rsl_rl", "Isaac-Velocity-Rough-H1-v0")
    if checkpoint is None:
        raise RuntimeError("official H1 checkpoint unavailable")
    env_cfg = H1RoughEnvCfg_PLAY()
    env_cfg.scene.num_envs = 1
    env_cfg.scene.clone_in_fabric = False
    env_cfg.episode_length_s = max(20.0, args.steps * env_cfg.sim.dt * env_cfg.decimation * 2.0)
    env_cfg.curriculum = None
    env_cfg.sim.device = args.device or "cuda:0"
    env_cfg.commands.base_velocity.ranges.lin_vel_x = (args.forward_speed, args.forward_speed)
    env_cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
    env_cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
    env_cfg.commands.base_velocity.resampling_time_range = (1000.0, 1000.0)
    env_cfg.commands.base_velocity.rel_standing_envs = 0.0
    env_cfg.commands.base_velocity.rel_heading_envs = 0.0
    env_cfg.commands.base_velocity.heading_command = False
    env_cfg.events.reset_base.params = {
        "pose_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)},
        "velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0), "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0)},
    }
    env = ManagerBasedRLEnv(env_cfg, render_mode="rgb_array")
    wrapped = RslRlVecEnvWrapper(env)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".raw.mp4")
    font_path = Path(r"C:\Windows\Fonts\msyh.ttc")
    if not font_path.is_file():
        raise FileNotFoundError(font_path)
    title_font = ImageFont.truetype(str(font_path), 23)
    detail_font = ImageFont.truetype(str(font_path), 16)
    try:
        runner = OnPolicyRunner(wrapped, H1RoughPPORunnerCfg().to_dict(), log_dir=None, device=env.device)
        runner.load(checkpoint)
        policy = runner.get_inference_policy(device=env.device)
        observations, _ = wrapped.reset()
        # D3D12/Hydra can return black buffers while the RGB render product is
        # still warming up.  Render before recording and reject those buffers
        # instead of allowing an apparently valid but unusable MP4.
        for _ in range(60):
            env.sim.render()
        resolution = (1280, 720)
        writer = cv2.VideoWriter(str(temporary), cv2.VideoWriter_fourcc(*"mp4v"), 30, resolution)
        if not writer.isOpened():
            raise RuntimeError("could not open video writer")
        frames = 0
        try:
            for step_index in range(args.steps):
                with torch.inference_mode():
                    actions = policy(observations)
                    observations, _, _, _ = wrapped.step(actions)
                root = env.scene["robot"].data.root_pos_w[0]
                root_x, root_y = float(root[0].item()), float(root[1].item())
                env.sim.set_camera_view(eye=(root_x - 5.2, root_y - 7.6, 6.2), target=(root_x + 3.2, root_y + 3.2, 0.75))
                env.sim.render()
                rgb = env.render(recompute=False)
                if rgb is not None and rgb.size:
                    image = Image.fromarray(rgb)
                    if float(np.asarray(image).mean()) < 8.0:
                        continue
                    if image.size != resolution:
                        image = image.resize(resolution, Image.Resampling.LANCZOS)
                    draw = ImageDraw.Draw(image, "RGBA")
                    draw.rectangle((0, 0, image.width, 76), fill=(18, 24, 30, 225))
                    draw.text((20, 8), "实体迷宫 · H1 预训练低层行走", font=title_font, fill=(246, 248, 250, 255))
                    draw.text(
                        (20, 43),
                        f"物理墙体 {len(walls)} 段  |  指令前进 {args.forward_speed:.1f} m/s  |  LLM高层规划：待接入",
                        font=detail_font,
                        fill=(169, 219, 235, 255),
                    )
                    writer.write(cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR))
                    frames += 1
        finally:
            writer.release()
        temporary.replace(args.output)
        metadata = {
            "asset_type": "physical_maze_low_level_setup",
            "truth_label": "pretrained H1 low-level locomotion only; no LLM planning connected",
            "output": str(args.output.resolve()),
            "seed": args.seed,
            "walls": len(walls),
            "steps": args.steps,
            "frames": frames,
            "fps": 30,
            "duration_seconds": frames / 30.0,
            "checkpoint": str(checkpoint),
        }
        args.output.with_suffix(".json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(metadata, ensure_ascii=False, sort_keys=True), flush=True)
    finally:
        wrapped.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
