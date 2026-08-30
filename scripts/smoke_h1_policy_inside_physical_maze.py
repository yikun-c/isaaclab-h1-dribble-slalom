"""Verify H1's published locomotion policy starts inside the collidable maze.

The previous wall smoke only proved that wall prims existed.  This smoke removes
the rough-terrain origin mismatch by retaining the rough-policy observation
shape while switching its terrain source to a plane, so the one environment's
origin is the logical maze start at world (0, 0).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
parser = argparse.ArgumentParser(description="Smoke-test pretrained H1 inside collidable maze geometry.")
parser.add_argument("--seed", type=int, default=2026)
parser.add_argument("--steps", type=int, default=48)
parser.add_argument("--forward-speed", type=float, default=0.30)
parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "artifacts/h1/policy_inside_physical_maze_v1.json")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
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


def main() -> None:
    import torch
    from rsl_rl.runners import OnPolicyRunner

    import isaaclab.sim as sim_utils
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.sim.utils.stage import get_current_stage
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
    from isaaclab_tasks.manager_based.locomotion.velocity.config.h1.agents.rsl_rl_ppo_cfg import H1RoughPPORunnerCfg
    from isaaclab_tasks.manager_based.locomotion.velocity.config.h1.rough_env_cfg import H1RoughEnvCfg_PLAY

    from maze_agent import build_task, maze_wall_specs

    if args.steps <= 0:
        raise ValueError("steps must be positive")
    task = build_task(9, 9, args.seed)
    # The selected default seed has east as the only start exit, matching yaw 0.
    if not task.layout.can_move(task.start, task.layout.accessible_neighbors(task.start)[0][0]):
        raise RuntimeError("invalid start topology")
    walls = maze_wall_specs(task)
    for spec in walls:
        wall_cfg = sim_utils.CuboidCfg(
            size=spec.size,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.20, 0.40, 0.62), roughness=0.64),
        )
        wall_cfg.func(f"/World/Maze/{spec.name}", wall_cfg, translation=spec.translation)
    checkpoint = get_published_pretrained_checkpoint("rsl_rl", "Isaac-Velocity-Rough-H1-v0")
    if checkpoint is None:
        raise RuntimeError("official H1 checkpoint unavailable")
    cfg = H1RoughEnvCfg_PLAY()
    cfg.scene.num_envs = 1
    cfg.scene.clone_in_fabric = False
    # Preserve the rough-policy height-scan observation term, but make origin
    # deterministic instead of sampling a distant rough-terrain tile.
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None
    cfg.curriculum = None
    cfg.sim.device = args.device or "cuda:0"
    cfg.episode_length_s = 30.0
    cfg.commands.base_velocity.resampling_time_range = (1000.0, 1000.0)
    cfg.commands.base_velocity.rel_standing_envs = 0.0
    cfg.commands.base_velocity.rel_heading_envs = 0.0
    cfg.commands.base_velocity.heading_command = False
    cfg.events.reset_base.params = {
        "pose_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)},
        "velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0), "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0)},
    }
    env = ManagerBasedRLEnv(cfg)
    wrapped = RslRlVecEnvWrapper(env)
    try:
        runner = OnPolicyRunner(wrapped, H1RoughPPORunnerCfg().to_dict(), log_dir=None, device=env.device)
        runner.load(checkpoint)
        policy = runner.get_inference_policy(device=env.device)
        observations, _ = wrapped.reset()
        stage = get_current_stage()
        spawned_walls = sum(stage.GetPrimAtPath(f"/World/Maze/{spec.name}").IsValid() for spec in walls)
        root_before = env.scene["robot"].data.root_pos_w[0].detach().cpu()
        if spawned_walls != len(walls):
            raise RuntimeError(f"wall stage mismatch: {spawned_walls}/{len(walls)}")
        if max(abs(float(root_before[0])), abs(float(root_before[1]))) > 0.65:
            raise RuntimeError(f"robot did not reset at maze start: {root_before.tolist()}")
        term = env.command_manager.get_term("base_velocity")
        target = torch.tensor([[args.forward_speed, 0.0, 0.0]], device=env.device, dtype=term.vel_command_b.dtype)
        for _ in range(args.steps):
            term.vel_command_b[:] = target
            with torch.inference_mode():
                actions = policy(observations)
                observations, _, _, _ = wrapped.step(actions)
            if not torch.isfinite(env.scene["robot"].data.root_pos_w).all():
                raise RuntimeError("non-finite H1 root state")
        root_after = env.scene["robot"].data.root_pos_w[0].detach().cpu()
        applied = term.vel_command_b[0].detach().cpu().tolist()
        payload = {
            "result": "H1_POLICY_INSIDE_PHYSICAL_MAZE_SMOKE_OK",
            "truth_label": "pretrained low-level forward command inside collidable maze start; no LLM planner connected",
            "maze_seed": args.seed,
            "walls": len(walls),
            "spawned_walls": spawned_walls,
            "steps": args.steps,
            "command": applied,
            "root_before_w": root_before.tolist(),
            "root_after_w": root_after.tolist(),
            "checkpoint": str(checkpoint),
        }
        write_json_atomic(args.output, payload)
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True), flush=True)
    finally:
        wrapped.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
