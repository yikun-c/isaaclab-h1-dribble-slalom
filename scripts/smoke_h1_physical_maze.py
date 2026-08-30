"""Spawn one collidable, volumetric maze around H1 and verify Isaac physics initialization."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

parser = argparse.ArgumentParser(description="Smoke-test H1 plus collidable physical maze walls.")
parser.add_argument("--seed", type=int, default=2026)
parser.add_argument("--cell-size", type=float, default=1.8)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app


def main() -> None:
    import torch

    import isaaclab.sim as sim_utils
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.sim.utils.stage import get_current_stage
    from isaaclab_tasks.manager_based.locomotion.velocity.config.h1.flat_env_cfg import H1FlatEnvCfg_PLAY

    from maze_agent import build_task, maze_wall_specs

    task = build_task(9, 9, args.seed)
    walls = maze_wall_specs(task, cell_size=args.cell_size)
    for spec in walls:
        wall_cfg = sim_utils.CuboidCfg(
            size=spec.size,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.22, 0.35, 0.48), roughness=0.65),
        )
        wall_cfg.func(f"/World/Maze/{spec.name}", wall_cfg, translation=spec.translation)
    cfg = H1FlatEnvCfg_PLAY()
    cfg.scene.num_envs = 1
    cfg.sim.device = args.device or "cuda:0"
    env = ManagerBasedRLEnv(cfg)
    try:
        observations, _ = env.reset()
        actions = torch.zeros(env.action_space.shape, device=env.device)
        observations, rewards, terminated, truncated, _ = env.step(actions)
        stage = get_current_stage()
        existing = sum(stage.GetPrimAtPath(f"/World/Maze/{spec.name}").IsValid() for spec in walls)
        assert torch.isfinite(rewards).all()
        assert existing == len(walls)
        print(
            f"H1_PHYSICAL_MAZE_SMOKE_OK seed={args.seed} walls={len(walls)} "
            f"stage_walls={existing} action_shape={tuple(env.action_space.shape)}",
            flush=True,
        )
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
