"""Check whether ordinary headless env.render works without RTX sensor extensions."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
parser = argparse.ArgumentParser(description="Smoke-test non-Replicator offscreen H1 maze rendering.")
parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "artifacts/h1/offscreen_render_smoke_v1.json")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
# Deliberately do not set args.enable_cameras: this distinguishes ordinary
# environment rendering from the failing RTX/Replicator sensor stack.
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app


def main() -> None:
    import numpy as np
    import torch

    import isaaclab.sim as sim_utils
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab_tasks.manager_based.locomotion.velocity.config.h1.flat_env_cfg import H1FlatEnvCfg_PLAY

    from maze_agent import build_task, maze_wall_specs

    task = build_task(9, 9, 2026)
    walls = maze_wall_specs(task, cell_size=3.6)
    for spec in walls:
        cfg = sim_utils.CuboidCfg(
            size=spec.size,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.20, 0.40, 0.62), roughness=0.64),
        )
        cfg.func(f"/World/Maze/{spec.name}", cfg, translation=spec.translation)
    env_cfg = H1FlatEnvCfg_PLAY()
    env_cfg.scene.num_envs = 1
    env_cfg.scene.clone_in_fabric = False
    env_cfg.scene.terrain.terrain_type = "plane"
    env_cfg.scene.terrain.terrain_generator = None
    env_cfg.sim.device = args.device or "cuda:0"
    env = ManagerBasedRLEnv(env_cfg, render_mode="rgb_array")
    try:
        env.reset()
        for _ in range(20):
            env.render(recompute=True)
        rgb = env.render(recompute=True)
        result = {
            "result": "H1_OFFSCREEN_RENDER_SMOKE_OK" if rgb is not None and rgb.size else "H1_OFFSCREEN_RENDER_UNAVAILABLE",
            "renderer": "ordinary_env_render_without_enable_cameras",
            "shape": list(rgb.shape) if rgb is not None else None,
            "mean_rgb": float(np.asarray(rgb).mean()) if rgb is not None and rgb.size else None,
            "non_black": bool(float(np.asarray(rgb).mean()) >= 8.0) if rgb is not None and rgb.size else False,
            "walls": len(walls),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary, args.output)
        print(json.dumps(result, ensure_ascii=False), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
