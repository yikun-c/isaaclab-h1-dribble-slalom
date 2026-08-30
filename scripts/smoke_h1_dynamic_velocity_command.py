"""Verify that the published H1 policy accepts changing velocity commands at runtime.

This is deliberately a low-level interface smoke, not a maze-navigation claim.
It exercises the exact command buffer that a later macro-action executor will
write, records all applied commands, and rejects non-finite robot state.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from isaaclab.app import AppLauncher


PROJECT_ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description="Smoke-test runtime H1 velocity command updates.")
parser.add_argument("--ticks-per-phase", type=int, default=24)
parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "artifacts/h1/dynamic_velocity_command_v2.json")
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

    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
    from isaaclab_tasks.manager_based.locomotion.velocity.config.h1.agents.rsl_rl_ppo_cfg import H1RoughPPORunnerCfg
    from isaaclab_tasks.manager_based.locomotion.velocity.config.h1.rough_env_cfg import H1RoughEnvCfg_PLAY

    if args.ticks_per_phase <= 0:
        raise ValueError("ticks-per-phase must be positive")
    checkpoint = get_published_pretrained_checkpoint("rsl_rl", "Isaac-Velocity-Rough-H1-v0")
    if checkpoint is None:
        raise RuntimeError("official H1 checkpoint unavailable")
    cfg = H1RoughEnvCfg_PLAY()
    cfg.scene.num_envs = 1
    cfg.scene.clone_in_fabric = False
    cfg.curriculum = None
    cfg.sim.device = args.device or "cuda:0"
    cfg.episode_length_s = 30.0
    cfg.commands.base_velocity.resampling_time_range = (1000.0, 1000.0)
    cfg.commands.base_velocity.rel_standing_envs = 0.0
    cfg.commands.base_velocity.rel_heading_envs = 0.0
    cfg.commands.base_velocity.heading_command = False
    env = ManagerBasedRLEnv(cfg)
    wrapped = RslRlVecEnvWrapper(env)
    phases = (("forward", (0.35, 0.0, 0.0)), ("walking_positive_yaw", (0.12, 0.0, 0.55)), ("stand", (0.0, 0.0, 0.0)))
    records: list[dict] = []
    try:
        runner = OnPolicyRunner(wrapped, H1RoughPPORunnerCfg().to_dict(), log_dir=None, device=env.device)
        runner.load(checkpoint)
        policy = runner.get_inference_policy(device=env.device)
        observations, _ = wrapped.reset()
        term = env.command_manager.get_term("base_velocity")
        for phase, values in phases:
            target = torch.tensor(values, device=env.device, dtype=term.vel_command_b.dtype).unsqueeze(0)
            start_position = env.scene["robot"].data.root_pos_w[0].detach().cpu().tolist()
            start_heading = float(env.scene["robot"].data.heading_w[0].item())
            for _ in range(args.ticks_per_phase):
                # CommandManager may update each environment step; write immediately
                # before policy inference so this is the measured runtime interface.
                term.vel_command_b[:] = target
                with torch.inference_mode():
                    actions = policy(observations)
                    observations, _, _, _ = wrapped.step(actions)
                if not torch.isfinite(env.scene["robot"].data.root_pos_w).all():
                    raise RuntimeError(f"non-finite robot state during {phase}")
            applied = term.vel_command_b[0].detach().cpu().tolist()
            end_position = env.scene["robot"].data.root_pos_w[0].detach().cpu().tolist()
            end_heading = float(env.scene["robot"].data.heading_w[0].item())
            if any(abs(actual - expected) > 1e-6 for actual, expected in zip(applied, values)):
                raise RuntimeError(f"runtime command mismatch for {phase}: expected={values}, actual={applied}")
            records.append({"phase": phase, "target": list(values), "applied": applied, "start_pos_w": start_position, "end_pos_w": end_position, "start_heading_w": start_heading, "end_heading_w": end_heading, "heading_delta_rad": end_heading - start_heading})
        payload = {
            "result": "H1_DYNAMIC_VELOCITY_COMMAND_SMOKE_OK",
            "checkpoint": str(checkpoint),
            "ticks_per_phase": args.ticks_per_phase,
            "records": records,
            "truth_label": "runtime low-level velocity-command interface only; no maze planner connected",
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
