"""Verify the official pretrained H1 velocity policy under a fixed forward command."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Smoke-test the official pretrained H1 velocity policy.")
parser.add_argument("--steps", type=int, default=60)
parser.add_argument("--forward-speed", type=float, default=0.4)
parser.add_argument("--turn-rate", type=float, default=0.0)
parser.add_argument("--use-default-command", action="store_true")
parser.add_argument("--heartbeat", type=Path, default=Path("artifacts/h1/pretrained_locomotion_heartbeat_v1.json"))
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app


def write_heartbeat(payload: dict) -> None:
    args.heartbeat.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.heartbeat.with_suffix(args.heartbeat.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, args.heartbeat)


def main() -> None:
    import torch
    from rsl_rl.runners import OnPolicyRunner

    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
    from isaaclab_tasks.manager_based.locomotion.velocity.config.h1.agents.rsl_rl_ppo_cfg import H1RoughPPORunnerCfg
    from isaaclab_tasks.manager_based.locomotion.velocity.config.h1.rough_env_cfg import H1RoughEnvCfg_PLAY

    task_name = "Isaac-Velocity-Rough-H1-v0"
    checkpoint = get_published_pretrained_checkpoint("rsl_rl", task_name)
    if checkpoint is None:
        raise RuntimeError("official H1 pretrained checkpoint is unavailable")
    print("H1_PRETRAINED_SMOKE_STAGE=checkpoint_ready", flush=True)
    cfg = H1RoughEnvCfg_PLAY()
    cfg.scene.num_envs = 1
    cfg.episode_length_s = 60.0
    cfg.curriculum = None
    cfg.sim.device = args.device or "cuda:0"
    if not args.use_default_command:
        cfg.commands.base_velocity.ranges.lin_vel_x = (args.forward_speed, args.forward_speed)
        cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        cfg.commands.base_velocity.ranges.ang_vel_z = (args.turn_rate, args.turn_rate)
        cfg.commands.base_velocity.resampling_time_range = (1000.0, 1000.0)
        cfg.commands.base_velocity.rel_standing_envs = 0.0
        cfg.commands.base_velocity.rel_heading_envs = 0.0
        cfg.commands.base_velocity.heading_command = False
    wrapped = RslRlVecEnvWrapper(ManagerBasedRLEnv(cfg=cfg))
    try:
        print("H1_PRETRAINED_SMOKE_STAGE=env_created", flush=True)
        runner_cfg = H1RoughPPORunnerCfg()
        runner = OnPolicyRunner(wrapped, runner_cfg.to_dict(), log_dir=None, device=wrapped.unwrapped.device)
        print("H1_PRETRAINED_SMOKE_STAGE=runner_created", flush=True)
        runner.load(checkpoint)
        policy = runner.get_inference_policy(device=wrapped.unwrapped.device)
        print("H1_PRETRAINED_SMOKE_STAGE=policy_loaded", flush=True)
        observations, _ = wrapped.reset()
        print("H1_PRETRAINED_SMOKE_STAGE=reset_complete", flush=True)
        write_heartbeat({"stage": "reset_complete", "completed_steps": 0})
        completed_steps = 0
        for step_index in range(args.steps):
            if step_index == 0:
                print("H1_PRETRAINED_SMOKE_STAGE=before_policy", flush=True)
            with torch.inference_mode():
                action = policy(observations)
                if step_index == 0:
                    print(f"H1_PRETRAINED_SMOKE_STAGE=policy_action_shape_{tuple(action.shape)}", flush=True)
                observations, _, _, _ = wrapped.step(action)
            if (step_index + 1) % 5 == 0 or step_index + 1 == args.steps:
                print(f"H1_PRETRAINED_SMOKE_STAGE=step_{step_index + 1}_complete", flush=True)
            completed_steps += 1
            write_heartbeat({"stage": "stepping", "completed_steps": completed_steps})
        if completed_steps != args.steps:
            raise RuntimeError(f"simulation app stopped after {completed_steps}/{args.steps} policy steps")
        write_heartbeat({"stage": "completed", "completed_steps": completed_steps})
        print("H1_PRETRAINED_SMOKE_STAGE=steps_complete", flush=True)
        print(
            f"H1_PRETRAINED_LOCOMOTION_SMOKE_OK steps={args.steps} forward_speed={args.forward_speed} "
            f"checkpoint={checkpoint}",
            flush=True,
        )
    finally:
        wrapped.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
