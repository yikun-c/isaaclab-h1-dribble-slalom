"""Bounded Isaac Lab smoke test for the official H1 flat velocity environment."""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Step the official H1 flat velocity environment with zero actions.")
parser.add_argument("--num-envs", type=int, default=1)
parser.add_argument("--steps", type=int, default=5)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app


def main() -> None:
    import torch

    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab_tasks.manager_based.locomotion.velocity.config.h1.flat_env_cfg import H1FlatEnvCfg_PLAY

    cfg = H1FlatEnvCfg_PLAY()
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = args.device or "cuda:0"
    print("H1_VELOCITY_SMOKE_STAGE=config_ready", flush=True)
    env = ManagerBasedRLEnv(cfg)
    try:
        print("H1_VELOCITY_SMOKE_STAGE=env_created", flush=True)
        observations, _ = env.reset()
        print("H1_VELOCITY_SMOKE_STAGE=reset_complete", flush=True)
        for _ in range(args.steps):
            actions = torch.zeros(env.action_space.shape, device=env.device)
            observations, rewards, terminated, truncated, _ = env.step(actions)
            assert torch.isfinite(rewards).all()
            assert terminated.dtype == torch.bool
            assert truncated.dtype == torch.bool
        print("H1_VELOCITY_SMOKE_STAGE=steps_complete", flush=True)
        print(
            f"H1_VELOCITY_SMOKE_OK envs={args.num_envs} steps={args.steps} "
            f"action_shape={tuple(env.action_space.shape)} device={env.device} "
            f"observation_keys={sorted(observations.keys())}",
            flush=True,
        )
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
