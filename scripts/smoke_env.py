from __future__ import annotations

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

parser = argparse.ArgumentParser(description="Smoke-test the H1 dribble-slalom environment.")
parser.add_argument("--num-envs", type=int, default=16)
parser.add_argument("--steps", type=int, default=120)
parser.add_argument("--stage", type=int, choices=(0, 1, 2, 3), default=3)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch

from dribble_agent.dribble_env import DribbleSlalomEnv, DribbleSlalomEnvCfg


def main() -> None:
    cfg = DribbleSlalomEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.forced_stage = args.stage
    cfg.sim.device = args.device or "cuda:0"
    env = DribbleSlalomEnv(cfg)
    obs, _ = env.reset()
    assert obs["policy"].shape == (args.num_envs, cfg.observation_space)

    for _ in range(args.steps):
        actions = torch.zeros((args.num_envs, cfg.action_space), device=env.device)
        obs, rewards, terminated, truncated, _ = env.step(actions)
        assert torch.isfinite(obs["policy"]).all()
        assert torch.isfinite(rewards).all()
        assert terminated.dtype == torch.bool
        assert truncated.dtype == torch.bool

    print(
        f"SMOKE_OK envs={args.num_envs} steps={args.steps} "
        f"obs={tuple(obs['policy'].shape)} device={env.device} stage={env.curriculum_stage} "
        f"active_poles={env.active_poles}"
    )
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
