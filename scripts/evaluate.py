from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

parser = argparse.ArgumentParser(description="Evaluate complete deterministic dribble-slalom episodes.")
parser.add_argument("checkpoint", type=Path)
parser.add_argument("--num-envs", type=int, default=512)
parser.add_argument("--episodes", type=int, default=4096)
parser.add_argument("--stage", type=int, choices=(0, 1, 2, 3), default=3)
parser.add_argument("--seed", type=int, default=2026)
parser.add_argument("--output", type=Path)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from dribble_agent.dribble_env import DribbleSlalomEnv, DribbleSlalomEnvCfg
from dribble_agent.ppo_cfg import DribbleSlalomPPORunnerCfg


def write_json_atomic(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> None:
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    env_cfg = DribbleSlalomEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.forced_stage = args.stage
    env_cfg.seed = args.seed
    env_cfg.sim.device = args.device or "cuda:0"

    agent_cfg = DribbleSlalomPPORunnerCfg()
    agent_cfg.device = args.device or "cuda:0"
    env = DribbleSlalomEnv(env_cfg)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(str(checkpoint))
    policy = runner.get_inference_policy(device=env.device)

    obs = wrapped.get_observations()
    with torch.inference_mode():
        while int(env.completed_episodes.item()) < args.episodes:
            actions = policy(obs)
            obs, _, _, _ = wrapped.step(actions)

    completed = int(env.completed_episodes.item())
    active_poles = args.stage + 1
    result = {
        "checkpoint": str(checkpoint),
        "stage": args.stage,
        "active_poles": active_poles,
        "seed": args.seed,
        "requested_episodes": args.episodes,
        "completed_episodes": completed,
        "route_successes": int(env.completed_route_successes.item()),
        "route_success_rate": float(env.completed_route_successes.item() / completed),
        "goals": int(env.completed_goals.item()),
        "goal_rate": float(env.completed_goals.item() / completed),
        "gate_passes": int(env.completed_gate_passes.item()),
        "mean_gate_passes": float(env.completed_gate_passes.item() / completed),
        "maximum_gate_passes": active_poles,
        "falls": int(env.completed_falls.item()),
        "fall_rate": float(env.completed_falls.item() / completed),
        "wrong_routes": int(env.completed_wrong_routes.item()),
        "wrong_route_rate": float(env.completed_wrong_routes.item() / completed),
    }
    output = args.output or checkpoint.with_name(f"{checkpoint.stem}_stage{args.stage}_evaluation.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(output, result)
    print(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True))
    wrapped.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
