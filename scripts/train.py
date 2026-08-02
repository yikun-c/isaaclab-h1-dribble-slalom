from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from isaaclab.app import AppLauncher


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

parser = argparse.ArgumentParser(description="Train H1 to dribble through a slalom and finish with a shot.")
parser.add_argument("--num-envs", type=int, default=8192)
parser.add_argument("--max-iterations", type=int, default=1600)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--run-name", type=str, default="stage0_one_pole")
parser.add_argument("--resume", type=Path)
parser.add_argument("--curriculum-step-offset", type=int, default=0)
parser.add_argument("--forced-stage", type=int, choices=(0, 1, 2, 3))
parser.add_argument("--start-route-index", type=int, choices=(0, 1, 2, 3))
parser.add_argument("--start-route-fraction", type=float, default=1.0)
parser.add_argument("--action-noise-std", type=float)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab.utils.io import dump_yaml
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
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    env_cfg = DribbleSlalomEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = args.seed
    env_cfg.sim.device = args.device or "cuda:0"
    env_cfg.curriculum_step_offset = args.curriculum_step_offset
    env_cfg.forced_stage = args.forced_stage
    env_cfg.start_route_index = args.start_route_index
    env_cfg.start_route_fraction = args.start_route_fraction

    agent_cfg = DribbleSlalomPPORunnerCfg()
    agent_cfg.max_iterations = args.max_iterations
    agent_cfg.seed = args.seed
    agent_cfg.run_name = args.run_name
    agent_cfg.device = args.device or "cuda:0"

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = PROJECT_ROOT / "runs" / agent_cfg.experiment_name / f"{timestamp}_{args.run_name}"
    log_dir.mkdir(parents=True, exist_ok=False)
    status_path = log_dir / "run_status.json"
    write_json_atomic(
        status_path,
        {
            "status": "initializing",
            "num_envs": args.num_envs,
            "max_iterations": args.max_iterations,
            "seed": args.seed,
            "device": agent_cfg.device,
            "curriculum_step_offset": args.curriculum_step_offset,
            "forced_stage": args.forced_stage,
            "start_route_index": args.start_route_index,
            "start_route_fraction": args.start_route_fraction,
            "started_at": datetime.now().isoformat(timespec="seconds"),
        },
    )
    env_cfg.log_dir = str(log_dir)
    print(f"[INFO] run_dir={log_dir}", flush=True)
    dump_yaml(str(log_dir / "env.yaml"), env_cfg)
    dump_yaml(str(log_dir / "agent.yaml"), agent_cfg)

    print(f"[INFO] configs_saved=1", flush=True)
    print(f"[INFO] gpu={torch.cuda.get_device_name(0)} envs={args.num_envs} seed={args.seed}", flush=True)
    env = DribbleSlalomEnv(env_cfg)
    print("[INFO] environment_ready=1", flush=True)
    wrapped_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(wrapped_env, agent_cfg.to_dict(), log_dir=str(log_dir), device=agent_cfg.device)
    print("[INFO] runner_ready=1", flush=True)
    if args.resume:
        runner.load(str(args.resume.resolve()))
    if args.action_noise_std is not None:
        with torch.no_grad():
            runner.alg.policy.std.fill_(args.action_noise_std)
        print(f"[INFO] action_noise_std={args.action_noise_std}", flush=True)

    started = time.time()
    write_json_atomic(
        status_path,
        {
            "status": "training",
            "num_envs": args.num_envs,
            "max_iterations": args.max_iterations,
            "seed": args.seed,
            "device": agent_cfg.device,
            "curriculum_step_offset": args.curriculum_step_offset,
            "forced_stage": args.forced_stage,
            "start_route_index": args.start_route_index,
            "start_route_fraction": args.start_route_fraction,
            "started_at": datetime.now().isoformat(timespec="seconds"),
        },
    )
    try:
        runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    except BaseException as exc:
        write_json_atomic(
            status_path,
            {
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "elapsed_seconds": round(time.time() - started, 3),
            },
        )
        raise
    else:
        elapsed = time.time() - started
        write_json_atomic(
            status_path,
            {
                "status": "complete",
                "num_envs": args.num_envs,
                "iterations": agent_cfg.max_iterations,
                "total_transitions": args.num_envs * agent_cfg.num_steps_per_env * agent_cfg.max_iterations,
                "elapsed_seconds": round(elapsed, 3),
                "completed_at": datetime.now().isoformat(timespec="seconds"),
            },
        )
        print(f"[INFO] training_complete=1 seconds={elapsed:.2f}", flush=True)
    finally:
        wrapped_env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
