"""Run one real Qwen3.5 tool call through the physical H1 velocity interface.

It deliberately verifies only one decision at the deterministic maze start. It
does not represent an end-to-end maze-navigation success: metric/pose feedback,
macro completion, and multi-decision planning still require their own gate.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "runtime" / "qwen35_transformers"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Smoke-test one Qwen3.5 planner decision through H1 velocity execution.")
parser.add_argument("--seed", type=int, default=2026)
parser.add_argument("--adapter-dir", type=Path, default=PROJECT_ROOT / "runs/qwen35_sft/2026-08-30_21-50-28_qwen3_5_2b_maze_memory_sft_dev200_v1/adapter")
parser.add_argument("--model-dir", type=Path, default=PROJECT_ROOT / "models/qwen3_5_2b")
parser.add_argument("--ticks", type=int, default=48)
parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "artifacts/h1/qwen35_h1_bridge_one_decision_v1.json")
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


def planner_input(task, state, memory):
    from maze_agent import observe

    local = observe(task, state)
    return {
        "instruction": task.instruction,
        "local_perception": {
            "front_open": local.front_open,
            "left_open": local.left_open,
            "right_open": local.right_open,
            "rear_open": local.rear_open,
            "current_landmarks": list(local.current_landmarks),
            "adjacent_landmarks": list(local.adjacent_landmarks),
        },
        "state": {"heading": state.heading.value, "checkpoint_complete": state.checkpoint_complete, "last_result": state.last_result},
        "memory": memory.compact_summary(state),
    }


def velocity_for_action(action):
    from maze_agent import Action

    mapping = {
        Action.MOVE_FORWARD: (0.30, 0.0, 0.0),
        Action.TURN_LEFT: (0.0, 0.0, 0.55),
        Action.TURN_RIGHT: (0.0, 0.0, -0.55),
        Action.BACKTRACK: (-0.20, 0.0, 0.0),
        Action.STOP: (0.0, 0.0, 0.0),
    }
    return mapping[action]


def main() -> None:
    import torch
    from peft import PeftModel
    from rsl_rl.runners import OnPolicyRunner
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    import isaaclab.sim as sim_utils
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.sim.utils.stage import get_current_stage
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
    from isaaclab_tasks.manager_based.locomotion.velocity.config.h1.agents.rsl_rl_ppo_cfg import H1RoughPPORunnerCfg
    from isaaclab_tasks.manager_based.locomotion.velocity.config.h1.rough_env_cfg import H1RoughEnvCfg_PLAY

    from maze_agent import TopologicalMemory, build_task, parse_planner_response
    from maze_agent.core import reset
    from maze_agent.physical_maze import maze_wall_specs
    from maze_agent.sft import SYSTEM_PROMPT

    if args.ticks <= 0 or not args.adapter_dir.is_dir() or not args.model_dir.is_dir():
        raise ValueError("ticks must be positive and model/adapter paths must exist")
    task = build_task(9, 9, args.seed)
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
    model = base = None
    try:
        runner = OnPolicyRunner(wrapped, H1RoughPPORunnerCfg().to_dict(), log_dir=None, device=env.device)
        runner.load(checkpoint)
        locomotion = runner.get_inference_policy(device=env.device)
        observations, _ = wrapped.reset()
        root_before = env.scene["robot"].data.root_pos_w[0].detach().cpu().tolist()
        stage = get_current_stage()
        spawned_walls = sum(stage.GetPrimAtPath(f"/World/Maze/{spec.name}").IsValid() for spec in walls)
        if spawned_walls != len(walls) or max(abs(root_before[0]), abs(root_before[1])) > 0.65:
            raise RuntimeError("H1 was not initialized at the physical maze start")

        state = reset(task)
        memory = TopologicalMemory()
        memory.record_observation(task, state)
        payload = planner_input(task, state, memory)
        processor = AutoProcessor.from_pretrained(args.adapter_dir, local_files_only=True)
        base = AutoModelForMultimodalLM.from_pretrained(args.model_dir, local_files_only=True, dtype=torch.bfloat16).to("cuda").eval()
        model = PeftModel.from_pretrained(base, args.adapter_dir, local_files_only=True).eval()
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": [{"type": "text", "text": json.dumps(payload, ensure_ascii=False)}]}]
        prompt = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=prompt, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            started = time.perf_counter()
            output_ids = model.generate(**inputs, max_new_tokens=64, do_sample=False)
            latency_ms = (time.perf_counter() - started) * 1000.0
        raw = processor.decode(output_ids[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True)
        decision = parse_planner_response(raw)
        target_values = velocity_for_action(decision.action)
        term = env.command_manager.get_term("base_velocity")
        target = torch.tensor([target_values], device=env.device, dtype=term.vel_command_b.dtype)
        for _ in range(args.ticks):
            term.vel_command_b[:] = target
            with torch.inference_mode():
                joint_actions = locomotion(observations)
                observations, _, _, _ = wrapped.step(joint_actions)
            if not torch.isfinite(env.scene["robot"].data.root_pos_w).all():
                raise RuntimeError("non-finite H1 root state after LLM command")
        root_after = env.scene["robot"].data.root_pos_w[0].detach().cpu().tolist()
        applied = term.vel_command_b[0].detach().cpu().tolist()
        report = {
            "result": "QWEN35_H1_BRIDGE_ONE_DECISION_OK",
            "truth_label": "one real Qwen3.5 planner tool call mapped to a pretrained H1 velocity command inside collidable maze geometry; not an end-to-end navigation success",
            "maze_seed": args.seed,
            "planner_input": payload,
            "planner_raw_response": raw,
            "planner_action": decision.action.value,
            "planner_valid": decision.valid,
            "planner_fallback_reason": decision.fallback_reason,
            "planner_latency_ms": latency_ms,
            "velocity_command": applied,
            "ticks": args.ticks,
            "root_before_w": root_before,
            "root_after_w": root_after,
            "walls": len(walls),
            "spawned_walls": spawned_walls,
            "model_dir": str(args.model_dir.resolve()),
            "adapter_dir": str(args.adapter_dir.resolve()),
            "locomotion_checkpoint": str(checkpoint),
        }
        write_json_atomic(args.output, report)
        print(json.dumps(report, ensure_ascii=False, sort_keys=True), flush=True)
    finally:
        del model, base
        wrapped.close()
        import torch
        torch.cuda.empty_cache()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
