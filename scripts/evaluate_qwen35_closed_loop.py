"""Closed-loop development evaluation for a Qwen3.5 causal-memory maze adapter."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "runtime" / "qwen35_transformers"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from maze_agent import TopologicalMemory, build_task, decision_event, guard_action, observe, parse_planner_response, sense_physical_maze, step
from maze_agent.core import reset
from maze_agent.sft import SYSTEM_PROMPT


def write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def build_input(task, state, memory: TopologicalMemory, use_physical_wall_rays: bool = False) -> dict:
    local = observe(task, state)
    ray_ranges = None
    if use_physical_wall_rays:
        ray_ranges = sense_physical_maze(task, state.position, state.heading)
        openings = ray_ranges.open_by_direction(minimum_clearance_m=1.0)
    else:
        openings = {"front": local.front_open, "left": local.left_open, "right": local.right_open, "rear": local.rear_open}
    return {
        "instruction": task.instruction,
        "local_perception": {
            "front_open": openings["front"],
            "left_open": openings["left"],
            "right_open": openings["right"],
            "rear_open": openings["rear"],
            "current_landmarks": list(local.current_landmarks),
            "adjacent_landmarks": list(local.adjacent_landmarks),
        },
        "state": {
            "heading": state.heading.value,
            "checkpoint_complete": state.checkpoint_complete,
            "last_result": state.last_result,
        },
        "memory": memory.compact_summary(state),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a Qwen3.5 LoRA adapter on closed-loop development mazes.")
    parser.add_argument("--model-dir", type=Path, default=PROJECT_ROOT / "models" / "qwen3_5_2b")
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "artifacts" / "maze" / "splits_v1.json")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max-decisions", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--execution-guard", action="store_true", help="Use the explicitly labelled local-memory executor guard.")
    parser.add_argument("--guard-revisit-threshold", type=int, default=2)
    parser.add_argument("--physical-wall-rays", action="store_true", help="Derive boolean planner openings from physical wall ray ranges.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    seeds = manifest["development_seeds"][: args.episodes]

    import torch
    from peft import PeftModel
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    processor = AutoProcessor.from_pretrained(args.adapter_dir, local_files_only=True)
    base = AutoModelForMultimodalLM.from_pretrained(
        args.model_dir, local_files_only=True, dtype=torch.bfloat16
    ).to("cuda").eval()
    model = PeftModel.from_pretrained(base, args.adapter_dir, local_files_only=True).eval()
    episodes: list[dict] = []
    total_started = time.perf_counter()
    with torch.inference_mode():
        for maze_seed in seeds:
            task = build_task(manifest["width"], manifest["height"], maze_seed)
            state = reset(task)
            memory = TopologicalMemory()
            events: list[dict] = []
            seen: set[tuple[tuple[int, int], str, bool]] = set()
            loops = 0
            guard_overrides = 0
            for _ in range(args.max_decisions):
                memory.record_observation(task, state)
                memory_before = memory.compact_summary(state)
                state_key = (state.position, state.heading.value, state.checkpoint_complete)
                loops += int(state_key in seen)
                seen.add(state_key)
                payload = build_input(task, state, memory, args.physical_wall_rays)
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": [{"type": "text", "text": json.dumps(payload, ensure_ascii=False)}]},
                ]
                prompt = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = processor(text=prompt, return_tensors="pt").to("cuda")
                started = time.perf_counter()
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                )
                latency_ms = (time.perf_counter() - started) * 1000
                raw = processor.decode(output_ids[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True)
                proposed = parse_planner_response(raw)
                decision = proposed
                guard_reason = None
                if args.execution_guard:
                    guarded = guard_action(task, state, memory, proposed.action, args.guard_revisit_threshold)
                    guard_reason = guarded.reason
                    guard_overrides += int(guarded.overridden)
                    if guarded.overridden:
                        decision = replace(
                            proposed,
                            action=guarded.action,
                            fallback_reason=f"memory_guard:{guarded.reason}",
                        )
                after = step(task, state, decision.action)
                event = decision_event(
                        task,
                        state,
                        decision,
                        after,
                        memory_before,
                        raw,
                        latency_ms=latency_ms,
                        token_count=None,
                    )
                if args.execution_guard:
                    event["planner"]["proposed_action"] = proposed.action.value
                    event["planner"]["guard_reason"] = guard_reason
                if args.physical_wall_rays:
                    ranges = sense_physical_maze(task, state.position, state.heading)
                    event["physical_wall_ranges_m"] = {
                        "front": ranges.front_m,
                        "left": ranges.left_m,
                        "right": ranges.right_m,
                        "rear": ranges.rear_m,
                    }
                events.append(event)
                memory.record_transition(task, state, decision.action, after)
                state = after
                if state.terminated:
                    break
            episodes.append(
                {
                    "maze_seed": maze_seed,
                    "task_id": task.task_id,
                    "success": state.success,
                    "terminal_reason": state.terminal_reason or "decision_budget_exhausted",
                    "decisions": len(events),
                    "collisions": state.collisions,
                    "loop_observations": loops,
                    "valid_outputs": sum(event["planner"]["valid"] for event in events),
                    "guard_overrides": guard_overrides,
                    "events": events,
                }
            )
            # Preserve completed episodes if a desktop-session interruption
            # occurs during a later generation call.  The final report remains
            # authoritative only when this sidecar says ``complete``.
            write_json_atomic(
                args.output.with_suffix(args.output.suffix + ".progress.json"),
                {
                    "status": "running",
                    "episodes_requested": len(seeds),
                    "episodes_completed": len(episodes),
                    "episodes_detail": episodes,
                },
            )
    elapsed = time.perf_counter() - total_started
    decisions = sum(episode["decisions"] for episode in episodes)
    report = {
        "evaluation_version": "qwen35-maze-closed-loop-v2",
        "scope": "development-only causal-memory closed-loop mazes; final splits were not loaded",
        "execution_interface": "LLM plus local-memory guard" if args.execution_guard else "LLM proposed action directly executed",
        "guard_revisit_threshold": args.guard_revisit_threshold if args.execution_guard else None,
        "perception_source": "physical_wall_ray_ranges" if args.physical_wall_rays else "grid_topology_adapter",
        "model_dir": str(args.model_dir.resolve()),
        "adapter_dir": str(args.adapter_dir.resolve()),
        "episodes": len(episodes),
        "successes": sum(episode["success"] for episode in episodes),
        "success_rate": sum(episode["success"] for episode in episodes) / len(episodes),
        "total_decisions": decisions,
        "valid_output_rate": sum(episode["valid_outputs"] for episode in episodes) / max(1, decisions),
        "mean_collisions": sum(episode["collisions"] for episode in episodes) / len(episodes),
        "mean_loop_observations": sum(episode["loop_observations"] for episode in episodes) / len(episodes),
        "total_guard_overrides": sum(episode["guard_overrides"] for episode in episodes),
        "elapsed_seconds": elapsed,
        "episodes_detail": episodes,
    }
    write_json_atomic(args.output, report)
    write_json_atomic(
        args.output.with_suffix(args.output.suffix + ".progress.json"),
        {
            "status": "complete",
            "episodes_requested": len(seeds),
            "episodes_completed": len(episodes),
            "report_path": str(args.output.resolve()),
        },
    )
    print(json.dumps({key: value for key, value in report.items() if key != "episodes_detail"}, ensure_ascii=False, sort_keys=True))
    del model, base
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
