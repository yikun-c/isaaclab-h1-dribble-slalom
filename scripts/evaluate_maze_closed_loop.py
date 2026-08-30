"""Closed-loop development evaluation for a local maze-planner model.

The evaluator never loads IID-final or OOD-final seeds. It logs each planner action
and physical outcome, so a valid JSON answer is not confused with task success.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from maze_agent import (
    TopologicalMemory,
    build_task,
    decision_event,
    observe,
    parse_planner_response,
    step,
)
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


def planner_input(task, state, memory: TopologicalMemory, include_memory: bool) -> dict:
    local = observe(task, state)
    payload = {
        "instruction": task.instruction,
        "local_perception": {
            "front_open": local.front_open,
            "left_open": local.left_open,
            "right_open": local.right_open,
            "rear_open": local.rear_open,
            "current_landmarks": list(local.current_landmarks),
            "adjacent_landmarks": list(local.adjacent_landmarks),
        },
        "state": {
            "heading": state.heading.value,
            "checkpoint_complete": state.checkpoint_complete,
            "last_result": state.last_result,
        },
    }
    if include_memory:
        payload["memory"] = memory.compact_summary(state)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run closed-loop development mazes with a local LoRA adapter.")
    parser.add_argument("--model-dir", type=Path, default=PROJECT_ROOT / "models" / "qwen2_5_1_5b_instruct")
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "artifacts" / "maze" / "splits_v1.json")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max-decisions", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--with-memory", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.episodes <= 0 or args.max_decisions <= 0:
        raise ValueError("episodes and max-decisions must be positive")

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    seeds = manifest["development_seeds"][: args.episodes]

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.adapter_dir, local_files_only=True)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, local_files_only=True, torch_dtype=torch.float16
    ).to("cuda").eval()
    model = PeftModel.from_pretrained(model, args.adapter_dir, local_files_only=True).eval()

    episodes: list[dict] = []
    total_started = time.perf_counter()
    with torch.inference_mode():
        for maze_seed in seeds:
            task = build_task(manifest["width"], manifest["height"], maze_seed)
            state = reset(task)
            memory = TopologicalMemory()
            events: list[dict] = []
            seen_states: set[tuple[tuple[int, int], str, bool]] = set()
            loops = 0
            for _ in range(args.max_decisions):
                memory.record_observation(task, state)
                memory_before = memory.compact_summary(state)
                key = (state.position, state.heading.value, state.checkpoint_complete)
                loops += int(key in seen_states)
                seen_states.add(key)
                payload = planner_input(task, state, memory, args.with_memory)
                prompt = tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": json.dumps(payload, ensure_ascii=False, sort_keys=True)},
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
                started = time.perf_counter()
                generated = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
                latency_ms = (time.perf_counter() - started) * 1000
                raw = tokenizer.decode(generated[0][inputs.input_ids.shape[1] :], skip_special_tokens=True)
                decision = parse_planner_response(raw)
                after = step(task, state, decision.action)
                events.append(
                    decision_event(
                        task,
                        state,
                        decision,
                        after,
                        memory_before,
                        raw,
                        latency_ms=latency_ms,
                        token_count=None,
                    )
                )
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
                    "events": events,
                }
            )
    elapsed = time.perf_counter() - total_started
    total_decisions = sum(episode["decisions"] for episode in episodes)
    summary = {
        "evaluation_version": "maze-closed-loop-v1",
        "scope": "development-only closed-loop mazes; final splits were not loaded",
        "model_dir": str(args.model_dir.resolve()),
        "adapter_dir": str(args.adapter_dir.resolve()),
        "manifest": str(args.manifest.resolve()),
        "with_memory": args.with_memory,
        "episodes": len(episodes),
        "successes": sum(episode["success"] for episode in episodes),
        "success_rate": sum(episode["success"] for episode in episodes) / len(episodes),
        "total_decisions": total_decisions,
        "valid_output_rate": sum(episode["valid_outputs"] for episode in episodes) / max(1, total_decisions),
        "mean_collisions": sum(episode["collisions"] for episode in episodes) / len(episodes),
        "mean_loop_observations": sum(episode["loop_observations"] for episode in episodes) / len(episodes),
        "elapsed_seconds": elapsed,
        "episodes_detail": episodes,
    }
    write_json_atomic(args.output, summary)
    print(json.dumps({key: value for key, value in summary.items() if key != "episodes_detail"}, ensure_ascii=False, sort_keys=True))
    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
