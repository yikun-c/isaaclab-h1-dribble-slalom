"""Evaluate structured action accuracy on development-only expert states.

This is an action-level tool-calling evaluation, not a closed-loop navigation claim.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from maze_agent import TopologicalMemory, astar_plan, build_task, observe, parse_planner_response, step
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


def task_examples(seed: int, width: int, height: int, with_memory: bool) -> list[dict]:
    task = build_task(width, height, seed)
    state = reset(task)
    memory = TopologicalMemory()
    examples: list[dict] = []
    for decision_index, action in enumerate(astar_plan(task)):
        local = observe(task, state)
        memory.record_observation(task, state)
        planner_input = {
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
        if with_memory:
            planner_input["memory"] = memory.compact_summary(state)
        examples.append(
            {
                "maze_seed": seed,
                "decision_index": decision_index,
                "input": planner_input,
                "target_action": action.value,
            }
        )
        after = step(task, state, action)
        memory.record_transition(task, state, action, after)
        state = after
    return examples


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate base or LoRA maze tool calls on development-only states.")
    parser.add_argument("--model-dir", type=Path, default=PROJECT_ROOT / "models" / "qwen2_5_1_5b_instruct")
    parser.add_argument("--adapter-dir", type=Path)
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "artifacts" / "maze" / "splits_v1.json")
    parser.add_argument("--max-examples", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--with-memory", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.max_examples <= 0 or args.batch_size <= 0:
        raise ValueError("max-examples and batch-size must be positive")

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    all_examples = [
        example
        for maze_seed in manifest["development_seeds"]
        for example in task_examples(maze_seed, manifest["width"], manifest["height"], args.with_memory)
    ]
    sampler = random.Random(args.seed)
    sampled = sampler.sample(all_examples, min(args.max_examples, len(all_examples)))

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if args.adapter_dir:
        from peft import PeftModel

    tokenizer = AutoTokenizer.from_pretrained(args.adapter_dir or args.model_dir, local_files_only=True)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, local_files_only=True, torch_dtype=torch.float16
    ).to("cuda").eval()
    if args.adapter_dir:
        model = PeftModel.from_pretrained(model, args.adapter_dir, local_files_only=True).eval()

    records: list[dict] = []
    started = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, len(sampled), args.batch_size):
            batch = sampled[start : start + args.batch_size]
            prompts = [
                tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": json.dumps(item["input"], ensure_ascii=False, sort_keys=True)},
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for item in batch
            ]
            inputs = tokenizer(prompts, return_tensors="pt", padding=True).to("cuda")
            generated = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
            response_tokens = generated[:, inputs.input_ids.shape[1] :]
            raw_outputs = tokenizer.batch_decode(response_tokens, skip_special_tokens=True)
            for item, raw in zip(batch, raw_outputs, strict=True):
                decision = parse_planner_response(raw)
                records.append(
                    {
                        **item,
                        "raw_output": raw,
                        "parsed_action": decision.action.value,
                        "valid": decision.valid,
                        "fallback_reason": decision.fallback_reason,
                        "exact_action_match": decision.valid and decision.action.value == item["target_action"],
                    }
                )
    elapsed = time.perf_counter() - started
    valid = sum(record["valid"] for record in records)
    exact = sum(record["exact_action_match"] for record in records)
    summary = {
        "evaluation_version": "maze-tool-calls-v1",
        "scope": "development-only expert states; action-level, not closed-loop navigation",
        "with_memory": args.with_memory,
        "model_dir": str(args.model_dir.resolve()),
        "adapter_dir": str(args.adapter_dir.resolve()) if args.adapter_dir else None,
        "manifest": str(args.manifest.resolve()),
        "examples": len(records),
        "valid_json_rate": valid / len(records),
        "exact_action_accuracy": exact / len(records),
        "elapsed_seconds": elapsed,
        "examples_per_second": len(records) / elapsed,
        "records": records,
    }
    write_json_atomic(args.output, summary)
    print(json.dumps({key: value for key, value in summary.items() if key != "records"}, ensure_ascii=False, sort_keys=True))
    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
