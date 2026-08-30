"""Development-only action-level evaluation for Qwen3.5 maze adapters."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "runtime" / "qwen35_transformers"))
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


def task_examples(seed: int, width: int, height: int) -> list[dict]:
    task = build_task(width, height, seed)
    state = reset(task)
    memory = TopologicalMemory()
    examples: list[dict] = []
    for decision_index, action in enumerate(astar_plan(task)):
        memory.record_observation(task, state)
        local = observe(task, state)
        examples.append(
            {
                "maze_seed": seed,
                "decision_index": decision_index,
                "input": {
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
                    "memory": memory.compact_summary(state),
                },
                "target_action": action.value,
            }
        )
        after = step(task, state, action)
        memory.record_transition(task, state, action, after)
        state = after
    return examples


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Qwen3.5 LoRA tool calls on development-only states.")
    parser.add_argument("--model-dir", type=Path, default=PROJECT_ROOT / "models" / "qwen3_5_2b")
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "artifacts" / "maze" / "splits_v1.json")
    parser.add_argument("--max-examples", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    examples = [
        example
        for maze_seed in manifest["development_seeds"]
        for example in task_examples(maze_seed, manifest["width"], manifest["height"])
    ]
    sample = random.Random(args.seed).sample(examples, min(args.max_examples, len(examples)))

    import torch
    from peft import PeftModel
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    processor = AutoProcessor.from_pretrained(args.adapter_dir, local_files_only=True)
    processor.tokenizer.pad_token = processor.tokenizer.pad_token or processor.tokenizer.eos_token
    processor.tokenizer.padding_side = "left"
    base = AutoModelForMultimodalLM.from_pretrained(
        args.model_dir, local_files_only=True, dtype=torch.bfloat16
    ).to("cuda").eval()
    model = PeftModel.from_pretrained(base, args.adapter_dir, local_files_only=True).eval()
    records: list[dict] = []
    started = time.perf_counter()
    with torch.inference_mode():
        for offset in range(0, len(sample), args.batch_size):
            batch = sample[offset : offset + args.batch_size]
            prompts = [
                processor.tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": [{"type": "text", "text": json.dumps(item["input"], ensure_ascii=False)}]},
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for item in batch
            ]
            inputs = processor(text=prompts, return_tensors="pt", padding=True).to("cuda")
            output_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
            )
            response_ids = output_ids[:, inputs["input_ids"].shape[-1] :]
            outputs = processor.batch_decode(response_ids, skip_special_tokens=True)
            for item, raw in zip(batch, outputs, strict=True):
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
    report = {
        "evaluation_version": "qwen35-maze-tool-calls-v1",
        "scope": "development-only expert states with causal external-memory input; not closed-loop navigation",
        "model_dir": str(args.model_dir.resolve()),
        "adapter_dir": str(args.adapter_dir.resolve()),
        "manifest": str(args.manifest.resolve()),
        "examples": len(records),
        "valid_json_rate": valid / len(records),
        "exact_action_accuracy": exact / len(records),
        "elapsed_seconds": elapsed,
        "examples_per_second": len(records) / elapsed,
        "records": records,
    }
    write_json_atomic(args.output, report)
    print(json.dumps({key: value for key, value in report.items() if key != "records"}, ensure_ascii=False, sort_keys=True))
    del model, base
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
