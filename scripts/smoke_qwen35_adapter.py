"""Verify one Qwen3.5 LoRA adapter on a causal-memory maze state."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "runtime" / "qwen35_transformers"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from maze_agent import TopologicalMemory, build_task, observe, parse_planner_response
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test a Qwen3.5 maze LoRA adapter.")
    parser.add_argument("--model-dir", type=Path, default=PROJECT_ROOT / "models" / "qwen3_5_2b")
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    task = build_task(9, 9, args.seed)
    state = reset(task)
    memory = TopologicalMemory()
    memory.record_observation(task, state)
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
        "memory": memory.compact_summary(state),
    }
    processor = AutoProcessor.from_pretrained(args.adapter_dir, local_files_only=True)
    base = AutoModelForMultimodalLM.from_pretrained(
        args.model_dir, local_files_only=True, dtype=torch.bfloat16
    ).to("cuda").eval()
    model = PeftModel.from_pretrained(base, args.adapter_dir, local_files_only=True).eval()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [{"type": "text", "text": json.dumps(payload, ensure_ascii=False)}]},
    ]
    prompt = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=prompt, return_tensors="pt").to("cuda")
    with torch.inference_mode():
        output_ids = model.generate(**inputs, max_new_tokens=96, do_sample=False)
    raw = processor.decode(output_ids[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True)
    decision = parse_planner_response(raw)
    report = {
        "adapter_dir": str(args.adapter_dir.resolve()),
        "seed": args.seed,
        "input": payload,
        "raw_output": raw,
        "parsed": {
            "action": decision.action.value,
            "decision_summary": decision.decision_summary,
            "valid": decision.valid,
            "fallback_reason": decision.fallback_reason,
        },
    }
    write_json_atomic(args.output, report)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    del output_ids, inputs, model, base
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
