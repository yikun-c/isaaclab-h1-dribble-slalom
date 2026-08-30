"""Load a saved LoRA adapter and verify one real structured maze decision."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from maze_agent import build_task, observe, parse_planner_response
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
    parser = argparse.ArgumentParser(description="Verify one local LoRA maze-planner decision.")
    parser.add_argument("--model-dir", type=Path, default=PROJECT_ROOT / "models" / "qwen2_5_1_5b_instruct")
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    task = build_task(9, 9, args.seed)
    state = reset(task)
    observation = observe(task, state)
    planner_input = {
        "instruction": task.instruction,
        "local_perception": {
            "front_open": observation.front_open,
            "left_open": observation.left_open,
            "right_open": observation.right_open,
            "rear_open": observation.rear_open,
            "current_landmarks": list(observation.current_landmarks),
            "adjacent_landmarks": list(observation.adjacent_landmarks),
        },
        "state": {
            "heading": state.heading.value,
            "checkpoint_complete": state.checkpoint_complete,
            "last_result": state.last_result,
        },
    }
    tokenizer = AutoTokenizer.from_pretrained(args.adapter_dir, local_files_only=True)
    base = AutoModelForCausalLM.from_pretrained(
        args.model_dir, local_files_only=True, torch_dtype=torch.float16
    ).to("cuda").eval()
    model = PeftModel.from_pretrained(base, args.adapter_dir, local_files_only=True).eval()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(planner_input, ensure_ascii=False, sort_keys=True)},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=96,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    raw = tokenizer.decode(generated_ids[0][inputs.input_ids.shape[1] :], skip_special_tokens=True)
    decision = parse_planner_response(raw)
    report = {
        "adapter_dir": str(args.adapter_dir.resolve()),
        "model_dir": str(args.model_dir.resolve()),
        "seed": args.seed,
        "input": planner_input,
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
    del generated_ids, inputs, model, base
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
