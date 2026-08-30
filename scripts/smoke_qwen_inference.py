"""Bounded local inference smoke test for the pinned maze-planner baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one short local Qwen inference and report peak VRAM.")
    parser.add_argument("--model-dir", type=Path, default=PROJECT_ROOT / "models" / "qwen2_5_1_5b_instruct")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        local_files_only=True,
        torch_dtype=torch.float16,
    ).to("cuda").eval()
    messages = [
        {
            "role": "system",
            "content": (
                "Return exactly one JSON object with action and decision_summary. action must be one of "
                "MOVE_FORWARD, TURN_LEFT, TURN_RIGHT, BACKTRACK, STOP."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {"front_open": True, "left_open": False, "right_open": True}, ensure_ascii=False
            ),
        },
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = tokenizer.decode(output[0][inputs.input_ids.shape[1] :], skip_special_tokens=True)
    report = {
        "generated": generated,
        "allocated_mb": round(torch.cuda.max_memory_allocated() / 1024 / 1024, 1),
        "reserved_mb": round(torch.cuda.max_memory_reserved() / 1024 / 1024, 1),
    }
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    del output, inputs, model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
