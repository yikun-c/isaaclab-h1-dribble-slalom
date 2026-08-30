"""Run one bounded text-only Qwen3.5 multimodal-model inference in an isolated runtime."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_ROOT = PROJECT_ROOT / "runtime" / "qwen35_transformers"
sys.path.insert(0, str(RUNTIME_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test Qwen3.5-2B text tool generation.")
    parser.add_argument("--model-dir", type=Path, default=PROJECT_ROOT / "models" / "qwen3_5_2b")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    args = parser.parse_args()
    if not args.model_dir.is_dir():
        raise FileNotFoundError(args.model_dir)

    import torch
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    processor = AutoProcessor.from_pretrained(args.model_dir, local_files_only=True)
    model = AutoModelForMultimodalLM.from_pretrained(
        args.model_dir,
        local_files_only=True,
        dtype=torch.bfloat16,
    ).to("cuda").eval()
    messages = [
        {
            "role": "system",
            "content": "Return exactly one JSON object with action and decision_summary. No analysis or markdown.",
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": '{"front_open":true,"left_open":false,"right_open":true}',
                }
            ],
        },
    ]
    prompt = processor.tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )
    inputs = processor(text=prompt, return_tensors="pt").to("cuda")
    input_length = inputs["input_ids"].shape[-1]
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
        )
    response = processor.decode(outputs[0][input_length:], skip_special_tokens=True)
    print(
        json.dumps(
            {
                "response": response,
                "allocated_mb": round(torch.cuda.max_memory_allocated() / 1024 / 1024, 1),
                "reserved_mb": round(torch.cuda.max_memory_reserved() / 1024 / 1024, 1),
            },
            ensure_ascii=False,
        )
    )
    del outputs, inputs, model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
