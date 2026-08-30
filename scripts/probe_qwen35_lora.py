"""Probe Qwen3.5 LoRA compatibility without starting a training run."""

from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "runtime" / "qwen35_transformers"))


def main() -> None:
    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForMultimodalLM

    model_dir = PROJECT_ROOT / "models" / "qwen3_5_2b"
    model = AutoModelForMultimodalLM.from_pretrained(
        model_dir, local_files_only=True, dtype=torch.bfloat16
    ).to("cuda")
    candidates = [
        name
        for name, _ in model.named_modules()
        if name.endswith(("q_proj", "k_proj", "v_proj", "o_proj"))
    ]
    if not candidates:
        raise RuntimeError("no standard attention projection modules found")
    adapted = get_peft_model(
        model,
        LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=8,
            lora_alpha=16,
            lora_dropout=0.0,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        ),
    )
    trainable = sum(parameter.numel() for parameter in adapted.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in adapted.parameters())
    print(
        json.dumps(
            {
                "candidate_count": len(candidates),
                "candidate_examples": candidates[:8],
                "trainable_parameters": trainable,
                "total_parameters": total,
                "trainable_percent": 100 * trainable / total,
            },
            ensure_ascii=False,
        )
    )
    del adapted, model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
