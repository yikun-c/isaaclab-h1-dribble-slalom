"""Run a bounded LoRA SFT smoke test for the structured maze tool protocol."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from maze_agent.sft import collate_sft_batch, tokenize_example


def write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Bounded LoRA SFT for the maze tool-response dataset.")
    parser.add_argument("--model-dir", type=Path, default=PROJECT_ROOT / "models" / "qwen2_5_1_5b_instruct")
    parser.add_argument("--dataset", type=Path, default=PROJECT_ROOT / "assets" / "datasets" / "maze_sft_smoke_v1.jsonl")
    parser.add_argument("--run-name", default="qwen2_5_1_5b_maze_sft_smoke_v1")
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2.0e-4)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    if args.max_steps <= 0 or args.batch_size <= 0 or args.gradient_accumulation <= 0:
        raise ValueError("max-steps, batch-size and gradient-accumulation must be positive")
    if not args.model_dir.is_dir():
        raise FileNotFoundError(f"model directory not found: {args.model_dir}")
    if not args.dataset.is_file():
        raise FileNotFoundError(f"dataset not found: {args.dataset}")

    import torch
    from datasets import Dataset
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this bounded SFT run")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = args.output_dir or PROJECT_ROOT / "runs" / "maze_sft" / f"{timestamp}_{args.run_name}"
    run_dir.mkdir(parents=True, exist_ok=False)
    status_path = run_dir / "run_status.json"
    write_json_atomic(
        status_path,
        {
            "status": "initializing",
            "model_dir": str(args.model_dir.resolve()),
            "dataset": str(args.dataset.resolve()),
            "seed": args.seed,
            "max_steps": args.max_steps,
            "created_at": datetime.now().isoformat(timespec="seconds"),
        },
    )

    rows = [json.loads(line) for line in args.dataset.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError("SFT dataset is empty")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    tokenizer.padding_side = "right"
    encoded = [tokenize_example(tokenizer, row, args.max_length) for row in rows]
    dataset = Dataset.from_list(encoded)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        local_files_only=True,
        torch_dtype=torch.float16,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    lora = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank,
        lora_alpha=args.lora_rank * 2,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora)
    training_args = TrainingArguments(
        output_dir=str(run_dir),
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        logging_steps=1,
        save_strategy="steps",
        save_steps=args.max_steps,
        save_total_limit=1,
        report_to=[],
        fp16=True,
        dataloader_num_workers=0,
        remove_unused_columns=False,
        seed=args.seed,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=lambda features: collate_sft_batch(tokenizer, features),
    )
    write_json_atomic(status_path, {"status": "training", "run_dir": str(run_dir), "dataset_rows": len(rows)})
    train_result = trainer.train()
    adapter_dir = run_dir / "adapter"
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    metrics = {key: float(value) if isinstance(value, (int, float)) else value for key, value in train_result.metrics.items()}
    summary = {
        "status": "completed",
        "run_dir": str(run_dir.resolve()),
        "adapter_dir": str(adapter_dir.resolve()),
        "dataset_rows": len(rows),
        "seed": args.seed,
        "max_steps": args.max_steps,
        "metrics": metrics,
    }
    write_json_atomic(run_dir / "train_summary.json", summary)
    write_json_atomic(status_path, summary)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
