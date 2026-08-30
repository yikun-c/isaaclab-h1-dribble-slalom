"""Bounded manual DPO for Qwen3.5 LoRA maze pairs, with explicit reference model."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "runtime" / "qwen35_transformers"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from maze_agent.sft import SYSTEM_PROMPT


def write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def encode_pair(tokenizer, prompt: dict, response: str, max_length: int):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [{"type": "text", "text": json.dumps(prompt, ensure_ascii=False)}]},
    ]
    prefix = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    full = tokenizer.apply_chat_template(messages + [{"role": "assistant", "content": response}], tokenize=False, add_generation_prompt=False)
    prefix_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full, truncation=True, max_length=max_length, add_special_tokens=False)["input_ids"]
    if len(full_ids) <= len(prefix_ids):
        raise ValueError("response was truncated entirely")
    return full_ids, len(prefix_ids)


def sequence_response_logprob(model, input_ids, response_start: int):
    import torch
    import torch.nn.functional as F

    logits = model(input_ids=input_ids, use_cache=False).logits[:, :-1, :]
    labels = input_ids[:, 1:]
    token_logps = F.log_softmax(logits.float(), dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    # labels index j corresponds to full token j+1. Response begins at prefix length.
    mask = torch.zeros_like(token_logps, dtype=torch.bool)
    mask[:, max(0, response_start - 1) :] = True
    return (token_logps * mask).sum(dim=-1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run bounded Qwen3.5 LoRA DPO with a frozen adapter reference.")
    parser.add_argument("--model-dir", type=Path, default=PROJECT_ROOT / "models/qwen3_5_2b")
    parser.add_argument("--reference-adapter", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, default=PROJECT_ROOT / "assets/datasets/maze_dpo_pairs_smoke_v1.jsonl")
    parser.add_argument("--run-name", default="qwen3_5_2b_maze_dpo_smoke_v1")
    parser.add_argument("--max-steps", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.max_steps <= 0 or args.max_length <= 0 or args.beta <= 0:
        raise ValueError("max-steps, max-length and beta must be positive")
    if not args.model_dir.is_dir() or not args.reference_adapter.is_dir() or not args.pairs.is_file():
        raise FileNotFoundError("model, reference adapter, or pair data is missing")

    import torch
    import torch.nn.functional as F
    from peft import PeftModel
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    rows = [json.loads(line) for line in args.pairs.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError("pair dataset is empty")
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = args.output_dir or PROJECT_ROOT / "runs/qwen35_dpo" / f"{timestamp}_{args.run_name}"
    run_dir.mkdir(parents=True, exist_ok=False)
    status_path = run_dir / "run_status.json"
    write_json_atomic(status_path, {"status": "initializing", "pairs": str(args.pairs.resolve()), "reference_adapter": str(args.reference_adapter.resolve()), "max_steps": args.max_steps})
    processor = AutoProcessor.from_pretrained(args.reference_adapter, local_files_only=True)
    tokenizer = processor.tokenizer
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    reference_base = AutoModelForMultimodalLM.from_pretrained(args.model_dir, local_files_only=True, dtype=torch.bfloat16).to("cuda").eval()
    reference = PeftModel.from_pretrained(reference_base, args.reference_adapter, local_files_only=True).eval()
    for parameter in reference.parameters():
        parameter.requires_grad_(False)
    policy_base = AutoModelForMultimodalLM.from_pretrained(args.model_dir, local_files_only=True, dtype=torch.bfloat16).to("cuda")
    policy = PeftModel.from_pretrained(policy_base, args.reference_adapter, local_files_only=True, is_trainable=True)
    policy.config.use_cache = False
    policy.gradient_checkpointing_enable()
    optimizer = torch.optim.AdamW((p for p in policy.parameters() if p.requires_grad), lr=args.learning_rate)
    metrics: list[dict] = []
    try:
        for step_index in range(args.max_steps):
            row = rows[step_index % len(rows)]
            chosen_ids, chosen_start = encode_pair(tokenizer, row["prompt"], row["chosen"], args.max_length)
            rejected_ids, rejected_start = encode_pair(tokenizer, row["prompt"], row["rejected"], args.max_length)
            chosen_tensor = torch.tensor([chosen_ids], device="cuda", dtype=torch.long)
            rejected_tensor = torch.tensor([rejected_ids], device="cuda", dtype=torch.long)
            with torch.no_grad():
                ref_chosen = sequence_response_logprob(reference, chosen_tensor, chosen_start)
                ref_rejected = sequence_response_logprob(reference, rejected_tensor, rejected_start)
            policy_chosen = sequence_response_logprob(policy, chosen_tensor, chosen_start)
            policy_rejected = sequence_response_logprob(policy, rejected_tensor, rejected_start)
            margin = (policy_chosen - policy_rejected) - (ref_chosen - ref_rejected)
            loss = -F.logsigmoid(args.beta * margin).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_((p for p in policy.parameters() if p.requires_grad), 1.0)
            optimizer.step()
            item = {"step": step_index + 1, "loss": float(loss.detach().cpu()), "margin": float(margin.detach().mean().cpu()), "chosen_action": row["chosen_action"], "rejected_action": row["rejected_action"]}
            metrics.append(item)
            write_json_atomic(status_path, {"status": "training", "run_dir": str(run_dir.resolve()), "latest": item})
        adapter_dir = run_dir / "adapter"
        policy.save_pretrained(adapter_dir)
        processor.save_pretrained(adapter_dir)
        summary = {"status": "completed", "run_dir": str(run_dir.resolve()), "adapter_dir": str(adapter_dir.resolve()), "reference_adapter": str(args.reference_adapter.resolve()), "pairs": str(args.pairs.resolve()), "max_steps": args.max_steps, "beta": args.beta, "metrics": metrics}
        write_json_atomic(run_dir / "train_summary.json", summary)
        write_json_atomic(status_path, summary)
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    finally:
        del policy, policy_base, reference, reference_base
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
