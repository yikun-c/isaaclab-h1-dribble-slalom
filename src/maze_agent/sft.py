"""Prompt construction and label masking for the maze planner SFT experiment."""

from __future__ import annotations

import json
from typing import Any


SYSTEM_PROMPT = (
    "You control only a high-level maze tool. Return exactly one JSON object with the keys "
    '"action" and "decision_summary". action must be one of MOVE_FORWARD, TURN_LEFT, '
    "TURN_RIGHT, BACKTRACK, STOP. Do not include analysis, markdown, or extra keys."
)


def build_messages(example: dict[str, Any]) -> list[dict[str, str]]:
    target = example["target"]
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(example["input"], ensure_ascii=False, sort_keys=True)},
        {"role": "assistant", "content": json.dumps(target, ensure_ascii=False, sort_keys=True)},
    ]


def tokenize_example(tokenizer: Any, example: dict[str, Any], max_length: int) -> dict[str, list[int]]:
    """Mask user/system tokens so supervised loss is only on the planner tool response."""
    messages = build_messages(example)
    prefix = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
    full = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    full_encoded = tokenizer(full, truncation=True, max_length=max_length, add_special_tokens=False)
    prefix_encoded = tokenizer(prefix, truncation=True, max_length=max_length, add_special_tokens=False)
    input_ids = full_encoded["input_ids"]
    attention_mask = full_encoded["attention_mask"]
    labels = list(input_ids)
    for index in range(min(len(prefix_encoded["input_ids"]), len(labels))):
        labels[index] = -100
    if all(label == -100 for label in labels):
        raise ValueError("assistant target was truncated; increase max_length or shorten input")
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def collate_sft_batch(tokenizer: Any, features: list[dict[str, list[int]]]) -> dict[str, Any]:
    import torch

    maximum = max(len(feature["input_ids"]) for feature in features)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        raise ValueError("tokenizer must define a pad token")
    batch = {"input_ids": [], "attention_mask": [], "labels": []}
    for feature in features:
        padding = maximum - len(feature["input_ids"])
        batch["input_ids"].append(feature["input_ids"] + [pad_id] * padding)
        batch["attention_mask"].append(feature["attention_mask"] + [0] * padding)
        batch["labels"].append(feature["labels"] + [-100] * padding)
    return {key: torch.tensor(value, dtype=torch.long) for key, value in batch.items()}
