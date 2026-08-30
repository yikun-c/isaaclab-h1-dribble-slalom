from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from maze_agent.sft import SYSTEM_PROMPT, build_messages


def test_sft_messages_preserve_exact_public_tool_contract() -> None:
    example = {
        "input": {"instruction": "visit blue then exit"},
        "target": {"action": "MOVE_FORWARD", "decision_summary": "Advance."},
    }

    messages = build_messages(example)

    assert len(messages) == 3
    assert messages[0]["content"] == SYSTEM_PROMPT
    assert '"action": "MOVE_FORWARD"' in messages[2]["content"]
    assert "chain of thought" not in messages[0]["content"].lower()
