from __future__ import annotations

import json
from typing import Any, Mapping

SYSTEM_INSTRUCTION = (
    "Extract expense details as JSON with keys: "
    "amount (number or null), currency (ISO code, default USD), "
    "category (one of: food_drink, groceries, transport, travel, "
    "entertainment, shopping, bills, health, subscriptions, other), "
    "merchant (string or null), description (string or null), "
    "date (string or null)."
)


def build_user_prompt(user_text: str) -> str:
    return f"{SYSTEM_INSTRUCTION}\nInput: {user_text.strip()}"


def _gemma_wrap(user: str, assistant: str | None) -> str:
    turns = f"<start_of_turn>user\n{user}\n<end_of_turn>\n<start_of_turn>model\n"
    if assistant is not None:
        turns += f"{assistant}\n<end_of_turn>"
    return turns


def format_training_example(user_text: str, target: Mapping[str, Any]) -> dict:
    assistant = json.dumps(target, ensure_ascii=False, separators=(", ", ": "))
    text = _gemma_wrap(build_user_prompt(user_text), assistant)
    return {"text": text}


def format_inference_prompt(user_text: str) -> str:
    return _gemma_wrap(build_user_prompt(user_text), None)
