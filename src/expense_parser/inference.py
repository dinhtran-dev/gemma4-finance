from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .fallback import parse_fallback
from .prompt import format_inference_prompt
from .schema import Expense


_JSON_OBJECT_RE = re.compile(r"\{[\s\S]*\}")


def _extract_json(text: str) -> Optional[dict]:
    m = _JSON_OBJECT_RE.search(text)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


@dataclass
class ParseResult:
    expense: Expense
    raw_output: str
    used_fallback: bool


class ExpenseParser:
    def __init__(
        self,
        model_path: str = "google/gemma-3-270m",
        adapter_path: Optional[str] = None,
        max_tokens: int = 200,
    ) -> None:
        from mlx_lm import load  # imported lazily so non-MLX machines can still import the module

        self._model, self._tokenizer = load(
            model_path,
            adapter_path=adapter_path if adapter_path and Path(adapter_path).exists() else None,
        )
        self._max_tokens = max_tokens

    def _generate(self, prompt: str) -> str:
        from mlx_lm import generate

        return generate(
            self._model,
            self._tokenizer,
            prompt=prompt,
            max_tokens=self._max_tokens,
            verbose=False,
        )

    def parse(self, user_text: str) -> ParseResult:
        prompt = format_inference_prompt(user_text)
        raw = self._generate(prompt)
        payload = _extract_json(raw)
        if payload is not None:
            try:
                return ParseResult(Expense.model_validate(payload), raw, False)
            except ValueError:
                pass
        return ParseResult(parse_fallback(user_text), raw, True)
