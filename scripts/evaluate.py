"""Evaluate a fine-tuned Gemma adapter on the held-out test set.

Reports JSON validity rate, exact-match accuracy, per-field F1, and amount MAE.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from expense_parser.prompt import format_inference_prompt  # noqa: E402
from expense_parser.schema import Expense  # noqa: E402


FIELDS = ("amount", "currency", "category", "merchant", "description", "date")
JSON_RE = re.compile(r"\{[\s\S]*\}")


def _extract_json(raw: str) -> Optional[dict]:
    m = JSON_RE.search(raw)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def _load_test_records(path: Path) -> list[tuple[str, dict]]:
    """Recover (input, target) pairs from a Gemma-formatted JSONL test file."""
    pattern = re.compile(
        r"<start_of_turn>user\n(.*?)\n<end_of_turn>\n"
        r"<start_of_turn>model\n(.*?)\n<end_of_turn>",
        re.DOTALL,
    )
    records: list[tuple[str, dict]] = []
    with path.open() as f:
        for line in f:
            obj = json.loads(line)
            m = pattern.search(obj["text"])
            if not m:
                continue
            user_block, model_block = m.group(1), m.group(2)
            user_text = user_block.split("Input: ", 1)[-1]
            target = json.loads(model_block)
            records.append((user_text, target))
    return records


def _fields_equal(a: Any, b: Any) -> bool:
    if isinstance(a, float) or isinstance(b, float):
        try:
            return abs(float(a) - float(b)) < 1e-6
        except (TypeError, ValueError):
            return False
    return a == b


def evaluate(model_path: str, adapter_path: str, data_dir: str, limit: int) -> dict:
    from mlx_lm import generate, load

    model, tokenizer = load(model_path, adapter_path=adapter_path)

    test_path = Path(data_dir) / "test.jsonl"
    records = _load_test_records(test_path)
    if limit > 0:
        records = records[:limit]

    valid_count = 0
    exact_count = 0
    amount_errors: list[float] = []
    field_stats: dict[str, dict[str, int]] = defaultdict(
        lambda: {"correct": 0, "total": 0}
    )

    for user_text, target in records:
        prompt = format_inference_prompt(user_text)
        raw = generate(model, tokenizer, prompt=prompt, max_tokens=200, temp=0.0, verbose=False)
        payload = _extract_json(raw)

        parsed_valid = False
        pred: dict = {}
        if payload is not None:
            try:
                pred = Expense.model_validate(payload).model_dump()
                parsed_valid = True
            except Exception:
                parsed_valid = False

        if parsed_valid:
            valid_count += 1
            if all(_fields_equal(pred.get(k), target.get(k)) for k in FIELDS):
                exact_count += 1
            for k in FIELDS:
                field_stats[k]["total"] += 1
                if _fields_equal(pred.get(k), target.get(k)):
                    field_stats[k]["correct"] += 1
            if isinstance(pred.get("amount"), (int, float)) and isinstance(target.get("amount"), (int, float)):
                amount_errors.append(abs(float(pred["amount"]) - float(target["amount"])))
        else:
            for k in FIELDS:
                field_stats[k]["total"] += 1

    n = max(1, len(records))
    return {
        "n": len(records),
        "json_validity": valid_count / n,
        "exact_match": exact_count / n,
        "amount_mae": sum(amount_errors) / len(amount_errors) if amount_errors else None,
        "field_accuracy": {
            k: (v["correct"] / v["total"]) if v["total"] else None
            for k, v in field_stats.items()
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="google/gemma-3-270m")
    ap.add_argument("--adapter-path", default="./adapters/gemma3-270m-expense")
    ap.add_argument("--data", default="./data")
    ap.add_argument("--limit", type=int, default=0,
                    help="Evaluate only the first N records (0 = all)")
    args = ap.parse_args()

    results = evaluate(args.model, args.adapter_path, args.data, args.limit)
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
