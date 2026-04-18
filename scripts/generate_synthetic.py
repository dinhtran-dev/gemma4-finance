"""Generate synthetic expense examples with Claude.

Requires ANTHROPIC_API_KEY. Outputs raw JSONL matching seed_examples.jsonl format
(fields: input, output). Every line MUST be hand-reviewed before training.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from expense_parser.schema import Category, Expense  # noqa: E402


GEN_SYSTEM = """You generate training examples for an expense-parsing model.
Produce diverse, realistic one-line user utterances and their structured JSON.

Output format: one example per line, each a JSON object:
{"input": "<natural language>", "output": {"amount": ..., "currency": "...", "category": "...", "merchant": ..., "description": ..., "date": ...}}

Rules:
- category must be one of: food_drink, groceries, transport, travel, entertainment, shopping, bills, health, subscriptions, other
- currency is an ISO 4217 code (default USD)
- merchant/description/date may be null
- Amount may be null when the user is vague (e.g. "a couple bucks")
- Vary phrasing heavily: terse, conversational, past/present, abbreviated, misspelled merchants
- Cover: named + unnamed merchants, multi-currency, dates (today/yesterday/last Tuesday/ISO/none), refunds (negative), multi-item, splits, tips
- No duplicates, no numbering, no explanations outside the JSON
"""


def _build_user_prompt(category: str, n: int) -> str:
    return (
        f"Generate {n} diverse expense examples, biased toward the '{category}' "
        f"category but include occasional others to avoid narrow distributions. "
        f"Use the JSONL format from the system instructions. Output only JSONL."
    )


def _validate_line(line: str) -> dict | None:
    line = line.strip()
    if not line or line.startswith("```"):
        return None
    try:
        rec = json.loads(line)
    except json.JSONDecodeError:
        return None
    if "input" not in rec or "output" not in rec:
        return None
    try:
        Expense.model_validate(rec["output"])
    except Exception:
        return None
    return rec


def generate(category: Category, n: int, model: str) -> list[dict]:
    from anthropic import Anthropic

    client = Anthropic()
    msg = client.messages.create(
        model=model,
        max_tokens=4096,
        system=GEN_SYSTEM,
        messages=[{"role": "user", "content": _build_user_prompt(category.value, n)}],
    )
    text = "".join(block.text for block in msg.content if block.type == "text")

    results: list[dict] = []
    for line in text.splitlines():
        rec = _validate_line(line)
        if rec is not None:
            results.append(rec)
    return results


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, help="Output JSONL path")
    ap.add_argument("--per-category", type=int, default=50,
                    help="Examples requested per category")
    ap.add_argument("--model", default="claude-opus-4-7",
                    help="Anthropic model id")
    args = ap.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY is not set", file=sys.stderr)
        return 1

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    with out_path.open("w") as f:
        for cat in Category:
            print(f"Generating {args.per_category} for {cat.value}...", file=sys.stderr)
            records = generate(cat, args.per_category, args.model)
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            total += len(records)
            print(f"  kept {len(records)}", file=sys.stderr)

    print(f"Wrote {total} validated examples to {out_path}", file=sys.stderr)
    print("Hand-review every line before adding to training data.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
