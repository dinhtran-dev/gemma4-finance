"""Generate synthetic expense examples with the Claude CLI.

Shells out to `claude -p` per batch. Uses your existing Claude Code auth —
no ANTHROPIC_API_KEY required. Output is raw JSONL matching seed_examples.jsonl
(fields: input, output). Every line MUST be hand-reviewed before training.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from expense_parser.schema import Category, Expense  # noqa: E402


GEN_INSTRUCTIONS = """You generate training examples for an expense-parsing model.
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
- No duplicates, no numbering, no markdown fences, no explanations outside the JSON
- Output ONLY the JSONL, one record per line.
"""


def _build_prompt(category: str, n: int) -> str:
    return (
        f"{GEN_INSTRUCTIONS}\n\n"
        f"Generate {n} diverse expense examples, biased toward the "
        f"'{category}' category but include occasional others to avoid narrow "
        f"distributions. Output ONLY JSONL — one record per line."
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


def _run_claude(prompt: str, model: str | None, timeout: int) -> str:
    cmd = ["claude", "-p", prompt]
    if model:
        cmd.extend(["--model", model])
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"claude CLI exited {result.returncode}: {result.stderr.strip()[:500]}"
        )
    return result.stdout


def generate(category: Category, n: int, model: str | None, timeout: int) -> list[dict]:
    prompt = _build_prompt(category.value, n)
    text = _run_claude(prompt, model, timeout)
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
                    help="Examples requested per category per batch")
    ap.add_argument("--batches-per-category", type=int, default=1,
                    help="Run multiple batches per category to reach higher volume")
    ap.add_argument("--model", default=None,
                    help="Optional model override passed to `claude --model`")
    ap.add_argument("--timeout", type=int, default=600,
                    help="Per-batch timeout in seconds")
    args = ap.parse_args()

    if shutil.which("claude") is None:
        print("claude CLI not found on PATH", file=sys.stderr)
        return 1

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    with out_path.open("w") as f:
        for cat in Category:
            for batch in range(args.batches_per_category):
                label = f"{cat.value} (batch {batch + 1}/{args.batches_per_category})"
                print(f"Generating {args.per_category} for {label}...", file=sys.stderr)
                try:
                    records = generate(cat, args.per_category, args.model, args.timeout)
                except (subprocess.TimeoutExpired, RuntimeError) as e:
                    print(f"  skipped: {e}", file=sys.stderr)
                    continue
                for rec in records:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                total += len(records)
                print(f"  kept {len(records)}", file=sys.stderr)

    print(f"Wrote {total} validated examples to {out_path}", file=sys.stderr)
    print("Hand-review every line before adding to training data.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
