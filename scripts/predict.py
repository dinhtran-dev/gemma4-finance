"""One-shot CLI: parse a single expense utterance."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from expense_parser.inference import ExpenseParser  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("text", help="Expense description to parse")
    ap.add_argument("--model", default="google/gemma-3-270m")
    ap.add_argument("--adapter-path", default="./adapters/gemma3-270m-expense")
    args = ap.parse_args()

    parser = ExpenseParser(model_path=args.model, adapter_path=args.adapter_path)
    result = parser.parse(args.text)
    print(json.dumps({
        "expense": result.expense.model_dump(),
        "used_fallback": result.used_fallback,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
