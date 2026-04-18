"""Build train/valid/test JSONL in mlx-lm format.

Reads raw records with {input, output} fields from one or more source JSONLs,
deduplicates by normalized input, shuffles deterministically, stratifies the
split by category, and formats each record with the Gemma chat template.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from expense_parser.prompt import format_training_example  # noqa: E402
from expense_parser.schema import Expense  # noqa: E402


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _load(paths: list[Path]) -> list[dict]:
    seen: set[str] = set()
    out: list[dict] = []
    for p in paths:
        with p.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                key = _norm(rec["input"])
                if key in seen:
                    continue
                seen.add(key)
                Expense.model_validate(rec["output"])
                out.append(rec)
    return out


def _stratified_split(
    records: list[dict], val_frac: float, test_frac: float, seed: int
) -> tuple[list[dict], list[dict], list[dict]]:
    buckets: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        buckets[rec["output"].get("category", "other")].append(rec)

    rng = random.Random(seed)
    train, valid, test = [], [], []
    for cat, items in buckets.items():
        rng.shuffle(items)
        n = len(items)
        n_test = max(1, int(round(n * test_frac))) if n > 2 else 0
        n_val = max(1, int(round(n * val_frac))) if n > 2 else 0
        test.extend(items[:n_test])
        valid.extend(items[n_test:n_test + n_val])
        train.extend(items[n_test + n_val:])

    rng.shuffle(train)
    rng.shuffle(valid)
    rng.shuffle(test)
    return train, valid, test


def _write(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for rec in records:
            formatted = format_training_example(rec["input"], rec["output"])
            f.write(json.dumps(formatted, ensure_ascii=False) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sources", nargs="+", required=True,
                    help="Raw JSONL file(s) with {input, output} records")
    ap.add_argument("--out-dir", default="data",
                    help="Directory to write train/valid/test.jsonl")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--test-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    records = _load([Path(p) for p in args.sources])
    if not records:
        print("No records loaded", file=sys.stderr)
        return 1

    train, valid, test = _stratified_split(
        records, args.val_frac, args.test_frac, args.seed
    )

    out = Path(args.out_dir)
    _write(out / "train.jsonl", train)
    _write(out / "valid.jsonl", valid)
    _write(out / "test.jsonl", test)

    print(f"train={len(train)} valid={len(valid)} test={len(test)} "
          f"(from {len(records)} unique records) -> {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
