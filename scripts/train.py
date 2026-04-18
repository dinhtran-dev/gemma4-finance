"""LoRA fine-tune Gemma 3 270M on the prepared expense dataset.

Thin wrapper around mlx-lm's tuner API using the hyperparameters from the plan.
Meant to be run on Apple Silicon (Mac Studio M4 Max or similar).
"""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="google/gemma-3-270m")
    ap.add_argument("--data", default="./data")
    ap.add_argument("--adapter-path", default="./adapters/gemma3-270m-expense")
    ap.add_argument("--iters", type=int, default=1000)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--learning-rate", type=float, default=2e-4)
    ap.add_argument("--num-layers", type=int, default=16)
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--lora-alpha", type=float, default=32.0)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--max-seq-length", type=int, default=512)
    ap.add_argument("--steps-per-report", type=int, default=10)
    ap.add_argument("--steps-per-eval", type=int, default=100)
    ap.add_argument("--steps-per-save", type=int, default=200)
    ap.add_argument("--val-batches", type=int, default=20)
    args = ap.parse_args()

    from mlx_lm import load
    from mlx_lm.tuner import TrainingArgs, linear_to_lora_layers, train
    from mlx_lm.tuner.datasets import load_dataset

    model, tokenizer = load(args.model)

    lora_config = {
        "rank": args.lora_rank,
        "alpha": args.lora_alpha,
        "dropout": args.lora_dropout,
        "scale": 2.0,
    }
    linear_to_lora_layers(model, num_layers=args.num_layers, config=lora_config)

    train_set, valid_set, _ = load_dataset(args.data, tokenizer)

    adapter_dir = Path(args.adapter_path)
    adapter_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArgs(
        batch_size=args.batch_size,
        iters=args.iters,
        val_batches=args.val_batches,
        steps_per_report=args.steps_per_report,
        steps_per_eval=args.steps_per_eval,
        steps_per_save=args.steps_per_save,
        adapter_file=str(adapter_dir / "adapters.safetensors"),
        max_seq_length=args.max_seq_length,
    )

    train(model, tokenizer, training_args, train_set, valid_set)
    print(f"Adapter saved to {adapter_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
