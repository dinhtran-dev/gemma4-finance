# Fine-Tuning Gemma 3 270M for Expense Parsing

### Target environment: Mac Studio M4 Max, 36GB unified memory

An implementation plan for building a natural-language expense tracker powered by a fine-tuned Gemma 3 270M model, running entirely on Apple Silicon.

-----

## 1. Project Goal

Fine-tune Gemma 3 270M to convert natural language expense descriptions into structured JSON, enabling an expense-tracking app where users log spending conversationally (e.g. "grabbed lunch at Chipotle for $14").

**Target output schema:**

```json
{
  "amount": 14.00,
  "currency": "USD",
  "category": "food_drink",
  "merchant": "Chipotle",
  "description": "lunch",
  "date": "today"
}
```

**Success criteria:**

- ≥95% valid JSON output
- ≥90% field-level accuracy on held-out test set
- Inference latency under 200ms on target device
- Model size under 500MB after quantization

-----

## 2. Why MLX on Apple Silicon

The M4 Max with 36GB unified memory is an excellent fine-tuning platform for models this size. Key tooling choices:

| Option                     | Verdict                                                                                        |
|----------------------------|------------------------------------------------------------------------------------------------|
| **MLX + mlx-lm**           | Recommended. Native Apple Silicon, fastest, lowest memory overhead, first-class LoRA support   |
| PyTorch with MPS backend   | Works but slower, some ops fall back to CPU                                                    |
| Unsloth                    | CUDA-only, won't run on Mac                                                                    |
| bitsandbytes               | CUDA-only                                                                                      |

MLX leverages the unified memory architecture — no CPU↔GPU transfers. For a 270M model, training runs almost entirely in cache.

-----

## 3. Repository Layout

```
gemma4-finance/
├── data/
│   └── seed/seed_examples.jsonl    # Hand-written seed records (input → output)
├── src/expense_parser/
│   ├── schema.py                   # Pydantic Expense + Category enum
│   ├── prompt.py                   # Gemma chat-template formatter
│   ├── fallback.py                 # Regex/heuristic parser (used when model output is invalid)
│   └── inference.py                # MLX inference wrapper with schema validation
├── scripts/
│   ├── generate_synthetic.py       # Claude-backed synthetic expansion
│   ├── build_dataset.py            # Dedup, stratified split, Gemma-format JSONL
│   ├── train.py                    # LoRA fine-tune via mlx-lm tuner
│   ├── evaluate.py                 # JSON validity, exact-match, per-field F1, amount MAE
│   └── predict.py                  # One-shot CLI for a single utterance
├── tests/test_schema_and_prompt.py # Offline unit tests (no MLX required)
├── Modelfile                       # Ollama deployment template
├── requirements.txt
└── README.md
```

-----

## 4. Environment Setup

### System prerequisites

- macOS 14+ (Sonoma or later)
- Python 3.10+
- Xcode Command Line Tools

### Install dependencies

```bash
python3 -m venv ~/envs/gemma-ft
source ~/envs/gemma-ft/bin/activate
pip install -r requirements.txt
```

### Hugging Face access

Accept the Gemma license on the Hugging Face model page, then:

```bash
huggingface-cli login
```

### Verify the setup

```bash
python -c "import mlx.core as mx; print(mx.default_device())"
# Expected: Device(gpu, 0)
```

### Resource expectations on M4 Max (36GB)

| Stage                                       | Memory   | Time            |
|---------------------------------------------|----------|-----------------|
| Model load (bf16)                           | ~1.5 GB  | seconds         |
| LoRA fine-tuning (1.5k examples, 4 epochs)  | ~3–5 GB  | 5–15 minutes    |
| Full-parameter fine-tuning (alternative)    | ~6–10 GB | 15–40 minutes   |
| Inference                                   | ~1 GB    | <100 ms/response|

With 36GB you could even fine-tune Gemma 3 **1B** or **4B** if the 270M ends up undersized. Start small — 270M is almost certainly enough for this task.

-----

## 5. Dataset Preparation

### Target size

- **Training set:** 1,000–2,000 examples
- **Validation set:** 150–250 examples
- **Test set:** 150–250 examples (held out until the end)

### Pipeline

1. **Seed set (manual):** Hand-write 50–100 diverse examples covering the coverage matrix below. A starter set lives in `data/seed/seed_examples.jsonl` (extend it).
2. **Synthetic expansion:** `python scripts/generate_synthetic.py --out data/synthetic/raw.jsonl --per-category 50` (shells out to the `claude` CLI — uses your existing Claude Code auth; add `--batches-per-category N` to scale volume).
3. **Human review:** Hand-check every synthetic example — bad labels poison training.
4. **Build splits:** `python scripts/build_dataset.py --sources data/seed/seed_examples.jsonl data/synthetic/raw.jsonl` deduplicates, stratifies by category, and writes `data/{train,valid,test}.jsonl` in mlx-lm format with the Gemma chat template already applied.

### Coverage matrix

| Dimension      | Variations to include                                                                       |
|----------------|---------------------------------------------------------------------------------------------|
| Amount format  | `$15`, `15 bucks`, `fifteen dollars`, `15.50`, `€20`, `¥1200`                               |
| Category       | food, transport, groceries, entertainment, bills, shopping, health, travel, subscriptions   |
| Phrasing       | terse (`uber 22`), conversational, past tense, present tense                                |
| Merchant       | named, unnamed, abbreviated, misspelled                                                     |
| Date           | today, yesterday, last Tuesday, explicit dates, none                                        |
| Edge cases     | missing fields, multi-item, tips, splits, refunds (negative amounts)                        |

### Seed record format

Raw records live in `data/seed/seed_examples.jsonl`, one per line:

```json
{"input": "spent $15 on coffee at Starbucks", "output": {"amount": 15.00, "currency": "USD", "category": "food_drink", "merchant": "Starbucks", "description": "coffee", "date": "today"}}
```

`build_dataset.py` formats these into the mlx-lm `{"text": "<start_of_turn>user\n...<end_of_turn>"}` form automatically.

-----

## 6. Training

### Option A: Command-line (simplest)

```bash
mlx_lm.lora \
  --model google/gemma-3-270m \
  --train \
  --data ./data \
  --iters 1000 \
  --batch-size 8 \
  --learning-rate 2e-4 \
  --num-layers 16 \
  --adapter-path ./adapters/gemma3-270m-expense \
  --save-every 200 \
  --steps-per-eval 100 \
  --val-batches 20
```

### Option B: Python wrapper in this repo

```bash
python scripts/train.py \
  --data ./data \
  --adapter-path ./adapters/gemma3-270m-expense \
  --iters 1000 \
  --batch-size 8
```

### Recommended hyperparameters

| Parameter      | Value    | Notes                                  |
|----------------|----------|----------------------------------------|
| LoRA rank      | 16       | Increase to 32 if underfitting         |
| LoRA alpha     | 32       | Typically 2× rank                      |
| LoRA dropout   | 0.05     |                                        |
| Learning rate  | 2e-4     | Lower to 1e-4 if loss is unstable      |
| Batch size     | 8        | Can push to 16 on 36GB                 |
| Iterations     | 800–1500 | ~4 epochs on 1.5k examples             |
| Max seq length | 512      | Expense strings are short; 256 is fine |

Watch for: training loss decreasing smoothly; validation loss tracking training loss (divergence = overfitting); stop when val loss plateaus or rises.

-----

## 7. Evaluation

```bash
python scripts/evaluate.py --adapter-path ./adapters/gemma3-270m-expense
```

Reports:

- **JSON validity rate** — outputs that parse and validate against the schema
- **Exact-match accuracy** — full-record correctness
- **Per-field accuracy** — amount, currency, category, merchant, description, date
- **Amount MAE** — mean absolute error on numeric amounts

Maintain ~30 "canary" examples covering tricky cases. Run every checkpoint; any regression blocks shipping.

-----

## 8. Deployment

### Fuse adapters

```bash
mlx_lm.fuse \
  --model google/gemma-3-270m \
  --adapter-path ./adapters/gemma3-270m-expense \
  --save-path ./models/gemma3-270m-expense-merged
```

### Convert to GGUF

```bash
python /path/to/llama.cpp/convert_hf_to_gguf.py \
  ./models/gemma3-270m-expense-merged \
  --outtype q4_k_m \
  --outfile ./gemma3-270m-expense.gguf
```

Size expectations: bf16 MLX ~540 MB · Q8_0 GGUF ~290 MB · Q4_K_M GGUF ~180 MB (recommended for mobile).

### Ollama

```bash
ollama create gemma3-expense -f Modelfile
ollama run gemma3-expense
```

### Deployment targets

| Target            | Runtime                                        |
|-------------------|------------------------------------------------|
| Mac/iOS app       | MLX directly (see `ios/ExpenseParser/`)        |
| Android app       | llama.cpp bindings                             |
| Backend API       | Ollama or MLX server                           |
| Local desktop app | MLX (best on Mac), llama.cpp (cross-platform)  |

An iOS SwiftUI demo that runs the fused model on-device via MLX-Swift lives in
`ios/ExpenseParser/` — see that folder's README for Xcode setup.

### Inference wrapper

`ExpenseParser` (`src/expense_parser/inference.py`) formats the chat-template prompt, runs MLX generation, extracts the JSON block, validates it against the Pydantic schema, and falls back to `parse_fallback()` on any failure — returning a `ParseResult` with a `used_fallback` flag so the app can decide how to handle low-confidence outputs.

```python
from expense_parser.inference import ExpenseParser

parser = ExpenseParser(adapter_path="./adapters/gemma3-270m-expense")
result = parser.parse("uber home 22")
print(result.expense.model_dump(), result.used_fallback)
```

-----

## 9. Iteration Loop

1. Log inputs where output was rejected, edited, or failed to parse.
2. Sample ~50/week for manual labeling.
3. Add to the training set and retrain every 2–4 weeks.
4. Track per-category accuracy over time.

Training speed on M4 Max makes this project data-bound rather than compute-bound.

-----

## 10. Risks & Mitigations

| Risk                                        | Mitigation                                                      |
|---------------------------------------------|-----------------------------------------------------------------|
| Model hallucinates fields                   | Validate JSON against schema; fall back to rule-based parser    |
| Category drift / inconsistency              | Enforce fixed category enum in training data                    |
| Ambiguous amounts (e.g. "a couple bucks")   | Return `null` + confidence; prompt user to confirm              |
| Overfitting on synthetic data               | Ensure real-user examples make up ≥30% of training set after v1 |
| 270M is too small for edge cases            | M4 Max can easily fine-tune Gemma 3 1B — escalate if needed     |
| Privacy (expenses are sensitive)            | Run inference on-device; never log raw inputs without consent   |

-----

## 11. Quick start

```bash
pip install -r requirements.txt
huggingface-cli login

# 1. Expand dataset
python scripts/generate_synthetic.py --out data/synthetic/raw.jsonl --per-category 50
#    ...hand-review data/synthetic/raw.jsonl...

# 2. Build splits
python scripts/build_dataset.py \
  --sources data/seed/seed_examples.jsonl data/synthetic/raw.jsonl

# 3. Smoke test (50 iters on the current tiny seed set)
python scripts/train.py --iters 50

# 4. Full run
python scripts/train.py --iters 1000

# 5. Evaluate
python scripts/evaluate.py

# 6. Predict
python scripts/predict.py "grabbed lunch at Chipotle for $14"
```

Offline unit tests (no MLX required): `python -m pytest tests/`.
