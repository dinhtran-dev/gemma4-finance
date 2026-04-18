# ExpenseParser iOS (MLX-Swift)

Minimal SwiftUI demo that runs `gemma3-270m-expense-merged` on-device via MLX-Swift.
Type an expense description, get structured JSON back.

Swift sources live under `ios/ExpenseParser/ExpenseParser/` and are tracked in
git. The `.xcodeproj`, build output, and bundled model weights are gitignored —
each developer creates the Xcode project locally once and adds the model
directory to the target.

## Requirements

- macOS with Xcode 15.3+
- A real iOS device (iPhone 13 or newer recommended). MLX runs on Metal; the
  iOS Simulator works for small models but performance is poor — use a device
  for real testing.
- The fused MLX model directory produced by step 7 of the root README:
  `models/gemma3-270m-expense-merged/` (contains `config.json`,
  `tokenizer.json`, weight shards, etc.).

## 1. Create the Xcode project

Skip this section if `ios/ExpenseParser/ExpenseParser.xcodeproj` already exists
on your machine.

1. **File → New → Project → iOS → App**.
2. Product name: `ExpenseParser`. Interface: **SwiftUI**. Language: **Swift**.
   Deployment target: **iOS 17.0+**.
3. Save it into `ios/` — Xcode creates `ios/ExpenseParser/` containing the
   `.xcodeproj` and an inner `ExpenseParser/` source folder that already holds
   the tracked `.swift` files.
4. In the Xcode navigator, right-click the `ExpenseParser` group → **Add
   Files to "ExpenseParser"…** and select the `Model/` directory under
   `ios/ExpenseParser/ExpenseParser/`.
   - **Copy items if needed**: off
   - **Added folders**: **Create groups**
   - **Add to targets**: `ExpenseParser`

## 2. Add MLX-Swift

In Xcode: **File → Add Package Dependencies…**

- URL: `https://github.com/ml-explore/mlx-swift-examples`
- Dependency Rule: **Up to Next Major Version** from `2.0.0`
- Add these products to the `ExpenseParser` target:
  - `MLX`
  - `MLXLLM`
  - `MLXLMCommon`

(The `mlx-swift-examples` package transitively pulls in `mlx-swift`,
`swift-transformers`, and tokenizer dependencies.)

## 3. Bundle the model

From the repo root, the fused model should live at
`models/gemma3-270m-expense-merged/`.

1. In Finder, drag that **directory** into the Xcode project navigator.
2. In the dialog, select:
   - **Create folder references** (blue folder icon — NOT "Create groups")
   - **Copy items if needed** = on
   - **Add to target: ExpenseParser**

This preserves the directory structure so MLX can load it as a single model
path at runtime. `ExpenseInference` looks for a bundle resource named
`gemma3-270m-expense-merged` — keep the directory name or update the
`bundleDirName` argument in `ExpenseParserApp`.

> Note: the bf16 model is ~540 MB. For smaller app size, regenerate the merged
> model in a quantized MLX format (`mlx_lm.convert --quantize --q-bits 4 ...`)
> and bundle that instead. The copied model directory inside
> `ios/ExpenseParser/ExpenseParser/` is gitignored.

## 4. Capabilities & Info.plist

No special entitlements needed. If Xcode complains about bitcode or the binary
size, set **Build Settings → Enable Bitcode = No** (default in modern Xcode).

## 5. Run

- Select your iPhone as the run destination, plug it in, Cmd-R.
- First launch takes a few seconds to load the model into GPU memory.
- Subsequent inferences should be well under 200 ms on an A17/M-class chip.

## File layout

```
ios/
├── README.md                                # this file
└── ExpenseParser/                           # Xcode project root
    ├── ExpenseParser.xcodeproj              # gitignored, generated per dev
    └── ExpenseParser/                       # tracked Swift sources
        ├── ExpenseParserApp.swift           # @main entry
        ├── ContentView.swift                # SwiftUI view + view-model
        ├── Model/
        │   ├── Expense.swift                # Codable struct + Category enum + JSON extractor
        │   ├── PromptTemplate.swift         # Gemma chat-template builder (mirrors prompt.py)
        │   └── ExpenseInference.swift       # MLX-Swift inference actor
        └── gemma3-270m-expense-merged/      # gitignored — bundled at build time
```

## Troubleshooting

- **"Model bundle not found"** — the model directory wasn't added as a folder
  reference, or the name doesn't match. Confirm a blue folder icon in the
  project navigator and that the folder name is `gemma3-270m-expense-merged`.
- **Tokenizer load errors** — ensure `tokenizer.json` and
  `tokenizer_config.json` are inside the bundled directory (they should be,
  after `mlx_lm.fuse`).
- **`ModelConfiguration(directory:)` not found** — `mlx-swift-examples` API
  has changed across versions. If the init signature differs, check the
  package's current `MLXLLM/Configuration.swift` and adjust accordingly; the
  concept (point it at a local directory) is stable even when the exact
  initializer name moves.
- **Slow first token** — expected; MLX JIT-compiles kernels on first use.
  Keep the `ExpenseInference` actor alive between parses (we do — it's
  owned by `ExpenseViewModel`).
