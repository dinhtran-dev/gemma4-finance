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

From the repo root, the fused model lives at
`models/gemma3-270m-expense-merged/`. Copy it into the Xcode source folder:

```bash
cp -R models/gemma3-270m-expense-merged \
      ios/ExpenseParser/ExpenseParser/
```

Xcode's synchronized folders will pick it up automatically — you should see
`gemma3-270m-expense-merged` appear in the project navigator within a few
seconds. No drag-and-drop needed, no import dialog.

`ExpenseInference.resolveModelURL` accepts either bundle layout that Xcode
might produce:

- **Folder reference** — `<bundle>/gemma3-270m-expense-merged/{config.json, …}`
- **Flattened** (Xcode groups each file individually) —
  `<bundle>/{config.json, model.safetensors, …}`

So whichever way the auto-sync adds it, the app will find the weights as long
as `config.json` and `model.safetensors` end up in the bundle.

> Note: the bf16 model is ~540 MB. For smaller app size, regenerate the merged
> model in a quantized MLX format (`mlx_lm.convert --quantize --q-bits 4 ...`)
> and copy that instead. The model directory inside
> `ios/ExpenseParser/ExpenseParser/` is gitignored.

## 4. Permissions (speech + microphone)

The app records short voice descriptions and transcribes them on-device.
You need to declare microphone + speech recognition usage or iOS will kill
the app on first recording attempt.

In Xcode: **target → General → Info → Custom iOS Target Properties**, add:

| Key                                    | Value                                                |
|----------------------------------------|------------------------------------------------------|
| `NSMicrophoneUsageDescription`         | `ExpenseParser uses the microphone to record your spoken expense descriptions.` |
| `NSSpeechRecognitionUsageDescription`  | `ExpenseParser uses on-device speech recognition to turn your spoken expense into text.` |

(Or under **Build Settings**, set `INFOPLIST_KEY_NSMicrophoneUsageDescription`
and `INFOPLIST_KEY_NSSpeechRecognitionUsageDescription` to the same strings —
same effect.)

`SpeechTranscriber.beginRecording()` sets `requiresOnDeviceRecognition =
supportsOnDeviceRecognition`, so modern iPhones never send audio to Apple's
servers; devices without on-device support fall back to cloud recognition.

## 5. Capabilities & Info.plist

No other entitlements needed. If Xcode complains about bitcode or the binary
size, set **Build Settings → Enable Bitcode = No** (default in modern Xcode).

## 6. Run

- Select your iPhone as the run destination, plug it in, Cmd-R.
- First launch takes a few seconds to load the model into GPU memory.
- Tap the big mic button to start dictation; tap again to stop. The live
  transcript streams into the text field; edit if needed, then tap **Parse**.
- Every successful parse is persisted. Tap the clock icon in the top-right to
  open the history list; swipe a row to delete.
- Subsequent inferences should be well under 200 ms on an A17/M-class chip.

## Storage

Parsed expenses go into a SQLite database via SwiftData. On device the file
lives under the app's sandbox at:

```
Library/Application Support/default.store
```

(plus `.store-shm` / `.store-wal` journal files). Each row captures:

- `createdAt` — timestamp of when the parse was saved
- `amount`, `currency`, `categoryRaw`, `merchant`, `descriptionText`
- `dateText` — the natural-language date the user spoke ("yesterday")
- `rawInput` / `rawOutput` — the original utterance and the model's reply,
  for later re-training or error analysis

## File layout

```
ios/
├── README.md                                # this file
└── ExpenseParser/                           # Xcode project root
    ├── ExpenseParser.xcodeproj              # gitignored, generated per dev
    └── ExpenseParser/                       # tracked Swift sources
        ├── ExpenseParserApp.swift           # @main entry
        ├── ContentView.swift                # SwiftUI view + view-model
        ├── HistoryView.swift                # SwiftData-backed list of saved expenses
        ├── Model/
        │   ├── Expense.swift                # Codable struct + Category enum + JSON extractor
        │   ├── PromptTemplate.swift         # Gemma chat-template builder (mirrors prompt.py)
        │   ├── ExpenseInference.swift       # MLX-Swift inference actor
        │   ├── ExpenseRecord.swift          # @Model — SwiftData/SQLite persistence
        │   └── SpeechTranscriber.swift      # SFSpeechRecognizer live dictation
        └── gemma3-270m-expense-merged/      # gitignored — bundled at build time
```

## Troubleshooting

- **"Model bundle not found"** — the load-state card prints the full contents
  of `Bundle.main.resourcePath`. If `config.json` / `model.safetensors` aren't
  in the listing, the copy didn't make it into the target. Confirm the folder
  exists on disk at `ios/ExpenseParser/ExpenseParser/gemma3-270m-expense-merged/`
  and that its files show target membership in Xcode's File Inspector.
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
