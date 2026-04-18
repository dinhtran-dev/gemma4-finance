import Foundation
import MLX
import MLXLLM
import MLXLMCommon

/// Loads the fused Gemma model from the app bundle and runs inference.
/// The model directory must be added to the Xcode project as a *folder reference*
/// (blue folder) named `gemma3-270m-expense-merged` so MLX sees a single directory
/// containing `config.json`, tokenizer files, and weight shards at runtime.
actor ExpenseInference {
    enum InferenceError: Error, LocalizedError {
        case modelBundleMissing(searched: String, listing: String)

        var errorDescription: String? {
            switch self {
            case .modelBundleMissing(let searched, let listing):
                return """
                    Model bundle "\(searched)" not found in app bundle.
                    Add the folder to the Xcode target as a *folder reference* (blue folder).

                    Bundle contents:
                    \(listing)
                    """
            }
        }
    }

    private let bundleDirName: String
    private var container: ModelContainer?

    init(bundleDirName: String = "gemma3-270m-expense-merged") {
        self.bundleDirName = bundleDirName
    }

    var isLoaded: Bool { container != nil }

    func ensureLoaded() async throws {
        if container != nil { return }

        guard let url = Bundle.main.url(forResource: bundleDirName, withExtension: nil) else {
            throw InferenceError.modelBundleMissing(
                searched: bundleDirName,
                listing: Self.listBundle()
            )
        }

        MLX.GPU.set(cacheLimit: 32 * 1024 * 1024)

        let configuration = ModelConfiguration(directory: url)
        container = try await LLMModelFactory.shared.loadContainer(configuration: configuration)
    }

    /// Runs greedy decoding and returns the raw model output.
    func generate(_ userText: String, maxTokens: Int = 200) async throws -> String {
        try await ensureLoaded()
        guard let container else { return "" }

        let prompt = PromptTemplate.chat(userText)
        let params = GenerateParameters(maxTokens: maxTokens, temperature: 0.0)

        return try await container.perform { context in
            let userInput = UserInput(prompt: prompt)
            let input = try await context.processor.prepare(input: userInput)

            let didGenerate: ([Int]) -> GenerateDisposition = { _ in .more }
            let result = try MLXLMCommon.generate(
                input: input,
                parameters: params,
                context: context,
                didGenerate: didGenerate
            )
            return result.output
        }
    }

    /// Returns a parsed Expense if the output decoded successfully, plus the raw string.
    func parse(_ userText: String) async throws -> (Expense?, String) {
        let raw = try await generate(userText)
        return (ExpenseJSON.parse(raw), raw)
    }

    /// Walks the main bundle's resourcePath and returns a printable tree.
    /// Subdirectories are listed one level deep so we can spot the model folder
    /// (or its absence) without flooding the output.
    nonisolated private static func listBundle() -> String {
        guard let root = Bundle.main.resourcePath else { return "(no resourcePath)" }
        let fm = FileManager.default
        var lines: [String] = ["resourcePath: \(root)"]

        let top: [String]
        do {
            top = try fm.contentsOfDirectory(atPath: root).sorted()
        } catch {
            return lines.joined(separator: "\n") + "\nfailed to list: \(error.localizedDescription)"
        }

        for name in top {
            let full = (root as NSString).appendingPathComponent(name)
            var isDir: ObjCBool = false
            fm.fileExists(atPath: full, isDirectory: &isDir)
            if isDir.boolValue {
                lines.append("  \(name)/")
                if let inner = try? fm.contentsOfDirectory(atPath: full).sorted() {
                    for child in inner.prefix(20) {
                        lines.append("    \(child)")
                    }
                    if inner.count > 20 {
                        lines.append("    ... (+\(inner.count - 20) more)")
                    }
                }
            } else {
                lines.append("  \(name)")
            }
        }
        return lines.joined(separator: "\n")
    }
}
