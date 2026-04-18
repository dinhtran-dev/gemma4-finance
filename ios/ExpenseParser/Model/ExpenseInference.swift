import Foundation
import MLX
import MLXLLM
import MLXLMCommon

/// Loads the fused Gemma model from the app bundle and runs inference.
/// The model directory must be added to the Xcode project as a *folder reference*
/// (blue folder) named `gemma3-270m-expense-merged` so MLX sees a single directory
/// containing `config.json`, tokenizer files, and weight shards at runtime.
actor ExpenseInference {
    enum InferenceError: Error {
        case modelBundleMissing
    }

    private let bundleDirName: String
    private var container: ModelContainer?

    init(bundleDirName: String = "gemma3-270m-expense-merged") {
        self.bundleDirName = bundleDirName
    }

    func ensureLoaded() async throws {
        if container != nil { return }

        guard let url = Bundle.main.url(forResource: bundleDirName, withExtension: nil) else {
            throw InferenceError.modelBundleMissing
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
            let input = try await context.processor.prepare(input: .init(prompt: prompt))
            let result = try MLXLMCommon.generate(
                input: input,
                parameters: params,
                context: context
            ) { _ in .more }
            return result.output
        }
    }

    /// Returns a parsed Expense if the output decoded successfully, plus the raw string.
    func parse(_ userText: String) async throws -> (Expense?, String) {
        let raw = try await generate(userText)
        return (ExpenseJSON.parse(raw), raw)
    }
}
