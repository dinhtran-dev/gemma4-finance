import Foundation

/// Must stay in sync with `src/expense_parser/prompt.py`.
enum PromptTemplate {
    static let systemInstruction = """
        Extract expense details as JSON with keys: \
        amount (number or null), currency (ISO code, default USD), \
        category (one of: food_drink, groceries, transport, travel, \
        entertainment, shopping, bills, health, subscriptions, other), \
        merchant (string or null), description (string or null), \
        date (string or null).
        """

    static func userPrompt(_ userText: String) -> String {
        "\(systemInstruction)\nInput: \(userText.trimmingCharacters(in: .whitespacesAndNewlines))"
    }

    /// Wraps the user prompt in Gemma chat-template turn markers.
    static func chat(_ userText: String) -> String {
        let user = userPrompt(userText)
        return "<start_of_turn>user\n\(user)\n<end_of_turn>\n<start_of_turn>model\n"
    }
}
