import Foundation
import SwiftData

/// Persisted record of a parsed expense. Stored by SwiftData in a SQLite
/// database inside the app's Application Support directory.
@Model
final class ExpenseRecord {
    @Attribute(.unique) var id: UUID
    var createdAt: Date
    var amount: Double?
    var currency: String?
    var categoryRaw: String?
    var merchant: String?
    var descriptionText: String?
    /// Natural-language date the user mentioned ("yesterday", "last Tuesday").
    var dateText: String?
    var rawInput: String
    var rawOutput: String

    var category: Category? {
        get { categoryRaw.flatMap(Category.init(rawValue:)) }
        set { categoryRaw = newValue?.rawValue }
    }

    init(
        id: UUID = UUID(),
        createdAt: Date = .now,
        expense: Expense,
        rawInput: String,
        rawOutput: String
    ) {
        self.id = id
        self.createdAt = createdAt
        self.amount = expense.amount
        self.currency = expense.currency
        self.categoryRaw = expense.category?.rawValue
        self.merchant = expense.merchant
        self.descriptionText = expense.description
        self.dateText = expense.date
        self.rawInput = rawInput
        self.rawOutput = rawOutput
    }
}
