import Foundation

/// Mirrors `src/expense_parser/schema.py`.
enum Category: String, Codable, CaseIterable {
    case foodDrink = "food_drink"
    case groceries
    case transport
    case travel
    case entertainment
    case shopping
    case bills
    case health
    case subscriptions
    case other

    var display: String {
        switch self {
        case .foodDrink: return "Food & Drink"
        case .groceries: return "Groceries"
        case .transport: return "Transport"
        case .travel: return "Travel"
        case .entertainment: return "Entertainment"
        case .shopping: return "Shopping"
        case .bills: return "Bills"
        case .health: return "Health"
        case .subscriptions: return "Subscriptions"
        case .other: return "Other"
        }
    }
}

struct Expense: Codable, Equatable {
    var amount: Double?
    var currency: String?
    var category: Category?
    var merchant: String?
    var description: String?
    var date: String?

    enum CodingKeys: String, CodingKey {
        case amount, currency, category, merchant, description, date
    }

    init(
        amount: Double? = nil,
        currency: String? = "USD",
        category: Category? = .other,
        merchant: String? = nil,
        description: String? = nil,
        date: String? = nil
    ) {
        self.amount = amount
        self.currency = currency
        self.category = category
        self.merchant = merchant
        self.description = description
        self.date = date
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        amount = try c.decodeIfPresent(Double.self, forKey: .amount)
        currency = (try c.decodeIfPresent(String.self, forKey: .currency))?.uppercased() ?? "USD"
        category = try c.decodeIfPresent(Category.self, forKey: .category) ?? .other
        merchant = try c.decodeIfPresent(String.self, forKey: .merchant)
        description = try c.decodeIfPresent(String.self, forKey: .description)
        date = try c.decodeIfPresent(String.self, forKey: .date)
    }
}

enum ExpenseJSON {
    /// Extracts the first `{...}` block from model output and decodes it.
    static func parse(_ raw: String) -> Expense? {
        guard let start = raw.firstIndex(of: "{"),
              let end = raw.lastIndex(of: "}"),
              start < end else { return nil }
        let slice = raw[start...end]
        guard let data = String(slice).data(using: .utf8) else { return nil }
        return try? JSONDecoder().decode(Expense.self, from: data)
    }
}
