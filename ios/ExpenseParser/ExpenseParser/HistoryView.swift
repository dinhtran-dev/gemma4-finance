import SwiftData
import SwiftUI

struct HistoryView: View {
    @Environment(\.modelContext) private var modelContext
    @Query(sort: \ExpenseRecord.createdAt, order: .reverse) private var records: [ExpenseRecord]

    private static let dateFormatter: DateFormatter = {
        let f = DateFormatter()
        f.dateStyle = .medium
        f.timeStyle = .short
        return f
    }()

    var body: some View {
        Group {
            if records.isEmpty {
                ContentUnavailableView(
                    "No expenses yet",
                    systemImage: "tray",
                    description: Text("Parsed expenses show up here.")
                )
            } else {
                List {
                    ForEach(records) { record in
                        row(record)
                    }
                    .onDelete(perform: delete)
                }
            }
        }
        .navigationTitle("History")
        .toolbar {
            if !records.isEmpty {
                EditButton()
            }
        }
    }

    @ViewBuilder
    private func row(_ r: ExpenseRecord) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack(alignment: .firstTextBaseline) {
                Text(amountString(r))
                    .font(.headline)
                Spacer()
                Text(Self.dateFormatter.string(from: r.createdAt))
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            Text(r.merchant ?? r.descriptionText ?? r.rawInput)
                .font(.subheadline)
                .foregroundStyle(.secondary)
                .lineLimit(2)
            HStack(spacing: 8) {
                if let cat = r.category {
                    categoryPill(cat.display)
                }
                if let d = r.dateText, !d.isEmpty {
                    Text(d)
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }
            }
        }
        .padding(.vertical, 2)
    }

    private func categoryPill(_ text: String) -> some View {
        Text(text)
            .font(.caption2.weight(.medium))
            .padding(.horizontal, 8)
            .padding(.vertical, 2)
            .background(Color.accentColor.opacity(0.15), in: Capsule())
            .foregroundStyle(Color.accentColor)
    }

    private func amountString(_ r: ExpenseRecord) -> String {
        guard let amount = r.amount else { return "—" }
        let symbol: String
        switch (r.currency ?? "USD").uppercased() {
        case "USD": symbol = "$"
        case "EUR": symbol = "€"
        case "GBP": symbol = "£"
        case "JPY": symbol = "¥"
        default: symbol = (r.currency ?? "") + " "
        }
        return String(format: "%@%.2f", symbol, amount)
    }

    private func delete(at offsets: IndexSet) {
        for index in offsets {
            modelContext.delete(records[index])
        }
    }
}

#Preview {
    NavigationStack {
        HistoryView()
    }
    .modelContainer(for: ExpenseRecord.self, inMemory: true)
}
