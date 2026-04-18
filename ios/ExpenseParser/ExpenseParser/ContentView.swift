import Combine
import SwiftUI

@MainActor
final class ExpenseViewModel: ObservableObject {
    enum LoadState: Equatable {
        case idle
        case loading
        case loaded
        case failed(String)
    }

    @Published var input: String = ""
    @Published var expense: Expense?
    @Published var rawOutput: String = ""
    @Published var isParsing: Bool = false
    @Published var errorMessage: String?
    @Published var loadState: LoadState = .idle

    private let inference = ExpenseInference()

    func preload() async {
        if loadState == .loading || loadState == .loaded { return }
        loadState = .loading
        do {
            try await inference.ensureLoaded()
            loadState = .loaded
        } catch {
            loadState = .failed(error.localizedDescription)
        }
    }

    func parse() async {
        let text = input.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return }

        isParsing = true
        errorMessage = nil
        expense = nil
        rawOutput = ""
        defer { isParsing = false }

        do {
            let (parsed, raw) = try await inference.parse(text)
            self.rawOutput = raw
            self.expense = parsed
            if parsed == nil {
                errorMessage = "Model output did not parse as valid JSON."
            }
        } catch {
            errorMessage = error.localizedDescription
        }
    }
}

struct ContentView: View {
    @EnvironmentObject var vm: ExpenseViewModel

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    loadStateCard
                    inputCard
                    if vm.isParsing {
                        HStack(spacing: 8) {
                            ProgressView()
                            Text("Parsing on device...")
                                .foregroundStyle(.secondary)
                        }
                        .padding(.horizontal, 4)
                    }
                    if let expense = vm.expense {
                        resultCard(expense)
                    }
                    if let msg = vm.errorMessage {
                        Text(msg)
                            .font(.footnote)
                            .foregroundStyle(.red)
                            .padding(.horizontal, 4)
                    }
                    if !vm.rawOutput.isEmpty {
                        rawCard
                    }
                }
                .padding()
            }
            .navigationTitle("Expense Parser")
        }
        .task { await vm.preload() }
    }

    @ViewBuilder
    private var loadStateCard: some View {
        switch vm.loadState {
        case .idle, .loading:
            HStack(spacing: 8) {
                ProgressView()
                Text("Loading model on device...")
                    .foregroundStyle(.secondary)
            }
            .padding()
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(Color(.secondarySystemBackground), in: RoundedRectangle(cornerRadius: 12))
        case .loaded:
            HStack(spacing: 8) {
                Image(systemName: "checkmark.seal.fill")
                    .foregroundStyle(.green)
                Text("Model loaded")
                    .foregroundStyle(.secondary)
            }
            .padding(.horizontal, 4)
        case .failed(let message):
            VStack(alignment: .leading, spacing: 6) {
                Label("Model failed to load", systemImage: "exclamationmark.triangle.fill")
                    .foregroundStyle(.red)
                    .font(.headline)
                Text(message)
                    .font(.system(.footnote, design: .monospaced))
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
            .padding()
            .background(Color(.secondarySystemBackground), in: RoundedRectangle(cornerRadius: 12))
        }
    }

    private var inputCard: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Describe an expense")
                .font(.headline)
            TextField("e.g. grabbed lunch at Chipotle for $14", text: $vm.input, axis: .vertical)
                .lineLimit(2...5)
                .textFieldStyle(.roundedBorder)
                .submitLabel(.send)
                .onSubmit { Task { await vm.parse() } }
            Button {
                Task { await vm.parse() }
            } label: {
                Label("Parse", systemImage: "sparkles")
                    .frame(maxWidth: .infinity)
            }
            .buttonStyle(.borderedProminent)
            .disabled(parseDisabled)
        }
        .padding()
        .background(Color(.secondarySystemBackground), in: RoundedRectangle(cornerRadius: 12))
    }

    private var parseDisabled: Bool {
        if vm.isParsing { return true }
        if vm.input.trimmingCharacters(in: .whitespaces).isEmpty { return true }
        if case .failed = vm.loadState { return true }
        return false
    }

    private func resultCard(_ e: Expense) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Parsed")
                .font(.headline)
            row("Amount", e.amount.map { String(format: "%.2f", $0) } ?? "—")
            row("Currency", e.currency ?? "—")
            row("Category", e.category?.display ?? "—")
            row("Merchant", e.merchant ?? "—")
            row("Description", e.description ?? "—")
            row("Date", e.date ?? "—")
        }
        .padding()
        .background(Color(.secondarySystemBackground), in: RoundedRectangle(cornerRadius: 12))
    }

    private func row(_ label: String, _ value: String) -> some View {
        HStack(alignment: .top) {
            Text(label)
                .foregroundStyle(.secondary)
                .frame(width: 100, alignment: .leading)
            Text(value)
                .textSelection(.enabled)
            Spacer(minLength: 0)
        }
        .font(.subheadline)
    }

    private var rawCard: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Raw model output")
                .font(.headline)
            Text(vm.rawOutput)
                .font(.system(.footnote, design: .monospaced))
                .textSelection(.enabled)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding()
        .background(Color(.secondarySystemBackground), in: RoundedRectangle(cornerRadius: 12))
    }
}

#Preview {
    ContentView().environmentObject(ExpenseViewModel())
}
