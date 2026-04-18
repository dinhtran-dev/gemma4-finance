import SwiftUI

@MainActor
final class ExpenseViewModel: ObservableObject {
    @Published var input: String = ""
    @Published var expense: Expense?
    @Published var rawOutput: String = ""
    @Published var isLoading: Bool = false
    @Published var errorMessage: String?

    private let inference = ExpenseInference()

    func parse() async {
        let text = input.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return }

        isLoading = true
        errorMessage = nil
        expense = nil
        rawOutput = ""
        defer { isLoading = false }

        do {
            let (parsed, raw) = try await inference.parse(text)
            self.rawOutput = raw
            self.expense = parsed
            if parsed == nil {
                errorMessage = "Model output did not parse as valid JSON."
            }
        } catch ExpenseInference.InferenceError.modelBundleMissing {
            errorMessage = "Model bundle not found. Add gemma3-270m-expense-merged to the app target as a folder reference."
        } catch {
            errorMessage = "Inference failed: \(error.localizedDescription)"
        }
    }
}

struct ContentView: View {
    @EnvironmentObject var vm: ExpenseViewModel

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    inputCard
                    if vm.isLoading {
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
            .disabled(vm.isLoading || vm.input.trimmingCharacters(in: .whitespaces).isEmpty)
        }
        .padding()
        .background(Color(.secondarySystemBackground), in: RoundedRectangle(cornerRadius: 12))
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
