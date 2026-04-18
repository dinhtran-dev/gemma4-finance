import Combine
import SwiftData
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
    @Published var lastSavedAt: Date?

    private let inference = ExpenseInference()
    private var modelContext: ModelContext?

    func attach(modelContext: ModelContext) {
        self.modelContext = modelContext
    }

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
            if let parsed {
                persist(parsed, rawInput: text, rawOutput: raw)
            } else {
                errorMessage = "Model output did not parse as valid JSON."
            }
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    private func persist(_ e: Expense, rawInput: String, rawOutput: String) {
        guard let modelContext else { return }
        let record = ExpenseRecord(expense: e, rawInput: rawInput, rawOutput: rawOutput)
        modelContext.insert(record)
        do {
            try modelContext.save()
            lastSavedAt = record.createdAt
        } catch {
            errorMessage = "Saved to memory but failed to write to SQLite: \(error.localizedDescription)"
        }
    }
}

struct ContentView: View {
    @Environment(\.modelContext) private var modelContext
    @EnvironmentObject var vm: ExpenseViewModel
    @StateObject private var transcriber = SpeechTranscriber()

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
                    if let msg = transcriber.errorMessage {
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
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    NavigationLink {
                        HistoryView()
                    } label: {
                        Image(systemName: "clock.arrow.circlepath")
                    }
                    .accessibilityLabel("History")
                }
            }
        }
        .task {
            vm.attach(modelContext: modelContext)
            await vm.preload()
        }
        .onChange(of: transcriber.transcript) { _, newValue in
            vm.input = newValue
        }
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
        VStack(spacing: 16) {
            Text(transcriber.isRecording ? "Listening…" : "Tap the mic and describe an expense")
                .font(.headline)
                .foregroundStyle(.primary)

            micButton

            TextField("e.g. grabbed lunch at Chipotle for $14",
                      text: $vm.input, axis: .vertical)
                .lineLimit(2...5)
                .textFieldStyle(.roundedBorder)
                .disabled(transcriber.isRecording)

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
        .frame(maxWidth: .infinity)
        .background(Color(.secondarySystemBackground), in: RoundedRectangle(cornerRadius: 12))
    }

    private var micButton: some View {
        Button {
            Task { await transcriber.toggle() }
        } label: {
            ZStack {
                Circle()
                    .fill(transcriber.isRecording ? Color.red : Color.accentColor)
                    .frame(width: 96, height: 96)
                    .shadow(radius: transcriber.isRecording ? 8 : 2)
                Image(systemName: transcriber.isRecording ? "stop.fill" : "mic.fill")
                    .foregroundStyle(.white)
                    .font(.system(size: 36, weight: .semibold))
            }
        }
        .buttonStyle(.plain)
        .accessibilityLabel(transcriber.isRecording ? "Stop recording" : "Start recording")
        .disabled(vm.isParsing)
    }

    private var parseDisabled: Bool {
        if vm.isParsing { return true }
        if transcriber.isRecording { return true }
        if vm.input.trimmingCharacters(in: .whitespaces).isEmpty { return true }
        if case .failed = vm.loadState { return true }
        return false
    }

    private func resultCard(_ e: Expense) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text("Parsed")
                    .font(.headline)
                Spacer()
                if vm.lastSavedAt != nil {
                    Label("Saved", systemImage: "checkmark.circle.fill")
                        .labelStyle(.titleAndIcon)
                        .font(.caption)
                        .foregroundStyle(.green)
                }
            }
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
    ContentView()
        .environmentObject(ExpenseViewModel())
        .modelContainer(for: ExpenseRecord.self, inMemory: true)
}
