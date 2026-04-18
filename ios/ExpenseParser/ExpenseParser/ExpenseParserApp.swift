import SwiftData
import SwiftUI

@main
struct ExpenseParserApp: App {
    @StateObject private var viewModel = ExpenseViewModel()

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(viewModel)
        }
        .modelContainer(for: ExpenseRecord.self)
    }
}
