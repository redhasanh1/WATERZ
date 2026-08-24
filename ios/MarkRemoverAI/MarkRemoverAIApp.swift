import SwiftUI

@main
struct MarkRemoverAIApp: App {
    @StateObject private var appState = AppState()

    var body: some Scene {
        WindowGroup {
            RootView()
                .environmentObject(appState)
                .tint(Theme.accent)
        }
    }
}

struct RootView: View {
    @EnvironmentObject private var appState: AppState

    var body: some View {
        Group {
            switch appState.phase {
            case .launching:
                VStack(spacing: 16) {
                    ProgressView().tint(Theme.accent)
                    Text("MarkRemoverAI").font(.headline).foregroundStyle(.secondary)
                }
            case .signedOut:
                LoginView()
            case .signedIn:
                HomeView()
            }
        }
        .animation(.easeInOut(duration: 0.2), value: appState.phase)
        .task { await appState.restoreSession() }
    }
}
