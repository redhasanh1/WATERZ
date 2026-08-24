import SwiftUI

@main
struct MarkRemoverAIApp: App {
    @StateObject private var appState = AppState()
    @StateObject private var store = Store()

    var body: some Scene {
        WindowGroup {
            RootView()
                .environmentObject(appState)
                .environmentObject(store)
                .tint(Theme.accent)
                .task {
                    APIClient.bootstrap()
                    // Transactions that land while the app was closed need a
                    // place to write the new balance.
                    Store.activeAppState = appState
                    await store.redeemUnfinished(appState: appState)
                }
        }
    }
}

struct RootView: View {
    @EnvironmentObject private var appState: AppState

    var body: some View {
        Group {
            #if DEBUG
            // `-show_paywall 1` opens the store without a login, so the
            // StoreKit config can be exercised on a simulator.
            if UserDefaults.standard.bool(forKey: "show_paywall") {
                PaywallView(isPresented: false)
            } else {
                content
            }
            #else
            content
            #endif
        }
        .animation(.easeInOut(duration: 0.2), value: appState.phase)
        .task { await appState.restoreSession() }
    }

    @ViewBuilder
    private var content: some View {
        Group {
            switch appState.phase {
            case .launching:
                VStack(spacing: 16) {
                    ProgressView().tint(Theme.accent)
                    Text("ObjectRemoverAI").font(.headline).foregroundStyle(.secondary)
                }
            case .signedOut:
                LoginView()
            case .signedIn:
                MainTabs()
            }
        }
    }
}


/// Bottom tabs so the app isn't a single dead-end screen. Create is the job;
/// the other two are the things people reach for when it doesn't go smoothly.
struct MainTabs: View {
    @EnvironmentObject private var appState: AppState
    @State private var selection = 0

    var body: some View {
        TabView(selection: $selection) {
            HomeView()
                .tabItem { Label("Create", systemImage: "wand.and.stars") }
                .tag(0)

            BackgroundView()
                .tabItem { Label("Background", systemImage: "person.and.background.dotted") }
                .tag(1)
                .tint(Theme.orange)

            GuideView()
                .tabItem { Label("Guide", systemImage: "questionmark.circle") }
                .tag(2)

            ProfileView(embedded: true)
                .tabItem { Label("Profile", systemImage: "person.crop.circle") }
                .tag(3)
                .badge(appState.credits < 1 ? "!" : nil)
        }
    }
}
