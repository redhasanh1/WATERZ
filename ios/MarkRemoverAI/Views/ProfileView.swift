import SwiftUI

struct ProfileView: View {
    /// As a tab there's nothing to dismiss, so the Done button is dropped.
    var embedded = false

    @EnvironmentObject private var appState: AppState
    @Environment(\.dismiss) private var dismiss

    @State private var showPaywall = false
    @State private var showConsole = false
    @State private var showSignOutConfirm = false
    @State private var showHostPicker = false
    @State private var host = APIClient.currentBaseURL

    var body: some View {
        NavigationStack {
            List {
                Section {
                    HStack(spacing: 14) {
                        Circle()
                            .fill(Theme.heroGradient)
                            .frame(width: 54, height: 54)
                            .overlay(
                                Text(initials)
                                    .font(.title3.bold())
                                    .foregroundStyle(.white)
                            )

                        VStack(alignment: .leading, spacing: 3) {
                            Text(appState.user?.name ?? "Signed in")
                                .font(.headline)
                            Text(appState.user?.email ?? "")
                                .font(.subheadline)
                                .foregroundStyle(.secondary)
                                .lineLimit(1)
                        }
                    }
                    .padding(.vertical, 6)
                }

                Section("Credits") {
                    HStack {
                        Label("Balance", systemImage: "bolt.fill")
                        Spacer()
                        Text("\(Int(appState.credits))")
                            .font(.headline.monospacedDigit())
                            .foregroundStyle(Theme.accent)
                    }
                    Button {
                        showPaywall = true
                    } label: {
                        Label("Get more credits", systemImage: "plus.circle.fill")
                    }
                    Text("One credit removes an object from one video.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }

                Section("Service") {
                    HStack {
                        Label("GPU workers", systemImage: "cpu")
                        Spacer()
                        HStack(spacing: 6) {
                            Circle()
                                .fill(appState.workerOnline == true ? Theme.positive : Theme.warning)
                                .frame(width: 8, height: 8)
                            Text(appState.workerOnline == true ? "Online" : "Unreachable")
                                .foregroundStyle(.secondary)
                        }
                        .font(.subheadline)
                    }

                    Button { showHostPicker = true } label: {
                        VStack(alignment: .leading, spacing: 2) {
                            Label("Server", systemImage: "network")
                            Text(host.replacingOccurrences(of: "https://", with: ""))
                                .font(.caption)
                                .foregroundStyle(.secondary)
                                .lineLimit(1)
                        }
                    }

                    Button { showConsole = true } label: {
                        Label("Console", systemImage: "text.alignleft")
                    }
                }

                Section("Help") {
                    Button {
                        openSupportMail()
                    } label: {
                        Label("Email support", systemImage: "envelope")
                    }
                    Text(SupportMail.address)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .textSelection(.enabled)
                }

                Section {
                    Link(destination: URL(string: "https://markremoverai.com/terms")!) {
                        Label("Terms of service", systemImage: "doc.text")
                    }
                    Link(destination: URL(string: "https://markremoverai.com/privacy")!) {
                        Label("Privacy policy", systemImage: "hand.raised")
                    }
                }

                Section {
                    Button(role: .destructive) {
                        showSignOutConfirm = true
                    } label: {
                        Label("Sign out", systemImage: "rectangle.portrait.and.arrow.right")
                    }
                }
            }
            .navigationTitle("Profile")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                if !embedded {
                    ToolbarItem(placement: .topBarTrailing) {
                        Button("Done") { dismiss() }
                    }
                }
            }
            .sheet(isPresented: $showPaywall) { PaywallView() }
            .sheet(isPresented: $showConsole) { ConsoleView() }
            .confirmationDialog("Sign out?", isPresented: $showSignOutConfirm, titleVisibility: .visible) {
                Button("Sign out", role: .destructive) {
                    Task {
                        await appState.signOut()
                        if !embedded { dismiss() }
                    }
                }
                Button("Cancel", role: .cancel) {}
            }
            .confirmationDialog("Server", isPresented: $showHostPicker, titleVisibility: .visible) {
                Button("Railway (recommended)") { select(APIClient.railwayBase) }
                Button("markremoverai.com") { select(APIClient.productionBase) }
                Button("Cancel", role: .cancel) {}
            } message: {
                Text("The app talks to Railway directly so a blocked domain can't take it offline.")
            }
            .task { await appState.refreshUser() }
        }
    }

    private var initials: String {
        let source = appState.user?.name?.trimmingCharacters(in: .whitespaces).isEmpty == false
            ? appState.user!.name!
            : (appState.user?.email ?? "?")
        return String(source.prefix(1)).uppercased()
    }

    /// Pre-fills the details we'd otherwise have to ask for. Nothing is sent
    /// until the user hits send in Mail — this only composes a draft.
    private func openSupportMail() {
        let url = SupportMail.composeURL(
            email: appState.user?.email,
            credits: appState.credits,
            host: host,
            recentErrors: LogStore.shared.recentErrors(limit: 6)
        )
        guard let url else { return }
        UIApplication.shared.open(url)
    }

    private func select(_ value: String) {
        APIClient.setBaseURL(value)
        host = value
        Task { await appState.refreshHealth() }
    }
}
