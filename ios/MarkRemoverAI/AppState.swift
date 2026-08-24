import SwiftUI

@MainActor
final class AppState: ObservableObject {
    enum Phase: Equatable {
        case launching
        case signedOut
        case signedIn(User)
    }

    @Published private(set) var phase: Phase = .launching
    @Published private(set) var workerOnline: Bool?

    var user: User? {
        if case .signedIn(let user) = phase { return user }
        return nil
    }

    var credits: Double { user?.credits ?? 0 }

    func restoreSession() async {
        // The Flask session cookie survives launches, so this is the whole
        // "am I still logged in" check. It is bounded so a stalled network
        // can never leave the app sitting on the splash screen.
        let status = try? await withTimeout(seconds: 10) {
            try await APIClient.shared.authStatus()
        }

        if let status, status.authenticated, let user = status.user {
            phase = .signedIn(user)
        } else {
            phase = .signedOut
        }
        await refreshHealth()
    }

    func refreshHealth() async {
        let health = try? await withTimeout(seconds: 10) {
            try await APIClient.shared.health()
        }
        workerOnline = health?.status == "ok"
    }

    func signIn(email: String, password: String) async throws {
        let user = try await APIClient.shared.login(email: email, password: password)
        phase = .signedIn(user)
    }

    func register(email: String, password: String, name: String) async throws {
        try await APIClient.shared.register(email: email, password: password, name: name)
    }

    func signOut() async {
        await APIClient.shared.logout()
        phase = .signedOut
    }

    /// Called after a job finishes so the balance on screen matches the server.
    func refreshUser() async {
        guard let status = try? await APIClient.shared.authStatus(),
              status.authenticated, let user = status.user else { return }
        phase = .signedIn(user)
    }
}


struct TimedOut: Error {}

/// Races work against a deadline. Whichever finishes first wins; the loser is
/// cancelled.
func withTimeout<T: Sendable>(
    seconds: Double,
    operation: @escaping @Sendable () async throws -> T
) async throws -> T {
    try await withThrowingTaskGroup(of: T.self) { group in
        group.addTask { try await operation() }
        group.addTask {
            try await Task.sleep(for: .seconds(seconds))
            throw TimedOut()
        }
        guard let first = try await group.next() else { throw TimedOut() }
        group.cancelAll()
        return first
    }
}
