import AuthenticationServices
import Foundation

/// Google sign-in without pulling in an SDK: the site's own OAuth flow runs
/// inside an authentication session, and the server hands back a one-time code
/// because the cookie that web view receives isn't ours to keep.
@MainActor
final class GoogleSignIn: NSObject, ASWebAuthenticationPresentationContextProviding {
    static let callbackScheme = "markremoverai"

    private var session: ASWebAuthenticationSession?

    func start(baseURL: URL) async throws -> String {
        let url = baseURL
            .appendingPathComponent("auth/google")
            .appending(queryItems: [URLQueryItem(name: "native", value: "1")])

        let callback: URL = try await withCheckedThrowingContinuation { continuation in
            let session = ASWebAuthenticationSession(
                url: url,
                callbackURLScheme: Self.callbackScheme
            ) { callbackURL, error in
                if let callbackURL {
                    continuation.resume(returning: callbackURL)
                } else if let error {
                    continuation.resume(throwing: error)
                } else {
                    continuation.resume(throwing: SocialAuthError.cancelled)
                }
            }
            session.presentationContextProvider = self
            // Keep the shared cookie jar so "sign in with the account you're
            // already using in Safari" works the way people expect.
            session.prefersEphemeralWebBrowserSession = false
            self.session = session
            session.start()
        }

        let components = URLComponents(url: callback, resolvingAgainstBaseURL: false)
        let items = components?.queryItems ?? []

        if let error = items.first(where: { $0.name == "error" })?.value {
            throw SocialAuthError.server(error)
        }
        guard let code = items.first(where: { $0.name == "code" })?.value, !code.isEmpty else {
            throw SocialAuthError.server("No sign-in code came back.")
        }
        return code
    }

    nonisolated func presentationAnchor(for session: ASWebAuthenticationSession) -> ASPresentationAnchor {
        MainActor.assumeIsolated {
            let scene = UIApplication.shared.connectedScenes
                .compactMap { $0 as? UIWindowScene }
                .first { $0.activationState == .foregroundActive }
            return scene?.keyWindow ?? ASPresentationAnchor()
        }
    }
}

enum SocialAuthError: LocalizedError {
    case cancelled
    case missingToken
    case server(String)

    var errorDescription: String? {
        switch self {
        case .cancelled: return nil                     // user backed out; say nothing
        case .missingToken: return "Apple didn't return a sign-in token."
        case .server(let message): return message
        }
    }

    /// ASWebAuthenticationSession reports a user-initiated dismissal as an
    /// error, and that shouldn't surface as a failure banner.
    static func isCancellation(_ error: Error) -> Bool {
        if case SocialAuthError.cancelled = error { return true }
        let nsError = error as NSError
        return nsError.domain == ASWebAuthenticationSessionErrorDomain
            && nsError.code == ASWebAuthenticationSessionError.canceledLogin.rawValue
    }
}
