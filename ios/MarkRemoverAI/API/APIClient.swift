import Foundation

enum APIError: LocalizedError {
    case http(Int, String)
    case transport(Error)
    case decoding
    case needsVerification(String)
    case notAuthenticated
    case outOfCredits
    case workerOffline

    var errorDescription: String? {
        switch self {
        case .http(_, let message): return message
        case .transport: return "Can't reach MarkRemoverAI. Check your connection."
        case .decoding: return "The server sent something unexpected."
        case .needsVerification(let email): return "Verify \(email) first — check your inbox."
        case .notAuthenticated: return "Sign in to keep going."
        case .outOfCredits: return "You're out of credits."
        case .workerOffline: return "The GPU worker isn't answering. Try again in a moment."
        }
    }
}

actor APIClient {
    static let shared = APIClient()

    /// Overridable so a build can be pointed at the Railway host directly —
    /// some networks (school/library filters) refuse to resolve the apex domain.
    static let baseURLDefaultsKey = "api_base_url"
    static let productionBase = "https://markremoverai.com"
    static let railwayBase = "https://user-interface-ui-production.up.railway.app"

    private let session: URLSession

    private init() {
        let config = URLSessionConfiguration.default
        // The backend authenticates with a Flask session cookie, so the cookie
        // jar is the whole login state. It persists across launches for free.
        config.httpCookieAcceptPolicy = .always
        config.httpShouldSetCookies = true
        config.httpCookieStorage = .shared
        config.timeoutIntervalForRequest = 30
        config.timeoutIntervalForResource = 600
        // Deliberately false: with it on, an unresolvable host reports
        // "waiting" forever instead of failing, which strands the launch check.
        config.waitsForConnectivity = false
        session = URLSession(configuration: config)
    }

    nonisolated var baseURL: URL {
        let stored = UserDefaults.standard.string(forKey: Self.baseURLDefaultsKey)
        return URL(string: stored ?? Self.productionBase)!
    }

    /// Some networks (school and library DNS filters) refuse to resolve the
    /// apex domain while the Railway host answers fine. Let the user flip.
    nonisolated static func setBaseURL(_ value: String) {
        UserDefaults.standard.set(value, forKey: baseURLDefaultsKey)
    }

    nonisolated static var currentBaseURL: String {
        UserDefaults.standard.string(forKey: baseURLDefaultsKey) ?? productionBase
    }

    // MARK: - Core

    private func request(
        _ path: String,
        method: String = "GET",
        json: [String: Any]? = nil
    ) async throws -> Data {
        var req = URLRequest(url: baseURL.appendingPathComponent(path))
        req.httpMethod = method
        req.setValue("application/json", forHTTPHeaderField: "Accept")
        if let json {
            req.setValue("application/json", forHTTPHeaderField: "Content-Type")
            req.httpBody = try JSONSerialization.data(withJSONObject: json)
        }

        let data: Data
        let response: URLResponse
        do {
            (data, response) = try await session.data(for: req)
        } catch {
            throw APIError.transport(error)
        }

        let code = (response as? HTTPURLResponse)?.statusCode ?? 0
        guard (200..<300).contains(code) else {
            throw Self.mapFailure(code: code, data: data)
        }
        return data
    }

    private static func mapFailure(code: Int, data: Data) -> APIError {
        let decoded = try? JSONDecoder().decode(APIErrorResponse.self, from: data)
        let message = decoded?.text ?? "Request failed (\(code))."

        switch code {
        case 401: return .notAuthenticated
        case 402: return .outOfCredits
        case 403 where decoded?.needsVerification == true:
            return .needsVerification(message)
        case 504: return .workerOffline
        default: return .http(code, message)
        }
    }

    private func decode<T: Decodable>(_ type: T.Type, from data: Data) throws -> T {
        do { return try JSONDecoder().decode(T.self, from: data) }
        catch { throw APIError.decoding }
    }

    // MARK: - Health

    func health() async throws -> HealthResponse {
        try decode(HealthResponse.self, from: try await request("api/health"))
    }

    // MARK: - Auth

    func authStatus() async throws -> AuthStatusResponse {
        try decode(AuthStatusResponse.self, from: try await request("api/auth/status"))
    }

    func login(email: String, password: String) async throws -> User {
        let data = try await request(
            "api/auth/login",
            method: "POST",
            json: ["email": email, "password": password]
        )
        return try decode(AuthSuccessResponse.self, from: data).user
    }

    func register(email: String, password: String, name: String) async throws {
        _ = try await request(
            "api/auth/register",
            method: "POST",
            json: ["email": email, "password": password, "name": name]
        )
    }

    /// Trades the one-time code from the native Google flow for a real session
    /// cookie on this URLSession.
    func exchangeGoogleCode(_ code: String) async throws -> User {
        let data = try await request(
            "api/auth/exchange",
            method: "POST",
            json: ["code": code]
        )
        return try decode(AuthSuccessResponse.self, from: data).user
    }

    /// Apple only reveals the name and address on the first authorization, so
    /// they are forwarded alongside the token rather than looked up later.
    func signInWithApple(identityToken: String, email: String?, name: String?) async throws -> User {
        var payload: [String: Any] = ["identity_token": identityToken]
        if let email, !email.isEmpty { payload["email"] = email }
        if let name, !name.isEmpty { payload["name"] = name }

        let data = try await request("api/auth/apple", method: "POST", json: payload)
        return try decode(AuthSuccessResponse.self, from: data).user
    }

    func logout() async {
        _ = try? await request("api/auth/logout", method: "POST")
        // Clear the jar too, so a stale cookie can't resurrect the session.
        HTTPCookieStorage.shared.cookies(for: baseURL)?.forEach {
            HTTPCookieStorage.shared.deleteCookie($0)
        }
    }

    // MARK: - Upload

    /// Mirrors the web flow: ask the API for a B2 ticket, PUT the bytes straight
    /// to B2 (Railway never sees them), then tell the API it landed.
    func upload(fileURL: URL, filename: String, contentType: String) async throws -> String {
        let ticket: UploadURLResponse = try decode(
            UploadURLResponse.self,
            from: try await request(
                "api/get-upload-url",
                method: "POST",
                json: ["filename": filename, "content_type": contentType]
            )
        )

        try await uploadToB2(fileURL: fileURL, ticket: ticket, contentType: contentType)

        let done: UploadCompleteResponse = try decode(
            UploadCompleteResponse.self,
            from: try await request(
                "api/upload-complete",
                method: "POST",
                json: [
                    "task_id": ticket.taskId,
                    "remote_path": ticket.remotePath,
                    "cdn_url": ticket.cdnURL,
                    "filename": filename
                ]
            )
        )
        return done.taskId
    }

    private func uploadToB2(fileURL: URL, ticket: UploadURLResponse, contentType: String) async throws {
        guard let url = URL(string: ticket.uploadURL) else { throw APIError.decoding }

        var req = URLRequest(url: url)
        req.httpMethod = "POST"
        req.setValue(ticket.authToken, forHTTPHeaderField: "Authorization")
        req.setValue(contentType, forHTTPHeaderField: "Content-Type")
        // B2 wants the path percent-encoded, and the backend opts out of the
        // checksum the same way the web client does.
        let encodedPath = ticket.remotePath
            .addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? ticket.remotePath
        req.setValue(encodedPath, forHTTPHeaderField: "X-Bz-File-Name")
        req.setValue("do_not_verify", forHTTPHeaderField: "X-Bz-Content-Sha1")

        let (data, response): (Data, URLResponse)
        do {
            // Streaming from the file keeps a 4K video off the heap.
            (data, response) = try await session.upload(for: req, fromFile: fileURL)
        } catch {
            throw APIError.transport(error)
        }

        let code = (response as? HTTPURLResponse)?.statusCode ?? 0
        guard (200..<300).contains(code) else {
            let body = String(data: data, encoding: .utf8) ?? ""
            throw APIError.http(code, "Upload to storage failed (\(code)). \(body.prefix(120))")
        }
    }

    // MARK: - Selection & processing

    /// Round-trips one frame to the SAM2 worker to preview what a set of taps
    /// selects. Returns the base64 PNG mask, or nil when the worker is quiet.
    func previewMask(
        frameBase64PNG: String,
        frameIndex: Int,
        points: [SelectionPoint],
        videoWidth: Int,
        videoHeight: Int
    ) async throws -> String? {
        let payload: [String: Any] = [
            "frame_data": frameBase64PNG,
            "frame_index": frameIndex,
            "points": points.map { ["x": $0.x, "y": $0.y, "label": $0.label] },
            "video_width": videoWidth,
            "video_height": videoHeight
        ]
        let data = try await request("api/sam2/select-object", method: "POST", json: payload)
        return try decode(SelectObjectResponse.self, from: data).mask
    }

    func processVideo(
        taskId: String,
        points: [SelectionPoint],
        videoWidth: Int,
        videoHeight: Int,
        frameIndex: Int
    ) async throws -> String {
        let payload: [String: Any] = [
            "task_id": taskId,
            "prompt_mode": "point",
            "points": points.map { ["x": $0.x, "y": $0.y, "label": $0.label] },
            "video_width": videoWidth,
            "video_height": videoHeight,
            "frame_index": frameIndex
        ]
        let data = try await request("api/sam2/process-video", method: "POST", json: payload)
        let result = try decode(ProcessVideoResponse.self, from: data)
        guard let jobId = result.jobId else {
            throw APIError.http(500, result.message ?? "The server didn't return a job id.")
        }
        return jobId
    }

    // MARK: - Purchases

    /// Hands a StoreKit signed transaction to the backend, which verifies it
    /// against Apple's roots and moves the credits. Returns the new balance.
    func redeemApplePurchase(signedTransaction jws: String) async throws -> Double {
        let data = try await request(
            "api/billing/apple/redeem",
            method: "POST",
            json: ["signed_transaction": jws]
        )
        return try decode(RedeemResponse.self, from: data).credits
    }

    func jobStatus(jobId: String) async throws -> JobStatusResponse {
        try decode(JobStatusResponse.self, from: try await request("api/sam2/status/\(jobId)"))
    }

    /// Result paths come back either absolute or relative to the API host.
    nonisolated func absoluteResultURL(_ raw: String) -> URL? {
        if raw.hasPrefix("http") { return URL(string: raw) }
        return URL(string: raw, relativeTo: baseURL)?.absoluteURL
    }

    func download(_ url: URL) async throws -> URL {
        do {
            let (temp, _) = try await session.download(from: url)
            let dest = FileManager.default.temporaryDirectory
                .appendingPathComponent("markremover-\(UUID().uuidString).mp4")
            try? FileManager.default.removeItem(at: dest)
            try FileManager.default.moveItem(at: temp, to: dest)
            return dest
        } catch {
            throw APIError.transport(error)
        }
    }
}
