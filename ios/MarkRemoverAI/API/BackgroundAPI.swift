import Foundation

/// Background replacement runs on its own job pipeline (`objrem/…`), separate
/// from the object-removal one — different upload, different job id, different
/// status endpoint. Keeping it in its own file stops the two getting confused.
extension APIClient {

    struct BackgroundJob {
        let jobId: String
        let uploadURL: String
        let authToken: String
        let remotePath: String
        let cdnURL: String
    }

    func backgroundUpload(
        fileURL: URL,
        filename: String,
        contentType: String,
        width: Int,
        height: Int,
        fps: Double,
        duration: Double
    ) async throws -> String {
        let ticket: BackgroundTicket = try decode(
            BackgroundTicket.self,
            from: try await request(
                "api/object-removal/get-upload-url",
                method: "POST",
                json: ["filename": filename]
            )
        )

        try await uploadToB2(
            fileURL: fileURL,
            uploadURL: ticket.uploadURL,
            authToken: ticket.authToken,
            remotePath: ticket.remotePath,
            contentType: contentType
        )

        _ = try await request(
            "api/object-removal/upload-complete",
            method: "POST",
            json: [
                "job_id": ticket.jobId,
                "cdn_url": ticket.cdnURL,
                "width": width,
                "height": height,
                "fps": fps,
                "duration": duration,
                "frame_count": max(1, Int((duration * fps).rounded()))
            ]
        )
        return ticket.jobId
    }

    /// Marks the subject to keep, then follows it through the clip.
    func backgroundSelect(jobId: String, points: [SelectionPoint], frameIndex: Int) async throws {
        _ = try await request(
            "api/object-removal/select",
            method: "POST",
            json: [
                "job_id": jobId,
                "points": points.map { ["x": $0.x, "y": $0.y, "label": $0.label] },
                "frame_index": frameIndex
            ]
        )
    }

    func backgroundTrack(jobId: String) async throws {
        _ = try await request(
            "api/object-removal/track",
            method: "POST",
            json: ["job_id": jobId]
        )
    }

    func backgroundExport(jobId: String, style: BackgroundStyle) async throws {
        var payload: [String: Any] = [
            "job_id": jobId,
            "operation": "keep_object",
            "background": style.background,
            "dilation": 0,
            "format": style.format
        ]
        if let color = style.hexColor { payload["bg_color"] = color }
        if let blur = style.blurAmount { payload["blur_amount"] = blur }

        _ = try await request("api/object-removal/export", method: "POST", json: payload)
    }

    func backgroundStatus(jobId: String) async throws -> JobStatusResponse {
        try decode(
            JobStatusResponse.self,
            from: try await request("api/object-removal/status/\(jobId)")
        )
    }

    nonisolated func backgroundDownloadURL(jobId: String) -> URL {
        baseURL.appendingPathComponent("api/object-removal/download/\(jobId)")
    }
}

private struct BackgroundTicket: Codable {
    let status: String
    let jobId: String
    let uploadURL: String
    let authToken: String
    let remotePath: String
    let cdnURL: String

    enum CodingKeys: String, CodingKey {
        case status
        case jobId = "job_id"
        case uploadURL = "upload_url"
        case authToken = "auth_token"
        case remotePath = "remote_path"
        case cdnURL = "cdn_url"
    }
}

/// What replaces the background once the subject is isolated.
enum BackgroundStyle: String, CaseIterable, Identifiable {
    case transparent, blur, green, black, white

    var id: String { rawValue }

    var title: String {
        switch self {
        case .transparent: return "Transparent"
        case .blur: return "Blurred"
        case .green: return "Green screen"
        case .black: return "Black"
        case .white: return "White"
        }
    }

    var detail: String {
        switch self {
        case .transparent: return "Alpha channel, for compositing"
        case .blur: return "Keeps the setting, drops the detail"
        case .green: return "Chroma key for your editor"
        case .black: return "Solid black"
        case .white: return "Solid white"
        }
    }

    var symbol: String {
        switch self {
        case .transparent: return "square.dashed"
        case .blur: return "drop.circle"
        case .green: return "flag.checkered"
        case .black: return "circle.fill"
        case .white: return "circle"
        }
    }

    var background: String {
        switch self {
        case .transparent: return "transparent"
        case .blur: return "blur"
        case .green, .black, .white: return "color"
        }
    }

    var hexColor: String? {
        switch self {
        case .green: return "#00FF00"
        case .black: return "#000000"
        case .white: return "#FFFFFF"
        default: return nil
        }
    }

    var blurAmount: Int? { self == .blur ? 25 : nil }

    /// MP4 has no alpha channel, so a transparent export has to be WebM.
    var format: String { self == .transparent ? "webm" : "mp4" }

    var fileExtension: String { format }
}
