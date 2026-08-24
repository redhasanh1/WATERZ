import AVKit
import PhotosUI
import SwiftUI

@MainActor
final class BackgroundModel: ObservableObject {
    enum Stage: Equatable {
        case picking, selecting, working(String), done(URL), failed(String)
    }

    @Published var stage: Stage = .picking
    @Published var frame: VideoFrame?
    @Published var points: [SelectionPoint] = []
    @Published var style: BackgroundStyle = .transparent
    @Published private(set) var sourceURL: URL?

    private var duration: Double = 0
    private var pollTask: Task<Void, Never>?

    var canProcess: Bool { frame != nil && !points.isEmpty }

    deinit { pollTask?.cancel() }

    func load(videoURL: URL) async {
        reset()
        sourceURL = videoURL
        duration = await VideoFrameExtractor.duration(of: videoURL)

        guard duration <= EditorModel.maxDuration else {
            stage = .failed("That clip is \(Int(duration.rounded()))s. The limit is 90 seconds.")
            return
        }
        do {
            frame = try await VideoFrameExtractor.frame(from: videoURL)
            stage = .selecting
        } catch {
            stage = .failed(error.localizedDescription)
        }
    }

    func reset() {
        pollTask?.cancel(); pollTask = nil
        stage = .picking
        frame = nil
        points = []
        sourceURL = nil
        duration = 0
    }

    func addPoint(normalized: CGPoint) {
        guard let frame else { return }
        points.append(SelectionPoint(
            x: Int((normalized.x * frame.pixelSize.width).rounded()),
            y: Int((normalized.y * frame.pixelSize.height).rounded()),
            label: 1
        ))
    }

    func undo() { if !points.isEmpty { points.removeLast() } }

    func run(appState: AppState) async {
        guard let sourceURL, let frame else { return }
        guard appState.credits >= 1 else {
            stage = .failed(APIError.outOfCredits.localizedDescription)
            return
        }

        do {
            stage = .working("Uploading…")
            let jobId = try await APIClient.shared.backgroundUpload(
                fileURL: sourceURL,
                filename: sourceURL.lastPathComponent,
                contentType: "video/mp4",
                width: Int(frame.pixelSize.width),
                height: Int(frame.pixelSize.height),
                fps: frame.fps,
                duration: duration
            )

            stage = .working("Finding the subject…")
            try await APIClient.shared.backgroundSelect(
                jobId: jobId, points: points, frameIndex: frame.frameIndex
            )

            stage = .working("Following it through the clip…")
            try await APIClient.shared.backgroundTrack(jobId: jobId)

            stage = .working("Applying the background…")
            try await APIClient.shared.backgroundExport(jobId: jobId, style: style)

            AppLog.info(.editor, "Background job \(jobId) as \(style.rawValue)")
            poll(jobId: jobId, appState: appState)
        } catch {
            AppLog.error(.editor, "Background failed: \(error.localizedDescription)")
            stage = .failed(error.localizedDescription)
        }
    }

    private func poll(jobId: String, appState: AppState) {
        pollTask?.cancel()
        pollTask = Task { [weak self] in
            let deadline = Date().addingTimeInterval(20 * 60)
            while !Task.isCancelled && Date() < deadline {
                try? await Task.sleep(for: .seconds(3))
                guard let status = try? await APIClient.shared.backgroundStatus(jobId: jobId) else { continue }

                switch status.status {
                case "export_complete", "completed":
                    let url = APIClient.shared.backgroundDownloadURL(jobId: jobId)
                    if let local = try? await APIClient.shared.download(url) {
                        await MainActor.run { self?.stage = .done(local) }
                    } else {
                        await MainActor.run { self?.stage = .failed("Finished, but the download failed.") }
                    }
                    await appState.refreshUser()
                    return
                case "failed", "error":
                    await MainActor.run { self?.stage = .failed(status.error ?? "The worker rejected it.") }
                    return
                default:
                    await MainActor.run {
                        self?.stage = .working(status.message ?? "Working…")
                    }
                }
            }
            await MainActor.run { self?.stage = .failed("Timed out after 20 minutes.") }
        }
    }
}

struct BackgroundView: View {
    @EnvironmentObject private var appState: AppState
    @StateObject private var model = BackgroundModel()
    @State private var pickerItem: PhotosPickerItem?

    var body: some View {
        NavigationStack {
            Group {
                switch model.stage {
                case .picking: picker
                case .selecting: selector
                case .working(let message): busy(message)
                case .done(let url): result(url)
                case .failed(let message): failure(message)
                }
            }
            .navigationTitle("Background")
            .navigationBarTitleDisplayMode(.inline)
        }
        .onChange(of: pickerItem) { _, item in
            guard let item else { return }
            Task {
                guard let movie = try? await item.loadTransferable(type: VideoFile.self) else {
                    model.stage = .failed("Couldn't read that video.")
                    return
                }
                await model.load(videoURL: movie.url)
            }
        }
    }

    private var picker: some View {
        ScrollView {
            VStack(spacing: 20) {
                ZStack {
                    RoundedRectangle(cornerRadius: 24, style: .continuous)
                        .fill(LinearGradient(
                            colors: [Color(red: 0.20, green: 0.65, blue: 0.75),
                                     Color(red: 0.42, green: 0.36, blue: 0.90)],
                            startPoint: .topLeading, endPoint: .bottomTrailing
                        ))
                    VStack(spacing: 12) {
                        Image(systemName: "person.and.background.dotted")
                            .font(.system(size: 46))
                            .foregroundStyle(.white)
                        Text("Keep the subject,\nreplace everything else")
                            .font(.title3.bold())
                            .foregroundStyle(.white)
                            .multilineTextAlignment(.center)
                    }
                    .padding(.vertical, 26)
                }
                .frame(height: 200)
                .padding(.horizontal, 16)

                PhotosPicker(selection: $pickerItem, matching: .videos, photoLibrary: .shared()) {
                    Label("Choose a video", systemImage: "photo.on.rectangle.angled")
                }
                .buttonStyle(PrimaryButtonStyle())
                .padding(.horizontal, 16)

                Text("Up to 90 seconds · 1 credit per clip")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            .padding(.vertical, 14)
        }
        .background(Color(.systemGroupedBackground))
    }

    private var selector: some View {
        VStack(spacing: 0) {
            if let frame = model.frame {
                VideoCanvas(
                    frame: frame,
                    selections: [],
                    activeSelectionID: nil,
                    isBusy: false,
                    onTap: { model.addPoint(normalized: $0) }
                )
                .overlay(alignment: .topLeading) {
                    ForEach(Array(model.points.enumerated()), id: \.offset) { _, _ in EmptyView() }
                }
                .padding(.horizontal, 16)
                .padding(.top, 8)
            }

            Text(model.points.isEmpty
                 ? "Tap the person or object you want to keep."
                 : "\(model.points.count) tap\(model.points.count == 1 ? "" : "s") — everything else becomes the background.")
                .font(.footnote)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
                .padding(.horizontal, 24)
                .padding(.top, 12)

            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 10) {
                    ForEach(BackgroundStyle.allCases) { option in
                        Button {
                            model.style = option
                            Haptics.tick()
                        } label: {
                            VStack(spacing: 5) {
                                Image(systemName: option.symbol).font(.title3)
                                Text(option.title).font(.caption2.weight(.medium))
                            }
                            .frame(width: 88, height: 68)
                            .background(
                                RoundedRectangle(cornerRadius: 12, style: .continuous)
                                    .fill(model.style == option ? Theme.accentSoft : Color(.tertiarySystemFill))
                            )
                            .foregroundStyle(model.style == option ? Theme.accent : .primary)
                        }
                        .buttonStyle(.plain)
                    }
                }
                .padding(.horizontal, 16)
            }
            .padding(.top, 14)

            Text(model.style.detail)
                .font(.caption)
                .foregroundStyle(.secondary)
                .padding(.top, 6)

            Spacer(minLength: 12)

            VStack(spacing: 10) {
                HStack(spacing: 10) {
                    Button("Undo") { model.undo() }.disabled(model.points.isEmpty)
                    Button("New video") { model.reset(); pickerItem = nil }
                }
                .font(.subheadline)
                .buttonStyle(.bordered)

                Button("Replace background — 1 credit") {
                    Task { await model.run(appState: appState) }
                }
                .buttonStyle(PrimaryButtonStyle(enabled: model.canProcess))
                .disabled(!model.canProcess)
            }
            .padding(.horizontal, 16)
            .padding(.bottom, 16)
        }
    }

    private func busy(_ message: String) -> some View {
        VStack(spacing: 16) {
            Spacer()
            ProgressView().scaleEffect(1.5).tint(Theme.accent)
            Text(message).font(.subheadline).foregroundStyle(.secondary)
            Spacer()
        }
    }

    private func result(_ url: URL) -> some View {
        VStack(spacing: 14) {
            VideoPlayer(player: AVPlayer(url: url))
                .clipShape(RoundedRectangle(cornerRadius: 16, style: .continuous))
                .padding(.horizontal, 16)
                .padding(.top, 8)

            if model.style == .transparent {
                Label("WebM with alpha — MP4 can't carry transparency.", systemImage: "info.circle")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            Spacer(minLength: 8)

            VStack(spacing: 10) {
                ShareLink(item: url) { Text("Share or save").frame(maxWidth: .infinity) }
                    .buttonStyle(.borderedProminent)
                Button("Do another") { model.reset(); pickerItem = nil }
                    .font(.subheadline)
            }
            .padding(.horizontal, 16)
            .padding(.bottom, 16)
        }
    }

    private func failure(_ message: String) -> some View {
        VStack(spacing: 16) {
            Spacer()
            Image(systemName: "exclamationmark.triangle.fill")
                .font(.system(size: 40)).foregroundStyle(Theme.warning)
            Text(message)
                .font(.subheadline).foregroundStyle(.secondary)
                .multilineTextAlignment(.center).padding(.horizontal, 30)
            Button("Start over") { model.reset(); pickerItem = nil }
                .buttonStyle(PrimaryButtonStyle())
                .padding(.horizontal, 30)
            Spacer()
        }
    }
}
