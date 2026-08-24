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
    /// The pipeline tracks each object_id separately, so a second subject needs
    /// its own id rather than more points on the first.
    @Published var currentObject = 0
    /// Hand-corrections to the SAM2 mask. The worker takes these in place of
    /// what it generated, so a bad edge can be fixed before tracking runs.
    let maskBuilder = StaticMaskBuilder()
    @Published var settings = BackgroundSettings()
    @Published private(set) var sourceURL: URL?

    private var duration: Double = 0
    private var pollTask: Task<Void, Never>?

    var canProcess: Bool { frame != nil && !points.isEmpty }


    deinit { pollTask?.cancel() }

    func load(videoURL: URL) async {
        reset()
        sourceURL = videoURL
        duration = await VideoFrameExtractor.duration(of: videoURL)

        do {
            let first = try await VideoFrameExtractor.frame(from: videoURL)
            frame = first
            maskBuilder.begin(videoSize: first.pixelSize)
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
        currentObject = 0
        maskBuilder.clear()
        sourceURL = nil
        duration = 0
    }

    func addPoint(normalized: CGPoint, label: Int = 1) {
        guard let frame else { return }
        points.append(SelectionPoint(
            x: Int((normalized.x * frame.pixelSize.width).rounded()),
            y: Int((normalized.y * frame.pixelSize.height).rounded()),
            label: label,
            objectId: currentObject
        ))
    }

    func nextObject() {
        guard points.contains(where: { $0.objectId == currentObject }) else { return }
        currentObject += 1
    }

    func undo() {
        guard !points.isEmpty else { return }
        points.removeLast()
        currentObject = points.map(\.objectId).max() ?? 0
    }

    var objectCount: Int { Set(points.map(\.objectId)).count }

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
            var corrections: [(objectId: Int, base64: String)] = []
            if !maskBuilder.isEmpty, let drawn = maskBuilder.exportBase64PNG() {
                corrections.append((objectId: currentObject, base64: drawn))
                AppLog.info(.editor, "Sending a hand-corrected mask for object \(currentObject)")
            }
            try await APIClient.shared.backgroundTrack(jobId: jobId, modifiedMasks: corrections)

            stage = .working("Applying the background…")
            try await APIClient.shared.backgroundExport(jobId: jobId, settings: settings)

            AppLog.info(.editor, "Background job \(jobId): \(settings.operation.rawValue) / \(settings.fill.rawValue), dilation \(Int(settings.dilation))")
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
    @State private var markMode = 1
    @State private var tool: BackgroundTool = .click
    @State private var brushSize: Double = 0.04

    /// Click places SAM2 points; the rest paint a correction over the mask.
    enum BackgroundTool: String, CaseIterable, Identifiable {
        case click, rectangle, brush, eraser
        var id: String { rawValue }
        var title: String {
            switch self {
            case .click: return "Click"
            case .rectangle: return "Box"
            case .brush: return "Brush"
            case .eraser: return "Erase"
            }
        }
        var symbol: String {
            switch self {
            case .click: return "hand.tap"
            case .rectangle: return "rectangle"
            case .brush: return "paintbrush.pointed"
            case .eraser: return "eraser"
            }
        }
        var drawTool: StaticMaskBuilder.Tool? {
            switch self {
            case .click: return nil
            case .rectangle: return .rectangle
            case .brush: return .brush
            case .eraser: return .eraser
            }
        }
    }

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

                Text("1 credit per clip · any length")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            .padding(.vertical, 14)
        }
        .background(Color(.systemGroupedBackground))
    }

    private var selector: some View {
        ScrollView {
            VStack(spacing: 0) {
            if let frame = model.frame {
                VideoCanvas(
                    frame: frame,
                    selections: [],
                    activeSelectionID: nil,
                    isBusy: false,
                    drawnMask: model.maskBuilder.preview,
                    isDrawing: tool != .click,
                    onDraw: { from, to in
                        guard let drawTool = tool.drawTool else { return }
                        model.maskBuilder.stroke(
                            from: from, to: to, tool: drawTool, brushFraction: brushSize
                        )
                    },
                    onTap: { location in
                        guard tool == .click else { return }
                        model.addPoint(normalized: location, label: markMode)
                    }
                )
                .padding(.horizontal, 16)
                .padding(.top, 8)
            }

            HStack(spacing: 8) {
                ForEach(BackgroundTool.allCases) { option in
                    Button {
                        tool = option
                        Haptics.tick()
                    } label: {
                        VStack(spacing: 3) {
                            Image(systemName: option.symbol).font(.subheadline)
                            Text(option.title).font(.caption2)
                        }
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 8)
                        .background(
                            RoundedRectangle(cornerRadius: 10, style: .continuous)
                                .fill(tool == option ? Theme.accentSoft : Color(.tertiarySystemFill))
                        )
                        .foregroundStyle(tool == option ? Theme.accent : .primary)
                    }
                    .buttonStyle(.plain)
                }
            }
            .padding(.horizontal, 16)
            .padding(.top, 12)

            if tool == .click {
                Picker("Tap mode", selection: $markMode) {
                    Text("Include").tag(1)
                    Text("Exclude").tag(0)
                }
                .pickerStyle(.segmented)
                .padding(.horizontal, 16)
                .padding(.top, 10)
            } else {
                HStack(spacing: 10) {
                    Image(systemName: "circle.fill").font(.system(size: 7))
                    Slider(value: $brushSize, in: 0.01...0.25).tint(Theme.accent)
                    Image(systemName: "circle.fill").font(.system(size: 15))
                    Button("Clear") { model.maskBuilder.clear() }
                        .font(.caption)
                        .buttonStyle(.bordered)
                        .disabled(model.maskBuilder.isEmpty)
                }
                .foregroundStyle(.secondary)
                .padding(.horizontal, 16)
                .padding(.top, 10)
            }

            Text(model.points.isEmpty
                 ? (model.settings.operation == .keepObject
                    ? "Tap what you want to keep."
                    : "Tap what you want gone.")
                 : "\(model.points.count) tap\(model.points.count == 1 ? "" : "s") across \(model.objectCount) subject\(model.objectCount == 1 ? "" : "s").")
                .font(.footnote)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
                .padding(.horizontal, 24)
                .padding(.top, 12)

            options
                .padding(.top, 12)

            Spacer(minLength: 12)

            VStack(spacing: 10) {
                HStack(spacing: 10) {
                    Button("Undo") { model.undo() }.disabled(model.points.isEmpty)
                    Button {
                        model.nextObject()
                        Haptics.tick()
                    } label: {
                        Label("Next subject", systemImage: "plus.circle")
                    }
                    .disabled(model.points.isEmpty)
                    Button("New video") { model.reset(); pickerItem = nil }
                }
                .font(.subheadline)
                .buttonStyle(.bordered)

                Button(model.settings.operation == .keepObject
                       ? "Replace background — 1 credit"
                       : "Remove selection — 1 credit") {
                    Task { await model.run(appState: appState) }
                }
                .buttonStyle(PrimaryButtonStyle(enabled: model.canProcess))
                .disabled(!model.canProcess)
            }
            .padding(.horizontal, 16)
            .padding(.bottom, 16)
            }
        }
        .background(Color(.systemGroupedBackground))
    }

    /// Everything the website exposes: what survives, what fills the rest, and
    /// the two sliders that decide how clean the cut-out looks.
    private var options: some View {
        VStack(spacing: 14) {
            VStack(spacing: 6) {
                Picker("Operation", selection: $model.settings.operation) {
                    ForEach(BackgroundOperation.allCases) { Text($0.title).tag($0) }
                }
                .pickerStyle(.segmented)
                Text(model.settings.operation.detail)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            HStack(spacing: 8) {
                ForEach(BackgroundFill.allCases) { option in
                    Button {
                        model.settings.fill = option
                        Haptics.tick()
                    } label: {
                        VStack(spacing: 5) {
                            Image(systemName: option.symbol).font(.title3)
                            Text(option.title).font(.caption2.weight(.medium))
                        }
                        .frame(maxWidth: .infinity, minHeight: 62)
                        .background(
                            RoundedRectangle(cornerRadius: 12, style: .continuous)
                                .fill(model.settings.fill == option ? Theme.accentSoft : Color(.tertiarySystemFill))
                        )
                        .foregroundStyle(model.settings.fill == option ? Theme.accent : .primary)
                    }
                    .buttonStyle(.plain)
                }
            }

            if model.settings.fill == .color {
                ColorPicker("Background colour", selection: $model.settings.color, supportsOpacity: false)
                    .font(.subheadline)
            }

            if model.settings.fill == .blur {
                slider("Blur", value: $model.settings.blurAmount, range: 5...50)
            }

            // Dilation grows the mask outward. A pixel or two removes the halo
            // of old background that otherwise clings to the edges.
            slider("Edge grow", value: $model.settings.dilation, range: 0...20)
        }
        .padding(.horizontal, 16)
    }

    private func slider(_ label: String, value: Binding<Double>, range: ClosedRange<Double>) -> some View {
        HStack(spacing: 10) {
            Text(label)
                .font(.caption)
                .foregroundStyle(.secondary)
                .frame(width: 68, alignment: .leading)
            Slider(value: value, in: range, step: 1).tint(Theme.accent)
            Text("\(Int(value.wrappedValue))")
                .font(.caption2.monospacedDigit())
                .foregroundStyle(.secondary)
                .frame(width: 24, alignment: .trailing)
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

            if model.settings.fill == .transparent {
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
