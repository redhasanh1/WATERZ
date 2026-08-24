import AVFoundation
import SwiftUI

/// The editing surface: the frame, whatever marks it up, and the gestures for
/// getting close enough to be accurate.
///
/// The website binds a single-finger tap and calls `preventDefault()`, which
/// disables pinch-zoom outright — a small corner watermark is then nearly
/// impossible to hit on a phone. Here you can pinch, drag, and double-tap in.
struct VideoCanvas: View {
    let frame: VideoFrame
    let selections: [Selection]
    let activeSelectionID: Int?
    let isBusy: Bool
    var maskOpacity: Double = 0.55
    /// Hold the peek control to drop the overlays and see the untouched frame.
    var peeking: Bool = false
    /// Painted mask for the fixed-watermark mode.
    var drawnMask: UIImage? = nil
    /// When true, dragging paints instead of panning.
    var isDrawing: Bool = false
    var onDraw: ((CGPoint, CGPoint) -> Void)? = nil
    var onDrawEnded: (() -> Void)? = nil
    var onTap: (CGPoint) -> Void

    @State private var zoom: CGFloat = 1
    @State private var committedZoom: CGFloat = 1
    @State private var offset: CGSize = .zero
    @State private var committedOffset: CGSize = .zero

    private var scale: CGFloat { min(max(zoom, 1), 6) }

    var body: some View {
        GeometryReader { geo in
            let rect = fittedRect(in: geo.size)

            ZStack(alignment: .topLeading) {
                Color.black

                Image(uiImage: frame.image)
                    .resizable()
                    .aspectRatio(contentMode: .fit)

                if !peeking {
                    // Masks are pre-tinted and alpha-cut, so this is a plain
                    // composite into the identical fitted rect as the frame —
                    // which is what keeps them aligned at any resolution.
                    if let drawnMask {
                        Image(uiImage: drawnMask)
                            .resizable()
                            .interpolation(.none)
                            .aspectRatio(contentMode: .fit)
                            .opacity(maskOpacity)
                            .allowsHitTesting(false)
                    }

                    ForEach(selections) { selection in
                        if let mask = selection.mask {
                            Image(uiImage: mask)
                                .resizable()
                                .interpolation(.none)
                                .aspectRatio(contentMode: .fit)
                                .opacity(maskOpacity)
                                .allowsHitTesting(false)
                        }
                    }

                    ForEach(selections) { selection in
                        ForEach(selection.points) { point in
                            marker(for: point, selection: selection, in: rect)
                        }
                    }
                }
            }
            .scaleEffect(scale)
            .offset(offset)
            .contentShape(Rectangle())
            .gesture(primaryGesture(geo: geo, rect: rect))
            .onTapGesture(count: 2) { location in
                withAnimation(.easeOut(duration: 0.22)) { toggleZoom(at: location, in: geo.size) }
                Haptics.tick()
            }
            .onTapGesture { location in
                guard !isDrawing else { return }
                guard let normalized = normalize(location, geo: geo, rect: rect) else { return }
                Haptics.tap()
                onTap(normalized)
            }
            .overlay(alignment: .topTrailing) { zoomBadge }
            .overlay {
                if isBusy {
                    ProgressView()
                        .tint(.white)
                        .padding(12)
                        .background(.black.opacity(0.5), in: Circle())
                }
            }
        }
        .aspectRatio(frame.pixelSize.width / max(frame.pixelSize.height, 1), contentMode: .fit)
        .clipShape(RoundedRectangle(cornerRadius: 16, style: .continuous))
    }

    // MARK: - Gestures

    /// Dragging paints while a draw tool is active, otherwise it pans. Pinch
    /// stays available in both, so you can zoom in before painting detail.
    private func primaryGesture(geo: GeometryProxy, rect: CGRect) -> some Gesture {
        let drag = DragGesture(minimumDistance: 0)
            .onChanged { value in
                if isDrawing {
                    guard let onDraw,
                          let from = normalize(value.startLocation, geo: geo, rect: rect, clamped: true),
                          let to = normalize(value.location, geo: geo, rect: rect, clamped: true)
                    else { return }
                    onDraw(from, to)
                } else if scale > 1 {
                    offset = CGSize(
                        width: committedOffset.width + value.translation.width,
                        height: committedOffset.height + value.translation.height
                    )
                }
            }
            .onEnded { _ in
                if isDrawing {
                    Haptics.tap()
                    onDrawEnded?()
                } else {
                    committedOffset = offset
                }
            }

        let pinch = MagnificationGesture()
            .onChanged { zoom = committedZoom * $0 }
            .onEnded { _ in
                committedZoom = scale
                zoom = scale
                if scale == 1 { resetPan() }
            }

        return pinch.simultaneously(with: drag)
    }

    private func toggleZoom(at location: CGPoint, in size: CGSize) {
        if scale > 1.01 {
            zoom = 1; committedZoom = 1; resetPan()
        } else {
            let target: CGFloat = 3
            let centre = CGPoint(x: size.width / 2, y: size.height / 2)
            zoom = target
            committedZoom = target
            offset = CGSize(
                width: (centre.x - location.x) * target,
                height: (centre.y - location.y) * target
            )
            committedOffset = offset
        }
    }

    private func resetPan() {
        offset = .zero
        committedOffset = .zero
    }

    /// Undoes the zoom/pan transform so a screen point maps back onto the frame.
    private func normalize(
        _ location: CGPoint, geo: GeometryProxy, rect: CGRect, clamped: Bool = false
    ) -> CGPoint? {
        let centre = CGPoint(x: geo.size.width / 2, y: geo.size.height / 2)
        let unprojected = CGPoint(
            x: (location.x - centre.x - offset.width) / scale + centre.x,
            y: (location.y - centre.y - offset.height) / scale + centre.y
        )
        if !clamped && !rect.contains(unprojected) { return nil }

        let x = (unprojected.x - rect.minX) / rect.width
        let y = (unprojected.y - rect.minY) / rect.height
        return clamped
            ? CGPoint(x: min(max(x, 0), 1), y: min(max(y, 0), 1))
            : CGPoint(x: x, y: y)
    }

    // MARK: - Pieces

    @ViewBuilder
    private var zoomBadge: some View {
        if scale > 1.01 {
            Button {
                withAnimation(.easeOut(duration: 0.2)) {
                    zoom = 1; committedZoom = 1; resetPan()
                }
            } label: {
                Label("\(String(format: "%.1f", scale))×", systemImage: "arrow.down.right.and.arrow.up.left")
                    .font(.caption2.weight(.semibold))
                    .padding(.horizontal, 9)
                    .padding(.vertical, 5)
                    .background(.ultraThinMaterial, in: Capsule())
            }
            .padding(8)
        }
    }

    /// Where the aspect-fit image actually sits inside the container.
    private func fittedRect(in container: CGSize) -> CGRect {
        let imageAspect = frame.pixelSize.width / max(frame.pixelSize.height, 1)
        let containerAspect = container.width / max(container.height, 1)

        var size = container
        if imageAspect > containerAspect {
            size.height = container.width / imageAspect
        } else {
            size.width = container.height * imageAspect
        }

        return CGRect(
            x: (container.width - size.width) / 2,
            y: (container.height - size.height) / 2,
            width: size.width,
            height: size.height
        )
    }

    private func marker(for point: SelectionPoint, selection: Selection, in rect: CGRect) -> some View {
        let x = rect.minX + CGFloat(point.x) / frame.pixelSize.width * rect.width
        let y = rect.minY + CGFloat(point.y) / frame.pixelSize.height * rect.height
        let keep = point.label == 0
        // Shrink markers as you zoom so they stop covering what you aimed at.
        let size = max(8, 17 / scale)
        let isActive = selection.id == activeSelectionID

        return ZStack {
            if keep {
                // A "keep" mark should read as a cut-out, not another object.
                Circle()
                    .fill(.black.opacity(0.6))
                    .overlay(
                        Image(systemName: "minus")
                            .font(.system(size: size * 0.66, weight: .black))
                            .foregroundStyle(.white)
                    )
            } else {
                Circle().fill(selection.color)
            }
        }
        .frame(width: size, height: size)
        .overlay(Circle().stroke(.white, lineWidth: max(1, (isActive ? 2.5 : 1.5) / scale)))
        .shadow(color: .black.opacity(0.5), radius: max(1, 2 / scale))
        .position(x: x, y: y)
        .allowsHitTesting(false)
    }
}

/// Frame picker. Thumbnails are generated on device rather than pulled from the
/// server's sprite endpoint — the clip is already local, so this costs no round
/// trip and works before the upload happens.
struct FrameScrubber: View {
    let videoURL: URL
    let duration: Double
    @Binding var time: Double
    var onCommit: (Double) -> Void

    @State private var thumbnails: [UIImage] = []

    var body: some View {
        VStack(spacing: 6) {
            ZStack(alignment: .leading) {
                HStack(spacing: 0) {
                    ForEach(Array(thumbnails.enumerated()), id: \.offset) { _, image in
                        Image(uiImage: image)
                            .resizable()
                            .aspectRatio(contentMode: .fill)
                            .frame(maxWidth: .infinity)
                            .frame(height: 44)
                            .clipped()
                    }
                }
                .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
                .overlay(
                    RoundedRectangle(cornerRadius: 8, style: .continuous)
                        .stroke(Color(.separator), lineWidth: 0.5)
                )

                GeometryReader { geo in
                    let x = duration > 0 ? CGFloat(time / duration) * geo.size.width : 0
                    RoundedRectangle(cornerRadius: 2)
                        .fill(.white)
                        .frame(width: 3, height: 50)
                        .shadow(radius: 2)
                        .position(x: min(max(x, 2), max(geo.size.width - 2, 2)), y: 22)
                }
                .frame(height: 44)
            }
            .frame(height: 44)

            Slider(value: $time, in: 0...max(duration, 0.1)) { editing in
                if !editing { onCommit(time) }
            }
            .tint(Theme.accent)

            HStack {
                Text(timeLabel(time))
                Spacer()
                Text(timeLabel(duration))
            }
            .font(.caption2.monospacedDigit())
            .foregroundStyle(.secondary)
        }
        .task { await loadThumbnails() }
    }

    private func timeLabel(_ seconds: Double) -> String {
        String(format: "%d:%05.2f", Int(seconds) / 60, seconds.truncatingRemainder(dividingBy: 60))
    }

    private func loadThumbnails() async {
        guard thumbnails.isEmpty, duration > 0 else { return }

        let asset = AVURLAsset(url: videoURL)
        let generator = AVAssetImageGenerator(asset: asset)
        generator.appliesPreferredTrackTransform = true
        generator.maximumSize = CGSize(width: 120, height: 120)
        // Tolerance is fine here: these are 44pt tall and only need to suggest
        // roughly where you are in the clip.
        generator.requestedTimeToleranceBefore = CMTime(seconds: 0.3, preferredTimescale: 600)
        generator.requestedTimeToleranceAfter = CMTime(seconds: 0.3, preferredTimescale: 600)

        var images: [UIImage] = []
        let count = 8
        for index in 0..<count {
            let seconds = duration * Double(index) / Double(count)
            if let cg = try? await generator.image(at: CMTime(seconds: seconds, preferredTimescale: 600)).image {
                images.append(UIImage(cgImage: cg))
            }
        }
        thumbnails = images
    }
}
