import AVFoundation
import SwiftUI

/// The editing surface: the chosen frame, the taps on it, the SAM2 mask, and a
/// scrubber for picking which frame to mark up. Pinch to zoom, drag to pan —
/// a watermark in a corner is often too small to hit accurately at fit-scale.
struct VideoCanvas: View {
    let frame: VideoFrame
    let selections: [Selection]
    let activeSelectionID: Int?
    let isBusy: Bool
    var maskOpacity: Double = 0.55
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

                // Each mask is already tinted and alpha-cut, so drawing it is
                // a plain composite — no per-pixel work, and it lands in the
                // identical fitted rect as the frame, which is what keeps it
                // pixel-aligned at any resolution.
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
            .scaleEffect(scale)
            .offset(offset)
            .contentShape(Rectangle())
            .onTapGesture { location in
                // Undo the zoom/pan so the tap maps back to the frame.
                let centre = CGPoint(x: geo.size.width / 2, y: geo.size.height / 2)
                let unscaled = CGPoint(
                    x: (location.x - centre.x - offset.width) / scale + centre.x,
                    y: (location.y - centre.y - offset.height) / scale + centre.y
                )
                guard rect.contains(unscaled) else { return }
                onTap(CGPoint(
                    x: (unscaled.x - rect.minX) / rect.width,
                    y: (unscaled.y - rect.minY) / rect.height
                ))
            }
            .gesture(
                MagnificationGesture()
                    .onChanged { zoom = committedZoom * $0 }
                    .onEnded { _ in
                        committedZoom = scale
                        zoom = scale
                        if scale == 1 { resetPan() }
                    }
                    .simultaneously(with:
                        DragGesture()
                            .onChanged { value in
                                guard scale > 1 else { return }
                                offset = CGSize(
                                    width: committedOffset.width + value.translation.width,
                                    height: committedOffset.height + value.translation.height
                                )
                            }
                            .onEnded { _ in committedOffset = offset }
                    )
            )
            .overlay(alignment: .topTrailing) {
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

    private func resetPan() {
        offset = .zero
        committedOffset = .zero
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

    @ViewBuilder
    private func marker(for point: SelectionPoint, selection: Selection, in rect: CGRect) -> some View {
        let x = rect.minX + CGFloat(point.x) / frame.pixelSize.width * rect.width
        let y = rect.minY + CGFloat(point.y) / frame.pixelSize.height * rect.height
        let keep = point.label == 0
        // Shrink markers as you zoom so they stop covering what you aimed at.
        let size = max(8, 17 / scale)
        let isActive = selection.id == activeSelectionID

        ZStack {
            if keep {
                // A "keep" mark reads as a cut-out, not another object.
                Circle()
                    .fill(.black.opacity(0.55))
                    .overlay(
                        Image(systemName: "minus")
                            .font(.system(size: size * 0.7, weight: .black))
                            .foregroundStyle(.white)
                    )
            } else {
                Circle().fill(selection.color)
            }
        }
        .frame(width: size, height: size)
        .overlay(
            Circle().stroke(.white, lineWidth: max(1, (isActive ? 2.5 : 1.5) / scale))
        )
        .shadow(color: .black.opacity(0.5), radius: max(1, 2 / scale))
        .position(x: x, y: y)
        .allowsHitTesting(false)
    }
}

/// Frame picker. Thumbnails are generated on device rather than pulled from the
/// server's sprite endpoint — the video is already local, so this needs no
/// round trip and works before the upload happens.
struct FrameScrubber: View {
    let videoURL: URL
    let duration: Double
    @Binding var time: Double
    var onCommit: (Double) -> Void

    @State private var thumbnails: [UIImage] = []
    @State private var isScrubbing = false

    var body: some View {
        VStack(spacing: 6) {
            ZStack(alignment: .leading) {
                HStack(spacing: 0) {
                    ForEach(Array(thumbnails.enumerated()), id: \.offset) { _, image in
                        Image(uiImage: image)
                            .resizable()
                            .aspectRatio(contentMode: .fill)
                            .frame(maxWidth: .infinity)
                            .frame(height: 46)
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
                        .frame(width: 3, height: 52)
                        .shadow(radius: 2)
                        .position(x: min(max(x, 2), geo.size.width - 2), y: 23)
                }
                .frame(height: 46)
            }
            .frame(height: 46)

            Slider(value: $time, in: 0...max(duration, 0.1)) { editing in
                isScrubbing = editing
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
        // Tolerance is fine here: these are 46pt tall and only need to suggest
        // roughly where you are in the clip.
        generator.requestedTimeToleranceBefore = CMTime(seconds: 0.3, preferredTimescale: 600)
        generator.requestedTimeToleranceAfter = CMTime(seconds: 0.3, preferredTimescale: 600)

        let count = 8
        var images: [UIImage] = []
        for index in 0..<count {
            let seconds = duration * Double(index) / Double(count)
            let time = CMTime(seconds: seconds, preferredTimescale: 600)
            if let cg = try? await generator.image(at: time).image {
                images.append(UIImage(cgImage: cg))
            }
        }
        thumbnails = images
    }
}
