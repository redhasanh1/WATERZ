import UIKit

/// A hand-drawn area for something that never moves. The bitmap is repeated on
/// every frame server-side, so no tracking runs at all — faster than SAM2 and
/// it cannot drift off something stationary.
@MainActor
final class StaticMaskBuilder: ObservableObject {
    enum Tool: String, CaseIterable, Identifiable {
        case rectangle, ellipse, polygon, brush, eraser
        var id: String { rawValue }

        var symbol: String {
            switch self {
            case .rectangle: return "rectangle"
            case .ellipse: return "circle"
            case .polygon: return "skew"
            case .brush: return "paintbrush.pointed"
            case .eraser: return "eraser"
            }
        }

        var title: String {
            switch self {
            case .rectangle: return "Box"
            case .ellipse: return "Oval"
            case .polygon: return "Shape"
            case .brush: return "Brush"
            case .eraser: return "Erase"
            }
        }
    }

    /// Cap on the drawing buffer. An 8K mask is 33M pixels; past 4K the extra
    /// precision buys nothing for a region mask and just costs memory.
    private static let maxEdge: CGFloat = 4096

    @Published private(set) var preview: UIImage?
    @Published private(set) var isEmpty = true
    /// Vertices of the polygon being drawn, 0…1 in frame space.
    @Published private(set) var pendingPolygon: [CGPoint] = []

    private var context: CGContext?
    private(set) var size: CGSize = .zero
    private var videoSize: CGSize = .zero

    /// Scale from mask space back up to video space on export.
    private var exportScale: CGFloat { videoSize.width / max(size.width, 1) }

    func begin(videoSize: CGSize) {
        self.videoSize = videoSize

        let longest = max(videoSize.width, videoSize.height)
        let scale = longest > Self.maxEdge ? Self.maxEdge / longest : 1
        size = CGSize(
            width: max(1, (videoSize.width * scale).rounded()),
            height: max(1, (videoSize.height * scale).rounded())
        )

        // 8-bit grey: the mask is on/off, and a full RGBA buffer at this size
        // would be four times the memory for nothing.
        context = CGContext(
            data: nil,
            width: Int(size.width),
            height: Int(size.height),
            bitsPerComponent: 8,
            bytesPerRow: 0,
            space: CGColorSpaceCreateDeviceGray(),
            bitmapInfo: CGImageAlphaInfo.none.rawValue
        )
        context?.setFillColor(UIColor.black.cgColor)
        context?.fill(CGRect(origin: .zero, size: size))
        isEmpty = true
        refresh()
    }

    /// Polygon is tap-to-place rather than drag, so it gets its own entry point.
    func addPolygonVertex(_ normalized: CGPoint) {
        pendingPolygon.append(normalized)
    }

    func undoPolygonVertex() {
        if !pendingPolygon.isEmpty { pendingPolygon.removeLast() }
    }

    /// Closes the shape and fills it. Fewer than three vertices isn't an area.
    func closePolygon() {
        guard let context, pendingPolygon.count >= 3 else {
            pendingPolygon = []
            return
        }
        context.setFillColor(UIColor.white.cgColor)
        context.beginPath()
        context.move(to: point(pendingPolygon[0]))
        for vertex in pendingPolygon.dropFirst() {
            context.addLine(to: point(vertex))
        }
        context.closePath()
        context.fillPath()

        pendingPolygon = []
        isEmpty = false
        refresh()
    }

    func clear() {
        pendingPolygon = []
        guard let context else { return }
        context.setFillColor(UIColor.black.cgColor)
        context.fill(CGRect(origin: .zero, size: size))
        isEmpty = true
        refresh()
    }

    /// `from`/`to` are 0…1 in frame space.
    func stroke(from: CGPoint, to: CGPoint, tool: Tool, brushFraction: CGFloat) {
        guard let context else { return }

        let a = point(from), b = point(to)
        let white = tool == .eraser ? UIColor.black.cgColor : UIColor.white.cgColor
        context.setFillColor(white)
        context.setStrokeColor(white)

        switch tool {
        case .rectangle:
            context.fill(CGRect(x: min(a.x, b.x), y: min(a.y, b.y),
                                width: abs(b.x - a.x), height: abs(b.y - a.y)))
        case .ellipse:
            context.fillEllipse(in: CGRect(x: min(a.x, b.x), y: min(a.y, b.y),
                                           width: abs(b.x - a.x), height: abs(b.y - a.y)))
        case .polygon:
            // Placed by tap, not drag — nothing to do on a stroke.
            return

        case .brush, .eraser:
            let width = max(2, brushFraction * size.width)
            context.setLineWidth(width)
            context.setLineCap(.round)
            context.setLineJoin(.round)
            context.beginPath()
            context.move(to: a)
            context.addLine(to: b)
            context.strokePath()
        }

        if tool != .eraser { isEmpty = false }
        refresh()
    }

    /// Full-resolution PNG for `/api/process-static-mask`.
    func exportBase64PNG() -> String? {
        guard let image = currentCGImage() else { return nil }

        let target = exportScale > 1.001 ? videoSize : size
        let format = UIGraphicsImageRendererFormat()
        format.scale = 1
        format.opaque = true

        let rendered = UIGraphicsImageRenderer(size: target, format: format).image { ctx in
            ctx.cgContext.interpolationQuality = .none
            UIImage(cgImage: image).draw(in: CGRect(origin: .zero, size: target))
        }
        return rendered.pngData()?.base64EncodedString()
    }

    private func point(_ normalized: CGPoint) -> CGPoint {
        // CoreGraphics origin is bottom-left; the frame's is top-left.
        CGPoint(x: normalized.x * size.width, y: (1 - normalized.y) * size.height)
    }

    private func currentCGImage() -> CGImage? { context?.makeImage() }

    private func refresh() {
        guard let image = currentCGImage() else { return }
        preview = MaskRenderer.tinted(CIImage(cgImage: image), color: SelectionPalette.uiColor(0))
    }
}
