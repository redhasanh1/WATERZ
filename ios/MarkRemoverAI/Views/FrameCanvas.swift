import SwiftUI

/// The frame with the taps drawn on it, plus the SAM2 mask when the worker
/// hands one back. Taps are reported normalized so the model can convert them
/// into video pixels regardless of how the frame is laid out on screen.
struct FrameCanvas: View {
    let frame: VideoFrame
    let points: [SelectionPoint]
    let maskImage: UIImage?
    let isBusy: Bool
    var onTap: (CGPoint) -> Void

    var body: some View {
        GeometryReader { geo in
            let rect = fittedRect(in: geo.size)

            ZStack(alignment: .topLeading) {
                Color.black

                Image(uiImage: frame.image)
                    .resizable()
                    .aspectRatio(contentMode: .fit)

                if let maskImage {
                    Image(uiImage: maskImage)
                        .resizable()
                        .aspectRatio(contentMode: .fit)
                        .blendMode(.screen)
                        .opacity(0.55)
                        .allowsHitTesting(false)
                }

                ForEach(points) { point in
                    marker(for: point, in: rect)
                }

                if isBusy {
                    ProgressView()
                        .tint(.white)
                        .padding(10)
                        .background(.black.opacity(0.45), in: Circle())
                        .position(x: rect.midX, y: rect.midY)
                }
            }
            .contentShape(Rectangle())
            .onTapGesture { location in
                guard rect.contains(location) else { return }
                onTap(CGPoint(
                    x: (location.x - rect.minX) / rect.width,
                    y: (location.y - rect.minY) / rect.height
                ))
            }
        }
        .aspectRatio(frame.pixelSize.width / max(frame.pixelSize.height, 1), contentMode: .fit)
        .clipShape(RoundedRectangle(cornerRadius: 16, style: .continuous))
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

    private func marker(for point: SelectionPoint, in rect: CGRect) -> some View {
        let x = rect.minX + CGFloat(point.x) / frame.pixelSize.width * rect.width
        let y = rect.minY + CGFloat(point.y) / frame.pixelSize.height * rect.height
        let keep = point.label == 0

        return Circle()
            .fill(keep ? Color.red : Theme.positive)
            .frame(width: 18, height: 18)
            .overlay(Circle().stroke(.white, lineWidth: 2.5))
            .shadow(radius: 3)
            .position(x: x, y: y)
            .allowsHitTesting(false)
    }
}
