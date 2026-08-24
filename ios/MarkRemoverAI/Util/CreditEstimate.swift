import Foundation

/// Mirrors the website's estimate so the number quoted in the app matches what
/// the job actually costs. Baseline is 720p for 10 seconds.
enum CreditEstimate {
    private static let basePixels: Double = 1280 * 720
    private static let baseDuration: Double = 10

    /// Background replacement is charged at a third of the removal rate.
    static func credits(duration: Double, size: CGSize, isBackground: Bool) -> Double {
        guard duration > 0, size.width > 0, size.height > 0 else { return 0.1 }

        let durationFactor = duration / baseDuration
        let resolutionFactor = ((size.width * size.height) / basePixels).squareRoot()
        let raw = durationFactor * 0.7 + resolutionFactor * 0.3
        let scaled = isBackground ? raw / 3 : raw

        return max(0.1, (scaled * 10).rounded() / 10)
    }

    static func label(_ credits: Double) -> String {
        credits == credits.rounded()
            ? String(Int(credits))
            : String(format: "%.1f", credits)
    }
}
