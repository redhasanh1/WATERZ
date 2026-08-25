import Foundation

/// Mirrors the website's estimate so the number quoted in the app matches what
/// the job actually costs. Baseline is 720p for 10 seconds.
enum CreditEstimate {
    private static let basePixels: Double = 1280 * 720
    private static let baseDuration: Double = 10

    /// The two tools price differently, and not symmetrically:
    ///
    /// - Object removal is a **flat 1 credit** whatever the clip. The website
    ///   computes the duration and resolution factors there and then discards
    ///   them — the formula is dead code on that page.
    /// - Background replacement genuinely scales, at a third of the rate.
    ///
    /// Applying the scaled formula to both quoted several credits for a long
    /// 4K removal the site charges 1 for.
    static func credits(duration: Double, size: CGSize, isBackground: Bool) -> Double {
        guard isBackground else { return 1 }
        guard duration > 0, size.width > 0, size.height > 0 else { return 0.1 }

        let durationFactor = duration / baseDuration
        let resolutionFactor = ((size.width * size.height) / basePixels).squareRoot()
        let raw = durationFactor * 0.7 + resolutionFactor * 0.3

        return max(0.1, ((raw / 3) * 10).rounded() / 10)
    }

    static func label(_ credits: Double) -> String {
        credits == credits.rounded()
            ? String(Int(credits))
            : String(format: "%.1f", credits)
    }
}


extension CreditEstimate {
    /// Balances can run to nine digits, which pushes the title out of the bar.
    /// Abbreviate anything over a thousand.
    static func compact(_ credits: Double) -> String {
        // Promote before formatting: 999,999,604 divided by a million is
        // 999.99, which prints as "1000M", which is not a number anyone writes.
        // Comparing against the tier boundary minus half a unit catches that.
        if credits >= 999_500_000 { return trim(credits / 1_000_000_000, "B") }
        if credits >= 999_500     { return trim(credits / 1_000_000, "M") }
        if credits >= 999.5       { return trim(credits / 1_000, "K") }
        return label(credits)
    }

    private static func trim(_ value: Double, _ suffix: String) -> String {
        // One decimal below ten, none above: 1.5K, 12K, 999K.
        value < 10
            ? String(format: "%.1f%@", value, suffix)
            : String(format: "%.0f%@", value, suffix)
    }
}
