import StoreKit

/// Credit packs, priced to match the website. Apple requires digital goods to
/// be sold through StoreKit, so these are consumables rather than Stripe links.
enum CreditPack: String, CaseIterable, Identifiable {
    case five = "com.markremoverai.app.credits5"
    case fifteen = "com.markremoverai.app.credits15"
    case sixty = "com.markremoverai.app.credits60"

    var id: String { rawValue }

    var credits: Int {
        switch self {
        case .five: return 5
        case .fifteen: return 15
        case .sixty: return 60
        }
    }

    var title: String {
        switch self {
        case .five: return "Starter"
        case .fifteen: return "Basic"
        case .sixty: return "Pro"
        }
    }

    var isBestValue: Bool { self == .sixty }
}

enum StoreError: LocalizedError {
    case unverified
    case redeemFailed(String)
    case productsUnavailable

    var errorDescription: String? {
        switch self {
        case .unverified:
            return "App Store couldn't verify that purchase."
        case .redeemFailed(let message):
            return message
        case .productsUnavailable:
            return "Credit packs aren't loading right now. Try again shortly."
        }
    }
}

@MainActor
final class Store: ObservableObject {
    @Published private(set) var products: [Product] = []
    @Published private(set) var isLoading = false
    @Published private(set) var purchasingID: String?

    private var updatesTask: Task<Void, Never>?

    init() {
        // Purchases can complete while the app is backgrounded, or be approved
        // later by a parent. This is the only place those arrive.
        updatesTask = Task { [weak self] in
            for await update in Transaction.updates {
                await self?.handle(update)
            }
        }
    }

    deinit { updatesTask?.cancel() }

    func loadProducts() async {
        guard products.isEmpty else { return }
        isLoading = true
        defer { isLoading = false }

        let loaded = try? await Product.products(for: CreditPack.allCases.map(\.rawValue))
        products = (loaded ?? []).sorted { $0.price < $1.price }
    }

    func pack(for product: Product) -> CreditPack? {
        CreditPack(rawValue: product.id)
    }

    /// Buys a pack and only finishes the transaction once the backend has
    /// actually banked the credits — an unfinished transaction is replayed by
    /// StoreKit, which is what makes a dropped network call recoverable.
    func purchase(_ product: Product, appState: AppState) async throws {
        purchasingID = product.id
        defer { purchasingID = nil }

        let result = try await product.purchase()

        switch result {
        case .success(let verification):
            try await redeem(verification, appState: appState)
        case .userCancelled:
            return
        case .pending:
            // Ask-to-buy and similar: it will arrive via Transaction.updates.
            return
        @unknown default:
            return
        }
    }

    /// Replays anything StoreKit still considers unfinished. Doubles as the
    /// "Restore" path for a purchase whose redeem call failed earlier.
    func redeemUnfinished(appState: AppState) async {
        for await result in Transaction.unfinished {
            try? await redeem(result, appState: appState)
        }
    }

    private func handle(_ result: VerificationResult<Transaction>) async {
        // AppState is the source of truth for the balance; refresh after.
        guard let appState = Store.activeAppState else { return }
        try? await redeem(result, appState: appState)
    }

    private func redeem(_ result: VerificationResult<Transaction>, appState: AppState) async throws {
        guard case .verified(let transaction) = result else {
            throw StoreError.unverified
        }

        do {
            let balance = try await APIClient.shared.redeemApplePurchase(
                signedTransaction: result.jwsRepresentation
            )
            await appState.applyCredits(balance)
            await transaction.finish()
        } catch {
            // Leave it unfinished on purpose so it can be retried rather than
            // the customer paying for credits that never arrived.
            throw StoreError.redeemFailed(error.localizedDescription)
        }
    }

    /// Set once at launch so background transaction updates have somewhere to
    /// write the new balance.
    nonisolated(unsafe) static var activeAppState: AppState?
}
