import StoreKit
import SwiftUI

struct PaywallView: View {
    /// False when the paywall is the root view (the debug entry point), where
    /// `dismiss()` has nothing to dismiss and a Done button would do nothing.
    var isPresented = true

    @EnvironmentObject private var appState: AppState
    @EnvironmentObject private var store: Store
    @Environment(\.dismiss) private var dismiss

    @State private var error: String?
    @State private var notice: String?

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 20) {
                    header

                    if store.isLoading && store.products.isEmpty {
                        ProgressView().padding(.vertical, 40)
                    } else if store.products.isEmpty {
                        unavailable
                    } else {
                        ForEach(store.products, id: \.id) { product in
                            packRow(product)
                        }
                    }

                    if let notice {
                        message(notice, color: Theme.positive)
                    }
                    if let error {
                        message(error, color: .red)
                    }

                    Button("Restore purchases") {
                        Task {
                            await store.redeemUnfinished(appState: appState)
                            notice = "Checked for anything unfinished."
                        }
                    }
                    .font(.subheadline)
                    .padding(.top, 4)

                    Text("Credits never expire. One credit removes an object from one video.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .multilineTextAlignment(.center)
                        .padding(.top, 4)
                }
                .padding(20)
            }
            .background(Color(.systemGroupedBackground))
            .navigationTitle("Get credits")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                if isPresented {
                    ToolbarItem(placement: .topBarTrailing) {
                        Button("Done") { dismiss() }
                    }
                }
            }
        }
        .task { await store.loadProducts() }
    }

    private var header: some View {
        VStack(spacing: 6) {
            Text("\(Int(appState.credits))")
                .font(.system(size: 44, weight: .bold, design: .rounded))
                .foregroundStyle(Theme.accent)
            Text(appState.credits == 1 ? "credit left" : "credits left")
                .font(.subheadline)
                .foregroundStyle(.secondary)
        }
        .padding(.bottom, 4)
    }

    private var unavailable: some View {
        VStack(spacing: 10) {
            Image(systemName: "cart.badge.questionmark")
                .font(.system(size: 34))
                .foregroundStyle(.secondary)
            Text(StoreError.productsUnavailable.localizedDescription)
                .font(.subheadline)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
        }
        .padding(.vertical, 30)
    }

    private func packRow(_ product: Product) -> some View {
        let pack = store.pack(for: product)
        let busy = store.purchasingID == product.id

        return Button {
            Task { await buy(product) }
        } label: {
            HStack(spacing: 14) {
                VStack(alignment: .leading, spacing: 3) {
                    HStack(spacing: 6) {
                        Text(pack?.title ?? product.displayName)
                            .font(.headline)
                        if pack?.isBestValue == true {
                            Text("BEST VALUE")
                                .font(.caption2.bold())
                                .padding(.horizontal, 7)
                                .padding(.vertical, 3)
                                .background(Theme.accentSoft)
                                .foregroundStyle(Theme.accent)
                                .clipShape(Capsule())
                        }
                    }
                    Text("\(pack?.credits ?? 0) credits")
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                }

                Spacer()

                if busy {
                    ProgressView()
                } else {
                    Text(product.displayPrice)
                        .font(.headline)
                        .foregroundStyle(Theme.accent)
                }
            }
            .padding(16)
            .background(Color(.secondarySystemGroupedBackground))
            .clipShape(RoundedRectangle(cornerRadius: 14, style: .continuous))
        }
        .buttonStyle(.plain)
        .disabled(store.purchasingID != nil)
    }

    private func message(_ text: String, color: Color) -> some View {
        Text(text)
            .font(.footnote)
            .foregroundStyle(color)
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(12)
            .background(color.opacity(0.1))
            .clipShape(RoundedRectangle(cornerRadius: 10, style: .continuous))
    }

    private func buy(_ product: Product) async {
        error = nil
        notice = nil
        do {
            try await store.purchase(product, appState: appState)
            if let pack = store.pack(for: product) {
                notice = "\(pack.credits) credits added."
            }
        } catch {
            self.error = error.localizedDescription
        }
    }
}
