import SwiftUI

/// The log, on screen. The failures that matter happen on a real phone on a
/// real network, where there is no Xcode console to read.
struct ConsoleView: View {
    @ObservedObject private var log = LogStore.shared
    @Environment(\.dismiss) private var dismiss

    @State private var filter: AppLog.Category?
    @State private var errorsOnly = false

    private var visible: [LogEntry] {
        log.entries.filter { entry in
            (filter == nil || entry.category == filter)
                && (!errorsOnly || entry.level == .error)
        }
    }

    var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                controls

                if visible.isEmpty {
                    ContentUnavailableView(
                        "Nothing logged yet",
                        systemImage: "text.alignleft",
                        description: Text("Actions you take in the app show up here.")
                    )
                } else {
                    ScrollViewReader { proxy in
                        List(visible) { entry in
                            Text(entry.line)
                                .font(.system(.caption, design: .monospaced))
                                .foregroundStyle(entry.level == .error ? .red : .primary)
                                .textSelection(.enabled)
                                .listRowInsets(EdgeInsets(top: 3, leading: 12, bottom: 3, trailing: 12))
                                .id(entry.id)
                        }
                        .listStyle(.plain)
                        .onChange(of: visible.count) { _, _ in
                            guard let last = visible.last else { return }
                            withAnimation { proxy.scrollTo(last.id, anchor: .bottom) }
                        }
                    }
                }
            }
            .navigationTitle("Console")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarLeading) {
                    Button("Clear") { log.clear() }
                }
                ToolbarItem(placement: .topBarTrailing) {
                    HStack(spacing: 14) {
                        ShareLink(item: log.transcript) {
                            Image(systemName: "square.and.arrow.up")
                        }
                        Button("Done") { dismiss() }
                    }
                }
            }
        }
    }

    private var controls: some View {
        VStack(spacing: 8) {
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 8) {
                    chip("All", active: filter == nil) { filter = nil }
                    ForEach(AppLog.Category.allCases, id: \.self) { category in
                        chip(category.rawValue, active: filter == category) { filter = category }
                    }
                }
                .padding(.horizontal, 12)
            }

            Toggle("Errors only", isOn: $errorsOnly)
                .font(.caption)
                .padding(.horizontal, 14)
        }
        .padding(.vertical, 10)
        .background(Color(.secondarySystemGroupedBackground))
    }

    private func chip(_ label: String, active: Bool, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            Text(label)
                .font(.caption.weight(.medium))
                .padding(.horizontal, 12)
                .padding(.vertical, 6)
                .background(active ? Theme.accent : Color(.tertiarySystemFill))
                .foregroundStyle(active ? .white : .primary)
                .clipShape(Capsule())
        }
        .buttonStyle(.plain)
    }
}
