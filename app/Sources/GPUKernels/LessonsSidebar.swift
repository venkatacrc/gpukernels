import SwiftUI

/// Lessons, each with a box that includes it in Cards and Print and a ring for lab progress.
struct LessonsSidebar: View {
    @EnvironmentObject private var store: CourseStore
    @EnvironmentObject private var progress: ProgressStore
    @EnvironmentObject private var navigator: Navigator

    private var selected: Set<String> { progress.selected(in: store.lessons) }

    private var openLesson: Binding<String?> {
        Binding(
            get: { navigator.view == .lessons ? navigator.current(in: store)?.id : nil },
            set: { id in if let id { navigator.open(id) } })
    }

    var body: some View {
        List(selection: openLesson) {
            Section {
                ForEach(store.lessons) { lesson in
                    row(lesson)
                }
            } header: {
                HStack {
                    Text("Lessons")
                    Spacer()
                    Button("Select all") { progress.setSelected(Set(store.lessons.map(\.id)), in: store.lessons) }
                    Button("Clear") { progress.setSelected([], in: store.lessons) }
                }
                .buttonStyle(.link)
                .font(.caption)
            }
        }
        .listStyle(.sidebar)
        .safeAreaInset(edge: .top) {
            HStack(spacing: 10) {
                TileMark(size: 30)
                Text("GPU Kernels → HSTU").font(.title3.weight(.semibold))
                Spacer()
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 8)
        }
        .safeAreaInset(edge: .bottom) {
            let cards = store.course.cards.filter { selected.contains($0.lesson) }.count
            Text("\(selected.count) lessons selected · \(cards) cards")
                .font(.callout.weight(.medium))
                .frame(maxWidth: .infinity)
                .padding(10)
                .background(.bar)
        }
    }

    private func row(_ lesson: Lesson) -> some View {
        HStack(alignment: .top, spacing: 8) {
            Toggle("Include \(lesson.label) in Cards and Print", isOn: Binding(
                get: { selected.contains(lesson.id) },
                set: { on in
                    var ids = selected
                    if on { ids.insert(lesson.id) } else { ids.remove(lesson.id) }
                    progress.setSelected(ids, in: store.lessons)
                }))
            .toggleStyle(.checkbox)
            .labelsHidden()
            Text(lesson.id)
                .font(.callout.monospacedDigit())
                .foregroundStyle(.secondary)
                .frame(width: 22, alignment: .leading)
            VStack(alignment: .leading, spacing: 2) {
                Text(lesson.title).lineLimit(2)
                Text(detail(lesson))
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
            }
            Spacer(minLength: 4)
            ProgressRing(fraction: progress.fraction(lesson))
                .frame(width: 18, height: 18)
        }
        .padding(.vertical, 2)
        .tag(lesson.id)
    }

    private func detail(_ lesson: Lesson) -> String {
        var parts = ["\(lesson.minutes) min"]
        if !lesson.track.isEmpty { parts.append(lesson.track) }
        parts.append("\(store.cards(of: lesson.id).count) cards")
        return parts.joined(separator: " · ")
    }
}

struct ProgressRing: View {
    let fraction: Double

    var body: some View {
        ZStack {
            Circle().stroke(Color.secondary.opacity(0.25), lineWidth: 3)
            Circle()
                .trim(from: 0, to: fraction)
                .stroke(Brand.green, style: StrokeStyle(lineWidth: 3, lineCap: .round))
                .rotationEffect(.degrees(-90))
        }
        .accessibilityLabel("\(Int((fraction * 100).rounded()))% done")
    }
}

enum Brand {
    static let green = Color(red: 0.46, green: 0.73, blue: 0.0)
    static let deep = Color(red: 0.25, green: 0.45, blue: 0.0)
}

/// Four tiles of a matrix with the output tile lit: the app icon in miniature.
struct TileMark: View {
    let size: CGFloat

    var body: some View {
        let gap = size * 0.08
        let tile = (size * 0.64 - gap) / 2
        RoundedRectangle(cornerRadius: size * 0.24, style: .continuous)
            .fill(LinearGradient(colors: [Brand.green, Brand.deep], startPoint: .top, endPoint: .bottom))
            .frame(width: size, height: size)
            .overlay {
                VStack(spacing: gap) {
                    HStack(spacing: gap) { cell(tile, lit: false); cell(tile, lit: false) }
                    HStack(spacing: gap) { cell(tile, lit: false); cell(tile, lit: true) }
                }
            }
    }

    private func cell(_ side: CGFloat, lit: Bool) -> some View {
        RoundedRectangle(cornerRadius: side * 0.2, style: .continuous)
            .fill(Color(red: 1, green: 0.99, blue: 0.94).opacity(lit ? 1 : 0.5))
            .frame(width: side, height: side)
    }
}
