import SwiftUI

struct ProgressPage: View {
    @EnvironmentObject private var store: CourseStore
    @EnvironmentObject private var progress: ProgressStore
    @EnvironmentObject private var navigator: Navigator

    private var lessons: [Lesson] { store.lessons }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                Text(store.course.title.uppercased())
                    .font(.caption.weight(.bold))
                    .tracking(0.6)
                    .foregroundStyle(Brand.deep)
                Text("Progress").font(.system(size: 30, weight: .bold))
                HStack(spacing: 10) {
                    let total = lessons.reduce(0) { $0 + $1.stepKeys.count }
                    let done = lessons.reduce(0) { $0 + progress.stepsDone($1) }
                    let fields: [(lesson: String, key: String)] = lessons.flatMap { lesson in
                        lesson.measureFields.map { (lesson: lesson.id, key: $0.key) }
                    }
                    let measured = fields.filter { !progress.measure($0.lesson, $0.key).isEmpty }.count
                    kpi("\(done)/\(total)", "lab steps done")
                    kpi("\(measured)/\(fields.count)", "measurements")
                    kpi("\(progress.count("known"))", "cards known")
                    kpi("\(progress.count("review"))", "cards to review")
                }
                Text("LESSONS")
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(.secondary)
                    .padding(.top, 8)
                VStack(spacing: 0) {
                    ForEach(lessons) { lesson in
                        lessonRow(lesson)
                        if lesson.id != lessons.last?.id { Divider() }
                    }
                }
                .background(RoundedRectangle(cornerRadius: 14, style: .continuous).fill(Color.primary.opacity(0.04)))
                Text("Progress is saved on this Mac. Use File › Export Progress to move it to the web app on your phone.")
                    .font(.callout)
                    .foregroundStyle(.secondary)
            }
            .frame(maxWidth: 760, alignment: .leading)
            .padding(32)
            .frame(maxWidth: .infinity)
        }
        .background(Color(nsColor: .textBackgroundColor))
        .clipShape(RoundedRectangle(cornerRadius: 20, style: .continuous))
        .overlay(RoundedRectangle(cornerRadius: 20, style: .continuous).strokeBorder(Color(nsColor: .separatorColor)))
        .shadow(color: .black.opacity(0.07), radius: 12, y: 5)
    }

    private func lessonRow(_ lesson: Lesson) -> some View {
        Button { navigator.open(lesson.id) } label: {
            VStack(alignment: .leading, spacing: 6) {
                HStack(spacing: 12) {
                    Text(lesson.id)
                        .font(.callout.monospacedDigit())
                        .foregroundStyle(.secondary)
                        .frame(width: 24, alignment: .leading)
                    VStack(alignment: .leading, spacing: 2) {
                        Text(lesson.title)
                        Text("\(progress.stepsDone(lesson)) of \(lesson.stepKeys.count) steps")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                    Spacer()
                    ProgressView(value: progress.fraction(lesson))
                        .tint(Brand.green)
                        .frame(width: 90)
                }
                ForEach(lesson.measureFields.filter { !progress.measure(lesson.id, $0.key).isEmpty }, id: \.key) { field in
                    measureLine(lesson, field)
                        .padding(.leading, 36)
                }
            }
            .padding(.horizontal, 14)
            .padding(.vertical, 10)
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
    }

    private func measureLine(_ lesson: Lesson, _ field: MeasureField) -> some View {
        let value = progress.measure(lesson.id, field.key)
        let cmp = Comparison(value: value, ref: field.ref)
        return HStack(spacing: 6) {
            Text("\(field.label): ").foregroundColor(.secondary)
                + Text("\(value) \(field.unit)").bold()
            if let cmp {
                Text(cmp.text).foregroundStyle(cmp.color)
            }
        }
        .font(.caption)
    }

    private func kpi(_ value: String, _ caption: String) -> some View {
        VStack(spacing: 2) {
            Text(value).font(.title2.weight(.semibold)).monospacedDigit()
            Text(caption).font(.caption).foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 12)
        .background(RoundedRectangle(cornerRadius: 14, style: .continuous).fill(Color.primary.opacity(0.04)))
    }
}

/// Your measurement against the course GPU's, coloured as in the web app.
struct Comparison {
    let text: String
    let color: Color

    init?(value: String, ref: Double?) {
        guard let ref, ref != 0, let v = Double(value) else { return nil }
        let r = v / ref
        color = r > 0.8 && r < 1.25 ? .green : (r > 0.5 && r < 2 ? .orange : .red)
        let yours = r >= 1 ? String(format: "%.2f×", r) : String(format: "%.0f%%", r * 100)
        text = "reference \(Self.format(ref)) · yours is \(yours)"
    }

    static func format(_ x: Double) -> String {
        if abs(x) >= 100 {
            return NumberFormatter.localizedString(from: NSNumber(value: x.rounded()), number: .decimal)
        }
        return String(format: "%.3g", x)
    }
}
