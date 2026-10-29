import Foundation

/// The course as written by tools/build_course.py into course.json, shared with the web app.
struct Course: Codable {
    var title: String
    var built: String
    var lessons: [Lesson]
    var cards: [Card]

    static let empty = Course(title: "GPU Kernels to HSTU", built: "", lessons: [], cards: [])
}

struct Lesson: Codable, Identifiable, Hashable {
    var id: String
    var title: String
    var summary: String
    var minutes: Int
    var lab: String
    var track: String
    var sections: [LessonSection]

    enum CodingKeys: String, CodingKey {
        case id, title, summary, minutes, lab, track, sections
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        id = try c.decode(String.self, forKey: .id)
        title = try c.decodeIfPresent(String.self, forKey: .title) ?? id
        summary = try c.decodeIfPresent(String.self, forKey: .summary) ?? ""
        minutes = try c.decodeIfPresent(Int.self, forKey: .minutes) ?? 0
        lab = try c.decodeIfPresent(String.self, forKey: .lab) ?? ""
        track = try c.decodeIfPresent(String.self, forKey: .track) ?? ""
        sections = try c.decodeIfPresent([LessonSection].self, forKey: .sections) ?? []
    }

    /// Keys of the steps a learner ticks, numbered the way the web app numbers them.
    var stepKeys: [String] {
        let kinds = ["write", "run", "measure", "explain"]
        return sections.filter { kinds.contains($0.kind) }.enumerated().map { "\($0.element.kind)\($0.offset)" }
    }

    var measureFields: [MeasureField] {
        sections.filter { $0.kind == "measure" }.flatMap { $0.fields ?? [] }
    }

    var label: String { "\(id). \(title)" }
}

struct LessonSection: Codable, Hashable {
    var kind: String
    var html: String?
    var fields: [MeasureField]?
}

struct MeasureField: Codable, Hashable {
    var key: String
    var label: String
    var ref: Double?
    var unit: String
}

struct Card: Codable, Identifiable, Hashable {
    var id: String
    var q: String
    var a: String
    var lesson: String
}

enum CourseView: String, CaseIterable, Identifiable {
    case lessons, cards, progress
    var id: String { rawValue }
    var title: String { rawValue.capitalized }
}

enum CardMode: String, CaseIterable, Identifiable {
    case learn, test
    var id: String { rawValue }
    var title: String { rawValue.capitalized }
}

enum CardFilter: String, CaseIterable, Identifiable {
    case all, unknown, review
    var id: String { rawValue }
    var title: String {
        switch self {
        case .all: "All cards"
        case .unknown: "Not yet known"
        case .review: "Review only"
        }
    }
}

/// Lab steps, measurements and card marks. The `state` object of the web app's
/// "Export progress" file, so the two apps can swap files.
struct CourseState: Codable {
    var steps: [String: [String: Bool]] = [:]
    var measures: [String: [String: String]] = [:]
    /// "known" or "review" per card id.
    var cards: [String: String] = [:]
    /// Lessons included in Cards and Print; nil means all of them.
    var selected: [String]?

    enum CodingKeys: String, CodingKey {
        case steps, measures, cards, selected
    }

    init() {}

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        steps = (try? c.decodeIfPresent([String: [String: Bool]].self, forKey: .steps)) ?? [:]
        measures = (try? c.decodeIfPresent([String: [String: String]].self, forKey: .measures)) ?? [:]
        cards = (try? c.decodeIfPresent([String: String].self, forKey: .cards)) ?? [:]
        selected = (try? c.decodeIfPresent([String].self, forKey: .selected)) ?? nil
    }
}

struct ProgressFile: Codable {
    var app = "gk-course"
    var version = 1
    var exported = ""
    var state = CourseState()
}
