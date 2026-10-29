import AppKit
import Foundation
import UniformTypeIdentifiers

@MainActor
final class CourseStore: ObservableObject {
    @Published private(set) var course: Course = .empty
    @Published var errorMessage: String?

    init() { load() }

    /// The copy inside the .app first, then the SwiftPM resource bundle (used by `swift run`).
    func load() {
        guard let url = Bundle.main.url(forResource: "course", withExtension: "json")
            ?? Bundle.module.url(forResource: "course", withExtension: "json") else {
            errorMessage = "course.json is missing from the app. Run scripts/build_app.sh again."
            return
        }
        do {
            course = try JSONDecoder().decode(Course.self, from: Data(contentsOf: url))
        } catch {
            errorMessage = "Could not read course.json: \(error.localizedDescription)"
        }
    }

    var lessons: [Lesson] { course.lessons }

    func lesson(_ id: String) -> Lesson? { course.lessons.first { $0.id == id } }
    func cards(of lessonID: String) -> [Card] { course.cards.filter { $0.lesson == lessonID } }

    func title(of lessonID: String) -> String { lesson(lessonID)?.label ?? lessonID }
}

@MainActor
final class ProgressStore: ObservableObject {
    @Published private(set) var data = CourseState()
    /// Bumped when progress is replaced wholesale, so open lesson pages reload.
    @Published private(set) var revision = 0
    @Published var errorMessage: String?

    static var fileURL: URL {
        FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("GPUKernels", isDirectory: true)
            .appendingPathComponent("progress.json")
    }

    init() {
        if let raw = try? Data(contentsOf: Self.fileURL),
           let file = try? JSONDecoder().decode(ProgressFile.self, from: raw) {
            data = file.state
        }
    }

    // ---- lessons ----

    func stepsDone(_ lesson: Lesson) -> Int {
        let done = data.steps[lesson.id] ?? [:]
        return lesson.stepKeys.filter { done[$0] == true }.count
    }

    func fraction(_ lesson: Lesson) -> Double {
        let n = lesson.stepKeys.count
        return n == 0 ? 0 : Double(stepsDone(lesson)) / Double(n)
    }

    func measure(_ lessonID: String, _ key: String) -> String {
        data.measures[lessonID]?[key] ?? ""
    }

    func measuresDone(_ lesson: Lesson) -> Bool {
        let fields = lesson.measureFields
        return !fields.isEmpty && fields.allSatisfy { !measure(lesson.id, $0.key).isEmpty }
    }

    func setStep(_ lessonID: String, _ key: String, on: Bool) {
        data.steps[lessonID, default: [:]][key] = on
        save()
    }

    func setMeasure(_ lesson: Lesson, _ key: String, value: String) {
        data.measures[lesson.id, default: [:]][key] = value
        if measuresDone(lesson), let step = lesson.stepKeys.first(where: { $0.hasPrefix("measure") }) {
            data.steps[lesson.id, default: [:]][step] = true
        }
        save()
    }

    func reveal(_ lessonID: String) {
        data.steps[lessonID, default: [:]]["revealed"] = true
        save()
    }

    // ---- cards ----

    func mark(_ cardID: String) -> String? { data.cards[cardID] }

    /// Toggles a "known" or "review" mark, as the Known and Review buttons do.
    func toggle(_ cardID: String, _ kind: String) {
        if data.cards[cardID] == kind {
            data.cards.removeValue(forKey: cardID)
        } else {
            data.cards[cardID] = kind
        }
        save()
    }

    func set(_ cardID: String, _ kind: String) {
        data.cards[cardID] = kind
        save()
    }

    func count(_ kind: String) -> Int { data.cards.values.filter { $0 == kind }.count }

    // ---- lesson selection for Cards and Print ----

    func selected(in lessons: [Lesson]) -> Set<String> {
        Set(data.selected ?? lessons.map(\.id))
    }

    func setSelected(_ ids: Set<String>, in lessons: [Lesson]) {
        let all = lessons.map(\.id)
        data.selected = ids.count == all.count ? nil : all.filter { ids.contains($0) }
        save()
    }

    // ---- files ----

    func exportProgress() {
        let panel = NSSavePanel()
        panel.title = "Export progress"
        panel.message = "The web app can import this file from its ☰ menu."
        let day = ISO8601DateFormatter().string(from: Date()).prefix(10)
        panel.nameFieldStringValue = "gk-progress-\(day).json"
        panel.allowedContentTypes = [.json]
        guard panel.runModal() == .OK, let url = panel.url else { return }
        do {
            let file = ProgressFile(exported: ISO8601DateFormatter().string(from: Date()), state: data)
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
            try encoder.encode(file).write(to: url)
        } catch {
            errorMessage = "Export failed: \(error.localizedDescription)"
        }
    }

    /// Replaces progress with a file exported from the web app or from this app, as the web app does.
    func importProgress() {
        let panel = NSOpenPanel()
        panel.title = "Import progress"
        panel.allowedContentTypes = [.json]
        panel.allowsMultipleSelection = false
        guard panel.runModal() == .OK, let url = panel.url else { return }
        do {
            let file = try JSONDecoder().decode(ProgressFile.self, from: Data(contentsOf: url))
            guard file.app == "gk-course" else {
                errorMessage = "That file is not a course progress export."
                return
            }
            data = file.state
            revision += 1
            save()
        } catch {
            errorMessage = "Import failed: \(error.localizedDescription)"
        }
    }

    func reset() {
        let alert = NSAlert()
        alert.messageText = "Reset progress?"
        alert.informativeText = "This clears every lab step, measurement and card mark stored by this app."
        alert.addButton(withTitle: "Reset")
        alert.addButton(withTitle: "Cancel")
        alert.buttons.first?.hasDestructiveAction = true
        guard alert.runModal() == .alertFirstButtonReturn else { return }
        let selected = data.selected
        data = CourseState()
        data.selected = selected
        revision += 1
        save()
    }

    private func save() {
        do {
            try FileManager.default.createDirectory(
                at: Self.fileURL.deletingLastPathComponent(), withIntermediateDirectories: true)
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.sortedKeys]
            try encoder.encode(ProgressFile(state: data)).write(to: Self.fileURL, options: .atomic)
        } catch {
            errorMessage = "Could not save progress: \(error.localizedDescription)"
        }
    }
}

/// What the window shows. Shared by the window and the menu commands.
@MainActor
final class Navigator: ObservableObject {
    private let defaults = UserDefaults.standard

    @Published var view: CourseView = .lessons {
        didSet { defaults.set(view.rawValue, forKey: "view") }
    }
    @Published var lessonID = "" {
        didSet { defaults.set(lessonID, forKey: "lesson") }
    }
    @Published var cardMode: CardMode = .learn {
        didSet { defaults.set(cardMode.rawValue, forKey: "cardMode") }
    }
    @Published var filter: CardFilter = .all {
        didSet { defaults.set(filter.rawValue, forKey: "filter") }
    }
    @Published var shuffle = false {
        didSet { defaults.set(shuffle, forKey: "shuffle") }
    }
    /// When set, Cards practises only this lesson's cards.
    @Published var focus: String?

    init() {
        let d = UserDefaults.standard
        view = CourseView(rawValue: d.string(forKey: "view") ?? "") ?? .lessons
        lessonID = d.string(forKey: "lesson") ?? ""
        cardMode = CardMode(rawValue: d.string(forKey: "cardMode") ?? "") ?? .learn
        filter = CardFilter(rawValue: d.string(forKey: "filter") ?? "") ?? .all
        shuffle = d.bool(forKey: "shuffle")
    }

    func current(in store: CourseStore) -> Lesson? {
        store.lesson(lessonID) ?? store.lessons.first
    }

    func open(_ id: String) {
        lessonID = id
        view = .lessons
    }

    func step(_ delta: Int, in store: CourseStore) {
        guard let lesson = current(in: store), let i = store.lessons.firstIndex(of: lesson),
              store.lessons.indices.contains(i + delta) else { return }
        open(store.lessons[i + delta].id)
    }

    func canStep(_ delta: Int, in store: CourseStore) -> Bool {
        guard let lesson = current(in: store), let i = store.lessons.firstIndex(of: lesson) else { return false }
        return store.lessons.indices.contains(i + delta)
    }

    func practise(_ lessonID: String) {
        focus = lessonID
        view = .cards
    }
}

/// The deck being practised in Cards.
@MainActor
final class StudySession: ObservableObject {
    @Published var deck: [Card] = []
    @Published var position = 0
    @Published var revealed = false
    @Published var finished = false
    @Published var answers: [String: Bool] = [:]

    var current: Card? { deck.indices.contains(position) ? deck[position] : nil }
    var got: Int { answers.values.filter { $0 }.count }
    var missed: [String] { deck.map(\.id).filter { answers[$0] == false } }

    func rebuild(store: CourseStore, progress: ProgressStore, navigator: Navigator) {
        let lessons: Set<String>
        if let focus = navigator.focus, store.lesson(focus) != nil {
            lessons = [focus]
        } else {
            lessons = progress.selected(in: store.lessons)
        }
        let filter = navigator.filter
        var cards = store.course.cards.filter { lessons.contains($0.lesson) }.filter { card in
            let mark = progress.mark(card.id)
            switch filter {
            case .all: return true
            case .unknown: return mark != "known"
            case .review: return mark == "review"
            }
        }
        if navigator.shuffle { cards.shuffle() }
        start(cards)
    }

    func start(_ cards: [Card]) {
        deck = cards
        position = 0
        revealed = false
        finished = false
        answers = [:]
    }

    func retestMissed() {
        let ids = Set(missed)
        start(deck.filter { ids.contains($0.id) })
    }

    func move(_ delta: Int, testing: Bool) {
        guard !deck.isEmpty, !finished else { return }
        let next = position + delta
        guard next >= 0 else { return }
        if next >= deck.count {
            if testing { finished = true }
            return
        }
        position = next
        revealed = false
    }

    func answer(_ got: Bool, progress: ProgressStore) {
        guard revealed, !finished, let card = current else { return }
        answers[card.id] = got
        progress.set(card.id, got ? "known" : "review")
        if position + 1 >= deck.count {
            finished = true
        } else {
            position += 1
        }
        revealed = false
    }
}
