import SwiftUI

struct ContentView: View {
    @EnvironmentObject private var store: CourseStore
    @EnvironmentObject private var progress: ProgressStore
    @EnvironmentObject private var navigator: Navigator
    @StateObject private var session = StudySession()

    private var testing: Bool { navigator.cardMode == .test }

    var body: some View {
        NavigationSplitView {
            LessonsSidebar()
                .navigationSplitViewColumnWidth(min: 260, ideal: 310)
        } detail: {
            detail
                .padding(24)
                .toolbar { toolbar }
                .navigationTitle(navigator.view == .lessons ? (navigator.current(in: store)?.label ?? "Lessons") : navigator.view.title)
        }
        .onAppear(perform: rebuild)
        .onChange(of: navigator.view) { if navigator.view == .cards { rebuild() } }
        .onChange(of: navigator.cardMode) { rebuild() }
        .onChange(of: navigator.filter) { rebuild() }
        .onChange(of: navigator.shuffle) { rebuild() }
        .onChange(of: navigator.focus) { rebuild() }
        .onChange(of: progress.data.selected) { rebuild() }
        .onChange(of: progress.revision) { rebuild() }
        .alert("Something went wrong", isPresented: Binding(
            get: { store.errorMessage != nil || progress.errorMessage != nil },
            set: { if !$0 { store.errorMessage = nil; progress.errorMessage = nil } })
        ) {
            Button("OK") { store.errorMessage = nil; progress.errorMessage = nil }
        } message: {
            Text(store.errorMessage ?? progress.errorMessage ?? "")
        }
    }

    private func rebuild() {
        session.rebuild(store: store, progress: progress, navigator: navigator)
    }

    @ViewBuilder
    private var detail: some View {
        switch navigator.view {
        case .lessons: lessonDetail
        case .cards: cardsDetail
        case .progress: ProgressPage()
        }
    }

    // ---- lessons ----

    @ViewBuilder
    private var lessonDetail: some View {
        if let lesson = navigator.current(in: store) {
            VStack(spacing: 16) {
                HStack {
                    Text("Lesson \(lesson.id) · \(lesson.minutes) min\(lesson.track.isEmpty ? "" : " · " + lesson.track)")
                        .font(.headline)
                    Spacer()
                    Text("\(progress.stepsDone(lesson)) of \(lesson.stepKeys.count) steps")
                        .font(.callout)
                        .foregroundStyle(.secondary)
                }
                LessonWebView(lesson: lesson, progress: progress)
                    .background(Color(nsColor: .textBackgroundColor))
                    .clipShape(RoundedRectangle(cornerRadius: 20, style: .continuous))
                    .overlay(RoundedRectangle(cornerRadius: 20, style: .continuous).strokeBorder(Color(nsColor: .separatorColor)))
                    .shadow(color: .black.opacity(0.07), radius: 12, y: 5)
                HStack(spacing: 12) {
                    Button { navigator.step(-1, in: store) } label: { Label("Previous lesson", systemImage: "chevron.left") }
                        .disabled(!navigator.canStep(-1, in: store))
                        .help("Previous lesson (⌘[)")
                    Button { navigator.step(1, in: store) } label: { Label("Next lesson", systemImage: "chevron.right") }
                        .disabled(!navigator.canStep(1, in: store))
                        .help("Next lesson (⌘])")
                    Spacer()
                    let count = store.cards(of: lesson.id).count
                    Button { navigator.practise(lesson.id) } label: {
                        Label("Practise \(count) cards", systemImage: "rectangle.on.rectangle.angled")
                    }
                    .disabled(count == 0)
                    .help("Cards from this lesson only (⌘L)")
                    Button { LessonPrinter.run([lesson], state: progress.data, title: lesson.label) } label: {
                        Label("Print", systemImage: "printer")
                    }
                    .help("Print or save as PDF (⌘P)")
                }
                .controlSize(.large)
            }
        } else {
            ContentUnavailableView("No lessons", systemImage: "book", description: Text("course.json has no lessons."))
        }
    }

    // ---- cards ----

    @ViewBuilder
    private var cardsDetail: some View {
        VStack(spacing: 18) {
            if let focus = navigator.focus, store.lesson(focus) != nil {
                HStack(spacing: 12) {
                    Text("Cards from \(store.title(of: focus))").font(.headline)
                    Spacer()
                    Button("All selected lessons") { navigator.focus = nil }
                }
            }
            if testing && session.finished {
                testSummary
            } else if let card = session.current {
                cardHeader
                FlashCardView(
                    card: card,
                    lessonTitle: store.title(of: card.lesson),
                    showAnswer: !testing || session.revealed,
                    mark: progress.mark(card.id))
                .onTapGesture { if testing { session.revealed = true } }
                cardControls(card)
            } else {
                ContentUnavailableView(
                    "No cards to practise",
                    systemImage: "rectangle.stack",
                    description: Text(emptyHint))
            }
        }
    }

    private var emptyHint: String {
        switch navigator.filter {
        case .review: return "No cards in the selected lessons are marked for review. Press R on a card to mark it."
        case .unknown: return "Every card in the selected lessons is marked known."
        case .all: return "Tick at least one lesson in the sidebar."
        }
    }

    private var cardHeader: some View {
        HStack {
            Text("Card \(session.position + 1) of \(session.deck.count)").font(.headline)
            Spacer()
            if testing {
                let answered = session.answers.count
                Text("Score \(session.got) of \(answered) · \(session.deck.count - answered) to go")
                    .font(.callout.weight(.medium))
            } else {
                let known = session.deck.filter { progress.mark($0.id) == "known" }.count
                let review = session.deck.filter { progress.mark($0.id) == "review" }.count
                Text("Known \(known) · Review \(review)")
                    .font(.callout)
                    .foregroundStyle(.secondary)
            }
        }
    }

    private func cardControls(_ card: Card) -> some View {
        HStack(spacing: 12) {
            Button { session.move(-1, testing: testing) } label: { Label("Previous", systemImage: "chevron.left") }
                .keyboardShortcut(.leftArrow, modifiers: [])
            if testing {
                if session.revealed {
                    Button { session.answer(false, progress: progress) } label: { Label("Missed it (M)", systemImage: "xmark.circle") }
                        .keyboardShortcut("m", modifiers: [])
                        .tint(.orange)
                    Button { session.answer(true, progress: progress) } label: { Label("Got it (G)", systemImage: "checkmark.circle") }
                        .keyboardShortcut("g", modifiers: [])
                        .tint(.green)
                        .buttonStyle(.borderedProminent)
                } else {
                    Button { session.revealed = true } label: { Label("Show answer", systemImage: "eye") }
                        .keyboardShortcut(.space, modifiers: [])
                        .buttonStyle(.borderedProminent)
                }
            }
            Button { session.move(1, testing: testing) } label: {
                Label(testing ? "Skip" : "Next", systemImage: "chevron.right")
            }
            .keyboardShortcut(.rightArrow, modifiers: [])
            Spacer()
            if !testing {
                let mark = progress.mark(card.id)
                Button { progress.toggle(card.id, "known") } label: {
                    Label("Known (K)", systemImage: mark == "known" ? "checkmark.circle.fill" : "checkmark.circle")
                }
                .keyboardShortcut("k", modifiers: [])
                Button { progress.toggle(card.id, "review") } label: {
                    Label("Review (R)", systemImage: mark == "review" ? "flag.fill" : "flag")
                }
                .keyboardShortcut("r", modifiers: [])
            }
        }
        .controlSize(.large)
    }

    @ViewBuilder
    private var testSummary: some View {
        let missed = session.missed
        let total = session.deck.count
        let percent = total == 0 ? 0 : Int((Double(session.got) / Double(total) * 100).rounded())
        VStack(spacing: 16) {
            Image(systemName: missed.isEmpty ? "star.circle.fill" : "checkmark.seal.fill")
                .font(.system(size: 56))
                .foregroundStyle(missed.isEmpty ? .yellow : .green)
            Text("Test complete").font(.largeTitle.weight(.semibold))
            Text("\(session.got) of \(total) correct (\(percent)%)").font(.title2)
            Text(missed.isEmpty
                 ? "Every card you answered is now marked known."
                 : "Cards you got are marked known; the \(missed.count) you missed are marked for review.")
                .foregroundStyle(.secondary)
            HStack(spacing: 12) {
                if !missed.isEmpty {
                    Button("Retest the \(missed.count) missed") { session.retestMissed() }
                        .buttonStyle(.borderedProminent)
                        .keyboardShortcut(.defaultAction)
                }
                Button("Start a new test") { rebuild() }
                Button("Switch to Learn") { navigator.cardMode = .learn }
            }
            .controlSize(.large)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    // ---- toolbar ----

    @ToolbarContentBuilder
    private var toolbar: some ToolbarContent {
        ToolbarItem(placement: .navigation) {
            Picker("View", selection: $navigator.view) {
                ForEach(CourseView.allCases) { Text($0.title).tag($0) }
            }
            .pickerStyle(.segmented)
            .help("Lessons, Cards, Progress (⌘1 to ⌘3)")
        }
        ToolbarItemGroup(placement: .primaryAction) {
            if navigator.view == .cards {
                Picker("Mode", selection: $navigator.cardMode) {
                    ForEach(CardMode.allCases) { Text($0.title).tag($0) }
                }
                .pickerStyle(.segmented)
                .help("Learn: see each card in full. Test: recall it, then mark whether you got it.")
                Picker("Cards", selection: $navigator.filter) {
                    ForEach(CardFilter.allCases) { Text($0.title).tag($0) }
                }
                .pickerStyle(.menu)
                Toggle(isOn: $navigator.shuffle) { Label("Shuffle", systemImage: "shuffle") }
                    .toggleStyle(.button)
                    .help("Shuffle the deck")
                Button { rebuild() } label: { Label("Restart", systemImage: "arrow.counterclockwise") }
                    .help("Start the deck again")
            }
        }
    }
}
