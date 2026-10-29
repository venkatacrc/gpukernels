import AppKit
import SwiftUI

final class AppDelegate: NSObject, NSApplicationDelegate {
    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApp.setActivationPolicy(.regular)
        NSApp.activate(ignoringOtherApps: true)
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool { true }
}

@main
struct GPUKernelsApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var appDelegate
    @StateObject private var store = CourseStore()
    @StateObject private var progress = ProgressStore()
    @StateObject private var navigator = Navigator()

    var body: some Scene {
        WindowGroup("GPU Kernels → HSTU") {
            ContentView()
                .environmentObject(store)
                .environmentObject(progress)
                .environmentObject(navigator)
                .frame(minWidth: 1000, minHeight: 680)
        }
        .commands {
            CommandGroup(replacing: .printItem) {
                Button("Print Lesson / Save as PDF…") {
                    if let lesson = navigator.current(in: store) {
                        LessonPrinter.run([lesson], state: progress.data, title: lesson.label)
                    }
                }
                .keyboardShortcut("p")
                Button("Print Selected Lessons…") {
                    let selected = progress.selected(in: store.lessons)
                    LessonPrinter.run(store.lessons.filter { selected.contains($0.id) },
                                      state: progress.data, title: store.course.title)
                }
                .keyboardShortcut("p", modifiers: [.command, .shift])
            }
            CommandGroup(after: .newItem) {
                Button("Export Progress…") { progress.exportProgress() }
                    .keyboardShortcut("e", modifiers: [.command, .shift])
                Button("Import Progress…") { progress.importProgress() }
                    .keyboardShortcut("i", modifiers: [.command, .shift])
                Divider()
                Button("Reset Progress…") { progress.reset() }
            }
            CommandMenu("Go") {
                ForEach(Array(CourseView.allCases.enumerated()), id: \.element) { index, view in
                    Button(view.title) { navigator.view = view }
                        .keyboardShortcut(KeyEquivalent(Character(String(index + 1))))
                }
                Divider()
                Button("Previous Lesson") { navigator.step(-1, in: store) }
                    .keyboardShortcut("[")
                Button("Next Lesson") { navigator.step(1, in: store) }
                    .keyboardShortcut("]")
                Button("Practise This Lesson's Cards") {
                    if let lesson = navigator.current(in: store) { navigator.practise(lesson.id) }
                }
                .keyboardShortcut("l")
            }
        }
    }
}
