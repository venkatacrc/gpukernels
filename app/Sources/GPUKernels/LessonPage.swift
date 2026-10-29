import AppKit
import SwiftUI
import WebKit

/// Builds the HTML for one or more lessons; Resources/lesson.js renders them in the page.
enum LessonPage {
    static let base = URL(string: "https://gpukernels.local/")!
    static let script = resource("lesson", "js")
    static let style = resource("lesson", "css")

    private struct Payload: Encodable {
        let lessons: [Lesson]
        let steps: [String: [String: Bool]]
        let measures: [String: [String: String]]
        let print: Bool
    }

    private static func resource(_ name: String, _ ext: String) -> String {
        guard let url = Bundle.main.url(forResource: name, withExtension: ext)
                ?? Bundle.module.url(forResource: name, withExtension: ext),
              let text = try? String(contentsOf: url, encoding: .utf8) else { return "" }
        return text
    }

    static func document(_ lessons: [Lesson], state: CourseState, print: Bool, title: String) -> String {
        let payload = Payload(lessons: lessons, steps: state.steps, measures: state.measures, print: print)
        let json = (try? JSONEncoder().encode(payload)).flatMap { String(data: $0, encoding: .utf8) } ?? "{}"
        // Keep "</script>" inside a lesson from closing the script tag.
        let safe = json.replacingOccurrences(of: "</", with: "<\\/")
        let escapedTitle = title.replacingOccurrences(of: "<", with: "&lt;")
        return """
        <!doctype html>
        <html><head><meta charset="utf-8"><title>\(escapedTitle)</title><style>\(style)</style></head>
        <body><main id="root"></main>
        <script>window.GK = \(safe);</script>
        <script>\(script)</script>
        </body></html>
        """
    }
}

/// The open lesson. Ticks, measurements and "show the explanation" come back as script messages.
struct LessonWebView: NSViewRepresentable {
    let lesson: Lesson
    let progress: ProgressStore

    func makeCoordinator() -> Coordinator { Coordinator(progress: progress, lesson: lesson) }

    func makeNSView(context: Context) -> WKWebView {
        let config = WKWebViewConfiguration()
        config.userContentController.add(context.coordinator, name: "gk")
        let view = WKWebView(frame: .zero, configuration: config)
        view.navigationDelegate = context.coordinator
        view.setValue(false, forKey: "drawsBackground")
        view.allowsMagnification = true
        return view
    }

    func updateNSView(_ view: WKWebView, context: Context) {
        context.coordinator.progress = progress
        context.coordinator.lesson = lesson
        let key = "\(lesson.id)#\(progress.revision)"
        guard context.coordinator.loaded != key else { return }
        context.coordinator.loaded = key
        view.loadHTMLString(
            LessonPage.document([lesson], state: progress.data, print: false, title: lesson.label),
            baseURL: LessonPage.base)
    }

    static func dismantleNSView(_ view: WKWebView, coordinator: Coordinator) {
        view.configuration.userContentController.removeScriptMessageHandler(forName: "gk")
    }

    @MainActor
    final class Coordinator: NSObject, WKScriptMessageHandler, WKNavigationDelegate {
        var progress: ProgressStore
        var lesson: Lesson
        var loaded = ""

        init(progress: ProgressStore, lesson: Lesson) {
            self.progress = progress
            self.lesson = lesson
        }

        func userContentController(_ controller: WKUserContentController, didReceive message: WKScriptMessage) {
            guard let body = message.body as? [String: Any], let type = body["type"] as? String else { return }
            let lessonID = body["lesson"] as? String ?? lesson.id
            switch type {
            case "step":
                progress.setStep(lessonID, body["key"] as? String ?? "", on: body["on"] as? Bool ?? false)
            case "measure":
                if lessonID == lesson.id {
                    progress.setMeasure(lesson, body["key"] as? String ?? "", value: body["value"] as? String ?? "")
                }
            case "reveal":
                progress.reveal(lessonID)
            case "copy":
                NSPasteboard.general.clearContents()
                NSPasteboard.general.setString(body["text"] as? String ?? "", forType: .string)
            default:
                break
            }
        }

        func webView(_ webView: WKWebView, decidePolicyFor navigationAction: WKNavigationAction) async -> WKNavigationActionPolicy {
            guard navigationAction.navigationType == .linkActivated, let url = navigationAction.request.url else {
                return .allow
            }
            if url.host != LessonPage.base.host { NSWorkspace.shared.open(url) }
            return .cancel
        }
    }
}

/// Prints lessons from an off-screen web view, so Cards and Progress can print the selected lessons.
@MainActor
final class LessonPrinter: NSObject, WKNavigationDelegate {
    private static var current: LessonPrinter?
    private let window: NSWindow
    private let webView: WKWebView
    private let title: String

    static func run(_ lessons: [Lesson], state: CourseState, title: String) {
        guard !lessons.isEmpty else {
            NSSound.beep()
            return
        }
        let printer = LessonPrinter(title: title)
        current = printer
        printer.webView.loadHTMLString(
            LessonPage.document(lessons, state: state, print: true, title: title), baseURL: LessonPage.base)
    }

    private init(title: String) {
        self.title = title
        let frame = NSRect(x: 0, y: 0, width: 780, height: 1000)
        webView = WKWebView(frame: frame, configuration: WKWebViewConfiguration())
        window = NSWindow(contentRect: frame.offsetBy(dx: -20000, dy: -20000), styleMask: [.borderless],
                          backing: .buffered, defer: false)
        window.isReleasedWhenClosed = false
        super.init()
        window.contentView = webView
        webView.navigationDelegate = self
    }

    func webView(_ webView: WKWebView, didFinish navigation: WKNavigation!) {
        let info = NSPrintInfo.shared.copy() as! NSPrintInfo
        info.topMargin = 40
        info.bottomMargin = 40
        info.leftMargin = 40
        info.rightMargin = 40
        info.horizontalPagination = .fit
        info.verticalPagination = .automatic
        info.isVerticallyCentered = false
        let op = webView.printOperation(with: info)
        op.jobTitle = title
        op.showsPrintPanel = true
        op.showsProgressPanel = true
        op.view?.frame = webView.bounds
        if let host = NSApp.keyWindow ?? NSApp.mainWindow {
            op.runModal(for: host, delegate: self,
                        didRun: #selector(printOperationDidRun(_:success:contextInfo:)), contextInfo: nil)
        } else {
            op.run()
            finish()
        }
    }

    @objc private func printOperationDidRun(_ operation: NSPrintOperation, success: Bool,
                                            contextInfo: UnsafeMutableRawPointer?) {
        finish()
    }

    private func finish() {
        window.close()
        LessonPrinter.current = nil
    }
}
