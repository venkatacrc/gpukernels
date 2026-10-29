import SwiftUI

struct FlashCardView: View {
    let card: Card
    let lessonTitle: String
    let showAnswer: Bool
    let mark: String?

    var body: some View {
        VStack(alignment: .leading, spacing: 18) {
            HStack {
                Text(lessonTitle)
                    .font(.callout.weight(.semibold))
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
                Spacer()
                if let mark {
                    Text(mark == "known" ? "Known" : "Review")
                        .font(.caption.weight(.semibold))
                        .padding(.horizontal, 9)
                        .padding(.vertical, 3)
                        .foregroundStyle(mark == "known" ? Color.green : Color.orange)
                        .background(Capsule().fill((mark == "known" ? Color.green : Color.orange).opacity(0.16)))
                }
            }
            RichText.make(card.q, size: 22)
                .font(.system(size: 22, weight: .semibold))
                .lineSpacing(3)
                .fixedSize(horizontal: false, vertical: true)
                .textSelection(.enabled)
            if showAnswer {
                Divider()
                RichText.make(card.a, size: 17)
                    .font(.system(size: 17))
                    .lineSpacing(4)
                    .fixedSize(horizontal: false, vertical: true)
                    .textSelection(.enabled)
            } else {
                Spacer(minLength: 0)
                Text("Recall the answer, then press Space or click the card.")
                    .font(.callout)
                    .foregroundStyle(.secondary)
                    .frame(maxWidth: .infinity)
            }
            Spacer(minLength: 0)
        }
        .padding(32)
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
        .background(RoundedRectangle(cornerRadius: 20, style: .continuous).fill(Color(nsColor: .textBackgroundColor)))
        .overlay(RoundedRectangle(cornerRadius: 20, style: .continuous)
            .strokeBorder(showAnswer ? Brand.green.opacity(0.45) : Color(nsColor: .separatorColor)))
        .shadow(color: .black.opacity(0.07), radius: 12, y: 5)
        .contentShape(Rectangle())
    }
}

/// Card text from course.json: plain text with `<code>` and `<em>` spans.
enum RichText {
    private static let pattern = try! NSRegularExpression(
        pattern: "<(code|em)>(.*?)</(?:code|em)>", options: [.dotMatchesLineSeparators])

    static func make(_ html: String, size: CGFloat) -> Text {
        let ns = html as NSString
        var out = Text(verbatim: "")
        var last = 0
        for match in pattern.matches(in: html, range: NSRange(location: 0, length: ns.length)) {
            let before = ns.substring(with: NSRange(location: last, length: match.range.location - last))
            out = out + Text(verbatim: plain(before))
            let tag = ns.substring(with: match.range(at: 1))
            let inner = Text(verbatim: plain(ns.substring(with: match.range(at: 2))))
            if tag == "code" {
                out = out + inner.font(.system(size: size * 0.88, design: .monospaced))
            } else {
                out = out + inner.italic()
            }
            last = match.range.location + match.range.length
        }
        return out + Text(verbatim: plain(ns.substring(from: last)))
    }

    private static func plain(_ s: String) -> String {
        s.replacingOccurrences(of: "<[^>]+>", with: "", options: .regularExpression)
            .replacingOccurrences(of: "&lt;", with: "<")
            .replacingOccurrences(of: "&gt;", with: ">")
            .replacingOccurrences(of: "&quot;", with: "\"")
            .replacingOccurrences(of: "&#39;", with: "'")
            .replacingOccurrences(of: "&amp;", with: "&")
    }
}
