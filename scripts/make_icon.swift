// Draws the app icon (a 3×3 grid of matrix tiles, output tile lit) and writes an .iconset
// folder for iconutil, or with --web the PNGs for the web app (icon-192/512 plus a
// full-bleed maskable-512 for Android).
// Usage: swift scripts/make_icon.swift OUT.iconset
//        swift scripts/make_icon.swift --web web/icons
import AppKit

enum Style { case mac, web, maskable }

let web = CommandLine.arguments[1] == "--web"
let out = URL(fileURLWithPath: CommandLine.arguments[web ? 2 : 1], isDirectory: true)
try FileManager.default.createDirectory(at: out, withIntermediateDirectories: true)

func render(_ px: Int, _ style: Style = .mac) -> Data {
    let rep = NSBitmapImageRep(
        bitmapDataPlanes: nil, pixelsWide: px, pixelsHigh: px, bitsPerSample: 8,
        samplesPerPixel: 4, hasAlpha: true, isPlanar: false, colorSpaceName: .deviceRGB,
        bytesPerRow: 0, bitsPerPixel: 0)!
    NSGraphicsContext.saveGraphicsState()
    NSGraphicsContext.current = NSGraphicsContext(bitmapImageRep: rep)
    let s = CGFloat(px)

    // macOS icon grid: the tile is inset ~10% with a ~22% corner radius. Android crops
    // maskable icons to its own shape, keeping only the central 80% circle.
    let inset: CGFloat = style == .mac ? 0.1 : style == .web ? 0.02 : 0
    let tile = NSRect(x: s * inset, y: s * inset, width: s * (1 - 2 * inset), height: s * (1 - 2 * inset))
    let radius = style == .maskable ? 0 : tile.width * 0.225
    NSGradient(colors: [
        NSColor(red: 0.46, green: 0.73, blue: 0.00, alpha: 1),
        NSColor(red: 0.20, green: 0.41, blue: 0.00, alpha: 1),
    ])!.draw(in: NSBezierPath(roundedRect: tile, xRadius: radius, yRadius: radius), angle: -90)

    let span = (style == .maskable ? s * 0.46 : tile.width * 0.56)
    let gap = span * 0.08
    let cell = (span - 2 * gap) / 3
    let left = tile.midX - span / 2
    let bottom = tile.midY - span / 2
    let shadow = NSShadow()
    shadow.shadowColor = NSColor(white: 0, alpha: 0.25)
    shadow.shadowOffset = NSSize(width: 0, height: -s * 0.01)
    shadow.shadowBlurRadius = s * 0.02
    for row in 0..<3 {
        for col in 0..<3 {
            // Rows count up from the bottom in AppKit; row 0 is the bottom row.
            let lit = row == 0 && col == 2
            let rect = NSRect(x: left + CGFloat(col) * (cell + gap), y: bottom + CGFloat(row) * (cell + gap),
                              width: cell, height: cell)
            NSGraphicsContext.saveGraphicsState()
            if lit { shadow.set() }
            NSColor(red: 1.0, green: 0.99, blue: 0.94, alpha: lit ? 1 : 0.47).setFill()
            NSBezierPath(roundedRect: rect, xRadius: cell * 0.18, yRadius: cell * 0.18).fill()
            NSGraphicsContext.restoreGraphicsState()
        }
    }

    NSGraphicsContext.restoreGraphicsState()
    return rep.representation(using: .png, properties: [:])!
}

if web {
    try render(192, .web).write(to: out.appendingPathComponent("icon-192.png"))
    try render(512, .web).write(to: out.appendingPathComponent("icon-512.png"))
    try render(512, .maskable).write(to: out.appendingPathComponent("maskable-512.png"))
} else {
    for size in [16, 32, 128, 256, 512] {
        try render(size).write(to: out.appendingPathComponent("icon_\(size)x\(size).png"))
        try render(size * 2).write(to: out.appendingPathComponent("icon_\(size)x\(size)@2x.png"))
    }
}
