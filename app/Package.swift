// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "GPUKernels",
    platforms: [.macOS(.v14)],
    targets: [
        .executableTarget(
            name: "GPUKernels",
            resources: [
                .copy("Resources/course.json"),
                .copy("Resources/lesson.js"),
                .copy("Resources/lesson.css"),
            ]
        )
    ]
)
