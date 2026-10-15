// Copyright © 2026 Faux Fox, Inc.

import CoreImage
import Foundation
import MLX
import MLXLMCommon

/// Turns container media into pixels for the shared EmbeddingGemma 2 processors.
enum EmbeddingGemma2MediaProcessor {
    static let softTokenBudgets: Set<Int> = [70, 140, 280, 560, 1120]

    /// `[1, 3, H, W]` pixels resized to `budget` soft tokens.
    static func image(
        _ input: EmbeddingGemma2Image, processor: EmbeddingGemma2ImageProcessor, budget: Int
    ) throws -> MLXArray {
        try frames([input], processor: processor, budget: budget)
    }

    /// `[frames, 3, H, W]` pixels, every frame at the `budget` size of the first.
    static func frames(
        _ inputs: [EmbeddingGemma2Image], processor: EmbeddingGemma2ImageProcessor, budget: Int
    ) throws -> MLXArray {
        guard softTokenBudgets.contains(budget) else {
            throw EmbeddingGemma2Error.invalidConfiguration("Unsupported soft-token budget")
        }
        var size: (height: Int, width: Int)?
        var pixels: [MLXArray] = []
        pixels.reserveCapacity(inputs.count)
        for input in inputs {
            try Task.checkCancellation()
            let image = try EmbeddingGemma2ImageProcessor.prepared(ciImage(input))
            if size == nil {
                let target = processor.configuration.aspectPreservingTargetSize(
                    for: image.extent.size, budget: budget)
                size = (Int(target.height), Int(target.width))
            }
            guard let size, size.height > 0, size.width > 0 else {
                throw EmbeddingGemma2Error.invalidInput("Image rounds to an empty patch grid")
            }
            pixels.append(
                EmbeddingGemma2ImageProcessor.pixels(
                    image, height: size.height, width: size.width))
        }
        guard !pixels.isEmpty else {
            throw EmbeddingGemma2Error.invalidInput("Video has no frames")
        }
        let result = pixels.count == 1 ? pixels[0] : concatenated(pixels, axis: 0)
        try checkedEval(result)
        return result
    }

    /// `[frames, 3, H, W]` pixels of a video.
    ///
    /// Keyframes are used as given. Timestamped frames are sampled at the processor's rate.
    static func video(
        _ input: EmbeddingGemma2Video, images: EmbeddingGemma2ImageProcessor,
        video: EmbeddingGemma2VideoProcessor
    ) async throws -> MLXArray {
        switch input {
        case .keyframes(let frames, let softTokenBudget):
            guard (1 ... video.maximumFrames).contains(frames.count) else {
                throw EmbeddingGemma2Error.invalidInput(
                    "Video requires between 1 and \(video.maximumFrames) keyframes")
            }
            try checkTimes(frames)
            return try self.frames(
                frames.map(\.image), processor: images, budget: softTokenBudget)
        case .frames(let frames):
            guard let first = frames.first else {
                throw EmbeddingGemma2Error.invalidInput("Video has no frames")
            }
            try checkTimes(frames)
            // Use the frame at or immediately before each sample time.
            let interval = 1 / video.framesPerSecond
            let span = frames[frames.count - 1].time - first.time
            guard span.isFinite, span >= 0, span / interval < Double(Int.max) else {
                throw EmbeddingGemma2Error.invalidInput("Video duration is invalid")
            }
            let count = Int(span / interval) + 1
            var cursor = 0
            var selected: [EmbeddingGemma2Image] = []
            for index in uniformIndices(count: count, maximum: video.maximumFrames) {
                let time = first.time + Double(index) * interval
                while cursor + 1 < frames.count, frames[cursor + 1].time <= time {
                    cursor += 1
                }
                selected.append(frames[cursor].image)
            }
            return try self.frames(selected, processor: images, budget: video.budget)
        case .file(let url):
            guard url.isFileURL else {
                throw EmbeddingGemma2Error.invalidInput("Video requires a local file URL")
            }
            return try await video.frames(for: .url(url))
        }
    }

    private static func checkTimes(_ frames: [EmbeddingGemma2VideoFrame]) throws {
        var previous = -Double.infinity
        for frame in frames {
            guard frame.time.isFinite, frame.time >= 0, frame.time > previous else {
                throw EmbeddingGemma2Error.invalidInput(
                    "Video timestamps must be finite, nonnegative and strictly increasing")
            }
            previous = frame.time
        }
    }

    /// `np.linspace(0, count - 1, maximum, dtype=int)` when `count` exceeds `maximum`.
    private static func uniformIndices(count: Int, maximum: Int) -> [Int] {
        if count <= maximum { return Array(0 ..< count) }
        let span = count - 1
        let steps = max(maximum - 1, 1)
        return (0 ..< maximum).map { index in
            (span / steps) * index + (span % steps) * index / steps
        }
    }

    private static func ciImage(_ input: EmbeddingGemma2Image) throws -> CIImage {
        switch input {
        case .file(let url):
            guard url.isFileURL else {
                throw EmbeddingGemma2Error.invalidInput("Image requires a local file URL")
            }
            do {
                return try UserInput.Image.url(url).asCIImage()
            } catch {
                throw EmbeddingGemma2Error.invalidInput("Cannot decode image file")
            }
        case .rgb(let bytes, let width, let height):
            let (pixels, overflow) = width.multipliedReportingOverflow(by: height)
            guard width > 0, height > 0, !overflow, pixels <= Int(Int32.max) / 4,
                bytes.count == pixels * 3,
                let colorSpace = CGColorSpace(name: CGColorSpace.sRGB)
            else {
                throw EmbeddingGemma2Error.invalidInput(
                    "Image must contain exactly width * height * 3 RGB bytes")
            }
            var rgba = [UInt8](repeating: 255, count: pixels * 4)
            for pixel in 0 ..< pixels {
                rgba[pixel * 4] = bytes[pixel * 3]
                rgba[pixel * 4 + 1] = bytes[pixel * 3 + 1]
                rgba[pixel * 4 + 2] = bytes[pixel * 3 + 2]
            }
            return CIImage(
                bitmapData: Data(rgba), bytesPerRow: width * 4,
                size: CGSize(width: width, height: height), format: .RGBA8,
                colorSpace: colorSpace)
        }
    }
}
