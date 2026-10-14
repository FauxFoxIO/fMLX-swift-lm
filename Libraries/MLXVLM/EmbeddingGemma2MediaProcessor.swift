// Copyright © 2026 Faux Fox, Inc.
// Portions © 2026 Google LLC and Hugging Face (Apache-2.0).
// Bicubic quantization adapted from PyTorch, copyright its contributors (BSD).
// See LICENSES/EmbeddingGemma2-NOTICE.md for source attribution and license texts.

import Foundation
import MLX

#if canImport(AVFoundation)
@preconcurrency import AVFoundation
#endif
#if canImport(CoreGraphics) && canImport(ImageIO)
import CoreGraphics
import ImageIO
#endif

enum EmbeddingGemma2MediaProcessor {
    static func image(_ input: EmbeddingGemma2Image, softTokenBudget: Int) throws
        -> EmbeddingGemma2PreparedImage
    {
        guard [70, 140, 280, 560, 1120].contains(softTokenBudget) else {
            throw EmbeddingGemma2Error.invalidConfiguration("Unsupported image soft-token budget")
        }
        let source = try rgb(input)
        let size = try targetSize(
            width: source.width, height: source.height, budget: softTokenBudget)
        let pixels = try resize(source, width: size.width, height: size.height)
        let gridWidth = size.width / 16
        let gridHeight = size.height / 16
        let patchCount = gridWidth * gridHeight
        let paddedCount = softTokenBudget * 9
        var patches = [Float](repeating: 0, count: paddedCount * 768)
        var positions = [Int32](repeating: -1, count: paddedCount * 2)
        for y in 0 ..< gridHeight {
            for x in 0 ..< gridWidth {
                let patch = y * gridWidth + x
                positions[patch * 2] = Int32(x)
                positions[patch * 2 + 1] = Int32(y)
                for py in 0 ..< 16 {
                    for px in 0 ..< 16 {
                        let sourceOffset = ((y * 16 + py) * size.width + x * 16 + px) * 3
                        let destination = patch * 768 + (py * 16 + px) * 3
                        for channel in 0 ..< 3 {
                            patches[destination + channel] =
                                Float(pixels[sourceOffset + channel]) * Float(1.0 / 255)
                        }
                    }
                }
            }
        }
        return EmbeddingGemma2PreparedImage(
            patches: MLXArray(patches, [1, paddedCount, 768]),
            positions: MLXArray(positions, [1, paddedCount, 2]),
            tokenCount: patchCount / 9)
    }

    static func audio(_ input: EmbeddingGemma2Audio, maximumTokens: Int) async throws
        -> EmbeddingGemma2PreparedAudio
    {
        try await EmbeddingGemma2AudioProcessor.process(input, maximumTokens: maximumTokens)
    }

    static func video(_ input: EmbeddingGemma2Video) async throws
        -> [EmbeddingGemma2PreparedImage]
    {
        switch input {
        case .keyframes(let frames, let softTokenBudget):
            guard (1 ... 32).contains(frames.count) else {
                throw EmbeddingGemma2Error.invalidInput("Video requires between 1 and 32 keyframes")
            }
            guard [70, 140, 280, 560, 1120].contains(softTokenBudget) else {
                throw EmbeddingGemma2Error.invalidConfiguration(
                    "Unsupported video soft-token budget")
            }
            var previous = -Double.infinity
            for frame in frames {
                guard frame.time.isFinite, frame.time >= 0, frame.time > previous else {
                    throw EmbeddingGemma2Error.invalidInput(
                        "Keyframe timestamps must be finite, nonnegative and strictly increasing")
                }
                previous = frame.time
            }
            var result: [EmbeddingGemma2PreparedImage] = []
            result.reserveCapacity(frames.count)
            for frame in frames {
                try Task.checkCancellation()
                result.append(try image(frame.image, softTokenBudget: softTokenBudget))
            }
            return result
        case .frames(let frames):
            guard let first = frames.first else {
                throw EmbeddingGemma2Error.invalidInput("Video has no frames")
            }
            var previous = -Double.infinity
            for frame in frames {
                guard frame.time.isFinite, frame.time > previous else {
                    throw EmbeddingGemma2Error.invalidInput(
                        "Video timestamps must be finite and strictly increasing")
                }
                previous = frame.time
            }
            // Use the frame at or immediately before each second, as floor(i * sourceFPS) does.
            let span = frames[frames.count - 1].time - first.time
            let count = try sampledFrameCount(duration: span, includeEndpoint: true)
            var cursor = 0
            var result: [EmbeddingGemma2PreparedImage] = []
            for second in uniformIndices(count: count) {
                try Task.checkCancellation()
                let time = first.time + Double(second)
                while cursor + 1 < frames.count, frames[cursor + 1].time <= time {
                    cursor += 1
                }
                result.append(try image(frames[cursor].image, softTokenBudget: 140))
            }
            return result
        case .file(let url):
            guard url.isFileURL else {
                throw EmbeddingGemma2Error.invalidInput("Video requires a local file URL")
            }
            #if canImport(AVFoundation) && canImport(CoreGraphics) && canImport(ImageIO)
            let asset = AVURLAsset(url: url)
            guard let track = try await asset.loadTracks(withMediaType: .video).first else {
                throw EmbeddingGemma2Error.invalidInput("File has no video track")
            }
            let range = try await track.load(.timeRange)
            let fps = Double(try await track.load(.nominalFrameRate))
            let timescale = try await track.load(.naturalTimeScale)
            let count = try sampledFrameCount(duration: range.duration.seconds)
            let generator = AVAssetImageGenerator(asset: asset)
            generator.appliesPreferredTrackTransform = true
            // Allow one media tick for nominal-FPS rounding without accepting a keyframe seek.
            let tolerance = CMTime(value: 1, timescale: max(1, timescale))
            generator.requestedTimeToleranceBefore = tolerance
            generator.requestedTimeToleranceAfter = tolerance
            defer { generator.cancelAllCGImageGeneration() }
            var result: [EmbeddingGemma2PreparedImage] = []
            for second in uniformIndices(count: count) {
                try Task.checkCancellation()
                let offset =
                    fps.isFinite && fps > 0 ? floor(Double(second) * fps) / fps : Double(second)
                let time = range.start + CMTime(seconds: offset, preferredTimescale: 600_000)
                let frame = try await generator.image(at: time).image
                let pixels = try rgb(frame)
                result.append(
                    try image(
                        .rgb(bytes: pixels.bytes, width: pixels.width, height: pixels.height),
                        softTokenBudget: 140))
            }
            return result
            #else
            throw EmbeddingGemma2Error.invalidInput(
                "Native video decoding is unavailable on this platform")
            #endif
        }
    }

    private struct RGB {
        let bytes: [UInt8]
        let width: Int
        let height: Int
    }

    private static func rgb(_ input: EmbeddingGemma2Image) throws -> RGB {
        switch input {
        case .rgb(let bytes, let width, let height):
            guard width > 0, height > 0,
                try elementCount(width, height, 3) == bytes.count
            else {
                throw EmbeddingGemma2Error.invalidInput(
                    "Image must contain exactly width * height * 3 RGB bytes")
            }
            return RGB(bytes: bytes, width: width, height: height)
        case .file(let url):
            guard url.isFileURL else {
                throw EmbeddingGemma2Error.invalidInput("Image requires a local file URL")
            }
            #if canImport(CoreGraphics) && canImport(ImageIO)
            guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
                let decoded = CGImageSourceCreateImageAtIndex(source, 0, nil)
            else {
                throw EmbeddingGemma2Error.invalidInput("Cannot decode image file")
            }
            let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [String: Any]
            let orientation =
                (properties?[kCGImagePropertyOrientation as String] as? NSNumber)?.intValue ?? 1
            if orientation != 1 {
                let options: [String: Any] = [
                    kCGImageSourceCreateThumbnailFromImageAlways as String: true,
                    kCGImageSourceCreateThumbnailWithTransform as String: true,
                    kCGImageSourceThumbnailMaxPixelSize as String: max(
                        decoded.width, decoded.height),
                ]
                guard
                    let oriented = CGImageSourceCreateThumbnailAtIndex(
                        source, 0, options as CFDictionary)
                else {
                    throw EmbeddingGemma2Error.invalidInput("Cannot apply image orientation")
                }
                return try rgb(oriented)
            }
            return try rgb(decoded)
            #else
            throw EmbeddingGemma2Error.invalidInput(
                "Native image decoding is unavailable on this platform")
            #endif
        }
    }

    #if canImport(CoreGraphics) && canImport(ImageIO)
    private static func rgb(_ image: CGImage) throws -> RGB {
        let width = image.width
        let height = image.height
        var rgba = [UInt8](repeating: 0, count: try elementCount(width, height, 4))
        let rendered = rgba.withUnsafeMutableBytes { bytes -> Bool in
            guard let colorSpace = CGColorSpace(name: CGColorSpace.sRGB),
                let context = CGContext(
                    data: bytes.baseAddress, width: width, height: height, bitsPerComponent: 8,
                    bytesPerRow: width * 4, space: colorSpace,
                    bitmapInfo: CGBitmapInfo.byteOrder32Big.rawValue
                        | CGImageAlphaInfo.noneSkipLast.rawValue)
            else { return false }
            context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
            return true
        }
        guard rendered else {
            throw EmbeddingGemma2Error.invalidInput("Cannot render decoded image as RGB")
        }
        var bytes = [UInt8](repeating: 0, count: try elementCount(width, height, 3))
        for pixel in 0 ..< width * height {
            for channel in 0 ..< 3 { bytes[pixel * 3 + channel] = rgba[pixel * 4 + channel] }
        }
        return RGB(bytes: bytes, width: width, height: height)
    }
    #endif

    private static func elementCount(_ width: Int, _ height: Int, _ channels: Int) throws -> Int {
        let (pixels, firstOverflow) = width.multipliedReportingOverflow(by: height)
        let (count, secondOverflow) = pixels.multipliedReportingOverflow(by: channels)
        guard width > 0, height > 0, !firstOverflow, !secondOverflow, count <= Int(Int32.max) else {
            throw EmbeddingGemma2Error.invalidInput(
                "Image dimensions exceed the native array range")
        }
        return count
    }

    private static func targetSize(width: Int, height: Int, budget: Int) throws
        -> (width: Int, height: Int)
    {
        let targetPixels = budget * 9 * 16 * 16
        let factor = sqrt(Double(targetPixels) / (Double(width) * Double(height)))
        var h = Int(floor(factor * Double(height) / 48)) * 48
        var w = Int(floor(factor * Double(width) / 48)) * 48
        if h == 0 && w == 0 {
            throw EmbeddingGemma2Error.invalidInput("Image rounds to an empty patch grid")
        }
        if h == 0 {
            h = 48
            w = min(width / height, budget) * 48
        } else if w == 0 {
            w = 48
            h = min(height / width, budget) * 48
        }
        guard w > 0, h > 0, w * h <= targetPixels else {
            throw EmbeddingGemma2Error.invalidInput("Resized image exceeds its patch budget")
        }
        return (w, h)
    }

    private struct AxisFilter {
        let starts: [Int]
        let weights: [[Int64]]
        let precision: Int
    }

    // Torchvision's uint8 antialiased bicubic path uses normalized Keys (-0.5) weights.
    private static func axisFilter(input: Int, output: Int) -> AxisFilter {
        let scale = Double(input) / Double(output)
        let support = 2 * max(1, scale)
        let inverseScale = 1 / max(1, scale)
        var starts: [Int] = []
        var allWeights: [[Double]] = []
        var maximumWeight = 0.0
        for index in 0 ..< output {
            let center = scale * (Double(index) + 0.5)
            let start = max(0, Int(center - support + 0.5))
            let end = min(input, Int(center + support + 0.5))
            var weights = (start ..< end).map { position -> Double in
                let x = abs((Double(position) - center + 0.5) * inverseScale)
                if x < 1 { return ((1.5 * x - 2.5) * x) * x + 1 }
                if x < 2 { return ((-0.5 * x + 2.5) * x - 4) * x + 2 }
                return 0
            }
            let total = weights.reduce(0, +)
            if total != 0 { weights = weights.map { $0 / total } }
            maximumWeight = max(maximumWeight, weights.max() ?? 0)
            starts.append(start)
            allWeights.append(weights)
        }
        var precision = 0
        while precision < 22 {
            if Int(0.5 + maximumWeight * Double(1 << (precision + 1))) >= 32768 { break }
            precision += 1
        }
        let factor = Double(1 << precision)
        let weights = allWeights.map { row in
            row.map { value -> Int64 in
                let scaled = value * factor
                return Int64(scaled < 0 ? scaled - 0.5 : scaled + 0.5)
            }
        }
        return AxisFilter(starts: starts, weights: weights, precision: precision)
    }

    private static func resize(_ source: RGB, width: Int, height: Int) throws -> [UInt8] {
        if source.width == width && source.height == height { return source.bytes }
        let horizontal = axisFilter(input: source.width, output: width)
        let vertical = axisFilter(input: source.height, output: height)
        var intermediate = [UInt8](repeating: 0, count: try elementCount(width, source.height, 3))
        for y in 0 ..< source.height {
            for x in 0 ..< width {
                for channel in 0 ..< 3 {
                    var value: Int64 = 1 << (horizontal.precision - 1)
                    for (tap, weight) in horizontal.weights[x].enumerated() {
                        value +=
                            Int64(
                                source.bytes[
                                    (y * source.width + horizontal.starts[x] + tap) * 3 + channel])
                            * weight
                    }
                    intermediate[(y * width + x) * 3 + channel] = UInt8(
                        clamping: value >> horizontal.precision)
                }
            }
        }
        var result = [UInt8](repeating: 0, count: try elementCount(width, height, 3))
        for y in 0 ..< height {
            for x in 0 ..< width {
                for channel in 0 ..< 3 {
                    var value: Int64 = 1 << (vertical.precision - 1)
                    for (tap, weight) in vertical.weights[y].enumerated() {
                        value +=
                            Int64(
                                intermediate[((vertical.starts[y] + tap) * width + x) * 3 + channel]
                            ) * weight
                    }
                    result[(y * width + x) * 3 + channel] = UInt8(
                        clamping: value >> vertical.precision)
                }
            }
        }
        return result
    }

    private static func sampledFrameCount(duration: Double, includeEndpoint: Bool = false) throws
        -> Int
    {
        guard duration.isFinite, duration >= 0, duration < Double(Int.max) else {
            throw EmbeddingGemma2Error.invalidInput("Video duration is invalid")
        }
        return max(1, Int(duration) + (includeEndpoint ? 1 : 0))
    }

    private static func uniformIndices(count: Int) -> [Int] {
        if count <= 32 { return Array(0 ..< count) }
        // Integer linspace: include both endpoints, matching np.linspace(..., dtype=int).
        return (0 ..< 32).map { index in
            let span = count - 1
            return (span / 31) * index + (span % 31) * index / 31
        }
    }
}
