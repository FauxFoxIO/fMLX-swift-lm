// Copyright © 2026 Faux Fox, Inc.

import Foundation
import MLX

#if canImport(CoreGraphics) && canImport(ImageIO)
import CoreGraphics
import ImageIO
#endif

#if canImport(AVFoundation)
@preconcurrency import AVFoundation
#endif

/// An 8-bit RGB image as an MLX array [height, width, 3].
struct DecisionRGBImage {
    let pixels: MLXArray
    var width: Int { pixels.dim(1) }
    var height: Int { pixels.dim(0) }

    init(pixels: MLXArray) {
        self.pixels = pixels
    }

    /// Packed RGB, row-major.
    init(bytes: [UInt8], width: Int, height: Int) {
        pixels = MLXArray(bytes, [height, width, 3])
    }

    var bytes: [UInt8] { pixels.asArray(UInt8.self) }

    func crop(x: Int, y: Int, width: Int, height: Int) -> DecisionRGBImage {
        DecisionRGBImage(pixels: pixels[y ..< (y + height), x ..< (x + width)])
    }
}

/// A decoded image as plain values, so it can cross into a model's serialized domain.
/// Pixel values are the file's own samples, without color management, as PIL reads them.
struct DecodedImage: Sendable {
    let bytes: [UInt8]
    let width: Int
    let height: Int
    let bytesPerRow: Int
    let bytesPerPixel: Int
    /// Byte offsets of red, green, and blue within a pixel.
    let channels: [Int]
    /// Byte offset of a premultiplied alpha value, when the samples carry one.
    let premultipliedAlpha: Int?
    /// EXIF orientation, 1 through 8.
    let orientation: Int

    /// Upright RGB [height, width, 3] as uint8.
    func pixels() -> MLXArray {
        let rows = MLXArray(bytes, [height, bytesPerRow])[0..., ..<(width * bytesPerPixel)]
        let pixels = rows.reshaped(height, width, bytesPerPixel)
        var rgb = take(pixels, MLXArray(channels.map(Int32.init)), axis: 2)
        if let premultipliedAlpha {
            let alpha = pixels[0..., 0..., premultipliedAlpha ..< (premultipliedAlpha + 1)]
                .asType(.float32)
            let straight = round(rgb.asType(.float32) * 255 / maximum(alpha, MLXArray(Float(1))))
            rgb = clip(straight, min: 0, max: 255).asType(.uint8)
        }
        return Self.upright(rgb, orientation: orientation)
    }

    /// PIL's `exif_transpose` as array flips and transposes.
    static func upright(_ x: MLXArray, orientation: Int) -> MLXArray {
        func flipRows(_ a: MLXArray) -> MLXArray { a[.stride(by: -1)] }
        func flipColumns(_ a: MLXArray) -> MLXArray { a[0..., .stride(by: -1)] }
        func transpose(_ a: MLXArray) -> MLXArray { a.transposed(1, 0, 2) }
        switch orientation {
        case 2: return flipColumns(x)
        case 3: return flipRows(flipColumns(x))
        case 4: return flipRows(x)
        case 5: return transpose(x)
        case 6: return flipColumns(transpose(x))
        case 7: return flipRows(flipColumns(transpose(x)))
        case 8: return flipRows(transpose(x))
        default: return x
        }
    }
}

enum DecisionMedia {
    static let maximumPixels = 64 * 1024 * 1024

    static func rgb(_ image: DecisionImage) throws -> DecisionRGBImage {
        DecisionRGBImage(pixels: try decode(image).pixels())
    }

    static func decode(_ image: DecisionImage) throws -> DecodedImage {
        switch image {
        case .rgb(let bytes, let width, let height):
            let (pixels, overflow) = width.multipliedReportingOverflow(by: height)
            guard width > 0, height > 0, !overflow, pixels <= maximumPixels,
                bytes.count == pixels * 3
            else {
                throw DecisionError.invalidInput(
                    "an RGB image needs width * height * 3 bytes and at most \(maximumPixels) pixels"
                )
            }
            return DecodedImage(
                bytes: bytes, width: width, height: height, bytesPerRow: width * 3,
                bytesPerPixel: 3, channels: [0, 1, 2], premultipliedAlpha: nil, orientation: 1)
        case .file(let url):
            guard url.isFileURL else {
                throw DecisionError.invalidInput("an image file needs a local file URL")
            }
            #if canImport(CoreGraphics) && canImport(ImageIO)
            guard let source = CGImageSourceCreateWithURL(url as CFURL, nil) else {
                throw DecisionError.invalidInput("cannot open the image file")
            }
            return try decode(source)
            #else
            throw DecisionError.unsupportedInput("image decoding is unavailable on this platform")
            #endif
        case .encoded(let data):
            #if canImport(CoreGraphics) && canImport(ImageIO)
            guard let source = CGImageSourceCreateWithData(data as CFData, nil) else {
                throw DecisionError.invalidInput("cannot read the encoded image")
            }
            return try decode(source)
            #else
            throw DecisionError.unsupportedInput("image decoding is unavailable on this platform")
            #endif
        }
    }

    #if canImport(CoreGraphics) && canImport(ImageIO)
    private static func decode(_ source: CGImageSource) throws -> DecodedImage {
        guard let image = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
            throw DecisionError.invalidInput("cannot decode the image")
        }
        guard image.width > 0, image.height > 0, image.width * image.height <= maximumPixels else {
            throw DecisionError.invalidInput(
                "the image is empty or has more than \(maximumPixels) pixels")
        }
        let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [String: Any]
        let orientation =
            (properties?[kCGImagePropertyOrientation as String] as? NSNumber)?.intValue ?? 1
        guard (1 ... 8).contains(orientation) else {
            throw DecisionError.invalidInput("invalid EXIF orientation \(orientation)")
        }
        return try samples(image, orientation: orientation)
            ?? render(image, orientation: orientation)
    }

    /// The decoded 8-bit samples of an RGB or gray image with straight or no alpha.
    private static func samples(_ image: CGImage, orientation: Int) throws -> DecodedImage? {
        guard image.bitsPerComponent == 8, let model = image.colorSpace?.model,
            model == .rgb || model == .monochrome,
            let data = image.dataProvider?.data
        else { return nil }
        let bytesPerPixel = image.bitsPerPixel / 8
        let components = model == .rgb ? 3 : 1
        let alpha = image.alphaInfo
        let hasAlphaSlot = alpha != .none
        guard bytesPerPixel == components + (hasAlphaSlot ? 1 : 0),
            alpha != .premultipliedFirst, alpha != .premultipliedLast, alpha != .alphaOnly,
            image.bytesPerRow >= image.width * bytesPerPixel,
            CFDataGetLength(data) >= image.bytesPerRow * (image.height - 1) + image.width
                * bytesPerPixel
        else { return nil }
        let alphaFirst = alpha == .first || alpha == .noneSkipFirst
        var color = model == .rgb ? [0, 1, 2] : [0, 0, 0]
        if hasAlphaSlot, alphaFirst { color = color.map { $0 + 1 } }
        switch image.byteOrderInfo {
        case .orderDefault, .order32Big, .order16Big:
            break
        case .order32Little where bytesPerPixel == 4:
            color = color.map { 3 - $0 }
        default:
            return nil
        }
        var bytes = [UInt8](repeating: 0, count: CFDataGetLength(data))
        CFDataGetBytes(data, CFRange(location: 0, length: bytes.count), &bytes)
        return DecodedImage(
            bytes: bytes, width: image.width, height: image.height, bytesPerRow: image.bytesPerRow,
            bytesPerPixel: bytesPerPixel, channels: color, premultipliedAlpha: nil,
            orientation: orientation)
    }

    /// Other layouts (16-bit, indexed, CMYK, premultiplied) drawn into 8-bit RGBA.
    private static func render(_ image: CGImage, orientation: Int) throws -> DecodedImage {
        let width = image.width
        let height = image.height
        var rgba = [UInt8](repeating: 0, count: width * height * 4)
        let space =
            image.colorSpace.flatMap { $0.model == .rgb ? $0 : nil }
            ?? CGColorSpace(name: CGColorSpace.sRGB)!
        let rendered = rgba.withUnsafeMutableBytes { bytes -> Bool in
            guard
                let context = CGContext(
                    data: bytes.baseAddress, width: width, height: height, bitsPerComponent: 8,
                    bytesPerRow: width * 4, space: space,
                    bitmapInfo: CGBitmapInfo.byteOrder32Big.rawValue
                        | CGImageAlphaInfo.premultipliedLast.rawValue)
            else { return false }
            context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
            return true
        }
        guard rendered else { throw DecisionError.invalidInput("cannot render the image as RGB") }
        return DecodedImage(
            bytes: rgba, width: width, height: height, bytesPerRow: width * 4, bytesPerPixel: 4,
            channels: [0, 1, 2], premultipliedAlpha: image.alphaInfo == .none ? nil : 3,
            orientation: orientation)
    }
    #endif

    /// Mono 16 kHz float samples, at most `maximumSeconds` long.
    static func waveform(_ audio: DecisionAudio, maximumSeconds: Int) async throws -> [Float] {
        let limit = maximumSeconds * 16_000
        switch audio {
        case .pcm(let samples, let rate, let channels):
            let kept = try frames(samples, sampleRate: rate, channels: channels, limit: limit)
            return try resample(
                try mix(Array(kept), channels: channels), sampleRate: rate, limit: limit)
        case .pcm16(let samples, let rate, let channels):
            // Scale after mixing, as the reference divides the mono int16 clip by 32768.
            let kept = try frames(samples, sampleRate: rate, channels: channels, limit: limit)
            let mono = try mix(kept.map { Float($0) }, channels: channels).map { $0 / 32768 }
            return try resample(mono, sampleRate: rate, limit: limit)
        case .file(let url):
            return try await decode(url, limit: limit)
        }
    }

    /// The leading frames that cover `limit` output samples, plus resampler history.
    private static func frames<T>(_ samples: [T], sampleRate: Int, channels: Int, limit: Int) throws
        -> ArraySlice<T>
    {
        guard sampleRate > 0, channels > 0, !samples.isEmpty, samples.count % channels == 0 else {
            throw DecisionError.invalidInput(
                "audio needs a positive rate, complete frames, and at least one sample")
        }
        let keep =
            sampleRate == 16_000
            ? limit
            : Int((Double(limit) * Double(sampleRate) / 16_000).rounded(.up)) + sampleRate / 10
        return samples.prefix(keep * channels)
    }

    private static func mix(_ samples: [Float], channels: Int) throws -> [Float] {
        guard channels > 0, samples.count % channels == 0, !samples.isEmpty else {
            throw DecisionError.invalidInput("audio needs complete frames and at least one sample")
        }
        guard samples.allSatisfy(\.isFinite) else {
            throw DecisionError.invalidInput("audio samples must be finite")
        }
        if channels == 1 { return samples }
        return stride(from: 0, to: samples.count, by: channels).map { start in
            samples[start ..< start + channels].reduce(0, +) / Float(channels)
        }
    }

    private static func resample(_ samples: [Float], sampleRate: Int, limit: Int) throws
        -> [Float]
    {
        guard sampleRate > 0 else { throw DecisionError.invalidInput("invalid sample rate") }
        if sampleRate == 16_000 { return Array(samples.prefix(limit)) }
        // Keep only the source span that covers the limit, plus converter history.
        let keep = Int((Double(limit) * Double(sampleRate) / 16_000).rounded(.up)) + sampleRate / 10
        let source = Array(samples.prefix(keep))
        #if canImport(AVFoundation)
        guard
            let sourceFormat = AVAudioFormat(
                commonFormat: .pcmFormatFloat32, sampleRate: Double(sampleRate), channels: 1,
                interleaved: false),
            let targetFormat = AVAudioFormat(
                commonFormat: .pcmFormatFloat32, sampleRate: 16_000, channels: 1,
                interleaved: false),
            let converter = AVAudioConverter(from: sourceFormat, to: targetFormat),
            let input = AVAudioPCMBuffer(pcmFormat: sourceFormat, frameCapacity: 4096),
            let output = AVAudioPCMBuffer(pcmFormat: targetFormat, frameCapacity: 4096)
        else {
            throw DecisionError.unsupportedInput("cannot create an audio sample-rate converter")
        }
        converter.sampleRateConverterQuality = AVAudioQuality.max.rawValue
        converter.primeMethod = .normal
        var cursor = 0
        var result = [Float]()
        while result.count < limit {
            let previous = cursor
            var error: NSError?
            output.frameLength = 0
            let status = converter.convert(to: output, error: &error) { requested, status in
                guard cursor < source.count else {
                    status.pointee = .endOfStream
                    return nil
                }
                let count = min(Int(requested), Int(input.frameCapacity), source.count - cursor)
                input.frameLength = AVAudioFrameCount(count)
                source.withUnsafeBufferPointer { buffer in
                    input.floatChannelData![0].update(
                        from: buffer.baseAddress! + cursor, count: count)
                }
                cursor += count
                status.pointee = .haveData
                return input
            }
            if let error { throw error }
            if status == .error {
                throw DecisionError.invalidInput("audio sample-rate conversion failed")
            }
            let count = Int(output.frameLength)
            if count > 0 {
                result.append(
                    contentsOf: UnsafeBufferPointer(
                        start: output.floatChannelData![0], count: count))
            }
            if status == .endOfStream { break }
            guard count > 0 || cursor > previous else {
                throw DecisionError.invalidInput("the audio converter stopped early")
            }
        }
        return Array(result.prefix(limit))
        #else
        throw DecisionError.unsupportedInput("audio resampling is unavailable on this platform")
        #endif
    }

    private static func decode(_ url: URL, limit: Int) async throws -> [Float] {
        guard url.isFileURL else {
            throw DecisionError.invalidInput("an audio file needs a local file URL")
        }
        #if canImport(AVFoundation)
        let asset = AVURLAsset(url: url)
        guard let track = try await asset.loadTracks(withMediaType: .audio).first else {
            throw DecisionError.invalidInput("the file has no audio track")
        }
        let descriptions = try await track.load(.formatDescriptions)
        guard let description = descriptions.first,
            let format = CMAudioFormatDescriptionGetStreamBasicDescription(description)?.pointee,
            format.mSampleRate.isFinite, format.mSampleRate > 0,
            format.mSampleRate.rounded() == format.mSampleRate, format.mChannelsPerFrame > 0
        else {
            throw DecisionError.invalidInput("the audio track has an unsupported format")
        }
        let sampleRate = Int(format.mSampleRate)
        let channels = Int(format.mChannelsPerFrame)
        let settings: [String: Any] = [
            AVFormatIDKey: kAudioFormatLinearPCM,
            AVSampleRateKey: format.mSampleRate,
            AVNumberOfChannelsKey: channels,
            AVLinearPCMBitDepthKey: 32,
            AVLinearPCMIsFloatKey: true,
            AVLinearPCMIsBigEndianKey: false,
            AVLinearPCMIsNonInterleaved: false,
        ]
        let reader = try AVAssetReader(asset: asset)
        let output = AVAssetReaderTrackOutput(track: track, outputSettings: settings)
        output.alwaysCopiesSampleData = false
        guard reader.canAdd(output) else {
            throw DecisionError.invalidInput("cannot decode the audio track as PCM")
        }
        reader.add(output)
        guard reader.startReading() else {
            throw reader.error ?? DecisionError.invalidInput("cannot start the audio decoder")
        }
        defer { reader.cancelReading() }
        let sourceLimit =
            Int((Double(limit) * Double(sampleRate) / 16_000).rounded(.up)) + sampleRate / 10
        var mono = [Float]()
        while mono.count < sourceLimit, let buffer = output.copyNextSampleBuffer() {
            try Task.checkCancellation()
            guard let block = CMSampleBufferGetDataBuffer(buffer) else {
                throw DecisionError.invalidInput("the audio decoder returned no PCM data")
            }
            let bytes = CMBlockBufferGetDataLength(block)
            guard bytes > 0, bytes % (MemoryLayout<Float>.size * channels) == 0 else {
                throw DecisionError.invalidInput("the audio decoder returned a partial frame")
            }
            var chunk = [Float](repeating: 0, count: bytes / MemoryLayout<Float>.size)
            let status = chunk.withUnsafeMutableBytes { destination in
                CMBlockBufferCopyDataBytes(
                    block, atOffset: 0, dataLength: bytes, destination: destination.baseAddress!)
            }
            guard status == kCMBlockBufferNoErr else {
                throw DecisionError.invalidInput("cannot copy decoded PCM samples")
            }
            mono.append(contentsOf: try mix(chunk, channels: channels))
        }
        if mono.count < sourceLimit, reader.status != .completed {
            throw reader.error ?? DecisionError.invalidInput("the audio decoder did not complete")
        }
        guard !mono.isEmpty else {
            throw DecisionError.invalidInput("the audio file has no samples")
        }
        return try resample(mono, sampleRate: sampleRate, limit: limit)
        #else
        throw DecisionError.unsupportedInput("audio decoding is unavailable on this platform")
        #endif
    }
}
