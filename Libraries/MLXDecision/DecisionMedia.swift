// Copyright © 2026 Faux Fox, Inc.

import Foundation

#if canImport(CoreGraphics) && canImport(ImageIO)
import CoreGraphics
import ImageIO
#endif

#if canImport(AVFoundation)
@preconcurrency import AVFoundation
#endif

/// Packed 8-bit RGB, row-major.
struct DecisionRGBImage: Sendable, Equatable {
    let bytes: [UInt8]
    let width: Int
    let height: Int

    func crop(x: Int, y: Int, width: Int, height: Int) -> DecisionRGBImage {
        var result = [UInt8](repeating: 0, count: width * height * 3)
        for row in 0 ..< height {
            let source = ((y + row) * self.width + x) * 3
            result.replaceSubrange(
                row * width * 3 ..< (row + 1) * width * 3,
                with: bytes[source ..< source + width * 3])
        }
        return DecisionRGBImage(bytes: result, width: width, height: height)
    }
}

enum DecisionMedia {
    static let maximumPixels = 64 * 1024 * 1024

    static func rgb(_ image: DecisionImage) throws -> DecisionRGBImage {
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
            return DecisionRGBImage(bytes: bytes, width: width, height: height)
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
    private static func decode(_ source: CGImageSource) throws -> DecisionRGBImage {
        guard let decoded = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
            throw DecisionError.invalidInput("cannot decode the image")
        }
        guard decoded.width * decoded.height <= maximumPixels else {
            throw DecisionError.invalidInput("the image has more than \(maximumPixels) pixels")
        }
        let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [String: Any]
        let orientation =
            (properties?[kCGImagePropertyOrientation as String] as? NSNumber)?.intValue ?? 1
        if orientation != 1 {
            let options: [String: Any] = [
                kCGImageSourceCreateThumbnailFromImageAlways as String: true,
                kCGImageSourceCreateThumbnailWithTransform as String: true,
                kCGImageSourceThumbnailMaxPixelSize as String: max(decoded.width, decoded.height),
            ]
            guard
                let oriented = CGImageSourceCreateThumbnailAtIndex(
                    source, 0, options as CFDictionary)
            else {
                throw DecisionError.invalidInput("cannot apply the image orientation")
            }
            return try render(oriented)
        }
        return try render(decoded)
    }

    private static func render(_ image: CGImage) throws -> DecisionRGBImage {
        let width = image.width
        let height = image.height
        var rgba = [UInt8](repeating: 0, count: width * height * 4)
        let rendered = rgba.withUnsafeMutableBytes { bytes -> Bool in
            guard let space = CGColorSpace(name: CGColorSpace.sRGB),
                let context = CGContext(
                    data: bytes.baseAddress, width: width, height: height, bitsPerComponent: 8,
                    bytesPerRow: width * 4, space: space,
                    bitmapInfo: CGBitmapInfo.byteOrder32Big.rawValue
                        | CGImageAlphaInfo.noneSkipLast.rawValue)
            else { return false }
            context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
            return true
        }
        guard rendered else { throw DecisionError.invalidInput("cannot render the image as RGB") }
        var bytes = [UInt8](repeating: 0, count: width * height * 3)
        for pixel in 0 ..< width * height {
            bytes[pixel * 3] = rgba[pixel * 4]
            bytes[pixel * 3 + 1] = rgba[pixel * 4 + 1]
            bytes[pixel * 3 + 2] = rgba[pixel * 4 + 2]
        }
        return DecisionRGBImage(bytes: bytes, width: width, height: height)
    }
    #endif

    /// Mono 16 kHz float samples, at most `maximumSeconds` long.
    static func waveform(_ audio: DecisionAudio, maximumSeconds: Int) async throws -> [Float] {
        let limit = maximumSeconds * 16_000
        switch audio {
        case .pcm(let samples, let rate, let channels):
            return try resample(
                try mix(samples, channels: channels), sampleRate: rate, limit: limit)
        case .pcm16(let samples, let rate, let channels):
            // Scale after mixing, as the reference divides the mono int16 clip by 32768.
            let floats = samples.map { Float($0) }
            let mono = try mix(floats, channels: channels).map { $0 / 32768 }
            return try resample(mono, sampleRate: rate, limit: limit)
        case .file(let url):
            return try await decode(url, limit: limit)
        }
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
