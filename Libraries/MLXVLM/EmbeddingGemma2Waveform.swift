// Copyright © 2026 Faux Fox, Inc.

import Foundation

#if canImport(AVFoundation)
@preconcurrency import AVFoundation
#endif

/// Decodes and resamples audio to mono samples for `EmbeddingGemma2AudioProcessor`.
/// Audio longer than the processor's 30-second window throws instead of being cut.
enum EmbeddingGemma2Waveform {
    static let sampleRate = 16_000

    static func samples(_ input: EmbeddingGemma2Audio) async throws -> [Float] {
        try Task.checkCancellation()
        let waveform: [Float]
        switch input {
        case .pcm(let samples, let sampleRate, let channels):
            guard sampleRate > 0, channels > 0, !samples.isEmpty,
                samples.count % channels == 0
            else {
                throw EmbeddingGemma2Error.invalidInput(
                    "PCM requires complete frames, a positive rate, and channels")
            }
            try checkLength(try resampledCount(samples.count / channels, sampleRate: sampleRate))
            let mono = try mix(samples, channels: channels)
            waveform = try resample(mono, sampleRate: sampleRate)
        case .file(let url):
            waveform = try await decode(url)
        }
        try Task.checkCancellation()
        try checkLength(waveform.count)
        guard waveform.allSatisfy(\.isFinite) else {
            throw EmbeddingGemma2Error.invalidInput("Resampled PCM samples must be finite")
        }
        return waveform
    }

    private static func checkLength(_ sampleCount: Int) throws {
        guard sampleCount <= EmbeddingGemma2AudioProcessor.maximumSampleCount else {
            throw EmbeddingGemma2Error.invalidInput(
                "Audio exceeds 30 seconds. Divide it into contiguous units.")
        }
    }

    private static func resampledCount(_ count: Int, sampleRate: Int) throws -> Int {
        guard count >= 0, sampleRate > 0 else {
            throw EmbeddingGemma2Error.invalidInput("Invalid audio sample count or rate")
        }
        let (scaled, overflow) = count.multipliedReportingOverflow(by: Self.sampleRate)
        guard !overflow else {
            throw EmbeddingGemma2Error.invalidInput(
                "Resampled audio length overflows the native array range")
        }
        return scaled / sampleRate + (scaled % sampleRate == 0 ? 0 : 1)
    }

    private static func mix(_ samples: [Float], channels: Int) throws -> [Float] {
        var mono = [Float](repeating: 0, count: samples.count / channels)
        for frame in mono.indices {
            if frame % 4096 == 0 { try Task.checkCancellation() }
            var total = 0.0
            for channel in 0 ..< channels {
                let value = samples[frame * channels + channel]
                guard value.isFinite else {
                    throw EmbeddingGemma2Error.invalidInput("PCM samples must be finite")
                }
                total += Double(value)
            }
            mono[frame] = Float(total / Double(channels))
        }
        return mono
    }

    private static func resample(_ samples: [Float], sampleRate: Int) throws -> [Float] {
        if sampleRate == Self.sampleRate { return samples }
        #if canImport(AVFoundation)
        guard
            let sourceFormat = AVAudioFormat(
                commonFormat: .pcmFormatFloat32, sampleRate: Double(sampleRate), channels: 1,
                interleaved: false),
            let targetFormat = AVAudioFormat(
                commonFormat: .pcmFormatFloat32, sampleRate: Double(Self.sampleRate), channels: 1,
                interleaved: false),
            let converter = AVAudioConverter(from: sourceFormat, to: targetFormat),
            let input = AVAudioPCMBuffer(pcmFormat: sourceFormat, frameCapacity: 4096),
            let output = AVAudioPCMBuffer(pcmFormat: targetFormat, frameCapacity: 4096)
        else {
            throw EmbeddingGemma2Error.invalidInput(
                "Cannot create native audio sample-rate converter")
        }
        converter.sampleRateConverterQuality = AVAudioQuality.max.rawValue
        converter.primeMethod = .normal
        var cursor = 0
        var result: [Float] = []
        while true {
            try Task.checkCancellation()
            let previousCursor = cursor
            var error: NSError?
            output.frameLength = 0
            let status = converter.convert(to: output, error: &error) { requested, status in
                guard cursor < samples.count else {
                    status.pointee = .endOfStream
                    return nil
                }
                let count = min(Int(requested), Int(input.frameCapacity), samples.count - cursor)
                input.frameLength = AVAudioFrameCount(count)
                samples.withUnsafeBufferPointer { source in
                    input.floatChannelData![0].update(
                        from: source.baseAddress! + cursor, count: count)
                }
                cursor += count
                status.pointee = .haveData
                return input
            }
            if let error { throw error }
            if status == .error {
                throw EmbeddingGemma2Error.invalidInput(
                    "Native audio sample-rate conversion failed")
            }
            let count = Int(output.frameLength)
            try checkLength(result.count + count)
            if count > 0 {
                result.append(
                    contentsOf: UnsafeBufferPointer(
                        start: output.floatChannelData![0], count: count))
            }
            if status == .endOfStream {
                guard cursor == samples.count else {
                    throw EmbeddingGemma2Error.invalidInput(
                        "Native audio converter ended before consuming all PCM frames")
                }
                break
            }
            guard count > 0 || cursor > previousCursor else {
                throw EmbeddingGemma2Error.invalidInput(
                    "Native audio converter stopped before end of stream")
            }
        }
        return result
        #else
        throw EmbeddingGemma2Error.invalidInput(
            "Native audio resampling is unavailable on this platform")
        #endif
    }

    private static func decode(_ url: URL) async throws -> [Float] {
        guard url.isFileURL else {
            throw EmbeddingGemma2Error.invalidInput("Audio requires a local file URL")
        }
        #if canImport(AVFoundation)
        let asset = AVURLAsset(url: url)
        guard let track = try await asset.loadTracks(withMediaType: .audio).first else {
            throw EmbeddingGemma2Error.invalidInput("File has no audio track")
        }
        let descriptions = try await track.load(.formatDescriptions)
        guard let description = descriptions.first,
            let format = CMAudioFormatDescriptionGetStreamBasicDescription(description)?.pointee,
            format.mSampleRate.isFinite, format.mSampleRate > 0,
            format.mSampleRate.rounded() == format.mSampleRate,
            format.mSampleRate < Double(Int.max), format.mChannelsPerFrame > 0
        else {
            throw EmbeddingGemma2Error.invalidInput("Audio track has an unsupported PCM format")
        }
        let sampleRate = Int(format.mSampleRate)
        let channels = Int(format.mChannelsPerFrame)
        let duration = try await track.load(.timeRange).duration.seconds
        guard duration.isFinite, duration >= 0 else {
            throw EmbeddingGemma2Error.invalidInput("Audio track duration is invalid")
        }
        // Metadata is a preflight estimate. The decoded and resampled lengths are checked again.
        let estimate = floor(duration * Double(Self.sampleRate))
        guard estimate < Double(Int.max) else {
            throw EmbeddingGemma2Error.invalidInput("Audio duration exceeds the native array range")
        }
        try checkLength(Int(estimate))
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
            throw EmbeddingGemma2Error.invalidInput("Cannot decode audio track as native PCM")
        }
        reader.add(output)
        guard reader.startReading() else {
            throw reader.error
                ?? EmbeddingGemma2Error.invalidInput("Cannot start native audio decoder")
        }
        defer { reader.cancelReading() }
        var mono: [Float] = []
        while let buffer = output.copyNextSampleBuffer() {
            try Task.checkCancellation()
            guard let block = CMSampleBufferGetDataBuffer(buffer) else {
                throw EmbeddingGemma2Error.invalidInput(
                    "Audio decoder returned a buffer without PCM data")
            }
            let bytes = CMBlockBufferGetDataLength(block)
            guard bytes > 0, bytes % (MemoryLayout<Float>.size * channels) == 0 else {
                throw EmbeddingGemma2Error.invalidInput(
                    "Audio decoder returned an incomplete PCM frame")
            }
            let sampleCount = bytes / MemoryLayout<Float>.size
            let expected = try resampledCount(
                mono.count + sampleCount / channels, sampleRate: sampleRate)
            try checkLength(expected)
            var chunk = [Float](repeating: 0, count: sampleCount)
            let status = chunk.withUnsafeMutableBytes { destination in
                CMBlockBufferCopyDataBytes(
                    block, atOffset: 0, dataLength: bytes,
                    destination: destination.baseAddress!)
            }
            guard status == kCMBlockBufferNoErr else {
                throw EmbeddingGemma2Error.invalidInput("Cannot copy decoded PCM samples")
            }
            mono.append(contentsOf: try mix(chunk, channels: channels))
        }
        guard reader.status == .completed else {
            throw reader.error
                ?? EmbeddingGemma2Error.invalidInput("Native audio decoder did not complete")
        }
        guard !mono.isEmpty else {
            throw EmbeddingGemma2Error.invalidInput("Audio file contains no PCM frames")
        }
        return try resample(mono, sampleRate: sampleRate)
        #else
        throw EmbeddingGemma2Error.invalidInput(
            "Native audio decoding is unavailable on this platform")
        #endif
    }
}
