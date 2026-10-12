// Copyright © 2026 Faux Fox, Inc.
// Adapted from Hugging Face Transformers' Gemma4AudioFeatureExtractor and audio_utils,
// Copyright 2026 Google LLC and the HuggingFace Team.
// Licensed under the Apache License, Version 2.0.
// You may obtain a copy at https://www.apache.org/licenses/LICENSE-2.0.
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

import Foundation
import MLX

#if canImport(AVFoundation)
@preconcurrency import AVFoundation
#endif

enum EmbeddingGemma2AudioProcessor {
    static func process(_ input: EmbeddingGemma2Audio, maximumTokens: Int) async throws
        -> EmbeddingGemma2PreparedAudio
    {
        guard maximumTokens > 0 else {
            throw EmbeddingGemma2Error.invalidConfiguration("Audio token budget must be positive")
        }
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
            let expected = try resampledCount(samples.count / channels, sampleRate: sampleRate)
            try checkBudget(sampleCount: expected, maximumTokens: maximumTokens)
            let mono = try mix(samples, channels: channels)
            waveform = try resample(mono, sampleRate: sampleRate, maximumTokens: maximumTokens)
        case .file(let url):
            waveform = try await decode(url, maximumTokens: maximumTokens)
        }
        try Task.checkCancellation()
        return try features(waveform, maximumTokens: maximumTokens)
    }

    // Unfold requires 321 samples after prepending 160; only the first 320 are analyzed.
    private static func frameCount(sampleCount: Int) -> Int {
        sampleCount > 160 ? (sampleCount - 1) / 160 : 0
    }

    private static func tokens(frameCount: Int) -> Int {
        let first = frameCount / 2 + frameCount % 2
        return first / 2 + first % 2
    }

    private static func checkBudget(sampleCount: Int, maximumTokens: Int) throws {
        guard sampleCount >= 0, sampleCount <= Int(Int32.max) - 256 else {
            throw EmbeddingGemma2Error.invalidInput(
                "Audio sample count exceeds the native array range")
        }
        let count = tokens(frameCount: frameCount(sampleCount: sampleCount))
        if count > maximumTokens {
            throw EmbeddingGemma2Error.tokenLimit(count: count, limit: maximumTokens)
        }
    }

    private static func resampledCount(_ count: Int, sampleRate: Int) throws -> Int {
        guard count >= 0, sampleRate > 0 else {
            throw EmbeddingGemma2Error.invalidInput("Invalid audio sample count or rate")
        }
        let (scaled, overflow) = count.multipliedReportingOverflow(by: 16_000)
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

    private static func resample(_ samples: [Float], sampleRate: Int, maximumTokens: Int) throws
        -> [Float]
    {
        if sampleRate == 16_000 { return samples }
        #if canImport(AVFoundation)
        guard
            let sourceFormat = AVAudioFormat(
                commonFormat: .pcmFormatFloat32, sampleRate: Double(sampleRate), channels: 1,
                interleaved: false),
            let targetFormat = AVAudioFormat(
                commonFormat: .pcmFormatFloat32, sampleRate: 16_000, channels: 1, interleaved: false
            ),
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
            try checkBudget(sampleCount: result.count + count, maximumTokens: maximumTokens)
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

    private static func decode(_ url: URL, maximumTokens: Int) async throws -> [Float] {
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
        let estimate = floor(duration * 16_000)
        guard estimate < Double(Int.max) else {
            throw EmbeddingGemma2Error.invalidInput("Audio duration exceeds the native array range")
        }
        try checkBudget(sampleCount: Int(estimate), maximumTokens: maximumTokens)
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
            try checkBudget(sampleCount: expected, maximumTokens: maximumTokens)
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
        return try resample(mono, sampleRate: sampleRate, maximumTokens: maximumTokens)
        #else
        throw EmbeddingGemma2Error.invalidInput(
            "Native audio decoding is unavailable on this platform")
        #endif
    }

    private static func features(_ waveform: [Float], maximumTokens: Int) throws
        -> EmbeddingGemma2PreparedAudio
    {
        try checkBudget(sampleCount: waveform.count, maximumTokens: maximumTokens)
        let validFrames = frameCount(sampleCount: waveform.count)
        guard validFrames > 0 else {
            throw EmbeddingGemma2Error.invalidInput("Audio requires at least 161 samples at 16 kHz")
        }
        guard waveform.allSatisfy(\.isFinite) else {
            throw EmbeddingGemma2Error.invalidInput("Resampled PCM samples must be finite")
        }
        // Preserve the extractor's default waveform padding to a multiple of 128.
        let paddedCount = ((waveform.count + 127) / 128) * 128
        let frames = frameCount(sampleCount: paddedCount)
        guard frames <= Int(Int32.max) / 128 else {
            throw EmbeddingGemma2Error.invalidInput(
                "Audio feature tensor exceeds the native array range")
        }
        var padded = [Float](repeating: 0, count: paddedCount + 160)
        padded.replaceSubrange(160 ..< 160 + waveform.count, with: waveform)
        let window = MLXArray(
            (0 ..< 320).map { index in
                Float(0.5 - 0.5 * cos(2 * Double.pi * Double(index) / 320))
            })
        let unfolded = asStrided(MLXArray(padded), [frames, 321], strides: [160, 1])
        let analysis = unfolded[0..., 0 ..< 320] * window
        let spectrum = abs(MLXFFT.rfft(analysis, n: 512, axis: -1))
        let mel = matmul(spectrum, MLXArray(melFilters(), [257, 128]))
        let mask = MLXArray((0 ..< frames).map { $0 < validFrames }, [1, frames])
        let logMel = log(mel + Float(0.001)).expandedDimensions(axis: 0)
        return EmbeddingGemma2PreparedAudio(
            features: which(mask.expandedDimensions(axis: -1), logMel, MLXArray(Float(0))),
            validMask: mask,
            tokenCount: tokens(frameCount: validFrames))
    }

    private static func melFilters() -> [Float] {
        let maximumMel = 2595 * log10(1 + 8000.0 / 700)
        let edges = (0 ..< 130).map { index in
            700 * (pow(10, (maximumMel * Double(index) / 129) / 2595) - 1)
        }
        var filters = [Float](repeating: 0, count: 257 * 128)
        for frequency in 0 ..< 257 {
            let hertz = Double(frequency) * 8000 / 256
            for mel in 0 ..< 128 {
                let rising = (hertz - edges[mel]) / (edges[mel + 1] - edges[mel])
                let falling = (edges[mel + 2] - hertz) / (edges[mel + 2] - edges[mel + 1])
                filters[frequency * 128 + mel] = Float(max(0, min(rising, falling)))
            }
        }
        return filters
    }
}
