// Copyright © 2026 Faux Fox, Inc.

import Foundation
import MLX
import Testing

@testable import MLXVLM

/// Container media selection on top of the shared EmbeddingGemma 2 processors.
@Suite(.serialized)
struct EmbeddingGemma2ContainerTests {
    private static let processorConfig =
        #"{"processor_class":"Gemma4Processor","do_normalize":false,"patch_size":16,"pooling_kernel_size":3,"max_soft_tokens":280,"video_processor":{"fps":1.0,"max_frames":32,"max_soft_tokens":70,"overflow_strategy":"uniform","add_timestamps":false}}"#

    private static func processors() throws -> (
        EmbeddingGemma2ImageProcessor, EmbeddingGemma2VideoProcessor
    ) {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        try Data(processorConfig.utf8).write(
            to: directory.appendingPathComponent("processor_config.json"))
        return (
            try EmbeddingGemma2ImageProcessor(directory: directory),
            try EmbeddingGemma2VideoProcessor(directory: directory)
        )
    }

    @Test("Timestamped video chooses preceding frames and includes uniform endpoints")
    func videoSelection() async throws {
        let (images, videos) = try Self.processors()
        let frames = (0 ... 40).map { index in
            EmbeddingGemma2VideoFrame(
                image: .rgb(
                    bytes: Array(repeating: UInt8(index), count: 3),
                    width: 1, height: 1), time: Double(index) * 1.5)
        }
        let selected = try await EmbeddingGemma2MediaProcessor.video(
            .frames(frames), images: images, video: videos)
        #expect(selected.dim(0) == 32)
        let pixels = selected[0..., 0, 0, 0].asArray(Float.self)
        for index in pixels.indices {
            let second = 60 * index / 31
            let expected = Float(second * 2 / 3) / 255
            #expect(abs(pixels[index] - expected) < 0.5 / 255)
        }
        #expect(abs(pixels.last! - Float(40) / 255) < 0.5 / 255)
    }

    @Test("Selected video keyframes retain count and order while honoring the patch budget")
    func videoKeyframes() async throws {
        let (images, videos) = try Self.processors()
        let frames = [0.0, 5.0, 11.0].enumerated().map { index, time in
            EmbeddingGemma2VideoFrame(
                image: .rgb(
                    bytes: Array(repeating: UInt8(index * 80), count: 384 * 384 * 3),
                    width: 384, height: 384), time: time)
        }
        func tokens(_ pixels: MLXArray) -> Int {
            images.configuration.softTokenCount(height: pixels.dim(2), width: pixels.dim(3))
        }
        let compact = try await EmbeddingGemma2MediaProcessor.video(
            .keyframes(frames, softTokenBudget: 70), images: images, video: videos)
        let standard = try await EmbeddingGemma2MediaProcessor.video(
            .keyframes(frames, softTokenBudget: 140), images: images, video: videos)
        #expect(compact.dim(0) == 3)
        #expect(standard.dim(0) == 3)
        #expect(tokens(compact) == 64)
        #expect(tokens(standard) > tokens(compact))
        let first = compact[0..., 0, 0, 0].asArray(Float.self)
        for index in frames.indices {
            #expect(abs(first[index] - Float(index * 80) / 255) < 0.5 / 255)
        }
        let wide = EmbeddingGemma2VideoFrame(
            image: .rgb(
                bytes: Array(repeating: UInt8(128), count: 768 * 384 * 3),
                width: 768, height: 384), time: 0)
        let wideFrames = try await EmbeddingGemma2MediaProcessor.video(
            .keyframes([wide], softTokenBudget: 140), images: images, video: videos)
        #expect(tokens(wideFrames) == 128)
    }

    @Test("Selected video keyframes reject invalid counts, timestamps, and budgets")
    func invalidVideoKeyframes() async throws {
        let (images, videos) = try Self.processors()
        let rgb = EmbeddingGemma2Image.rgb(bytes: [0, 0, 0], width: 1, height: 1)
        let invalidTimes: [[Double]] = [
            [], [0, 0], [1, 0], [-1], [.nan], [.infinity], [-.infinity],
            (0 ... 32).map(Double.init),
        ]
        for times in invalidTimes {
            let frames = times.map { EmbeddingGemma2VideoFrame(image: rgb, time: $0) }
            await #expect(throws: EmbeddingGemma2Error.self) {
                try await EmbeddingGemma2MediaProcessor.video(
                    .keyframes(frames, softTokenBudget: 70), images: images, video: videos)
            }
        }
        for budget in [0, 69, 141, 1121] {
            await #expect(throws: EmbeddingGemma2Error.self) {
                try await EmbeddingGemma2MediaProcessor.video(
                    .keyframes([.init(image: rgb, time: 0)], softTokenBudget: budget),
                    images: images, video: videos)
            }
        }
    }
}
