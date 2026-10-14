import Foundation
import MLX
import Testing

@testable import MLXVLM

@Suite(.serialized)
struct EmbeddingGemma2Tests {
    @Test("Text normalization matches pinned MLX BF16 learned-scale rounding")
    func normalizationRounding() throws {
        let norm = EmbeddingGemma2RMSNorm(8, epsilon: 1e-6)
        try norm.update(
            parameters: .unflattened([
                "weight": MLXArray([Float(0.4), 1.1, 0.6, 0.12, 2.8, 0.73, 1.5, 1.8]).asType(
                    .bfloat16)
            ]), verify: .all)
        let input = MLXArray([Float(0.13), -0.71, 1.3, -3.25, 0.007, 2.03, -0.125, 0.56])
            .reshaped(1, 8).asType(.bfloat16)
        // Pinned MLX-VLM text nn.RMSNorm, evaluated with Python MLX 0.32.1.
        let expected: [Float] = [
            0.035400390625, -0.53515625, 0.53125, -0.265625,
            0.0133056640625, 1.0078125, -0.126953125, 0.68359375,
        ]
        #expect(norm(input).asType(.float32).asArray(Float.self) == expected)
    }

    @Test("Audio preserves validity at waveform-padding and subsampling boundaries")
    func audioBoundaries() async throws {
        for count in [161, 320, 321, 640, 641, 1601] {
            let audio = try await EmbeddingGemma2AudioProcessor.process(
                .pcm(samples: Array(repeating: 0, count: count), sampleRate: 16000),
                maximumTokens: 100)
            let frames = (count - 1) / 160
            #expect(audio.validMask.asArray(Bool.self).filter { $0 }.count == frames)
            #expect(audio.tokenCount == (frames + 3) / 4)
            let values = audio.features.asArray(Float.self)
            for (index, valid) in audio.validMask.asArray(Bool.self).enumerated() {
                let row = values[(index * 128) ..< ((index + 1) * 128)]
                #expect(row.allSatisfy { abs($0 - (valid ? log(Float(0.001)) : 0)) < 1e-5 })
            }
        }
        await #expect(throws: EmbeddingGemma2Error.self) {
            try await EmbeddingGemma2AudioProcessor.process(
                .pcm(samples: Array(repeating: 0, count: 160), sampleRate: 16000),
                maximumTokens: 100)
        }
        await #expect(throws: EmbeddingGemma2Error.self) {
            try await EmbeddingGemma2AudioProcessor.process(
                .pcm(samples: Array(repeating: 0, count: 1601), sampleRate: 16000),
                maximumTokens: 2)
        }
    }

    @Test("Image patches preserve RGB order and padded positions")
    func imagePatches() throws {
        let image = try EmbeddingGemma2MediaProcessor.image(
            .rgb(
                bytes: Array(repeating: [UInt8(255), 128, 0], count: 384 * 384).flatMap { $0 },
                width: 384, height: 384), softTokenBudget: 70)
        #expect(image.tokenCount == 64)
        #expect(image.patches.shape == [1, 630, 768])
        let pixels = image.patches.asArray(Float.self)
        #expect(pixels[0] == 1)
        #expect(abs(pixels[1] - Float(128) / 255) < 1e-7)
        #expect(pixels[2] == 0)
        #expect(pixels.suffix((630 - 576) * 768).allSatisfy { $0 == 0 })
        let positions = image.positions.asArray(Int32.self)
        #expect(Array(positions.prefix(4)) == [0, 0, 1, 0])
        #expect(positions[2 * 575] == 23)
        #expect(positions[2 * 575 + 1] == 23)
        #expect(positions.suffix((630 - 576) * 2).allSatisfy { $0 == -1 })
    }

    @Test("Timestamped video chooses preceding frames and includes uniform endpoints")
    func videoSelection() async throws {
        let frames = (0 ... 40).map { index in
            EmbeddingGemma2VideoFrame(
                image: .rgb(
                    bytes: Array(repeating: UInt8(index), count: 3),
                    width: 1, height: 1), time: Double(index) * 1.5)
        }
        let selected = try await EmbeddingGemma2MediaProcessor.video(.frames(frames))
        #expect(selected.count == 32)
        for index in selected.indices {
            let second = 60 * index / 31
            let expected = Float(second * 2 / 3) / 255
            #expect(abs(selected[index].patches.asArray(Float.self)[0] - expected) < 1e-7)
        }
        #expect(abs(selected.last!.patches.asArray(Float.self)[0] - Float(40) / 255) < 1e-7)
    }

    @Test("Selected video keyframes retain count and order while honoring the patch budget")
    func videoKeyframes() async throws {
        let frames = [0.0, 5.0, 11.0].enumerated().map { index, time in
            EmbeddingGemma2VideoFrame(
                image: .rgb(
                    bytes: Array(repeating: UInt8(index * 80), count: 384 * 384 * 3),
                    width: 384, height: 384), time: time)
        }
        let compact = try await EmbeddingGemma2MediaProcessor.video(
            .keyframes(frames, softTokenBudget: 70))
        let standard = try await EmbeddingGemma2MediaProcessor.video(
            .keyframes(frames, softTokenBudget: 140))
        #expect(compact.count == 3)
        #expect(standard.count == 3)
        for index in frames.indices {
            #expect(compact[index].tokenCount == 64)
            #expect(compact[index].patches.shape == [1, 630, 768])
            #expect(standard[index].patches.shape == [1, 1260, 768])
            #expect(standard[index].tokenCount > compact[index].tokenCount)
            let expected = Float(index * 80) / 255
            #expect(abs(compact[index].patches.asArray(Float.self)[0] - expected) < 1e-7)
        }
        let wide = EmbeddingGemma2VideoFrame(
            image: .rgb(
                bytes: Array(repeating: UInt8(128), count: 768 * 384 * 3),
                width: 768, height: 384), time: 0)
        let wideFrames = try await EmbeddingGemma2MediaProcessor.video(
            .keyframes([wide], softTokenBudget: 140))
        #expect(wideFrames.first?.tokenCount == 128)
    }

    @Test("Selected video keyframes reject invalid counts, timestamps, and budgets")
    func invalidVideoKeyframes() async throws {
        let rgb = EmbeddingGemma2Image.rgb(bytes: [0, 0, 0], width: 1, height: 1)
        let invalidTimes: [[Double]] = [
            [], [0, 0], [1, 0], [-1], [.nan], [.infinity], [-.infinity],
            (0 ... 32).map(Double.init),
        ]
        for times in invalidTimes {
            let frames = times.map { EmbeddingGemma2VideoFrame(image: rgb, time: $0) }
            await #expect(throws: EmbeddingGemma2Error.self) {
                try await EmbeddingGemma2MediaProcessor.video(
                    .keyframes(frames, softTokenBudget: 70))
            }
        }
        for budget in [0, 69, 141, 1121] {
            await #expect(throws: EmbeddingGemma2Error.self) {
                try await EmbeddingGemma2MediaProcessor.video(
                    .keyframes([.init(image: rgb, time: 0)], softTokenBudget: budget))
            }
        }
    }

    @Test(
        "Pinned checkpoint executes all modalities and reloads identical text embeddings",
        .enabled(if: ProcessInfo.processInfo.environment["EMBEDDINGGEMMA2_MODEL_DIRECTORY"] != nil))
    func pinnedCheckpoint() async throws {
        let environment = ProcessInfo.processInfo.environment
        let directory = URL(
            fileURLWithPath: try #require(environment["EMBEDDINGGEMMA2_MODEL_DIRECTORY"]))
        let output = URL(
            fileURLWithPath: try #require(environment["EMBEDDINGGEMMA2_PARITY_DIRECTORY"]))
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
        let precision = try #require(
            EmbeddingGemma2Precision(
                rawValue: environment["EMBEDDINGGEMMA2_PRECISION"] ?? "bfloat16"))
        let container = try await EmbeddingGemma2Loader.loadContainer(
            from: directory,
            artifact: .init(
                identifier: "mlx-community/embeddinggemma-2-bf16",
                revision: "1a4ffddb7905d3f63486748deabe091a01fb6201"), precision: precision)
        #expect(container.resources.checkpointBytes == 1_488_743_984)
        let tokenizer = try await EmbeddingGemma2Tokenizer.load(from: directory)
        let config = try JSONDecoder().decode(
            EmbeddingGemma2Configuration.self,
            from: Data(contentsOf: directory.appendingPathComponent("config.json")))
        let rgb = EmbeddingGemma2Image.rgb(
            bytes: (0 ..< (384 * 384 * 3)).map { UInt8($0 % 251) },
            width: 384, height: 384)
        let pcm = (0 ..< 1601).map { Float(sin(Double($0) * 2 * .pi * 440 / 16000) * 0.25) }
        let image = try EmbeddingGemma2MediaProcessor.image(rgb, softTokenBudget: 70)
        let video = try await EmbeddingGemma2MediaProcessor.video(
            .frames([.init(image: rgb, time: 0)]))
        let audio = try await EmbeddingGemma2MediaProcessor.audio(
            .pcm(samples: pcm, sampleRate: 16000), maximumTokens: 100)
        try save(
            arrays: [
                "image": image.patches, "image_positions": image.positions,
                "video": video[0].patches, "video_positions": video[0].positions,
                "audio": audio.features, "audio_mask": audio.validMask,
            ],
            url: output.appendingPathComponent("inputs.safetensors"))
        func token(_ id: Int) throws -> String { try #require(tokenizer.convertIdToToken(id)) }
        func media(_ id: Int, count: Int, begin: Int, end: Int) throws -> String {
            try token(begin) + String(repeating: token(id), count: count) + token(end)
        }
        let prefix = "title: none | text: "
        let imagePrompt = try media(
            config.imageToken, count: image.tokenCount, begin: config.beginImage,
            end: config.endImage)
        let audioPrompt = try media(
            config.audioToken, count: audio.tokenCount, begin: config.beginAudio,
            end: config.endAudio)
        let videoPrompt = try media(
            config.videoToken, count: video[0].tokenCount, begin: config.beginImage,
            end: config.endImage)
        let cases: [(String, [EmbeddingGemma2Content], String)] = [
            ("text", [.text("Mars is the Red Planet.")], "Mars is the Red Planet."),
            ("query", [.text("Which planet is red?")], "Which planet is red?"),
            ("image", [.image(rgb)], imagePrompt),
            ("audio", [.audio(.pcm(samples: pcm, sampleRate: 16000))], audioPrompt),
            ("video", [.video(.frames([.init(image: rgb, time: 0)]))], videoPrompt),
            (
                "mixed",
                [
                    .text("Mars: "), .image(rgb), .text(" observation "),
                    .audio(.pcm(samples: pcm, sampleRate: 16000)),
                    .video(.frames([.init(image: rgb, time: 0)])),
                ], "Mars: " + imagePrompt + " observation " + audioPrompt + videoPrompt
            ),
        ]
        var records = [[String: Any]]()
        for (name, content, prompt) in cases {
            let start = Date()
            let purpose: EmbeddingGemma2Purpose = name == "query" ? .query() : .document()
            let result = try await container.embed(
                content, purpose: purpose, imageSoftTokenBudget: 70)
            let actualPrefix = name == "query" ? "task: search result | query: " : prefix
            let ids = try tokenizer.encode(text: actualPrefix + prompt, addSpecialTokens: true)
            #expect(result.tokenCount == ids.count)
            #expect(result.values.allSatisfy { $0.isFinite })
            if let referencePath = environment["EMBEDDINGGEMMA2_REFERENCE_FILE"] {
                let reference = try JSONDecoder().decode(
                    Reference.self,
                    from: Data(contentsOf: URL(fileURLWithPath: referencePath)))
                #expect(reference.modelRevision == "1a4ffddb7905d3f63486748deabe091a01fb6201")
                #expect(reference.sourceRevision == EmbeddingGemma2Artifact.pythonReferenceRevision)
                let expected = try #require(reference.results.first { $0.name == name })
                let error =
                    zip(result.values, expected.values).map { abs($0 - $1) }.max() ?? .infinity
                #expect(expected.values.count == result.values.count)
                #expect(error < (precision == .float32 ? 1e-5 : 5e-3))
                let dot = zip(result.values, expected.values).reduce(Float(0)) { $0 + $1.0 * $1.1 }
                let expectedNorm = sqrt(expected.values.reduce(Float(0)) { $0 + $1 * $1 })
                #expect(dot / expectedNorm > 0.999)
            }
            records.append([
                "name": name, "tokens": ids, "values": result.values,
                "seconds": Date().timeIntervalSince(start),
            ])
        }
        #expect(await container.loadedModalities == [.text, .vision, .audio])
        let mixed = try #require(cases.first { $0.0 == "mixed" })
        let beforeMediaUnload = try await container.embed(mixed.1, imageSoftTokenBudget: 70)
        await container.unload([.vision, .audio])
        #expect(await container.loadedModalities == [.text])
        let afterMediaReload = try await container.embed(mixed.1, imageSoftTokenBudget: 70)
        #expect(beforeMediaUnload.values == afterMediaReload.values)
        await #expect(throws: EmbeddingGemma2Error.self) {
            try await container.embed(text: String(repeating: "Mars ", count: 8193))
        }
        let first = try await container.embed(text: "Mars is the Red Planet.")
        await container.unload()
        #expect(await container.loadedModalities == [])
        let reloaded = try await container.embed(text: "Mars is the Red Planet.")
        #expect(first.values == reloaded.values)
        for dimension in EmbeddingGemma2Dimension.allCases {
            let result = try await container.embed(
                text: "Mars is the Red Planet.", dimension: dimension)
            #expect(result.values.count == dimension.rawValue)
            let prefix = first.values.prefix(dimension.rawValue)
            let norm = sqrt(prefix.reduce(Float(0)) { $0 + $1 * $1 })
            #expect(zip(result.values, prefix).allSatisfy { abs($0 - $1 / norm) < 1e-5 })
        }
        try JSONSerialization.data(withJSONObject: records, options: [.prettyPrinted, .sortedKeys])
            .write(to: output.appendingPathComponent("swift.json"))
        print("EmbeddingGemma2 MLX peak bytes: \(Memory.peakMemory)")
        await container.unload()
    }

    private struct Reference: Decodable {
        let modelRevision: String
        let sourceRevision: String
        let results: [Vector]
        enum CodingKeys: String, CodingKey {
            case modelRevision = "model_revision"
            case sourceRevision = "source_revision"
            case results
        }
        struct Vector: Decodable {
            let name: String
            let values: [Float]
            enum CodingKeys: String, CodingKey {
                case name
                case values = "reference_values"
            }
        }
    }
}
