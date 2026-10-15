// Copyright © 2026 Faux Fox, Inc.

import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXDecision

/// A random tiny d1-omni in the PyTorch checkpoint layout, checked against the PyTorch reference.
@Suite(.serialized)
struct D1OmniTinyModelTests {
    static let directory = Fixtures.directory.appendingPathComponent("tiny-d1-omni")

    static func configuration() throws -> D1OmniConfiguration {
        let configuration = try JSONDecoder().decode(
            D1OmniConfiguration.self,
            from: Data(contentsOf: directory.appendingPathComponent("config.json")))
        try configuration.validate()
        return configuration
    }

    static let source = D1OmniWeightSource(
        files: [directory.appendingPathComponent("model.safetensors")], quantization: nil,
        precision: .checkpoint)

    static func network() throws -> D1OmniNetwork {
        let network = D1OmniNetwork(try configuration())
        try source.load(.text, into: network)
        return network
    }

    static func runtime() throws -> D1OmniRuntime {
        D1OmniRuntime(
            network: try network(), tokenizer: FakeD1Tokenizer(), special: fakeD1Special,
            source: source)
    }

    static func reference() throws -> [String: Any] {
        try Fixtures.json("tiny-d1-omni/reference.json") as! [String: Any]
    }

    static func rows(_ entries: [[String: Any]], prefix: MLXArray?) -> [D1OmniNetwork.Row] {
        entries.map { entry in
            let kind: D1OmniQuestion.Kind =
                switch entry["type"] as! String {
                case "choice": .choice
                case "score": .score
                default: .noul
                }
            return D1OmniNetwork.Row(
                prefix: prefix, ids: entry["ids"] as! [Int], markers: entry["markers"] as! [Int],
                kind: kind)
        }
    }

    @Test("Text rows give the reference logits, batched with padding")
    func textLogits() throws {
        let network = try Self.network()
        let text = try Self.reference()["text"] as! [String: Any]
        let entries = text["rows"] as! [[String: Any]]
        let expected = text["logits"] as! [[Double]]
        let logits = network.logits(Self.rows(entries, prefix: nil))
        for (row, reference) in zip(logits, expected) {
            expectClose(row.map(Double.init), reference, tolerance: 2e-4)
        }
        // A row alone matches the same row in a padded batch.
        let single = network.logits(Self.rows([entries[2]], prefix: nil))
        expectClose(single[0].map(Double.init), expected[2], tolerance: 2e-4)
    }

    @Test("Image prefixes and logits match the reference, tiled and whole")
    func imagePrefix() throws {
        let network = try Self.network()
        let vision = D1OmniVision(try Self.configuration())
        try Self.source.load(.vision, into: vision)
        let reference = try Self.reference()
        let text = reference["text"] as! [String: Any]
        let entries = Array((text["rows"] as! [[String: Any]]).prefix(2))
        for entry in reference["images"] as! [[String: Any]] {
            let width = entry["width"] as! Int
            let height = entry["height"] as! Int
            let image = try D1OmniImageProcessor.prepare(
                DecisionRGBImage(
                    bytes: Fixtures.rgb(width: width, height: height), width: width, height: height)
            )
            let prefix = vision([image])
            #expect(prefix.shape == entry["prefix_shape"] as! [Int])
            expectSummary(prefix, entry["prefix"] as! [String: Any], tolerance: 2e-3)
            let logits = network.logits(Self.rows(entries, prefix: prefix))
            for (row, expected) in zip(logits, entry["logits"] as! [[Double]]) {
                expectClose(row.map(Double.init), expected, tolerance: 2e-3)
            }
        }
    }

    @Test("Log-mel features, encoder frames, and audio logits match the reference")
    func audioPrefix() throws {
        let network = try Self.network()
        let audio = D1OmniAudio(try Self.configuration())
        try Self.source.load(.audio, into: audio)
        let reference = try Self.reference()
        let text = reference["text"] as! [String: Any]
        let entries = Array((text["rows"] as! [[String: Any]]).suffix(2))
        for entry in reference["audio"] as! [[String: Any]] {
            let pcm = Fixtures.pcm(count: entry["samples"] as! Int)
            #expect(Fixtures.checksum(pcm) == entry["pcm_checksum"] as! Int64)
            let samples = D1OmniMelFrontend.clip(pcm.map { Float($0) / 32768 })
            let (mel, frames) = D1OmniMelFrontend.features(samples, bins: 128)
            #expect(frames == entry["frames"] as! Int)
            let melShape = entry["mel_shape"] as! [Int]
            #expect(mel.shape == [melShape[1], melShape[0]])
            expectSummary(mel.transposed(), entry["mel"] as! [String: Any], tolerance: 2e-3)
            let encoded = audio.encoder(mel[.newAxis], valid: frames)
            #expect(encoded.dim(1) == entry["encoded_length"] as! Int)
            let prefix = audio(pcm.map { Float($0) / 32768 })
            #expect(prefix.shape == entry["prefix_shape"] as! [Int])
            expectSummary(prefix, entry["prefix"] as! [String: Any], tolerance: 2e-3)
            let logits = network.logits(Self.rows(entries, prefix: prefix))
            for (row, expected) in zip(logits, entry["logits"] as! [[Double]]) {
                expectClose(row.map(Double.init), expected, tolerance: 2e-3)
            }
        }
    }

    @Test("Requests run end to end: prompts, media, calibration, and yes/no order")
    func endToEnd() async throws {
        let runtime = try Self.runtime()
        let artifact = DecisionArtifact(identifier: "tiny", revision: "fixture")
        for entry in try Self.reference()["end_to_end"] as! [[String: Any]] {
            let questions = (entry["questions"] as! [[String: Any]]).map(decisionQuestion)
            var request = DecisionRequest(state: entry["state"] as? String, questions: questions)
            if let media = entry["media"] as? [Any] {
                if media[0] as! String == "image" {
                    let (width, height) = (media[1] as! Int, media[2] as! Int)
                    request.images = [
                        .rgb(
                            bytes: Fixtures.rgb(width: width, height: height), width: width,
                            height: height)
                    ]
                } else {
                    request.audio = .pcm16(
                        samples: Fixtures.pcm(count: media[1] as! Int), sampleRate: 16000)
                }
            }
            let result = try await runtime.decide(
                [request], tokenBudget: 65536, artifact: artifact)[0]
            #expect(result.inputTokens == entry["input_tokens"] as! Int)
            for (distribution, expected) in zip(
                result.distributions, entry["probabilities"] as! [[Double]])
            {
                expectClose(distribution.probabilities, expected, tolerance: 2e-3)
                #expect(abs(distribution.probabilities.reduce(0, +) - 1) < 1e-9)
            }
            #expect(result.distributions[0].outcomes == ["yes", "no"])
            #expect(result.distributions[0].temperature == (entry["media"] is NSNull ? 1.75 : 1))
        }
    }

    @Test("Several requests batch together and keep their order")
    func batching() async throws {
        let runtime = try Self.runtime()
        let artifact = DecisionArtifact(identifier: "tiny", revision: "fixture")
        let question = DecisionQuestion(
            id: "team", instructions: "Which team?",
            kind: .choice([DecisionOption("billing"), DecisionOption("technical")]))
        let states = ["short", String(repeating: "long state ", count: 40), "medium length state"]
        let requests = states.map { DecisionRequest(state: $0, questions: [question]) }
        let together = try await runtime.decide(requests, tokenBudget: 65536, artifact: artifact)
        let tight = try await runtime.decide(requests, tokenBudget: 64, artifact: artifact)
        for state in states.indices {
            let alone = try await runtime.decide(
                [requests[state]], tokenBudget: 65536, artifact: artifact)
            expectClose(
                together[state].distributions[0].probabilities,
                alone[0].distributions[0].probabilities, tolerance: 1e-4)
            expectClose(
                tight[state].distributions[0].probabilities,
                alone[0].distributions[0].probabilities, tolerance: 1e-4)
        }
        await #expect(throws: DecisionError.self) {
            try await runtime.decide(
                [
                    DecisionRequest(
                        state: "x", questions: [question],
                        images: [.rgb(bytes: [0, 0, 0], width: 1, height: 1)],
                        audio: .pcm(samples: [0], sampleRate: 16000))
                ],
                tokenBudget: 65536, artifact: artifact)
        }
        await #expect(throws: DecisionError.self) {
            try await runtime.decide(
                [
                    DecisionRequest(
                        state: "x",
                        questions: [
                            DecisionQuestion(
                                id: "bad", instructions: "", kind: .choice([DecisionOption("only")])
                            )
                        ])
                ],
                tokenBudget: 65536, artifact: artifact)
        }
    }

    @Test("Quantized and float32 loads keep the module tree and outputs")
    func precisionAndQuantization() throws {
        let directory = Fixtures.directory.appendingPathComponent("tiny-d1-omni")
        let configuration = try JSONDecoder().decode(
            D1OmniConfiguration.self,
            from: Data(contentsOf: directory.appendingPathComponent("config.json")))
        let weights = D1OmniPart.text.weights(
            D1OmniNetwork.sanitize(
                try DecisionWeights.load([directory.appendingPathComponent("model.safetensors")])))
        let reference = try Self.network()
        let text = try Self.reference()["text"] as! [String: Any]
        let rows = Self.rows(text["rows"] as! [[String: Any]], prefix: nil)
        let expected = reference.logits(rows)

        // Quantize the trunk's linear layers as an MLX export would, then load them back.
        var quantized = weights
        var dequantized = weights
        for (key, value) in weights
        where key.hasPrefix("encoder.layers.") && key.hasSuffix(".weight") && value.ndim == 2
            && value.dim(-1) % 64 == 0
        {
            let (w, scales, biases) = MLX.quantized(value, groupSize: 64, bits: 8)
            let base = String(key.dropLast(".weight".count))
            quantized[key] = w
            quantized["\(base).scales"] = scales
            quantized["\(base).biases"] = biases!
            dequantized[key] = MLX.dequantized(
                w, scales: scales, biases: biases, groupSize: 64, bits: 8)
        }
        let network = D1OmniNetwork(configuration)
        try DecisionWeights.apply(
            quantized, to: network, groupSize: 64, bits: 8, mode: "affine", precision: .checkpoint)
        #expect(network.encoder.layers[0].feedForward.w1 is QuantizedLinear)
        #expect(!(network.head.typeEmbedding is QuantizedEmbedding))
        let plain = D1OmniNetwork(configuration)
        try DecisionWeights.apply(
            dequantized, to: plain, groupSize: nil, bits: nil, mode: nil, precision: .checkpoint)
        for ((row, reference), quantizedRow) in zip(
            zip(plain.logits(rows), expected), network.logits(rows))
        {
            expectClose(quantizedRow.map(Double.init), row.map(Double.init), tolerance: 1e-3)
            expectClose(row.map(Double.init), reference.map(Double.init), tolerance: 0.5)
        }

        // Half-precision weights run in float16, or in float32 when asked.
        let half = weights.mapValues { $0.dtype == .float32 ? $0.asType(.float16) : $0 }
        let float16 = D1OmniNetwork(configuration)
        try DecisionWeights.apply(
            half, to: float16, groupSize: nil, bits: nil, mode: nil, precision: .checkpoint)
        #expect(float16.dtype == .float16)
        let float32 = D1OmniNetwork(configuration)
        try DecisionWeights.apply(
            half, to: float32, groupSize: nil, bits: nil, mode: nil, precision: .float32)
        #expect(float32.dtype == .float32)
        for ((a, b), reference) in zip(zip(float16.logits(rows), float32.logits(rows)), expected) {
            expectClose(a.map(Double.init), reference.map(Double.init), tolerance: 0.05)
            expectClose(b.map(Double.init), reference.map(Double.init), tolerance: 0.05)
        }

        let missing = weights.filter { $0.key != "head.type_emb.weight" }
        #expect(throws: DecisionError.self) {
            try DecisionWeights.apply(
                missing, to: D1OmniNetwork(configuration), groupSize: nil, bits: nil, mode: nil,
                precision: .checkpoint)
        }
        #expect(throws: DecisionError.self) {
            try DecisionWeights.apply(
                quantized, to: D1OmniNetwork(configuration), groupSize: nil, bits: nil, mode: nil,
                precision: .checkpoint)
        }
    }

    @Test("Image and audio towers load with their first request and unload on demand")
    func lazyTowers() async throws {
        let runtime = try Self.runtime()
        #expect(runtime.loadedModalities == [.text])
        let artifact = DecisionArtifact(identifier: "tiny", revision: "fixture")
        let question = DecisionQuestion(id: "q", instructions: "Holds?", kind: .yesNo())
        let image = DecisionRequest(
            state: nil, questions: [question],
            images: [.rgb(bytes: Fixtures.rgb(width: 64, height: 64), width: 64, height: 64)])
        let first = try await runtime.decide([image], tokenBudget: 65536, artifact: artifact)
        #expect(runtime.loadedModalities == [.text, .image])
        runtime.unload([.image, .text])
        #expect(runtime.loadedModalities == [.text])
        let again = try await runtime.decide([image], tokenBudget: 65536, artifact: artifact)
        #expect(first[0].distributions[0].probabilities == again[0].distributions[0].probabilities)
        let audio = DecisionRequest(
            state: nil, questions: [question],
            audio: .pcm16(samples: Fixtures.pcm(count: 9000), sampleRate: 16000))
        _ = try await runtime.decide([audio], tokenBudget: 65536, artifact: artifact)
        #expect(runtime.loadedModalities == [.text, .image, .audio])
    }
}
