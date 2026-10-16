// Copyright © 2026 Faux Fox, Inc.

import Foundation
import MLX
import MLXNN
import XCTest

@testable import MLXLLM
@testable import MLXLMCommon

/// Tiny models that every rank builds with identical weights from a fixed seed.
enum TensorParallelFixture: String, CaseIterable, Codable {
    case llama
    case llamaQuantized
    case qwen3
    case qwen3MoE
    case qwen3MoEQuantized

    var seed: UInt64 {
        UInt64(Self.allCases.firstIndex(of: self)! + 1)
    }

    var quantization: BaseConfiguration.PerLayerQuantization? {
        switch self {
        case .llamaQuantized, .qwen3MoEQuantized:
            .init(quantization: .init(groupSize: 64, bits: 4), perLayerQuantization: [:])
        default:
            nil
        }
    }

    func makeModel() throws -> any LanguageModel {
        let decoder = JSONDecoder()
        switch self {
        case .llama, .llamaQuantized:
            let json = """
                {"model_type": "llama", "hidden_size": 128, "num_hidden_layers": 2,
                 "intermediate_size": 256, "num_attention_heads": 4, "num_key_value_heads": 2,
                 "head_dim": 32, "rms_norm_eps": 1e-5, "vocab_size": 256,
                 "tie_word_embeddings": false}
                """
            return LlamaModel(try decoder.decode(LlamaConfiguration.self, from: Data(json.utf8)))
        case .qwen3:
            let json = """
                {"hidden_size": 128, "num_hidden_layers": 2, "intermediate_size": 256,
                 "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 32,
                 "rms_norm_eps": 1e-6, "vocab_size": 256, "tie_word_embeddings": true}
                """
            return Qwen3Model(try decoder.decode(Qwen3Configuration.self, from: Data(json.utf8)))
        case .qwen3MoE, .qwen3MoEQuantized:
            // layer 0 is dense, layer 1 is MoE
            let json = """
                {"hidden_size": 128, "num_hidden_layers": 2, "intermediate_size": 256,
                 "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 32,
                 "num_experts": 8, "num_experts_per_tok": 2, "decoder_sparse_step": 1,
                 "mlp_only_layers": [0], "moe_intermediate_size": 128, "norm_topk_prob": true,
                 "rms_norm_eps": 1e-6, "vocab_size": 256, "tie_word_embeddings": false}
                """
            return Qwen3MoEModel(
                try decoder.decode(Qwen3MoEConfiguration.self, from: Data(json.utf8)))
        }
    }

    /// Write a seeded checkpoint to `directory`.
    func writeCheckpoint(to directory: URL) throws {
        MLXRandom.seed(seed)
        let model = try makeModel()
        if let quantization {
            let (groupSize, bits, mode) = quantization.quantization!.asTuple
            quantize(model: model, groupSize: groupSize, bits: bits, mode: mode)
        }
        let weights = Dictionary(uniqueKeysWithValues: model.parameters().flattened())
        eval(Array(weights.values))
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try save(arrays: weights, url: directory.appendingPathComponent("model.safetensors"))
    }

    func loadModel(from directory: URL, group: MLXDistributed.Group?) async throws
        -> any LanguageModel
    {
        let model = try makeModel()
        if let group {
            try await loadWeights(
                modelDirectory: directory, model: model, tensorParallelGroup: group,
                perLayerQuantization: quantization)
        } else {
            try await loadWeights(
                modelDirectory: directory, model: model, perLayerQuantization: quantization)
        }
        model.train(false)
        return model
    }
}

/// What one rank observed, written for the launching test to compare across ranks.
struct TensorParallelRankReport: Codable {
    struct FixtureReport: Codable {
        var fixture: TensorParallelFixture
        var maxLogitDifference: Float
        var referenceGreedy: [Int]
        var shardedGreedy: [Int]
        var sampled: [Int]
    }

    var rank: Int
    var size: Int
    var fixtures: [FixtureReport]
}

/// Prompt long enough that the MoE sorts its 2 x 40 expert assignments.
private let parityPrompt = (0 ..< 40).map { ($0 * 37 + 11) % 256 }

/// Runs as one rank of a group. The launching test starts it in two processes; on its own
/// it skips.
final class TensorParallelRankTests: XCTestCase {

    func testRank() async throws {
        let environment = ProcessInfo.processInfo.environment
        guard let resultDirectory = environment["FMLX_TP_RESULT_DIR"] else {
            throw XCTSkip("runs only as a rank started by TensorParallelTests")
        }

        let group = try MLXDistributed.initialize(backend: .ring, strict: true)
        let scratch = FileManager.default.temporaryDirectory
            .appendingPathComponent("fmlx-tp-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: scratch) }

        var report = TensorParallelRankReport(rank: group.rank, size: group.size, fixtures: [])
        func log(_ message: String) {
            FileHandle.standardError.write(Data("[rank \(group.rank)] \(message)\n".utf8))
        }
        for fixture in TensorParallelFixture.allCases {
            log("\(fixture): load")
            let directory = scratch.appendingPathComponent(fixture.rawValue)
            try fixture.writeCheckpoint(to: directory)

            let reference = try await fixture.loadModel(from: directory, group: nil)
            let sharded = try await fixture.loadModel(from: directory, group: group)

            log("\(fixture): logits")
            let tokens = MLXArray(parityPrompt.map { Int32($0) })[.newAxis]
            let expected = reference(tokens, cache: nil)
            let actual = sharded(tokens, cache: nil)
            let difference = abs(expected - actual).max().item(Float.self)

            var iterator = try TokenIterator(
                input: LMInput(tokens: MLXArray(parityPrompt.map { Int32($0) })),
                model: reference,
                parameters: GenerateParameters(maxTokens: 12, temperature: 0))
            var referenceGreedy = [Int]()
            while let token = iterator.next() {
                referenceGreedy.append(token)
            }

            let greedy = TensorParallelGenerationRequest(promptTokens: parityPrompt, maxTokens: 12)
            // unseeded, so each rank's own sample differs; rank 0's token must win
            let sampled = TensorParallelGenerationRequest(
                promptTokens: parityPrompt, maxTokens: 12, temperature: 1)

            log("\(fixture): generate")
            let generator = try TensorParallelGenerator(model: sharded)
            var shardedGreedy = [Int]()
            var sampledTokens = [Int]()
            if generator.isLeader {
                shardedGreedy = try generator.generate(greedy)
                sampledTokens = try generator.generate(sampled)
                try generator.shutdown()
            } else {
                var results = [[Int]]()
                try generator.followRequests { _, tokens in results.append(tokens) }
                XCTAssertEqual(results.count, 2)
                shardedGreedy = results.first ?? []
                sampledTokens = results.last ?? []
            }

            report.fixtures.append(
                .init(
                    fixture: fixture, maxLogitDifference: difference,
                    referenceGreedy: referenceGreedy, shardedGreedy: shardedGreedy,
                    sampled: sampledTokens))
        }

        let url = URL(fileURLWithPath: resultDirectory)
            .appendingPathComponent("rank-\(group.rank).json")
        try JSONEncoder().encode(report).write(to: url)
    }
}

final class TensorParallelTests: XCTestCase {

    /// Shards across a group of one, where every sharded layer must equal the original.
    func testShardingAcrossOneRankMatchesUnshardedModel() throws {
        let group = try MLXDistributed.initialize(backend: .ring, strict: false)
        XCTAssertEqual(group.size, 1)

        for fixture in TensorParallelFixture.allCases {
            MLXRandom.seed(fixture.seed)
            let reference = try fixture.makeModel()
            let sharded = try fixture.makeModel()
            try sharded.update(parameters: reference.parameters(), verify: .all)
            try XCTUnwrap(sharded as? any ShardableModel).shard(group: group)
            XCTAssertTrue((sharded as? any ShardableModel)?.tensorParallelGroup === group)

            let tokens = MLXArray(parityPrompt.map { Int32($0) })[.newAxis]
            XCTAssertTrue(
                allClose(reference(tokens, cache: nil), sharded(tokens, cache: nil)).item(
                    Bool.self),
                "\(fixture)")
        }
    }

    func testRequestRoundTripsThroughJSON() throws {
        let request = TensorParallelGenerationRequest(
            promptTokens: [1, 2, 3], maxTokens: 9, temperature: 0.7, topP: 0.9, seed: 5,
            prefillStepSize: 256, kvBits: 8)
        let decoded = try JSONDecoder().decode(
            TensorParallelGenerationRequest.self, from: JSONEncoder().encode(request))
        XCTAssertEqual(decoded, request)
        XCTAssertEqual(decoded.generateParameters.maxTokens, 9)
        XCTAssertEqual(decoded.generateParameters.prefill.stepSize, 256)
    }

    /// Starts two copies of ``TensorParallelRankTests/testRank()`` over the ring backend on
    /// localhost, then compares what each rank saw.
    func testTwoRanksOverRingBackendMatchUnshardedModel() throws {
        let environment = ProcessInfo.processInfo.environment
        guard environment["FMLX_TP_RESULT_DIR"] == nil else {
            throw XCTSkip("already running as a rank")
        }

        let bundle = Bundle(for: Self.self).bundleURL
        let scratch = FileManager.default.temporaryDirectory
            .appendingPathComponent("fmlx-tp-launch-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: scratch, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: scratch) }

        let port = Int.random(in: 20000 ..< 60000)
        let hostfile = scratch.appendingPathComponent("hostfile.json")
        try JSONEncoder().encode([["127.0.0.1:\(port)"], ["127.0.0.1:\(port + 1)"]])
            .write(to: hostfile)

        let ranks = try (0 ..< 2).map { rank -> (Process, URL) in
            let process = Process()
            process.executableURL = URL(fileURLWithPath: "/usr/bin/xcrun")
            process.arguments = [
                "xctest", "-XCTest", "MLXLMTests.TensorParallelRankTests/testRank", bundle.path,
            ]
            // drop the test runner's injection, or the child waits for the IDE session
            var childEnvironment = environment.filter { key, _ in
                !(key.hasPrefix("XCTest") || key.hasPrefix("XCInject")
                    || key.hasPrefix("__XCTest") || key == "DYLD_INSERT_LIBRARIES")
            }
            childEnvironment["MLX_RANK"] = "\(rank)"
            childEnvironment["MLX_HOSTFILE"] = hostfile.path
            childEnvironment["FMLX_TP_RESULT_DIR"] = scratch.path
            process.environment = childEnvironment
            let log = scratch.appendingPathComponent("rank-\(rank).log")
            FileManager.default.createFile(atPath: log.path, contents: nil)
            let handle = try FileHandle(forWritingTo: log)
            process.standardOutput = handle
            process.standardError = handle
            try process.run()
            return (process, log)
        }

        let deadline = Date().addingTimeInterval(600)
        while ranks.contains(where: { $0.0.isRunning }), Date() < deadline {
            Thread.sleep(forTimeInterval: 0.2)
        }
        for (process, _) in ranks where process.isRunning {
            process.terminate()
        }

        for (rank, (process, log)) in ranks.enumerated() {
            process.waitUntilExit()
            let output = (try? String(contentsOf: log, encoding: .utf8)) ?? ""
            XCTAssertEqual(
                process.terminationStatus, 0,
                "rank \(rank) failed:\n\(output.suffix(4000))")
        }

        let reports = try (0 ..< 2).map { rank in
            try JSONDecoder().decode(
                TensorParallelRankReport.self,
                from: Data(contentsOf: scratch.appendingPathComponent("rank-\(rank).json")))
        }
        XCTAssertEqual(reports.map(\.size), [2, 2])

        for (leader, follower) in zip(reports[0].fixtures, reports[1].fixtures) {
            let name = leader.fixture.rawValue
            for report in [leader, follower] {
                XCTAssertLessThan(report.maxLogitDifference, 1e-4, name)
                XCTAssertEqual(report.shardedGreedy, report.referenceGreedy, name)
            }
            XCTAssertEqual(leader.referenceGreedy.count, 12, name)
            XCTAssertEqual(leader.sampled.count, 12, name)
            XCTAssertEqual(follower.sampled, leader.sampled, name)
            print(
                "tensor parallel \(name): max |logit diff| rank0=\(leader.maxLogitDifference) "
                    + "rank1=\(follower.maxLogitDifference) greedy=\(leader.shardedGreedy) "
                    + "sampled=\(leader.sampled)")
        }
    }
}
