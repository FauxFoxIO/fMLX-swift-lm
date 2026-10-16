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

/// Parity fixtures from `scripts/parity` whose heads and expert widths split across two ranks.
enum TensorParallelParityFixture: String, CaseIterable, Codable {
    case glmMoEDSA = "glm_moe_dsa"
    case museSparkMeta = "muse_spark_meta"
    case museSparkNVFP4 = "muse_spark_nvfp4_tp"
    case deepseekV41 = "deepseek_v41"
    case glm5Next = "glm5_next"

    var hasMTP: Bool { self == .glmMoEDSA }

    /// Fixtures whose speculative drafter is checked with ``TensorParallelRankReport/SpeculativeReport``.
    var hasDrafter: Bool { self == .deepseekV41 || self == .glm5Next }

    /// mlx-lm's two-rank sharded logits and greedy tokens, from `scripts/parity/muse_spark_tp.py`.
    ///
    /// Muse runs its branches in bf16, and a sharded layer sums bf16 partial products. On these
    /// tiny fixtures that rounding moves near-tied expert choices, so the sharded model is
    /// compared with mlx-lm's sharded model rather than with the unsharded one. GLM, GLM-5.3-Flash
    /// and DeepSeek-V4.1 run in float32 and are compared with the unsharded model. (mlx-lm's
    /// DeepSeek-V4.1 has no Engram, so it cannot give a sharded reference.)
    func shardedReference() throws -> (logits: MLXArray, greedy: [Int])? {
        let url = try ParityFixture(rawValue).directory
            .appending(component: "expected_tp2.safetensors")
        guard FileManager.default.fileExists(atPath: url.path) else { return nil }
        let arrays = try loadArrays(url: url)
        guard let logits = arrays["logits"], let greedy = arrays["greedy"] else {
            throw ParityFixtureError.missing("\(rawValue)/expected_tp2.safetensors")
        }
        return (logits, greedy[0].asArray(Int.self))
    }

    func loadModel(group: MLXDistributed.Group?) async throws -> any LanguageModel {
        let fixture = try ParityFixture(rawValue)
        guard let group else { return try await fixture.loadLanguageModel() }
        let data = try Data(contentsOf: fixture.directory.appending(component: "config.json"))
        let base = try JSONDecoder().decode(BaseConfiguration.self, from: data)
        let model = try await LLMTypeRegistry.shared.createModel(
            configuration: data, modelType: base.modelType)
        try await loadWeights(
            modelDirectory: fixture.directory, model: model, tensorParallelGroup: group,
            perLayerQuantization: base.perLayerQuantization)
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

    struct ParityReport: Codable {
        var fixture: TensorParallelParityFixture
        /// Against mlx-lm's sharded logits when the fixture has them, else the unsharded model.
        var maxLogitDifference: Float
        var unshardedLogitDifference: Float
        /// mlx-lm's sharded greedy tokens when the fixture has them, else the unsharded model's.
        var referenceGreedy: [Int]
        var shardedGreedy: [Int]
        var sampled: [Int]
        var mtp: MTPReport?
        var speculative: SpeculativeReport?
    }

    /// A sharded DSpark or GLM-5.3-Flash MTP drafter.
    struct SpeculativeReport: Codable {
        /// Sharded against unsharded draft logits for the same target hidden states.
        var maxDraftLogitDifference: Float
        var greedy: [Int]
        var proposed: Int
        var accepted: Int
        var passthroughReason: String?
        var generatorGreedy: [Int]
        var generatorSampled: [Int]
        /// With ``ScriptedContinuationDrafter``, so that rounds accept some drafts and the
        /// sharded target rewinds its caches.
        var scriptedGreedy: [Int]
        var scriptedProposed: Int
        var scriptedAccepted: Int
        var scriptedGeneratorSampled: [Int]
    }

    struct MTPReport: Codable {
        /// Sharded against unsharded MTP head logits on the target's hidden states.
        var maxHeadLogitDifference: Float
        var greedy: [Int]
        var proposed: Int
        var accepted: Int
        var passthroughReason: String?
        var generatorGreedy: [Int]
        var generatorSampled: [Int]
        /// With a head that agrees with the target, so that rounds accept drafts.
        var agreeingExpected: [Int]
        var agreeingGreedy: [Int]
        var agreeingProposed: Int
        var agreeingAccepted: Int
        var agreeingSampled: [Int]
    }

    var rank: Int
    var size: Int
    var fixtures: [FixtureReport]
    var parity: [ParityReport] = []
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

        for fixture in TensorParallelParityFixture.allCases {
            log("\(fixture): load")
            report.parity.append(try await runParity(fixture, group: group))
        }

        let url = URL(fileURLWithPath: resultDirectory)
            .appendingPathComponent("rank-\(group.rank).json")
        try JSONEncoder().encode(report).write(to: url)
    }
}

extension TensorParallelRankTests {
    /// Runs on every rank in the same order, since the sharded model communicates.
    fileprivate func runParity(
        _ fixture: TensorParallelParityFixture, group: MLXDistributed.Group
    ) async throws -> TensorParallelRankReport.ParityReport {
        let parity = try ParityFixture(fixture.rawValue)
        let reference = try await fixture.loadModel(group: nil)
        let sharded = try await fixture.loadModel(group: group)
        XCTAssertTrue((sharded as? any ShardableModel)?.tensorParallelGroup === group)

        let logits = sharded(parity.inputIDs, cache: nil).asType(.float32)
        let unshardedDifference = abs(
            reference(parity.inputIDs, cache: nil).asType(.float32) - logits
        ).max().item(Float.self)

        let prompt = parity.inputIDs[0].asArray(Int.self)
        var difference = unshardedDifference
        var referenceGreedy = [Int]()
        if let shardedReference = try fixture.shardedReference() {
            difference = abs(shardedReference.logits - logits).max().item(Float.self)
            referenceGreedy = shardedReference.greedy
        } else {
            var iterator = try TokenIterator(
                input: LMInput(tokens: MLXArray(prompt.map { Int32($0) })), model: reference,
                parameters: GenerateParameters(maxTokens: 12, temperature: 0))
            while let token = iterator.next() {
                referenceGreedy.append(token)
            }
        }

        let greedy = TensorParallelGenerationRequest(promptTokens: prompt, maxTokens: 12)
        let sampled = TensorParallelGenerationRequest(
            promptTokens: prompt, maxTokens: 12, temperature: 1)
        let (shardedGreedy, sampledTokens) = try generate(
            TensorParallelGenerator(model: sharded), greedy, sampled)

        var mtp: TensorParallelRankReport.MTPReport?
        if fixture.hasMTP, let target = sharded as? GLMMoEDSAModel,
            let referenceTarget = reference as? GLMMoEDSAModel
        {
            mtp = try await runMTP(
                parity, target: target, referenceTarget: referenceTarget, prompt: prompt,
                greedy: greedy, sampled: sampled, group: group)
        }
        var speculative: TensorParallelRankReport.SpeculativeReport?
        if fixture.hasDrafter {
            speculative = try await runDrafter(
                fixture, parity: parity, target: sharded, referenceTarget: reference,
                prompt: prompt, greedy: greedy, sampled: sampled, group: group)
        }
        return .init(
            fixture: fixture, maxLogitDifference: difference,
            unshardedLogitDifference: unshardedDifference, referenceGreedy: referenceGreedy,
            shardedGreedy: shardedGreedy, sampled: sampledTokens, mtp: mtp,
            speculative: speculative)
    }

    /// Loads the fixture's drafter sharded and not, compares their draft logits on the
    /// reference target's hidden states, then decodes speculatively on the sharded target.
    private func runDrafter(
        _ fixture: TensorParallelParityFixture, parity: ParityFixture, target: any LanguageModel,
        referenceTarget: any LanguageModel, prompt: [Int], greedy: TensorParallelGenerationRequest,
        sampled: TensorParallelGenerationRequest, group: MLXDistributed.Group
    ) async throws -> TensorParallelRankReport.SpeculativeReport {
        let drafter: any MTPDrafterModel
        let draftDifference: Float
        switch fixture {
        case .deepseekV41:
            let directory = parity.directory.deletingLastPathComponent()
                .appending(component: "deepseek_v41_dspark")
            let loadedReference = try await DeepseekV41DSparkDraftModel.load(directory: directory)
            let loadedSharded = try await DeepseekV41DSparkDraftModel.load(
                directory: directory, tensorParallelGroup: group)
            let reference = try XCTUnwrap(loadedReference)
            let sharded = try XCTUnwrap(loadedSharded)
            XCTAssertTrue(sharded.tensorParallelGroup === group)
            let referenceTarget = try XCTUnwrap(referenceTarget as? DeepseekV41Model)
            var request = LMOutput.State()
            request[mtpEmitFlagKey] = true
            request[mtpHiddenLayerIDsKey] = reference.targetHiddenLayerIDs
            let hidden = try XCTUnwrap(
                referenceTarget(LMInput.Text(tokens: parity.inputIDs), cache: nil, state: request)
                    .state?[mtpLastHiddenStatesKey])
            func draftLogits(_ drafter: DeepseekV41DSparkDraftModel) -> MLXArray {
                var state = drafter.makeState(parameters: nil)
                drafter.prepareDrafterChunk(
                    target: referenceTarget, shiftedTokens: parity.inputIDs, targetHidden: hidden,
                    isFinal: true, state: &state, sampler: ArgMaxSampler())
                return drafter.draftBlock(
                    target: referenceTarget, lastToken: MLXArray([Int32(1)]),
                    lastHidden: hidden[0..., (-1)...], sharedKV: [:], positionDeltas: nil,
                    queryOffset: hidden.dim(1), blockSize: 6, state: &state,
                    sampler: ArgMaxSampler()
                ).logits
            }
            draftDifference = abs(draftLogits(reference) - draftLogits(sharded)).max()
                .item(Float.self)
            drafter = sharded
        case .glm5Next:
            let loadedReference = try await GLM5NextMTPDraftModel.load(
                directory: parity.directory)
            let loadedSharded = try await GLM5NextMTPDraftModel.load(
                directory: parity.directory, tensorParallelGroup: group)
            let reference = try XCTUnwrap(loadedReference)
            let sharded = try XCTUnwrap(loadedSharded)
            XCTAssertTrue(sharded.tensorParallelGroup === group)
            let referenceTarget = try XCTUnwrap(referenceTarget as? GLM5NextModel)
            var state = LMOutput.State()
            state[mtpEmitFlagKey] = true
            let hidden = try XCTUnwrap(
                referenceTarget(LMInput.Text(tokens: parity.inputIDs), cache: nil, state: state)
                    .state?[mtpLastHiddenStatesKey])
            let embeddings = referenceTarget.mtpEmbedding(parity.inputIDs)
            func headLogits(_ drafter: GLM5NextMTPDraftModel) -> MLXArray {
                referenceTarget.lmHead(
                    drafter(inputEmbeddings: embeddings, hidden: hidden, cache: KVCacheSimple()))
            }
            draftDifference = abs(headLogits(reference) - headLogits(sharded)).max()
                .item(Float.self)
            drafter = sharded
        default:
            preconditionFailure("\(fixture) has no drafter")
        }

        var speculative = try MTPSpeculativeTokenIterator(
            input: LMInput(tokens: MLXArray(prompt.map { Int32($0) })), mainModel: target,
            drafter: drafter, parameters: greedy.generateParameters, blockSize: 3)
        var speculativeGreedy = [Int]()
        while let token = speculative.next() {
            speculativeGreedy.append(token)
        }
        let (generatorGreedy, generatorSampled) = try generate(
            TensorParallelGenerator(model: target, drafter: drafter, draftBlockSize: 3),
            greedy, sampled)

        // The unsharded greedy continuation, with every fourth draft wrong.
        var baseline = try TokenIterator(
            input: LMInput(tokens: MLXArray(prompt.map { Int32($0) })), model: referenceTarget,
            parameters: greedy.generateParameters)
        var continuation = [Int]()
        while let token = baseline.next() {
            continuation.append(token)
        }
        let scripted = ScriptedContinuationDrafter(
            promptLength: prompt.count, continuation: continuation,
            vocabularySize: parity.logits.dim(-1))
        var scriptedIterator = try MTPSpeculativeTokenIterator(
            input: LMInput(tokens: MLXArray(prompt.map { Int32($0) })), mainModel: target,
            drafter: scripted, parameters: greedy.generateParameters, blockSize: 3)
        var scriptedGreedy = [Int]()
        while let token = scriptedIterator.next() {
            scriptedGreedy.append(token)
        }
        let (_, scriptedSampled) = try generate(
            TensorParallelGenerator(model: target, drafter: scripted, draftBlockSize: 3),
            greedy, sampled)
        return .init(
            maxDraftLogitDifference: draftDifference, greedy: speculativeGreedy,
            proposed: speculative.proposedCount, accepted: speculative.acceptedCount,
            passthroughReason: speculative.passthroughReason, generatorGreedy: generatorGreedy,
            generatorSampled: generatorSampled, scriptedGreedy: scriptedGreedy,
            scriptedProposed: scriptedIterator.proposedCount,
            scriptedAccepted: scriptedIterator.acceptedCount,
            scriptedGeneratorSampled: scriptedSampled)
    }

    private func runMTP(
        _ parity: ParityFixture, target: GLMMoEDSAModel, referenceTarget: GLMMoEDSAModel,
        prompt: [Int], greedy: TensorParallelGenerationRequest,
        sampled: TensorParallelGenerationRequest, group: MLXDistributed.Group
    ) async throws -> TensorParallelRankReport.MTPReport {
        let loadedReference = try await GLMMoEDSAMTPDraftModel.load(directory: parity.directory)
        let referenceDrafter = try XCTUnwrap(loadedReference)
        let loaded = try await GLMMoEDSAMTPDraftModel.load(
            directory: parity.directory, tensorParallelGroup: group)
        let drafter = try XCTUnwrap(loaded)
        XCTAssertTrue(drafter.tensorParallelGroup === group)

        // The head on the same hidden states, sharded and not.
        var state = LMOutput.State()
        state[mtpEmitFlagKey] = true
        let hidden = try XCTUnwrap(
            referenceTarget(LMInput.Text(tokens: parity.inputIDs), cache: nil, state: state)
                .state?[mtpLastHiddenStatesKey])
        let embeddings = referenceTarget.mtpEmbedding(parity.inputIDs)
        let expected = referenceTarget.lmHead(
            referenceDrafter(inputEmbeddings: embeddings, hidden: hidden, cache: KVCacheSimple()))
        let actual = target.lmHead(
            drafter(inputEmbeddings: embeddings, hidden: hidden, cache: KVCacheSimple()))
        let headDifference = abs(expected - actual).max().item(Float.self)

        var speculative = try MTPSpeculativeTokenIterator(
            input: LMInput(tokens: MLXArray(prompt.map { Int32($0) })), mainModel: target,
            drafter: drafter, parameters: greedy.generateParameters, blockSize: 3)
        var speculativeGreedy = [Int]()
        while let token = speculative.next() {
            speculativeGreedy.append(token)
        }

        let (generatorGreedy, generatorSampled) = try generate(
            TensorParallelGenerator(model: target, drafter: drafter, draftBlockSize: 3),
            greedy, sampled)

        makeBigram(referenceTarget, drafter: referenceDrafter)
        makeBigram(target, drafter: drafter)
        var baseline = try TokenIterator(
            input: LMInput(tokens: MLXArray(prompt.map { Int32($0) })), model: referenceTarget,
            parameters: greedy.generateParameters)
        var agreeingExpected = [Int]()
        while let token = baseline.next() {
            agreeingExpected.append(token)
        }
        var agreeing = try MTPSpeculativeTokenIterator(
            input: LMInput(tokens: MLXArray(prompt.map { Int32($0) })), mainModel: target,
            drafter: drafter, parameters: greedy.generateParameters, blockSize: 4)
        var agreeingGreedy = [Int]()
        while let token = agreeing.next() {
            agreeingGreedy.append(token)
        }
        let (_, agreeingSampled) = try generate(
            TensorParallelGenerator(model: target, drafter: drafter, draftBlockSize: 4),
            greedy, sampled)

        return .init(
            maxHeadLogitDifference: headDifference, greedy: speculativeGreedy,
            proposed: speculative.proposedCount, accepted: speculative.acceptedCount,
            passthroughReason: speculative.passthroughReason, generatorGreedy: generatorGreedy,
            generatorSampled: generatorSampled, agreeingExpected: agreeingExpected,
            agreeingGreedy: agreeingGreedy, agreeingProposed: agreeing.proposedCount,
            agreeingAccepted: agreeing.acceptedCount, agreeingSampled: agreeingSampled)
    }

    /// As `GLMMoEDSAMTPTests`: the target predicts from the current token only, and the MTP
    /// head becomes the same function of its input token.
    private func makeBigram(_ model: GLMMoEDSAModel, drafter: GLMMoEDSAMTPDraftModel) {
        func zeroOutputs(_ module: Module) {
            let zeroed = module.parameters().flattened().compactMap { key, value in
                key.hasSuffix("o_proj.weight") || key.hasSuffix("down_proj.weight")
                    ? (key, MLXArray.zeros(like: value)) : nil
            }
            module.update(parameters: ModuleParameters.unflattened(zeroed))
        }
        zeroOutputs(model)
        zeroOutputs(drafter)
        let hidden = model.configuration.hiddenSize
        drafter.update(
            parameters: ModuleParameters.unflattened([
                "eh_proj.weight": concatenated(
                    [MLXArray.eye(hidden), MLXArray.zeros([hidden, hidden])], axis: 1),
                "enorm.weight": MLXArray.ones([hidden]),
                "shared_head.norm.weight": model.model.norm.weight,
            ]))
        eval(model, drafter)
    }

    /// Rank 0 leads both requests; the others follow.
    private func generate(
        _ generator: TensorParallelGenerator, _ first: TensorParallelGenerationRequest,
        _ second: TensorParallelGenerationRequest
    ) throws -> ([Int], [Int]) {
        if generator.isLeader {
            let tokens = (try generator.generate(first), try generator.generate(second))
            try generator.shutdown()
            return tokens
        }
        var results = [[Int]]()
        try generator.followRequests { _, tokens in results.append(tokens) }
        XCTAssertEqual(results.count, 2)
        return (results.first ?? [], results.last ?? [])
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

    /// Shards the parity fixtures across a group of one, which must change nothing.
    func testParityFixturesShardAcrossOneRank() async throws {
        let group = try MLXDistributed.initialize(backend: .ring, strict: false)
        XCTAssertEqual(group.size, 1)

        for fixture in TensorParallelParityFixture.allCases {
            let parity = try ParityFixture(fixture.rawValue)
            let reference = try await fixture.loadModel(group: nil)
            let sharded = try await fixture.loadModel(group: nil)
            try XCTUnwrap(sharded as? any ShardableModel).shard(group: group)
            XCTAssertTrue(
                allClose(
                    reference(parity.inputIDs, cache: nil), sharded(parity.inputIDs, cache: nil)
                )
                .item(Bool.self),
                "\(fixture)")
        }

        let parity = try ParityFixture(TensorParallelParityFixture.glmMoEDSA.rawValue)
        let loadedDrafter = try await GLMMoEDSAMTPDraftModel.load(directory: parity.directory)
        let loadedReference = try await GLMMoEDSAMTPDraftModel.load(directory: parity.directory)
        let drafter = try XCTUnwrap(loadedDrafter)
        let reference = try XCTUnwrap(loadedReference)
        try drafter.shard(group: group)
        let tokens = MLXArray.zeros([1, 4, 32])
        XCTAssertTrue(
            allClose(
                reference(inputEmbeddings: tokens, hidden: tokens + 1, cache: KVCacheSimple()),
                drafter(inputEmbeddings: tokens, hidden: tokens + 1, cache: KVCacheSimple())
            ).item(Bool.self))
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
    /// localhost, then compares what each rank saw. Muse fixtures compare with mlx-lm's sharded
    /// model instead of the unsharded one (see ``TensorParallelParityFixture``).
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

        XCTAssertEqual(
            reports.map { $0.parity.map(\.fixture) },
            Array(repeating: TensorParallelParityFixture.allCases, count: 2))
        for (leader, follower) in zip(reports[0].parity, reports[1].parity) {
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
                    + "rank1=\(follower.maxLogitDifference) "
                    + "(unsharded \(leader.unshardedLogitDifference)) "
                    + "greedy=\(leader.shardedGreedy) sampled=\(leader.sampled)")

            if leader.fixture.hasDrafter {
                let speculative = try XCTUnwrap(leader.speculative, name)
                let followerSpeculative = try XCTUnwrap(follower.speculative, name)
                for report in [speculative, followerSpeculative] {
                    XCTAssertLessThan(report.maxDraftLogitDifference, 1e-4, name)
                    XCTAssertEqual(report.greedy, leader.referenceGreedy, name)
                    XCTAssertEqual(report.generatorGreedy, leader.referenceGreedy, name)
                    XCTAssertNil(report.passthroughReason, name)
                    XCTAssertGreaterThan(report.proposed, 0, name)
                    XCTAssertEqual(report.scriptedGreedy, leader.referenceGreedy, name)
                    XCTAssertGreaterThan(report.scriptedAccepted, 0, name)
                    XCTAssertLessThan(report.scriptedAccepted, report.scriptedProposed, name)
                }
                XCTAssertEqual(followerSpeculative.accepted, speculative.accepted, name)
                XCTAssertEqual(
                    followerSpeculative.scriptedAccepted, speculative.scriptedAccepted, name)
                XCTAssertEqual(speculative.scriptedGeneratorSampled.count, 12, name)
                XCTAssertEqual(
                    followerSpeculative.scriptedGeneratorSampled,
                    speculative.scriptedGeneratorSampled, name)
                XCTAssertEqual(followerSpeculative.proposed, speculative.proposed, name)
                XCTAssertEqual(speculative.generatorSampled.count, 12, name)
                XCTAssertEqual(
                    followerSpeculative.generatorSampled, speculative.generatorSampled, name)
                print(
                    "tensor parallel \(name) drafter: draft |logit diff| "
                        + "rank0=\(speculative.maxDraftLogitDifference) "
                        + "rank1=\(followerSpeculative.maxDraftLogitDifference) "
                        + "greedy=\(speculative.greedy) "
                        + "accepted \(speculative.accepted)/\(speculative.proposed) "
                        + "sampled=\(speculative.generatorSampled); scripted accepted "
                        + "\(speculative.scriptedAccepted)/\(speculative.scriptedProposed) "
                        + "sampled=\(speculative.scriptedGeneratorSampled)")
            }

            guard leader.fixture.hasMTP else { continue }
            let mtp = try XCTUnwrap(leader.mtp, name)
            let followerMTP = try XCTUnwrap(follower.mtp, name)
            for report in [mtp, followerMTP] {
                XCTAssertLessThan(report.maxHeadLogitDifference, 1e-4, name)
                XCTAssertEqual(report.greedy, leader.referenceGreedy, name)
                XCTAssertEqual(report.generatorGreedy, leader.referenceGreedy, name)
                XCTAssertNil(report.passthroughReason, name)
                XCTAssertGreaterThan(report.proposed, 0, name)
            }
            XCTAssertEqual(followerMTP.accepted, mtp.accepted, name)
            XCTAssertEqual(followerMTP.proposed, mtp.proposed, name)
            XCTAssertEqual(mtp.generatorSampled.count, 12, name)
            XCTAssertEqual(followerMTP.generatorSampled, mtp.generatorSampled, name)
            for report in [mtp, followerMTP] {
                XCTAssertEqual(report.agreeingGreedy, report.agreeingExpected, name)
                XCTAssertGreaterThanOrEqual(
                    report.agreeingAccepted * 10, report.agreeingProposed * 9)
            }
            XCTAssertEqual(followerMTP.agreeingAccepted, mtp.agreeingAccepted, name)
            XCTAssertEqual(mtp.agreeingSampled.count, 12, name)
            XCTAssertEqual(followerMTP.agreeingSampled, mtp.agreeingSampled, name)
            print(
                "tensor parallel \(name) MTP: head |logit diff| rank0=\(mtp.maxHeadLogitDifference) "
                    + "rank1=\(followerMTP.maxHeadLogitDifference) greedy=\(mtp.greedy) "
                    + "accepted \(mtp.accepted)/\(mtp.proposed) sampled=\(mtp.generatorSampled); "
                    + "agreeing head accepted \(mtp.agreeingAccepted)/\(mtp.agreeingProposed) "
                    + "sampled=\(mtp.agreeingSampled)")
        }
    }
}
