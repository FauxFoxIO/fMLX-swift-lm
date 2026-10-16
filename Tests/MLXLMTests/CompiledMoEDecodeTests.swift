// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXNN
import XCTest

@testable import MLXLLM
@testable import MLXLMCommon

/// Compiled batch-1 decode for MoE decoder stacks must match the eager path.
final class CompiledMoEDecodeTests: XCTestCase {
    private struct Snapshot: Equatable {
        let shape: [Int]
        let dtype: DType
        let bytes: Data

        init(_ array: MLXArray) {
            let contents = array.asData(access: .copy)
            self.shape = contents.shape
            self.dtype = contents.dType
            self.bytes = contents.data
        }
    }

    private struct Run: Equatable {
        let logits: [Snapshot]
        let cacheState: [Snapshot]
        let offsets: [Int]
    }

    private typealias Model = Module & LanguageModel & CompiledDecodeConfigurable

    // Dimensions are multiples of 64 so every projection quantizes.
    private func qwen3MoE() throws -> Qwen3MoEModel {
        let json = """
            {
                "model_type": "qwen3_moe", "hidden_size": 128, "num_hidden_layers": 3,
                "intermediate_size": 128, "num_attention_heads": 2, "num_key_value_heads": 1,
                "head_dim": 64, "num_experts": 8, "num_experts_per_tok": 2,
                "decoder_sparse_step": 1, "mlp_only_layers": [0], "moe_intermediate_size": 64,
                "rms_norm_eps": 1e-6, "vocab_size": 64, "rope_theta": 10000.0,
                "norm_topk_prob": true, "tie_word_embeddings": false
            }
            """
        return Qwen3MoEModel(
            try JSONDecoder().decode(Qwen3MoEConfiguration.self, from: Data(json.utf8)))
    }

    private func deepseekV3() throws -> DeepseekV3Model {
        let json = """
            {
                "model_type": "deepseek_v3", "vocab_size": 64, "hidden_size": 128,
                "intermediate_size": 128, "moe_intermediate_size": 64,
                "num_hidden_layers": 3, "num_attention_heads": 2, "num_key_value_heads": 2,
                "n_routed_experts": 8, "n_shared_experts": 1, "num_experts_per_tok": 2,
                "n_group": 2, "topk_group": 1, "norm_topk_prob": true,
                "routed_scaling_factor": 2.5, "first_k_dense_replace": 1, "moe_layer_freq": 1,
                "q_lora_rank": 64, "kv_lora_rank": 64, "qk_rope_head_dim": 32,
                "qk_nope_head_dim": 32, "v_head_dim": 64, "rms_norm_eps": 1e-6,
                "rope_theta": 10000.0, "max_position_embeddings": 256, "attention_bias": false
            }
            """
        return DeepseekV3Model(
            try JSONDecoder().decode(DeepseekV3Configuration.self, from: Data(json.utf8)))
    }

    private func glm4MoE() throws -> GLM4MoEModel {
        let json = """
            {
                "model_type": "glm4_moe", "vocab_size": 64, "hidden_size": 128,
                "intermediate_size": 128, "max_position_embeddings": 256,
                "moe_intermediate_size": 64, "norm_topk_prob": true,
                "num_attention_heads": 2, "n_group": 1, "head_dim": 64, "topk_group": 1,
                "n_shared_experts": 1, "n_routed_experts": 8, "routed_scaling_factor": 1.0,
                "num_experts_per_tok": 2, "first_k_dense_replace": 1, "num_hidden_layers": 3,
                "num_key_value_heads": 1, "rms_norm_eps": 1e-5, "rope_theta": 10000.0,
                "use_qk_norm": true, "tie_word_embeddings": false, "attention_bias": true,
                "partial_rotary_factor": 0.5, "scoring_func": "sigmoid", "topk_method": "noaux_tc"
            }
            """
        return GLM4MoEModel(
            try JSONDecoder().decode(GLM4MoEConfiguration.self, from: Data(json.utf8)))
    }

    private func models() throws -> [(String, () throws -> Model)] {
        [("qwen3_moe", qwen3MoE), ("deepseek_v3", deepseekV3), ("glm4_moe", glm4MoE)]
    }

    /// Random bfloat16 weights, quantized like a 4-bit checkpoint, then prepared.
    private func materialize(_ model: Model, seed: UInt64, quantized: Bool) throws {
        MLXRandom.seed(seed)
        let random = model.parameters().mapValues { value in
            (MLXRandom.normal(value.shape) * 0.1).asType(.bfloat16)
        }
        model.update(parameters: random)
        if quantized {
            quantize(model: model, groupSize: 64, bits: 4)
        }
        eval(model)
        try model.prepare()
    }

    private func run(
        _ model: Model, compiled: Bool, batch: Int = 1,
        cache makeCache: (Model) throws -> [KVCache] = { try $0.newCache(parameters: nil) }
    ) throws -> Run {
        model.compiledDecodeEnabled = compiled
        let cache = try makeCache(model)
        let prompt = MLXArray((0 ..< 5 * batch).map { Int32($0 * 7 % 64) }).reshaped(batch, 5)
        var logits = [model(prompt, cache: cache)]
        var next = logits[0][0..., -1, 0...].argMax(axis: -1).reshaped(batch, 1)
        for _ in 0 ..< 6 {
            let output = model(next, cache: cache)
            logits.append(output)
            next = output[0..., -1, 0...].argMax(axis: -1).reshaped(batch, 1)
        }
        eval(logits)
        return Run(
            logits: logits.map(Snapshot.init),
            cacheState: cache.flatMap(\.state).map(Snapshot.init),
            offsets: cache.map(\.offset))
    }

    private func compiledSegments(_ model: Model) -> Int {
        switch model {
        case let model as Qwen3MoEModel: model.compiledDecode.compiledCount
        case let model as DeepseekV3Model: model.compiledDecode.compiledCount
        case let model as GLM4MoEModel: model.compiledDecode.compiledCount
        default: 0
        }
    }

    func testCompiledDecodeMatchesEagerExactly() throws {
        for (name, make) in try models() {
            for quantized in [true, false] {
                let model = try make()
                try materialize(model, seed: 5, quantized: quantized)
                let eager = try run(model, compiled: false)
                XCTAssertEqual(compiledSegments(model), 0, name)
                let compiled = try run(model, compiled: true)
                XCTAssertEqual(compiledSegments(model), 4, "\(name): one trace per segment")
                XCTAssertEqual(compiled, eager, "\(name) quantized=\(quantized)")
            }
        }
    }

    func testCompiledDecodeSupportsOtherCaches() throws {
        let caches: [(String, (Model) throws -> [KVCache])] = [
            ("rotating", { _ in (0 ..< 3).map { _ in RotatingKVCache(maxSize: 8) } }),
            ("quantized", { _ in (0 ..< 3).map { _ in QuantizedKVCache(groupSize: 32) } }),
        ]
        for (name, make) in try models() {
            let model = try make()
            try materialize(model, seed: 9, quantized: true)
            for (cacheName, makeCache) in caches {
                let eager = try run(model, compiled: false, cache: makeCache)
                let compiled = try run(model, compiled: true, cache: makeCache)
                XCTAssertEqual(compiled, eager, "\(name) with \(cacheName) cache")
            }
        }
    }

    func testBatchedDecodeFallsBackToEager() throws {
        for (name, make) in try models() {
            let model = try make()
            try materialize(model, seed: 13, quantized: true)
            let eager = try run(model, compiled: false, batch: 2)
            let compiled = try run(model, compiled: true, batch: 2)
            XCTAssertEqual(compiledSegments(model), 0, "\(name): batch 2 stays eager")
            XCTAssertEqual(compiled, eager, name)
        }
    }

    /// `TokenIterator` (compiled, raw-buffer cache evaluation) emits the
    /// same greedy tokens as a plain eager loop.
    func testTokenIteratorMatchesEagerGreedyLoop() throws {
        for (name, make) in try models() {
            let model = try make()
            try materialize(model, seed: 17, quantized: true)
            let prompt = MLXArray((0 ..< 7).map { Int32($0 * 5 % 64) })

            model.compiledDecodeEnabled = false
            let cache = try model.newCache(parameters: nil)
            var logits = model(prompt.reshaped(1, -1), cache: cache)
            var expected: [Int] = []
            for _ in 0 ..< 12 {
                let token = logits[0..., -1, 0...].argMax(axis: -1)
                expected.append(token.item(Int.self))
                logits = model(token.reshaped(1, 1), cache: cache)
            }

            model.compiledDecodeEnabled = true
            var iterator = try TokenIterator(
                input: LMInput(tokens: prompt), model: model,
                parameters: GenerateParameters(maxTokens: 12, temperature: 0))
            var actual: [Int] = []
            while let token = iterator.next() { actual.append(token) }
            XCTAssertEqual(actual, expected, name)
            XCTAssertGreaterThan(compiledSegments(model), 0, name)
        }
    }
}
