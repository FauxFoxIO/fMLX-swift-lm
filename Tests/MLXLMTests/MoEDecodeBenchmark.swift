// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXNN
import XCTest

@testable import MLXLLM
@testable import MLXLMCommon

/// Opt-in batch-1 decode benchmark for MoE models with random 4-bit weights.
///
/// Run with `TEST_RUNNER_FMLX_MOE_DECODE_BENCH=1` (xcodebuild strips the
/// prefix) in a Release build with testability enabled. Optional settings:
/// - `FMLX_MOE_DECODE_BENCH_PRESET`: `qwen3moe` (Qwen3-30B-A3B dims, default),
///   `qwen3moe-large` (hidden 4096, 128 experts top-8, expert dim 1536),
///   `qwen3moe-tiny` (the same graph at tiny widths, so decode is CPU-bound),
///   `glm4moe` (GLM-4.5-Air dims), or `deepseekv3` (reduced DeepSeek-V3 dims).
/// - `FMLX_MOE_DECODE_BENCH_LAYERS`: decoder layer count override.
/// - `FMLX_MOE_DECODE_BENCH_TOKENS`: timed decode tokens per round (default 128).
/// - `FMLX_MOE_DECODE_BENCH_ROUNDS`: interleaved rounds per arm (default 5).
/// - `FMLX_MOE_DECODE_BENCH_ARMS`: comma-separated arms from `base` (eager)
///   and `compiled` (default `base,compiled`).
///
/// Arms alternate within one process so they share machine load. Each arm
/// reports median tok/s through `TokenIterator`, the median CPU time to build
/// one decode step's graph (forward plus argmax, before evaluation), and the
/// median unpipelined step time (build plus evaluation).
final class MoEDecodeBenchmark: XCTestCase {
    private var environment: [String: String] { ProcessInfo.processInfo.environment }

    private struct Arm {
        let name: String
        let compiled: Bool
    }

    func testDecodeThroughput() throws {
        guard environment["FMLX_MOE_DECODE_BENCH"] == "1" else {
            throw XCTSkip("Set TEST_RUNNER_FMLX_MOE_DECODE_BENCH=1 to benchmark")
        }
        let preset = environment["FMLX_MOE_DECODE_BENCH_PRESET"] ?? "qwen3moe"
        let layers = environment["FMLX_MOE_DECODE_BENCH_LAYERS"].flatMap(Int.init)
        let tokens = environment["FMLX_MOE_DECODE_BENCH_TOKENS"].flatMap(Int.init) ?? 128
        let rounds = environment["FMLX_MOE_DECODE_BENCH_ROUNDS"].flatMap(Int.init) ?? 5
        let armNames = (environment["FMLX_MOE_DECODE_BENCH_ARMS"] ?? "base,compiled")
            .split(separator: ",").map(String.init)
        let label = "[MOEDECODEBENCH] \(preset) layers=\(layers.map(String.init) ?? "default")"

        MLXRandom.seed(0)
        let model = try Self.makeModel(preset: preset, layers: layers)
        print("\(label) resident \(Memory.activeMemory / 1_048_576) MiB")
        let arms = armNames.map { Arm(name: $0, compiled: $0 == "compiled") }

        let prompt = MLXArray((0 ..< 32).map { Int32(($0 * 7919) % 1000 + 10) })
        var rates = [[Double]](repeating: [], count: arms.count)
        var greedy = [[Int]](repeating: [], count: arms.count)
        var builds = [[Double]](repeating: [], count: arms.count)
        var steps = [[Double]](repeating: [], count: arms.count)
        for round in 0 ..< rounds {
            let order =
                round.isMultiple(of: 2)
                ? Array(arms.indices) : Array(arms.indices.reversed())
            for index in order {
                let arm = arms[index]
                Self.setCompiledDecode(model, arm.compiled)
                MLXRandom.seed(1)
                // Greedy on the first round to compare arms, sampled afterwards
                // so routing varies like real text.
                let parameters = GenerateParameters(
                    maxTokens: tokens + 8, temperature: round == 0 ? 0 : 1)
                var iterator = try TokenIterator(
                    input: LMInput(tokens: prompt), model: model, parameters: parameters)
                var generated: [Int] = []
                for _ in 0 ..< 8 { generated.append(iterator.next()!) }
                let start = Date.timeIntervalSinceReferenceDate
                for _ in 0 ..< tokens { generated.append(iterator.next()!) }
                let elapsed = Date.timeIntervalSinceReferenceDate - start
                rates[index].append(Double(tokens) / elapsed)
                if round == 0 { greedy[index] = generated }
                let timing = try Self.stepMicroseconds(model, prompt: prompt)
                builds[index] += timing.build
                steps[index] += timing.step
            }
        }

        for (index, arm) in arms.enumerated() {
            let sorted = rates[index].sorted()
            let median = sorted[sorted.count / 2]
            let build = builds[index].sorted()
            let step = steps[index].sorted()
            print(
                String(
                    format:
                        "%@ arm %@: %.2f tok/s median (%.3f ms/tok; min %.2f max %.2f) | graph build median %.0f min %.0f µs/tok | synchronous step median %.0f min %.0f µs/tok",
                    label, arm.name, median, 1000 / median, sorted.first!, sorted.last!,
                    build[build.count / 2], build[0], step[step.count / 2], step[0]))
            let matches = greedy[index] == greedy[0]
            print("\(label) arm \(arm.name) greedy tokens match first arm: \(matches)")
        }
    }

    /// Run unpipelined decode steps, timing each graph construction and each
    /// whole step (construction plus evaluation).
    private static func stepMicroseconds(
        _ model: any LanguageModel, prompt: MLXArray
    ) throws -> (build: [Double], step: [Double]) {
        let cache = try model.newCache(parameters: nil)
        eval(model(prompt.reshaped(1, -1), cache: cache))
        var y = MLXArray([Int32(11)]).reshaped(1, 1)
        var builds: [Double] = []
        var steps: [Double] = []
        for step in 0 ..< 24 {
            let start = Date.timeIntervalSinceReferenceDate
            let next = model(y, cache: cache)[0..., -1, 0...].argMax(axis: -1)
            let built = Date.timeIntervalSinceReferenceDate - start
            eval([next] + cache.flatMap { $0.state })
            let done = Date.timeIntervalSinceReferenceDate - start
            if step >= 4 {
                builds.append(built * 1_000_000)
                steps.append(done * 1_000_000)
            }
            y = next.reshaped(1, 1)
        }
        return (builds, steps)
    }

    private static func setCompiledDecode(_ model: any LanguageModel, _ enabled: Bool) {
        (model as? CompiledDecodeConfigurable)?.compiledDecodeEnabled = enabled
    }

    /// CPU cost of collecting cache arrays for the per-token `asyncEval`.
    func testCacheEvaluationArrayCost() throws {
        guard environment["FMLX_MOE_DECODE_BENCH"] == "1" else {
            throw XCTSkip("Set TEST_RUNNER_FMLX_MOE_DECODE_BENCH=1 to benchmark")
        }
        let caches: [KVCache] = (0 ..< 48).map { _ in
            let cache = KVCacheSimple()
            _ = cache.update(
                keys: MLXArray.zeros([1, 4, 300, 128], dtype: .bfloat16),
                values: MLXArray.zeros([1, 4, 300, 128], dtype: .bfloat16))
            return cache
        }
        eval(caches.flatMap { $0.innerState() })
        let iterations = 2_000
        var samples = [[Double]](repeating: [], count: 2)
        for round in 0 ..< 10 {
            for arm in (round.isMultiple(of: 2) ? [0, 1] : [1, 0]) {
                let start = Date.timeIntervalSinceReferenceDate
                var count = 0
                for _ in 0 ..< iterations {
                    let arrays =
                        arm == 0
                        ? caches.flatMap { $0.state }
                        : caches.flatMap { decodeEvaluationArrays($0) }
                    count += arrays.count
                }
                XCTAssertEqual(count, iterations * 96)
                samples[arm].append(
                    (Date.timeIntervalSinceReferenceDate - start) * 1_000_000 / Double(iterations))
            }
        }
        let medians = samples.map { $0.sorted()[5] }
        print(
            String(
                format:
                    "[MOEDECODEBENCH] 48-layer cache arrays: state %.1f µs/token, raw %.1f µs/token",
                medians[0], medians[1]))
    }

    /// Separate gate and up gathered matvecs against one fused gather on the
    /// concatenated weight, at decode shapes. On an M4 Max the fused gather
    /// measured within noise of the separate pair, so `SwitchGLU` keeps them
    /// separate.
    func testGateUpGatherKernels() throws {
        guard environment["FMLX_MOE_DECODE_BENCH"] == "1" else {
            throw XCTSkip("Set TEST_RUNNER_FMLX_MOE_DECODE_BENCH=1 to benchmark")
        }
        let shapes = [(2048, 768), (4096, 1536), (4096, 1408)]
        for (inputDims, hiddenDims) in shapes {
            let layers = 16
            let experts = 128
            MLXRandom.seed(3)
            var gates: [(MLXArray, MLXArray, MLXArray?)] = []
            var ups: [(MLXArray, MLXArray, MLXArray?)] = []
            var fused: [(MLXArray, MLXArray, MLXArray?)] = []
            var stacked: [(MLXArray, MLXArray, MLXArray?)] = []
            for _ in 0 ..< layers {
                let g = MLX.quantized(
                    MLXRandom.normal([experts, hiddenDims, inputDims]).asType(.bfloat16),
                    groupSize: 64, bits: 4)
                let u = MLX.quantized(
                    MLXRandom.normal([experts, hiddenDims, inputDims]).asType(.bfloat16),
                    groupSize: 64, bits: 4)
                let f = (
                    concatenated([g.wq, u.wq], axis: 1),
                    concatenated([g.scales, u.scales], axis: 1),
                    concatenated([g.biases!, u.biases!], axis: 1)
                )
                let st = (
                    concatenated([g.wq, u.wq], axis: 0),
                    concatenated([g.scales, u.scales], axis: 0),
                    concatenated([g.biases!, u.biases!], axis: 0)
                )
                eval(g.wq, g.scales, g.biases!, u.wq, u.scales, u.biases!, f.0, f.1, f.2)
                eval(st.0, st.1, st.2)
                stacked.append(st)
                gates.append((g.wq, g.scales, g.biases))
                ups.append((u.wq, u.scales, u.biases))
                fused.append((f.0, f.1, f.2))
            }
            let x = MLXRandom.normal([1, 1, 1, inputDims]).asType(.bfloat16)
            let indices = (0 ..< layers).map { _ in
                MLXRandom.randInt(0 ..< experts, [1, 8]).asType(.uint32)
            }
            eval([x] + indices)

            func gather(_ x: MLXArray, _ w: (MLXArray, MLXArray, MLXArray?), _ i: MLXArray)
                -> MLXArray
            {
                MLX.gatherQuantizedMM(
                    x, w.0, scales: w.1, biases: w.2, rhsIndices: i, transpose: true,
                    groupSize: 64, bits: 4, mode: .affine, sortedIndices: false)
            }
            let separate = {
                (0 ..< layers).map { l in
                    compiledSiluProduct(
                        gather(x, gates[l], indices[l]), gather(x, ups[l], indices[l]))
                }
            }
            let combined = {
                (0 ..< layers).map { l in
                    let gateUp = gather(x, fused[l], indices[l])
                    return compiledSiluProduct(
                        gateUp[.ellipsis, ..<hiddenDims], gateUp[.ellipsis, hiddenDims...])
                }
            }
            let xs = MLX.expandedDimensions(x, axis: 0)
            let stackedBody = {
                (0 ..< layers).map { l in
                    let both = concatenated(
                        [indices[l], indices[l] + UInt32(experts)], axis: -1)
                    let out = gather(xs, stacked[l], both)
                    return compiledSiluProduct(
                        out[.ellipsis, ..<8, 0..., 0...], out[.ellipsis, 8..., 0..., 0...])
                }
            }
            let a = separate()
            let b = combined()
            let c = stackedBody()
            eval(a + b + c)
            for l in 0 ..< layers {
                XCTAssertEqual(
                    a[l].asType(.float32).asArray(Float.self),
                    b[l].asType(.float32).asArray(Float.self))
                XCTAssertEqual(
                    a[l].asType(.float32).asArray(Float.self),
                    c[l].asType(.float32).asArray(Float.self))
            }
            let views = {
                (0 ..< layers).map { l in
                    let f = fused[l]
                    let g = (
                        f.0[0..., ..<hiddenDims], f.1[0..., ..<hiddenDims],
                        f.2?[0..., ..<hiddenDims]
                    )
                    let u = (
                        f.0[0..., hiddenDims...], f.1[0..., hiddenDims...],
                        f.2?[0..., hiddenDims...]
                    )
                    return compiledSiluProduct(
                        gather(x, g, indices[l]), gather(x, u, indices[l]))
                }
            }
            let bodies = [separate, combined, stackedBody, views]
            var samples = [[Double]](repeating: [], count: 4)
            for round in 0 ..< 41 {
                for arm in (round.isMultiple(of: 2) ? [0, 1, 2, 3] : [3, 2, 1, 0]) {
                    let body = bodies[arm]
                    let start = Date.timeIntervalSinceReferenceDate
                    eval((0 ..< 20).flatMap { _ in body() })
                    samples[arm].append(
                        (Date.timeIntervalSinceReferenceDate - start) * 1_000_000
                            / Double(20 * layers))
                }
            }
            let medians = samples.map { $0.sorted()[20] }
            print(
                String(
                    format:
                        "[MOEDECODEBENCH] gate+up K=%d H=%d: separate %.1f, fused [E,2H] %.1f, stacked [2E,H] %.1f, separate on views %.1f µs/layer",
                    inputDims, hiddenDims, medians[0], medians[1], medians[2], medians[3]))
        }
    }

    // MARK: - Synthetic models

    static func makeModel(preset: String, layers: Int?) throws -> any LanguageModel {
        let model: Module & LanguageModel
        switch preset {
        case "qwen3moe-tiny":
            // Qwen3-30B-A3B graph shape with tiny widths: CPU-bound decode.
            let json = """
                {
                    "model_type": "qwen3_moe", "hidden_size": 256,
                    "num_hidden_layers": \(layers ?? 48), "intermediate_size": 512,
                    "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 64,
                    "num_experts": 128, "num_experts_per_tok": 8,
                    "decoder_sparse_step": 1, "mlp_only_layers": [],
                    "moe_intermediate_size": 128, "rms_norm_eps": 1e-6, "vocab_size": 4096,
                    "rope_theta": 1000000.0, "norm_topk_prob": true,
                    "tie_word_embeddings": false
                }
                """
            model = Qwen3MoEModel(
                try JSONDecoder().decode(Qwen3MoEConfiguration.self, from: Data(json.utf8)))
        case "qwen3moe", "qwen3moe-large":
            let large = preset == "qwen3moe-large"
            let json = """
                {
                    "model_type": "qwen3_moe",
                    "hidden_size": \(large ? 4096 : 2048),
                    "num_hidden_layers": \(layers ?? (large ? 94 : 48)),
                    "intermediate_size": \(large ? 12288 : 6144),
                    "num_attention_heads": \(large ? 64 : 32),
                    "num_key_value_heads": 4, "head_dim": 128,
                    "num_experts": 128, "num_experts_per_tok": 8,
                    "decoder_sparse_step": 1, "mlp_only_layers": [],
                    "moe_intermediate_size": \(large ? 1536 : 768),
                    "rms_norm_eps": 1e-6, "vocab_size": 151936,
                    "rope_theta": 1000000.0, "norm_topk_prob": true,
                    "tie_word_embeddings": false
                }
                """
            model = Qwen3MoEModel(
                try JSONDecoder().decode(Qwen3MoEConfiguration.self, from: Data(json.utf8)))
        case "glm4moe":
            let json = """
                {
                    "model_type": "glm4_moe", "vocab_size": 151552, "hidden_size": 4096,
                    "intermediate_size": 10944, "max_position_embeddings": 131072,
                    "moe_intermediate_size": 1408, "norm_topk_prob": true,
                    "num_attention_heads": 96, "n_group": 1, "head_dim": 128,
                    "topk_group": 1, "n_shared_experts": 1, "n_routed_experts": 128,
                    "routed_scaling_factor": 1.0, "num_experts_per_tok": 8,
                    "first_k_dense_replace": 1, "num_hidden_layers": \(layers ?? 46),
                    "num_key_value_heads": 8, "rms_norm_eps": 1e-5,
                    "rope_theta": 1000000.0, "use_qk_norm": false,
                    "tie_word_embeddings": false, "attention_bias": true,
                    "partial_rotary_factor": 0.5, "scoring_func": "sigmoid",
                    "topk_method": "noaux_tc"
                }
                """
            model = GLM4MoEModel(
                try JSONDecoder().decode(GLM4MoEConfiguration.self, from: Data(json.utf8)))
        case "deepseekv3":
            // DeepSeek-V3 routing and MLA with reduced width so it fits on one Mac.
            let json = """
                {
                    "model_type": "deepseek_v3", "vocab_size": 129280, "hidden_size": 4096,
                    "intermediate_size": 11264, "moe_intermediate_size": 1024,
                    "num_hidden_layers": \(layers ?? 24), "num_attention_heads": 32,
                    "num_key_value_heads": 32, "n_routed_experts": 256,
                    "n_shared_experts": 1, "num_experts_per_tok": 8, "n_group": 8,
                    "topk_group": 4, "norm_topk_prob": true, "routed_scaling_factor": 2.5,
                    "first_k_dense_replace": 1, "moe_layer_freq": 1, "q_lora_rank": 1536,
                    "kv_lora_rank": 512, "qk_rope_head_dim": 64, "qk_nope_head_dim": 128,
                    "v_head_dim": 128, "rms_norm_eps": 1e-6, "rope_theta": 10000.0,
                    "max_position_embeddings": 4096, "attention_bias": false
                }
                """
            model = DeepseekV3Model(
                try JSONDecoder().decode(DeepseekV3Configuration.self, from: Data(json.utf8)))
        default:
            throw XCTSkip("Unknown preset \(preset)")
        }
        materializeQuantized(model)
        return model
    }

    /// Cast to bfloat16 and quantize one layer at a time, so only one layer's
    /// float32 random weights are resident at once.
    static func materializeQuantized(_ model: Module) {
        for (_, child) in model.leafModules().flattened() {
            child.update(parameters: child.parameters().mapValues { $0.asType(.bfloat16) })
        }
        quantize(model: model, groupSize: 64, bits: 4)
        for (_, child) in model.leafModules().flattened() {
            eval(child)
        }
        eval(model)
    }
}
