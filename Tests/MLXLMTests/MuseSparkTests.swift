// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

/// Parity with mlx-lm's `muse_spark_open` on fixtures from `scripts/parity/muse_spark.py`.
///
/// The port follows the reference's bf16 roundings, so the logits match to well within the bound.
struct MuseSparkParityTests {
    static let tolerance: Float = 1e-3

    static func prefillDifference(_ name: String) async throws -> Float {
        let fixture = try ParityFixture(name)
        let model = try await fixture.loadLanguageModel()
        return fixture.maxAbsDifference(model(fixture.inputIDs, cache: nil))
    }

    @Test("Meta reference names match mlx-lm")
    func metaNames() async throws {
        let difference = try await Self.prefillDifference("muse_spark_meta")
        #expect(difference < Self.tolerance, "max |Δ logit| \(difference)")
    }

    @Test("Transformers names match mlx-lm")
    func transformersNames() async throws {
        let difference = try await Self.prefillDifference("muse_spark_hf")
        #expect(difference < Self.tolerance, "max |Δ logit| \(difference)")
    }

    @Test("Nvfp4 experts with a global scale match mlx-lm")
    func nvfp4GlobalScale() async throws {
        let fixture = try ParityFixture("muse_spark_nvfp4")
        let model = try await fixture.loadLanguageModel()
        let experts = try #require(model as? MuseSparkModel).model.layers[1].feedForward.switchMLP
        let gate = try #require(experts.gateProj as? QuantizedSwitchLinear)
        #expect(gate.mode == .nvfp4)
        #expect(gate.globalScale?.shape == [8])
        let difference = fixture.maxAbsDifference(model(fixture.inputIDs, cache: nil))
        #expect(difference < Self.tolerance, "max |Δ logit| \(difference)")
    }

    @Test(
        "Decoding through the cache matches the prefill reference",
        arguments: [
            "muse_spark_meta", "muse_spark_nvfp4",
        ])
    func cachedDecode(name: String) async throws {
        let fixture = try ParityFixture(name)
        let model = try await fixture.loadLanguageModel()
        let cache = try model.newCache(parameters: nil)
        let length = fixture.inputIDs.dim(1)
        // The prompt is longer than the sliding window, so the rotating caches wrap.
        var steps = [model(fixture.inputIDs[0..., ..<5], cache: cache)]
        for t in 5 ..< length {
            steps.append(model(fixture.inputIDs[0..., t ..< (t + 1)], cache: cache))
        }
        let logits = concatenated(steps, axis: 1)
        let difference = fixture.maxAbsDifference(logits)
        #expect(difference < Self.tolerance, "max |Δ logit| \(difference)")
    }

    /// The mlx-community layout: `language_model.` prefix, split `switch_mlp` experts and the
    /// post-expert norm on `mlp`.
    @Test("mlx-community names match mlx-lm")
    func mlxCommunityNames() async throws {
        let fixture = try ParityFixture("muse_spark_hf")
        var weights = [String: MLXArray]()
        for (key, value) in try loadArrays(
            url: fixture.directory.appending(component: "model.safetensors"))
        {
            var name = key.replacingOccurrences(
                of: "model.language_model.", with: "language_model.model.")
            name = name.replacingOccurrences(of: "lm_head.", with: "language_model.lm_head.")
            if name.hasSuffix("mlp.experts.gate_up_proj") {
                let parts = split(value, parts: 2, axis: 1)
                let base = name.replacingOccurrences(
                    of: "experts.gate_up_proj", with: "switch_mlp.")
                weights[base + "gate_proj.weight"] = parts[0]
                weights[base + "up_proj.weight"] = parts[1]
                continue
            }
            name = name.replacingOccurrences(
                of: "mlp.experts.down_proj", with: "mlp.switch_mlp.down_proj.weight")
            name = name.replacingOccurrences(
                of: "mlp.experts.post_expert_norm", with: "mlp.post_expert_norm")
            weights[name] = value
        }
        let model = try await fixture.loadLanguageModel()
        try load(weights, into: model)
        let difference = fixture.maxAbsDifference(model(fixture.inputIDs, cache: nil))
        #expect(difference < Self.tolerance, "max |Δ logit| \(difference)")
    }

    /// Weights in this model's layout, as fMLX conversion saves them, and with the fused
    /// `qkvg_proj` that mlx-lm conversion saves.
    @Test("Converted checkpoints load unchanged", arguments: [false, true])
    func convertedCheckpoint(fusedAttention: Bool) async throws {
        let fixture = try ParityFixture("muse_spark_meta")
        let source = try await fixture.loadLanguageModel()
        var weights = Dictionary(uniqueKeysWithValues: source.parameters().flattened())
        if fusedAttention {
            for layer in 0 ..< 4 {
                let attention = "model.layers.\(layer).self_attn."
                let parts = ["q_proj", "k_proj", "v_proj", "o_gate_proj"].map {
                    weights.removeValue(forKey: attention + $0 + ".weight")!
                }
                weights[attention + "qkvg_proj.weight"] = concatenated(parts, axis: 0)
            }
        }
        let model = try await fixture.loadLanguageModel()
        try load(weights, into: model)
        let difference = fixture.maxAbsDifference(model(fixture.inputIDs, cache: nil))
        #expect(difference < Self.tolerance, "max |Δ logit| \(difference)")
    }

    private func load(_ weights: [String: MLXArray], into model: LanguageModel) throws {
        let checkpoint = try model.prepareCheckpoint(ModelCheckpoint(weights: weights))
        try model.update(
            parameters: ModuleParameters.unflattened(checkpoint.weights), verify: [.all])
    }
}

struct MuseSparkCheckpointNameTests {
    @Test(
        "Checkpoint names map to the text model",
        arguments: [
            (
                "model.language_model.layers.3.self_attn.gate_proj.weight",
                "model.layers.3.self_attn.o_gate_proj.weight"
            ),
            (
                "language_model.model.layers.3.self_attn.gate_proj",
                "model.layers.3.self_attn.o_gate_proj"
            ),
            ("model.layers.0.mlp.gate.weight", "model.layers.0.feed_forward.router"),
            (
                "model.layers.0.mlp.gate.e_score_correction_bias",
                "model.layers.0.feed_forward.expert_bias"
            ),
            (
                "model.layers.0.mlp.switch_mlp.down_proj.scales",
                "model.layers.0.feed_forward.switch_mlp.down_proj.scales"
            ),
            (
                "model.layers.0.mlp.post_expert_norm.weight",
                "model.layers.0.feed_forward.post_expert_norm.weight"
            ),
            (
                "model.layers.0.mlp.pre_expert_proj.biases",
                "model.layers.0.feed_forward.pooler.biases"
            ),
            (
                "model.language_model.layers.0.mlp.experts.down_proj_weight_scale_2",
                "model.layers.0.feed_forward.experts.w2_weight_scale_2"
            ),
            (
                "model.layers.0.self_attn.pre_qkv_proj_norm.weight",
                "model.layers.0.pre_attn_norm.weight"
            ),
            ("model.layers.0.post_feedforward_residual_gate.gate", "model.layers.0.add_ffn.gate"),
            ("language_model.lm_head", "lm_head"),
            (
                "model.layers.0.feed_forward.router.weight",
                "model.layers.0.feed_forward.router.weight"
            ),
        ])
    func mapped(source: String, expected: String) {
        #expect(MuseSparkCheckpoint.mapName(source) == expected)
    }

    @Test(
        "Vision, audio and activation scales are not loaded",
        arguments: [
            "model.vision_tower.layers.0.attn.qkv_proj.weight",
            "model.audio_tower.proj.weight",
            "modalities.vision_perception.encoder.vision_projection.weight",
            "vision_model.encoder.layers.0.ln_1.weight",
            "model.layers.0.feed_forward.experts.w13_input_scale",
            "model.language_model.layers.0.self_attn.k_scale",
        ])
    func dropped(source: String) {
        #expect(MuseSparkCheckpoint.mapName(source) == nil)
    }
}
