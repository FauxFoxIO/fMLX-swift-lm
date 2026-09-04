// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import XCTest

@testable import MLXLLM

final class Qwen38CompatibilityTests: XCTestCase {

    private func configuration() throws -> Qwen35TextConfiguration {
        let json = """
            {
                "architectures": ["Qwen3_5ForConditionalGeneration"],
                "model_type": "qwen3_5",
                "text_config": {
                    "model_type": "qwen3_5_text",
                    "hidden_size": 32, "num_hidden_layers": 4, "intermediate_size": 64,
                    "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 16,
                    "linear_num_value_heads": 6, "linear_num_key_heads": 2,
                    "linear_key_head_dim": 16, "linear_value_head_dim": 16,
                    "linear_conv_kernel_dim": 4, "vocab_size": 32,
                    "full_attention_interval": 4,
                    "layer_types": [
                        "linear_attention", "linear_attention", "linear_attention", "full_attention"
                    ],
                    "hidden_act": "silu", "output_gate_type": "swish", "attn_output_gate": true,
                    "mamba_ssm_dtype": "float32", "rms_norm_eps": 0.000001,
                    "mtp_num_hidden_layers": 1, "mtp_use_dedicated_embeddings": false,
                    "tie_word_embeddings": false, "max_position_embeddings": 262144,
                    "rope_parameters": {
                        "rope_type": "default", "rope_theta": 10000000,
                        "partial_rotary_factor": 0.25,
                        "mrope_interleaved": true, "mrope_section": [11, 11, 10]
                    }
                }
            }
            """
        return try JSONDecoder().decode(Qwen35Configuration.self, from: Data(json.utf8)).textConfig
    }

    func testSwishMetadataKeepsFullAttentionGateSigmoid() throws {
        let attention = Qwen35Attention(try configuration())
        attention.update(
            modules: ModuleChildren.unflattened([("o_proj", Linear(weight: MLXArray.eye(32)))]))
        let gates = (0 ..< 64).map { Float($0 % 5 - 2) }
        let output = attention.mergeHeadsAndProject(
            attention: MLXArray.ones([1, 2, 2, 16]),
            gate: MLXArray(gates).reshaped(1, 2, 32))
        let expected = gates.map { Float(1.0 / (1.0 + exp(-Double($0)))) }
        let actual = output.asArray(Float.self)

        for (actual, expected) in zip(actual, expected) {
            XCTAssertEqual(actual, expected, accuracy: 1e-6)
        }
    }

    func testSwishMetadataUsesSiLUForGatedDeltaNet() throws {
        let config = try configuration()
        let gdn = Qwen35GatedDeltaNet(config)
        let gates = (0 ..< 16).map { Float($0 % 5 - 2) }
        let output = gdn.norm(
            MLXArray.ones([1, 1, 1, 16]),
            gate: MLXArray(gates).reshaped(1, 1, 1, 16))
        let scale = 1.0 / sqrt(1.0 + Double(config.rmsNormEps))
        let expected = gates.map { Float(scale * Double($0) / (1.0 + exp(-Double($0)))) }

        for (actual, expected) in zip(output.asArray(Float.self), expected) {
            XCTAssertEqual(actual, expected, accuracy: 1e-6)
        }
        XCTAssertEqual(gdn.zeroStates(batch: 1, dtype: .bfloat16).rec.dtype, .float32)
    }

    func testTextRoPEUsesCheckpointThetaAndOnlyRotates64Dimensions() throws {
        var config = try configuration()
        config.headDim = 256
        XCTAssertEqual(config.ropeTheta, 10_000_000)
        XCTAssertEqual(config.partialRotaryFactor, 0.25)
        let attention = Qwen35Attention(config)
        let values = (0 ..< 512).map { Float($0 % 13 - 6) / 7 }
        let output = applyRotaryPosition(
            attention.rope, to: MLXArray(values).reshaped(1, 1, 2, 256), offset: .scalar(7))
        var expected = values
        for token in 0 ..< 2 {
            for pair in 0 ..< 32 {
                let angle = Double(token + 7) / pow(10_000_000.0, Double(2 * pair) / 64)
                let first = token * 256 + pair
                let second = first + 32
                expected[first] = Float(
                    Double(values[first]) * cos(angle) - Double(values[second]) * sin(angle))
                expected[second] = Float(
                    Double(values[second]) * cos(angle) + Double(values[first]) * sin(angle))
            }
        }

        for (actual, expected) in zip(output.asArray(Float.self), expected) {
            XCTAssertEqual(actual, expected, accuracy: 1e-5)
        }
    }

    func testConvertedCheckpointNormWeightsRemainUnshifted() throws {
        let model = Qwen35TextModel(try configuration())
        let norm = MLXArray([Float(0.75), 1.25])
        let normKeys = [
            "model.norm.weight", "model.layers.0.input_layernorm.weight",
            "model.layers.0.post_attention_layernorm.weight",
            "model.layers.3.self_attn.q_norm.weight", "model.layers.3.self_attn.k_norm.weight",
            "model.layers.0.linear_attn.norm.weight",
        ]
        var weights = Dictionary(uniqueKeysWithValues: normKeys.map { ($0, norm) })
        weights["model.layers.0.linear_attn.conv1d.weight"] = MLXArray.zeros([160, 4, 1])
        let sanitized = model.sanitize(weights: weights)

        for key in normKeys {
            let actual = try XCTUnwrap(sanitized[key])
            XCTAssertEqual(actual.asArray(Float.self), norm.asArray(Float.self), key)
        }
    }

    func testCompiledDecodeMatchesFullPrefillWithQwen38Metadata() throws {
        let config = try configuration()
        let model = withRandomState(MLXRandom.RandomState(seed: 38)) {
            Qwen35TextModel(config)
        }
        let cache = try model.newCache(parameters: nil)
        XCTAssertEqual(cache.filter { $0 is MambaCache }.count, 3)
        var tokens: [Int32] = [1, 4, 5]
        eval(model(MLXArray(tokens).reshaped(1, tokens.count), cache: cache))

        for token in [Int32(6), Int32(7)] {
            tokens.append(token)
            let decoded = model(MLXArray([token]).reshaped(1, 1), cache: cache)
            let full = model(MLXArray(tokens).reshaped(1, tokens.count), cache: nil)
            let expected = full[0..., (tokens.count - 1)..., 0...]
            XCTAssertTrue(allClose(decoded, expected, rtol: 1e-4, atol: 1e-4).item(Bool.self))
            for layerCache in cache {
                if let recurrent = layerCache as? MambaCache {
                    let conv = try XCTUnwrap(recurrent[0])
                    let state = try XCTUnwrap(recurrent[1])
                    let convDimensions =
                        2 * config.linearNumKeyHeads * config.linearKeyHeadDim
                        + config.linearNumValueHeads * config.linearValueHeadDim
                    XCTAssertEqual(conv.shape, [1, config.linearConvKernelDim - 1, convDimensions])
                    XCTAssertEqual(
                        state.shape,
                        [
                            1, config.linearNumValueHeads, config.linearValueHeadDim,
                            config.linearKeyHeadDim,
                        ])
                    XCTAssertEqual(state.dtype, .float32)
                } else {
                    XCTAssertEqual(layerCache.offset, tokens.count)
                }
            }
        }
        XCTAssertGreaterThan(model.model.compiledDecodeSegmentCount, 0)
    }
}
