// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import XCTest

@testable import MLXLLM

final class Qwen35QLoRATests: XCTestCase {

    private func configuration() throws -> Qwen35TextConfiguration {
        let json = """
            {
                "model_type": "qwen3_5",
                "hidden_size": 64,
                "num_hidden_layers": 2,
                "intermediate_size": 128,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_dim": 32,
                "linear_num_value_heads": 4,
                "linear_num_key_heads": 2,
                "linear_key_head_dim": 32,
                "linear_value_head_dim": 32,
                "linear_conv_kernel_dim": 4,
                "vocab_size": 32,
                "full_attention_interval": 2,
                "num_experts": 0,
                "num_experts_per_tok": 0
            }
            """
        return try JSONDecoder().decode(
            Qwen35TextConfiguration.self, from: Data(json.utf8))
    }

    /// Regression for #561. Real Qwen3.5 MLX checkpoints use bf16 base
    /// tensors while converted PEFT adapters commonly use fp16. Wrapping the
    /// three linear-attention projections used to promote their results to
    /// fp32, changing the custom gated-delta kernel specialization and making
    /// the first generation command buffer hit macOS's GPU watchdog.
    func testFP16QLoRAKeepsQwen35LinearAttentionOnBaseDType() throws {
        let model = Qwen35TextModel(try configuration())
        model.update(parameters: model.parameters().mapValues { $0.asType(.bfloat16) })

        let targetPaths: Set<String> = [
            "model.layers.0.linear_attn.in_proj_qkv",
            "model.layers.0.linear_attn.in_proj_z",
            "model.layers.0.linear_attn.out_proj",
        ]
        quantize(model: model, groupSize: 32, bits: 4) { path, _ in
            targetPaths.contains(path)
        }

        let rank = 4
        let adapter = LoRAContainer(
            configuration: .init(
                numLayers: 2,
                loraParameters: .init(
                    rank: rank, scale: 1.0,
                    keys: [
                        "linear_attn.in_proj_qkv",
                        "linear_attn.in_proj_z",
                        "linear_attn.out_proj",
                    ])),
            parameters: ModuleParameters.unflattened([
                "model.layers.0.linear_attn.in_proj_qkv.lora_a": fp16Adapter([64, rank]),
                "model.layers.0.linear_attn.in_proj_qkv.lora_b": fp16Adapter([rank, 256]),
                "model.layers.0.linear_attn.in_proj_z.lora_a": fp16Adapter([64, rank]),
                "model.layers.0.linear_attn.in_proj_z.lora_b": fp16Adapter([rank, 128]),
                "model.layers.0.linear_attn.out_proj.lora_a": fp16Adapter([128, rank]),
                "model.layers.0.linear_attn.out_proj.lora_b": fp16Adapter([rank, 64]),
            ]))
        try adapter.load(into: model)

        let gdn = try XCTUnwrap(
            model.modules().compactMap { $0 as? Qwen35GatedDeltaNet }.first)
        XCTAssertTrue(gdn.inProjQKV is QLoRALinear)
        XCTAssertTrue(gdn.inProjZ is QLoRALinear)
        XCTAssertTrue(gdn.outProj is QLoRALinear)

        let hidden = MLXArray.ones([1, 15, 64], dtype: .bfloat16)
        XCTAssertEqual(gdn.inProjQKV(hidden).dtype, .bfloat16)
        XCTAssertEqual(gdn.inProjZ(hidden).dtype, .bfloat16)
        XCTAssertEqual(
            gdn.outProj(MLXArray.ones([1, 15, 128], dtype: .bfloat16)).dtype,
            .bfloat16)

        let cache = try model.newCache(parameters: nil)
        let prefill = model(
            MLXArray(Array(Int32(1) ... Int32(15))).reshaped(1, 15), cache: cache)
        eval(prefill, cache)
        XCTAssertEqual(prefill.dtype, .bfloat16)
        XCTAssertEqual(prefill.shape, [1, 15, 32])

        let decode = model(MLXArray([Int32(16)]).reshaped(1, 1), cache: cache)
        eval(decode, cache)
        XCTAssertEqual(decode.dtype, .bfloat16)
        XCTAssertEqual(decode.shape, [1, 1, 32])
    }

    private func fp16Adapter(_ shape: [Int]) -> MLXArray {
        MLXArray.full(shape, values: MLXArray(0.001), dtype: .float16)
    }

    func testPreparedQuantizedVerificationTracksAdapterLifecycle() throws {
        let config = try configuration()
        let model = withRandomState(MLXRandom.RandomState(seed: 331)) {
            Qwen35TextModel(config)
        }
        model.update(parameters: model.parameters().mapValues { $0.asType(.bfloat16) })
        quantize(model: model, groupSize: 32, bits: 4)
        model.train(false)
        try model.prepare()
        let layer = try XCTUnwrap(
            model.modules().compactMap { $0 as? Qwen35DecoderLayer }.first(where: \.isLinear))
        let gdn = try XCTUnwrap(layer.linearAttn)
        XCTAssertTrue(gdn.hasFusedInputProjection)
        let input = MLXArray((0 ..< 128).map { Float($0 % 17 + 1) / 17 })
            .reshaped(1, 2, 64).asType(.bfloat16)

        func verify() throws -> [Float] {
            let seed = MambaCache()
            eval(
                layer(
                    MLXArray.ones([1, 3, 64], dtype: .bfloat16), attentionMask: .causal,
                    ssmMask: nil, cache: seed))
            let compiledCache = try XCTUnwrap(seed.copy() as? MambaCache)
            let referenceCache = try XCTUnwrap(seed.copy() as? MambaCache)
            layer.compiledVerificationEnabled = true
            let actual = layer(
                input, attentionMask: .none, ssmMask: nil, cache: compiledCache,
                checkpointAfter: 1)
            eval(actual, compiledCache)
            XCTAssertTrue(layer.hasCompiledVerificationLayer)
            layer.compiledVerificationEnabled = false
            let expected = layer(
                input, attentionMask: .none, ssmMask: nil, cache: referenceCache,
                checkpointAfter: 1)
            eval(expected, referenceCache)
            XCTAssertTrue(allClose(actual, expected, rtol: 1e-4, atol: 1e-4).item(Bool.self))
            for (a, b) in zip(compiledCache.state, referenceCache.state) {
                XCTAssertTrue(allClose(a, b, rtol: 1e-4, atol: 1e-4).item(Bool.self))
            }
            XCTAssertTrue(compiledCache.restoreSpeculativeCheckpoint())
            XCTAssertTrue(referenceCache.restoreSpeculativeCheckpoint())
            XCTAssertEqual(compiledCache.offset, referenceCache.offset)
            for (a, b) in zip(compiledCache.state, referenceCache.state) {
                XCTAssertTrue(allClose(a, b, rtol: 1e-4, atol: 1e-4).item(Bool.self))
            }
            return actual.asType(.float32).asArray(Float.self)
        }

        let baseline = try verify()
        let rank = 4
        let adapter = LoRAContainer(
            configuration: .init(
                numLayers: 2,
                loraParameters: .init(
                    rank: rank, scale: 1, keys: ["linear_attn.in_proj_qkv"])),
            parameters: ModuleParameters.unflattened([
                "model.layers.0.linear_attn.in_proj_qkv.lora_a": MLXArray.full(
                    [64, rank], values: MLXArray(0.1), dtype: .float16),
                "model.layers.0.linear_attn.in_proj_qkv.lora_b": MLXArray.full(
                    [rank, 256], values: MLXArray(0.1), dtype: .float16),
            ]))
        try adapter.load(into: model)
        XCTAssertFalse(layer.hasCompiledVerificationLayer)
        XCTAssertFalse(gdn.hasFusedInputProjection)
        XCTAssertNotEqual(try verify(), baseline)

        adapter.unload(from: model)
        XCTAssertFalse(layer.hasCompiledVerificationLayer)
        XCTAssertEqual(try verify(), baseline)
        model.invalidateCompiledTraces()
        try model.prepare()
        XCTAssertTrue(gdn.hasFusedInputProjection)
        XCTAssertEqual(try verify(), baseline)

        try adapter.fuse(with: model)
        XCTAssertFalse(layer.hasCompiledVerificationLayer)
        XCTAssertFalse(gdn.inProjQKV is QLoRALinear)
        XCTAssertNotEqual(try verify(), baseline)
    }
}
