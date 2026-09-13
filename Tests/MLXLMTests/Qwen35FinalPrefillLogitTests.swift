// Copyright © 2026 Faux Fox.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import XCTest

@testable import MLXLLM

final class Qwen35FinalPrefillLogitTests: XCTestCase {
    func testFinalPrefillLogitsMatchTheLastFullProjectionAndPreserveKV() async throws {
        let configuration = try JSONDecoder().decode(
            Qwen35TextConfiguration.self,
            from: Data(
                """
                {"model_type":"qwen3_5_text","hidden_size":64,"num_hidden_layers":2,
                "intermediate_size":128,"num_attention_heads":2,"num_key_value_heads":1,
                "head_dim":32,"linear_num_value_heads":2,"linear_num_key_heads":1,
                "linear_key_head_dim":32,"linear_value_head_dim":32,"linear_conv_kernel_dim":4,
                "vocab_size":100,"full_attention_interval":2,"tie_word_embeddings":true,
                "rope_theta":10000000.0,"partial_rotary_factor":0.25}
                """.utf8))
        let model = withRandomState(MLXRandom.RandomState(seed: 381)) {
            Qwen35TextModel(configuration)
        }
        let input = MLXArray([Int32(3), 7, 11]).reshaped([1, 3])
        let fullCache = try model.newCache(parameters: nil)
        let finalCache = try model.newCache(parameters: nil)

        let full = try await model.scheduledForward(input, cache: fullCache)
        let final = try await model.scheduledFinalPrefillForward(input, cache: finalCache)
        let expected = full[0..., (full.dim(1) - 1) ..< full.dim(1), 0...]
        let fullState = fullCache.flatMap { $0.innerState() }
        let finalState = finalCache.flatMap { $0.innerState() }
        eval([expected, final] + fullState + finalState)

        XCTAssertEqual(final.shape, [1, 1, configuration.vocabularySize])
        XCTAssertLessThanOrEqual(MLX.max(abs(final - expected)).item(Float.self), 1e-5)
        XCTAssertEqual(
            argMax(final, axis: -1).item(Int.self),
            argMax(expected, axis: -1).item(Int.self))
        XCTAssertEqual(finalState.count, fullState.count)
        for (optimized, baseline) in zip(finalState, fullState) {
            XCTAssertTrue(allClose(optimized, baseline, rtol: 0, atol: 0).item(Bool.self))
        }
    }

    func testInstalledCheckpointFinalPrefillParity() async throws {
        guard let path = ProcessInfo.processInfo.environment["FMLX_QWEN_FINAL_PREFILL_PARITY_MODEL"]
        else {
            throw XCTSkip("Set FMLX_QWEN_FINAL_PREFILL_PARITY_MODEL to an installed Qwen 3.5 model")
        }
        let model = try await NativeTextModelLoader.load(directory: URL(fileURLWithPath: path))
        for length in [1, 2, 7, 31, 128, 511, 512] {
            let input = MLXArray((0 ..< length).map { Int32(1_000 + ($0 * 37) % 997) })
                .expandedDimensions(axis: 0)
            let fullCache = try model.newCache(parameters: nil)
            let finalCache = try model.newCache(parameters: nil)

            let full = try await model.scheduledForward(input, cache: fullCache)
            let final = try await model.scheduledFinalPrefillForward(input, cache: finalCache)
            let expected = full[0..., (full.dim(1) - 1) ..< full.dim(1), 0...]
            eval(expected, final)

            let maxAbsDifference = MLX.max(abs(final - expected)).item(Float.self)
            print(
                "FMLX_QWEN_FINAL_PREFILL_PARITY length=\(length) maxAbsDifference=\(maxAbsDifference)"
            )
            XCTAssertLessThanOrEqual(maxAbsDifference, 0.125)
            XCTAssertEqual(
                argMax(final, axis: -1).item(Int.self),
                argMax(expected, axis: -1).item(Int.self))
        }
    }
}
