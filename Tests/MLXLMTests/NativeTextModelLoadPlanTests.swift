// Copyright © 2026 Faux Fox.

import Foundation
import XCTest

@testable import MLXLLM

final class NativeTextModelLoadPlanTests: XCTestCase {
    func testResidentQwenSelectionExcludesOnlyNonTargetNamespaces() {
        XCTAssertFalse(qwen35ResidentTensorSelection.contains("mtp.fc.weight"))
        XCTAssertFalse(qwen35ResidentTensorSelection.contains("vision_tower.patch.weight"))
        XCTAssertFalse(qwen35ResidentTensorSelection.contains("model.visual.patch.weight"))
        XCTAssertTrue(
            qwen35ResidentTensorSelection.contains(
                "language_model.model.layers.0.mlp.switch_mlp.gate_proj.weight"))
        XCTAssertTrue(qwen35ResidentTensorSelection.contains("model.embed_tokens.weight"))
    }

    func testStreamedPlanSeparatesRoutedExpertsFromDenseWeights() throws {
        let directory = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        try writeCheckpoint(in: directory)

        let streamed = try NativeTextModelLoader.loadPlan(
            directory: directory,
            policy: .streamedExperts(.init(maximumResidentBytesPerLayer: 96)))
        XCTAssertEqual(streamed.denseWeightBytes, 4)
        XCTAssertEqual(streamed.routedExpertWeightBytes, 192)
        XCTAssertEqual(streamed.maximumResidentExpertBytes, 192)
        XCTAssertEqual(streamed.maximumResidentWeightBytes, 196)

        let resident = try NativeTextModelLoader.loadPlan(directory: directory)
        XCTAssertEqual(resident.maximumResidentExpertBytes, 192)
        XCTAssertEqual(resident.maximumResidentWeightBytes, 196)
    }

    func testStreamedPlanRejectsNegativeResidentBudget() throws {
        let directory = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        try writeCheckpoint(in: directory)

        XCTAssertThrowsError(
            try NativeTextModelLoader.loadPlan(
                directory: directory,
                policy: .streamedExperts(.init(maximumResidentBytesPerLayer: -1)))
        ) { error in
            XCTAssertEqual(
                error as? NativeTextModelLoadingError,
                .invalidStreamedExpertConfiguration)
        }
    }

    func testEdge0PlanRejectsNegativeBudgetAndOrdinaryQwenCheckpoint() throws {
        let directory = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        try writeCheckpoint(in: directory)

        XCTAssertThrowsError(
            try NativeTextModelLoader.loadPlan(
                directory: directory,
                policy: .edge0(.init(maximumResidentBytesPerLayer: -1)))
        ) {
            XCTAssertEqual(
                $0 as? NativeTextModelLoadingError,
                .invalidStreamedExpertConfiguration)
        }
        XCTAssertThrowsError(
            try NativeTextModelLoader.loadPlan(
                directory: directory,
                policy: .edge0(.init(maximumResidentBytesPerLayer: 96)))
        ) {
            XCTAssertEqual($0 as? NativeTextModelLoadingError, .incompatibleEdge0Checkpoint)
        }
    }

    func testEdge0AcceptsPinnedCheckpointArchitecture() throws {
        let data = Data(
            """
            {
              "model_type": "qwen3_5_moe",
              "text_config": {
                "model_type": "qwen3_5_moe",
                "hidden_size": 2048,
                "num_hidden_layers": 40,
                "num_attention_heads": 16,
                "num_key_value_heads": 2,
                "head_dim": 256,
                "full_attention_interval": 4,
                "num_experts": 256,
                "num_experts_per_tok": 8,
                "moe_intermediate_size": 512,
                "shared_expert_intermediate_size": 512,
                "linear_num_value_heads": 32,
                "linear_num_key_heads": 16,
                "linear_key_head_dim": 128,
                "linear_value_head_dim": 128,
                "linear_conv_kernel_dim": 4,
                "vocab_size": 248320
              }
            }
            """.utf8)
        let configuration = try JSONDecoder().decode(Qwen35Configuration.self, from: data)

        XCTAssertNoThrow(
            try validateEdge0Checkpoint(
                baseModelType: configuration.modelType,
                configuration: configuration.textConfig))
    }

    func testEdge0RejectsCheckpointWithDifferentVocabulary() throws {
        let data = Data(
            """
            {
              "model_type": "qwen3_5_moe",
              "text_config": {
                "model_type": "qwen3_5_moe",
                "hidden_size": 2048,
                "num_hidden_layers": 40,
                "num_attention_heads": 16,
                "num_key_value_heads": 2,
                "head_dim": 256,
                "full_attention_interval": 4,
                "num_experts": 256,
                "num_experts_per_tok": 8,
                "moe_intermediate_size": 512,
                "shared_expert_intermediate_size": 512,
                "linear_num_value_heads": 32,
                "linear_num_key_heads": 16,
                "linear_key_head_dim": 128,
                "linear_value_head_dim": 128,
                "linear_conv_kernel_dim": 4,
                "vocab_size": 32000
              }
            }
            """.utf8)
        let configuration = try JSONDecoder().decode(Qwen35Configuration.self, from: data)

        XCTAssertThrowsError(
            try validateEdge0Checkpoint(
                baseModelType: configuration.modelType,
                configuration: configuration.textConfig)
        ) {
            XCTAssertEqual(
                $0 as? NativeTextModelLoadingError,
                .incompatibleEdge0Checkpoint)
        }
    }

    private func writeCheckpoint(in directory: URL) throws {
        let config: [String: Any] = [
            "model_type": "qwen3_5_text",
            "text_config": [
                "model_type": "qwen3_5_text",
                "num_hidden_layers": 2,
                "num_experts": 2,
                "num_experts_per_tok": 1,
            ],
        ]
        try JSONSerialization.data(withJSONObject: config).write(
            to: directory.appendingPathComponent("config.json"))

        var header = [String: [String: Any]]()
        var offset = 0
        func add(_ name: String, dtype: String, shape: [Int], bytes: Int) {
            header[name] = [
                "dtype": dtype,
                "shape": shape,
                "data_offsets": [offset, offset + bytes],
            ]
            offset += bytes
        }

        add("model.embed_tokens.weight", dtype: "F16", shape: [2], bytes: 4)
        for layer in 0 ..< 2 {
            let prefix = "model.layers.\(layer).mlp.switch_mlp"
            for projection in ["gate_proj", "up_proj", "down_proj"] {
                add("\(prefix).\(projection).weight", dtype: "U32", shape: [2, 2, 1], bytes: 16)
                add("\(prefix).\(projection).scales", dtype: "F16", shape: [2, 2, 1], bytes: 8)
                add("\(prefix).\(projection).biases", dtype: "F16", shape: [2, 2, 1], bytes: 8)
            }
        }
        let headerData = try JSONSerialization.data(withJSONObject: header)
        var headerSize = UInt64(headerData.count).littleEndian
        var contents = Data(bytes: &headerSize, count: MemoryLayout<UInt64>.size)
        contents.append(headerData)
        contents.append(Data(repeating: 0, count: offset))
        try contents.write(to: directory.appendingPathComponent("model.safetensors"))
    }

    private func makeTemporaryDirectory() throws -> URL {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(
            "NativeTextModelLoadPlanTests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: false)
        return directory
    }
}
