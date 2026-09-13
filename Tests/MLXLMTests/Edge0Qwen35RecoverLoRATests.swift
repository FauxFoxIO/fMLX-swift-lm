// Copyright © 2026 Faux Fox.

import Foundation
import MLX
import XCTest

@testable import MLXLLM

final class Edge0Qwen35RecoverLoRATests: XCTestCase {
    func testFixedTargetInventoryCoversEveryRecoverProjection() {
        let targets = Edge0Qwen35RecoverLoRA.expectedTargetPaths

        XCTAssertEqual(targets.count, 310)
        XCTAssertEqual(Set(targets).count, 310)
        XCTAssertTrue(targets.contains("language_model.model.layers.0.linear_attn.in_proj_qkv"))
        XCTAssertTrue(targets.contains("language_model.model.layers.3.self_attn.o_proj"))
        XCTAssertTrue(
            targets.contains("language_model.model.layers.39.mlp.shared_expert.down_proj"))
    }

    func testAttachmentRejectsAnyModelOutsideTheFixedEdge0Shape() throws {
        let target = "language_model.model.layers.0.mlp.shared_expert.gate_proj"
        let adapter = try Edge0Qwen35RecoverLoRA.load(
            weights: fixtureWeights(target: target), configuration: .init(expectedTargetCount: 1))
        let model = Qwen35TextModel(try tinyQwenConfiguration())

        XCTAssertThrowsError(try model.attachRecoverLoRA(adapter)) {
            XCTAssertEqual($0 as? Edge0Qwen35RecoverLoRAAttachmentError, .incompatibleModel)
        }
    }

    func testLoadsOnlyTheFixedSidecarArtifactAndRetainsExactTargetPath() throws {
        let directory = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }

        let target = "language_model.model.layers.0.mlp.shared_expert.gate_proj"
        try save(
            arrays: fixtureWeights(target: target),
            metadata: validLoRAMetadata,
            url: Edge0Qwen35RecoverLoRA.artifactURL(in: directory))

        let adapter = try Edge0Qwen35RecoverLoRA.load(
            from: directory, configuration: .init(expectedTargetCount: 1))

        XCTAssertEqual(adapter.targetPaths, [target])
        XCTAssertEqual(adapter.adapter(for: target)?.loraA.dtype, .float16)
        XCTAssertEqual(adapter.adapter(for: target)?.loraB.dtype, .float16)
    }

    func testFixedArtifactLocationDoesNotDiscoverOtherSafetensorsFiles() throws {
        let directory = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }

        try save(
            arrays: fixtureWeights(
                target: "language_model.model.layers.0.mlp.shared_expert.gate_proj"),
            url: directory.appendingPathComponent("other_adapter.safetensors"))

        XCTAssertThrowsError(
            try Edge0Qwen35RecoverLoRA.load(
                from: directory, configuration: .init(expectedTargetCount: 1))
        ) { error in
            XCTAssertEqual(
                error as? Edge0Qwen35RecoverLoRAArtifactError,
                .missingArtifact(Edge0Qwen35RecoverLoRA.artifactURL(in: directory)))
        }
    }

    func testValidatesConfiguredTargetCountAndFP16Pairs() {
        let target = "language_model.model.layers.0.mlp.shared_expert.gate_proj"
        XCTAssertThrowsError(
            try Edge0Qwen35RecoverLoRA.load(
                weights: fixtureWeights(target: target),
                configuration: .init(expectedTargetCount: 2))
        ) { error in
            XCTAssertEqual(
                error as? Edge0Qwen35RecoverLoRAArtifactError,
                .targetCountMismatch(expected: 2, actual: 1))
        }

        var weights = fixtureWeights(target: target)
        weights["\(target).lora_A"] = weights["\(target).lora_A"]!.asType(.float32)
        XCTAssertThrowsError(
            try Edge0Qwen35RecoverLoRA.load(
                weights: weights, configuration: .init(expectedTargetCount: 1))
        ) { error in
            XCTAssertEqual(
                error as? Edge0Qwen35RecoverLoRAArtifactError,
                .invalidWeightDType(target: target, component: "A", actual: .float32))
        }

        weights = fixtureWeights(target: target)
        weights.removeValue(forKey: "\(target).lora_B")
        XCTAssertThrowsError(
            try Edge0Qwen35RecoverLoRA.load(
                weights: weights, configuration: .init(expectedTargetCount: 1))
        ) { error in
            XCTAssertEqual(
                error as? Edge0Qwen35RecoverLoRAArtifactError,
                .missingComponent(target: target, component: "B"))
        }
    }

    func testAppliesFP16DeltaAndPreservesBaseOutputDType() throws {
        let target = "language_model.model.layers.0.mlp.shared_expert.gate_proj"
        let adapter = try Edge0Qwen35RecoverLoRA.load(
            weights: fixtureWeights(target: target), configuration: .init(expectedTargetCount: 1))
        let output = try adapter.applying(
            to: MLXArray.ones([512], dtype: .float16),
            input: fixtureInput(4, 5),
            forTarget: target)
        eval(output)

        // A's first row is [1, 2], B's first entry is 3: 1 + 2 * (4 + 10) * 3.
        XCTAssertEqual(output.dtype, .float16)
        XCTAssertEqual(output[0].item(Float.self), 85)

        let fp16InputDelta = try adapter.applying(
            to: MLXArray.zeros([512]),
            input: fixtureInput(1.0001, 0),
            forTarget: target)
        eval(fp16InputDelta)
        XCTAssertEqual(fp16InputDelta.dtype, .float32)
        XCTAssertEqual(fp16InputDelta[0].item(Float.self), 6)
    }

    func testFileLoadRejectsMissingMetadata() throws {
        let directory = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = Edge0Qwen35RecoverLoRA.artifactURL(in: directory)
        try save(
            arrays: fixtureWeights(
                target: "language_model.model.layers.0.mlp.shared_expert.gate_proj"),
            url: url)

        XCTAssertThrowsError(
            try Edge0Qwen35RecoverLoRA.load(
                fromArtifact: url, configuration: .init(expectedTargetCount: 1))
        ) {
            XCTAssertEqual(
                $0 as? Edge0Qwen35ArtifactMetadataError,
                .missingNestedMetadata)
        }
    }
}

private func tinyQwenConfiguration() throws -> Qwen35TextConfiguration {
    let json = """
        {
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
            "num_experts": 4,
            "num_experts_per_tok": 2,
            "shared_expert_intermediate_size": 32,
            "moe_intermediate_size": 64
        }
        """
    return try JSONDecoder().decode(Qwen35TextConfiguration.self, from: Data(json.utf8))
}

private func fixtureWeights(target: String) -> [String: MLXArray] {
    var a = Array(repeating: Float(0), count: Edge0Qwen35RecoverLoRAConfiguration.rank * 2048)
    a[0] = 1
    a[1] = 2
    var b = Array(repeating: Float(0), count: 512 * Edge0Qwen35RecoverLoRAConfiguration.rank)
    b[0] = 3
    return [
        "\(target).lora_A": MLXArray(a)
            .reshaped(Edge0Qwen35RecoverLoRAConfiguration.rank, 2048)
            .asType(.float16),
        "\(target).lora_B": MLXArray(b)
            .reshaped(512, Edge0Qwen35RecoverLoRAConfiguration.rank)
            .asType(.float16),
    ]
}

private func fixtureInput(_ first: Float, _ second: Float) -> MLXArray {
    var values = Array(repeating: Float(0), count: 2048)
    values[0] = first
    values[1] = second
    return MLXArray(values)
}

private let validLoRAMetadata = [
    "__metadata__": """
    {"model":"edge0-35b","kind":"lora","K":"4","r":"16","alpha":"32","source":"fixture","source_md5":"dbdef1af692986ad1937562c0d2aab7f","converted":"2026-09-08T09:59:20+00:00","format_version":"1"}
    """
]

private func makeTemporaryDirectory() throws -> URL {
    let directory = FileManager.default.temporaryDirectory
        .appendingPathComponent("Edge0Qwen35RecoverLoRATests-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: false)
    return directory
}
