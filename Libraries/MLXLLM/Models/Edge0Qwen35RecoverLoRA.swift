// Copyright © 2026 Faux Fox.

import Foundation
import MLX

/// The fixed Recover-LoRA layout shipped with Edge0's Qwen 3.5 35B artifact.
///
/// The expected target count is configurable only to support fixture artifacts.
/// Production uses all 310 A/B pairs (620 tensors).
public struct Edge0Qwen35RecoverLoRAConfiguration: Sendable, Equatable {
    public static let rank = 16
    public static let alpha: Float = 32

    /// `nil` permits a partial artifact, intended only for tools that select targets explicitly.
    public let expectedTargetCount: Int?

    public static let edge0_35b = Self(expectedTargetCount: 310)

    public init(expectedTargetCount: Int? = 310) {
        precondition(expectedTargetCount.map { $0 >= 0 } ?? true)
        self.expectedTargetCount = expectedTargetCount
    }

    public var scale: Float {
        Self.alpha / Float(Self.rank)
    }
}

public enum Edge0Qwen35RecoverLoRAArtifactError: Error, Equatable, LocalizedError {
    case missingArtifact(URL)
    case unexpectedTensor(String)
    case missingComponent(target: String, component: String)
    case targetCountMismatch(expected: Int, actual: Int)
    case invalidWeightDType(target: String, component: String, actual: DType)
    case invalidWeightRank(target: String, component: String, actual: [Int])
    case invalidAdapterRank(target: String, expected: Int, a: [Int], b: [Int])
    case invalidTargetShape(
        target: String, expectedA: [Int]?, expectedB: [Int]?, a: [Int], b: [Int])

    public var errorDescription: String? {
        switch self {
        case .missingArtifact(let url):
            "Edge0 Recover-LoRA artifact is missing: \(url.path)"
        case .unexpectedTensor(let name):
            "Edge0 Recover-LoRA has an unexpected tensor: \(name)"
        case .missingComponent(let target, let component):
            "Edge0 Recover-LoRA target \(target) is missing lora_\(component)"
        case .targetCountMismatch(let expected, let actual):
            "Edge0 Recover-LoRA has \(actual) targets, expected \(expected)"
        case .invalidWeightDType(let target, let component, let actual):
            "Edge0 Recover-LoRA \(target).lora_\(component) has dtype \(actual), expected Float16"
        case .invalidWeightRank(let target, let component, let actual):
            "Edge0 Recover-LoRA \(target).lora_\(component) has shape \(actual), expected rank 2"
        case .invalidAdapterRank(let target, let expected, let a, let b):
            "Edge0 Recover-LoRA \(target) has A shape \(a) and B shape \(b), expected rank \(expected)"
        case .invalidTargetShape(let target, let expectedA, let expectedB, let a, let b):
            "Edge0 Recover-LoRA \(target) has A shape \(a) and B shape \(b), expected \(String(describing: expectedA)) and \(String(describing: expectedB))"
        }
    }
}

public enum Edge0Qwen35RecoverLoRAAttachmentError: Error, Equatable, LocalizedError {
    case incompatibleModel
    case targetPathsMismatch(expected: Int, actual: Int)

    public var errorDescription: String? {
        switch self {
        case .incompatibleModel:
            "Edge0 Recover-LoRA requires the 40-layer Qwen 3.5 MoE architecture"
        case .targetPathsMismatch(let expected, let actual):
            "Edge0 Recover-LoRA targets do not match this Qwen model (expected \(expected), got \(actual))"
        }
    }
}

/// One unmerged Recover-LoRA target.
///
/// `loraA` is `[rank, input]` and `loraB` is `[output, rank]`, matching the
/// names and orientation in the Edge0 safetensors artifact.
public struct Edge0Qwen35RecoverLoRATarget {
    public let path: String
    public let loraA: MLXArray
    public let loraB: MLXArray

    /// Applies this adapter's parallel delta to a base linear output.
    ///
    /// This is deliberately independent of a particular linear class so a
    /// later Qwen integration can retain its streamed, quantized base layer.
    public func applying(to baseOutput: MLXArray, input: MLXArray) -> MLXArray {
        let adapterInput = input.asType(.float16)
        let delta = matmul(matmul(adapterInput, loraA.T), loraB.T)
        return baseOutput
            + (Edge0Qwen35RecoverLoRAConfiguration.edge0_35b.scale * delta)
            .asType(baseOutput.dtype)
    }
}

/// Loads Edge0's standalone Recover-LoRA artifact without downloading or reading base weights.
///
/// Call ``adapter(for:)`` or ``applying(to:input:forTarget:)`` from the Qwen
/// integration when replacing a compatible base linear with its parallel delta path.
public struct Edge0Qwen35RecoverLoRA {
    public static let artifactFileName = "lora_edge0_35b.safetensors"

    static let expectedTargetPaths: [String] = {
        (0 ..< 40).flatMap { layer in
            let prefix = "language_model.model.layers.\(layer)."
            let sharedExpert = [
                "mlp.shared_expert.gate_proj",
                "mlp.shared_expert.up_proj",
                "mlp.shared_expert.down_proj",
            ]
            if (layer + 1).isMultiple(of: 4) {
                return [
                    prefix + "self_attn.q_proj",
                    prefix + "self_attn.k_proj",
                    prefix + "self_attn.v_proj",
                    prefix + "self_attn.o_proj",
                ] + sharedExpert.map { prefix + $0 }
            }
            return [
                prefix + "linear_attn.in_proj_qkv",
                prefix + "linear_attn.in_proj_z",
                prefix + "linear_attn.in_proj_b",
                prefix + "linear_attn.in_proj_a",
                prefix + "linear_attn.out_proj",
            ] + sharedExpert.map { prefix + $0 }
        }
    }()

    public let configuration: Edge0Qwen35RecoverLoRAConfiguration
    private let targetsByPath: [String: Edge0Qwen35RecoverLoRATarget]

    public var targetPaths: [String] {
        targetsByPath.keys.sorted()
    }

    public init(
        configuration: Edge0Qwen35RecoverLoRAConfiguration,
        targets: [Edge0Qwen35RecoverLoRATarget]
    ) {
        self.configuration = configuration
        self.targetsByPath = Dictionary(uniqueKeysWithValues: targets.map { ($0.path, $0) })
    }

    /// Returns the fixed artifact location next to the Qwen checkpoint.
    public static func artifactURL(in modelDirectory: URL) -> URL {
        modelDirectory.appendingPathComponent(artifactFileName)
    }

    /// Loads only `lora_edge0_35b.safetensors` from the model directory.
    public static func load(
        from modelDirectory: URL,
        configuration: Edge0Qwen35RecoverLoRAConfiguration = .edge0_35b
    ) throws -> Edge0Qwen35RecoverLoRA {
        let url = artifactURL(in: modelDirectory)
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw Edge0Qwen35RecoverLoRAArtifactError.missingArtifact(url)
        }
        return try load(fromArtifact: url, configuration: configuration)
    }

    public static func load(
        fromArtifact url: URL,
        configuration: Edge0Qwen35RecoverLoRAConfiguration = .edge0_35b
    ) throws -> Edge0Qwen35RecoverLoRA {
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw Edge0Qwen35RecoverLoRAArtifactError.missingArtifact(url)
        }
        let (weights, metadata) = try MLX.loadArraysAndMetadata(url: url)
        try validateEdge0Qwen35ArtifactMetadata(metadata, kind: .lora)
        return try load(weights: weights, configuration: configuration)
    }

    public static func load(
        weights: [String: MLXArray],
        configuration: Edge0Qwen35RecoverLoRAConfiguration = .edge0_35b
    ) throws -> Edge0Qwen35RecoverLoRA {
        var aByPath = [String: MLXArray]()
        var bByPath = [String: MLXArray]()

        for (name, weight) in weights {
            if name.hasSuffix(".lora_A") {
                let path = String(name.dropLast(".lora_A".count))
                guard !path.isEmpty else {
                    throw Edge0Qwen35RecoverLoRAArtifactError.unexpectedTensor(name)
                }
                aByPath[path] = weight
            } else if name.hasSuffix(".lora_B") {
                let path = String(name.dropLast(".lora_B".count))
                guard !path.isEmpty else {
                    throw Edge0Qwen35RecoverLoRAArtifactError.unexpectedTensor(name)
                }
                bByPath[path] = weight
            } else {
                throw Edge0Qwen35RecoverLoRAArtifactError.unexpectedTensor(name)
            }
        }

        let paths = Set(aByPath.keys).union(bByPath.keys).sorted()
        if let expected = configuration.expectedTargetCount, paths.count != expected {
            throw Edge0Qwen35RecoverLoRAArtifactError.targetCountMismatch(
                expected: expected, actual: paths.count)
        }

        var targets = [Edge0Qwen35RecoverLoRATarget]()
        targets.reserveCapacity(paths.count)
        for path in paths {
            guard let loraA = aByPath[path] else {
                throw Edge0Qwen35RecoverLoRAArtifactError.missingComponent(
                    target: path, component: "A")
            }
            guard let loraB = bByPath[path] else {
                throw Edge0Qwen35RecoverLoRAArtifactError.missingComponent(
                    target: path, component: "B")
            }
            try validate(loraA, target: path, component: "A")
            try validate(loraB, target: path, component: "B")
            let expected = expectedShapes(for: path)
            guard let expected, loraA.shape == expected.a, loraB.shape == expected.b
            else {
                throw Edge0Qwen35RecoverLoRAArtifactError.invalidTargetShape(
                    target: path,
                    expectedA: expected?.a,
                    expectedB: expected?.b,
                    a: loraA.shape,
                    b: loraB.shape)
            }
            targets.append(Edge0Qwen35RecoverLoRATarget(path: path, loraA: loraA, loraB: loraB))
        }
        return Self(configuration: configuration, targets: targets)
    }

    public func adapter(for targetPath: String) -> Edge0Qwen35RecoverLoRATarget? {
        targetsByPath[targetPath]
    }

    func adapters(for targetPaths: [String]) -> [String: Edge0Qwen35RecoverLoRATarget] {
        Dictionary(
            uniqueKeysWithValues: targetPaths.compactMap { path in
                adapter(for: path).map { (path, $0) }
            })
    }

    /// Applies an adapter selected by its exact safetensors target path.
    public func applying(
        to baseOutput: MLXArray,
        input: MLXArray,
        forTarget targetPath: String
    ) throws -> MLXArray {
        guard let target = adapter(for: targetPath) else {
            throw Edge0Qwen35RecoverLoRAArtifactError.missingComponent(
                target: targetPath, component: "A/B")
        }
        return target.applying(to: baseOutput, input: input)
    }
}

private func validate(
    _ weight: MLXArray,
    target: String,
    component: String
) throws {
    guard weight.dtype == .float16 else {
        throw Edge0Qwen35RecoverLoRAArtifactError.invalidWeightDType(
            target: target, component: component, actual: weight.dtype)
    }
    guard weight.ndim == 2 else {
        throw Edge0Qwen35RecoverLoRAArtifactError.invalidWeightRank(
            target: target, component: component, actual: weight.shape)
    }
}

private func expectedShapes(for path: String) -> (a: [Int], b: [Int])? {
    let prefix = "language_model.model.layers."
    guard path.hasPrefix(prefix) else { return nil }
    let remainder = path.dropFirst(prefix.count)
    guard let separator = remainder.firstIndex(of: "."),
        let layer = Int(remainder[..<separator]), (0 ..< 40).contains(layer)
    else { return nil }
    let suffix = String(remainder[remainder.index(after: separator)...])
    let rank = Edge0Qwen35RecoverLoRAConfiguration.rank
    let isFullAttention = (layer + 1).isMultiple(of: 4)

    switch suffix {
    case "linear_attn.in_proj_qkv" where !isFullAttention:
        return ([rank, 2048], [8192, rank])
    case "linear_attn.in_proj_z" where !isFullAttention:
        return ([rank, 2048], [4096, rank])
    case "linear_attn.in_proj_a" where !isFullAttention:
        return ([rank, 2048], [32, rank])
    case "linear_attn.in_proj_b" where !isFullAttention:
        return ([rank, 2048], [32, rank])
    case "linear_attn.out_proj" where !isFullAttention:
        return ([rank, 4096], [2048, rank])
    case "self_attn.q_proj" where isFullAttention:
        return ([rank, 2048], [8192, rank])
    case "self_attn.k_proj" where isFullAttention:
        return ([rank, 2048], [512, rank])
    case "self_attn.v_proj" where isFullAttention:
        return ([rank, 2048], [512, rank])
    case "self_attn.o_proj" where isFullAttention:
        return ([rank, 4096], [2048, rank])
    case "mlp.shared_expert.gate_proj", "mlp.shared_expert.up_proj":
        return ([rank, 2048], [512, rank])
    case "mlp.shared_expert.down_proj":
        return ([rank, 512], [2048, rank])
    default:
        return nil
    }
}
