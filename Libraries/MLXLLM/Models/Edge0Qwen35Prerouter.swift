// Copyright © 2026 Faux Fox.

import Foundation
import MLX
import MLXNN

/// The fixed layout of Edge0's Qwen 3.5 35B trained prerouter artifact.
///
/// A head owned by layer `N` predicts the four experts consumed by `N + 1`.
/// Layer 39 remains on its real router even though owner 38 exists in the
/// artifact and upstream computes its unused terminal prediction.
public struct Edge0Qwen35PrerouterConfiguration: Sendable, Equatable {
    public let layerCount: Int
    public let hiddenSize: Int
    public let expertCount: Int
    public let topK: Int
    public let prerouterHiddenSize: Int
    public let owners: [Int]

    public static let edge0_35b = Self(
        layerCount: 40,
        hiddenSize: 2048,
        expertCount: 256,
        topK: 4,
        prerouterHiddenSize: 512,
        owners: Array(6 ... 38))

    public init(
        layerCount: Int,
        hiddenSize: Int,
        expertCount: Int,
        topK: Int,
        prerouterHiddenSize: Int,
        owners: [Int]
    ) {
        self.layerCount = layerCount
        self.hiddenSize = hiddenSize
        self.expertCount = expertCount
        self.topK = topK
        self.prerouterHiddenSize = prerouterHiddenSize
        self.owners = owners
    }

    var featureSize: Int { hiddenSize + 2 * expertCount }
}

public enum Edge0Qwen35PrerouterArtifactError: Error, Equatable, LocalizedError {
    case missingArtifact(URL)
    case missingWeight(owner: Int, component: String)
    case invalidWeightShape(name: String, expected: [Int], actual: [Int])
    case invalidWeightDType(name: String, actual: DType)
    case invalidOwner(owner: Int)
    case invalidExecutedShape([Int])
    case missingCurrentExecuted(owner: Int)
    case invalidHiddenShape(owner: Int, expected: Int, actual: [Int])

    public var errorDescription: String? {
        switch self {
        case .missingArtifact(let url):
            "Edge0 prerouter artifact is missing: \(url.path)"
        case .missingWeight(let owner, let component):
            "Edge0 prerouter is missing layers.\(owner).\(component).weight"
        case .invalidWeightShape(let name, let expected, let actual):
            "Edge0 prerouter weight \(name) has shape \(actual), expected \(expected)"
        case .invalidWeightDType(let name, let actual):
            "Edge0 prerouter weight \(name) has non-floating dtype \(actual)"
        case .invalidOwner(let owner):
            "Layer \(owner) does not own an Edge0 prerouter head"
        case .invalidExecutedShape(let shape):
            "Executed expert IDs must have shape [batch, tokens, topK], got \(shape)"
        case .missingCurrentExecuted(let owner):
            "Layer \(owner) has no executed expert feature for this token"
        case .invalidHiddenShape(let owner, let expected, let actual):
            "Layer \(owner) hidden state has shape \(actual), expected trailing width \(expected)"
        }
    }
}

/// A trained Edge0 pre-routing head.
///
/// The artifact is fp16. Inputs are converted to fp16 before either branch,
/// matching Edge0's MLX implementation and its PyTorch training export.
public final class Edge0Qwen35PrerouterHead: Module {
    @ModuleInfo(key: "fc1") public var fc1: Linear
    @ModuleInfo(key: "fc2") public var fc2: Linear
    @ModuleInfo(key: "linear_init") public var linearInit: Linear

    public let configuration: Edge0Qwen35PrerouterConfiguration

    public init(configuration: Edge0Qwen35PrerouterConfiguration) {
        self.configuration = configuration
        _fc1.wrappedValue = Linear(
            configuration.featureSize, configuration.prerouterHiddenSize, bias: false)
        _fc2.wrappedValue = Linear(
            configuration.prerouterHiddenSize, configuration.expertCount, bias: false)
        _linearInit.wrappedValue = Linear(
            configuration.featureSize, configuration.expertCount, bias: false)
        super.init()
        apply { $0.asType(.float16) }
    }

    public func callAsFunction(
        hidden: MLXArray,
        executed: MLXArray,
        previousExecuted: MLXArray
    ) -> MLXArray {
        let features = concatenated([hidden, executed, previousExecuted], axis: -1)
            .asType(.float16)
        // MLXNN.gelu uses the exact erf form, matching torch F.gelu's default.
        return linearInit(features) + fc2(MLXNN.gelu(fc1(features)))
    }
}

/// A completed selection retained for one consumer layer until it is replaced.
public struct Edge0Qwen35PrerouterPrediction {
    public let logits: MLXArray
    public let expertIDs: MLXArray
    public let scores: MLXArray
}

/// Cross-token prerouter state with Edge0's double-buffer transition.
///
/// Call ``swap()`` after staging all owner predictions at a forward-step
/// boundary. Pending predictions deliberately survive the swap; they are
/// consumed by the next token and overwritten by the next boundary.
public final class Edge0Qwen35PrerouterState {
    public let configuration: Edge0Qwen35PrerouterConfiguration

    private var executed: [MLXArray?]
    private var previousExecuted: [MLXArray?]
    private var predictions: [Edge0Qwen35PrerouterPrediction?]

    public init(configuration: Edge0Qwen35PrerouterConfiguration = .edge0_35b) {
        self.configuration = configuration
        self.executed = Array(repeating: nil, count: configuration.layerCount)
        self.previousExecuted = Array(repeating: nil, count: configuration.layerCount)
        self.predictions = Array(repeating: nil, count: configuration.layerCount)
    }

    /// Records the actual executed router IDs for one layer, rather than gate
    /// teacher IDs. This is the feature convention trained by Edge0 35B.
    public func recordExecuted(_ expertIDs: MLXArray, for layer: Int) throws {
        try validateLayer(layer)
        guard expertIDs.ndim == 3, expertIDs.dim(-1) == configuration.topK else {
            throw Edge0Qwen35PrerouterArtifactError.invalidExecutedShape(expertIDs.shape)
        }
        let expertAxis = MLXArray(0 ..< configuration.expertCount).asType(expertIDs.dtype)
        let oneHot = (expertIDs[.ellipsis, .newAxis] .== expertAxis)
            .asType(.float16)
            .sum(axis: -2)
        executed[layer] = oneHot
    }

    public func currentExecuted(for owner: Int) throws -> MLXArray {
        try validateOwner(owner)
        guard let value = executed[owner] else {
            throw Edge0Qwen35PrerouterArtifactError.missingCurrentExecuted(owner: owner)
        }
        return value
    }

    public func previousExecuted(for owner: Int, matching current: MLXArray) throws -> MLXArray {
        try validateOwner(owner)
        return previousExecuted[owner]
            ?? MLXArray.zeros(current.shape, dtype: current.dtype)
    }

    public func prediction(forConsumer layer: Int) -> Edge0Qwen35PrerouterPrediction? {
        guard predictions.indices.contains(layer) else { return nil }
        return predictions[layer]
    }

    /// Advances only the executed-feature buffers. Do not clear predictions:
    /// an owner at token t stages its consumer's selection for token t + 1.
    public func swap() {
        previousExecuted = executed
        executed = Array(repeating: nil, count: configuration.layerCount)
    }

    public func reset() {
        executed = Array(repeating: nil, count: configuration.layerCount)
        previousExecuted = Array(repeating: nil, count: configuration.layerCount)
        predictions = Array(repeating: nil, count: configuration.layerCount)
    }

    fileprivate func store(_ prediction: Edge0Qwen35PrerouterPrediction, from owner: Int) {
        predictions[owner + 1] = prediction
    }

    private func validateLayer(_ layer: Int) throws {
        guard executed.indices.contains(layer) else {
            throw Edge0Qwen35PrerouterArtifactError.invalidOwner(owner: layer)
        }
    }

    private func validateOwner(_ owner: Int) throws {
        guard configuration.owners.contains(owner) else {
            throw Edge0Qwen35PrerouterArtifactError.invalidOwner(owner: owner)
        }
    }
}

/// Loads and stages Edge0's `prerouter_edge0_35b.safetensors` heads.
///
/// This is intentionally standalone. It exposes the output needed by the
/// streamed Qwen runtime without changing the generic Qwen router path.
public final class Edge0Qwen35Prerouter {
    public static let artifactFileName = "prerouter_edge0_35b.safetensors"

    public let configuration: Edge0Qwen35PrerouterConfiguration
    private(set) var heads: [Int: Edge0Qwen35PrerouterHead]

    public init(configuration: Edge0Qwen35PrerouterConfiguration = .edge0_35b) {
        self.configuration = configuration
        self.heads = Dictionary(
            uniqueKeysWithValues: configuration.owners.map {
                ($0, Edge0Qwen35PrerouterHead(configuration: configuration))
            })
    }

    public static func load(
        from modelDirectory: URL,
        configuration: Edge0Qwen35PrerouterConfiguration = .edge0_35b
    ) throws -> Edge0Qwen35Prerouter {
        let url = modelDirectory.appendingPathComponent(artifactFileName)
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw Edge0Qwen35PrerouterArtifactError.missingArtifact(url)
        }
        return try load(fromArtifact: url, configuration: configuration)
    }

    public static func load(
        fromArtifact url: URL,
        configuration: Edge0Qwen35PrerouterConfiguration = .edge0_35b
    ) throws -> Edge0Qwen35Prerouter {
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw Edge0Qwen35PrerouterArtifactError.missingArtifact(url)
        }
        let (weights, metadata) = try MLX.loadArraysAndMetadata(url: url)
        try validateEdge0Qwen35ArtifactMetadata(
            metadata, kind: .prerouter, owners: configuration.owners)
        return try load(weights: weights, configuration: configuration)
    }

    public static func load(
        weights: [String: MLXArray],
        configuration: Edge0Qwen35PrerouterConfiguration = .edge0_35b
    ) throws -> Edge0Qwen35Prerouter {
        let prerouter = Edge0Qwen35Prerouter(configuration: configuration)
        for owner in configuration.owners {
            guard let head = prerouter.heads[owner] else { continue }
            let parameters = [
                "fc1.weight": try validatedWeight(
                    weights, owner: owner, component: "fc1",
                    shape: [configuration.prerouterHiddenSize, configuration.featureSize]),
                "fc2.weight": try validatedWeight(
                    weights, owner: owner, component: "fc2",
                    shape: [configuration.expertCount, configuration.prerouterHiddenSize]),
                "linear_init.weight": try validatedWeight(
                    weights, owner: owner, component: "linear_init",
                    shape: [configuration.expertCount, configuration.featureSize]),
            ]
            try head.update(parameters: ModuleParameters.unflattened(parameters), verify: [.all])
        }
        return prerouter
    }

    public func stage(
        state: Edge0Qwen35PrerouterState,
        hiddenByOwner: [Int: MLXArray]
    ) throws {
        precondition(state.configuration == configuration, "mismatched Edge0 prerouter state")
        for owner in configuration.owners {
            guard let hidden = hiddenByOwner[owner] else { continue }
            guard hidden.ndim == 3, hidden.dim(-1) == configuration.hiddenSize else {
                throw Edge0Qwen35PrerouterArtifactError.invalidHiddenShape(
                    owner: owner, expected: configuration.hiddenSize, actual: hidden.shape)
            }
            guard let head = heads[owner] else { continue }
            let current = try state.currentExecuted(for: owner)
            let previous = try state.previousExecuted(for: owner, matching: current)
            let logits = head(hidden: hidden, executed: current, previousExecuted: previous)
            let gates = MLX.softmax(logits, axis: -1, precise: true)
            let kth = configuration.expertCount - configuration.topK
            let expertIDs = MLX.argPartition(gates, kth: kth, axis: -1)[.ellipsis, kth...]
            let selected = MLX.takeAlong(gates, expertIDs, axis: -1)
            let scores = selected / selected.sum(axis: -1, keepDims: true)
            state.store(
                Edge0Qwen35PrerouterPrediction(
                    logits: logits, expertIDs: expertIDs, scores: scores),
                from: owner)
        }
    }
}

private func validatedWeight(
    _ weights: [String: MLXArray],
    owner: Int,
    component: String,
    shape: [Int]
) throws -> MLXArray {
    let names = [
        "layers.\(owner).\(component).weight",
        "layers.\(owner).mlp.prerouter.\(component).weight",
    ]
    guard let name = names.first(where: { weights[$0] != nil }), let weight = weights[name] else {
        throw Edge0Qwen35PrerouterArtifactError.missingWeight(owner: owner, component: component)
    }
    guard weight.dtype.isFloatingPoint else {
        throw Edge0Qwen35PrerouterArtifactError.invalidWeightDType(name: name, actual: weight.dtype)
    }
    guard weight.shape == shape else {
        throw Edge0Qwen35PrerouterArtifactError.invalidWeightShape(
            name: name, expected: shape, actual: weight.shape)
    }
    return weight.asType(.float16)
}
