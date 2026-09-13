// Copyright © 2026 Faux Fox.

import Foundation
import MLX

/// Immutable identity for the Edge0 model and implementation this profile supports.
public struct Edge0Qwen35InferenceIdentity: Sendable, Equatable {
    public let modelID: String
    public let modelRevision: String
    public let implementationID: String
    public let implementationRevision: String

    public init(
        modelID: String,
        modelRevision: String,
        implementationID: String,
        implementationRevision: String
    ) {
        self.modelID = modelID
        self.modelRevision = modelRevision
        self.implementationID = implementationID
        self.implementationRevision = implementationRevision
    }
}

/// The explicit Edge0-35B deployment contract.
///
/// This profile deliberately does not install a router replacement. It records
/// the data an attached streamed-Qwen runtime would need, while keeping layer
/// 39 on its real router. The published artifact has a head for owner 38, but
/// this profile does not consume that terminal prediction.
public struct Edge0Qwen35InferenceProfile: Sendable, Equatable {
    public static let edge0_35b = Self(
        identity: .init(
            modelID: "Edge0/Edge0-35B-A3B-preview",
            modelRevision: "1ff9f4478890faec0368c5463b621d1036d5b518",
            implementationID: "https://github.com/Edge0-AI/Edge0",
            implementationRevision: "ae1ee2d343f88d6a9d3c9304d6a853353041a382"),
        prerouterConfiguration: .edge0_35b)

    public let identity: Edge0Qwen35InferenceIdentity
    public let prerouterConfiguration: Edge0Qwen35PrerouterConfiguration

    /// Artifact owners. The sidecar contains 33 heads, owners 6 through 38.
    public var artifactOwners: [Int] { prerouterConfiguration.owners }
    /// Consumers whose decode routing may be replaced by a staged prediction.
    public var predictedConsumers: [Int] { Array(7 ... 38) }
    /// Owner 38's artifact head is never consumed under this terminal policy.
    public var terminalRouterLayer: Int { 39 }
    public var topK: Int { 4 }
    public var supportsMTP: Bool { false }
    public var supportsBatching: Bool { false }
    public var hiddenClamp: ClosedRange<Float> { -1000 ... 1000 }

    public init(
        identity: Edge0Qwen35InferenceIdentity,
        prerouterConfiguration: Edge0Qwen35PrerouterConfiguration
    ) {
        self.identity = identity
        self.prerouterConfiguration = prerouterConfiguration
    }

    public func validate() throws {
        let configuration = prerouterConfiguration
        guard identity == Self.edge0_35b.identity,
            configuration.layerCount == 40,
            configuration.hiddenSize == 2048,
            configuration.expertCount == 256,
            configuration.topK == topK,
            configuration.prerouterHiddenSize == 512,
            configuration.owners == Array(6 ... 38),
            predictedConsumers == Array(7 ... 38),
            terminalRouterLayer == configuration.layerCount - 1
        else {
            throw Edge0Qwen35InferenceProfileError.invalidProfile
        }
    }

    /// Loads only the pinned sidecar name from a checkpoint directory, then
    /// verifies that its fixed tensor layout belongs to this deployment tier.
    public func loadPrerouter(from modelDirectory: URL) throws -> Edge0Qwen35Prerouter {
        try validate()
        let prerouter = try Edge0Qwen35Prerouter.load(
            from: modelDirectory, configuration: prerouterConfiguration)
        try validateAttachment(prerouter)
        return prerouter
    }

    /// Rejects a sidecar attached to another Edge0 tier or a modified profile.
    public func validateAttachment(_ prerouter: Edge0Qwen35Prerouter) throws {
        try validate()
        guard prerouter.configuration == prerouterConfiguration else {
            throw Edge0Qwen35InferenceProfileError.prerouterConfigurationMismatch
        }
    }

    public func validateRequest(batchSize: Int, mtpEnabled: Bool) throws {
        guard batchSize == 1 else {
            throw Edge0Qwen35InferenceProfileError.batchUnsupported(batchSize)
        }
        guard !mtpEnabled else {
            throw Edge0Qwen35InferenceProfileError.mtpUnsupported
        }
    }

    /// Clips finite values to the deployment guard band and turns every NaN or
    /// infinity into a finite value before it can poison subsequent routers.
    public func stabilizedHidden(_ hidden: MLXArray) -> MLXArray {
        MLX.clip(
            MLX.nanToNum(
                hidden, nan: 0, posInf: hiddenClamp.upperBound, negInf: hiddenClamp.lowerBound),
            min: hiddenClamp.lowerBound, max: hiddenClamp.upperBound)
    }
}

public enum Edge0Qwen35InferenceProfileError: Error, Equatable, LocalizedError {
    case invalidProfile
    case prerouterConfigurationMismatch
    case batchUnsupported(Int)
    case mtpUnsupported
    case invalidPredictedConsumer(Int)

    public var errorDescription: String? {
        switch self {
        case .invalidProfile:
            "The Edge0 Qwen35 inference profile does not match its pinned deployment contract"
        case .prerouterConfigurationMismatch:
            "The prerouter sidecar does not match the Edge0 Qwen35 inference profile"
        case .batchUnsupported(let batchSize):
            "Edge0 Qwen35 prerouting requires batch size 1, got \(batchSize)"
        case .mtpUnsupported:
            "Edge0 Qwen35 prerouting cannot be combined with MTP verification"
        case .invalidPredictedConsumer(let layer):
            "Layer \(layer) is not an Edge0 Qwen35 predicted consumer"
        }
    }
}

public enum Edge0Qwen35InferenceStateDisposition: Sendable, Equatable {
    case fresh
    case prefixReset
    case checkpointReset
}

enum Edge0Qwen35ForwardPhase: Equatable {
    case prefillRealRouter
    case prefillDecodeLike
    case decode
}

/// Tracks Edge0's pinned 2048-token logical prefill chunks independently of
/// the scheduler's smaller microbatches.
final class Edge0Qwen35PhaseController {
    static let pinnedPrefillChunkSize = 2048

    private var promptTokenCount = 0
    private var prefillTokenOffset = 0

    func begin(promptTokenCount: Int) {
        precondition(promptTokenCount >= 0)
        self.promptTokenCount = promptTokenCount
        prefillTokenOffset = 0
    }

    func prefillPhase(forwardTokenCount: Int) -> Edge0Qwen35ForwardPhase {
        precondition(forwardTokenCount > 0)
        defer { prefillTokenOffset += forwardTokenCount }

        let chunkStart =
            prefillTokenOffset / Self.pinnedPrefillChunkSize
            * Self.pinnedPrefillChunkSize
        let logicalChunkCount = min(
            Self.pinnedPrefillChunkSize, max(0, promptTokenCount - chunkStart))
        return logicalChunkCount == 1 && forwardTokenCount == 1
            ? .prefillDecodeLike : .prefillRealRouter
    }

    var decodePhase: Edge0Qwen35ForwardPhase { .decode }

    func reset() {
        promptTokenCount = 0
        prefillTokenOffset = 0
    }
}

/// Per-request state that makes the profile attachable without changing Qwen.
///
/// A runtime records actual executed experts before staging. At a forward-step
/// boundary, call ``advance()``. Prefix reuse and checkpoint restoration must
/// reset this state because neither carries safe cross-token prerouter tensors.
public final class Edge0Qwen35InferenceRequestState {
    public let profile: Edge0Qwen35InferenceProfile
    public let prerouterState: Edge0Qwen35PrerouterState
    public private(set) var disposition: Edge0Qwen35InferenceStateDisposition = .fresh

    private var currentExecuted: [MLXArray?]
    private var previousExecuted: [MLXArray?]
    private var currentPredicted: [MLXArray?]
    private var previousPredicted: [MLXArray?]

    public init(profile: Edge0Qwen35InferenceProfile = .edge0_35b) throws {
        try profile.validate()
        self.profile = profile
        self.prerouterState = Edge0Qwen35PrerouterState(
            configuration: profile.prerouterConfiguration)
        let count = profile.prerouterConfiguration.layerCount
        currentExecuted = Array(repeating: nil, count: count)
        previousExecuted = Array(repeating: nil, count: count)
        currentPredicted = Array(repeating: nil, count: count)
        previousPredicted = Array(repeating: nil, count: count)
    }

    public func recordExecuted(_ expertIDs: MLXArray, for layer: Int) throws {
        try prerouterState.recordExecuted(expertIDs, for: layer)
        currentExecuted[layer] = oneHot(
            expertIDs, expertCount: profile.prerouterConfiguration.expertCount)
    }

    public func previousExecutedOneHot(for layer: Int) -> MLXArray? {
        previousExecuted.indices.contains(layer) ? previousExecuted[layer] : nil
    }

    public func previousPredictedOneHot(forConsumer layer: Int) throws -> MLXArray? {
        try validatePredictedConsumer(layer)
        return previousPredicted[layer]
    }

    public func prediction(forConsumer layer: Int) throws -> Edge0Qwen35PrerouterPrediction? {
        try validatePredictedConsumer(layer)
        return prerouterState.prediction(forConsumer: layer)
    }

    /// Stages only owners 6...37. Artifact owner 38 is intentionally excluded,
    /// leaving terminal layer 39 on its ordinary Qwen router.
    public func stage(
        prerouter: Edge0Qwen35Prerouter,
        hiddenByOwner: [Int: MLXArray]
    ) throws {
        try profile.validateAttachment(prerouter)
        let permittedOwners = Set(profile.predictedConsumers.map { $0 - 1 })
        let permittedHidden = hiddenByOwner.filter { permittedOwners.contains($0.key) }
        try prerouter.stage(state: prerouterState, hiddenByOwner: permittedHidden)
        for consumer in profile.predictedConsumers {
            guard let prediction = prerouterState.prediction(forConsumer: consumer) else {
                continue
            }
            currentPredicted[consumer] = oneHot(
                prediction.expertIDs, expertCount: profile.prerouterConfiguration.expertCount)
        }
    }

    /// Rolls per-token features while retaining pending inner predictions for
    /// the next forward. This is the only valid boundary transition.
    public func advance() {
        prerouterState.swap()
        previousExecuted = currentExecuted
        currentExecuted = Array(repeating: nil, count: currentExecuted.count)
        previousPredicted = currentPredicted
        currentPredicted = Array(repeating: nil, count: currentPredicted.count)
    }

    public func resetForNewRequest() {
        reset(disposition: .fresh)
    }

    public func resetForPrefixReuse() {
        reset(disposition: .prefixReset)
    }

    /// Checkpoints exclude transient MLX prerouter tensors. Call after a cache
    /// restore before any new decode token is attached to the request.
    public func resetForCheckpointRestore() {
        reset(disposition: .checkpointReset)
    }

    private func reset(disposition: Edge0Qwen35InferenceStateDisposition) {
        prerouterState.reset()
        currentExecuted = Array(repeating: nil, count: currentExecuted.count)
        previousExecuted = Array(repeating: nil, count: previousExecuted.count)
        currentPredicted = Array(repeating: nil, count: currentPredicted.count)
        previousPredicted = Array(repeating: nil, count: previousPredicted.count)
        self.disposition = disposition
    }

    private func validatePredictedConsumer(_ layer: Int) throws {
        guard profile.predictedConsumers.contains(layer) else {
            throw Edge0Qwen35InferenceProfileError.invalidPredictedConsumer(layer)
        }
    }
}

private func oneHot(_ expertIDs: MLXArray, expertCount: Int) -> MLXArray {
    let expertAxis = MLXArray(0 ..< expertCount).asType(expertIDs.dtype)
    return (expertIDs[.ellipsis, .newAxis] .== expertAxis).asType(.float16).sum(axis: -2)
}
