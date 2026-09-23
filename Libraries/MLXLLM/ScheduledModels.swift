// Copyright © 2026 Faux Fox.

import Foundation
import MLX
import MLXLMCommon

func qwen35ScheduledRecurrentStateBytes(
    _ configuration: Qwen35TextConfiguration, verificationBlockSize: Int
) -> Int {
    let layers =
        configuration.hiddenLayers
        - configuration.hiddenLayers / configuration.fullAttentionInterval
    let convDimensions =
        2 * configuration.linearNumKeyHeads * configuration.linearKeyHeadDim
        + configuration.linearNumValueHeads * configuration.linearValueHeadDim
    let state =
        configuration.linearNumValueHeads * configuration.linearValueHeadDim
        * configuration.linearKeyHeadDim
    return layers * ((configuration.linearConvKernelDim - 1) * convDimensions + state)
        * MemoryLayout<Float>.size * Swift.max(verificationBlockSize, 1)
}

extension LlamaModel {
    public var scheduledSupportsBatchDecode: Bool { true }
}

extension Qwen3Model: ScheduledTextModel {
    public var scheduledCacheBytesPerToken: Int {
        configuration.hiddenLayers * configuration.kvHeads * configuration.headDim * 8
    }
    public var scheduledSupportsBatchDecode: Bool { true }
    public nonisolated(nonsending) func scheduledForward(
        _ tokens: MLXArray,
        cache: [KVCache]
    ) async throws -> MLXArray {
        callAsFunction(tokens, cache: cache)
    }
}

extension Qwen35TextModel: ScheduledTextModel {
    public var scheduledMTPArchitectureID: String? {
        usesEdge0 ? nil : "qwen3_5:\(configuration.hiddenSize)"
    }
    public var scheduledCacheBytesPerToken: Int {
        let attentionLayers = configuration.hiddenLayers / configuration.fullAttentionInterval
        return attentionLayers * configuration.kvHeads
            * (configuration.headDim ?? configuration.hiddenSize / configuration.attentionHeads) * 8
    }
    public var scheduledRecurrentStateBytes: Int {
        scheduledRecurrentStateBytes(forVerificationBlockSize: 4)
    }
    public func scheduledRecurrentStateBytes(forVerificationBlockSize verificationBlockSize: Int)
        -> Int
    {
        qwen35ScheduledRecurrentStateBytes(
            configuration, verificationBlockSize: verificationBlockSize)
    }
    public var scheduledSupportsBatchDecode: Bool { !usesStreamedExperts }
    public var scheduledAdditionalResidentWeightBytes: Int {
        streamedExpertResidentWeightBytes
    }
    public var scheduledRequiresExclusiveExecution: Bool { usesStreamedExperts }
    public var scheduledMaximumForwardTokens: Int? {
        usesEdge0 ? 1 : usesStreamedExperts ? 2 : nil
    }
    public var scheduledSupportsPrefixCache: Bool { !usesEdge0 }
    public var scheduledFirstTokenGreedy: Bool { usesEdge0 }
    public func scheduledBeginRequest() throws { beginEdge0Request() }
    public func scheduledBeginRequest(promptTokenCount: Int) throws {
        beginEdge0Request(promptTokenCount: promptTokenCount)
    }
    public func scheduledEndRequest() { endEdge0Request() }
    public nonisolated(nonsending) func scheduledForward(
        _ tokens: MLXArray,
        cache: [KVCache]
    ) async throws -> MLXArray {
        if usesStreamedExperts {
            return try await streamedScheduledForward(tokens, cache: cache)
        }
        return callAsFunction(tokens, cache: cache)
    }
}

extension Qwen35Model: ScheduledTextModel {
    public var scheduledMTPArchitectureID: String? { languageModel.scheduledMTPArchitectureID }
    public var scheduledCacheBytesPerToken: Int { languageModel.scheduledCacheBytesPerToken }
    public var scheduledRecurrentStateBytes: Int { languageModel.scheduledRecurrentStateBytes }
    public func scheduledRecurrentStateBytes(forVerificationBlockSize verificationBlockSize: Int)
        -> Int
    {
        languageModel.scheduledRecurrentStateBytes(
            forVerificationBlockSize: verificationBlockSize)
    }
    public var scheduledSupportsBatchDecode: Bool { !usesStreamedExperts }
    public var scheduledAdditionalResidentWeightBytes: Int {
        streamedExpertResidentWeightBytes
    }
    public var scheduledRequiresExclusiveExecution: Bool { usesStreamedExperts }
    public var scheduledMaximumForwardTokens: Int? {
        usesEdge0 ? 1 : usesStreamedExperts ? 2 : nil
    }
    public var scheduledSupportsPrefixCache: Bool { !usesEdge0 }
    public var scheduledFirstTokenGreedy: Bool { usesEdge0 }
    public func scheduledBeginRequest() throws { beginEdge0Request() }
    public func scheduledBeginRequest(promptTokenCount: Int) throws {
        beginEdge0Request(promptTokenCount: promptTokenCount)
    }
    public func scheduledEndRequest() { endEdge0Request() }
    public nonisolated(nonsending) func scheduledForward(
        _ tokens: MLXArray,
        cache: [KVCache]
    ) async throws -> MLXArray {
        if usesStreamedExperts {
            return try await streamedScheduledForward(tokens, cache: cache)
        }
        return callAsFunction(tokens, cache: cache)
    }
}

/// Chooses whether routed Qwen 3.5 MoE experts remain in the model or are read
/// from the installed checkpoint when decode routes select them.
public enum NativeTextModelLoadPolicy: Sendable, Equatable {
    /// Preserve the established eager loading behavior.
    case resident

    /// Keep dense weights resident and retain only a bounded expert cache per layer.
    case streamedExperts(StreamedExpertLoadConfiguration)

    /// Load the pinned Edge0 35B profile with K=4, Recover-LoRA and prerouting.
    case edge0(Edge0Qwen35LoadConfiguration)
}

/// Bounded residency configuration for ``NativeTextModelLoadPolicy/streamedExperts(_:)``.
public struct StreamedExpertLoadConfiguration: Sendable, Equatable {
    /// Maximum bytes retained by each routed-expert layer. A miss is usable even
    /// when this is zero; it is released after that decode plan completes.
    public let maximumResidentBytesPerLayer: Int

    public init(maximumResidentBytesPerLayer: Int) {
        self.maximumResidentBytesPerLayer = maximumResidentBytesPerLayer
    }
}

/// Bounded routed-expert residency for the pinned Edge0 35B deployment.
public struct Edge0Qwen35LoadConfiguration: Sendable, Equatable {
    public let maximumResidentBytesPerLayer: Int

    public init(maximumResidentBytesPerLayer: Int) {
        self.maximumResidentBytesPerLayer = maximumResidentBytesPerLayer
    }
}

/// Header-derived weight accounting for a native text-model load.
///
/// `denseWeightBytes` and `routedExpertWeightBytes` describe checkpoint payload
/// bytes for the target text model. They do not include transient MLX workspaces.
public struct NativeTextModelLoadPlan: Sendable, Equatable {
    public let policy: NativeTextModelLoadPolicy
    public let denseWeightBytes: Int
    public let routedExpertWeightBytes: Int
    public let maximumResidentExpertBytes: Int

    /// The maximum persistent model-weight payload for this policy.
    public var maximumResidentWeightBytes: Int {
        denseWeightBytes + maximumResidentExpertBytes
    }

    public init(
        policy: NativeTextModelLoadPolicy, denseWeightBytes: Int,
        routedExpertWeightBytes: Int, maximumResidentExpertBytes: Int
    ) {
        self.policy = policy
        self.denseWeightBytes = denseWeightBytes
        self.routedExpertWeightBytes = routedExpertWeightBytes
        self.maximumResidentExpertBytes = maximumResidentExpertBytes
    }
}

/// Errors reported when an opt-in streamed expert load cannot preserve the
/// checkpoint's routed-expert layout exactly.
public enum NativeTextModelLoadingError: Error, Equatable, Sendable {
    case invalidStreamedExpertConfiguration
    case streamedExpertsUnsupported(String)
    case incompatibleEdge0Checkpoint
}

/// Local-only weight loader for the runtime's audited text architectures. FMLXText's
/// NativeTextModel pairs it with checkpoint-specific tokenization. Neither downloads assets.
public enum NativeTextModelLoader {
    /// Inspects installed safetensor headers without materializing model arrays.
    public static func loadPlan(
        directory: URL, policy: NativeTextModelLoadPolicy = .resident
    ) throws -> NativeTextModelLoadPlan {
        let data = try Data(contentsOf: directory.appendingPathComponent("config.json"))
        let base = try JSONDecoder().decode(BaseConfiguration.self, from: data)

        let streamingConfiguration: StreamedExpertLoadConfiguration?
        switch policy {
        case .resident:
            streamingConfiguration = nil
        case .streamedExperts(let configuration):
            streamingConfiguration = configuration
        case .edge0(let configuration):
            streamingConfiguration = .init(
                maximumResidentBytesPerLayer: configuration.maximumResidentBytesPerLayer)
        }
        guard let configuration = streamingConfiguration else {
            let bytes = try Qwen35StreamedExpertCheckpoint.totalTargetWeightBytes(
                directory: directory)
            return NativeTextModelLoadPlan(
                policy: policy, denseWeightBytes: bytes.dense,
                routedExpertWeightBytes: bytes.routedExperts,
                maximumResidentExpertBytes: bytes.routedExperts)
        }
        guard configuration.maximumResidentBytesPerLayer >= 0 else {
            throw NativeTextModelLoadingError.invalidStreamedExpertConfiguration
        }
        var qwen = try JSONDecoder().decode(Qwen35Configuration.self, from: data)
        if case .edge0 = policy {
            try validateEdge0Checkpoint(
                baseModelType: base.modelType, configuration: qwen.textConfig)
            qwen.textConfig.numExpertsPerTok = Edge0Qwen35InferenceProfile.edge0_35b.topK
        }
        let checkpoint = try Qwen35StreamedExpertCheckpoint(
            directory: directory, baseModelType: base.modelType,
            configuration: qwen.textConfig,
            quantization: base.perLayerQuantization?.quantization)
        let resident = checkpoint.layerExpertBytes.reduce(0) { partial, layerBytes in
            partial + min(layerBytes, configuration.maximumResidentBytesPerLayer)
        }
        return NativeTextModelLoadPlan(
            policy: policy, denseWeightBytes: checkpoint.denseWeightBytes,
            routedExpertWeightBytes: checkpoint.routedExpertWeightBytes,
            maximumResidentExpertBytes: resident)
    }

    /// Loads a preserved head from the same checkpoint as the target. A config flag
    /// alone is insufficient: ordinary MLX conversions often strip all head tensors.
    public static func loadCombinedMTP(directory: URL) async throws -> sending Qwen35MTPDraftModel?
    {
        let data = try Data(contentsOf: directory.appendingPathComponent("config.json"))
        let base = try JSONDecoder().decode(BaseConfiguration.self, from: data)
        guard ["qwen3_5", "qwen3_5_moe", "qwen3_5_text"].contains(base.modelType) else {
            return nil
        }
        let configuration = try JSONDecoder().decode(Qwen35Configuration.self, from: data)
        guard configuration.textConfig.mtpNumHiddenLayers > 0 else { return nil }
        let mtpWeights = WeightTensorNameSelection.prefixed("mtp.")
        let containsHead = try containsWeightTensor(
            in: directory, tensorNameSelection: mtpWeights)
        guard containsHead else { return nil }
        let object = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        let quantization = object?["quantization"] as? [String: Any]
        let mixed =
            configuration.mixedPreservedNorms
            && quantization?["mtp"] as? String == "preserved"
        let drafter = Qwen35MTPDraftModel(
            configuration.textConfig, preconvertedNorms: base.perLayerQuantization != nil,
            mixedPreservedNorms: mixed)
        try await loadWeights(
            modelDirectory: directory, model: drafter,
            tensorNameSelection: mtpWeights,
            perLayerQuantization: base.perLayerQuantization)
        try Task.checkCancellation()
        return drafter
    }

    /// Loads a standalone, preconverted MLX Qwen MTP head; the caller supplies its target.
    public static func loadMTP(directory: URL) async throws -> sending Qwen35MTPDraftModel {
        let data = try Data(contentsOf: directory.appendingPathComponent("config.json"))
        let base = try JSONDecoder().decode(BaseConfiguration.self, from: data)
        guard base.modelType == "qwen3_5_mtp" else {
            throw ConcurrentTextRuntimeError.invalidConfiguration
        }
        let configuration = try JSONDecoder().decode(Qwen35Configuration.self, from: data)
        let model = Qwen35MTPDraftModel(configuration, preconvertedNorms: true)
        try await loadWeights(
            modelDirectory: directory, model: model,
            perLayerQuantization: base.perLayerQuantization)
        return model
    }

    /// Loads either the Qwen MTP head or the published trained DFlash2 drafter.
    public static func loadSpeculativeDrafter(directory: URL) async throws
        -> sending any IncrementalMTPDrafterModel
    {
        let data = try Data(contentsOf: directory.appendingPathComponent("config.json"))
        let base = try JSONDecoder().decode(BaseConfiguration.self, from: data)
        if base.modelType == "qwen3_5_mtp" {
            return try await loadMTP(directory: directory)
        }
        guard base.modelType == "qwen3" else {
            throw ConcurrentTextRuntimeError.invalidConfiguration
        }
        let configuration = try JSONDecoder().decode(DFlash2Configuration.self, from: data)
        try configuration.validateModelConfiguration()
        let model = DFlash2DraftModel(configuration)
        try await loadWeights(
            modelDirectory: directory, model: model,
            perLayerQuantization: base.perLayerQuantization)
        return model
    }

    /// Loads the MTP head embedded in a Qwen 3.5-family target checkpoint.
    /// Returns `nil` when the checkpoint does not declare an embedded head.
    public static func loadEmbeddedMTP(directory: URL) async throws
        -> sending Qwen35MTPDraftModel?
    {
        let data = try Data(contentsOf: directory.appendingPathComponent("config.json"))
        let base = try JSONDecoder().decode(BaseConfiguration.self, from: data)
        guard ["qwen3_5", "qwen3_5_moe"].contains(base.modelType) else { return nil }
        let configuration = try JSONDecoder().decode(Qwen35Configuration.self, from: data)
        guard configuration.textConfig.mtpNumHiddenLayers > 0 else { return nil }
        let mtpWeights = WeightTensorNameSelection.prefixed("mtp.")
        guard try containsWeightTensor(in: directory, tensorNameSelection: mtpWeights) else {
            return nil
        }
        let model = Qwen35MTPDraftModel(configuration, preconvertedNorms: true)
        try await loadWeights(
            modelDirectory: directory, model: model,
            tensorNameSelection: mtpWeights,
            perLayerQuantization: base.perLayerQuantization)
        return model
    }

    public static func load(
        directory: URL, policy: NativeTextModelLoadPolicy = .resident
    ) async throws -> sending any ScheduledTextModel {
        let data = try Data(contentsOf: directory.appendingPathComponent("config.json"))
        let base = try JSONDecoder().decode(BaseConfiguration.self, from: data)

        let streamed: StreamedExpertLoadConfiguration?
        let edge0: Bool
        switch policy {
        case .resident:
            streamed = nil
            edge0 = false
        case .streamedExperts(let configuration):
            streamed = configuration
            edge0 = false
        case .edge0(let configuration):
            streamed = .init(
                maximumResidentBytesPerLayer: configuration.maximumResidentBytesPerLayer)
            edge0 = true
        }
        if let streamed {
            guard streamed.maximumResidentBytesPerLayer >= 0 else {
                throw NativeTextModelLoadingError.invalidStreamedExpertConfiguration
            }
            guard ["qwen3_5", "qwen3_5_moe", "qwen3_5_text"].contains(base.modelType) else {
                throw NativeTextModelLoadingError.streamedExpertsUnsupported(
                    "Streaming is currently qualified only for Qwen 3.5 MoE")
            }
            var configuration = try JSONDecoder().decode(Qwen35Configuration.self, from: data)
            if edge0 {
                try validateEdge0Checkpoint(
                    baseModelType: base.modelType, configuration: configuration.textConfig)
                configuration.textConfig.numExpertsPerTok =
                    Edge0Qwen35InferenceProfile.edge0_35b.topK
            }
            let checkpoint = try Qwen35StreamedExpertCheckpoint(
                directory: directory, baseModelType: base.modelType,
                configuration: configuration.textConfig,
                quantization: base.perLayerQuantization?.quantization)
            let maximumResidentBytesPerLayer = min(
                streamed.maximumResidentBytesPerLayer, checkpoint.layerExpertBytes[0])
            let stores = try (0 ..< configuration.textConfig.hiddenLayers).map {
                try checkpoint.makeStore(
                    layer: $0, maximumResidentBytes: maximumResidentBytesPerLayer)
            }
            let selection = WeightTensorNameSelection.excluding { name in
                Qwen35StreamedExpertCheckpoint.isRoutedExpertWeight(name)
                    || name.hasPrefix("mtp.")
                    || name.hasPrefix("vision_tower")
                    || name.hasPrefix("model.visual")
            }

            switch base.modelType {
            case "qwen3_5", "qwen3_5_moe":
                let model: Qwen35Model =
                    base.modelType == "qwen3_5_moe"
                    ? Qwen35MoEModel(configuration, streamedExperts: true)
                    : Qwen35Model(configuration, streamedExperts: true)
                try await loadWeights(
                    modelDirectory: directory, model: model, tensorNameSelection: selection,
                    perLayerQuantization: base.perLayerQuantization)
                try model.configureStreamedExperts(
                    stores: stores, groupSize: checkpoint.groupSize,
                    maximumResidentBytesPerLayer: maximumResidentBytesPerLayer)
                if edge0 {
                    let profile = Edge0Qwen35InferenceProfile.edge0_35b
                    let prerouter = try profile.loadPrerouter(from: directory)
                    let recoverLoRA = try Edge0Qwen35RecoverLoRA.load(from: directory)
                    try model.attachRecoverLoRA(recoverLoRA)
                    try model.configureEdge0(profile: profile, prerouter: prerouter)
                }
                return model
            case "qwen3_5_text":
                let model = Qwen35TextModel(
                    configuration.textConfig,
                    mixedPreservedNorms: configuration.mixedPreservedNorms,
                    streamedExperts: true)
                try await loadWeights(
                    modelDirectory: directory, model: model, tensorNameSelection: selection,
                    perLayerQuantization: base.perLayerQuantization)
                try model.configureStreamedExperts(
                    stores: stores, groupSize: checkpoint.groupSize,
                    maximumResidentBytesPerLayer: maximumResidentBytesPerLayer)
                return model
            default:
                throw NativeTextModelLoadingError.streamedExpertsUnsupported(
                    "Streaming is currently qualified only for Qwen 3.5 MoE")
            }
        }

        let model: any ScheduledTextModel
        let tensorNameSelection: WeightTensorNameSelection
        switch base.modelType {
        case "llama", "mistral":
            model = LlamaModel(try JSONDecoder().decode(LlamaConfiguration.self, from: data))
            tensorNameSelection = .all
        case "qwen3":
            model = Qwen3Model(try JSONDecoder().decode(Qwen3Configuration.self, from: data))
            tensorNameSelection = .all
        case "qwen3_5", "qwen3_5_moe":
            let configuration = try JSONDecoder().decode(Qwen35Configuration.self, from: data)
            model =
                base.modelType == "qwen3_5_moe"
                ? Qwen35MoEModel(configuration) : Qwen35Model(configuration)
            tensorNameSelection = qwen35ResidentTensorSelection
        case "qwen3_5_text":
            let configuration = try JSONDecoder().decode(Qwen35Configuration.self, from: data)
            model = Qwen35TextModel(
                configuration.textConfig,
                mixedPreservedNorms: configuration.mixedPreservedNorms)
            tensorNameSelection = qwen35ResidentTensorSelection
        case "prism_hadamard_qwen35":
            let configuration = try JSONDecoder().decode(
                PrismHadamardQwen35Configuration.self, from: data)
            try configuration.validateModelConfiguration()
            model = PrismHadamardQwen35Model(configuration)
            tensorNameSelection = .all
        default:
            throw ConcurrentTextRuntimeError.unsupportedCache
        }
        try await loadWeights(
            modelDirectory: directory, model: model,
            tensorNameSelection: tensorNameSelection,
            perLayerQuantization: base.perLayerQuantization)
        return model
    }
}

/// The resident target owns neither combined-checkpoint vision tensors nor the
/// optional MTP head. NativeTextModel loads the latter into its drafter separately.
let qwen35ResidentTensorSelection = WeightTensorNameSelection.excluding { name in
    name.hasPrefix("mtp.")
        || name.hasPrefix("vision_tower")
        || name.hasPrefix("model.visual")
}

func validateEdge0Checkpoint(
    baseModelType: String, configuration: Qwen35TextConfiguration
) throws {
    let headDim =
        configuration.headDim
        ?? (configuration.attentionHeads == 0
            ? 0 : configuration.hiddenSize / configuration.attentionHeads)
    guard baseModelType == "qwen3_5_moe",
        configuration.hiddenSize == 2048,
        configuration.hiddenLayers == 40,
        configuration.vocabularySize == 248_320,
        configuration.numExperts == 256,
        configuration.numExpertsPerTok == 8,
        configuration.moeIntermediateSize == 512,
        configuration.sharedExpertIntermediateSize == 512,
        configuration.fullAttentionInterval == 4,
        configuration.attentionHeads == 16,
        configuration.kvHeads == 2,
        headDim == 256,
        configuration.linearNumValueHeads == 32,
        configuration.linearNumKeyHeads == 16,
        configuration.linearKeyHeadDim == 128,
        configuration.linearValueHeadDim == 128,
        configuration.linearConvKernelDim == 4
    else {
        throw NativeTextModelLoadingError.incompatibleEdge0Checkpoint
    }
}

/// Header-only description of the direct, outer-expert safetensor layout used
/// by Qwen 3.5 MLX MoE checkpoints.
struct Qwen35StreamedExpertCheckpoint: @unchecked Sendable {
    struct Tensor: Sendable {
        let reader: SafetensorRangeReader
        let metadata: SafetensorTensor
    }

    let configuration: Qwen35TextConfiguration
    let tensors: [String: Tensor]
    let prefixes: [String]
    let groupSize: Int
    let layerExpertBytes: [Int]
    let denseWeightBytes: Int
    let routedExpertWeightBytes: Int

    init(
        directory: URL, baseModelType: String, configuration: Qwen35TextConfiguration,
        quantization: BaseConfiguration.Quantization? = nil
    ) throws {
        guard ["qwen3_5", "qwen3_5_moe", "qwen3_5_text"].contains(baseModelType),
            configuration.numExperts > 0, configuration.numExpertsPerTok > 0,
            configuration.numExpertsPerTok <= configuration.numExperts
        else {
            throw NativeTextModelLoadingError.streamedExpertsUnsupported(
                "Streaming requires a Qwen 3.5 MoE checkpoint")
        }

        var indexed = [String: Tensor]()
        var totalTargetBytes = 0
        var routedBytes = 0
        for url in try safetensorWeightURLs(in: directory) {
            let reader = try SafetensorRangeReader(url: url)
            for metadata in reader.tensors {
                guard indexed[metadata.name] == nil else {
                    throw NativeTextModelLoadingError.streamedExpertsUnsupported(
                        "Duplicate safetensor name: \(metadata.name)")
                }
                indexed[metadata.name] = Tensor(reader: reader, metadata: metadata)
                if Self.isTargetWeight(metadata.name) {
                    let bytes = try Self.intBytes(metadata.byteCount, name: metadata.name)
                    totalTargetBytes += bytes
                    if Self.isRoutedExpertWeight(metadata.name) { routedBytes += bytes }
                }
            }
        }

        let prefixes = try (0 ..< configuration.hiddenLayers).map { layer in
            try Self.resolvePrefix(layer: layer, tensors: indexed)
        }
        let groupSize = try Self.resolveGroupSize(
            prefix: prefixes[0], tensors: indexed, configuration: configuration)
        if let quantization {
            guard quantization.bits == 4, quantization.mode == .affine,
                quantization.groupSize == groupSize
            else {
                throw NativeTextModelLoadingError.streamedExpertsUnsupported(
                    "Streaming requires affine four-bit expert quantization")
            }
        }
        let layerExpertBytes = try prefixes.map { prefix in
            try Self.bytes(for: prefix, tensors: indexed, expertCount: configuration.numExperts)
        }
        guard routedBytes == layerExpertBytes.reduce(0, +) else {
            throw NativeTextModelLoadingError.streamedExpertsUnsupported(
                "Routed expert tensors do not match the Qwen 3.5 layer layout")
        }

        self.configuration = configuration
        self.tensors = indexed
        self.prefixes = prefixes
        self.groupSize = groupSize
        self.layerExpertBytes = layerExpertBytes
        self.routedExpertWeightBytes = routedBytes
        self.denseWeightBytes = totalTargetBytes - routedBytes
    }

    func makeStore(
        layer: Int, maximumResidentBytes: Int
    ) throws -> ExpertWeightStore<Int, StreamedQuantizedExpertWeights> {
        guard layerExpertBytes.indices.contains(layer), maximumResidentBytes >= 0 else {
            throw NativeTextModelLoadingError.invalidStreamedExpertConfiguration
        }
        return try ExpertWeightStore(capacityBytes: maximumResidentBytes) { [self] expert in
            try await loadExpert(layer: layer, expert: expert)
        }
    }

    private func loadExpert(
        layer: Int, expert: Int
    ) async throws -> ExpertWeight<StreamedQuantizedExpertWeights> {
        guard prefixes.indices.contains(layer), (0 ..< configuration.numExperts).contains(expert)
        else {
            throw NativeTextModelLoadingError.streamedExpertsUnsupported("Invalid routed expert")
        }
        let prefix = prefixes[layer]
        async let gateWeight = readData(
            name: "\(prefix).gate_proj.weight", expert: expert, required: true)
        async let gateScales = readData(
            name: "\(prefix).gate_proj.scales", expert: expert, required: true)
        async let gateBiases = readData(
            name: "\(prefix).gate_proj.biases", expert: expert, required: false)
        async let upWeight = readData(
            name: "\(prefix).up_proj.weight", expert: expert, required: true)
        async let upScales = readData(
            name: "\(prefix).up_proj.scales", expert: expert, required: true)
        async let upBiases = readData(
            name: "\(prefix).up_proj.biases", expert: expert, required: false)
        async let downWeight = readData(
            name: "\(prefix).down_proj.weight", expert: expert, required: true)
        async let downScales = readData(
            name: "\(prefix).down_proj.scales", expert: expert, required: true)
        async let downBiases = readData(
            name: "\(prefix).down_proj.biases", expert: expert, required: false)

        let parts = try await (
            gateWeight, gateScales, gateBiases,
            upWeight, upScales, upBiases,
            downWeight, downScales, downBiases
        )
        let gateWeightArray = try array(name: "\(prefix).gate_proj.weight", data: parts.0!)
        let gateScalesArray = try array(name: "\(prefix).gate_proj.scales", data: parts.1!)
        let gateBiasesArray = try parts.2.map {
            try array(name: "\(prefix).gate_proj.biases", data: $0)
        }
        let upWeightArray = try array(name: "\(prefix).up_proj.weight", data: parts.3!)
        let upScalesArray = try array(name: "\(prefix).up_proj.scales", data: parts.4!)
        let upBiasesArray = try parts.5.map {
            try array(name: "\(prefix).up_proj.biases", data: $0)
        }
        let downWeightArray = try array(name: "\(prefix).down_proj.weight", data: parts.6!)
        let downScalesArray = try array(name: "\(prefix).down_proj.scales", data: parts.7!)
        let downBiasesArray = try parts.8.map {
            try array(name: "\(prefix).down_proj.biases", data: $0)
        }
        let bytes =
            (parts.0?.count ?? 0) + (parts.1?.count ?? 0) + (parts.2?.count ?? 0)
            + (parts.3?.count ?? 0) + (parts.4?.count ?? 0) + (parts.5?.count ?? 0)
            + (parts.6?.count ?? 0) + (parts.7?.count ?? 0) + (parts.8?.count ?? 0)
        let payload = try StreamedQuantizedExpertWeights(
            gateWeight: gateWeightArray, gateScales: gateScalesArray, gateBiases: gateBiasesArray,
            upWeight: upWeightArray, upScales: upScalesArray, upBiases: upBiasesArray,
            downWeight: downWeightArray, downScales: downScalesArray, downBiases: downBiasesArray,
            byteCount: bytes)
        return try ExpertWeight(unchecked: payload, byteCount: bytes)
    }

    private func readData(name: String, expert: Int, required: Bool) async throws -> Data? {
        guard let tensor = tensors[name] else {
            if !required { return nil }
            throw NativeTextModelLoadingError.streamedExpertsUnsupported(
                "Missing \(name)")
        }
        let isWeight = name.hasSuffix(".weight")
        let valid =
            isWeight
            ? tensor.metadata.dataType == .uint32
            : tensor.metadata.dataType == .float16 || tensor.metadata.dataType == .bfloat16
        guard valid else {
            throw NativeTextModelLoadingError.streamedExpertsUnsupported(
                "\(name) has an unsupported safetensor dtype")
        }
        return try await tensor.reader.readSlice(
            named: name, leadingRange: UInt64(expert) ..< UInt64(expert + 1))
    }

    private func array(name: String, data: Data) throws -> MLXArray {
        guard let tensor = tensors[name] else {
            throw NativeTextModelLoadingError.streamedExpertsUnsupported("Missing \(name)")
        }
        let shape = tensor.metadata.shape.dropFirst().map { Int($0) }
        let dtype: DType
        switch tensor.metadata.dataType {
        case .uint32: dtype = .uint32
        case .float16: dtype = .float16
        case .bfloat16: dtype = .bfloat16
        default:
            throw NativeTextModelLoadingError.streamedExpertsUnsupported(
                "\(name) has an unsupported safetensor dtype")
        }
        return MLXArray(data, shape, dtype: dtype)
    }

    static func totalTargetWeightBytes(directory: URL) throws -> (dense: Int, routedExperts: Int) {
        var total = 0
        var routed = 0
        for url in try safetensorWeightURLs(in: directory) {
            for tensor in try SafetensorRangeReader(url: url).tensors
            where isTargetWeight(tensor.name) {
                let bytes = try intBytes(tensor.byteCount, name: tensor.name)
                total += bytes
                if isRoutedExpertWeight(tensor.name) { routed += bytes }
            }
        }
        return (total - routed, routed)
    }

    static func isRoutedExpertWeight(_ name: String) -> Bool {
        let suffixes = [
            ".gate_proj.weight", ".gate_proj.scales", ".gate_proj.biases",
            ".up_proj.weight", ".up_proj.scales", ".up_proj.biases",
            ".down_proj.weight", ".down_proj.scales", ".down_proj.biases",
        ]
        return name.contains(".mlp.switch_mlp.")
            && suffixes.contains(where: { name.hasSuffix($0) })
    }

    private static func isTargetWeight(_ name: String) -> Bool {
        !name.hasPrefix("mtp.")
            && !name.hasPrefix("vision_tower")
            && !name.hasPrefix("model.visual")
    }

    private static func resolvePrefix(layer: Int, tensors: [String: Tensor]) throws -> String {
        let candidates = [
            "language_model.model.layers.\(layer).mlp.switch_mlp",
            "model.layers.\(layer).mlp.switch_mlp",
            "layers.\(layer).mlp.switch_mlp",
        ]
        guard let prefix = candidates.first(where: { tensors["\($0).gate_proj.weight"] != nil })
        else {
            throw NativeTextModelLoadingError.streamedExpertsUnsupported(
                "Layer \(layer) has no direct switch_mlp expert tensors")
        }
        return prefix
    }

    private static func resolveGroupSize(
        prefix: String, tensors: [String: Tensor], configuration: Qwen35TextConfiguration
    ) throws -> Int {
        guard let scales = tensors["\(prefix).gate_proj.scales"],
            scales.metadata.shape.count == 3,
            scales.metadata.shape[2] > 0,
            configuration.hiddenSize.isMultiple(of: Int(scales.metadata.shape[2]))
        else {
            throw NativeTextModelLoadingError.streamedExpertsUnsupported(
                "Routed expert scales do not describe a quantization group")
        }
        let groupSize = configuration.hiddenSize / Int(scales.metadata.shape[2])
        guard groupSize > 0 else {
            throw NativeTextModelLoadingError.streamedExpertsUnsupported(
                "Routed expert quantization group is invalid")
        }
        return groupSize
    }

    private static func bytes(
        for prefix: String, tensors: [String: Tensor], expertCount: Int
    ) throws -> Int {
        var total = 0
        for projection in ["gate_proj", "up_proj", "down_proj"] {
            for component in ["weight", "scales"] {
                let name = "\(prefix).\(projection).\(component)"
                guard let tensor = tensors[name], tensor.metadata.shape.first == UInt64(expertCount)
                else {
                    throw NativeTextModelLoadingError.streamedExpertsUnsupported(
                        "\(name) is not an outer-expert tensor")
                }
                total += try intBytes(tensor.metadata.byteCount, name: name)
            }
            let biasName = "\(prefix).\(projection).biases"
            if let tensor = tensors[biasName] {
                guard tensor.metadata.shape.first == UInt64(expertCount) else {
                    throw NativeTextModelLoadingError.streamedExpertsUnsupported(
                        "\(biasName) is not an outer-expert tensor")
                }
                total += try intBytes(tensor.metadata.byteCount, name: biasName)
            }
        }
        return total
    }

    private static func intBytes(_ bytes: UInt64, name: String) throws -> Int {
        guard let value = Int(exactly: bytes) else {
            throw NativeTextModelLoadingError.streamedExpertsUnsupported(
                "\(name) exceeds this platform's addressable size")
        }
        return value
    }
}
