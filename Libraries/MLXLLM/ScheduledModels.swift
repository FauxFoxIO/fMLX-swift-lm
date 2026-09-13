// Copyright © 2026 Faux Fox.

import Foundation
import MLX
import MLXLMCommon

extension LlamaModel {
    public var scheduledSupportsBatchDecode: Bool { true }
}

extension Qwen3Model: ScheduledTextModel {
    public var scheduledCacheBytesPerToken: Int {
        configuration.hiddenLayers * configuration.kvHeads * configuration.headDim * 8
    }
    public var scheduledSupportsBatchDecode: Bool { true }
    public func scheduledForward(_ tokens: MLXArray, cache: [KVCache]) throws -> MLXArray {
        callAsFunction(tokens, cache: cache)
    }
}

extension Qwen35TextModel: ScheduledTextModel {
    public var scheduledMTPArchitectureID: String? { "qwen3_5:\(configuration.hiddenSize)" }
    public var scheduledCacheBytesPerToken: Int {
        let attentionLayers = configuration.hiddenLayers / configuration.fullAttentionInterval
        return attentionLayers * configuration.kvHeads
            * (configuration.headDim ?? configuration.hiddenSize / configuration.attentionHeads) * 8
    }
    public var scheduledRecurrentStateBytes: Int {
        let c = configuration
        let layers = c.hiddenLayers - c.hiddenLayers / c.fullAttentionInterval
        let convDimensions =
            2 * c.linearNumKeyHeads * c.linearKeyHeadDim
            + c.linearNumValueHeads * c.linearValueHeadDim
        let state = c.linearNumValueHeads * c.linearValueHeadDim * c.linearKeyHeadDim
        // Include the live state and a possible speculative rollback checkpoint.
        return layers * ((c.linearConvKernelDim - 1) * convDimensions + state) * 8
    }
    public var scheduledSupportsBatchDecode: Bool { true }
    public func scheduledForward(_ tokens: MLXArray, cache: [KVCache]) throws -> MLXArray {
        callAsFunction(tokens, cache: cache)
    }
}

extension Qwen35Model: ScheduledTextModel {
    public var scheduledMTPArchitectureID: String? { languageModel.scheduledMTPArchitectureID }
    public var scheduledCacheBytesPerToken: Int { languageModel.scheduledCacheBytesPerToken }
    public var scheduledRecurrentStateBytes: Int { languageModel.scheduledRecurrentStateBytes }
    public var scheduledSupportsBatchDecode: Bool { true }
    public func scheduledForward(_ tokens: MLXArray, cache: [KVCache]) throws -> MLXArray {
        callAsFunction(tokens, cache: cache)
    }
}

/// Local-only weight loader for the runtime's audited text architectures. FMLXText's
/// NativeTextModel pairs it with checkpoint-specific tokenization. Neither downloads assets.
public enum NativeTextModelLoader {
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
        let files = try safetensorWeightURLs(in: directory)
        var containsHead = false
        for file in files {
            let handle = try FileHandle(forReadingFrom: file)
            defer { try? handle.close() }
            guard let prefix = try handle.read(upToCount: 8), prefix.count == 8 else {
                throw ConcurrentTextRuntimeError.invalidConfiguration
            }
            let length = prefix.enumerated().reduce(UInt64(0)) {
                $0 | UInt64($1.element) << (8 * $1.offset)
            }
            guard length > 0, length <= 16 * 1024 * 1024,
                let header = try handle.read(upToCount: Int(length)), header.count == Int(length),
                let tensors = try JSONSerialization.jsonObject(with: header) as? [String: Any]
            else { throw ConcurrentTextRuntimeError.invalidConfiguration }
            if tensors.keys.contains(where: { $0.hasPrefix("mtp.") }) {
                containsHead = true
                break
            }
        }
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
            perLayerQuantization: base.perLayerQuantization)
        try Task.checkCancellation()
        return drafter
    }

    /// Loads a standalone, preconverted MLX Qwen MTP head; the caller supplies its matching target.
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
        let model = Qwen35MTPDraftModel(configuration, preconvertedNorms: true)
        try await loadWeights(
            modelDirectory: directory, model: model,
            perLayerQuantization: base.perLayerQuantization)
        return model
    }

    public static func load(directory: URL) async throws -> sending any ScheduledTextModel {
        let data = try Data(contentsOf: directory.appendingPathComponent("config.json"))
        let base = try JSONDecoder().decode(BaseConfiguration.self, from: data)
        let model: any ScheduledTextModel
        switch base.modelType {
        case "llama", "mistral":
            model = LlamaModel(try JSONDecoder().decode(LlamaConfiguration.self, from: data))
        case "qwen3":
            model = Qwen3Model(try JSONDecoder().decode(Qwen3Configuration.self, from: data))
        case "qwen3_5", "qwen3_5_moe":
            let configuration = try JSONDecoder().decode(Qwen35Configuration.self, from: data)
            model =
                base.modelType == "qwen3_5_moe"
                ? Qwen35MoEModel(configuration) : Qwen35Model(configuration)
        case "qwen3_5_text":
            let configuration = try JSONDecoder().decode(Qwen35Configuration.self, from: data)
            model = Qwen35TextModel(
                configuration.textConfig,
                mixedPreservedNorms: configuration.mixedPreservedNorms)
        default:
            throw ConcurrentTextRuntimeError.unsupportedCache
        }
        try await loadWeights(
            modelDirectory: directory, model: model,
            perLayerQuantization: base.perLayerQuantization)
        return model
    }
}
