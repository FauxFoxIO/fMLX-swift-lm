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
            model = Qwen35TextModel(
                try JSONDecoder().decode(Qwen35TextConfiguration.self, from: data))
        default:
            throw ConcurrentTextRuntimeError.unsupportedCache
        }
        try await loadWeights(
            modelDirectory: directory, model: model,
            perLayerQuantization: base.perLayerQuantization)
        return model
    }
}
