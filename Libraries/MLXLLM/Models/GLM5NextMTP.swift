//
//  GLM5NextMTP.swift
//  LLM
//
//  GLM-5.3-Flash multi-token prediction head (`layers.<num_hidden_layers>`), drafted with the
//  generic DeepSeek-style MTP drafter.
//

import Foundation
import MLX
import MLXLMCommon
import MLXNN

extension GLM5NextModel: MTPTargetModel {
    public var mtpTargetArchitectureID: String {
        "glm5_next:\(configuration.hiddenSize):\(configuration.vocabularySize)"
    }
    public var mtpEmbedding: Embedding { model.embedTokens }
    public var mtpOutputHead: Linear? { lmHead }
}

/// The MTP decoder layer. The checkpoint has no hyper-connection weights for it, so it uses a
/// plain residual around sparse attention and the MoE, like SGLang's NextN layer.
final class GLM5NextMTPLayer: Module, MTPDecoderBlock {
    @ModuleInfo(key: "self_attn") var attention: GLM5NextSparseAttention
    @ModuleInfo(key: "mlp") var mlp: GLM5NextMoE
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    init(_ config: GLM5NextConfiguration) {
        _attention.wrappedValue = GLM5NextSparseAttention(config, hasIndexer: true)
        _mlp.wrappedValue = GLM5NextMoE(config)
        _inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, cache: KVCache) -> MLXArray {
        let h = x + attention(inputLayerNorm(x), cache: cache, sharedSelection: nil).0
        return h + mlp(postAttentionLayerNorm(h))
    }

    func makeCache() -> KVCache { KVCacheSimple() }
}

/// The GLM-5.3-Flash MTP head. It loads from the target's checkpoint, reading only
/// `layers.<num_hidden_layers>.*`, and consumes the target's final post-norm hidden states.
///
/// `index_share_for_mtp_iteration` lets serving stacks reuse the first step's top-k on later
/// drafted positions; this drafter recomputes the indexer at every step.
public final class GLM5NextMTPDraftModel: DeepSeekMTPDraftModel {
    let configuration: GLM5NextConfiguration

    public init(_ configuration: GLM5NextConfiguration) {
        self.configuration = configuration
        super.init(
            hiddenSize: configuration.hiddenSize, rmsNormEps: configuration.rmsNormEps,
            block: GLM5NextMTPLayer(configuration),
            targetArchitectureID:
                "glm5_next:\(configuration.hiddenSize):\(configuration.vocabularySize)",
            cacheBytesPerToken: (configuration.kvLoraRank + 2 * configuration.indexHeadDim) * 2)
    }

    static func layerPrefixes(_ configuration: GLM5NextConfiguration) -> [String] {
        let layer = configuration.hiddenLayers
        return [
            "model.language_model.layers.\(layer).", "model.layers.\(layer).",
            "language_model.model.layers.\(layer).",
        ]
    }

    /// Maps the MTP layer's `enorm`, `hnorm`, `eh_proj`, and `shared_head` to the drafter and
    /// its decoder weights to `mtp_block`, then applies the target's conversions.
    public override func prepareCheckpoint(_ checkpoint: ModelCheckpoint) throws
        -> ModelCheckpoint
    {
        let prefix = "model.layers.\(configuration.hiddenLayers)."
        let own: Set = ["enorm", "hnorm", "eh_proj", "shared_head"]
        var checkpoint = try checkpoint.mapNames { name in
            guard let mapped = GLM5NextCheckpoint.canonicalName(name),
                mapped.hasPrefix(prefix)
            else { return nil }
            let rest = String(mapped.dropFirst(prefix.count))
            // The embedding and output head are the target's.
            if rest.hasPrefix("embed_tokens.") || rest.hasPrefix("shared_head.head.") {
                return nil
            }
            let head = String(rest.prefix { $0 != "." })
            return own.contains(head) ? rest : "mtp_block.\(rest)"
        }
        try GLM5NextCheckpoint.normalize(
            &checkpoint, configuration: configuration, layerPrefixes: ["mtp_block"])
        return checkpoint
    }

    /// Loads the MTP head from a `glm5_next` checkpoint directory, or returns `nil` when the
    /// checkpoint has none.
    public static func load(directory: URL) async throws -> GLM5NextMTPDraftModel? {
        let data = try Data(contentsOf: directory.appending(component: "config.json"))
        let base = try JSONDecoder.json5().decode(BaseConfiguration.self, from: data)
        guard base.modelType == "glm5_next" else { return nil }
        let configuration = try JSONDecoder.json5().decode(GLM5NextConfiguration.self, from: data)
        guard configuration.numNextnPredictLayers > 0 else { return nil }
        let prefixes = layerPrefixes(configuration)
        let selection = WeightTensorNameSelection.excluding { name in
            !prefixes.contains { name.hasPrefix($0) }
        }
        guard try containsWeightTensor(in: directory, tensorNameSelection: selection) else {
            return nil
        }
        let drafter = GLM5NextMTPDraftModel(configuration)
        try await loadWeights(
            modelDirectory: directory, model: drafter, tensorNameSelection: selection,
            perLayerQuantization: base.perLayerQuantization)
        return drafter
    }
}

/// Registers the GLM-5.3-Flash MTP head with ``MTPDrafterTypeRegistry/shared``, so
/// `MTPDrafterModelFactory` can load it from the target's own checkpoint directory.
public enum GLM5NextMTPRegistration {
    public static func register() async {
        await MTPDrafterTypeRegistry.shared.registerModelType("glm5_next") { data in
            GLM5NextMTPDraftModel(
                try JSONDecoder.json5().decode(GLM5NextConfiguration.self, from: data))
        }
    }
}
