//
//  GLMMoEDSAMTP.swift
//  LLM
//
//  GLM-5.x multi-token prediction head (`model.layers.<num_hidden_layers>`), drafted with the
//  generic DeepSeek-style MTP drafter.
//

import Foundation
import MLX
import MLXLMCommon
import MLXNN

extension GLMMoEDSAModel: MTPTargetModel {
    public var mtpTargetArchitectureID: String {
        "glm_moe_dsa:\(configuration.hiddenSize):\(configuration.vocabularySize)"
    }
    public var mtpEmbedding: Embedding { model.embedTokens }
    public var mtpOutputHead: Linear? { lmHead }
}

extension GLMMoEDSADecoderLayer: MTPDecoderBlock {
    func callAsFunction(_ x: MLXArray, cache: KVCache) -> MLXArray {
        self(x, cache: cache, sharedSelection: nil).0
    }

    func makeCache() -> KVCache { KVCacheSimple() }
}

/// The GLM-5.x MTP layer. It has its own indexer and MoE, and loads from the target's
/// checkpoint, reading only `model.layers.<num_hidden_layers>.*`.
///
/// `index_share_for_mtp_iteration` lets serving stacks reuse the first step's top-k on later
/// drafted positions; this drafter recomputes the indexer at every step.
public final class GLMMoEDSAMTPDraftModel: DeepSeekMTPDraftModel {
    let configuration: GLMMoEDSAConfiguration

    public init(_ configuration: GLMMoEDSAConfiguration) {
        self.configuration = configuration
        let latentWidth =
            configuration.kvLoraRank + configuration.qkRopeHeadDim + configuration.indexHeadDim
        super.init(
            hiddenSize: configuration.hiddenSize, rmsNormEps: configuration.rmsNormEps,
            block: GLMMoEDSADecoderLayer(configuration, hasIndexer: true, sparseMLP: true),
            targetArchitectureID:
                "glm_moe_dsa:\(configuration.hiddenSize):\(configuration.vocabularySize)",
            cacheBytesPerToken: latentWidth * 2)
    }

    private var layerPrefix: String { "model.layers.\(configuration.hiddenLayers)" }

    public override var excludedCheckpointNamespaces: [String] {
        ["model.embed_tokens", "model.norm", "lm_head"]
            + (0 ..< configuration.hiddenLayers).map { "model.layers.\($0)" }
    }

    /// Maps `model.layers.<N>.{enorm,hnorm,eh_proj,shared_head}` to the drafter and the
    /// layer's decoder weights to `mtp_block`, then applies the target's conversions.
    public override func prepareCheckpoint(_ checkpoint: ModelCheckpoint) throws
        -> ModelCheckpoint
    {
        let prefix = layerPrefix + "."
        let own: Set = ["enorm", "hnorm", "eh_proj", "shared_head"]
        var checkpoint = try checkpoint.mapNames { name in
            guard name.hasPrefix(prefix) else { return nil }
            let rest = String(name.dropFirst(prefix.count))
            // The embedding and output head are the target's.
            if rest.hasPrefix("embed_tokens.") || rest.hasPrefix("shared_head.head.") {
                return nil
            }
            let head = String(rest.prefix { $0 != "." })
            return own.contains(head) ? rest : "mtp_block.\(rest)"
        }
        try GLMMoEDSACheckpoint.normalize(
            &checkpoint, configuration: configuration, layerPrefixes: ["mtp_block"])
        return checkpoint
    }

    /// Loads the MTP head from a `glm_moe_dsa` checkpoint directory, or returns `nil` when the
    /// checkpoint has none (mlx-community conversions usually drop it).
    public static func load(directory: URL) async throws -> GLMMoEDSAMTPDraftModel? {
        let data = try Data(contentsOf: directory.appending(component: "config.json"))
        let base = try JSONDecoder.json5().decode(BaseConfiguration.self, from: data)
        guard base.modelType == "glm_moe_dsa" else { return nil }
        let configuration = try JSONDecoder.json5().decode(
            GLMMoEDSAConfiguration.self, from: data)
        guard configuration.numNextnPredictLayers > 0 else { return nil }
        let selection = WeightTensorNameSelection.prefixed(
            "model.layers.\(configuration.hiddenLayers).")
        guard try containsWeightTensor(in: directory, tensorNameSelection: selection) else {
            return nil
        }
        let drafter = GLMMoEDSAMTPDraftModel(configuration)
        try await loadWeights(
            modelDirectory: directory, model: drafter, tensorNameSelection: selection,
            perLayerQuantization: base.perLayerQuantization)
        return drafter
    }
}

/// Registers the GLM-5.x MTP head with ``MTPDrafterTypeRegistry/shared``, so
/// `MTPDrafterModelFactory` can load it from the target's own checkpoint directory.
public enum GLMMoEDSAMTPRegistration {
    public static func register() async {
        await MTPDrafterTypeRegistry.shared.registerModelType("glm_moe_dsa") { data in
            GLMMoEDSAMTPDraftModel(
                try JSONDecoder.json5().decode(GLMMoEDSAConfiguration.self, from: data))
        }
    }
}
