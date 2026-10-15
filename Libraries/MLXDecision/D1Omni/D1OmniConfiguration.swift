// Copyright © 2026 Faux Fox, Inc.

import Foundation

/// `config.json` of a `d1_omni` checkpoint (LiquidAI/d1-omni-600M and its MLX exports).
public struct D1OmniConfiguration: Codable, Sendable {
    public struct Text: Codable, Sendable {
        public let vocabularySize: Int
        public let hiddenSize: Int
        public let intermediateSize: Int
        public let hiddenLayers: Int
        public let attentionHeads: Int
        public let keyValueHeads: Int
        public let layerTypes: [String]
        public let normEps: Float
        public let convolutionCache: Int
        public let feedForwardMultiplier: Double?
        public let feedForwardMultipleOf: Int
        public let ropeTheta: Float

        enum CodingKeys: String, CodingKey {
            case vocabularySize = "vocab_size"
            case hiddenSize = "hidden_size"
            case intermediateSize = "intermediate_size"
            case hiddenLayers = "num_hidden_layers"
            case attentionHeads = "num_attention_heads"
            case keyValueHeads = "num_key_value_heads"
            case layerTypes = "layer_types"
            case normEps = "norm_eps"
            case convolutionCache = "conv_L_cache"
            case feedForwardMultiplier = "block_ffn_dim_multiplier"
            case feedForwardMultipleOf = "block_multiple_of"
            case ropeTheta = "rope_theta"
        }

        /// LFM2's adjusted feed-forward width.
        var feedForwardSize: Int {
            var size = Int(2 * Double(intermediateSize) / 3)
            if let feedForwardMultiplier { size = Int(feedForwardMultiplier * Double(size)) }
            return feedForwardMultipleOf
                * ((size + feedForwardMultipleOf - 1) / feedForwardMultipleOf)
        }
    }

    public struct Vision: Codable, Sendable {
        public let hiddenSize: Int
        public let intermediateSize: Int
        public let hiddenLayers: Int
        public let attentionHeads: Int
        public let channels: Int
        public let patches: Int
        public let patchSize: Int
        public let layerNormEps: Float
        public let hiddenAct: String

        enum CodingKeys: String, CodingKey {
            case hiddenSize = "hidden_size"
            case intermediateSize = "intermediate_size"
            case hiddenLayers = "num_hidden_layers"
            case attentionHeads = "num_attention_heads"
            case channels = "num_channels"
            case patches = "num_patches"
            case patchSize = "patch_size"
            case layerNormEps = "layer_norm_eps"
            case hiddenAct = "hidden_act"
        }
    }

    public struct Audio: Codable, Sendable {
        public let melBins: Int
        public let layers: Int
        public let modelSize: Int
        public let subsamplingChannels: Int
        public let feedForwardExpansion: Int
        public let heads: Int
        public let convolutionKernel: Int
        public let residualWidth: Int

        enum CodingKeys: String, CodingKey {
            case melBins = "feat_in"
            case layers = "n_layers"
            case modelSize = "d_model"
            case subsamplingChannels = "subsampling_conv_channels"
            case feedForwardExpansion = "ff_expansion_factor"
            case heads = "n_heads"
            case convolutionKernel = "conv_kernel_size"
            case residualWidth = "residual_width"
        }
    }

    public struct Quantization: Codable, Sendable {
        public let groupSize: Int
        public let bits: Int
        public let mode: String?

        enum CodingKeys: String, CodingKey {
            case groupSize = "group_size"
            case bits, mode
        }
    }

    public let modelType: String
    public let text: Text
    public let vision: Vision
    public let audio: Audio
    public let projectorHiddenSize: Int
    public let headLayers: Int
    public let maxLength: Int
    public let imageTextLength: Int
    public let audioTextLength: Int
    /// Per-type text calibration, keyed like `choice:3-5`, with `choice`, `score`, `noul` fallbacks.
    public let temperatures: [String: Double]
    public let bosTokenID: Int
    public let quantization: Quantization?

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case text = "text_config"
        case vision = "vision_config"
        case audio = "audio_config"
        case projectorHiddenSize = "projector_hidden_size"
        case headLayers = "head_layers"
        case maxLength = "max_length"
        case imageTextLength = "image_text_length"
        case audioTextLength = "audio_text_length"
        case temperatures
        case bosTokenID = "bos_token_id"
        case quantization
    }

    func validate() throws {
        guard modelType == "d1_omni" else {
            throw DecisionError.invalidConfiguration("model_type \(modelType) is not d1_omni")
        }
        guard text.layerTypes.count == text.hiddenLayers,
            text.layerTypes.allSatisfy({ $0 == "conv" || $0 == "full_attention" }),
            text.hiddenSize % text.attentionHeads == 0,
            text.attentionHeads % text.keyValueHeads == 0, text.convolutionCache == 3,
            text.hiddenSize % 64 == 0
        else {
            throw DecisionError.invalidConfiguration("unsupported d1-omni text trunk")
        }
        guard vision.hiddenSize % vision.attentionHeads == 0,
            Int(Double(vision.patches).squareRoot()) * Int(Double(vision.patches).squareRoot())
                == vision.patches, vision.channels == 3
        else {
            throw DecisionError.invalidConfiguration("unsupported d1-omni vision tower")
        }
        guard audio.melBins == 128, audio.modelSize % audio.heads == 0,
            audio.convolutionKernel % 2 == 1
        else {
            throw DecisionError.invalidConfiguration("unsupported d1-omni audio encoder")
        }
        guard maxLength > 64, imageTextLength > 0, audioTextLength > 0,
            temperatures.values.allSatisfy({ $0 > 0 && $0.isFinite })
        else {
            throw DecisionError.invalidConfiguration("invalid d1-omni lengths or temperatures")
        }
    }

    func temperature(for question: D1OmniQuestion) -> Double {
        temperatures[question.temperatureKey] ?? temperatures[question.kind.name] ?? 1
    }
}
