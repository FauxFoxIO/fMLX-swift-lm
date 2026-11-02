// Copyright © 2026 Faux Fox, Inc.

import Foundation
import MLX

/// `config.json` of an `lfm3_vl` d1 checkpoint (Liquid d1-2B and its fine-tunes) and of its MLX exports.
public struct D1LFM3Configuration: Codable, Sendable {
    public struct Text: Codable, Sendable {
        public let vocabularySize: Int
        public let hiddenSize: Int
        public let intermediateSize: Int
        public let attentionHeads: Int
        public let keyValueHeads: Int
        public let headDim: Int
        public let layerTypes: [String]
        public let normEps: Float
        public let convolutionCache: Int
        public let slidingWindow: Int
        public let rope: [String: Rope]

        public struct Rope: Codable, Sendable {
            public let theta: Float
            public let partialRotaryFactor: Float?

            enum CodingKeys: String, CodingKey {
                case theta = "rope_theta"
                case partialRotaryFactor = "partial_rotary_factor"
            }
        }

        enum CodingKeys: String, CodingKey {
            case vocabularySize = "vocab_size"
            case hiddenSize = "hidden_size"
            case intermediateSize = "intermediate_size"
            case attentionHeads = "num_attention_heads"
            case keyValueHeads = "num_key_value_heads"
            case headDim = "head_dim"
            case layerTypes = "layer_types"
            case normEps = "norm_eps"
            case convolutionCache = "conv_L_cache"
            case slidingWindow = "sliding_window"
            case rope = "rope_parameters"
        }
    }

    public struct Vision: Codable, Sendable {
        public let hiddenSize: Int
        public let intermediateSize: Int
        public let attentionHeads: Int
        public let keyValueHeads: Int
        public let layerTypes: [String]
        public let normEps: Float
        public let patchSize: Int
        public let mergeSize: Int
        public let temporalPatchSize: Int
        public let channels: Int
        public let convolutionKernel: Int
        public let rope: Text.Rope

        enum CodingKeys: String, CodingKey {
            case hiddenSize = "hidden_size"
            case intermediateSize = "intermediate_size"
            case attentionHeads = "num_attention_heads"
            case keyValueHeads = "num_key_value_heads"
            case layerTypes = "layer_types"
            case normEps = "norm_eps"
            case patchSize = "patch_size"
            case mergeSize = "spatial_merge_size"
            case temporalPatchSize = "temporal_patch_size"
            case channels = "in_channels"
            case convolutionKernel = "conv_kernel_size"
            case rope = "rope_parameters"
        }

        /// Values in one pixel patch: channels × frames × patch × patch.
        public var patchValues: Int { channels * temporalPatchSize * patchSize * patchSize }
    }

    /// Per-tensor quantization of an MLX export: a default plus overrides keyed by module path.
    /// A `false` override keeps that module in floating point.
    public struct Quantization: Codable, Sendable {
        public struct Setting: Codable, Sendable, Hashable {
            public var groupSize: Int
            public var bits: Int
            public var mode: String

            public init(groupSize: Int, bits: Int, mode: String = "affine") {
                self.groupSize = groupSize
                self.bits = bits
                self.mode = mode
            }

            enum CodingKeys: String, CodingKey {
                case groupSize = "group_size"
                case bits, mode
            }

            public init(from decoder: Decoder) throws {
                let c = try decoder.container(keyedBy: CodingKeys.self)
                groupSize = try c.decode(Int.self, forKey: .groupSize)
                bits = try c.decode(Int.self, forKey: .bits)
                mode = try c.decodeIfPresent(String.self, forKey: .mode) ?? "affine"
            }

            public var quantizationMode: QuantizationMode {
                QuantizationMode(rawValue: mode) ?? .affine
            }
        }

        public var defaults: Setting
        public var overrides: [String: Setting?]

        public init(defaults: Setting, overrides: [String: Setting?] = [:]) {
            self.defaults = defaults
            self.overrides = overrides
        }

        /// The setting for the module at `path`, or nil to keep it in floating point.
        public func setting(for path: String) -> Setting? {
            if let override = overrides[path] { return override }
            return defaults
        }

        struct Key: CodingKey {
            var stringValue: String
            var intValue: Int? { nil }
            init(stringValue: String) { self.stringValue = stringValue }
            init?(intValue: Int) { nil }
        }

        public init(from decoder: Decoder) throws {
            defaults = try Setting(from: decoder)
            let c = try decoder.container(keyedBy: Key.self)
            var overrides = [String: Setting?]()
            for key in c.allKeys where !["group_size", "bits", "mode"].contains(key.stringValue) {
                if (try? c.decode(Bool.self, forKey: key)) == false {
                    overrides[key.stringValue] = .some(nil)
                } else {
                    overrides[key.stringValue] = try c.decode(Setting.self, forKey: key)
                }
            }
            self.overrides = overrides
        }

        public func encode(to encoder: Encoder) throws {
            var c = encoder.container(keyedBy: Key.self)
            try c.encode(defaults.groupSize, forKey: Key(stringValue: "group_size"))
            try c.encode(defaults.bits, forKey: Key(stringValue: "bits"))
            try c.encode(defaults.mode, forKey: Key(stringValue: "mode"))
            for (path, setting) in overrides.sorted(by: { $0.key < $1.key }) {
                if let setting {
                    try c.encode(setting, forKey: Key(stringValue: path))
                } else {
                    try c.encode(false, forKey: Key(stringValue: path))
                }
            }
        }
    }

    public let text: Text
    public let vision: Vision?
    public let projectorHiddenSize: Int
    public let projectorLayerNormEps: Float
    public let imageTokenId: Int
    public let codes: Int?
    public let attention: String?
    public let setSize: Int?
    public let visionMaximumPixels: Int?
    public var quantization: Quantization?

    enum CodingKeys: String, CodingKey {
        case text = "text_config"
        case vision = "vision_config"
        case projectorHiddenSize = "projector_hidden_size"
        case projectorLayerNormEps = "projector_layer_norm_eps"
        case imageTokenId = "image_token_id"
        case codes = "d1_codes"
        case attention = "d1_attention"
        case setSize = "d1_set_size"
        case visionMaximumPixels = "d1_vision_max_pixels"
        case quantization
    }

    public static func load(_ directory: URL) throws -> D1LFM3Configuration {
        let data = try Data(contentsOf: directory.appendingPathComponent("config.json"))
        let configuration = try JSONDecoder().decode(D1LFM3Configuration.self, from: data)
        guard configuration.stateAttention != nil else {
            throw DecisionError.invalidConfiguration(
                "d1 attention \(configuration.attention ?? "") is not supported; `set` and `causal_state` are"
            )
        }
        return configuration
    }

    /// `d1_attention`: absent or `set` → `.set`; `causal_state` → `.causalState`; anything else nil.
    public var stateAttention: D1LFM3StateAttention? {
        switch attention {
        case nil, "set": .set
        case "causal_state": .causalState
        default: nil
        }
    }
}

/// How d1's full-attention layers treat the state. `.set` (round 1): the state reads every position,
/// questions included, so nothing can be cached. `.causalState`: the state reads only earlier state, and
/// each question reads the state and its own group, so a state prefix can be cached
/// (``D1LFM3PrefixCache``).
public enum D1LFM3StateAttention: String, Sendable {
    case set
    case causalState = "causal_state"
}
