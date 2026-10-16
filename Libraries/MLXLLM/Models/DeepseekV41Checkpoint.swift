//
//  DeepseekV41Checkpoint.swift
//  LLM
//
//  Loads DeepSeek-V4.1 checkpoints in DeepSeek's native layout and in mlx-community's.
//
//  The native checkpoint stores FP8 E4M3 weights with one UE8M0 scale per 32x32 tile and FP4
//  E2M1 routed experts (two per byte) with one UE8M0 scale per 32 inputs. MLX's mxfp8 and
//  mxfp4 formats use the same element encodings with one UE8M0 scale per 32 inputs of a row,
//  so the weights are repacked without requantizing: the bytes are reinterpreted and each
//  tile scale is repeated for the tile's 32 rows. Port of mlx-lm's `deepseek_v41.sanitize`.
//

import Foundation
import MLX
import MLXLMCommon

enum DeepseekV41Checkpoint {
    /// Tensors a text model never reads. Engram tables are memory-mapped separately.
    static func excludedNamespaces(_ configuration: DeepseekV41Configuration) -> [String] {
        ["mtp", "vision", "aligner", "image_start", "image_end", "image_newline", "model.mtp"]
            + configuration.engramLayerIDs.flatMap {
                ["layers.\($0).engram.embed", "model.layers.\($0).engram.embed"]
            }
    }

    /// Maps a native tensor or layer name to its module path, or `nil` to drop it.
    /// Names that already start with `model.` or `lm_head` pass through.
    static func rename(_ name: String) -> String? {
        if name.hasPrefix("model.") || name.hasPrefix("lm_head") {
            return name
        }
        if name.hasPrefix("mtp.") || name.hasPrefix("vision.") || name.hasPrefix("aligner.")
            || name.hasPrefix("image_") || name.hasSuffix(".gate.bias_vl")
            || name.contains(".engram.embed")
        {
            return nil
        }
        var parts = name.split(separator: ".").map(String.init)
        switch parts.first {
        case "head": parts[0] = "lm_head"
        case "embed": parts[0] = "model.embed_tokens"
        case "norm": parts[0] = "model.norm"
        case "layers":
            var out = ["model", "layers"]
            for (i, part) in parts.enumerated().dropFirst() {
                let previous = parts[i - 1]
                out.append(contentsOf: renameComponent(part, after: previous, at: i, in: parts))
            }
            return out.joined(separator: ".")
        default: break
        }
        return parts.joined(separator: ".")
    }

    private static let projections = ["w1": "gate_proj", "w2": "down_proj", "w3": "up_proj"]

    private static func renameComponent(
        _ part: String, after previous: String, at index: Int, in parts: [String]
    ) -> [String] {
        switch part {
        case "attn": return ["self_attn"]
        case "ffn": return ["mlp"]
        case "attn_norm": return ["input_layernorm"]
        case "ffn_norm": return ["post_attention_layernorm"]
        case "attn_sink": return ["sinks"]
        case "wq_a": return ["q_a_proj"]
        case "wq_b": return ["q_b_proj"]
        case "wkv": return [previous == "engram" ? "wkv" : "kv_proj"]
        case "wgate": return ["gate_proj"]
        case "wo_a": return ["o_a_proj"]
        case "wo_b": return ["o_b_proj"]
        case "q_norm": return ["q_a_norm"]
        case "norm" where previous == "compressor": return ["kv_norm"]
        case "wk" where previous == "indexer": return ["k_proj"]
        case "bias" where previous == "gate": return ["e_score_correction_bias"]
        case "w1", "w2",
            "w3"
        where previous == "shared_experts" || (index >= 2 && parts[index - 2] == "experts"):
            return [projections[part]!]
        default:
            for (prefix, module) in [("hc_attn_", "attn_hc"), ("hc_ffn_", "ffn_hc")]
            where part.hasPrefix(prefix) {
                return [module, String(part.dropFirst(prefix.count))]
            }
            return [part]
        }
    }

    static func quantization(bits: Int) -> BaseConfiguration.Quantization {
        let json = #"{"group_size":32,"bits":\#(bits),"mode":"mxfp\#(bits)"}"#
        return try! JSONDecoder().decode(BaseConfiguration.Quantization.self, from: Data(json.utf8))
    }

    /// Repacks a native FP8 (`uint8`) or packed FP4 (`int8`) weight and its UE8M0 scale as
    /// MLX mxfp8 or mxfp4 `(weight, scales, settings)`.
    static func repack(weight: MLXArray, scale: MLXArray) throws -> (
        MLXArray, MLXArray, BaseConfiguration.Quantization
    ) {
        var scale = scale
        if scale.dtype != .uint8 {
            scale = (log2(scale.asType(.float32)) + 127).asType(.uint8)
        }
        let rows = weight.dim(-2)
        switch weight.dtype {
        case .int8:
            guard scale.dim(-2) == rows, scale.dim(-1) * 16 == weight.dim(-1) else {
                throw DeepseekV41CheckpointError.shape("FP4", weight.shape, scale.shape)
            }
            return (weight.view(dtype: .uint32), scale, quantization(bits: 4))
        case .uint8:
            guard scale.dim(-1) * 32 == weight.dim(-1), scale.dim(-2) * 32 >= rows else {
                throw DeepseekV41CheckpointError.shape("FP8", weight.shape, scale.shape)
            }
            if scale.dim(-2) != rows {
                // One scale per 32x32 tile becomes one per row and 32 inputs.
                scale = repeated(scale, count: 32, axis: -2)[.ellipsis, ..<rows, 0...]
            }
            return (weight.view(dtype: .uint32), scale, quantization(bits: 8))
        default:
            throw DeepseekV41CheckpointError.dtype(weight.dtype)
        }
    }

    static func prepare(_ checkpoint: ModelCheckpoint, configuration: DeepseekV41Configuration)
        throws -> ModelCheckpoint
    {
        try prepare(
            checkpoint, rename: rename,
            layerPrefixes: (0 ..< configuration.hiddenLayers).map { "model.layers.\($0)" },
            routedExperts: configuration.nRoutedExperts, oGroups: configuration.oGroups)
    }

    /// Renames, repacks native FP8/FP4 weights, stacks the routed experts of each layer under
    /// `layerPrefixes`, and splits `wo_a` into its groups.
    static func prepare(
        _ checkpoint: ModelCheckpoint, rename: (String) -> String?, layerPrefixes: [String],
        routedExperts: Int, oGroups: Int
    ) throws -> ModelCheckpoint {
        var checkpoint = try checkpoint.mapNames(rename)
        var weights = checkpoint.weights
        var settings = checkpoint.perLayerQuantization?.perLayerQuantization ?? [:]

        for key in weights.keys where key.hasSuffix(".scale") {
            let path = String(key.dropLast(".scale".count))
            guard let weight = weights["\(path).weight"], let scale = weights[key] else {
                continue
            }
            let (packed, scales, option) = try repack(weight: weight, scale: scale)
            weights[key] = nil
            weights["\(path).weight"] = packed
            weights["\(path).scales"] = scales
            settings[path] = .quantize(option)
        }

        for (layer, layerPrefix) in layerPrefixes.enumerated() {
            let prefix = "\(layerPrefix).mlp.experts"
            for projection in ["gate_proj", "up_proj", "down_proj"] {
                for part in ["weight", "scales", "biases"]
                where weights["\(prefix).0.\(projection).\(part)"] != nil {
                    let parts = (0 ..< routedExperts).compactMap {
                        weights.removeValue(forKey: "\(prefix).\($0).\(projection).\(part)")
                    }
                    guard parts.count == routedExperts else {
                        throw DeepseekV41CheckpointError.experts(layer, parts.count, routedExperts)
                    }
                    weights["\(prefix).switch_mlp.\(projection).\(part)"] = stacked(parts)
                }
                let first = settings["\(prefix).0.\(projection)"]
                for e in 0 ..< routedExperts {
                    settings["\(prefix).\(e).\(projection)"] = nil
                }
                if let first {
                    settings["\(prefix).switch_mlp.\(projection)"] = first
                }
            }

            // wo_a is block diagonal, so split it into its groups.
            let oA = "\(layerPrefix).self_attn.o_a_proj"
            for part in ["weight", "scales", "biases"] {
                if let value = weights["\(oA).\(part)"], value.ndim == 2 {
                    weights["\(oA).\(part)"] = value.reshaped(oGroups, -1, value.dim(-1))
                }
            }
        }

        checkpoint.weights = weights
        if !settings.isEmpty || checkpoint.perLayerQuantization != nil {
            checkpoint.perLayerQuantization = .init(
                quantization: checkpoint.perLayerQuantization?.quantization,
                perLayerQuantization: settings)
        }
        return checkpoint
    }
}

enum DeepseekV41CheckpointError: LocalizedError {
    case shape(String, [Int], [Int])
    case dtype(DType)
    case experts(Int, Int, Int)

    var errorDescription: String? {
        switch self {
        case .shape(let kind, let weight, let scale):
            "\(kind) weight \(weight) does not match its scale \(scale)."
        case .dtype(let dtype):
            "A scaled weight has dtype \(dtype); expected FP8 (uint8) or packed FP4 (int8)."
        case .experts(let layer, let found, let expected):
            "Layer \(layer) has \(found) routed experts, the config declares \(expected)."
        }
    }
}
