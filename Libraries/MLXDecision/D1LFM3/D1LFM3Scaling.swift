// Copyright © 2026 Faux Fox, Inc.
// Activation-aware channel scaling (AWQ, Lin et al. 2023) folded into the LFM3 text trunk.

import Foundation
import MLX

/// Rescales input channels of the trunk's linear layers before quantization, so channels with
/// large activations get a finer share of each group's grid. Every scale folds into the
/// producer of that input, so the float model is unchanged:
/// - `operator_norm` and `ffn_norm` weights take `1 / s` for the layers that read them;
/// - `w2`'s inputs fold into the rows of `w3`, `o_proj`'s into the rows of `v_proj` (shared by
///   the query heads of one key-value head), and the short convolution's `out_proj` into the
///   `C` rows of `in_proj`.
/// Each fold searches `s = importance^α` for the α with the least importance-weighted
/// quantization error of the layers it scales.
enum D1LFM3Scaling {
    static let alphas: [Float] = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

    struct Fold {
        /// Layers whose input columns take `s`.
        var consumers: [String]
        /// Where `1 / s` goes: a norm weight, or rows of a linear layer (`rows` selects them).
        var producer: String
        var rows: Range<Int>?
        /// Maps consumer columns to `s` entries (GQA: one value channel feeds several heads).
        var expand: ((MLXArray) -> MLXArray)?
        var importance: String
    }

    static func folds(_ text: D1LFM3Configuration.Text) -> [Fold] {
        var result = [Fold]()
        let d = text.hiddenSize
        for (i, type) in text.layerTypes.enumerated() {
            let p = "language_model.layers.\(i)"
            if type == "conv" {
                result.append(
                    Fold(
                        consumers: ["\(p).conv.in_proj"], producer: "\(p).operator_norm",
                        importance: "\(p).conv.in_proj"))
                result.append(
                    Fold(
                        consumers: ["\(p).conv.out_proj"], producer: "\(p).conv.in_proj",
                        rows: d ..< 2 * d, importance: "\(p).conv.out_proj"))
            } else {
                let a = "\(p).self_attn"
                result.append(
                    Fold(
                        consumers: ["\(a).q_proj", "\(a).k_proj", "\(a).v_proj", "\(a).gate_proj"],
                        producer: "\(p).operator_norm", importance: "\(a).q_proj"))
                let groups = text.attentionHeads / text.keyValueHeads
                let headDim = text.headDim
                let kv = text.keyValueHeads
                // s over v's [kv × headDim] channels → o_proj's [heads × headDim] columns.
                result.append(
                    Fold(
                        consumers: ["\(a).o_proj"], producer: "\(a).v_proj",
                        rows: 0 ..< kv * headDim,
                        expand: { s in
                            broadcast(
                                s.reshaped(kv, 1, headDim), to: [kv, groups, headDim]
                            ).reshaped(-1)
                        }, importance: "\(a).o_proj"))
            }
            result.append(
                Fold(
                    consumers: ["\(p).feed_forward.w1", "\(p).feed_forward.w3"],
                    producer: "\(p).ffn_norm", importance: "\(p).feed_forward.w1"))
            result.append(
                Fold(
                    consumers: ["\(p).feed_forward.w2"], producer: "\(p).feed_forward.w3",
                    rows: 0 ..< text.intermediateSize, importance: "\(p).feed_forward.w2"))
        }
        return result
    }

    /// Importance-weighted squared error of quantizing `w` [O, I] with input scale `s` [I].
    static func error(
        _ w: MLXArray, s: MLXArray, importance: MLXArray,
        setting: D1LFM3Configuration.Quantization.Setting, scaleType: DType
    ) -> Float {
        let scaled = w * s
        let q = D1LFM3Quantizer.quantize(
            scaled, setting: setting, method: .search, importance: importance / (s * s),
            scaleType: scaleType)
        let back =
            dequantized(
                q.weight, scales: q.scales, biases: q.biases, groupSize: setting.groupSize,
                bits: setting.bits
            ).asType(.float32) / s
        let e = (back - w) * (back - w) * importance
        return e.sum().item(Float.self)
    }

    /// Folds the chosen scales into `weights` (sanitized names) and `importance` (each scaled
    /// layer's inputs shrink by `s`, so their mean square by `s²`). Returns α per fold.
    static func apply(
        _ weights: inout [String: MLXArray], importance: inout [String: MLXArray],
        configuration: D1LFM3Configuration.Text, recipe: D1LFM3Recipe, scaleType: DType
    ) -> [String: Float] {
        var chosen = [String: Float]()
        for fold in folds(configuration) {
            guard let raw = importance[fold.importance] else { continue }
            let settings = fold.consumers.map { recipe.setting(for: $0) }
            guard settings.allSatisfy({ $0 != nil && $0!.mode == "affine" }) else { continue }
            var imp = raw.asType(.float32)
            if fold.expand != nil {
                // The value channel's importance: the mean over the heads it feeds.
                let kv = configuration.keyValueHeads
                let heads = configuration.attentionHeads
                imp = imp.reshaped(kv, heads / kv, configuration.headDim).mean(axis: 1).reshaped(-1)
            }
            let base = maximum(imp, MLXArray(Float(1e-12)))
            var best: (Float, Float, MLXArray)? = nil
            for alpha in alphas {
                var s = pow(base, alpha / 2)
                s = s / sqrt(s.max() * s.min())
                let columns = fold.expand?(s) ?? s
                var total: Float = 0
                for (path, setting) in zip(fold.consumers, settings) {
                    guard let w = weights["\(path).weight"]?.asType(.float32) else { continue }
                    total += error(
                        w, s: columns, importance: importance[path]!.asType(.float32),
                        setting: setting!, scaleType: scaleType)
                }
                if best == nil || total < best!.0 { best = (total, alpha, s) }
            }
            guard let (_, alpha, s) = best, alpha > 0 else {
                chosen[fold.importance] = 0
                continue
            }
            chosen[fold.importance] = alpha
            let columns = fold.expand?(s) ?? s
            for path in fold.consumers {
                let key = "\(path).weight"
                if let w = weights[key] {
                    weights[key] = w.asType(.float32) * columns
                    eval(weights[key]!)
                }
                if let i = importance[path] {
                    importance[path] = i.asType(.float32) / (columns * columns)
                }
            }
            if let rows = fold.rows {
                let key = "\(fold.producer).weight"
                if let w = weights[key] {
                    var w32 = w.asType(.float32)
                    w32[rows.lowerBound ..< rows.upperBound] =
                        w32[rows.lowerBound ..< rows.upperBound] / s.expandedDimensions(axis: 1)
                    weights[key] = w32
                }
            } else {
                let key = "\(fold.producer).weight"
                if let w = weights[key] {
                    weights[key] = w.asType(.float32) / s
                }
            }
            eval(weights["\(fold.producer).weight"] ?? MLXArray(0))
        }
        return chosen
    }
}
